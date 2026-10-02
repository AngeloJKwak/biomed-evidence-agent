"""PubMed client built on NCBI E-utilities (esearch + efetch).

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
No API key is required; without one NCBI allows ~3 requests/second.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET

import httpx

from biomed_agent.schemas import SourceDocument
from biomed_agent.sources.http import get_with_retry

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "biomed-evidence-agent"


class PubMedClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        api_key: str | None = None,
        email: str | None = None,
    ) -> None:
        self._http = http
        self._api_key = api_key
        self._email = email
        # Stay under NCBI's per-second limit when queries fan out in parallel.
        self._limiter = asyncio.Semaphore(3 if api_key is None else 8)

    def _params(self, **extra: str | int) -> dict[str, str | int]:
        params: dict[str, str | int] = {"db": "pubmed", "tool": TOOL_NAME, **extra}
        if self._api_key:
            params["api_key"] = self._api_key
        if self._email:
            params["email"] = self._email
        return params

    async def search(self, query: str, max_results: int = 10) -> list[str]:
        """Return PMIDs for a query, most relevant first."""
        async with self._limiter:
            resp = await get_with_retry(
                self._http,
                f"{EUTILS}/esearch.fcgi",
                params=self._params(
                    term=query, retmax=max_results, retmode="json", sort="relevance"
                ),
            )
        return list(resp.json().get("esearchresult", {}).get("idlist", []))

    async def fetch(self, pmids: list[str]) -> list[SourceDocument]:
        """Fetch title/abstract/metadata for PMIDs. Records without an abstract are skipped."""
        if not pmids:
            return []
        async with self._limiter:
            resp = await get_with_retry(
                self._http,
                f"{EUTILS}/efetch.fcgi",
                params=self._params(id=",".join(pmids), retmode="xml"),
            )
        return parse_pubmed_xml(resp.text)


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_pubmed_xml(xml_text: str) -> list[SourceDocument]:
    root = ET.fromstring(xml_text)
    docs: list[SourceDocument] = []
    for article in root.iter("PubmedArticle"):
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        title = _text(article.find(".//ArticleTitle"))

        sections = []
        for part in article.findall(".//Abstract/AbstractText"):
            label = part.get("Label")
            body = _text(part)
            if body:
                sections.append(f"{label}: {body}" if label else body)
        abstract = "\n".join(sections)
        if not pmid or not abstract:
            continue

        year_text = _text(article.find(".//JournalIssue/PubDate/Year")) or _text(
            article.find(".//ArticleDate/Year")
        )
        pub_types = [_text(pt) for pt in article.findall(".//PublicationTypeList/PublicationType")]
        doi = next(
            (_text(e) for e in article.findall(".//ELocationID") if e.get("EIdType") == "doi"),
            "",
        )

        docs.append(
            SourceDocument(
                source_id=f"PMID:{pmid}",
                source_type="pubmed",
                title=title,
                text=abstract,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                year=int(year_text) if year_text.isdigit() else None,
                venue=_text(article.find(".//Journal/ISOAbbreviation"))
                or _text(article.find(".//Journal/Title")),
                metadata={
                    k: v
                    for k, v in {
                        "publication_types": "; ".join(p for p in pub_types if p),
                        "doi": doi,
                    }.items()
                    if v
                },
            )
        )
    return docs
