from __future__ import annotations

import json

import httpx
import respx

from biomed_agent.sources.clinicaltrials import API, ClinicalTrialsClient, study_to_document
from biomed_agent.sources.pubmed import EUTILS, PubMedClient, parse_pubmed_xml


def test_parse_pubmed_xml_extracts_structured_abstract(pubmed_xml: str) -> None:
    docs = {d.source_id: d for d in parse_pubmed_xml(pubmed_xml)}

    select = docs["PMID:37952131"]
    assert select.title.startswith("Semaglutide and Cardiovascular Outcomes in Obesity")
    assert select.year == 2023
    assert select.venue == "N Engl J Med"
    assert select.url == "https://pubmed.ncbi.nlm.nih.gov/37952131/"
    # Labelled abstract sections are preserved as "LABEL: text" lines.
    assert "METHODS:" in select.text and "RESULTS:" in select.text
    assert "17,604 patients" in select.text
    assert select.metadata["doi"] == "10.1056/NEJMoa2307563"
    # Reference-list PMIDs must not be mistaken for article PMIDs.
    assert all(d.source_type == "pubmed" for d in docs.values())
    assert len(docs) == 2


def test_study_to_document_flattens_record(ctgov_json: str) -> None:
    studies = json.loads(ctgov_json)["studies"]
    doc = study_to_document(studies[0])

    assert doc is not None
    assert doc.source_id == "NCT04972721"
    assert doc.source_type == "trial"
    assert doc.url == "https://clinicaltrials.gov/study/NCT04972721"
    assert "Status: COMPLETED" in doc.text
    assert "Conditions: Overweight, Obesity" in doc.text
    assert doc.metadata["status"] == "COMPLETED"


def test_study_without_nct_id_is_skipped() -> None:
    assert study_to_document({"protocolSection": {}}) is None


@respx.mock
async def test_pubmed_client_search_and_fetch(pubmed_xml: str) -> None:
    search = respx.get(f"{EUTILS}/esearch.fcgi").respond(
        json={"esearchresult": {"idlist": ["37952131", "34706925"]}}
    )
    respx.get(f"{EUTILS}/efetch.fcgi").respond(text=pubmed_xml)

    async with httpx.AsyncClient() as http:
        client = PubMedClient(http, email="dev@example.com")
        ids = await client.search("semaglutide AND cardiovascular", max_results=5)
        docs = await client.fetch(ids)

    params = search.calls.last.request.url.params
    assert params["term"] == "semaglutide AND cardiovascular"
    assert params["retmax"] == "5"
    assert params["email"] == "dev@example.com"
    assert ids == ["37952131", "34706925"]
    assert {d.source_id for d in docs} == {"PMID:37952131", "PMID:34706925"}


@respx.mock
async def test_pubmed_client_retries_rate_limit() -> None:
    route = respx.get(f"{EUTILS}/esearch.fcgi")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json={"esearchresult": {"idlist": ["1"]}}),
    ]
    async with httpx.AsyncClient() as http:
        assert await PubMedClient(http).search("x") == ["1"]
    assert route.call_count == 2


@respx.mock
async def test_clinicaltrials_client_search(ctgov_json: str) -> None:
    route = respx.get(API).respond(text=ctgov_json, headers={"content-type": "application/json"})
    async with httpx.AsyncClient() as http:
        docs = await ClinicalTrialsClient(http).search("semaglutide obesity", max_results=3)
    assert route.calls.last.request.url.params["query.term"] == "semaglutide obesity"
    assert [d.source_id for d in docs][0] == "NCT04972721"
    assert len(docs) == 3
