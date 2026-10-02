"""Deterministic citation checks. These run before (and independently of) the LLM verifier."""

from __future__ import annotations

import re

from biomed_agent.schemas import CitationIssue, DraftAnswer

_PMID = re.compile(r"^(?:PMID:?\s*)(\d+)$", re.IGNORECASE)
_NCT = re.compile(r"^(NCT\d{8})$", re.IGNORECASE)


def normalize_source_id(raw: str) -> str:
    """Map common variants ('pmid 123', 'PMID:123', 'nct01234567') to canonical ids."""
    s = raw.strip().strip("[]()")
    if m := _PMID.match(s):
        return f"PMID:{m.group(1)}"
    if m := _NCT.match(s):
        return m.group(1).upper()
    if s.isdigit():
        return f"PMID:{s}"
    return s


def normalize_citations(draft: DraftAnswer) -> DraftAnswer:
    for claim in draft.claims:
        claim.citations = list(dict.fromkeys(normalize_source_id(c) for c in claim.citations))
    return draft


def check_citations(draft: DraftAnswer, retrieved_ids: set[str]) -> list[CitationIssue]:
    """Flag claims with no citation and citations to sources that were never retrieved."""
    issues: list[CitationIssue] = []
    for i, claim in enumerate(draft.claims):
        if not claim.citations:
            issues.append(CitationIssue(claim_index=i, kind="uncited_claim", detail=claim.text))
        for cid in claim.citations:
            if cid not in retrieved_ids:
                issues.append(
                    CitationIssue(
                        claim_index=i,
                        kind="unknown_source",
                        detail=f"{cid} was not in the retrieved evidence",
                    )
                )
    return issues


def citation_metrics(draft: DraftAnswer, retrieved_ids: set[str]) -> dict[str, float]:
    """Precision of citations against retrieved ids, and share of claims carrying a citation."""
    cites = [c for claim in draft.claims for c in claim.citations]
    n_claims = len(draft.claims) or 1
    return {
        "citation_validity": (sum(c in retrieved_ids for c in cites) / len(cites))
        if cites
        else 0.0,
        "citation_coverage": sum(bool(c.citations) for c in draft.claims) / n_claims,
    }
