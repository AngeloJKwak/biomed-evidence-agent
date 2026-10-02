from __future__ import annotations

import pytest

from biomed_agent.grounding import (
    check_citations,
    citation_metrics,
    normalize_citations,
    normalize_source_id,
)
from biomed_agent.schemas import Claim, DraftAnswer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PMID:37952131", "PMID:37952131"),
        ("pmid 37952131", "PMID:37952131"),
        ("[PMID:37952131]", "PMID:37952131"),
        ("37952131", "PMID:37952131"),
        ("nct03574597", "NCT03574597"),
        ("NCT03574597", "NCT03574597"),
        ("doi:10.1/x", "doi:10.1/x"),
    ],
)
def test_normalize_source_id(raw: str, expected: str) -> None:
    assert normalize_source_id(raw) == expected


def _draft(*claims: tuple[str, list[str]]) -> DraftAnswer:
    return DraftAnswer(
        summary="s",
        claims=[Claim(text=t, citations=c) for t, c in claims],
        evidence_quality="q",
        limitations="l",
    )


def test_check_citations_flags_unknown_and_uncited() -> None:
    draft = _draft(
        ("supported", ["PMID:1"]),
        ("hallucinated source", ["PMID:999"]),
        ("no citation", []),
    )
    issues = check_citations(draft, {"PMID:1", "NCT00000001"})
    assert [(i.claim_index, i.kind) for i in issues] == [
        (1, "unknown_source"),
        (2, "uncited_claim"),
    ]


def test_normalize_citations_dedupes() -> None:
    draft = normalize_citations(_draft(("x", ["PMID:1", "pmid 1", "nct00000001"])))
    assert draft.claims[0].citations == ["PMID:1", "NCT00000001"]


def test_citation_metrics() -> None:
    draft = _draft(("a", ["PMID:1", "PMID:2"]), ("b", []))
    m = citation_metrics(draft, {"PMID:1"})
    assert m == {"citation_validity": 0.5, "citation_coverage": 0.5}
