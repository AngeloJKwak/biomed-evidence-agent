"""Domain types shared across retrieval, the agent graph, the API, and evals."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["pubmed", "trial"]


class SourceDocument(BaseModel):
    """A retrieved record from PubMed or ClinicalTrials.gov.

    `source_id` is the canonical citation key the model must use:
    ``PMID:<digits>`` for articles and ``NCT<digits>`` for registered trials.
    """

    source_id: str
    source_type: SourceType
    title: str
    text: str
    url: str
    year: int | None = None
    venue: str | None = None  # journal for articles, sponsor for trials
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    source_type: SourceType
    text: str
    score: float | None = None


# --------------------------------------------------------------------------- #
# LLM structured outputs
# --------------------------------------------------------------------------- #


class SearchPlan(BaseModel):
    """Output of the planning step."""

    pubmed_queries: list[str] = Field(
        description="1-3 PubMed queries. Use MeSH terms and boolean operators where useful."
    )
    trials_query: str | None = Field(
        default=None,
        description="A short ClinicalTrials.gov search term, or null if trials are irrelevant.",
    )
    rationale: str = Field(description="One or two sentences on the search strategy.")


class EvidenceAssessment(BaseModel):
    """Output of the sufficiency check after retrieval."""

    sufficient: bool
    missing: str = Field(description="What evidence is missing, or empty if sufficient.")
    followup_pubmed_queries: list[str] = Field(default_factory=list)
    followup_trials_query: str | None = None


class Claim(BaseModel):
    text: str = Field(description="A single factual statement.")
    citations: list[str] = Field(
        description="Source ids (e.g. 'PMID:37952131' or 'NCT03574597') that directly support it."
    )


class DraftAnswer(BaseModel):
    """Output of the generation step."""

    summary: str = Field(description="2-4 sentence direct answer to the question.")
    claims: list[Claim]
    evidence_quality: str = Field(
        description="Brief note on study designs, sample sizes, and consistency of the evidence."
    )
    limitations: str = Field(description="Gaps, conflicting findings, or caveats.")


class ClaimVerdict(BaseModel):
    claim_index: int
    supported: bool
    reason: str


class VerificationResult(BaseModel):
    """Output of the LLM groundedness check."""

    verdicts: list[ClaimVerdict]


# --------------------------------------------------------------------------- #
# Final API response
# --------------------------------------------------------------------------- #


class CitationIssue(BaseModel):
    claim_index: int
    kind: Literal["unknown_source", "uncited_claim", "unsupported"]
    detail: str


class UsageSummary(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    llm_calls: int = 0


class AnswerResponse(BaseModel):
    run_id: str
    question: str
    answer: DraftAnswer
    sources: list[SourceDocument]
    issues: list[CitationIssue]
    grounded: bool
    retrieval_rounds: int
    revisions: int
    latency_s: float
    usage: UsageSummary
