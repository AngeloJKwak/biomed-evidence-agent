from __future__ import annotations

import json

import httpx
import pytest
import respx

from biomed_agent.agent.graph import Dependencies, EvidenceAgent
from biomed_agent.retrieval.embeddings import HashingEmbedder
from biomed_agent.retrieval.store import VectorStore
from biomed_agent.schemas import (
    Claim,
    ClaimVerdict,
    DraftAnswer,
    EvidenceAssessment,
    SearchPlan,
    VerificationResult,
)
from biomed_agent.sources.clinicaltrials import API, ClinicalTrialsClient
from biomed_agent.sources.pubmed import EUTILS, PubMedClient
from tests.conftest import ScriptedLLM

QUESTION = "Does semaglutide reduce cardiovascular events in adults with obesity but no diabetes?"

PLAN = SearchPlan(
    pubmed_queries=["semaglutide AND cardiovascular outcomes AND obesity"],
    trials_query="semaglutide cardiovascular obesity",
    rationale="Intervention + outcome + population.",
)
SUFFICIENT = EvidenceAssessment(sufficient=True, missing="")
GOOD_DRAFT = DraftAnswer(
    summary="Yes. In SELECT, semaglutide 2.4 mg reduced major adverse cardiovascular events.",
    claims=[
        Claim(text="SELECT enrolled 17,604 patients.", citations=["PMID:37952131"]),
        Claim(text="SELECT-LIFE follows SELECT participants.", citations=["nct04972721"]),
    ],
    evidence_quality="One large phase 3 RCT.",
    limitations="Population had established cardiovascular disease.",
)
ALL_SUPPORTED = VerificationResult(
    verdicts=[
        ClaimVerdict(claim_index=0, supported=True, reason="stated"),
        ClaimVerdict(claim_index=1, supported=True, reason="stated"),
    ]
)


@pytest.fixture
def mocked_apis(pubmed_xml: str, ctgov_json: str):
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{EUTILS}/esearch.fcgi").respond(
            json={"esearchresult": {"idlist": ["37952131", "34706925"]}}
        )
        router.get(f"{EUTILS}/efetch.fcgi").respond(text=pubmed_xml)
        router.get(API).respond(text=ctgov_json, headers={"content-type": "application/json"})
        yield router


async def _agent(settings, llm, tmp_path) -> tuple[EvidenceAgent, httpx.AsyncClient]:
    http = httpx.AsyncClient()
    deps = Dependencies(
        settings=settings,
        llm=llm,
        pubmed=PubMedClient(http),
        trials=ClinicalTrialsClient(http),
        store=VectorStore(HashingEmbedder(), persist_dir=tmp_path / "chroma"),
    )
    return EvidenceAgent(deps), http


async def test_happy_path_is_grounded_and_writes_provenance(
    settings, mocked_apis, tmp_path
) -> None:
    llm = ScriptedLLM(
        {
            SearchPlan: [PLAN],
            EvidenceAssessment: [SUFFICIENT],
            DraftAnswer: [GOOD_DRAFT.model_copy(deep=True)],
            VerificationResult: [ALL_SUPPORTED],
        }
    )
    agent, http = await _agent(settings, llm, tmp_path)
    async with http:
        result = await agent.ask(QUESTION)

    assert [step for step, _ in llm.calls] == ["plan", "assess", "generate", "verify"]
    assert result.grounded and result.issues == []
    assert result.retrieval_rounds == 1 and result.revisions == 0
    # Citations are normalized to canonical ids.
    assert result.answer.claims[1].citations == ["NCT04972721"]
    # Cited sources are returned first.
    assert {s.source_id for s in result.sources[:2]} == {"PMID:37952131", "NCT04972721"}
    assert result.usage.llm_calls == 4

    # The generation prompt contains the evidence headers the model must cite from.
    generate_prompt = dict(llm.calls)["generate"]
    assert "source_id=PMID:37952131" in generate_prompt

    records = list(settings.runs_dir.rglob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["run_id"] == result.run_id
    assert record["queries"][0].startswith("pubmed: semaglutide")
    assert record["config"]["embedding_model"] == "hashing-256"
    assert all(len(p["content_sha256"]) == 16 for p in record["context_passages"])


async def test_unsupported_claim_triggers_one_revision(settings, mocked_apis, tmp_path) -> None:
    bad = DraftAnswer(
        summary="s",
        claims=[
            Claim(text="SELECT enrolled 17,604 patients.", citations=["PMID:37952131"]),
            Claim(text="Made-up claim.", citations=["PMID:11111111"]),
        ],
        evidence_quality="q",
        limitations="l",
    )
    llm = ScriptedLLM(
        {
            SearchPlan: [PLAN],
            EvidenceAssessment: [SUFFICIENT],
            DraftAnswer: [bad, GOOD_DRAFT.model_copy(deep=True)],
            VerificationResult: [
                VerificationResult(
                    verdicts=[ClaimVerdict(claim_index=0, supported=True, reason="ok")]
                ),
                ALL_SUPPORTED,
            ],
        }
    )
    agent, http = await _agent(settings, llm, tmp_path)
    async with http:
        result = await agent.ask(QUESTION)

    steps = [s for s, _ in llm.calls]
    assert steps == ["plan", "assess", "generate", "verify", "generate", "verify"]
    # The second generation receives reviewer feedback about the unknown source.
    second_generate = [p for s, p in llm.calls if s == "generate"][1]
    assert "PMID:11111111 was not in the retrieved evidence" in second_generate
    assert result.grounded and result.revisions == 1


async def test_insufficient_evidence_runs_followup_retrieval(
    settings, mocked_apis, tmp_path
) -> None:
    llm = ScriptedLLM(
        {
            SearchPlan: [PLAN],
            EvidenceAssessment: [
                EvidenceAssessment(
                    sufficient=False,
                    missing="no data on stroke",
                    followup_pubmed_queries=["semaglutide stroke"],
                )
            ],
            DraftAnswer: [GOOD_DRAFT.model_copy(deep=True)],
            VerificationResult: [ALL_SUPPORTED],
        }
    )
    agent, http = await _agent(settings, llm, tmp_path)
    async with http:
        result = await agent.ask(QUESTION)

    # Second round hits max_retrieval_rounds=2, so assess skips its LLM call.
    assert [s for s, _ in llm.calls] == ["plan", "assess", "generate", "verify"]
    assert result.retrieval_rounds == 2


async def test_no_results_returns_no_evidence_answer(settings, tmp_path) -> None:
    llm = ScriptedLLM(
        {
            SearchPlan: [PLAN.model_copy(update={"trials_query": None})],
            EvidenceAssessment: [SUFFICIENT],
        }
    )
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{EUTILS}/esearch.fcgi").respond(json={"esearchresult": {"idlist": []}})
        agent, http = await _agent(settings, llm, tmp_path)
        async with http:
            result = await agent.ask(QUESTION)

    assert result.answer.claims == []
    assert "No relevant" in result.answer.summary
    assert result.sources == []
    assert [s for s, _ in llm.calls] == ["plan", "assess"]


async def test_stream_emits_progress_events(settings, mocked_apis, tmp_path) -> None:
    llm = ScriptedLLM(
        {
            SearchPlan: [PLAN],
            EvidenceAssessment: [SUFFICIENT],
            DraftAnswer: [GOOD_DRAFT.model_copy(deep=True)],
            VerificationResult: [ALL_SUPPORTED],
        }
    )
    agent, http = await _agent(settings, llm, tmp_path)
    async with http:
        events = [e async for e in agent.stream(QUESTION)]

    assert [e["node"] for e in events] == [
        "plan",
        "retrieve",
        "assess",
        "generate",
        "verify",
        "done",
    ]
    retrieve = events[1]
    assert retrieve["new_sources"] == 5 and retrieve["round"] == 1
    assert events[-1]["result"]["grounded"] is True
