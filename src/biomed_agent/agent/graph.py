"""The evidence agent as a LangGraph state machine.

    plan -> retrieve -> assess --(insufficient, rounds left)--> retrieve
                           |
                           v
                       generate -> verify --(issues, revisions left)--> generate
                                      |
                                      v
                                  finalize

LLM steps use structured outputs; retrieval and citation checks are plain code.
"""

from __future__ import annotations

import asyncio
import operator
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from biomed_agent.agent import prompts
from biomed_agent.config import Settings
from biomed_agent.grounding import check_citations, normalize_citations
from biomed_agent.llm import StructuredLLM, UsageTracker
from biomed_agent.provenance import code_version, write_run_record
from biomed_agent.retrieval.store import VectorStore
from biomed_agent.schemas import (
    AnswerResponse,
    Chunk,
    CitationIssue,
    DraftAnswer,
    EvidenceAssessment,
    SearchPlan,
    SourceDocument,
    VerificationResult,
)
from biomed_agent.sources.clinicaltrials import ClinicalTrialsClient
from biomed_agent.sources.pubmed import PubMedClient
from biomed_agent.tracing import RunTrace, observe, trace_run, update_observation


def _merge_docs(
    left: dict[str, SourceDocument], right: dict[str, SourceDocument]
) -> dict[str, SourceDocument]:
    return {**left, **right}


class AgentState(TypedDict, total=False):
    question: str
    # Queued searches for the next retrieval round.
    pending_pubmed: list[str]
    pending_trials: str | None
    queries_run: Annotated[list[str], operator.add]
    docs: Annotated[dict[str, SourceDocument], _merge_docs]
    chunks: list[Chunk]
    retrieval_rounds: int
    draft: DraftAnswer | None
    issues: list[CitationIssue]
    feedback: str | None
    revisions: int
    grounded: bool


@dataclass
class Dependencies:
    settings: Settings
    llm: StructuredLLM
    pubmed: PubMedClient
    trials: ClinicalTrialsClient
    store: VectorStore


NO_EVIDENCE = DraftAnswer(
    summary="No relevant PubMed abstracts or registered trials were found for this question.",
    claims=[],
    evidence_quality="No evidence retrieved.",
    limitations="Try rephrasing the question with more specific drug, condition, or outcome terms.",
)


def build_graph(deps: Dependencies, tracker: UsageTracker):
    s = deps.settings

    @observe(name="plan-searches", as_type="chain")
    async def plan(state: AgentState) -> dict[str, Any]:
        update_observation(input={"question": state["question"]})
        result = await deps.llm.generate(
            step="plan",
            system=prompts.PLANNER_SYSTEM,
            prompt=prompts.planner_prompt(state["question"]),
            schema=SearchPlan,
            effort=s.plan_effort,
            tracker=tracker,
        )
        update_observation(output=result.model_dump())
        return {
            "pending_pubmed": result.pubmed_queries[:3],
            "pending_trials": result.trials_query,
        }

    @observe(name="retrieve-evidence", as_type="retriever")
    async def retrieve(state: AgentState) -> dict[str, Any]:
        known = state.get("docs", {})
        pubmed_qs = state.get("pending_pubmed", [])
        trials_q = state.get("pending_trials")
        round_ = state.get("retrieval_rounds", 0) + 1
        update_observation(
            input={"pubmed_queries": pubmed_qs, "trials_query": trials_q, "round": round_}
        )

        id_lists = await asyncio.gather(
            *(deps.pubmed.search(q, s.pubmed_max_results) for q in pubmed_qs)
        )
        new_pmids = list(
            dict.fromkeys(p for ids in id_lists for p in ids if f"PMID:{p}" not in known)
        )
        fetched, trials = await asyncio.gather(
            deps.pubmed.fetch(new_pmids),
            deps.trials.search(trials_q, s.trials_max_results) if trials_q else _empty(),
        )
        new_docs = {d.source_id: d for d in [*fetched, *trials] if d.source_id not in known}

        # Embedding is CPU-bound; keep it off the event loop.
        await asyncio.to_thread(deps.store.add_documents, list(new_docs.values()))
        all_ids = [*known.keys(), *new_docs.keys()]
        chunks = await asyncio.to_thread(deps.store.search, state["question"], all_ids, s.top_k)

        all_docs = {**known, **new_docs}
        update_observation(
            # The ranked passages are exactly what the model will see as evidence.
            output=[
                {
                    "source_id": c.source_id,
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                    "title": all_docs[c.source_id].title if c.source_id in all_docs else None,
                    "text": c.text,
                }
                for c in chunks
            ],
            metadata={
                "pmids_found": len(new_pmids),
                "pubmed_docs_added": len(fetched),
                "trials_added": len(trials),
                "total_sources": len(all_docs),
                "passages_returned": len(chunks),
                "top_k": s.top_k,
                "embedding_model": deps.store.embedder.name,
            },
        )
        return {
            "docs": new_docs,
            "chunks": chunks,
            "queries_run": [*(f"pubmed: {q}" for q in pubmed_qs)]
            + ([f"trials: {trials_q}"] if trials_q else []),
            "pending_pubmed": [],
            "pending_trials": None,
            "retrieval_rounds": round_,
        }

    @observe(name="assess-coverage", as_type="chain")
    async def assess(state: AgentState) -> dict[str, Any]:
        update_observation(
            input={
                "question": state["question"],
                "passages": len(state["chunks"]),
                "queries_run": state["queries_run"],
            }
        )
        if state["retrieval_rounds"] >= s.max_retrieval_rounds:
            update_observation(
                output={"decision": "proceed", "reason": "max retrieval rounds reached"}
            )
            return {}
        evidence = prompts.format_evidence(state["chunks"], state["docs"])
        result = await deps.llm.generate(
            step="assess",
            system=prompts.ASSESSOR_SYSTEM,
            prompt=prompts.assessor_prompt(state["question"], evidence, state["queries_run"]),
            schema=EvidenceAssessment,
            effort=s.plan_effort,
            tracker=tracker,
        )
        update_observation(output=result.model_dump())
        if result.sufficient or not (
            result.followup_pubmed_queries or result.followup_trials_query
        ):
            return {}
        return {
            "pending_pubmed": result.followup_pubmed_queries[:2],
            "pending_trials": result.followup_trials_query,
        }

    def after_assess(state: AgentState) -> str:
        if state.get("pending_pubmed") or state.get("pending_trials"):
            return "retrieve"
        return "generate"

    @observe(name="generate-answer", as_type="chain")
    async def generate(state: AgentState) -> dict[str, Any]:
        update_observation(
            input={
                "question": state["question"],
                "passages": len(state.get("chunks", [])),
                "revision_feedback": state.get("feedback"),
            }
        )
        if not state.get("chunks"):
            update_observation(output=NO_EVIDENCE.model_dump(), metadata={"no_evidence": True})
            return {"draft": NO_EVIDENCE.model_copy(deep=True)}
        evidence = prompts.format_evidence(state["chunks"], state["docs"])
        draft = await deps.llm.generate(
            step="generate",
            system=prompts.ANSWER_SYSTEM,
            prompt=prompts.answer_prompt(state["question"], evidence, state.get("feedback")),
            schema=DraftAnswer,
            effort=s.answer_effort,
            tracker=tracker,
        )
        draft = normalize_citations(draft)
        update_observation(output=draft.model_dump())
        return {"draft": draft}

    @observe(name="verify-citations", as_type="evaluator")
    async def verify(state: AgentState) -> dict[str, Any]:
        draft = state["draft"]
        assert draft is not None
        update_observation(input={"claims": [c.model_dump() for c in draft.claims]})
        cited_ids = {c.source_id for c in state.get("chunks", [])}
        issues = check_citations(draft, cited_ids)

        checkable = [
            i
            for i, c in enumerate(draft.claims)
            if c.citations and all(x in cited_ids for x in c.citations)
        ]
        if checkable:
            evidence = prompts.format_evidence(state["chunks"], state["docs"])
            result = await deps.llm.generate(
                step="verify",
                system=prompts.VERIFIER_SYSTEM,
                prompt=prompts.verifier_prompt(draft, evidence),
                schema=VerificationResult,
                effort=s.verify_effort,
                tracker=tracker,
            )
            for v in result.verdicts:
                if not v.supported and v.claim_index in checkable:
                    issues.append(
                        CitationIssue(
                            claim_index=v.claim_index, kind="unsupported", detail=v.reason
                        )
                    )

        feedback = (
            "\n".join(f"- Claim {i.claim_index} ({i.kind}): {i.detail}" for i in issues) or None
        )
        update_observation(
            output={"grounded": not issues, "issues": [i.model_dump() for i in issues]},
            metadata={"llm_checked_claims": len(checkable)},
            level="WARNING" if issues else None,
            status_message=f"{len(issues)} citation issue(s)" if issues else None,
        )
        return {"issues": issues, "grounded": not issues, "feedback": feedback}

    def after_verify(state: AgentState) -> str:
        if state["grounded"] or state.get("revisions", 0) >= s.max_revisions:
            return END
        return "revise"

    def revise(state: AgentState) -> dict[str, Any]:
        return {"revisions": state.get("revisions", 0) + 1}

    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("assess", assess)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("revise", revise)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "assess")
    g.add_conditional_edges("assess", after_assess, ["retrieve", "generate"])
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, ["revise", END])
    g.add_edge("revise", "generate")
    return g.compile()


async def _empty() -> list[SourceDocument]:
    return []


class EvidenceAgent:
    def __init__(self, deps: Dependencies) -> None:
        self.deps = deps

    async def stream(
        self, question: str, *, entrypoint: str = "library", session_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield one progress event per graph node, then `{"node": "done", "result": ...}`.

        `entrypoint` ("cli", "web", "eval", ...) becomes a trace tag; `session_id`
        groups related traces in Langfuse (e.g. every question in one eval run).

        The graph runs in its own task inside a single trace span, and events are
        handed back through a queue. Opening the span inside this generator instead
        would break tracing context across `yield`s (and across SSE clients).
        """
        queue: asyncio.Queue[dict[str, Any] | BaseException | None] = asyncio.Queue()
        task = asyncio.create_task(self._run(question, queue, entrypoint, session_id))
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not task.done():  # consumer went away (e.g. browser closed the stream)
                task.cancel()

    async def _run(
        self,
        question: str,
        queue: asyncio.Queue[dict[str, Any] | BaseException | None],
        entrypoint: str,
        session_id: str | None,
    ) -> None:
        run_id = uuid.uuid4().hex[:12]
        tracker = UsageTracker()
        graph = build_graph(self.deps, tracker)
        started = time.perf_counter()
        state: dict[str, Any] = {"question": question}
        s = self.deps.settings
        try:
            with trace_run(
                "answer-question",
                input={"question": question},
                # Propagated to every observation; keys must be alphanumeric.
                metadata={"runId": run_id, "promptVersion": s.prompt_version},
                tags=[f"entrypoint:{entrypoint}"],
                session_id=session_id,
                version=code_version(),
            ) as trace:
                async for update in graph.astream({"question": question}, stream_mode="updates"):
                    for node, delta in update.items():
                        delta = delta or {}
                        for key, value in delta.items():
                            if key == "docs":
                                state.setdefault("docs", {}).update(value)
                            elif key == "queries_run":
                                state.setdefault("queries_run", []).extend(value)
                            else:
                                state[key] = value
                        await queue.put({"node": node, **_progress(node, delta, state)})

                result = self._finish(
                    run_id, question, state, tracker, time.perf_counter() - started, trace
                )
                trace.set_output(answer_markdown(result))
                trace.score("grounded", float(result.grounded), boolean=True)
                trace.score("citation_issues", float(len(result.issues)))
                trace.score("revisions", float(result.revisions))
                await queue.put({"node": "done", "result": result.model_dump(mode="json")})
        except BaseException as e:  # surface failures (incl. cancellation) to the consumer
            await queue.put(e)
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            await queue.put(None)

    async def ask(
        self, question: str, *, entrypoint: str = "library", session_id: str | None = None
    ) -> AnswerResponse:
        final: dict[str, Any] | None = None
        async for event in self.stream(question, entrypoint=entrypoint, session_id=session_id):
            if event["node"] == "done":
                final = event["result"]
        assert final is not None
        return AnswerResponse.model_validate(final)

    def _finish(
        self,
        run_id: str,
        question: str,
        state: dict[str, Any],
        tracker: UsageTracker,
        latency: float,
        trace: RunTrace | None = None,
    ) -> AnswerResponse:
        draft: DraftAnswer = state.get("draft") or NO_EVIDENCE
        docs: dict[str, SourceDocument] = state.get("docs", {})
        cited = {c for claim in draft.claims for c in claim.citations}
        # Return cited sources first, then the rest of what was used as context.
        context_ids = list(dict.fromkeys(c.source_id for c in state.get("chunks", [])))
        ordered = [i for i in context_ids if i in cited] + [
            i for i in context_ids if i not in cited
        ]

        response = AnswerResponse(
            run_id=run_id,
            question=question,
            answer=draft,
            sources=[docs[i] for i in ordered if i in docs],
            issues=state.get("issues", []),
            grounded=state.get("grounded", False),
            retrieval_rounds=state.get("retrieval_rounds", 0),
            revisions=state.get("revisions", 0),
            latency_s=round(latency, 2),
            usage=tracker.summary(),
            trace_id=trace.trace_id if trace else None,
            trace_url=trace.trace_url if trace else None,
        )
        write_run_record(
            self.deps.settings,
            response,
            queries=state.get("queries_run", []),
            chunks=state.get("chunks", []),
            calls=tracker.calls,
            embedding_model=self.deps.store.embedder.name,
        )
        return response


def answer_markdown(result: AnswerResponse) -> str:
    """Trace-level output: what a reviewer needs to judge the answer at a glance."""
    a = result.answer
    lines = [a.summary, ""]
    lines += [f"{i + 1}. {c.text} [{', '.join(c.citations)}]" for i, c in enumerate(a.claims)]
    lines += [
        "",
        f"**Evidence quality:** {a.evidence_quality}",
        f"**Limitations:** {a.limitations}",
    ]
    if result.issues:
        lines += ["", "**Citation issues:**"]
        lines += [f"- claim {i.claim_index + 1} ({i.kind}): {i.detail}" for i in result.issues]
    return "\n".join(lines)


def _progress(node: str, delta: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Small, UI-friendly summary of what a node just did."""
    if node == "plan":
        return {
            "queries": delta.get("pending_pubmed", []),
            "trials_query": delta.get("pending_trials"),
        }
    if node == "retrieve":
        return {
            "round": delta.get("retrieval_rounds"),
            "new_sources": len(delta.get("docs", {})),
            "total_sources": len(state.get("docs", {})),
            "passages": len(delta.get("chunks", [])),
        }
    if node == "assess":
        return {"followup": bool(delta.get("pending_pubmed") or delta.get("pending_trials"))}
    if node == "generate":
        draft = delta.get("draft")
        return {"claims": len(draft.claims) if draft else 0}
    if node == "verify":
        return {"grounded": delta.get("grounded"), "issues": len(delta.get("issues", []))}
    return {}
