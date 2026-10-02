"""Evaluate the agent on a set of questions with known landmark-trial answers.

Metrics per question:
  source_recall    - share of expected sources (PMIDs / NCT ids) that reached the model's context
  cited_expected   - whether at least one expected source was cited in the answer
  citation_validity- share of citations pointing at retrieved sources (deterministic)
  citation_coverage- share of claims carrying at least one citation (deterministic)
  grounded         - passed the verifier with no citation issues
  fact_recall      - share of key facts the answer conveys (LLM judge, separate model)
  finding_correct  - answer's direction (positive / null) matches the trial's (LLM judge)

Every run makes real Claude API calls. Use --limit to try a couple of questions first.

    uv run python evals/run_eval.py --limit 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from biomed_agent.config import get_settings, load_env
from biomed_agent.factory import build_agent
from biomed_agent.grounding import citation_metrics
from biomed_agent.llm import MODEL_PRICES, ClaudeLLM, UsageTracker
from biomed_agent.schemas import AnswerResponse
from biomed_agent.tracing import flush, score_trace, trace_run

HERE = Path(__file__).parent

JUDGE_SYSTEM = """\
You grade a biomedical answer against reference facts taken from the abstract of a landmark trial.
For each reference fact, decide whether the answer conveys it: the same direction of effect and
population, with any numbers consistent (approximate values are fine). An answer that hedges but
still states the finding counts. Also decide whether the answer's overall conclusion matches the
reference finding type: "positive" means the intervention showed benefit on the main question,
"null" means it did not.
"""


class FactGrade(BaseModel):
    fact_index: int
    conveyed: bool
    reason: str


class JudgeResult(BaseModel):
    facts: list[FactGrade]
    finding_matches: bool = Field(description="Answer's conclusion matches the reference finding.")


def judge_prompt(item: dict, result: AnswerResponse) -> str:
    a = result.answer
    answer_text = "\n".join(
        [a.summary, *(f"- {c.text}" for c in a.claims), a.evidence_quality, a.limitations]
    )
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(item["key_facts"]))
    return (
        f"Question: {item['question']}\n\nReference finding type: {item['finding']}\n\n"
        f"Reference facts (0-indexed):\n{facts}\n\nAnswer to grade:\n{answer_text}"
    )


def cost_usd(usage_by_model: dict[str, tuple[int, int]]) -> float:
    total = 0.0
    for model, (tin, tout) in usage_by_model.items():
        pin, pout = MODEL_PRICES.get(model, (0.0, 0.0, 0.0, 0.0))[:2]
        total += tin / 1e6 * pin + tout / 1e6 * pout
    return round(total, 4)


async def evaluate(limit: int | None, ids: list[str] | None) -> dict:
    settings = get_settings()
    created_at = datetime.now(UTC).isoformat()
    # One Langfuse session per eval run groups every agent and judge trace it produced.
    session_id = f"eval-{created_at[:19].replace(':', '')}"
    items = [json.loads(line) for line in (HERE / "dataset.jsonl").read_text().splitlines() if line]
    if ids:
        items = [i for i in items if i["id"] in ids]
    if limit:
        items = items[:limit]

    agent, http = build_agent(settings)
    judge = ClaudeLLM.from_settings(settings, model=settings.judge_model)
    rows = []
    try:
        for item in items:
            print(f"> {item['id']}: {item['question']}")
            started = time.perf_counter()
            result = await agent.ask(item["question"], entrypoint="eval", session_id=session_id)
            judge_tracker = UsageTracker()
            with trace_run(
                "grade-answer",
                as_type="evaluator",
                input={
                    "question": item["question"],
                    "reference_finding": item["finding"],
                    "reference_facts": item["key_facts"],
                },
                metadata={"evalItem": item["id"], "agentTraceId": result.trace_id or ""},
                tags=["entrypoint:eval"],
                session_id=session_id,
            ) as judge_trace:
                verdict = await judge.generate(
                    step="judge",
                    system=JUDGE_SYSTEM,
                    prompt=judge_prompt(item, result),
                    schema=JudgeResult,
                    effort="medium",
                    tracker=judge_tracker,
                )
                judge_trace.set_output(verdict.model_dump())

            retrieved = {s.source_id for s in result.sources}
            cited = {c for claim in result.answer.claims for c in claim.citations}
            expected = set(item["expected_sources"])
            conveyed = [f.conveyed for f in verdict.facts]
            row = {
                "id": item["id"],
                "run_id": result.run_id,
                "source_recall": len(expected & retrieved) / len(expected),
                "cited_expected": bool(expected & cited),
                **citation_metrics(result.answer, retrieved),
                "grounded": result.grounded,
                "fact_recall": sum(conveyed) / len(conveyed) if conveyed else 0.0,
                "finding_correct": verdict.finding_matches,
                "retrieval_rounds": result.retrieval_rounds,
                "revisions": result.revisions,
                "latency_s": round(time.perf_counter() - started, 1),
                "cost_usd": cost_usd(
                    {
                        settings.llm_model: (result.usage.input_tokens, result.usage.output_tokens),
                        settings.judge_model: (
                            judge_tracker.summary().input_tokens,
                            judge_tracker.summary().output_tokens,
                        ),
                    }
                ),
                "judge_notes": [f.reason for f in verdict.facts if not f.conveyed],
                "trace_url": result.trace_url,
            }
            rows.append(row)
            if result.trace_id:
                # Eval metrics land on the run's trace, next to the agent's own scores.
                for metric in ("source_recall", "fact_recall", "citation_validity"):
                    score_trace(result.trace_id, f"eval_{metric}", float(row[metric]))
                score_trace(result.trace_id, "eval_finding_correct", float(row["finding_correct"]))
            print(
                f"  recall={row['source_recall']:.2f} cited={row['cited_expected']} "
                f"facts={row['fact_recall']:.2f} finding={row['finding_correct']} "
                f"grounded={row['grounded']} ${row['cost_usd']:.3f} {row['latency_s']}s"
            )
    finally:
        await http.aclose()
        flush()

    def mean(key: str) -> float:
        return round(statistics.mean(float(r[key]) for r in rows), 3) if rows else 0.0

    summary = {
        k: mean(k)
        for k in [
            "source_recall",
            "cited_expected",
            "citation_validity",
            "citation_coverage",
            "grounded",
            "fact_recall",
            "finding_correct",
            "latency_s",
        ]
    }
    summary["total_cost_usd"] = round(sum(r["cost_usd"] for r in rows), 3)
    return {
        "created_at": created_at,
        "langfuse_session_id": session_id,
        "config": {
            "llm_model": settings.llm_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "prompt_version": settings.prompt_version,
            "top_k": settings.top_k,
        },
        "n": len(rows),
        "summary": summary,
        "rows": rows,
    }


def to_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"# Eval report - {report['created_at'][:19]}Z",
        "",
        f"Model `{report['config']['llm_model']}` · judge `{report['config']['judge_model']}` · "
        f"embeddings `{report['config']['embedding_model']}` · n={report['n']}",
        "",
        "| Metric | Mean |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in s.items()),
        "",
        "| Question | Source recall | Cited expected | Fact recall | Finding | Grounded | Cost |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r['id']} | {r['source_recall']:.2f} | {'yes' if r['cited_expected'] else 'no'} | "
            f"{r['fact_recall']:.2f} | {'ok' if r['finding_correct'] else 'wrong'} | "
            f"{'yes' if r['grounded'] else 'no'} | ${r['cost_usd']:.3f} |"
            for r in report["rows"]
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Only run the first N questions")
    parser.add_argument("--ids", nargs="*", help="Only run these question ids")
    args = parser.parse_args()
    load_env()
    # Keep eval traffic out of development/production dashboards in Langfuse.
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")

    report = asyncio.run(evaluate(args.limit, args.ids))
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = report["created_at"][:19].replace(":", "")
    (out_dir / f"{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = to_markdown(report)
    (out_dir / f"{stamp}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
