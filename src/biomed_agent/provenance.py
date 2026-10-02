"""Run records: one JSON file per answer with everything needed to audit or reproduce it.

Each record captures the code version, model and prompt versions, every search
query, the exact passages shown to the model (with content hashes), per-call
token usage, and the final answer with its citation issues.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biomed_agent.config import Settings
    from biomed_agent.llm import CallRecord
    from biomed_agent.schemas import AnswerResponse, Chunk


@lru_cache
def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_run_record(
    settings: Settings,
    response: AnswerResponse,
    *,
    queries: list[str],
    chunks: list[Chunk],
    calls: list[CallRecord],
    embedding_model: str,
) -> dict:
    return {
        "run_id": response.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "code_version": _git_commit(),
        "config": {
            "llm_model": settings.llm_model,
            "efforts": {
                "plan": settings.plan_effort,
                "answer": settings.answer_effort,
                "verify": settings.verify_effort,
            },
            "embedding_model": embedding_model,
            "prompt_version": settings.prompt_version,
            "top_k": settings.top_k,
            "chunk_chars": settings.chunk_chars,
            "max_retrieval_rounds": settings.max_retrieval_rounds,
            "max_revisions": settings.max_revisions,
        },
        "question": response.question,
        "queries": queries,
        "sources": [
            {
                "source_id": d.source_id,
                "url": d.url,
                "title": d.title,
                "year": d.year,
                "content_sha256": sha256(d.text),
            }
            for d in response.sources
        ],
        "context_passages": [
            {"chunk_id": c.chunk_id, "score": c.score, "content_sha256": sha256(c.text)}
            for c in chunks
        ],
        "llm_calls": [asdict(c) for c in calls],
        "answer": response.answer.model_dump(),
        "issues": [i.model_dump() for i in response.issues],
        "grounded": response.grounded,
        "latency_s": response.latency_s,
    }


def write_run_record(settings: Settings, response: AnswerResponse, **kwargs) -> Path:
    record = build_run_record(settings, response, **kwargs)
    day = record["created_at"][:10]
    out_dir = settings.runs_dir / day
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{response.run_id}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
