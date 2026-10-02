from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from biomed_agent.config import Settings
from biomed_agent.llm import CallRecord, UsageTracker

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def pubmed_xml() -> str:
    return (FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8")


@pytest.fixture
def ctgov_json() -> str:
    return (FIXTURES / "ctgov_search.json").read_text(encoding="utf-8")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        chroma_dir=tmp_path / "chroma",
        runs_dir=tmp_path / "runs",
        max_retrieval_rounds=2,
        max_revisions=1,
    )


class ScriptedLLM:
    """Returns queued responses per schema, in order. Records every call."""

    model = "scripted"

    def __init__(self, responses: dict[type[BaseModel], list[BaseModel]]) -> None:
        self._responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[tuple[str, str]] = []

    async def generate(self, *, step, system, prompt, schema, effort, tracker=None):
        self.calls.append((step, prompt))
        if tracker is not None:
            tracker.calls.append(CallRecord(step, self.model, effort, 100, 50, 0, 0.0))
        queue = self._responses.get(schema)
        if not queue:
            raise AssertionError(f"no scripted response left for {schema.__name__} ({step})")
        return queue.pop(0)


@pytest.fixture
def tracker() -> UsageTracker:
    return UsageTracker()
