from __future__ import annotations

import json

from fastapi.testclient import TestClient

from biomed_agent.api import app


class FakeAgent:
    entrypoints: list[str] = []

    async def stream(self, question: str, *, entrypoint: str = "library", session_id=None):
        self.entrypoints.append(entrypoint)
        yield {"node": "plan", "queries": ["q"], "trials_query": None}
        yield {"node": "done", "result": {"run_id": "abc", "question": question}}


def _client() -> TestClient:
    # No `with` block: skip the lifespan so the real agent (and its embedding model) isn't built.
    client = TestClient(app)
    app.state.agent = FakeAgent()
    return client


def test_stream_endpoint_emits_progress_then_result() -> None:
    resp = _client().post("/api/ask/stream", json={"question": "Does aspirin prevent strokes?"})
    assert resp.status_code == 200
    events = [
        (block.split("\n")[0].removeprefix("event: "), block.split("data: ", 1)[1])
        for block in resp.text.replace("\r\n", "\n").strip().split("\n\n")
    ]
    assert [e for e, _ in events] == ["progress", "result"]
    result = json.loads(events[1][1])
    assert result == {"run_id": "abc", "question": "Does aspirin prevent strokes?"}
    assert FakeAgent.entrypoints[-1] == "web"


def test_question_validation() -> None:
    resp = _client().post("/api/ask/stream", json={"question": "hi"})
    assert resp.status_code == 422
