from __future__ import annotations

import pytest

from biomed_agent import tracing
from biomed_agent.config import load_env


def test_tracing_disabled_without_keys() -> None:
    assert tracing.tracing_enabled() is False


def test_tracing_enabled_with_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert tracing.tracing_enabled() is True
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    assert tracing.tracing_enabled() is False


async def test_observe_is_passthrough_when_disabled() -> None:
    @tracing.observe(name="x")
    async def double(n: int) -> int:
        return n * 2

    assert await double(21) == 42


def test_trace_run_noop_when_disabled() -> None:
    with tracing.trace_run("t", input={}, metadata={}, tags=[]) as run:
        run.score("grounded", 1.0, boolean=True)
        run.set_output({"a": 1})
        assert run.trace_id is None and run.trace_url is None


class _BrokenClient:
    def __getattr__(self, name: str):
        def fail(*args, **kwargs):
            raise ConnectionError("langfuse unreachable")

        return fail


def test_backend_failures_never_raise(caplog: pytest.LogCaptureFixture) -> None:
    run = tracing.RunTrace(client=_BrokenClient(), observation=_BrokenClient())
    assert run.trace_id is None
    assert run.trace_url is None
    run.set_output({"a": 1})
    run.score("grounded", 1.0, boolean=True)
    assert "langfuse unreachable" in caplog.text


def test_load_env_exports_without_overriding(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("LANGFUSE_PUBLIC_KEY=from-file\nBIOMED_TEST_ONLY=1\n")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "from-shell")
    monkeypatch.delenv("BIOMED_TEST_ONLY", raising=False)
    load_env(env)
    import os

    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "from-shell"
    assert os.environ["BIOMED_TEST_ONLY"] == "1"
    monkeypatch.delenv("BIOMED_TEST_ONLY")
