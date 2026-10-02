"""Optional Langfuse tracing.

Enabled when the `tracing` extra is installed and LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY are set (plus LANGFUSE_BASE_URL for a non-default region or
self-hosted instance). Otherwise every helper here is a no-op, so the agent
never depends on a tracing backend.

The check happens at call time rather than import time, so keys loaded from
`.env` by an entry point are picked up even though modules were imported first.

Trace shape for one question:

    evidence-agent (agent)               <- trace_run(); input, output, scores
      plan (chain)
        claude (generation)              <- model, tokens, effort
      retrieve (retriever)
      assess (chain)
        claude (generation)
      generate (chain) ...
      verify (evaluator) ...
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
R = TypeVar("R")

log = logging.getLogger("biomed_agent.tracing")


def _safe(fn: Callable[..., R]) -> Callable[..., R | None]:
    """Tracing must never break an answer: log backend errors and carry on."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> R | None:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - any tracing failure is non-fatal
            log.warning("Langfuse %s failed: %s", fn.__name__, e)
            return None

    return wrapper


@functools.cache
def _langfuse_installed() -> bool:
    try:
        import langfuse  # noqa: F401
    except ImportError:
        return False
    return True


def tracing_enabled() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"
        and _langfuse_installed()
    )


def observe(name: str, as_type: str | None = None) -> Callable[[F], F]:
    """Decorate an async function as a Langfuse observation (no-op when tracing is off)."""

    def decorator(fn: F) -> F:
        traced: Callable[..., Awaitable[Any]] | None = None

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal traced
            if not tracing_enabled():
                return await fn(*args, **kwargs)
            if traced is None:
                from langfuse import observe as lf_observe

                traced = lf_observe(name=name, as_type=as_type)(fn)  # type: ignore[arg-type]
            return await traced(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@_safe
def record_generation(
    *, model: str, input_tokens: int, output_tokens: int, metadata: dict[str, Any]
) -> None:
    if not tracing_enabled():
        return
    from langfuse import get_client

    get_client().update_current_generation(
        model=model,
        usage_details={"input": input_tokens, "output": output_tokens},
        metadata=metadata,
    )


class RunTrace:
    """Handle on the root observation of one agent run. No-op when tracing is off."""

    def __init__(self, client: Any = None, observation: Any = None) -> None:
        self._client = client
        self._obs = observation

    @functools.cached_property
    @_safe
    def trace_id(self) -> str | None:
        return self._client.get_current_trace_id() if self._client else None

    @functools.cached_property
    @_safe
    def trace_url(self) -> str | None:
        # Looks up the project id over the network on first use (cached by the SDK).
        if self._client is None or self.trace_id is None:
            return None
        return self._client.get_trace_url(trace_id=self.trace_id)

    @_safe
    def set_output(self, output: Any) -> None:
        if self._obs is not None:
            self._obs.update(output=output)

    @_safe
    def score(self, name: str, value: float, *, boolean: bool = False, comment: str = "") -> None:
        if self._client is not None:
            self._client.score_current_trace(
                name=name,
                value=value,
                data_type="BOOLEAN" if boolean else "NUMERIC",
                comment=comment or None,
            )


@contextmanager
def trace_run(
    name: str, *, input: Any, metadata: dict[str, Any], tags: list[str]
) -> Iterator[RunTrace]:
    """Open the root observation for one agent run; nested observations attach to it."""
    if not tracing_enabled():
        yield RunTrace()
        return
    from langfuse import get_client, propagate_attributes

    client = get_client()
    with (
        client.start_as_current_observation(
            name=name, as_type="agent", input=input, metadata=metadata
        ) as obs,
        propagate_attributes(trace_name=name, tags=tags),
    ):
        yield RunTrace(client, obs)


@_safe
def score_trace(trace_id: str, name: str, value: float, comment: str = "") -> None:
    """Attach a score to an existing trace (used by the eval harness)."""
    if not tracing_enabled():
        return
    from langfuse import get_client

    get_client().create_score(
        trace_id=trace_id, name=name, value=value, data_type="NUMERIC", comment=comment or None
    )


@_safe
def flush() -> None:
    if tracing_enabled():
        from langfuse import get_client

        get_client().flush()
