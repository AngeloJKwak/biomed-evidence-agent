"""Optional Langfuse tracing, following https://langfuse.com/docs/observability/best-practices.

Enabled when the `tracing` extra is installed and LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY are set (plus LANGFUSE_BASE_URL for the project's region or a
self-hosted instance). Otherwise every helper here is a no-op, so the agent never
depends on a tracing backend. The check happens at call time, so keys loaded from
`.env` by an entry point are picked up even though modules were imported first.

Trace shape for one question:

    answer-question (agent)            input: question · output: answer (markdown) · scores
      plan-searches (chain)
        write-search-queries (generation)   model, effort, tokens, cost, thinking summary
      retrieve-evidence (retriever)         queries in, ranked passages out
      assess-coverage (chain)
        judge-evidence-coverage (generation)
      generate-answer (chain)
        draft-cited-answer (generation)
      verify-citations (evaluator)
        check-claim-support (generation)

Conventions:
- Every decorated function sets its input/output explicitly (`capture_*=False`), so
  internal objects and full agent state never end up in a trace.
- Observation names are verb-first and stable; the model is an attribute, never a name.
- Inputs/outputs/metadata pass through a PII mask at export time (see `mask_pii`).
- Backend errors are logged and swallowed: tracing must never break an answer.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
R = TypeVar("R")

log = logging.getLogger("biomed_agent.tracing")


def _safe(fn: Callable[..., R]) -> Callable[..., R | None]:
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


# --------------------------------------------------------------------------- #
# PII masking
# --------------------------------------------------------------------------- #

# Questions are free text and may contain personal health details. These patterns
# catch common direct identifiers; they are a safety net, not de-identification.
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b(?:MRN|medical record (?:number|no\.?))[:#\s]*\w+", re.I), "[MRN]"),
    (re.compile(r"\b(?:DOB|date of birth)[:\s]*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", re.I), "[DOB]"),
]

# Span attributes that carry user-visible content.
_MASKED_ATTRIBUTE_PREFIXES = (
    "langfuse.observation.input",
    "langfuse.observation.output",
    "langfuse.observation.metadata",
    "langfuse.trace.input",
    "langfuse.trace.output",
    "langfuse.trace.metadata",
)


def mask_text(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def mask_pii(*, params: Any) -> Any:
    """`mask_otel_spans` hook: redact PII in content attributes before export."""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        changed = {
            key: masked
            for key, value in span.attributes.items()
            if isinstance(value, str)
            and key.startswith(_MASKED_ATTRIBUTE_PREFIXES)
            and (masked := mask_text(value)) != value
        }
        if changed:
            patches[identifier] = OtelSpanPatch(set_attributes=changed)
    return MaskOtelSpansResult(span_patches=patches)


@functools.cache
def _client() -> Any:
    """Create the process-wide Langfuse client once; `get_client()` then returns it.

    Environment (LANGFUSE_TRACING_ENVIRONMENT), release, and region come from env vars.
    """
    from langfuse import Langfuse

    mask = os.getenv("BIOMED_TRACE_MASK_PII", "true").lower() != "false"
    return Langfuse(mask_otel_spans=mask_pii if mask else None)


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


def observe(name: str, as_type: str | None = None) -> Callable[[F], F]:
    """Decorate an async function as a Langfuse observation (no-op when tracing is off).

    Automatic argument/return capture is off: call `update_observation` (or
    `update_generation`) inside the function with what a reviewer needs to see.
    """

    def decorator(fn: F) -> F:
        traced: Callable[..., Awaitable[Any]] | None = None

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal traced
            if not tracing_enabled():
                return await fn(*args, **kwargs)
            if traced is None:
                from langfuse import observe as lf_observe

                _client()
                traced = lf_observe(
                    name=name, as_type=as_type, capture_input=False, capture_output=False
                )(fn)  # type: ignore[arg-type]
            return await traced(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@_safe
def update_observation(
    *,
    input: Any = None,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
) -> None:
    """Set input/output/metadata on the current (non-generation) observation."""
    if not tracing_enabled():
        return
    _client().update_current_span(
        input=input, output=output, metadata=metadata, level=level, status_message=status_message
    )


@_safe
def update_generation(**fields: Any) -> None:
    """Set fields (input, output, model, usage_details, cost_details, ...) on the current
    generation. See `Langfuse.update_current_generation` for accepted keyword arguments."""
    if not tracing_enabled():
        return
    _client().update_current_generation(**fields)


# --------------------------------------------------------------------------- #
# Trace roots and scores
# --------------------------------------------------------------------------- #


class RunTrace:
    """Handle on the root observation of one run. No-op when tracing is off."""

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


_METADATA_KEY = re.compile(r"^[A-Za-z0-9]+$")


@contextmanager
def trace_run(
    name: str,
    *,
    input: Any,
    as_type: str = "agent",
    metadata: dict[str, str] | None = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
    version: str | None = None,
) -> Iterator[RunTrace]:
    """Open the root observation of a trace; nested observations attach to it.

    `metadata` is propagated to every child observation so traces can be filtered on
    it. Langfuse requires alphanumeric keys and string values of at most 200 chars;
    anything else is dropped here rather than silently by the backend.
    """
    if not tracing_enabled():
        yield RunTrace()
        return
    from langfuse import propagate_attributes

    propagated = {
        k: v for k, v in (metadata or {}).items() if _METADATA_KEY.match(k) and len(v) <= 200
    }
    client = _client()
    with (
        client.start_as_current_observation(name=name, as_type=as_type, input=input) as obs,
        propagate_attributes(
            trace_name=name,
            tags=tags,
            metadata=propagated or None,
            session_id=session_id,
            version=version,
        ),
    ):
        yield RunTrace(client, obs)


@_safe
def score_trace(trace_id: str, name: str, value: float, comment: str = "") -> None:
    """Attach a score to an existing trace (used by the eval harness)."""
    if not tracing_enabled():
        return
    _client().create_score(
        trace_id=trace_id, name=name, value=value, data_type="NUMERIC", comment=comment or None
    )


@_safe
def flush() -> None:
    if tracing_enabled():
        _client().flush()
