"""Optional Langfuse tracing.

Enabled when the `tracing` extra is installed and LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY are set (plus LANGFUSE_HOST for self-hosted). Otherwise
every helper here is a no-op, so the agent never depends on a tracing backend.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _enabled() -> bool:
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    try:
        import langfuse  # noqa: F401
    except ImportError:
        return False
    return True


TRACING_ENABLED = _enabled()


def observe(name: str, as_type: str | None = None) -> Callable[[F], F]:
    if not TRACING_ENABLED:
        return lambda fn: fn
    from langfuse import observe as lf_observe

    return lf_observe(name=name, as_type=as_type)  # type: ignore[arg-type,return-value]


def record_generation(
    *, model: str, input_tokens: int, output_tokens: int, metadata: dict[str, Any]
) -> None:
    if not TRACING_ENABLED:
        return
    from langfuse import get_client

    get_client().update_current_generation(
        model=model,
        usage_details={"input": input_tokens, "output": output_tokens},
        metadata=metadata,
    )


def flush() -> None:
    if TRACING_ENABLED:
        from langfuse import get_client

        get_client().flush()
