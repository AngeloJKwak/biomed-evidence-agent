"""Thin structured-output layer over the Anthropic SDK.

Every LLM step in the graph asks for a Pydantic model back, using the SDK's
`messages.parse` helper (JSON-schema constrained decoding). Token usage is
accumulated per run so it can be logged in the provenance record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from biomed_agent.config import Effort, Settings
from biomed_agent.schemas import UsageSummary
from biomed_agent.tracing import observe, record_generation

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


@dataclass
class CallRecord:
    step: str
    model: str
    effort: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    latency_s: float
    request_id: str | None = None


@dataclass
class UsageTracker:
    calls: list[CallRecord] = field(default_factory=list)

    def summary(self) -> UsageSummary:
        return UsageSummary(
            input_tokens=sum(c.input_tokens for c in self.calls),
            output_tokens=sum(c.output_tokens for c in self.calls),
            cache_read_input_tokens=sum(c.cache_read_input_tokens for c in self.calls),
            llm_calls=len(self.calls),
        )


class StructuredLLM(Protocol):
    model: str

    async def generate(
        self,
        *,
        step: str,
        system: str,
        prompt: str,
        schema: type[T],
        effort: Effort,
        tracker: UsageTracker | None = None,
    ) -> T: ...


class ClaudeLLM:
    def __init__(
        self,
        model: str,
        max_tokens: int = 16000,
        fallbacks: bool = True,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._fallbacks = fallbacks
        # Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self._client = client or anthropic.AsyncAnthropic()

    @classmethod
    def from_settings(cls, settings: Settings, model: str | None = None) -> ClaudeLLM:
        return cls(
            model=model or settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            fallbacks=settings.llm_fallbacks,
        )

    @observe(name="claude", as_type="generation")
    async def generate(
        self,
        *,
        step: str,
        system: str,
        prompt: str,
        schema: type[T],
        effort: Effort,
        tracker: UsageTracker | None = None,
    ) -> T:
        extra: dict = {}
        if self._fallbacks:
            # On a safety-classifier decline, the API re-runs the request on a
            # recommended fallback model instead of returning a refusal.
            extra = {
                "extra_headers": {"anthropic-beta": FALLBACK_BETA},
                "extra_body": {"fallbacks": "default"},
            }

        start = time.perf_counter()
        response = await self._client.messages.parse(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
            **extra,
        )
        latency = time.perf_counter() - start
        record_generation(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            metadata={"step": step, "effort": effort, "stop_reason": response.stop_reason},
        )

        if tracker is not None:
            usage = response.usage
            tracker.calls.append(
                CallRecord(
                    step=step,
                    model=response.model,
                    effort=effort,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens or 0,
                    latency_s=round(latency, 3),
                    request_id=getattr(response, "_request_id", None),
                )
            )

        if response.stop_reason == "refusal":
            details = response.stop_details
            raise LLMRefusal(
                f"{step}: model declined"
                + (f" ({details.category}): {details.explanation}" if details else "")
            )
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{step}: output truncated at max_tokens={self._max_tokens}")
        if response.parsed_output is None:
            raise LLMError(f"{step}: no structured output returned")
        return response.parsed_output
