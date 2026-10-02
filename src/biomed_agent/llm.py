"""Thin structured-output layer over the Anthropic SDK.

Every LLM step in the graph asks for a Pydantic model back, using the SDK's
`messages.parse` helper (JSON-schema constrained decoding). Token usage is
accumulated per run so it can be logged in the provenance record, and each call
is traced as a Langfuse generation with its prompt, output, thinking summary,
token usage, and cost.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from biomed_agent.config import Effort, Settings
from biomed_agent.schemas import UsageSummary
from biomed_agent.tracing import observe, update_generation

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
MODEL_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
}

# Trace names for each step's model call: verb-first, stable, never the model's name.
GENERATION_NAMES = {
    "plan": "write-search-queries",
    "assess": "judge-evidence-coverage",
    "generate": "draft-cited-answer",
    "verify": "check-claim-support",
    "judge": "grade-answer",
}


def usage_details(usage: anthropic.types.Usage) -> dict[str, int]:
    """Token buckets as Langfuse expects them: each token counted exactly once."""
    details = {"input": usage.input_tokens, "output": usage.output_tokens}
    if usage.cache_read_input_tokens:
        details["cache_read_input_tokens"] = usage.cache_read_input_tokens
    if usage.cache_creation_input_tokens:
        details["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
    return details


def cost_details(model: str, usage: dict[str, int]) -> dict[str, float] | None:
    """USD cost per token bucket, or None for models without a known price."""
    prices = MODEL_PRICES.get(model)
    if prices is None:
        return None
    p_in, p_out, p_cache_read, p_cache_write = prices
    per_bucket = {
        "input": p_in,
        "output": p_out,
        "cache_read_input_tokens": p_cache_read,
        "cache_creation_input_tokens": p_cache_write,
    }
    return {k: round(n / 1e6 * per_bucket[k], 8) for k, n in usage.items()}


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

    @observe(name="llm-call", as_type="generation")
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

        messages = [{"role": "user", "content": prompt}]
        # Summarized thinking costs nothing extra (thinking runs either way on this
        # model family) and puts the model's reasoning for each step in the trace.
        thinking = {"type": "adaptive", "display": "summarized"}
        update_generation(
            name=GENERATION_NAMES.get(step, f"{step}-llm-call"),
            input=[{"role": "system", "content": system}, *messages],
            model=self.model,
            model_parameters={
                "effort": effort,
                "max_tokens": self._max_tokens,
                "thinking": "adaptive/summarized",
                "output_schema": schema.__name__,
            },
        )

        start = time.perf_counter()
        response = await self._client.messages.parse(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
            thinking=thinking,
            output_format=schema,
            output_config={"effort": effort},
            **extra,
        )
        latency = time.perf_counter() - start

        usage = usage_details(response.usage)
        thinking_summary = "\n\n".join(
            b.thinking for b in response.content if b.type == "thinking" and b.thinking
        )
        output: dict[str, str] = {
            "role": "assistant",
            "content": (
                json.dumps(response.parsed_output.model_dump(), indent=2)
                if response.parsed_output is not None
                else "".join(b.text for b in response.content if b.type == "text")
            ),
        }
        if thinking_summary:
            output["thinking"] = thinking_summary
        update_generation(
            output=output,
            model=response.model,  # differs from self.model if a refusal fallback ran
            usage_details=usage,
            cost_details=cost_details(response.model, usage),
            metadata={
                "step": step,
                "stop_reason": response.stop_reason,
                "request_id": getattr(response, "_request_id", None),
            },
            level="WARNING" if response.stop_reason != "end_turn" else None,
            status_message=None
            if response.stop_reason == "end_turn"
            else f"stop_reason={response.stop_reason}",
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
