"""Shared HTTP helpers for the public data sources."""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

USER_AGENT = "biomed-evidence-agent/0.1 (+https://github.com/AngeloJKwak/biomed-evidence-agent)"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=0.5, max=8),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def get_with_retry(
    http: httpx.AsyncClient, url: str, params: dict | None = None
) -> httpx.Response:
    resp = await http.get(url, params=params)
    resp.raise_for_status()
    return resp


def make_http_client(timeout_s: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout_s, headers={"User-Agent": USER_AGENT})
