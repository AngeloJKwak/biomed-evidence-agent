"""Runtime configuration, loaded from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BIOMED_", extra="ignore")

    # --- LLM -----------------------------------------------------------------
    # Generation + planning model. Judge model is used only by the eval harness.
    llm_model: str = "claude-opus-5-5"
    judge_model: str = "claude-sonnet-5-5"
    plan_effort: Effort = "low"
    answer_effort: Effort = "high"
    verify_effort: Effort = "medium"
    llm_max_tokens: int = 16000
    # Server-side refusal fallback (routes a declined request to another model).
    llm_fallbacks: bool = True

    # --- Retrieval -----------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: Path = Path(".data/chroma")
    pubmed_max_results: int = 12
    trials_max_results: int = 8
    chunk_chars: int = 1200
    chunk_overlap: int = 200
    top_k: int = 12

    # --- Agent loop ----------------------------------------------------------
    max_retrieval_rounds: int = 2
    max_revisions: int = 1

    # --- External APIs -------------------------------------------------------
    # Optional NCBI key raises the E-utilities rate limit from 3 to 10 req/s.
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None
    http_timeout_s: float = 30.0
    # Verify TLS against the OS certificate store instead of certifi's bundle. Needed behind
    # corporate proxies or antivirus HTTPS scanning that re-signs certificates.
    use_os_trust_store: bool = True

    # --- Provenance ----------------------------------------------------------
    runs_dir: Path = Path(".data/runs")
    prompt_version: str = "2026-10-01"


@lru_cache
def get_settings() -> Settings:
    return Settings()
