"""Wire concrete dependencies from settings."""

from __future__ import annotations

import httpx

from biomed_agent.agent.graph import Dependencies, EvidenceAgent
from biomed_agent.config import Settings, get_settings
from biomed_agent.llm import ClaudeLLM
from biomed_agent.retrieval.embeddings import SentenceTransformerEmbedder
from biomed_agent.retrieval.store import VectorStore
from biomed_agent.sources.clinicaltrials import ClinicalTrialsClient
from biomed_agent.sources.http import make_http_client
from biomed_agent.sources.pubmed import PubMedClient


def build_agent(
    settings: Settings | None = None, http: httpx.AsyncClient | None = None
) -> tuple[EvidenceAgent, httpx.AsyncClient]:
    """Return the agent and the HTTP client it uses (caller owns closing it)."""
    settings = settings or get_settings()
    if settings.use_os_trust_store:
        import truststore

        truststore.inject_into_ssl()
    http = http or make_http_client(settings.http_timeout_s)
    store = VectorStore(
        SentenceTransformerEmbedder(settings.embedding_model),
        persist_dir=settings.chroma_dir,
        chunk_chars=settings.chunk_chars,
        chunk_overlap=settings.chunk_overlap,
    )
    deps = Dependencies(
        settings=settings,
        llm=ClaudeLLM.from_settings(settings),
        pubmed=PubMedClient(http, settings.ncbi_api_key, settings.ncbi_email),
        trials=ClinicalTrialsClient(http),
        store=store,
    )
    return EvidenceAgent(deps), http
