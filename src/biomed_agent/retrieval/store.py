"""Chroma-backed vector store with an embedding cache keyed by chunk id.

Documents fetched for one question are reused by later questions: chunks that
are already in the collection are not re-embedded. Queries are scoped to the
sources retrieved for the current question so stale results from earlier
questions never leak into an answer.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from biomed_agent.retrieval.chunking import chunk_document
from biomed_agent.retrieval.embeddings import Embedder
from biomed_agent.schemas import Chunk, SourceDocument


class VectorStore:
    def __init__(
        self,
        embedder: Embedder,
        persist_dir: Path | None = None,
        collection: str = "biomed",
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        self.embedder = embedder
        if persist_dir is None:
            client = chromadb.EphemeralClient()
        else:
            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))
        # One collection per embedding model: vectors from different models aren't comparable.
        safe_model = "".join(c if c.isalnum() else "-" for c in embedder.name)[:40]
        self._col = client.get_or_create_collection(
            name=f"{collection}-{safe_model}", metadata={"hnsw:space": "cosine"}
        )
        self._chunk_chars = chunk_chars
        self._chunk_overlap = chunk_overlap

    def add_documents(self, docs: list[SourceDocument]) -> int:
        """Chunk + embed + upsert. Returns the number of newly embedded chunks."""
        chunks = [
            c for d in docs for c in chunk_document(d, self._chunk_chars, self._chunk_overlap)
        ]
        if not chunks:
            return 0
        existing = set(self._col.get(ids=[c.chunk_id for c in chunks], include=[])["ids"])
        new = [c for c in chunks if c.chunk_id not in existing]
        if new:
            self._col.upsert(
                ids=[c.chunk_id for c in new],
                documents=[c.text for c in new],
                embeddings=self.embedder.embed_documents([c.text for c in new]),
                metadatas=[{"source_id": c.source_id, "source_type": c.source_type} for c in new],
            )
        return len(new)

    def search(
        self, query: str, source_ids: list[str], k: int = 12, max_per_source: int = 2
    ) -> list[Chunk]:
        """Semantic search restricted to `source_ids`.

        At most `max_per_source` chunks are kept per source so a single long
        record can't crowd out the rest of the evidence.
        """
        if not source_ids:
            return []
        where = (
            {"source_id": source_ids[0]}
            if len(source_ids) == 1
            else {"source_id": {"$in": source_ids}}
        )
        res = self._col.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k * 3,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[Chunk] = []
        per_source: dict[str, int] = {}
        for cid, text, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            source_id = str(meta["source_id"])
            if per_source.get(source_id, 0) >= max_per_source:
                continue
            per_source[source_id] = per_source.get(source_id, 0) + 1
            out.append(
                Chunk(
                    chunk_id=cid,
                    source_id=source_id,
                    source_type=meta["source_type"],  # type: ignore[arg-type]
                    text=text,
                    score=round(1.0 - float(dist), 4),
                )
            )
            if len(out) == k:
                break
        return out
