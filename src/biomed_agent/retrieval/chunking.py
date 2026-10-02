"""Split source documents into overlapping chunks on sentence-ish boundaries."""

from __future__ import annotations

import re

from biomed_agent.schemas import Chunk, SourceDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def chunk_document(doc: SourceDocument, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Chunk a document, prefixing each chunk with the title so it embeds with context.

    Most abstracts fit in a single chunk; long trial records and structured
    abstracts get split with `overlap` characters carried between chunks.
    """
    header = f"{doc.title}\n"
    budget = max(200, max_chars - len(header))
    pieces = [p.strip() for p in _SENTENCE_END.split(doc.text) if p.strip()]

    bodies: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > budget:
            bodies.append(current)
            current = current[-overlap:] + " " + piece if overlap else piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        bodies.append(current)

    return [
        Chunk(
            chunk_id=f"{doc.source_id}#{i}",
            source_id=doc.source_id,
            source_type=doc.source_type,
            text=header + body,
        )
        for i, body in enumerate(bodies)
    ]
