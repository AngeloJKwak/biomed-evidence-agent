from __future__ import annotations

from biomed_agent.retrieval.chunking import chunk_document
from biomed_agent.retrieval.embeddings import HashingEmbedder
from biomed_agent.retrieval.store import VectorStore
from biomed_agent.schemas import SourceDocument


def _doc(source_id: str, title: str, text: str) -> SourceDocument:
    return SourceDocument(
        source_id=source_id, source_type="pubmed", title=title, text=text, url="https://x"
    )


def test_short_document_is_single_chunk_with_title() -> None:
    chunks = chunk_document(_doc("PMID:1", "A title", "One sentence. Two sentences."))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "PMID:1#0"
    assert chunks[0].text.startswith("A title\n")


def test_long_document_splits_with_overlap() -> None:
    text = " ".join(f"Sentence number {i} about statins and LDL." for i in range(120))
    chunks = chunk_document(_doc("PMID:2", "T", text), max_chars=500, overlap=100)
    assert len(chunks) > 3
    assert all(len(c.text) <= 500 + 120 for c in chunks)  # budget + one sentence of slack
    # Consecutive chunks share overlapping text.
    tail = chunks[0].text[-60:]
    assert tail.split()[-1] in chunks[1].text


def test_store_caps_chunks_per_source(tmp_path) -> None:
    store = VectorStore(HashingEmbedder(), persist_dir=tmp_path, chunk_chars=300, chunk_overlap=0)
    long_text = " ".join(f"Statin trial result {i} on LDL cholesterol." for i in range(60))
    docs = [
        _doc("PMID:1", "Long statin record", long_text),
        _doc("PMID:2", "Short statin note", "Statins lower LDL cholesterol."),
    ]
    store.add_documents(docs)
    hits = store.search("statin LDL cholesterol", ["PMID:1", "PMID:2"], k=10, max_per_source=2)
    assert sum(h.source_id == "PMID:1" for h in hits) == 2
    assert "PMID:2" in {h.source_id for h in hits}


def test_store_scopes_search_to_requested_sources(tmp_path) -> None:
    store = VectorStore(HashingEmbedder(), persist_dir=tmp_path)
    docs = [
        _doc(
            "PMID:1", "Statins", "Statin therapy lowers LDL cholesterol and cardiovascular events."
        ),
        _doc("PMID:2", "Metformin", "Metformin improves glycemic control in type 2 diabetes."),
        _doc("PMID:3", "Statin myopathy", "Muscle symptoms are reported with statin therapy."),
    ]
    assert store.add_documents(docs) == 3
    # Re-adding is a no-op thanks to the chunk-id cache.
    assert store.add_documents(docs) == 0

    hits = store.search("statin LDL cholesterol", ["PMID:1", "PMID:2"], k=5)
    assert {h.source_id for h in hits} == {"PMID:1", "PMID:2"}
    assert hits[0].source_id == "PMID:1"
    assert hits[0].score is not None and hits[0].score > hits[-1].score

    assert store.search("anything", [], k=5) == []
    assert [h.source_id for h in store.search("statin", ["PMID:3"], k=5)] == ["PMID:3"]
