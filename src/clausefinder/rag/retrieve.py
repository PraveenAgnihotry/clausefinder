"""Embed query, top-k FAISS search, and return ranked chunk metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss

from clausefinder import config
from clausefinder.index import embed
from clausefinder.process import normalize


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Retrieved chunk payload including score and citation metadata."""

    chunk_id: int
    score: float
    doc_id: str
    source: str
    jurisdiction: str
    title: str
    section: str
    url: str
    text: str


@dataclass(slots=True)
class Retriever:
    """Bundle FAISS index and chunk rows to keep row mappings consistent."""

    index: faiss.Index
    chunk_store: list[dict[str, Any]]

    @classmethod
    def load(cls) -> Retriever:
        """Load persisted index and chunk metadata in strict chunk_id order."""
        index = faiss.read_index(str(config.FAISS_INDEX_PATH))

        conn = normalize.connect(config.SQLITE_DB_PATH)
        try:
            rows = conn.execute(
                """
				SELECT chunk_id, doc_id, source, jurisdiction, title, section, url, text
				FROM chunks
				ORDER BY chunk_id
				"""
            ).fetchall()
        finally:
            conn.close()

        chunk_store: list[dict[str, Any]] = []
        for position, row in enumerate(rows):
            chunk_id = int(row["chunk_id"])
            if chunk_id != position:
                raise ValueError(
                    "Invalid chunks table ordering: expected contiguous chunk_id values "
                    f"starting at 0, got chunk_id={chunk_id} at position={position}. "
                    "FAISS row mapping would be incorrect."
                )
            chunk_store.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": str(row["doc_id"]),
                    "source": str(row["source"]),
                    "jurisdiction": str(row["jurisdiction"]),
                    "title": str(row["title"]),
                    "section": str(row["section"]),
                    "url": str(row["url"]),
                    "text": str(row["text"]),
                }
            )

        return cls(index=index, chunk_store=chunk_store)

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Run top-k vector retrieval and return ranked chunks with metadata."""
        top_k = k or config.TOP_K
        query_vec = embed.embed_texts([query], is_query=True)
        scores, idxs = self.index.search(query_vec, top_k)

        results: list[RetrievedChunk] = []
        for idx, score in zip(idxs[0], scores[0], strict=False):
            if idx == -1:
                continue

            row = self.chunk_store[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=int(row["chunk_id"]),
                    score=float(score),
                    doc_id=str(row["doc_id"]),
                    source=str(row["source"]),
                    jurisdiction=str(row["jurisdiction"]),
                    title=str(row["title"]),
                    section=str(row["section"]),
                    url=str(row["url"]),
                    text=str(row["text"]),
                )
            )
        return results


_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    """Return a lazy singleton Retriever for reuse across Streamlit reruns."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever.load()
    return _RETRIEVER


def is_low_confidence(chunks: list[RetrievedChunk]) -> bool:
    """Flag potentially weak retrieval results using the top hit score."""
    # Advisory only: never drop results on score; the prompt handles refusal behavior.
    return not chunks or chunks[0].score < config.RETRIEVAL_MIN_SCORE
