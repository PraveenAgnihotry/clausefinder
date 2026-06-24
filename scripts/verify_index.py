"""Retrieval smoke test for the persisted ClauseFinder FAISS index.

Run from the repo root:
    uv run python scripts/verify_index.py
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import faiss

from clausefinder import config
from clausefinder.index import embed
from clausefinder.process import normalize

TOP_K = 5
SNIPPET_CHARS = 140
SAMPLE_QUERIES = [
    "What are the fire resistance requirements for doors?",
    "minimum width of stairs and height of handrails",
    "accessible sanitary accommodation for wheelchair users",
    "provision of cavity barriers in concealed spaces",
]


@dataclass(frozen=True, slots=True)
class ChunkRow:
    """Minimal chunk payload used to display retrieval results."""

    chunk_id: int
    source: str
    section: str
    url: str
    text: str


def _load_chunk_rows(conn: sqlite3.Connection) -> list[ChunkRow]:
    rows = conn.execute(
        """
        SELECT chunk_id, source, section, url, text
        FROM chunks
        ORDER BY chunk_id
        """
    ).fetchall()

    chunk_rows = [
        ChunkRow(
            chunk_id=int(row["chunk_id"]),
            source=str(row["source"]),
            section=str(row["section"]),
            url=str(row["url"]),
            text=str(row["text"]),
        )
        for row in rows
    ]

    mismatches = [(pos, row.chunk_id) for pos, row in enumerate(chunk_rows) if row.chunk_id != pos]
    if mismatches:
        preview = ", ".join(
            f"pos={position} chunk_id={chunk_id}" for position, chunk_id in mismatches[:5]
        )
        print(f"WARNING: non-contiguous chunk_id mapping detected: {preview}")

    assert not mismatches, "Expected chunk_id to match FAISS row position (0-based)."
    return chunk_rows


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _print_hits(
    query: str, scores_row: list[float], idx_row: list[int], chunk_rows: list[ChunkRow]
) -> None:
    print(f"\nQuery: {query}")
    for rank, (score, idx) in enumerate(zip(scores_row, idx_row, strict=False), start=1):
        if idx < 0 or idx >= len(chunk_rows):
            print(f"  {rank}. score={score:.4f}  idx={idx} (out of range)")
            continue

        row = chunk_rows[idx]
        print(
            f"  {rank}. score={score:.4f}  source={row.source}  section={row.section or '<empty>'}"
        )
        print(f"     url={row.url}")
        print(f"     text={_snippet(row.text)}")


def main() -> int:
    if not config.FAISS_INDEX_PATH.exists():
        print(f"Index file not found: {config.FAISS_INDEX_PATH}")
        return 1
    if not config.SQLITE_DB_PATH.exists():
        print(f"SQLite DB not found: {config.SQLITE_DB_PATH}")
        return 1

    index = faiss.read_index(str(config.FAISS_INDEX_PATH))

    conn = normalize.connect(config.SQLITE_DB_PATH)
    try:
        chunk_rows = _load_chunk_rows(conn)
    finally:
        conn.close()

    if index.ntotal != len(chunk_rows):
        print(
            "WARNING: FAISS/vector row count mismatch: "
            f"index.ntotal={index.ntotal}, chunks={len(chunk_rows)}"
        )
        raise AssertionError("FAISS rows must match chunks row count.")

    model = embed.load_embedder()

    print("=== ClauseFinder index retrieval smoke test ===")
    print(f"Index: {config.FAISS_INDEX_PATH}")
    print(f"DB: {config.SQLITE_DB_PATH}")
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Chunk rows: {len(chunk_rows)}")

    for query in SAMPLE_QUERIES:
        query_vec = embed.embed_texts([query], model=model, is_query=True)
        scores, idxs = index.search(query_vec, TOP_K)
        _print_hits(query, list(scores[0]), list(idxs[0]), chunk_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
