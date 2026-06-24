"""Build and persist chunk metadata plus a FAISS index."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clausefinder import config
from clausefinder.index import embed
from clausefinder.process import chunk, normalize

CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id INTEGER PRIMARY KEY,
  doc_id TEXT NOT NULL,
  source TEXT NOT NULL,
  jurisdiction TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  section TEXT NOT NULL DEFAULT '',
  url TEXT NOT NULL DEFAULT '',
  chunk_index INTEGER NOT NULL DEFAULT 0,
  n_chunks INTEGER NOT NULL DEFAULT 1,
  text TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  content_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


def init_chunks_schema(conn: sqlite3.Connection) -> None:
    """Initialize the chunks table and indexes if they do not exist."""
    conn.executescript(CHUNKS_DDL)
    conn.commit()


def write_chunks(
    conn: sqlite3.Connection,
    chunks: list[chunk.Chunk],
    *,
    replace: bool = True,
) -> None:
    """Persist chunks in deterministic order, with chunk_id from enumerate order."""
    if replace:
        conn.execute("DELETE FROM chunks")

    conn.executemany(
        """
                INSERT INTO chunks (
                    chunk_id,
                    doc_id,
                    source,
                    jurisdiction,
                    title,
                    section,
                    url,
                    chunk_index,
                    n_chunks,
                    text,
                    token_count,
                    content_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
        [
            (
                chunk_id,
                item.doc_id,
                item.source,
                item.jurisdiction,
                item.title,
                item.section,
                item.url,
                item.chunk_index,
                item.n_chunks,
                item.text,
                item.token_count,
                item.content_sha256,
            )
            for chunk_id, item in enumerate(chunks)
        ],
    )
    conn.commit()


def _iso_utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _portable_path(path: Path) -> str:
    """Return a repo-relative POSIX path, or just filename when outside repo root."""
    try:
        return path.relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _token_stats(chunks: list[chunk.Chunk]) -> tuple[int, int, int]:
    if not chunks:
        return 0, 0, 0
    counts = [item.token_count for item in chunks]
    return min(counts), int(statistics.median(counts)), max(counts)


def _split_stats(chunks: list[chunk.Chunk]) -> tuple[int, int]:
    if not chunks:
        return 0, 0
    doc_n_chunks: dict[str, int] = {}
    for item in chunks:
        doc_n_chunks[item.doc_id] = max(doc_n_chunks.get(item.doc_id, 0), item.n_chunks)
    split_docs = sum(1 for n in doc_n_chunks.values() if n > 1)
    return split_docs, max(doc_n_chunks.values())


def build(*, replace: bool = True) -> dict[str, Any]:
    """Build chunks and FAISS index, then persist artifacts to disk and SQLite."""
    conn = normalize.connect(config.SQLITE_DB_PATH)
    try:
        records = normalize.read_all(conn)
        count_tokens = embed.get_token_counter()
        chunks = chunk.chunk_documents(
            records,
            count_tokens=count_tokens,
            exclude_sources=config.EMBED_EXCLUDED_SOURCES,
        )

        # INVARIANT: chunk_id equals FAISS row position; all writes preserve this exact order.
        chunk_texts = [item.text for item in chunks]
        embeddings = embed.embed_texts(chunk_texts, is_query=False)
        assert embeddings.shape[0] == len(chunks)

        import faiss

        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        config.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))

        init_chunks_schema(conn)
        write_chunks(conn, chunks, replace=replace)

        documents_indexed = len({item.doc_id for item in chunks})
        meta = {
            "model": config.EMBEDDING_MODEL,
            "dim": dim,
            "metric": "ip_cosine",
            "normalized": config.EMBED_NORMALIZE,
            "query_instruction": config.QUERY_INSTRUCTION,
            "n_chunks": len(chunks),
            "n_documents_total": len(records),
            "n_documents_indexed": documents_indexed,
            "excluded_sources": sorted(config.EMBED_EXCLUDED_SOURCES),
            "faiss_index_path": _portable_path(config.FAISS_INDEX_PATH),
            "db_path": _portable_path(config.SQLITE_DB_PATH),
            "built_at": _iso_utc_now(),
        }
        config.INDEX_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        token_min, token_median, token_max = _token_stats(chunks)
        split_docs, max_n_chunks = _split_stats(chunks)
        return {
            "documents_read": len(records),
            "documents_indexed": documents_indexed,
            "total_chunks": len(chunks),
            "split_docs": split_docs,
            "max_n_chunks": max_n_chunks,
            "token_min": token_min,
            "token_median": token_median,
            "token_max": token_max,
            "embedding_dim": dim,
            "faiss_index_path": str(config.FAISS_INDEX_PATH),
            "db_path": str(config.SQLITE_DB_PATH),
            "index_meta_path": str(config.INDEX_META_PATH),
        }
    finally:
        conn.close()


def _select_preview_rows(rows: list[sqlite3.Row], max_items: int = 15) -> list[sqlite3.Row]:
    by_source: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)

    selected: list[sqlite3.Row] = []
    sources = sorted(by_source)
    offset = 0

    while len(selected) < max_items:
        added = False
        for source in sources:
            items = by_source[source]
            if offset < len(items):
                selected.append(items[offset])
                added = True
                if len(selected) >= max_items:
                    break
        if not added:
            break
        offset += 1

    if not any(int(row["n_chunks"]) > 1 for row in selected):
        multi = next((row for row in rows if int(row["n_chunks"]) > 1), None)
        if multi is not None:
            if len(selected) >= max_items:
                selected[-1] = multi
            else:
                selected.append(multi)

    return selected


def _snippet(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def _write_chunks_preview(conn: sqlite3.Connection, path: Path) -> None:
    rows = conn.execute(
        """
        SELECT chunk_id, chunk_index, n_chunks, source, section, url, token_count, text
        FROM chunks
        ORDER BY chunk_id
        """
    ).fetchall()
    selected = _select_preview_rows(rows, max_items=15)
    payload = [
        {
            "chunk_id": int(row["chunk_id"]),
            "chunk_index": int(row["chunk_index"]),
            "n_chunks": int(row["n_chunks"]),
            "source": str(row["source"]),
            "section": str(row["section"]),
            "url": str(row["url"]),
            "token_count": int(row["token_count"]),
            "text_snippet": _snippet(str(row["text"])),
        }
        for row in selected
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Build FAISS + chunks artifacts and print a verification summary."""
    summary = build(replace=True)

    conn = normalize.connect(config.SQLITE_DB_PATH)
    try:
        preview_path = config.PROCESSED_DIR / "chunks_preview.json"
        _write_chunks_preview(conn, preview_path)
    finally:
        conn.close()

    print(f"Documents read: {summary['documents_read']}")
    print(f"Documents indexed (after exclusion): {summary['documents_indexed']}")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Documents split across multiple chunks: {summary['split_docs']}")
    print(f"Max chunks for a single document: {summary['max_n_chunks']}")
    print(
        "Chunk token-count min/median/max: "
        f"{summary['token_min']}/{summary['token_median']}/{summary['token_max']}"
    )
    print(f"Embedding dim: {summary['embedding_dim']}")
    print(f"FAISS index path: {summary['faiss_index_path']}")
    print(f"SQLite DB path: {summary['db_path']}")
    print(f"Index metadata path: {summary['index_meta_path']}")
    print(f"Chunks preview path: {preview_path}")


if __name__ == "__main__":
    main()
