"""Split normalized documents into citation-friendly chunks with metadata."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from clausefinder import config
from clausefinder.process.normalize import NormalizedRecord, connect, read_all


@dataclass(frozen=True, slots=True)
class Chunk:
    """Chunk payload used by indexing and retrieval."""

    doc_id: str
    source: str
    jurisdiction: str
    title: str
    section: str
    url: str
    chunk_index: int
    n_chunks: int
    text: str
    token_count: int
    content_sha256: str


def approx_token_count(text: str) -> int:
    """Estimate tokens from characters using a fixed chars-per-token heuristic."""
    return max(1, math.ceil(len(text) / config.APPROX_CHARS_PER_TOKEN))


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines and newlines, stripping and dropping empty parts."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return [line for line in lines if line]


def _split_large_paragraph(
    paragraph: str,
    *,
    count_tokens: Callable[[str], int],
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    words = paragraph.split()
    if not words:
        cleaned = paragraph.strip()
        return [cleaned] if cleaned else []

    parts: list[str] = []
    start = 0

    while start < len(words):
        end = start
        best_end = start + 1

        while end < len(words):
            candidate = " ".join(words[start : end + 1])
            if count_tokens(candidate) <= target_tokens or end == start:
                best_end = end + 1
                end += 1
                continue
            break

        part_words = words[start:best_end]
        part_text = " ".join(part_words).strip()
        if part_text:
            parts.append(part_text)

        if best_end >= len(words):
            break

        part_tokens = max(1, count_tokens(part_text))
        avg_tokens_per_word = max(1.0, part_tokens / max(1, len(part_words)))
        overlap_words = max(1, int(round(overlap_tokens / avg_tokens_per_word)))
        next_start = max(start + 1, best_end - overlap_words)
        start = next_start

    return parts


def _overlap_suffix(
    paragraphs: list[str],
    *,
    count_tokens: Callable[[str], int],
    overlap_tokens: int,
) -> list[str]:
    if not paragraphs:
        return []

    suffix: list[str] = []
    for paragraph in reversed(paragraphs):
        if not suffix:
            suffix.insert(0, paragraph)
            if count_tokens(paragraph) >= overlap_tokens:
                break
            continue

        candidate = [paragraph, *suffix]
        candidate_tokens = count_tokens("\n\n".join(candidate))
        if candidate_tokens > overlap_tokens:
            break
        suffix = candidate

    return suffix


def split_text(
    text: str,
    *,
    count_tokens: Callable[[str], int],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into greedy paragraph-packed chunks with deterministic overlap."""
    cleaned = text.strip()
    if not cleaned:
        return [" "]

    effective_target = max(1, min(target_tokens, max_tokens))
    paragraphs = _split_paragraphs(cleaned)
    if not paragraphs:
        return [cleaned]

    units: list[str] = []
    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)
        if paragraph_tokens <= effective_target:
            units.append(paragraph)
            continue

        units.extend(
            _split_large_paragraph(
                paragraph,
                count_tokens=count_tokens,
                target_tokens=effective_target,
                overlap_tokens=overlap_tokens,
            )
        )

    units = [unit for unit in units if unit.strip()]
    if not units:
        return [cleaned]

    chunks: list[str] = []
    start_idx = 0

    while start_idx < len(units):
        chunk_units: list[str] = []
        cursor = start_idx

        while cursor < len(units):
            candidate_units = [*chunk_units, units[cursor]]
            candidate_text = "\n\n".join(candidate_units)
            candidate_tokens = count_tokens(candidate_text)
            if candidate_tokens <= effective_target or not chunk_units:
                chunk_units.append(units[cursor])
                cursor += 1
                continue
            break

        chunk_text = "\n\n".join(chunk_units).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if cursor >= len(units):
            break

        overlap_units = _overlap_suffix(
            chunk_units,
            count_tokens=count_tokens,
            overlap_tokens=overlap_tokens,
        )
        next_start = cursor - len(overlap_units)
        if next_start <= start_idx:
            next_start = start_idx + 1
        start_idx = next_start

    return chunks or [cleaned]


def _content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_record(
    rec: NormalizedRecord,
    *,
    count_tokens: Callable[[str], int] = approx_token_count,
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    max_tokens: int = config.CHUNK_MAX_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk one normalized record, preserving all source metadata."""
    total_tokens = count_tokens(rec.text)
    if total_tokens <= max_tokens:
        return [
            Chunk(
                doc_id=rec.doc_id,
                source=rec.source,
                jurisdiction=rec.jurisdiction,
                title=rec.title,
                section=rec.section,
                url=rec.url,
                chunk_index=0,
                n_chunks=1,
                text=rec.text,
                token_count=total_tokens,
                content_sha256=_content_sha256(rec.text),
            )
        ]

    parts = split_text(
        rec.text,
        count_tokens=count_tokens,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    n_chunks = len(parts)

    return [
        Chunk(
            doc_id=rec.doc_id,
            source=rec.source,
            jurisdiction=rec.jurisdiction,
            title=rec.title,
            section=rec.section,
            url=rec.url,
            chunk_index=index,
            n_chunks=n_chunks,
            text=part,
            token_count=count_tokens(part),
            content_sha256=_content_sha256(part),
        )
        for index, part in enumerate(parts)
    ]


def chunk_documents(
    records: Iterable[NormalizedRecord],
    *,
    count_tokens: Callable[[str], int] = approx_token_count,
    exclude_sources: set[str] = config.EMBED_EXCLUDED_SOURCES,
    **kw: int,
) -> list[Chunk]:
    """Chunk all eligible records in order, skipping excluded sources and empty text."""
    chunks: list[Chunk] = []
    for rec in records:
        if rec.source in exclude_sources:
            continue
        if not rec.text.strip():
            continue
        chunks.extend(chunk_record(rec, count_tokens=count_tokens, **kw))
    return chunks


def _select_preview_chunks(chunks: list[Chunk], max_items: int = 15) -> list[Chunk]:
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source].append(chunk)

    selected: list[Chunk] = []
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

    has_multichunk = any(item.n_chunks > 1 for item in selected)
    if not has_multichunk:
        multi = next((item for item in chunks if item.n_chunks > 1), None)
        if multi is not None:
            if len(selected) >= max_items:
                selected[-1] = multi
            else:
                selected.append(multi)

    return selected


def _write_preview(path: Path, chunks: list[Chunk]) -> None:
    preview = _select_preview_chunks(chunks, max_items=15)
    payload = [asdict(chunk) for chunk in preview]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(
    records: list[NormalizedRecord],
    chunks: list[Chunk],
    *,
    exclude_sources: set[str],
) -> None:
    docs_after_exclusion = sum(1 for rec in records if rec.source not in exclude_sources)

    by_doc: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        by_doc[chunk.doc_id] += 1

    split_docs = sum(1 for n in by_doc.values() if n > 1)
    max_n_chunks = max(by_doc.values()) if by_doc else 0

    token_counts = [chunk.token_count for chunk in chunks]
    token_min = min(token_counts) if token_counts else 0
    token_median = int(statistics.median(token_counts)) if token_counts else 0
    token_max = max(token_counts) if token_counts else 0

    print("Token counter: approximate token counts; build_index uses the real tokenizer")
    print(f"Documents read: {len(records)}")
    print(f"Documents after source-exclusion: {docs_after_exclusion}")
    print(f"Chunks produced: {len(chunks)}")
    print(f"Documents split across multiple chunks: {split_docs}")
    print(f"Max chunks for a single document: {max_n_chunks}")
    print(f"Chunk token-count min/median/max: {token_min}/{token_median}/{token_max}")


def main() -> None:
    """Run chunking over normalized documents and write a preview JSON."""
    conn = connect(config.SQLITE_DB_PATH)
    try:
        records = read_all(conn)
    finally:
        conn.close()

    chunks = chunk_documents(records)
    _print_summary(records, chunks, exclude_sources=config.EMBED_EXCLUDED_SOURCES)

    preview_path = config.PROCESSED_DIR / "chunks_preview.json"
    _write_preview(preview_path, chunks)
    print(f"Wrote chunks preview JSON: {preview_path}")


if __name__ == "__main__":
    main()
