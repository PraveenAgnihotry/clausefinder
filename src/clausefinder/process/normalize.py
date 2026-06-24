"""Normalize parsed records into a unified schema and persist to SQLite."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from clausefinder.config import MANIFEST_PATH, PROCESSED_DIR, PROJECT_ROOT, SQLITE_DB_PATH
from clausefinder.ingest.parse_html import parse_html
from clausefinder.ingest.parse_pdf import parse_pdf
from clausefinder.ingest.parse_structured import parse_structured
from clausefinder.ingest.records import ParsedRecord

logger = logging.getLogger(__name__)

DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS documents (
	doc_id TEXT PRIMARY KEY,
	source TEXT NOT NULL,
	jurisdiction TEXT NOT NULL DEFAULT '',
	title TEXT NOT NULL DEFAULT '',
	section TEXT NOT NULL DEFAULT '',
	text TEXT NOT NULL,
	url TEXT NOT NULL DEFAULT '',
	content_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
"""


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    """Unified normalized document record used for chunking and retrieval."""

    doc_id: str
    source: str
    jurisdiction: str
    title: str
    section: str
    text: str
    url: str
    content_sha256: str


@dataclass(slots=True)
class _NormalizationResult:
    records: list[NormalizedRecord]
    pair_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    disambiguated_by_pair: Counter[tuple[str, str]] = field(default_factory=Counter)
    dropped_by_pair: Counter[tuple[str, str]] = field(default_factory=Counter)


def _slug(s: str) -> str:
    """Return a lowercase slug with non-alphanumeric runs collapsed to dashes."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _select_parser(path: Path) -> Callable[[Path, str], list[ParsedRecord]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf
    if suffix == ".html":
        return parse_html
    if suffix == ".json":
        return parse_structured
    raise ValueError(f"No parser configured for file type: {path}")


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Load manifest records from disk."""
    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a JSON array: {path}")
    manifest: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entries must be JSON objects: {path}")
        manifest.append(item)
    return manifest


def build_manifest_by_basename(manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a basename -> manifest entry map and validate uniqueness."""
    manifest_by_basename: dict[str, dict[str, Any]] = {}
    for entry in manifest:
        local_path = _as_text(entry.get("local_path"))
        basename = Path(local_path).name
        if not basename:
            raise ValueError(f"Manifest entry has empty local_path: {entry}")
        if basename in manifest_by_basename:
            raise ValueError(f"Manifest has duplicate basename: {basename}")
        manifest_by_basename[basename] = entry
    return manifest_by_basename


def parse_all_sources(manifest: list[dict[str, Any]]) -> list[ParsedRecord]:
    """Parse all manifest-listed sources in manifest order."""
    parsed_records: list[ParsedRecord] = []

    for entry in manifest:
        local_path_raw = _as_text(entry.get("local_path"))
        source_id = _as_text(entry.get("id"))
        source_path = Path(local_path_raw)
        full_path = source_path if source_path.is_absolute() else PROJECT_ROOT / source_path

        if not full_path.exists():
            raise FileNotFoundError(f"Manifest source file not found: {full_path}")

        parser = _select_parser(full_path)
        logger.info("Parsing source_id=%s path=%s", source_id, full_path)
        parsed_records.extend(parser(full_path, source_id))

    return parsed_records


def _make_unique_doc_id(base: str, content_sha256: str, used_doc_ids: set[str]) -> str:
    preferred_widths = (8, 12, 16, 24, 32, 40, 64)
    for width in preferred_widths:
        candidate = f"{base}-{content_sha256[:width]}"
        if candidate not in used_doc_ids:
            return candidate

    suffix = 2
    while True:
        candidate = f"{base}-{content_sha256}-{suffix}"
        if candidate not in used_doc_ids:
            return candidate
        suffix += 1


def _normalize_records_with_stats(
    records: list[ParsedRecord],
    manifest_by_basename: dict[str, dict[str, Any]],
) -> _NormalizationResult:
    normalized: list[NormalizedRecord] = []
    seen_by_base: dict[str, dict[str, str]] = defaultdict(dict)
    used_doc_ids: set[str] = set()

    pair_counts: Counter[tuple[str, str]] = Counter()
    disambiguated_by_pair: Counter[tuple[str, str]] = Counter()
    dropped_by_pair: Counter[tuple[str, str]] = Counter()

    for record in records:
        source_path = _as_text(record.source_path)
        key = Path(source_path).name
        if not key:
            raise ValueError(f"Parsed record has empty source_path: {record}")

        entry = manifest_by_basename.get(key)
        if entry is None:
            raise ValueError(
                f"Parsed record source_path is missing from manifest: {record.source_path}"
            )

        source = _as_text(entry.get("id"))
        url = _as_text(entry.get("url"))
        jurisdiction = _as_text(entry.get("jurisdiction"))

        raw_title = _as_text(record.title).strip()
        title = raw_title if raw_title else _as_text(entry.get("source_name"))
        section = _as_text(record.section)
        text = _as_text(record.text)
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

        source_slug = _slug(source)
        if section:
            base = f"{source_slug}::{_slug(section)}"
        else:
            base = f"{source_slug}::{content_sha256[:10]}"

        pair_key = (source, section)
        pair_counts[pair_key] += 1

        if content_sha256 in seen_by_base[base]:
            dropped_by_pair[pair_key] += 1
            continue

        if not seen_by_base[base]:
            doc_id = base
        else:
            doc_id = _make_unique_doc_id(base, content_sha256, used_doc_ids)
            disambiguated_by_pair[pair_key] += 1

        seen_by_base[base][content_sha256] = doc_id
        used_doc_ids.add(doc_id)

        normalized.append(
            NormalizedRecord(
                doc_id=doc_id,
                source=source,
                jurisdiction=jurisdiction,
                title=title,
                section=section,
                text=text,
                url=url,
                content_sha256=content_sha256,
            )
        )

    return _NormalizationResult(
        records=normalized,
        pair_counts=pair_counts,
        disambiguated_by_pair=disambiguated_by_pair,
        dropped_by_pair=dropped_by_pair,
    )


def normalize_records(
    records: list[ParsedRecord],
    manifest_by_basename: dict[str, dict[str, Any]],
) -> list[NormalizedRecord]:
    """Normalize parsed records into the unified string-only schema."""
    return _normalize_records_with_stats(records, manifest_by_basename).records


def connect(db_path: Path) -> sqlite3.Connection:
    """Create a SQLite connection and ensure the parent directory exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize idempotent SQLite schema."""
    conn.executescript(DOCUMENTS_DDL)
    conn.commit()


def write_documents(
    conn: sqlite3.Connection,
    records: list[NormalizedRecord],
    *,
    replace: bool = True,
) -> None:
    """Write normalized records into SQLite, replacing existing rows by default."""
    if replace:
        conn.execute("DELETE FROM documents")

    conn.executemany(
        """
		INSERT INTO documents (
			doc_id, source, jurisdiction, title, section, text, url, content_sha256
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		""",
        [
            (
                record.doc_id,
                record.source,
                record.jurisdiction,
                record.title,
                record.section,
                record.text,
                record.url,
                record.content_sha256,
            )
            for record in records
        ],
    )
    conn.commit()


def read_all(conn: sqlite3.Connection) -> list[NormalizedRecord]:
    """Return all normalized records from SQLite."""
    rows = conn.execute(
        """
		SELECT doc_id, source, jurisdiction, title, section, text, url, content_sha256
		FROM documents
		ORDER BY source, section, doc_id
		"""
    ).fetchall()
    return [
        NormalizedRecord(
            doc_id=_as_text(row["doc_id"]),
            source=_as_text(row["source"]),
            jurisdiction=_as_text(row["jurisdiction"]),
            title=_as_text(row["title"]),
            section=_as_text(row["section"]),
            text=_as_text(row["text"]),
            url=_as_text(row["url"]),
            content_sha256=_as_text(row["content_sha256"]),
        )
        for row in rows
    ]


def get_document(conn: sqlite3.Connection, doc_id: str) -> NormalizedRecord | None:
    """Fetch one normalized record by doc_id."""
    row = conn.execute(
        """
		SELECT doc_id, source, jurisdiction, title, section, text, url, content_sha256
		FROM documents
		WHERE doc_id = ?
		""",
        (doc_id,),
    ).fetchone()

    if row is None:
        return None

    return NormalizedRecord(
        doc_id=_as_text(row["doc_id"]),
        source=_as_text(row["source"]),
        jurisdiction=_as_text(row["jurisdiction"]),
        title=_as_text(row["title"]),
        section=_as_text(row["section"]),
        text=_as_text(row["text"]),
        url=_as_text(row["url"]),
        content_sha256=_as_text(row["content_sha256"]),
    )


def _select_preview_records(
    records: list[NormalizedRecord], max_items: int = 20
) -> list[NormalizedRecord]:
    by_source: dict[str, list[NormalizedRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)

    selected: list[NormalizedRecord] = []
    sources = list(by_source)
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

    return selected


def _write_preview(path: Path, records: list[NormalizedRecord]) -> None:
    preview_records = _select_preview_records(records, max_items=20)
    payload = [asdict(record) for record in preview_records]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(records: list[NormalizedRecord], stats: _NormalizationResult) -> None:
    text_lengths = [len(record.text) for record in records]
    min_len = min(text_lengths) if text_lengths else 0
    median_len = int(statistics.median(text_lengths)) if text_lengths else 0
    max_len = max(text_lengths) if text_lengths else 0

    per_source = Counter(record.source for record in records)
    per_jurisdiction = Counter(record.jurisdiction for record in records)
    empty_section_count = sum(not record.section.strip() for record in records)
    short_text_count = sum(
        (not record.text.strip()) or len(record.text.strip()) < 30 for record in records
    )

    print(f"Total rows: {len(records)}")

    print("Rows per source:")
    for source, count in sorted(per_source.items()):
        print(f"  {source}: {count}")

    print("Rows per jurisdiction:")
    for jurisdiction, count in sorted(per_jurisdiction.items()):
        print(f"  {jurisdiction or '<empty>'}: {count}")

    print(f"Rows with empty section: {empty_section_count}")
    print(f"Rows with empty or <30 char text: {short_text_count}")
    print(f"Text length min/median/max: {min_len}/{median_len}/{max_len}")

    repeated_pairs = [(pair, count) for pair, count in stats.pair_counts.items() if count > 1]
    repeated_pairs.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))

    print("Repeated (source, section) pairs (>1 parsed records):")
    if not repeated_pairs:
        print("  <none>")
        return

    for (source, section), count in repeated_pairs:
        disambiguated = stats.disambiguated_by_pair.get((source, section), 0)
        dropped = stats.dropped_by_pair.get((source, section), 0)
        section_display = section if section else "<empty>"
        print(
            "  "
            f"source={source} | section={section_display} | "
            f"count={count} | disambiguated={disambiguated} | dropped_exact_duplicates={dropped}"
        )


def main() -> None:
    """Run Step 3: parse all sources, normalize records, and persist to SQLite."""
    logging.basicConfig(level=logging.INFO)

    manifest = load_manifest(MANIFEST_PATH)
    manifest_by_basename = build_manifest_by_basename(manifest)

    parsed_records = parse_all_sources(manifest)
    result = _normalize_records_with_stats(parsed_records, manifest_by_basename)

    conn = connect(SQLITE_DB_PATH)
    try:
        init_schema(conn)
        write_documents(conn, result.records, replace=True)
        persisted_records = read_all(conn)
    finally:
        conn.close()

    _print_summary(persisted_records, result)

    preview_path = PROCESSED_DIR / "normalized_preview.json"
    _write_preview(preview_path, result.records)
    print(f"Wrote normalized preview JSON: {preview_path}")


if __name__ == "__main__":
    main()
