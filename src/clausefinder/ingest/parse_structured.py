"""Parse structured Approved Documents catalogue data into parsed records."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from clausefinder.ingest.records import ParsedRecord
from clausefinder.process import clean

logger = logging.getLogger(__name__)


def _as_text(value: Any) -> str | None:
    """Return a stripped string representation for scalar values."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def _normalize_section(part_value: Any) -> str | None:
    """Normalize a catalogue part code to a section string like 'Part B'."""
    part_text = _as_text(part_value)
    if not part_text:
        return None
    if part_text.lower().startswith("part "):
        return f"Part {part_text[5:].strip().upper()}"
    return f"Part {part_text.upper()}"


def _build_text(entry: dict[str, Any], section: str | None, title: str | None) -> str:
    """Synthesize a short retrievable text snippet from available fields."""
    part_code = section.replace("Part ", "") if section else None
    if part_code and title:
        lead = f"Approved Document {part_code} ({title})."
    elif title:
        lead = f"{title}."
    elif section:
        lead = f"Approved Document {part_code}."
    else:
        lead = "Approved Document entry."

    parts: list[str] = [lead]

    summary = (
        _as_text(entry.get("scope"))
        or _as_text(entry.get("summary"))
        or _as_text(entry.get("description"))
    )
    if summary:
        parts.append(f"{summary}.")

    updated_at = _as_text(entry.get("public_updated_at"))
    if updated_at:
        parts.append(f"Updated: {updated_at}.")

    withdrawn = entry.get("withdrawn")
    if withdrawn is True:
        parts.append("Status: withdrawn.")

    url = _as_text(entry.get("url")) or _as_text(entry.get("source_url"))
    if url:
        parts.append(f"Source: {url}.")

    return clean.clean_text(" ".join(parts))


def parse_structured(path: Path, source_id: str) -> list[ParsedRecord]:
    """Parse local catalogue JSON into one record per valid entry."""
    source_path = str(path)
    records: list[ParsedRecord] = []
    seen: set[tuple[str | None, str | None, str]] = set()

    with path.open("r", encoding="utf-8") as infile:
        raw_data = json.load(infile)

    if isinstance(raw_data, list):
        entries = raw_data
    else:
        logger.warning(
            "Structured parse summary file=%s entries_read=0 records=0 skipped=1 reason=unexpected_root",
            source_path,
        )
        return []

    entries_read = len(entries)
    skipped = 0

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            skipped += 1
            logger.warning(
                "Skipping malformed catalogue entry index=%d file=%s reason=not_object",
                index,
                source_path,
            )
            continue

        section = _normalize_section(entry.get("part"))
        title = _as_text(entry.get("title")) or _as_text(entry.get("name"))

        if not section and not title:
            skipped += 1
            logger.warning(
                "Skipping malformed catalogue entry index=%d file=%s reason=missing_title_and_part",
                index,
                source_path,
            )
            continue

        text = _build_text(entry, section=section, title=title)
        if not text:
            skipped += 1
            logger.warning(
                "Skipping malformed catalogue entry index=%d file=%s reason=empty_text",
                index,
                source_path,
            )
            continue

        dedupe_key = (section, title, text)
        if dedupe_key in seen:
            skipped += 1
            logger.info(
                "Skipping duplicate catalogue entry index=%d file=%s",
                index,
                source_path,
            )
            continue

        seen.add(dedupe_key)
        records.append(
            ParsedRecord(
                source_id=source_id,
                section=section,
                title=title,
                text=text,
                source_path=source_path,
            )
        )

    logger.info(
        "Structured parse summary file=%s entries_read=%d records=%d skipped=%d",
        source_path,
        entries_read,
        len(records),
        skipped,
    )
    return records
