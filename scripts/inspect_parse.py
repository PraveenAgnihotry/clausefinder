"""CLI utility to inspect parser outputs across known raw sources."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from clausefinder import config
from clausefinder.ingest.parse_html import parse_html
from clausefinder.ingest.parse_pdf import parse_pdf
from clausefinder.ingest.parse_structured import parse_structured
from clausefinder.ingest.records import ParsedRecord

logger = logging.getLogger(__name__)

# In Step 3, this mapping will be derived from data/manifest.json instead.
KNOWN_SOURCES: list[tuple[str, str]] = [
    ("pdf/approved_document_b.pdf", "ad_part_b"),
    ("pdf/approved_document_k.pdf", "ad_part_k"),
    ("pdf/approved_document_m.pdf", "ad_part_m"),
    ("html/reg_03.html", "building_regs_2010_reg_3"),
    ("html/reg_04.html", "building_regs_2010_reg_4"),
    ("html/reg_05.html", "building_regs_2010_reg_5"),
    ("html/reg_06.html", "building_regs_2010_reg_6"),
    ("html/reg_07.html", "building_regs_2010_reg_7"),
    ("html/reg_08.html", "building_regs_2010_reg_8"),
    ("html/reg_09.html", "building_regs_2010_reg_9"),
    ("html/schedule_1.html", "building_regs_2010_schedule_1"),
    ("structured/approved_documents_catalogue.json", "approved_documents_catalogue"),
]


def _select_parser(relative_path: str) -> Callable[[Path, str], list[ParsedRecord]]:
    suffix = Path(relative_path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf
    if suffix == ".html":
        return parse_html
    if suffix == ".json":
        return parse_structured
    raise ValueError(f"No parser configured for file type: {relative_path}")


def _print_summary(per_source: dict[str, list[ParsedRecord]]) -> None:
    print("source_id                                total  with_section  text_len_lt_30")
    print("-" * 78)
    for source_id, records in per_source.items():
        count = len(records)
        with_section = sum(record.section is not None for record in records)
        orphan_fragments = sum(len(record.text) < 30 for record in records)
        section_window = [
            record.section if record.section is not None else "<None>" for record in records[:25]
        ]
        section_preview = " | ".join(section_window) if section_window else "<none>"

        print(f"{source_id:<40} {count:>5} {with_section:>13} {orphan_fragments:>15}")
        print(f"  sections[1..25]: {section_preview}")


def _write_outputs(per_source: dict[str, list[ParsedRecord]]) -> tuple[Path, Path]:
    preview = {
        source_id: [asdict(record) for record in records[:3]]
        for source_id, records in per_source.items()
    }
    full = {
        source_id: [asdict(record) for record in records]
        for source_id, records in per_source.items()
    }

    preview_path = config.PROCESSED_DIR / "parsed_preview.json"
    preview_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")

    full_path = config.PROCESSED_DIR / "parsed_full.json"
    full_path.write_text(json.dumps(full, indent=2), encoding="utf-8")

    return preview_path, full_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    per_source_records: dict[str, list[ParsedRecord]] = {}

    for relative_path, source_id in KNOWN_SOURCES:
        full_path = config.RAW_DIR / relative_path
        if not full_path.exists():
            logger.warning("Skipping missing source file: %s", full_path)
            per_source_records[source_id] = []
            continue

        parser = _select_parser(relative_path)
        logger.info("Parsing source_id=%s path=%s", source_id, full_path)
        try:
            per_source_records[source_id] = parser(full_path, source_id)
        except Exception:
            logger.exception("Parser failed for source_id=%s path=%s", source_id, full_path)
            per_source_records[source_id] = []

    _print_summary(per_source_records)
    preview_path, full_path = _write_outputs(per_source_records)
    print(f"\nWrote preview JSON: {preview_path}")
    print(f"Wrote full JSON: {full_path}")


if __name__ == "__main__":
    main()
