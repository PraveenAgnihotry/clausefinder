"""CLI utility to inspect parser outputs across known raw sources."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Callable

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
    print("source_id                                records  with_section  avg_len  max_len")
    print("-" * 78)
    for source_id, records in per_source.items():
        lengths = [len(record.text) for record in records]
        count = len(records)
        with_section = sum(record.section is not None for record in records)
        avg_len = int(sum(lengths) / count) if count else 0
        max_len = max(lengths) if lengths else 0
        print(
            f"{source_id:<40} {count:>7} {with_section:>13} {avg_len:>8} {max_len:>8}"
        )


def _write_preview(per_source: dict[str, list[ParsedRecord]]) -> Path:
    preview = {source_id: [asdict(record) for record in records[:3]] for source_id, records in per_source.items()}
    output_path = config.PROCESSED_DIR / "parsed_preview.json"
    output_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    return output_path


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
    preview_path = _write_preview(per_source_records)
    print(f"\nWrote preview JSON: {preview_path}")


if __name__ == "__main__":
    main()
