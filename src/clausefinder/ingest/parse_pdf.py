"""Parse PDF sources into common parsed records."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

import fitz

from clausefinder.ingest.records import ParsedRecord
from clausefinder.process import clean

logger = logging.getLogger(__name__)

REQUIREMENT_CODE_RE = re.compile(
	r"^(?P<section>[A-Z]\d+(?:\(\d+\))?)\b(?:\s*[-:])?\s*(?P<title>.*)$"
)
NUMBERED_PARAGRAPH_RE = re.compile(
	r"^(?P<section>\d+(?:\.\d+)+)\b(?:\s*[-:])?\s*(?P<title>.*)$"
)
REGULATION_RE = re.compile(
	r"^(?P<section>reg\.?\s*\d+[A-Za-z]?)\b(?:\s*[-:])?\s*(?P<title>.*)$",
	re.IGNORECASE,
)

TOC_DOTTED_LEADER_RE = re.compile(r"\.{3,}")
HYPHENATED_BREAK_RE = re.compile(r"(?<=[^\W\d_])-\s*\n\s*(?=[a-z])")

REPEATED_LINE_MIN_FRACTION = 0.5
REPEATED_LINE_MAX_LEN = 80
TOC_DOTTED_LINE_FRACTION = 0.5


@dataclass(slots=True)
class _PageLines:
	page: int
	lines: list[str]


def _extract_page_lines(page: fitz.Page) -> list[str]:
	blocks = page.get_text("blocks")
	text_blocks = [block for block in blocks if len(block) >= 7 and int(block[6]) == 0]
	text_blocks.sort(key=lambda block: (round(float(block[1])), float(block[0])))

	# Known limitation: tables and figures are out of scope and may appear as loose text.
	page_text = "\n".join(str(block[4]) for block in text_blocks if str(block[4]).strip())
	return [line.strip() for line in page_text.splitlines() if line.strip()]


def _is_toc_page(lines: list[str]) -> bool:
	if not lines:
		return False
	dotted_lines = sum(1 for line in lines if TOC_DOTTED_LEADER_RE.search(line))
	return (dotted_lines / len(lines)) >= TOC_DOTTED_LINE_FRACTION


def _match_section_header(line: str) -> tuple[str, str | None] | None:
	stripped = line.strip()
	for pattern in (REQUIREMENT_CODE_RE, NUMBERED_PARAGRAPH_RE, REGULATION_RE):
		match = pattern.match(stripped)
		if not match:
			continue
		section = match.group("section").strip()
		title = (match.group("title") or "").strip() or None
		return section, title
	return None


def _append_record(
	records: list[ParsedRecord],
	*,
	source_id: str,
	source_path: str,
	section: str | None,
	title: str | None,
	page: int,
	lines: list[str],
) -> int:
	raw_text = "\n".join(lines)
	hyphenation_fixes = len(HYPHENATED_BREAK_RE.findall(raw_text))
	cleaned_text = clean.clean_text(raw_text)
	if not cleaned_text:
		return hyphenation_fixes

	records.append(
		ParsedRecord(
			source_id=source_id,
			section=section,
			title=title,
			text=cleaned_text,
			page=page,
			source_path=source_path,
		)
	)
	return hyphenation_fixes


def parse_pdf(path: Path, source_id: str) -> list[ParsedRecord]:
	"""Parse a PDF file into section-aware records using text blocks."""
	records: list[ParsedRecord] = []
	source_path = str(path)

	stripped_repeated_lines = 0
	hyphenation_fixes = 0
	skipped_blank_or_image_pages = 0
	skipped_toc_pages = 0

	with fitz.open(path) as document:
		total_pages = document.page_count
		extracted_pages: list[_PageLines] = []

		for page_index in range(total_pages):
			page_number = page_index + 1
			page = document.load_page(page_index)
			lines = _extract_page_lines(page)
			if not lines:
				skipped_blank_or_image_pages += 1
				logger.debug("Skipping image-only/blank page %s in %s", page_number, source_path)
				continue
			extracted_pages.append(_PageLines(page=page_number, lines=lines))

	repeated_lines = clean.find_repeated_lines(
		[page.lines for page in extracted_pages],
		min_fraction=REPEATED_LINE_MIN_FRACTION,
		max_len=REPEATED_LINE_MAX_LEN,
	)

	filtered_pages: list[_PageLines] = []
	for page in extracted_pages:
		filtered_lines: list[str] = []
		for line in page.lines:
			if line in repeated_lines:
				stripped_repeated_lines += 1
				continue
			if clean.is_boilerplate_line(line):
				continue
			filtered_lines.append(line)

		if not filtered_lines:
			skipped_blank_or_image_pages += 1
			logger.debug("Skipping blank page %s after cleanup in %s", page.page, source_path)
			continue

		if _is_toc_page(filtered_lines):
			skipped_toc_pages += 1
			logger.debug("Skipping TOC-like page %s in %s", page.page, source_path)
			continue

		filtered_pages.append(_PageLines(page=page.page, lines=filtered_lines))

	current_section: str | None = None
	current_title: str | None = None
	current_page: int | None = None
	current_lines: list[str] = []

	for page in filtered_pages:
		page_orphan_lines: list[str] = []

		for line in page.lines:
			header = _match_section_header(line)
			if header is None:
				if current_section is None:
					page_orphan_lines.append(line)
				else:
					current_lines.append(line)
				continue

			if current_section is not None:
				hyphenation_fixes += _append_record(
					records,
					source_id=source_id,
					source_path=source_path,
					section=current_section,
					title=current_title,
					page=current_page or page.page,
					lines=current_lines,
				)

			if page_orphan_lines:
				hyphenation_fixes += _append_record(
					records,
					source_id=source_id,
					source_path=source_path,
					section=None,
					title=None,
					page=page.page,
					lines=page_orphan_lines,
				)
				page_orphan_lines = []

			current_section, current_title = header
			current_page = page.page
			current_lines = []

		if current_section is None and page_orphan_lines:
			hyphenation_fixes += _append_record(
				records,
				source_id=source_id,
				source_path=source_path,
				section=None,
				title=None,
				page=page.page,
				lines=page_orphan_lines,
			)

	if current_section is not None:
		hyphenation_fixes += _append_record(
			records,
			source_id=source_id,
			source_path=source_path,
			section=current_section,
			title=current_title,
			page=current_page or 1,
			lines=current_lines,
		)

	logger.info(
		"PDF parse summary file=%s pages=%d records=%d headers_footers_stripped=%d hyphenation_fixes=%d skipped_blank_or_image=%d skipped_toc=%d skipped_total=%d",
		source_path,
		total_pages,
		len(records),
		stripped_repeated_lines,
		hyphenation_fixes,
		skipped_blank_or_image_pages,
		skipped_toc_pages,
		skipped_blank_or_image_pages + skipped_toc_pages,
	)
	return records
