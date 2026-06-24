"""Parse PDF sources into common parsed records."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz

from clausefinder.ingest.records import ParsedRecord
from clausefinder.process import clean

logger = logging.getLogger(__name__)

REQUIREMENT_CODE_RE = re.compile(
    r"^(?P<section>[A-Z]\d+(?:\(\d+\))?)\b(?:\s*[-:])?\s*(?P<title>.*)$"
)
NUMBERED_PARAGRAPH_RE = re.compile(r"^(?P<section>\d+(?:\.\d+)+)\b(?:\s*[-:])?\s*(?P<title>.*)$")
REGULATION_RE = re.compile(
    r"^(?P<section>reg\.?\s*\d+[A-Za-z]?)\b(?:\s*[-:])?\s*(?P<title>.*)$",
    re.IGNORECASE,
)

TOC_DOTTED_LEADER_RE = re.compile(r"\.{3,}")
HYPHENATED_BREAK_RE = re.compile(r"(?<=[^\W\d_])-\s*\n\s*(?=[a-z])")
TOC_PREFIX_RE = re.compile(r"^(Requirement|Section|Appendix|Part)\b", re.IGNORECASE)
TOC_TITLE_EXTRACT_RE = re.compile(
    r"^(?:Requirement\s+[A-Z0-9().]+\s*[:\-]\s*|Section\s+\d+[A-Za-z]?\s*[:\-]\s*|Appendix\s+[A-Z0-9]+\s*[:\-]\s*|Part\s+[A-Z0-9]+\s*[:\-]\s*)(?P<title>.+)$",
    re.IGNORECASE,
)
LINE_ENDS_WITH_PAGE_TOKEN_RE = re.compile(
    r"(?:\s+|\.{2,}\s*)(?:\d+|[ivxlcdm]{1,6}|pp?|p)\s*$",
    re.IGNORECASE,
)
ROMAN_ONLY_RE = re.compile(
    r"^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)$",
    re.IGNORECASE,
)
P_OR_PP_RE = re.compile(r"^pp?$", re.IGNORECASE)

REPEATED_LINE_MIN_FRACTION = 0.5
REPEATED_LINE_MAX_LEN = 80
TOC_DOTTED_LINE_FRACTION = 0.5
TOC_SHORT_LINE_MAX = 90
TOC_STRONG_FRACTION = 0.45
TOC_WEAK_FRACTION = 0.3
REAL_SECTION_MIN_PROSE_CHARS = 200
MIN_RECORD_TEXT_CHARS = 30
MAX_REQUIREMENT_TITLE_CHARS = 60
NUMBERED_SECTION_ID_RE = re.compile(r"^\d+(?:\.\d+)+$")
REQUIREMENT_SECTION_ID_RE = re.compile(r"^[A-Z]\d+(?:\(\d+\))?$")


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


def _normalize_title_line(line: str) -> str:
    line = clean.normalize_unicode(line)
    line = TOC_DOTTED_LEADER_RE.sub(" ", line)
    line = re.sub(r"\s+", " ", line).strip().lower()
    line = LINE_ENDS_WITH_PAGE_TOKEN_RE.sub("", line).strip()
    return line


def _collect_section_titles(pages: list[_PageLines]) -> set[str]:
    titles: set[str] = set()
    for page in pages:
        for line in page.lines:
            header = _match_section_header(line)
            if header is not None and header[1]:
                normalized = _normalize_title_line(header[1])
                if normalized:
                    titles.add(normalized)

            match = TOC_TITLE_EXTRACT_RE.match(line.strip())
            if not match:
                continue
            normalized = _normalize_title_line(match.group("title"))
            if normalized:
                titles.add(normalized)
    return titles


def _looks_like_toc_index_line(line: str, known_titles: set[str]) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.lower() in {"contents", "index"}:
        return True
    if TOC_PREFIX_RE.match(stripped):
        return True
    if TOC_DOTTED_LEADER_RE.search(stripped):
        return True
    if clean.is_boilerplate_line(stripped):
        return True
    if ROMAN_ONLY_RE.fullmatch(stripped):
        return True
    if P_OR_PP_RE.fullmatch(stripped):
        return True

    normalized = _normalize_title_line(stripped)
    if normalized and normalized in known_titles:
        return True
    return False


def _toc_page_scores(lines: list[str], known_titles: set[str]) -> tuple[float, float]:
    if not lines:
        return 0.0, 0.0
    short_lines = sum(1 for line in lines if len(line) <= TOC_SHORT_LINE_MAX)
    toc_like_lines = sum(1 for line in lines if _looks_like_toc_index_line(line, known_titles))
    line_count = len(lines)
    return short_lines / line_count, toc_like_lines / line_count


def _find_toc_like_page_indices(pages: list[_PageLines], known_titles: set[str]) -> set[int]:
    strong_flags: list[bool] = []
    weak_flags: list[bool] = []

    for page in pages:
        short_ratio, toc_ratio = _toc_page_scores(page.lines, known_titles)
        is_strong = (
            (short_ratio >= 0.65 and toc_ratio >= TOC_STRONG_FRACTION)
            or (toc_ratio >= 0.7)
            or _is_toc_page(page.lines)
        )
        is_weak = short_ratio >= 0.7 and toc_ratio >= TOC_WEAK_FRACTION
        strong_flags.append(is_strong)
        weak_flags.append(is_weak)

    drop_indices: set[int] = set()
    index = 0
    while index < len(pages):
        if not strong_flags[index]:
            index += 1
            continue

        end_index = index
        while end_index + 1 < len(pages) and (
            strong_flags[end_index + 1] or weak_flags[end_index + 1]
        ):
            end_index += 1

        drop_indices.update(range(index, end_index + 1))
        index = end_index + 1

    return drop_indices


def _line_is_valid_header(lines: list[str], line_index: int) -> bool:
    header = _match_section_header(lines[line_index])
    if header is None:
        return False
    if LINE_ENDS_WITH_PAGE_TOKEN_RE.search(lines[line_index].strip()):
        return False
    if line_index + 1 < len(lines) and _match_section_header(lines[line_index + 1]) is not None:
        return False
    return True


def _prose_chars_until_next_header(
    pages: list[_PageLines],
    start_page_index: int,
    start_line_index: int,
) -> int:
    char_count = 0
    for page_index in range(start_page_index, len(pages)):
        lines = pages[page_index].lines
        line_index = start_line_index if page_index == start_page_index else 0
        while line_index < len(lines):
            if _line_is_valid_header(lines, line_index):
                return char_count
            char_count += len(lines[line_index])
            line_index += 1
    return char_count


def _find_content_start(pages: list[_PageLines]) -> tuple[int, int] | None:
    for page_index, page in enumerate(pages):
        for line_index, _line in enumerate(page.lines):
            if not _line_is_valid_header(page.lines, line_index):
                continue
            prose_chars = _prose_chars_until_next_header(pages, page_index, line_index + 1)
            if prose_chars >= REAL_SECTION_MIN_PROSE_CHARS:
                return page_index, line_index
    return None


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


def _normalize_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_heading_title(title: str) -> bool:
    normalized = _normalize_inline(title)
    if not normalized:
        return False
    if len(normalized) > MAX_REQUIREMENT_TITLE_CHARS:
        return False
    if normalized.endswith(","):
        return False
    first = normalized[0]
    if first in ".(":
        return False
    if first.isalpha() and first.islower():
        return False
    return True


def _prepare_title_and_lines(
    section: str | None,
    title: str | None,
    lines: list[str],
) -> tuple[str | None, list[str]]:
    effective_title = _normalize_inline(title) if title else None
    effective_lines = list(lines)

    if not effective_title:
        return None, effective_lines

    if NUMBERED_SECTION_ID_RE.fullmatch(section or ""):
        return None, [effective_title, *effective_lines]

    if REQUIREMENT_SECTION_ID_RE.fullmatch(section or "") and not _looks_like_heading_title(
        effective_title
    ):
        return None, [effective_title, *effective_lines]

    return effective_title, effective_lines


def _strip_title_from_body_start(title: str, body: str) -> str:
    title_tokens = [re.escape(token) for token in title.split() if token]
    if not title_tokens:
        return body
    title_pattern = r"\s+".join(title_tokens)

    pattern = re.compile(
        rf"^\s*{title_pattern}(?:\s*[:\-\u2013\u2014]\s*|\s+)?",
        re.IGNORECASE,
    )
    stripped = pattern.sub("", body, count=1).strip()
    return stripped if stripped else body


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
    effective_title, effective_lines = _prepare_title_and_lines(section, title, lines)
    raw_text = "\n".join(effective_lines)
    hyphenation_fixes = len(HYPHENATED_BREAK_RE.findall(raw_text))
    cleaned_text = clean.clean_text(raw_text)

    if effective_title and cleaned_text:
        cleaned_text = _strip_title_from_body_start(effective_title, cleaned_text)

    if not cleaned_text:
        return hyphenation_fixes
    if len(cleaned_text) < MIN_RECORD_TEXT_CHARS:
        return hyphenation_fixes

    records.append(
        ParsedRecord(
            source_id=source_id,
            section=section,
            title=effective_title,
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
    dropped_toc_pages: list[int] = []
    front_matter_pages_before_start: list[int] = []

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

        filtered_pages.append(_PageLines(page=page.page, lines=filtered_lines))

    known_titles = _collect_section_titles(filtered_pages)
    toc_drop_indices = _find_toc_like_page_indices(filtered_pages, known_titles)
    if toc_drop_indices:
        dropped_toc_pages = [filtered_pages[index].page for index in sorted(toc_drop_indices)]
        logger.info(
            "PDF parse TOC/index pages dropped file=%s pages=%s", source_path, dropped_toc_pages
        )

    non_toc_pages = [
        page for index, page in enumerate(filtered_pages) if index not in toc_drop_indices
    ]

    content_start = _find_content_start(non_toc_pages)
    if content_start is None:
        logger.info(
            "PDF parse content start not found file=%s; using first non-TOC page", source_path
        )
        start_page_index = 0
        start_line_index = 0
    else:
        start_page_index, start_line_index = content_start
        logger.info(
            "PDF parse content starts file=%s page=%d",
            source_path,
            non_toc_pages[start_page_index].page,
        )

    if non_toc_pages and start_page_index > 0:
        front_matter_pages_before_start = [page.page for page in non_toc_pages[:start_page_index]]
        logger.info(
            "PDF parse front-matter pages skipped file=%s pages=%s",
            source_path,
            front_matter_pages_before_start,
        )

    current_section: str | None = None
    current_title: str | None = None
    current_page: int | None = None
    current_lines: list[str] = []

    for page_index, page in enumerate(non_toc_pages):
        if page_index < start_page_index:
            continue

        page_lines = page.lines
        if page_index == start_page_index:
            page_lines = page_lines[start_line_index:]

        for line_index, line in enumerate(page_lines):
            if _line_is_valid_header(page_lines, line_index):
                header = _match_section_header(line)
                if header is None:
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

                current_section, current_title = header
                current_page = page.page
                current_lines = []
                continue

            if current_section is not None:
                current_lines.append(line)

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

    short_records = sum(1 for record in records if len(record.text) < 30)
    toc_and_front_matter_dropped = len(dropped_toc_pages) + len(front_matter_pages_before_start)

    logger.info(
        "PDF parse summary file=%s pages=%d toc_front_matter_pages_dropped=%d records=%d short_records_lt30=%d headers_footers_stripped=%d hyphenation_fixes=%d skipped_blank_or_image=%d",
        source_path,
        total_pages,
        toc_and_front_matter_dropped,
        len(records),
        short_records,
        stripped_repeated_lines,
        hyphenation_fixes,
        skipped_blank_or_image_pages,
    )
    return records
