"""Parse legislation HTML files into parsed records."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag
from bs4.exceptions import FeatureNotFound

from clausefinder.ingest.records import ParsedRecord
from clausefinder.process import clean

logger = logging.getLogger(__name__)

MAIN_CONTAINER_SELECTORS: tuple[str, ...] = (
    "#viewLegContents .LegSnippet",
    "#viewLegContents",
    "#viewLegSnippet",
    "div.LegSnippet",
    "#content",
)

NOISE_SELECTORS: tuple[str, ...] = (
    "script",
    "style",
    "noscript",
    "nav",
    "form",
    "#statusWarning",
    "#statusWarningSubSections",
    "#tools",
    "#viewPrintControl",
    ".help",
    ".eniw",
    ".LegCommentary",
    ".LegCommentaryBlock",
    ".LegAnnotations",
    ".LegFootnotes",
)

SKIP_LINE_RE = re.compile(r"^this section has no associated explanatory memorandum$", re.IGNORECASE)
REG_FILE_RE = re.compile(r"^reg_(\d+)", re.IGNORECASE)
REG_NO_RE = re.compile(r"^(\d+[A-Z]?)\.?$")
PART_RE = re.compile(r"\bPART\s+([A-Z0-9]+)\b", re.IGNORECASE)
PROSPECTIVE_HINT_RE = re.compile(
    r"prospective|not yet in force|future|alternative version|point in time",
    re.IGNORECASE,
)
PARAGRAPH_MARKER_RE = re.compile(r"^\s*(?:\[[^\]]+\]\s*)?(?:F\d+\s*)?\((\d+[A-Za-z]?)\)\b")


def _get_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")


def _select_main_container(soup: BeautifulSoup) -> Tag:
    for selector in MAIN_CONTAINER_SELECTORS:
        container = soup.select_one(selector)
        if container is not None:
            return container
    if soup.body is None:
        raise ValueError("Unable to locate a parseable HTML body")
    return soup.body


def _strip_noise(container: Tag) -> None:
    for selector in NOISE_SELECTORS:
        for node in container.select(selector):
            node.decompose()


def _visible_text(node: Tag) -> str:
    text = node.get_text(" ", strip=True)
    return clean.normalize_unicode(text)


def _extract_reg_section_from_tag(paragraph: Tag) -> str | None:
    marker = paragraph.select_one(".LegP1No")
    if marker is None:
        return None
    marker_text = clean.normalize_unicode(marker.get_text("", strip=True)).strip()
    match = REG_NO_RE.match(marker_text)
    if not match:
        return None
    return f"reg. {match.group(1)}"


def _extract_reg_section_from_filename(path: Path) -> str | None:
    match = REG_FILE_RE.match(path.stem)
    if not match:
        return None
    return f"reg. {int(match.group(1))}"


def _extract_title(container: Tag) -> str | None:
    heading = container.select_one(
        ".LegP1GroupTitleFirst, .LegP1GroupTitle, .LegTitleBlockTitle, h1, h2, h3"
    )
    if heading is None:
        return None
    title = _visible_text(heading)
    title = re.sub(r"\b[A-Z]\+[A-Z](?:\+[A-Z])?\b", "", title).strip()
    return title or None


def _extract_reg_number(path: Path) -> str | None:
    match = REG_FILE_RE.match(path.stem)
    if not match:
        return None
    return str(int(match.group(1)))


def _drop_prospective_container(path: Path, container: Tag) -> bool:
    """Drop a clearly marked prospective/alternative wrapper if present."""
    reg_number = _extract_reg_number(path)
    if reg_number is None:
        return False

    reg_markers: list[Tag] = []
    for marker in container.select("span.LegP1No"):
        marker_text = clean.normalize_unicode(marker.get_text("", strip=True)).strip().rstrip(".")
        if marker_text == reg_number:
            reg_markers.append(marker)

    if len(reg_markers) < 2:
        return False

    second_para = reg_markers[1].find_parent("p")
    if second_para is None:
        return False

    candidate = second_para
    while candidate is not None and candidate is not container:
        if candidate.name in {"div", "section"}:
            class_blob = " ".join(candidate.get("class", []))
            id_blob = candidate.get("id") or ""
            text_blob = candidate.get_text(" ", strip=True)[:300]
            if PROSPECTIVE_HINT_RE.search(f"{class_blob} {id_blob} {text_blob}"):
                candidate.decompose()
                logger.info(
                    "HTML dedupe file=%s prospective_container_removed=true",
                    path,
                )
                return True
        candidate = candidate.parent if isinstance(candidate.parent, Tag) else None

    return False


def _trim_duplicate_regulation_block(
    text: str, reg_number: str, source_path: str
) -> tuple[str, int, bool]:
    """Keep only the first regulation block if the regulation boundary repeats."""
    boundary_pattern = re.compile(
        rf"(?m)^\s*{re.escape(reg_number)}\s*\.?\s*(?:[\-\u2013\u2014]|\u00e2\u20ac\u201d)?"
    )
    matches = list(boundary_pattern.finditer(text))
    if len(matches) <= 1:
        return text, 0, False

    trimmed = text[: matches[1].start()].rstrip()
    removed_chars = len(text) - len(trimmed)
    logger.info(
        "HTML dedupe file=%s duplicate_regulation_boundary=true removed_chars=%d",
        source_path,
        removed_chars,
    )
    return trimmed, removed_chars, True


def _trim_duplicate_paragraphs(text: str, source_path: str) -> tuple[str, int, bool]:
    """Drop repeated numeric paragraph markers, keeping the first occurrence."""

    def _paragraph_marker_key(line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None

        # Direct paragraph marker: (1), (4), (4A), etc.
        direct = re.match(r"^\((\d+[A-Za-z]?)\)\b", stripped)
        if direct:
            return direct.group(1)

        # Amendment-led variants, e.g. "F5 [ F6 (4) ..." or "[F7 (4) ...".
        if stripped.startswith("[") or re.match(r"^F\d+\b", stripped):
            near_start = stripped[:80]
            marker = re.search(r"\((\d+[A-Za-z]?)\)", near_start)
            if marker:
                return marker.group(1)

        legacy = PARAGRAPH_MARKER_RE.match(stripped)
        if legacy:
            return legacy.group(1)
        return None

    seen: dict[str, str] = {}
    kept_lines: list[str] = []

    for line in text.splitlines():
        marker = _paragraph_marker_key(line)
        if marker is None:
            kept_lines.append(line)
            continue

        if marker in seen:
            first_line = seen[marker]
            if re.search(r"\bF\d+\b", first_line) or re.search(r"\bF\d+\b", line):
                continue

        seen[marker] = line
        kept_lines.append(line)

    trimmed = "\n".join(kept_lines)
    removed_chars = len(text) - len(trimmed)
    if removed_chars > 0:
        logger.info(
            "HTML dedupe file=%s duplicate_paragraph_markers=true removed_chars=%d",
            source_path,
            removed_chars,
        )
        return trimmed, removed_chars, True

    return text, 0, False


def _finalize_reg_text(path: Path, raw_text: str) -> tuple[str, bool, bool]:
    """Apply duplicate guards and clean final regulation text."""
    source_path = str(path)
    reg_number = _extract_reg_number(path)
    trimmed = raw_text
    dup_reg_dropped = False
    dup_para_dropped = False

    if reg_number is not None:
        trimmed, _removed, dup_reg_dropped = _trim_duplicate_regulation_block(
            trimmed, reg_number=reg_number, source_path=source_path
        )

    trimmed, _removed, dup_para_dropped = _trim_duplicate_paragraphs(trimmed, source_path)
    return clean.clean_text(trimmed), dup_reg_dropped, dup_para_dropped


def _parse_regulation(
    path: Path,
    source_id: str,
    container: Tag,
    *,
    prospective_container_removed: bool,
) -> list[ParsedRecord]:
    records: list[ParsedRecord] = []
    source_path = str(path)

    title_hint = _extract_title(container)
    fallback_section = _extract_reg_section_from_filename(path)

    current_section: str | None = None
    current_title: str | None = title_hint
    current_lines: list[str] = []

    flattened_links = 0
    skipped_empty_lines = 0
    dup_reg_dropped_any = False
    dup_para_dropped_any = False

    for paragraph in container.find_all("p"):
        flattened_links += len(paragraph.find_all("a"))

        line = _visible_text(paragraph)
        if not line or SKIP_LINE_RE.match(line):
            skipped_empty_lines += 1
            continue

        new_section = _extract_reg_section_from_tag(paragraph)
        if new_section and new_section != current_section:
            if current_lines:
                cleaned_text, dup_reg_dropped, dup_para_dropped = _finalize_reg_text(
                    path=path,
                    raw_text="\n".join(current_lines),
                )
                dup_reg_dropped_any = dup_reg_dropped_any or dup_reg_dropped
                dup_para_dropped_any = dup_para_dropped_any or dup_para_dropped
                if cleaned_text:
                    records.append(
                        ParsedRecord(
                            source_id=source_id,
                            section=current_section,
                            title=current_title,
                            text=cleaned_text,
                            source_path=source_path,
                        )
                    )
            current_section = new_section
            current_title = title_hint
            current_lines = []

        if current_section is None:
            current_section = fallback_section

        # Keep paragraph markers like (1), (a), (i) by preserving paragraph-level text.
        current_lines.append(line)

    if current_lines:
        cleaned_text, dup_reg_dropped, dup_para_dropped = _finalize_reg_text(
            path=path,
            raw_text="\n".join(current_lines),
        )
        dup_reg_dropped_any = dup_reg_dropped_any or dup_reg_dropped
        dup_para_dropped_any = dup_para_dropped_any or dup_para_dropped
        if cleaned_text:
            records.append(
                ParsedRecord(
                    source_id=source_id,
                    section=current_section,
                    title=current_title,
                    text=cleaned_text,
                    source_path=source_path,
                )
            )

    if not records:
        fallback_text, dup_reg_dropped, dup_para_dropped = _finalize_reg_text(
            path=path,
            raw_text=clean.normalize_unicode(container.get_text("\n", strip=True)),
        )
        dup_reg_dropped_any = dup_reg_dropped_any or dup_reg_dropped
        dup_para_dropped_any = dup_para_dropped_any or dup_para_dropped
        if fallback_text:
            records.append(
                ParsedRecord(
                    source_id=source_id,
                    section=fallback_section,
                    title=title_hint,
                    text=fallback_text,
                    source_path=source_path,
                )
            )

    logger.info(
        "HTML parse summary file=%s type=regulation records=%d flattened_links=%d skipped_empty=%d prospective_removed=%s duplicate_regulation_dropped=%s duplicate_paragraphs_dropped=%s",
        source_path,
        len(records),
        flattened_links,
        skipped_empty_lines,
        prospective_container_removed,
        dup_reg_dropped_any,
        dup_para_dropped_any,
    )
    return records


def _extract_part_section(part_title: str) -> str:
    match = PART_RE.search(part_title)
    if match:
        return f"Schedule 1, Part {match.group(1).upper()}"
    return "Schedule 1"


def _parse_schedule(path: Path, source_id: str, container: Tag) -> list[ParsedRecord]:
    records: list[ParsedRecord] = []
    source_path = str(path)

    flattened_links = 0
    skipped_empty_rows = 0

    tabular_blocks = container.select(".LegTabular")
    if not tabular_blocks:
        tabular_blocks = [container]

    for block in tabular_blocks:
        part_heading = block.select_one(".LegTableTitle")
        part_title = _visible_text(part_heading) if part_heading is not None else "Schedule 1"
        section = _extract_part_section(part_title)

        for table in block.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    skipped_empty_rows += 1
                    continue

                if row.find("th") is not None:
                    continue

                requirement_cell = cells[0]
                flattened_links += len(requirement_cell.find_all("a"))
                requirement_text = clean.clean_text(_visible_text(requirement_cell))
                if not requirement_text:
                    skipped_empty_rows += 1
                    continue

                records.append(
                    ParsedRecord(
                        source_id=source_id,
                        section=section,
                        title=part_title,
                        text=requirement_text,
                        source_path=source_path,
                    )
                )

    logger.info(
        "HTML parse summary file=%s type=schedule records=%d flattened_links=%d skipped_empty_rows=%d",
        source_path,
        len(records),
        flattened_links,
        skipped_empty_rows,
    )
    return records


def parse_html(path: Path, source_id: str) -> list[ParsedRecord]:
    """Parse a legislation HTML file into ParsedRecord entries."""
    source_path = str(path)
    filename = path.name.lower()

    if filename == "contents.html":
        logger.info(
            "HTML parse summary file=%s type=skipped records=0 reason=contents", source_path
        )
        return []

    html = path.read_text(encoding="utf-8")
    soup = _get_soup(html)
    container = _select_main_container(soup)
    _strip_noise(container)
    prospective_removed = _drop_prospective_container(path=path, container=container)

    if filename.startswith("schedule"):
        return _parse_schedule(path=path, source_id=source_id, container=container)
    return _parse_regulation(
        path=path,
        source_id=source_id,
        container=container,
        prospective_container_removed=prospective_removed,
    )
