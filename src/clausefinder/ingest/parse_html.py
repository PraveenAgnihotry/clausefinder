"""Parse legislation HTML files into parsed records."""

from __future__ import annotations

import logging
from pathlib import Path
import re

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


def _parse_regulation(path: Path, source_id: str, container: Tag) -> list[ParsedRecord]:
	records: list[ParsedRecord] = []
	source_path = str(path)

	title_hint = _extract_title(container)
	fallback_section = _extract_reg_section_from_filename(path)

	current_section: str | None = None
	current_title: str | None = title_hint
	current_lines: list[str] = []

	flattened_links = 0
	skipped_empty_lines = 0

	for paragraph in container.find_all("p"):
		flattened_links += len(paragraph.find_all("a"))

		line = _visible_text(paragraph)
		if not line or SKIP_LINE_RE.match(line):
			skipped_empty_lines += 1
			continue

		new_section = _extract_reg_section_from_tag(paragraph)
		if new_section and new_section != current_section:
			if current_lines:
				cleaned_text = clean.clean_text("\n".join(current_lines))
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
		cleaned_text = clean.clean_text("\n".join(current_lines))
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
		fallback_text = clean.clean_text(clean.normalize_unicode(container.get_text("\n", strip=True)))
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
		"HTML parse summary file=%s type=regulation records=%d flattened_links=%d skipped_empty=%d",
		source_path,
		len(records),
		flattened_links,
		skipped_empty_lines,
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
		logger.info("HTML parse summary file=%s type=skipped records=0 reason=contents", source_path)
		return []

	html = path.read_text(encoding="utf-8")
	soup = _get_soup(html)
	container = _select_main_container(soup)
	_strip_noise(container)

	if filename.startswith("schedule"):
		return _parse_schedule(path=path, source_id=source_id, container=container)
	return _parse_regulation(path=path, source_id=source_id, container=container)
