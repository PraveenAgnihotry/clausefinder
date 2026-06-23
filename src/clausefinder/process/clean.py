"""Pure text-cleaning helpers shared by ingest parsers."""

import math
import re
import unicodedata

_UNICODE_SPACES_RE = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")
_HYPHENATION_RE = re.compile(r"(?<=[^\W\d_])-\s*\n\s*(?=[a-z])")
_INTERNAL_WS_RE = re.compile(r"[ \t]+")
_ONLY_DIGITS_RE = re.compile(r"^\d+$")
_PAGE_X_RE = re.compile(r"^page\s+\d+$", re.IGNORECASE)
_PAGE_X_OF_Y_RE = re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE)

_LIGATURES = {
	"\ufb00": "ff",
	"\ufb01": "fi",
	"\ufb02": "fl",
	"\ufb03": "ffi",
	"\ufb04": "ffl",
}


def normalize_unicode(text: str) -> str:
	"""Normalize Unicode text while preserving semantic punctuation."""
	normalized = unicodedata.normalize("NFKC", text)
	for ligature, replacement in _LIGATURES.items():
		normalized = normalized.replace(ligature, replacement)
	normalized = _UNICODE_SPACES_RE.sub(" ", normalized)
	normalized = _ZERO_WIDTH_RE.sub("", normalized)
	return normalized


def fix_hyphenation(text: str) -> str:
	"""Join words split by line-end hyphenation when continuation is lowercase.

	Example:
		>>> fix_hyphenation("requ-\nirement")
		'requirement'
		>>> fix_hyphenation("Part-\nA")
		'Part-\nA'
	"""
	return _HYPHENATION_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
	"""Normalize line endings, spacing, and excessive blank-line runs."""
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	lines = text.split("\n")

	normalized_lines: list[str] = []
	for line in lines:
		# Collapse repeated spaces/tabs while preserving line structure.
		collapsed = _INTERNAL_WS_RE.sub(" ", line).rstrip()
		normalized_lines.append(collapsed)

	collapsed_blank_runs: list[str] = []
	blank_run = 0
	for line in normalized_lines:
		if line == "":
			blank_run += 1
			continue

		if blank_run >= 3:
			collapsed_blank_runs.append("")
		elif blank_run > 0:
			collapsed_blank_runs.extend([""] * blank_run)
		blank_run = 0
		collapsed_blank_runs.append(line)

	if blank_run >= 3:
		collapsed_blank_runs.append("")
	elif blank_run > 0:
		collapsed_blank_runs.extend([""] * blank_run)

	return "\n".join(collapsed_blank_runs)


def clean_text(text: str) -> str:
	"""Apply the canonical text-cleaning pipeline and trim edges."""
	cleaned = normalize_unicode(text)
	cleaned = fix_hyphenation(cleaned)
	cleaned = normalize_whitespace(cleaned)
	return cleaned.strip()


def find_repeated_lines(
	pages: list[list[str]], min_fraction: float = 0.5, max_len: int = 80
) -> set[str]:
	"""Return lines that repeat across enough pages to be header/footer candidates.

	Example:
		>>> pages = [["Building Regulations", "1"], ["Building Regulations", "2"], ["Building Regulations", "1"]]
		>>> find_repeated_lines(pages, min_fraction=2/3, max_len=80) == {"Building Regulations", "1"}
		True
	"""
	if not pages:
		return set()

	required_pages = math.ceil(len(pages) * min_fraction)
	page_hits: dict[str, int] = {}

	for page in pages:
		for line in set(page):
			page_hits[line] = page_hits.get(line, 0) + 1

	return {
		line
		for line, hits in page_hits.items()
		if len(line) <= max_len and hits >= required_pages
	}


def is_boilerplate_line(line: str) -> bool:
	"""Return whether a line is obvious pagination boilerplate or empty."""
	stripped = line.strip()
	if not stripped:
		return True

	# Keep this intentionally small: plain numbers and common page labels.
	return bool(
		_ONLY_DIGITS_RE.fullmatch(stripped)
		or _PAGE_X_RE.fullmatch(stripped)
		or _PAGE_X_OF_Y_RE.fullmatch(stripped)
	)
