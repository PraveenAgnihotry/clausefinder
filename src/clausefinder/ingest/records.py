"""Common parse-stage output shape for all ingest parsers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedRecord:
    source_id: str
    section: str | None
    title: str | None
    text: str
    page: int | None = None
    source_path: str | None = None
