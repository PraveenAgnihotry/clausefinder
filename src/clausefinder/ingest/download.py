"""Download public regulation sources reproducibly; record provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from clausefinder.config import MANIFEST_PATH, PROJECT_ROOT, RAW_DIR

LOGGER = logging.getLogger(__name__)

USER_AGENT = "clausefinder-dev/0.1 (data ingestion)"
REQUEST_TIMEOUT_SECONDS = 30
LEGISLATION_SLEEP_SECONDS = 0.5
STREAM_CHUNK_SIZE = 64 * 1024

LICENSE_NAME = "OGL v3.0"

GOV_UK_BASE_URL = "https://www.gov.uk"
GOV_UK_CONTENT_BASE_URL = f"{GOV_UK_BASE_URL}/api/content"
GOV_UK_COLLECTION_URL = f"{GOV_UK_CONTENT_BASE_URL}/government/collections/approved-documents"

LEGISLATION_BASE_URL = "https://www.legislation.gov.uk/uksi/2010/2214"

TARGET_PARTS: tuple[str, ...] = ("B", "M", "K")

FALLBACK_PDF_URLS: dict[str, str] = {
    "B": (
        "https://assets.publishing.service.gov.uk/media/67d2bb074702aacd2251cb94/"
        "Approved_Document_B_volume_1_Dwellings_2019_edition_incorporating_"
        "2020_2022_and_2025_amendments_collated_with_2026_and_2029_amendments.pdf"
    )
}

ALL_FORMATS = {"pdf", "html", "structured"}


@dataclass(frozen=True)
class ManifestRecord:
    """Manifest metadata for one saved source file."""

    id: str
    source_name: str
    publisher: str
    format: str
    url: str
    fetched_at: str
    license: str
    local_path: str
    bytes: int
    sha256: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serializable dict."""
        return {
            "id": self.id,
            "source_name": self.source_name,
            "publisher": self.publisher,
            "format": self.format,
            "url": self.url,
            "fetched_at": self.fetched_at,
            "license": self.license,
            "local_path": self.local_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "notes": self.notes,
        }


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mtime_iso(path: Path) -> str:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return mtime.isoformat(timespec="seconds").replace("+00:00", "Z")


def _relative_to_repo(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _hash_file(path: Path) -> tuple[int, str]:
    total_bytes = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(STREAM_CHUNK_SIZE), b""):
            digest.update(chunk)
            total_bytes += len(chunk)
    return total_bytes, digest.hexdigest()


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _ensure_output_dirs() -> dict[str, Path]:
    dirs = {
        "pdf": RAW_DIR / "pdf",
        "html": RAW_DIR / "html",
        "structured": RAW_DIR / "structured",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _get_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def _extract_part(text: str) -> str | None:
    match = re.search(r"approved\s+document\s+([a-z])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"approved-document-([a-z])(?:-|/|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _is_approved_document(item: dict[str, Any]) -> bool:
    title = str(item.get("title", ""))
    base_path = str(item.get("base_path", ""))
    return title.lower().startswith("approved document") or "approved-document" in base_path.lower()


def _document_matches_part(item: dict[str, Any], part: str) -> bool:
    title = str(item.get("title", ""))
    base_path = str(item.get("base_path", ""))
    text = f"{title} {base_path}".lower()
    part_lower = part.lower()
    if f"approved document {part_lower}" in text:
        return True
    if f"approved-document-{part_lower}" in text:
        return True
    inferred = _extract_part(text)
    return inferred == part.upper()


def _as_utc_datetime(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        return datetime.min.replace(tzinfo=UTC)
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_approved_documents_from_collection(payload: dict[str, Any]) -> list[dict[str, Any]]:
    links = payload.get("links")
    documents = links.get("documents") if isinstance(links, dict) else None
    if not isinstance(documents, list):
        top_keys = sorted(payload.keys())
        sample_item_keys: list[str] = []
        if isinstance(links, dict):
            for value in links.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    sample_item_keys = sorted(value[0].keys())
                    break
        LOGGER.error("GOV.UK collection top-level keys: %s", top_keys)
        LOGGER.error("Sample linked document keys: %s", sample_item_keys)
        raise RuntimeError("Missing links.documents in GOV.UK collection payload.")

    approved_docs = [
        doc for doc in documents if isinstance(doc, dict) and _is_approved_document(doc)
    ]
    LOGGER.info("Found %d approved-document entries in GOV.UK collection", len(approved_docs))
    return approved_docs


def _choose_document_for_part(
    part: str, approved_docs: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    candidates = [doc for doc in approved_docs if _document_matches_part(doc, part)]
    if not candidates:
        return None

    sorted_candidates = sorted(
        candidates,
        key=lambda doc: (
            _as_utc_datetime(doc.get("public_updated_at")),
            str(doc.get("title", "")).lower(),
        ),
        reverse=True,
    )
    if len(sorted_candidates) > 1:
        LOGGER.info(
            "Part %s matched %d GOV.UK documents; selecting most recently updated entry",
            part,
            len(sorted_candidates),
        )
    return sorted_candidates[0]


def _find_pdf_attachment_url(attachments: Any) -> str | None:
    if not isinstance(attachments, list):
        return None
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        content_type = str(attachment.get("content_type", "")).lower()
        raw_url = str(attachment.get("url", "")).strip()
        if not raw_url:
            continue
        if content_type == "application/pdf" or raw_url.lower().endswith(".pdf"):
            return urljoin(GOV_UK_BASE_URL, raw_url)
    return None


def _save_streamed_binary(
    session: requests.Session,
    url: str,
    destination: Path,
    force: bool,
) -> tuple[bool, str]:
    if destination.exists() and not force:
        LOGGER.info("Skipping existing file (use --force to re-download): %s", destination)
        return False, _mtime_iso(destination)

    LOGGER.info("Downloading %s -> %s", url, destination)
    with session.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
    return True, _utc_now_iso()


def _save_text_file(
    session: requests.Session,
    url: str,
    destination: Path,
    force: bool,
) -> tuple[bool, str]:
    if destination.exists() and not force:
        LOGGER.info("Skipping existing file (use --force to re-download): %s", destination)
        return False, _mtime_iso(destination)

    LOGGER.info("Downloading %s -> %s", url, destination)
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")
    return True, _utc_now_iso()


def _save_json_file(payload: Any, destination: Path, force: bool) -> tuple[bool, str]:
    if destination.exists() and not force:
        LOGGER.info("Skipping existing file (use --force to re-download): %s", destination)
        return False, _mtime_iso(destination)

    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True, _utc_now_iso()


def _make_manifest_record(
    *,
    record_id: str,
    source_name: str,
    publisher: str,
    file_format: str,
    url: str,
    fetched_at: str,
    local_path: Path,
    notes: str,
) -> ManifestRecord:
    total_bytes, digest = _hash_file(local_path)
    return ManifestRecord(
        id=record_id,
        source_name=source_name,
        publisher=publisher,
        format=file_format,
        url=url,
        fetched_at=fetched_at,
        license=LICENSE_NAME,
        local_path=_relative_to_repo(local_path),
        bytes=total_bytes,
        sha256=digest,
        notes=notes,
    )


def _normalize_catalogue(approved_docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for document in approved_docs:
        title = str(document.get("title", "")).strip()
        base_path = str(document.get("base_path", "")).strip()
        text_for_part = f"{title} {base_path}"
        normalized.append(
            {
                "part": _extract_part(text_for_part),
                "title": title,
                "base_path": base_path,
                "url": urljoin(GOV_UK_BASE_URL, base_path),
                "public_updated_at": document.get("public_updated_at"),
                "withdrawn": bool(document.get("withdrawn", False)),
            }
        )

    normalized.sort(key=lambda item: (str(item.get("part") or ""), str(item.get("title") or "")))
    return normalized


def _download_pdfs(
    session: requests.Session,
    approved_docs: Sequence[dict[str, Any]],
    pdf_dir: Path,
    force: bool,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for part in TARGET_PARTS:
        selected = _choose_document_for_part(part, approved_docs)
        if selected is None:
            LOGGER.warning("No approved-document entry found for Part %s", part)
            continue

        title = str(selected.get("title", "")).strip() or f"Approved Document {part}"
        base_path = str(selected.get("base_path", "")).strip()
        public_updated_at = str(selected.get("public_updated_at", "")).strip()

        attachment_url: str | None = None
        notes = f"Source catalogue path: {base_path}"

        if base_path:
            item_payload = _get_json(session, f"{GOV_UK_CONTENT_BASE_URL}{base_path}")
            details = item_payload.get("details") if isinstance(item_payload, dict) else None
            attachments = details.get("attachments") if isinstance(details, dict) else None
            attachment_url = _find_pdf_attachment_url(attachments)

        if attachment_url is None:
            fallback_url = FALLBACK_PDF_URLS.get(part)
            if fallback_url:
                attachment_url = fallback_url
                notes = "No PDF attachment found in content item; used verified fallback URL"
                LOGGER.warning("Using fallback PDF URL for Part %s", part)
            else:
                LOGGER.warning("No PDF attachment found for Part %s; skipping", part)
                continue

        target_file = pdf_dir / f"approved_document_{part.lower()}.pdf"
        downloaded, fetched_at = _save_streamed_binary(
            session=session,
            url=attachment_url,
            destination=target_file,
            force=force,
        )

        if not downloaded:
            notes = f"Existing file reused; {notes}"
        if public_updated_at:
            notes = f"{notes}; public_updated_at={public_updated_at}"

        records.append(
            _make_manifest_record(
                record_id=f"approved_document_{part.lower()}_pdf",
                source_name=title,
                publisher="UK Government (GOV.UK)",
                file_format="pdf",
                url=attachment_url,
                fetched_at=fetched_at,
                local_path=target_file,
                notes=notes,
            )
        )
    return records


def _download_structured_catalogue(
    approved_docs: Sequence[dict[str, Any]],
    structured_dir: Path,
    force: bool,
) -> list[ManifestRecord]:
    normalized = _normalize_catalogue(approved_docs)
    target_file = structured_dir / "approved_documents_catalogue.json"
    downloaded, fetched_at = _save_json_file(normalized, target_file, force=force)

    notes = f"Normalized from GOV.UK approved-documents collection ({len(normalized)} records)"
    if not downloaded:
        notes = f"Existing file reused; {notes}"

    record = _make_manifest_record(
        record_id="approved_documents_catalogue_json",
        source_name="Approved Documents Catalogue",
        publisher="UK Government (GOV.UK)",
        file_format="structured",
        url=GOV_UK_COLLECTION_URL,
        fetched_at=fetched_at,
        local_path=target_file,
        notes=notes,
    )
    return [record]


def _html_targets() -> list[tuple[str, str, str]]:
    targets = [
        ("contents", f"{LEGISLATION_BASE_URL}/contents", "Building Regulations 2010 contents"),
        (
            "schedule_1",
            f"{LEGISLATION_BASE_URL}/schedule/1",
            "Building Regulations 2010 schedule 1",
        ),
    ]
    for regulation in range(3, 10):
        targets.append(
            (
                f"reg_{regulation:02d}",
                f"{LEGISLATION_BASE_URL}/regulation/{regulation}",
                f"Building Regulations 2010 regulation {regulation}",
            )
        )
    return targets


def _download_html_pages(
    session: requests.Session,
    html_dir: Path,
    force: bool,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    targets = _html_targets()
    for index, (slug, url, source_name) in enumerate(targets):
        target_file = html_dir / f"{slug}.html"
        downloaded, fetched_at = _save_text_file(
            session=session,
            url=url,
            destination=target_file,
            force=force,
        )
        notes = "Raw HTML page from legislation.gov.uk"
        if not downloaded:
            notes = f"Existing file reused; {notes}"

        records.append(
            _make_manifest_record(
                record_id=f"building_regulations_2010_{slug}_html",
                source_name=source_name,
                publisher="The National Archives (legislation.gov.uk)",
                file_format="html",
                url=url,
                fetched_at=fetched_at,
                local_path=target_file,
                notes=notes,
            )
        )

        if downloaded and index < len(targets) - 1:
            time.sleep(LEGISLATION_SLEEP_SECONDS)
    return records


def _write_manifest(records: Sequence[ManifestRecord]) -> None:
    ordered = sorted(records, key=lambda item: item.id)
    payload = [item.as_dict() for item in ordered]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote manifest with %d records: %s", len(payload), MANIFEST_PATH)


def _print_summary(records: Iterable[ManifestRecord]) -> None:
    rows = [(item.id, item.format, str(item.bytes), item.local_path) for item in records]
    rows = sorted(rows, key=lambda row: row[0])

    print("\nDownload summary")
    if not rows:
        print("No records generated.")
        return

    headers = ("id", "format", "bytes", "path")
    widths = [
        max(len(headers[0]), *(len(row[0]) for row in rows)),
        max(len(headers[1]), *(len(row[1]) for row in rows)),
        max(len(headers[2]), *(len(row[2]) for row in rows)),
        max(len(headers[3]), *(len(row[3]) for row in rows)),
    ]

    header_line = (
        f"{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}  {headers[3]}"
    )
    separator = "-" * len(header_line)
    print(header_line)
    print(separator)
    for row in rows:
        print(f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]:>{widths[2]}}  {row[3]}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(ALL_FORMATS),
        help="Limit to specific source format(s). Repeatable.",
    )
    return parser.parse_args(argv)


def run(force: bool = False, only: Sequence[str] | None = None) -> list[ManifestRecord]:
    """Download requested raw sources and rewrite data/manifest.json."""
    requested_formats = set(only or ALL_FORMATS)
    output_dirs = _ensure_output_dirs()
    records: list[ManifestRecord] = []

    with _build_session() as session:
        approved_docs: list[dict[str, Any]] = []
        if "pdf" in requested_formats or "structured" in requested_formats:
            collection_payload = _get_json(session, GOV_UK_COLLECTION_URL)
            approved_docs = _load_approved_documents_from_collection(collection_payload)

        if "pdf" in requested_formats:
            records.extend(_download_pdfs(session, approved_docs, output_dirs["pdf"], force=force))

        if "structured" in requested_formats:
            records.extend(
                _download_structured_catalogue(
                    approved_docs=approved_docs,
                    structured_dir=output_dirs["structured"],
                    force=force,
                )
            )

        if "html" in requested_formats:
            records.extend(_download_html_pages(session, output_dirs["html"], force=force))

    _write_manifest(records)
    _print_summary(records)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """Run the raw-source downloader CLI."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    selected = args.only or sorted(ALL_FORMATS)
    LOGGER.info("Starting download with formats=%s force=%s", selected, args.force)

    run(force=args.force, only=selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
