"""Standalone verification checks for the normalized SQLite DB (Step 3).

Run from the repo root:
    uv run python scripts/verify_db.py
or point it at a specific DB file:
    uv run python scripts/verify_db.py data/processed/clausefinder.sqlite

Reads only; makes no changes. Prints a PASS/FLAG per check.
"""

from __future__ import annotations

import os
import re
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DB = Path("data/processed/clausefinder.sqlite")

# Sources whose section labels should be (near-)unique per row. A duplicate
# (source, section) within these is a red flag: e.g. an in-force AND a
# prospective version of the same provision both landing as rows.
LEGISLATION_PREFIX = "building_regulations_2010_"

# A leaked local path would look like  D:\...  or  data/raw  or  data\raw.
# (We use a drive-letter + backslash regex so "Appendix D:" does NOT match.)
PATH_RE = re.compile(r"[A-Za-z]:\\")
DATARAW_PATTERNS = ("data/raw", "data\\raw")
# Optional private-name guard: set CLAUSEFINDER_FORBIDDEN_TOKEN locally to also
# scan rows for a string that must never reach the public repo. The token is
# read from the environment so it is never hardcoded in this file.
_token = os.environ.get("CLAUSEFINDER_FORBIDDEN_TOKEN", "").strip()
NAME_RE = re.compile(rf"\b{re.escape(_token)}\b", re.IGNORECASE) if _token else None


def _header(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> int:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT doc_id, source, jurisdiction, title, section, text, url, content_sha256 "
        "FROM documents"
    ).fetchall()

    print("=== ClauseFinder DB verification ===")
    print(f"DB: {db_path}")
    print(f"Total rows: {len(rows)}")

    per_source = Counter(r["source"] for r in rows)
    _header("Rows per source")
    for source, n in sorted(per_source.items()):
        print(f"  {source:44s} {n}")

    _header("Rows per jurisdiction (expect: England only)")
    per_juris = Counter(r["jurisdiction"] for r in rows)
    for juris, n in sorted(per_juris.items()):
        print(f"  {juris or '<empty>':44s} {n}")
    non_england = sorted(j for j in per_juris if j != "England")
    print("  [PASS]" if not non_england else f"  [FLAG] non-England values: {non_england}")

    _header("CHECK 1: duplicate (source, section) on regulation pages (expect none)")
    # Only reg_NN pages: each is a single regulation and should yield ~1 row, so a
    # duplicate here is the in-force/prospective leak signal. The schedule is parsed
    # differently and legitimately has many rows per Part, so it is excluded here.
    dup_leg = conn.execute(
        "SELECT source, section, COUNT(*) n FROM documents "
        "WHERE section <> '' AND source LIKE ? "
        "GROUP BY source, section HAVING n > 1 ORDER BY n DESC",
        (LEGISLATION_PREFIX + "reg_%",),
    ).fetchall()
    if not dup_leg:
        print("  <none>  [PASS]  (no in-force/prospective duplicate rows)")
    else:
        print("  [FLAG] possible in-force/prospective leak:")
        for r in dup_leg:
            print(f"    source={r['source']} section={r['section']} count={r['n']}")

    _header("CHECK 2: rows per legislation source (reg pages expect 1; schedule many)")
    for source, n in sorted(per_source.items()):
        if source.startswith(LEGISLATION_PREFIX):
            is_reg = source.startswith(LEGISLATION_PREFIX + "reg_")
            flag = "   <-- expected 1, look at this" if (is_reg and n > 1) else ""
            print(f"  {source:44s} {n}{flag}")

    _header("CHECK 3: exact-duplicate text bodies (same sha, different doc_id)")
    dup_text = conn.execute(
        "SELECT content_sha256, COUNT(*) n FROM documents "
        "GROUP BY content_sha256 HAVING n > 1 ORDER BY n DESC"
    ).fetchall()
    if not dup_text:
        print("  <none>  [PASS]")
    else:
        print(
            f"  [CHECK] {len(dup_text)} repeated text bodies "
            "(a little boilerplate repetition can be benign):"
        )
        for r in dup_text[:10]:
            print(f"    sha={r['content_sha256'][:12]} count={r['n']}")

    _header("CHECK 4: local-path / private-name leak scan (expect none)")
    offenders: list[tuple[str, str]] = []
    for r in rows:
        for col in ("url", "title", "section", "text"):
            val = r[col] or ""
            hit_path = PATH_RE.search(val) or any(p in val for p in DATARAW_PATTERNS)
            hit_name = NAME_RE.search(val) if NAME_RE else None
            if hit_path or hit_name:
                offenders.append((r["doc_id"], col))
                break
    if not offenders:
        extra = (
            ""
            if NAME_RE
            else "  (set CLAUSEFINDER_FORBIDDEN_TOKEN to also scan for a private name)"
        )
        print(f"  <none>  [PASS]  (no local path leaked){extra}")
    else:
        print(f"  [FLAG] {len(offenders)} rows with a leaked path or name:")
        for doc_id, col in offenders[:15]:
            print(f"    doc_id={doc_id} column={col}")

    _header("CHECK 5: data-quality counts")
    lengths = [len(r["text"]) for r in rows]
    empty_section = sum(1 for r in rows if not (r["section"] or "").strip())
    short_text = sum(1 for r in rows if len((r["text"] or "").strip()) < 30)
    print(f"  rows with empty section : {empty_section}")
    print(f"  rows with <30 char text : {short_text}")
    if lengths:
        print(
            "  text length min/median/max : "
            f"{min(lengths)} / {int(statistics.median(lengths))} / {max(lengths)}"
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
