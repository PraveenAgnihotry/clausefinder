"""Run ClauseFinder evaluation cases and write summary reports."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from clausefinder import config
from clausefinder.rag.generate import Answer, answer_question

SLEEP_BETWEEN_SECONDS = 4.0
RATE_LIMIT_RETRY_SECONDS = 30.0
RATE_LIMIT_HINTS = ("rate", "quota", "429")
NA = "—"


class ExpectedSource(TypedDict):
    """One expected citation target for an evaluation case."""

    source: str
    section: str


class EvalCase(TypedDict):
    """Input schema for a single evaluation case from eval_set.jsonl."""

    id: str
    question: str
    part: str
    expected_behavior: str
    expected_sources: list[ExpectedSource]
    answer_must_contain: list[str]
    answer_must_not_contain: list[str]
    known_failure: bool
    verified: bool
    notes: str


@dataclass(slots=True)
class CaseResult:
    """Evaluation result for one case, including metrics and errors."""

    case_id: str
    expected_behavior: str
    known_failure: bool
    verified: bool
    checkable: bool
    has_expected_section: bool
    doc_hit: bool | None
    sec_hit: bool | None
    behavior_ok: bool | None
    faithfulness_ok: bool | None
    answered: bool | None
    known_failure_note: str | None
    error: str | None


def load_cases() -> list[dict[str, Any]]:
    """Load evaluation cases from eval/eval_set.jsonl, skipping blank lines."""
    path = Path(__file__).with_name("eval_set.jsonl")
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def normalize(value: str) -> str:
    """Normalize text for case-insensitive matching."""
    return value.strip().casefold()


def section_match(expected: str, retrieved: str) -> bool:
    """Match two section labels with startswith in either direction."""
    expected_norm = normalize(expected)
    retrieved_norm = normalize(retrieved)
    if not expected_norm or not retrieved_norm:
        return False
    return expected_norm.startswith(retrieved_norm) or retrieved_norm.startswith(expected_norm)


def retrieval_hits(case: EvalCase, answer: Answer) -> tuple[bool, bool]:
    """Return document-level and section-level retrieval hit flags."""
    retrieved = [(normalize(chunk.source), normalize(chunk.section)) for chunk in answer.sources]

    expected_sources = case.get("expected_sources", [])
    doc_hit = any(
        any(normalize(exp.get("source", "")) == src for src, _ in retrieved)
        for exp in expected_sources
    )

    sec_hit = any(
        (normalize(exp.get("source", "")) == src and section_match(exp.get("section", ""), sec))
        for exp in expected_sources
        if normalize(exp.get("section", ""))
        for src, sec in retrieved
    )

    return doc_hit, sec_hit


def _is_rate_limit_error(message: str) -> bool:
    """Check if an exception message looks like rate-limit or quota pressure."""
    message_norm = message.casefold()
    return any(hint in message_norm for hint in RATE_LIMIT_HINTS)


def _safe_error_message(exc: Exception) -> str:
    """Return a stable, non-empty error string."""
    text = str(exc).strip()
    if text:
        return text
    return repr(exc)


def _to_bool_text(value: bool | None) -> str:
    """Render bool values for markdown tables."""
    if value is None:
        return NA
    return "yes" if value else "no"


def _escape_cell(text: str) -> str:
    """Escape markdown-table special chars in a single cell value."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _metric_row(name: str, numerator: int, denominator: int) -> tuple[str, str, str]:
    """Build a display row for ratio metrics."""
    if denominator <= 0:
        return (name, NA, "0")
    value = f"{(numerator / denominator):.1%}"
    return (name, value, str(denominator))


def _classify_known_failure(
    *,
    known_failure: bool,
    expected_behavior: str,
    behavior_ok: bool | None,
    faithfulness_ok: bool | None,
    answered: bool | None,
    checkable: bool,
    error: str | None,
) -> str | None:
    """Label known-failure outcomes for tracking regressions and fixes."""
    if not known_failure:
        return None
    if error is not None:
        return "error"

    passed: bool
    if expected_behavior == "refuse":
        passed = behavior_ok is True
    elif checkable:
        passed = faithfulness_ok is True
    else:
        passed = answered is True

    if passed:
        return "unexpected pass - bug may be fixed"
    return "expected fail (documented)"


def _evaluate_case(case: EvalCase, answer: Answer) -> CaseResult:
    """Evaluate one successful model response against case expectations."""
    expected_behavior = case.get("expected_behavior", "answer")
    known_failure = bool(case.get("known_failure", False))
    verified = bool(case.get("verified", False))

    has_expected_section = any(
        bool(normalize(exp.get("section", ""))) for exp in case.get("expected_sources", [])
    )

    if expected_behavior == "refuse":
        behavior_ok = answer.refused is True
        known_failure_note = _classify_known_failure(
            known_failure=known_failure,
            expected_behavior=expected_behavior,
            behavior_ok=behavior_ok,
            faithfulness_ok=None,
            answered=None,
            checkable=False,
            error=None,
        )
        return CaseResult(
            case_id=case.get("id", ""),
            expected_behavior=expected_behavior,
            known_failure=known_failure,
            verified=verified,
            checkable=False,
            has_expected_section=has_expected_section,
            doc_hit=None,
            sec_hit=None,
            behavior_ok=behavior_ok,
            faithfulness_ok=None,
            answered=None,
            known_failure_note=known_failure_note,
            error=None,
        )

    answered = not answer.refused
    must_contain = case.get("answer_must_contain", [])
    must_not_contain = case.get("answer_must_not_contain", [])
    response_text = answer.answer.casefold()

    contains_ok = (
        all(token.casefold() in response_text for token in must_contain) if must_contain else None
    )
    not_contains_ok = (
        all(token.casefold() not in response_text for token in must_not_contain)
        if must_not_contain
        else None
    )
    checkable = bool(must_contain or must_not_contain)
    faithfulness_ok = (
        answered and contains_ok is not False and not_contains_ok is not False
        if checkable
        else None
    )
    doc_hit, sec_hit = retrieval_hits(case, answer)

    known_failure_note = _classify_known_failure(
        known_failure=known_failure,
        expected_behavior=expected_behavior,
        behavior_ok=None,
        faithfulness_ok=faithfulness_ok,
        answered=answered,
        checkable=checkable,
        error=None,
    )

    return CaseResult(
        case_id=case.get("id", ""),
        expected_behavior=expected_behavior,
        known_failure=known_failure,
        verified=verified,
        checkable=checkable,
        has_expected_section=has_expected_section,
        doc_hit=doc_hit,
        sec_hit=sec_hit,
        behavior_ok=answered,
        faithfulness_ok=faithfulness_ok,
        answered=answered,
        known_failure_note=known_failure_note,
        error=None,
    )


def _error_result(case: EvalCase, error: str) -> CaseResult:
    """Build a result object when model execution fails for a case."""
    expected_behavior = case.get("expected_behavior", "answer")
    known_failure = bool(case.get("known_failure", False))
    has_expected_section = any(
        bool(normalize(exp.get("section", ""))) for exp in case.get("expected_sources", [])
    )
    known_failure_note = _classify_known_failure(
        known_failure=known_failure,
        expected_behavior=expected_behavior,
        behavior_ok=None,
        faithfulness_ok=None,
        answered=None,
        checkable=False,
        error=error,
    )
    return CaseResult(
        case_id=case.get("id", ""),
        expected_behavior=expected_behavior,
        known_failure=known_failure,
        verified=bool(case.get("verified", False)),
        checkable=bool(case.get("answer_must_contain") or case.get("answer_must_not_contain")),
        has_expected_section=has_expected_section,
        doc_hit=None,
        sec_hit=None,
        behavior_ok=None,
        faithfulness_ok=None,
        answered=None,
        known_failure_note=known_failure_note,
        error=error,
    )


def run_all(cases: list[EvalCase]) -> list[CaseResult]:
    """Execute all cases with pacing and one-shot rate-limit retry."""
    results: list[CaseResult] = []
    total = len(cases)

    for index, case in enumerate(cases):
        question = case.get("question", "")
        answer: Answer | None = None
        error: str | None = None

        for attempt in range(2):
            try:
                answer = answer_question(question, k=config.TOP_K)
                break
            except Exception as exc:  # noqa: BLE001
                message = _safe_error_message(exc)
                if attempt == 0 and _is_rate_limit_error(message):
                    time.sleep(RATE_LIMIT_RETRY_SECONDS)
                    continue
                error = message
                break

        if answer is not None:
            results.append(_evaluate_case(case, answer))
        else:
            results.append(_error_result(case, error or "unknown error"))

        if index < total - 1:
            time.sleep(SLEEP_BETWEEN_SECONDS)

    return results


def aggregate_metrics(results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    """Compute headline metrics using requested denominators."""
    answer_cases = [r for r in results if r.expected_behavior == "answer" and r.error is None]
    answer_with_section = [r for r in answer_cases if r.has_expected_section]
    refuse_cases = [r for r in results if r.expected_behavior == "refuse" and r.error is None]

    faithfulness_verified = [
        r
        for r in answer_cases
        if r.checkable and r.verified and not r.known_failure and r.faithfulness_ok is not None
    ]
    faithfulness_all_annotated = [
        r
        for r in answer_cases
        if r.checkable and not r.known_failure and r.faithfulness_ok is not None
    ]

    known_failures = [r for r in results if r.known_failure]
    failed_as_expected = [
        r for r in known_failures if r.known_failure_note == "expected fail (documented)"
    ]
    unexpected_passes = [
        r for r in known_failures if r.known_failure_note == "unexpected pass - bug may be fixed"
    ]

    metrics: dict[str, tuple[int, int]] = {
        "Retrieval hit-rate (document-level)": (
            sum(1 for r in answer_cases if r.doc_hit is True),
            len(answer_cases),
        ),
        "Retrieval hit-rate (section-level)": (
            sum(1 for r in answer_with_section if r.sec_hit is True),
            len(answer_with_section),
        ),
        "Refusal accuracy": (
            sum(1 for r in refuse_cases if r.behavior_ok is True),
            len(refuse_cases),
        ),
        "Faithfulness (verified)": (
            sum(1 for r in faithfulness_verified if r.faithfulness_ok is True),
            len(faithfulness_verified),
        ),
        "Faithfulness (all annotated; includes unverified expectations)": (
            sum(1 for r in faithfulness_all_annotated if r.faithfulness_ok is True),
            len(faithfulness_all_annotated),
        ),
        "Known failures - failed as expected": (
            len(failed_as_expected),
            len(known_failures),
        ),
        "Known failures - unexpectedly passed": (
            len(unexpected_passes),
            len(known_failures),
        ),
        "Errors": (sum(1 for r in results if r.error is not None), len(results)),
    }
    return metrics


def print_summary(metrics: dict[str, tuple[int, int]]) -> None:
    """Print compact metric summary to stdout."""
    print("metric | value | n")
    print("--- | --- | ---")
    for name, (num, den) in metrics.items():
        if name == "Errors":
            value = str(num)
            n_display = str(den)
        else:
            _, value, n_display = _metric_row(name, num, den)
            if value != NA:
                value = f"{value} ({num}/{den})"
        print(f"{name} | {value} | {n_display}")


def _case_status(result: CaseResult) -> str:
    """Build per-case status text for behavior/faithfulness column."""
    if result.error:
        return NA
    if result.expected_behavior == "refuse":
        return "behavior ok" if result.behavior_ok else "behavior fail"
    if result.checkable:
        return "faithful" if result.faithfulness_ok else "not faithful"
    return "answered" if result.answered else "refused"


def write_results_md(results: list[CaseResult], metrics: dict[str, tuple[int, int]]) -> None:
    """Write markdown results to eval/results.md for README copy/paste."""
    out_path = Path(__file__).with_name("results.md")
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()

    lines = [
        "# Evaluation Results",
        "",
        f"Model: `{config.GEMINI_MODEL}`  ",
        f"Run UTC: `{timestamp}`",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value | n |",
        "| --- | --- | --- |",
    ]

    for name, (num, den) in metrics.items():
        if name == "Errors":
            value = str(num)
            n_display = str(den)
        else:
            _, pct, n_display = _metric_row(name, num, den)
            value = NA if pct == NA else f"{pct} ({num}/{den})"
        lines.append(f"| {name} | {value} | {n_display} |")

    lines.extend(
        [
            "",
            "## Per-Case Results",
            "",
            "| id | type | doc hit | section hit | behavior/faithfulness | known_fail | error |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for result in results:
        known_fail = result.known_failure_note if result.known_failure_note else NA
        error = _escape_cell(result.error) if result.error else NA
        lines.append(
            "| "
            f"{_escape_cell(result.case_id)} | "
            f"{result.expected_behavior} | "
            f"{_to_bool_text(result.doc_hit)} | "
            f"{_to_bool_text(result.sec_hit)} | "
            f"{_case_status(result)} | "
            f"{_escape_cell(known_fail)} | "
            f"{error} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the full evaluation harness and emit console + markdown reports."""
    raw_cases = load_cases()
    cases = [case for case in raw_cases if isinstance(case, dict)]
    typed_cases: list[EvalCase] = [case for case in cases]

    results = run_all(typed_cases)
    metrics = aggregate_metrics(results)
    print_summary(metrics)
    write_results_md(results, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
