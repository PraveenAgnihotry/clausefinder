"""CLI smoke test for end-to-end grounded question answering."""

from __future__ import annotations

import argparse

from clausefinder import config
from clausefinder.rag.generate import Answer, answer_question


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the ask CLI."""
    parser = argparse.ArgumentParser(description="Ask a grounded question to ClauseFinder.")
    parser.add_argument("question", type=str, help="Question text to answer.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"Optional top-k override (default: None, uses config.TOP_K={config.TOP_K}).",
    )
    return parser


def _print_result(result: Answer) -> None:
    """Print the answer payload in a human-readable CLI format."""
    print(f"Question: {result.query}")
    print(f"Answer: {result.answer}")
    print("Sources:")

    for i, source in enumerate(result.sources, start=1):
        print(
            f"[{i}] {source.source} \u2014 {source.section} "
            f"\u2014 score {source.score:.3f} \u2014 {source.url}"
        )

    print(f"Status: refused={result.refused} low_confidence={result.low_confidence}")


def main() -> int:
    """Run the ask CLI and return a process exit code."""
    args = _build_parser().parse_args()

    try:
        result = answer_question(args.question, k=args.k)
    except Exception as exc:
        print(f"Error: could not answer question: {exc}")
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
