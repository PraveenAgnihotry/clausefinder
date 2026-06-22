"""Smoke test: verify Gemini API connectivity via the google-genai SDK.

Run once to de-risk the external dependency:
    uv run python scripts/check_gemini.py

Requires GEMINI_API_KEY in .env (never committed).
"""

import os
import sys

from dotenv import load_dotenv
from google import genai

from clausefinder.config import GEMINI_MODEL


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Copy .env.example to .env and add your key.")
        return 1

    client = genai.Client(api_key=api_key)

    # 1) Confirm the key works and show which Gemini models it can access.
    print("Available Gemini models for this key:")
    try:
        for model in client.models.list():
            if "gemini" in model.name:
                print(f"  - {model.name}")
    except Exception as exc:
        print(f"  (could not list models: {exc})")

    # 2) Make one real generation call with the configured model.
    print(f"\nCalling {GEMINI_MODEL} ...")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with one short sentence confirming you are working.",
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        print(
            "If this is a model-not-found error, pick one of the model names listed above "
            "and update GEMINI_MODEL in src/clausefinder/config.py."
        )
        return 1

    print("SUCCESS. Response:")
    print(response.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
