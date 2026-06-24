# clausefinder

A Retrieval-Augmented Generation (RAG) assistant that answers building-regulation and
compliance questions from a corpus of **public** construction regulations, returning grounded
answers **with exact source citations** — or clearly stating when the answer is not in its sources.

corpus scoped to the building-regulations regime as it applies in England, plus the "known limitations" bullets above.

> Status: work in progress.

## What problem are you solving, and for whom?

_TODO_

## Why this matters (technical inspection & construction compliance)

_TODO_

## Data sources

_TODO_

## Technical decisions & trade-offs

_TODO_

## Production tomorrow vs. throw away

_TODO_

## With 3 more months

_TODO_

## Setup

```bash
# 1. Install uv: https://docs.astral.sh/uv/
# 2. Install dependencies (also installs this package in editable mode)
uv sync
# 3. Configure secrets
cp .env.example .env   # then add your GEMINI_API_KEY
# 4. Build the corpus + index (download -> process -> index)
uv run python scripts/build_all.py
# 5. Run the app
uv run streamlit run app/streamlit_app.py
```

## Architecture

ingest → parse/clean → normalize → chunk → embed → FAISS index → retrieve → grounded generation (Gemini) → Streamlit UI.
