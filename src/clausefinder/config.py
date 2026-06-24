"""Central configuration: paths, model names, retrieval params, and ML-cache setup.

IMPORTANT: import this module *before* importing transformers / sentence_transformers.
It redirects all model caches to the repo-local ./models directory so nothing is ever
written to the C-drive user cache.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# config.py lives at src/clausefinder/config.py -> repo root is parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# --- ML cache redirection (MUST run before importing any model library) ------
# Hard-set (not setdefault) so a global HF_HOME can never send caches to C:.
_MODELS = str(MODELS_DIR)
os.environ["HF_HOME"] = _MODELS
os.environ["HF_HUB_CACHE"] = _MODELS
os.environ["TRANSFORMERS_CACHE"] = _MODELS
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _MODELS

# Ensure runtime directories exist (safe on a fresh clone).
for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Models ------------------------------------------------------------------
# English corpus -> bge-small-en. Switch to "BAAI/bge-m3" only if multilingual sources added.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 64
EMBED_NORMALIZE = True
EMBED_DEVICE = None  # None -> auto: cuda if available else cpu
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Free-tier stable Flash. Verify availability for your key with client.models.list().
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_OUTPUT_TOKENS = 1024

# --- Retrieval / chunking ----------------------------------------------------
TOP_K = 5
RETRIEVAL_MIN_SCORE = 0.30
CHUNK_SIZE = 800  # characters; tune during indexing
CHUNK_OVERLAP = 150  # characters
CHUNK_MAX_TOKENS = 480
CHUNK_TARGET_TOKENS = 384
CHUNK_OVERLAP_TOKENS = 64
APPROX_CHARS_PER_TOKEN = 4
EMBED_EXCLUDED_SOURCES = {"approved_documents_catalogue_json"}

# --- Build artifacts ---------------------------------------------------------
FAISS_INDEX_PATH = PROCESSED_DIR / "clausefinder.faiss"
INDEX_META_PATH = PROCESSED_DIR / "index_meta.json"
SQLITE_DB_PATH = PROCESSED_DIR / "clausefinder.sqlite"
