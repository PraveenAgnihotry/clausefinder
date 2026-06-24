"""Local embedding helpers for documents and queries."""

# ruff: noqa: I001

import clausefinder.config as config

# Import config before sentence-transformers so cache env vars are set first.
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


def _resolve_device() -> str:
    """Resolve embedding device from config override or torch CUDA availability."""
    if config.EMBED_DEVICE:
        return config.EMBED_DEVICE

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=8)
def _load_embedder_cached(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(
        model_name,
        device=device,
        cache_folder=str(config.MODELS_DIR),
    )


def load_embedder(model_name: str | None = None, device: str | None = None) -> SentenceTransformer:
    """Load and cache the sentence-transformer embedder instance."""
    resolved_model = model_name or config.EMBEDDING_MODEL
    resolved_device = device or _resolve_device()
    return _load_embedder_cached(resolved_model, resolved_device)


def embed_texts(
    texts: list[str],
    *,
    model: SentenceTransformer | None = None,
    batch_size: int | None = None,
    normalize: bool | None = None,
    is_query: bool = False,
) -> np.ndarray:
    """Embed texts as float32 C-contiguous vectors suitable for FAISS."""
    resolved_model = model or load_embedder()
    resolved_batch_size = batch_size if batch_size is not None else config.EMBED_BATCH_SIZE
    resolved_normalize = normalize if normalize is not None else config.EMBED_NORMALIZE

    prepared_texts = texts
    if is_query and config.QUERY_INSTRUCTION:
        # BGE v1.5 expects instruction prefixes for queries only, not for documents.
        prepared_texts = [f"{config.QUERY_INSTRUCTION}{text}" for text in texts]

    if not prepared_texts:
        dim = embedding_dim(model=resolved_model)
        return np.ascontiguousarray(np.empty((0, dim), dtype="float32"), dtype="float32")

    embeddings = resolved_model.encode(
        prepared_texts,
        batch_size=resolved_batch_size,
        normalize_embeddings=resolved_normalize,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.ascontiguousarray(embeddings, dtype="float32")


def get_token_counter() -> Callable[[str], int]:
    """Return a token-count closure based on the embedder tokenizer."""
    tokenizer = load_embedder().tokenizer

    def _count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return _count_tokens


def embedding_dim(model: SentenceTransformer | None = None) -> int:
    """Return sentence embedding dimensionality for the configured model."""
    resolved_model = model or load_embedder()
    return resolved_model.get_sentence_embedding_dimension()
