"""Lazy, process-wide model singletons (ONNX via fastembed, CPU-friendly, no torch)."""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from ..config import get_settings

_lock = threading.Lock()


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding

    s = get_settings()
    with _lock:
        return TextEmbedding(model_name=s.embed_model, cache_dir=str(s.model_cache_dir))


@lru_cache(maxsize=1)
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    s = get_settings()
    with _lock:
        return TextCrossEncoder(model_name=s.rerank_model, cache_dir=str(s.model_cache_dir))


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.clip(norms, 1e-12, None)).astype("float32")


def embed_passages(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    # Small batches: ONNX pads every text in a batch to the longest one, so a batch of 16
    # chunks costs ~330 MB extra RAM and is *slower*; batch 1 adds ~15 MB (measured).
    batch_size = batch_size or get_settings().embed_batch_size
    vecs = list(embedder().passage_embed(texts, batch_size=batch_size))
    return _normalise(np.vstack(vecs))


def embed_query(text: str) -> np.ndarray:
    # passage/query asymmetry: fastembed adds the BGE retrieval instruction for queries
    vec = next(iter(embedder().query_embed(text)))
    return _normalise(np.asarray(vec)[None, :])


def rerank(query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    # small batches keep peak memory down; the cost in speed is negligible for 30 docs
    return [float(s) for s in reranker().rerank(query, docs, batch_size=8)]


def warmup() -> None:
    """Download / load models up front (used by the Docker build and server start-up)."""
    embed_query("warm up")
    if get_settings().use_reranker:
        rerank("warm up", ["warm up"])
