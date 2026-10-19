"""Lazy, process-wide model singletons (ONNX via fastembed, CPU-friendly, no torch)."""

from __future__ import annotations

import logging
import math
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()


def cpu_quota(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """CPU cores this container may actually use (cgroup quota), or None if unlimited.

    Hosts like Render's free plan expose all of the machine's cores to os.cpu_count() but
    allow the container ~0.1 of one. ONNX starts a thread per visible core; on a 0.1 CPU
    quota those threads mostly wait for each other, so sizing the pool to the quota matters.
    """
    try:
        v2 = root / "cpu.max"
        if v2.exists():
            quota, period = v2.read_text().split()[:2]
            return None if quota == "max" else int(quota) / int(period)
        q = root / "cpu" / "cpu.cfs_quota_us"
        p = root / "cpu" / "cpu.cfs_period_us"
        if q.exists() and p.exists():
            quota = int(q.read_text())
            return None if quota <= 0 else quota / int(p.read_text())
    except (OSError, ValueError):
        pass
    return None


@lru_cache(maxsize=1)
def model_threads() -> int | None:
    """Threads for the ONNX models: EMBED_THREADS if set, else the CPU quota, else ONNX's default."""
    s = get_settings()
    if s.embed_threads > 0:
        return s.embed_threads
    quota = cpu_quota()
    threads = None if quota is None else max(1, min(math.ceil(quota), os.cpu_count() or 1))
    log.info("cpu quota %s, visible cores %s -> model threads %s", quota, os.cpu_count(), threads or "default")
    return threads


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding

    s = get_settings()
    with _lock:
        return TextEmbedding(model_name=s.embed_model, cache_dir=str(s.model_cache_dir), threads=model_threads())


@lru_cache(maxsize=1)
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    s = get_settings()
    with _lock:
        return TextCrossEncoder(model_name=s.rerank_model, cache_dir=str(s.model_cache_dir), threads=model_threads())


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.clip(norms, 1e-12, None)).astype("float32")


def embed_passages(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    # One text at a time by default: ONNX pads a batch to its longest text, so on a manual's
    # chunks a batch of 16 was ~4x slower than batch 1 and used ~330 MB more RAM (measured).
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
