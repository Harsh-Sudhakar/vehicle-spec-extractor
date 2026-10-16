"""Hybrid retrieval: dense (BGE) + sparse (BM25) fused with Reciprocal Rank Fusion,
then re-scored with a cross-encoder.

* Dense search understands paraphrases ("how much oil does the diesel take").
* BM25 nails exact tokens that embeddings blur: part numbers, spec codes, "Nm".
* The cross-encoder reads query and chunk together, which fixes most ranking errors
  of the two first-stage retrievers.
"""

from __future__ import annotations

import time

from ..config import get_settings
from ..index.models import rerank
from ..index.store import DocIndex
from ..schema import RetrievedChunk

RRF_K = 60


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve(
    index: DocIndex,
    query: str,
    final_k: int | None = None,
    use_rerank: bool = True,
    timings: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    s = get_settings()
    final_k = final_k or s.final_k
    timings = timings if timings is not None else {}
    t0 = time.perf_counter()

    dense = index.dense_search(query, s.dense_k)
    sparse = index.sparse_search(query, s.sparse_k)
    dense_rank = {i: r for r, (i, _) in enumerate(dense)}
    sparse_rank = {i: r for r, (i, _) in enumerate(sparse)}

    fused = rrf([[i for i, _ in dense], [i for i, _ in sparse]])
    candidates = sorted(fused, key=fused.get, reverse=True)[: s.rerank_k]
    timings["retrieve"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()

    if use_rerank and candidates:
        ce = rerank(query, [index.chunks[i].embed_text() for i in candidates])
        scored = []
        for i, score in zip(candidates, ce):
            c = index.chunks[i]
            # Small priors: chunks that actually contain values, and table rows, are
            # more likely to answer a spec question than narrative text.
            score += 0.75 if c.has_spec else 0.0
            score += 0.25 if c.chunk_type == "table" else 0.0
            scored.append((i, score))
    else:
        scored = [(i, fused[i]) for i in candidates]

    scored.sort(key=lambda x: x[1], reverse=True)
    timings["rerank"] = round((time.perf_counter() - t0) * 1000, 1)
    return [
        RetrievedChunk(
            chunk=index.chunks[i],
            score=round(score, 4),
            dense_rank=dense_rank.get(i),
            sparse_rank=sparse_rank.get(i),
        )
        for i, score in scored[:final_k]
    ]
