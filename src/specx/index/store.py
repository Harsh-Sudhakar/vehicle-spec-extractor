"""Per-document index: FAISS (dense) + BM25 (sparse), persisted on disk.

Layout: ``data/indexes/<doc_id>/{chunks.jsonl, dense.faiss, info.json}``.
BM25 is rebuilt on load (it takes milliseconds) so only JSON + FAISS are stored.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ..config import get_settings
from ..schema import Chunk, DocInfo
from .models import embed_passages, embed_query

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by for from how in is it of on or the to what which with when where "
    "does do much many value spec specs specification specifications".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens. Compound tokens (WSS-M6C57-A2, 175/65) are kept and also split."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        out.append(tok)
        if any(c in tok for c in ".-/"):
            out.extend(p for p in re.split(r"[.\-/]", tok) if p and p not in _STOP)
    return out


class DocIndex:
    def __init__(self, info: DocInfo, chunks: list[Chunk], dense: faiss.Index):
        self.info = info
        self.chunks = chunks
        self.dense = dense
        self.bm25 = BM25Okapi([tokenize(c.embed_text()) for c in chunks])
        self.by_id = {c.id: c for c in chunks}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, info: DocInfo, chunks: list[Chunk], progress=None) -> "DocIndex":
        texts = [c.embed_text() for c in chunks]
        vecs = []
        step = 128
        for i in range(0, len(texts), step):
            vecs.append(embed_passages(texts[i : i + step]))
            if progress:
                progress("embed", min(1.0, (i + step) / max(1, len(texts))))
        matrix = np.vstack(vecs) if vecs else np.zeros((0, 384), dtype="float32")
        dense = faiss.IndexFlatIP(matrix.shape[1])
        dense.add(matrix)
        idx = cls(info, chunks, dense)
        idx.save()
        return idx

    # ------------------------------------------------------------------ io
    @staticmethod
    def dir_for(doc_id: str) -> Path:
        return get_settings().index_dir / doc_id

    def save(self) -> None:
        d = self.dir_for(self.info.doc_id)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        faiss.write_index(self.dense, str(d / "dense.faiss"))
        (d / "info.json").write_text(self.info.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, doc_id: str) -> "DocIndex":
        d = cls.dir_for(doc_id)
        info = DocInfo.model_validate_json((d / "info.json").read_text(encoding="utf-8"))
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        dense = faiss.read_index(str(d / "dense.faiss"))
        return cls(info, chunks, dense)

    @classmethod
    def exists(cls, doc_id: str) -> bool:
        d = cls.dir_for(doc_id)
        return (d / "info.json").exists() and (d / "dense.faiss").exists()

    @classmethod
    def delete(cls, doc_id: str) -> None:
        shutil.rmtree(cls.dir_for(doc_id), ignore_errors=True)

    @staticmethod
    def list_infos() -> list[DocInfo]:
        root = get_settings().index_dir
        if not root.exists():
            return []
        infos = []
        for p in sorted(root.iterdir()):
            f = p / "info.json"
            if f.exists():
                infos.append(DocInfo.model_validate_json(f.read_text(encoding="utf-8")))
        return infos

    # ------------------------------------------------------------------ search
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        scores, ids = self.dense.search(embed_query(query), min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks or not self.chunks:
            return []
        scores = self.bm25.get_scores(toks)
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
