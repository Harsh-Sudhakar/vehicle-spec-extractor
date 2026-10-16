"""High-level API used by the CLI, the web server and the notebook."""

from __future__ import annotations

import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import pymupdf

from .config import get_settings
from .index.store import DocIndex
from .ingest.chunk import build_chunks, file_hash
from .query.extract import extract_specs
from .query.retrieve import retrieve
from .schema import DocInfo, QueryResult

ProgressFn = Callable[[str, float], None]

_cache: "OrderedDict[str, DocIndex]" = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_SIZE = 4


def ingest(pdf_path: str | Path, filename: str | None = None, progress: ProgressFn | None = None,
           force: bool = False, enable_ocr: bool | None = None) -> DocInfo:
    """Parse, chunk, embed and persist a PDF. Re-ingesting the same file is a cache hit."""
    s = get_settings()
    pdf_path = Path(pdf_path)
    doc_id = file_hash(str(pdf_path))
    if DocIndex.exists(doc_id) and not force:
        return DocIndex.load(doc_id).info

    progress = progress or (lambda stage, frac: None)
    t0 = time.perf_counter()
    enable_ocr = s.enable_ocr if enable_ocr is None else enable_ocr
    chunks, stats = build_chunks(str(pdf_path), doc_id, enable_ocr=enable_ocr, progress=progress)

    with pymupdf.open(str(pdf_path)) as doc:
        title = (doc.metadata or {}).get("title") or None
    if title and title.lower() in ("untitled", "unknown"):
        title = None

    info = DocInfo(
        doc_id=doc_id,
        filename=filename or pdf_path.name,
        pages=stats.pages,
        chunks=len(chunks),
        table_chunks=sum(c.chunk_type == "table" for c in chunks),
        text_chunks=sum(c.chunk_type == "text" for c in chunks),
        ocr_pages=stats.ocr_pages,
        skipped_pages={"toc_or_index": stats.toc_pages, "image_only": stats.image_pages, "blank": stats.blank_pages},
        has_outline=stats.has_outline,
        title=title,
    )
    index = DocIndex.build(info, chunks, progress=progress)
    info.ingest_ms = round((time.perf_counter() - t0) * 1000)
    index.info = info
    index.save()

    # keep the PDF next to its index so the UI can render source pages
    dest = DocIndex.dir_for(doc_id) / "source.pdf"
    if pdf_path.resolve() != dest.resolve():
        shutil.copyfile(pdf_path, dest)
    with _cache_lock:
        _cache[doc_id] = index
    progress("done", 1.0)
    return info


def load_index(doc_id: str) -> DocIndex:
    with _cache_lock:
        if doc_id in _cache:
            _cache.move_to_end(doc_id)
            return _cache[doc_id]
    index = DocIndex.load(doc_id)
    with _cache_lock:
        _cache[doc_id] = index
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return index


def source_pdf(doc_id: str) -> Path:
    return DocIndex.dir_for(doc_id) / "source.pdf"


def query(doc_id: str, question: str, top_k: int | None = None, offline: bool = False) -> QueryResult:
    index = load_index(doc_id)
    timings: dict[str, float] = {}

    hits = retrieve(index, question, final_k=top_k, timings=timings)

    t = time.perf_counter()
    specs, rejected, model, mode = extract_specs(question, hits, offline=offline)
    timings["extract"] = round((time.perf_counter() - t) * 1000, 1)

    return QueryResult(query=question, specs=specs, chunks=hits, rejected=rejected,
                       timings_ms=timings, model=model, mode=mode)
