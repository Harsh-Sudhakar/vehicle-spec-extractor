"""FastAPI app: JSON API + the static web UI.

Run locally:  uvicorn specx.server:app --reload   (from the repo root, with PYTHONPATH=src)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import pymupdf
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, pipeline
from .config import get_settings
from .index.store import DocIndex
from .ingest.chunk import file_hash
from .ingest.ocr import ocr_available
from .query.llm import llm_available
from .schema import DocInfo

log = logging.getLogger("specx")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parents[2] / "web" / "static"

app = FastAPI(title="SpecX", version=__version__)

# --------------------------------------------------------------------------- jobs
_jobs: dict[str, dict] = {}
_ingest_lock = threading.Lock()  # one ingestion at a time keeps memory predictable


def _run_ingest(job_id: str, path: Path, filename: str, owner: str) -> None:
    job = _jobs[job_id]

    def progress(stage: str, frac: float) -> None:
        job.update(stage=stage, progress=round(frac, 3))

    job.update(status="queued", stage="queued")
    with _ingest_lock:
        job.update(status="running", started=time.time())
        try:
            doc_id = file_hash(str(path))
            before = DocIndex.read_info(doc_id) if DocIndex.exists(doc_id) else None
            info = pipeline.ingest(path, filename=filename, progress=progress)
            # A new file, or one another visitor uploaded: it becomes (also) this visitor's
            # private upload. Re-uploading a shared demo manual just opens the demo copy.
            if before is None or before.is_upload:
                info.owners = sorted(set(info.owners) | {owner})
                info.uploaded_at = time.time()
                info.filename = before.filename if before else filename
                pipeline.save_info(info)
            job.update(status="done", stage="done", progress=1.0, doc=_public(info, owner))
        except Exception as e:  # surface the reason to the UI
            log.exception("ingest failed")
            job.update(status="error", error=str(e) or e.__class__.__name__)
        finally:
            path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- rate limit
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request, cost: int = 1) -> None:
    limit = get_settings().rate_limit_per_min
    if limit <= 0:
        return
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) + cost > limit:
        raise HTTPException(429, "Too many requests - please wait a minute.")
    for _ in range(cost):
        q.append(now)


# --------------------------------------------------------------------------- visitors
# The browser creates a random id and sends it with every request (header, or ?v= for
# <img>/<a> links). Uploads are visible only to the ids that uploaded them. This keeps
# visitors' uploads apart on a public demo; it is not authentication.
_VISITOR_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _visitor(request: Request) -> str | None:
    raw = request.headers.get("x-specx-visitor") or request.query_params.get("v") or ""
    return hashlib.sha256(raw.encode()).hexdigest()[:24] if _VISITOR_ID.match(raw) else None


def _visible(info: DocInfo, owner: str | None) -> bool:
    return not info.is_upload or (owner is not None and owner in info.owners)


def _expires_at(info: DocInfo) -> str | None:
    ttl = get_settings().upload_ttl_hours
    if not info.is_upload or ttl <= 0:
        return None
    return dt.datetime.fromtimestamp(info.uploaded_at + ttl * 3600, dt.timezone.utc).isoformat(timespec="seconds")


def _public(info: DocInfo, owner: str | None) -> dict:
    """Document metadata for one visitor (other visitors' ids are never sent)."""
    d = info.model_dump(exclude={"owners", "uploaded_at"})
    d.update(kind="upload" if info.is_upload else "demo", expires_at=_expires_at(info))
    return d


def _doc_for(doc_id: str, request: Request) -> DocInfo:
    """The document, if this visitor may see it. Otherwise 404, so others' uploads stay invisible."""
    if not DocIndex.exists(doc_id):
        raise HTTPException(404, "Document not found.")
    info = DocIndex.read_info(doc_id)
    if not _visible(info, _visitor(request)):
        raise HTTPException(404, "Document not found.")
    return info


def _purge_expired() -> None:
    ttl = get_settings().upload_ttl_hours
    if ttl <= 0:
        return
    for info in DocIndex.list_infos():
        if info.is_upload and time.time() - info.uploaded_at > ttl * 3600:
            log.info("upload expired, deleting %s", info.filename)
            pipeline.forget(info.doc_id)


# --------------------------------------------------------------------------- api
class QueryIn(BaseModel):
    doc_id: str
    query: str = Field(min_length=2, max_length=300)
    top_k: int | None = Field(default=None, ge=3, le=15)
    offline: bool = False


@app.get("/api/health")
def health():
    s = get_settings()
    return {
        "version": __version__,
        "llm": llm_available(),
        "model": s.openrouter_model if llm_available() else None,
        "embed_model": s.embed_model,
        "rerank_model": s.rerank_model,
        "ocr": s.enable_ocr and ocr_available(),
        "max_upload_mb": s.max_upload_mb,
        "max_upload_pages": s.max_upload_pages,
        "upload_ttl_hours": s.upload_ttl_hours,
        "max_uploads_per_visitor": s.max_uploads_per_visitor,
        "reranker": s.use_reranker,
        "allow_upload": s.allow_upload,
        "default_doc": s.default_doc,
        "default_query": s.default_query,
    }


@app.get("/api/documents")
def documents(request: Request):
    """Demo manuals plus this visitor's own uploads; the default manual first, then by name."""
    _purge_expired()
    owner = _visitor(request)
    default = get_settings().default_doc
    infos = sorted(
        (i for i in DocIndex.list_infos() if _visible(i, owner)),
        key=lambda i: (default not in (i.filename, i.doc_id), i.is_upload, i.filename.lower()),
    )
    return [_public(i, owner) for i in infos]


@app.post("/api/documents")
async def upload(request: Request, background: BackgroundTasks, file: UploadFile = File(...)):
    _rate_limit(request, cost=3)
    s = get_settings()
    if not s.allow_upload:
        raise HTTPException(403, "Uploads are disabled on this demo.")
    owner = _visitor(request)
    if owner is None:
        raise HTTPException(400, "Missing visitor id - reload the page and try again.")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    _purge_expired()
    if s.max_uploads_per_visitor > 0:
        mine = sum(1 for i in DocIndex.list_infos() if i.is_upload and owner in i.owners)
        pending = sum(1 for j in _jobs.values() if j.get("owner") == owner and j["status"] in ("queued", "running"))
        if mine + pending >= s.max_uploads_per_visitor:
            raise HTTPException(
                429, f"You can keep up to {s.max_uploads_per_visitor} uploaded manuals at a time - delete one first.")
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(suffix=".pdf", dir=s.upload_dir)[1])
    size = 0
    with open(tmp, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > s.max_upload_mb * (1 << 20):
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, _too_big(f"{size / (1 << 20):.0f}+ MB"))
            out.write(chunk)
    try:
        with pymupdf.open(tmp) as d:  # counting pages reads only the page tree, not content
            pages = d.page_count
        if pages == 0:
            raise ValueError
    except Exception:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "That file could not be opened as a PDF.")
    if s.max_upload_pages and pages > s.max_upload_pages:
        tmp.unlink(missing_ok=True)
        raise HTTPException(413, _too_big(f"{pages} pages"))

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"id": job_id, "filename": file.filename, "status": "queued", "stage": "queued",
                     "progress": 0.0, "pages": pages, "owner": owner}
    background.add_task(_run_ingest, job_id, tmp, file.filename, owner)
    return {"job_id": job_id, "pages": pages}


def _too_big(yours: str) -> str:
    s = get_settings()
    limits = f"{s.max_upload_pages} pages / {s.max_upload_mb} MB" if s.max_upload_pages else f"{s.max_upload_mb} MB"
    return f"This demo indexes PDFs up to {limits} (yours: {yours}). Run it locally for larger manuals - see the README."


@app.get("/api/jobs/{job_id}")
def job(job_id: str, request: Request):
    j = _jobs.get(job_id)
    if j is None or j.get("owner") != _visitor(request):
        raise HTTPException(404, "Unknown job")
    return {k: v for k, v in j.items() if k != "owner"}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, request: Request):
    """Remove one of your uploads (the file is deleted once no visitor owns it)."""
    info = _doc_for(doc_id, request)
    if not info.is_upload:
        raise HTTPException(403, "The demo manual can't be deleted.")
    info.owners = [o for o in info.owners if o != _visitor(request)]
    if info.owners:
        pipeline.save_info(info)
    else:
        pipeline.forget(doc_id)
    return {"deleted": doc_id}


@app.post("/api/query")
def run_query(body: QueryIn, request: Request):
    _rate_limit(request)
    _doc_for(body.doc_id, request)
    try:
        result = pipeline.query(body.doc_id, body.query.strip(), top_k=body.top_k, offline=body.offline)
    except Exception as e:
        log.exception("query failed")
        raise HTTPException(502, f"Extraction failed: {e}")
    return JSONResponse(result.model_dump())


@app.get("/api/documents/{doc_id}/pages/{page}.png")
def page_image(request: Request, doc_id: str, page: int, hl: list[str] | None = Query(None), dpi: int = 110):
    """Render one source page, highlighting the given strings (value, quote) if found."""
    info = _doc_for(doc_id, request)
    path = pipeline.source_pdf(doc_id)
    if not path.exists():
        raise HTTPException(404, "Source PDF not available")
    dpi = max(60, min(dpi, 200))
    with pymupdf.open(path) as doc:
        if not 1 <= page <= doc.page_count:
            raise HTTPException(404, "Page out of range")
        p = doc[page - 1]
        for needle in (hl or [])[:6]:
            needle = needle.strip()
            if len(needle) < 2:
                continue
            for rect in p.search_for(needle, quads=False)[:8]:
                annot = p.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 0.45, 0.12))
                annot.set_opacity(0.45)
                annot.update()
        png = p.get_pixmap(dpi=dpi, annots=True).tobytes("png")
    cache = "private, max-age=600" if info.is_upload else "public, max-age=3600"
    return Response(png, media_type="image/png", headers={"Cache-Control": cache})


@app.get("/api/documents/{doc_id}/source.pdf")
def source(doc_id: str, request: Request):
    _doc_for(doc_id, request)
    path = pipeline.source_pdf(doc_id)
    if not path.exists():
        raise HTTPException(404, "Source PDF not available")
    return FileResponse(path, media_type="application/pdf")


# --------------------------------------------------------------------------- startup
def _pull_prebuilt_indexes() -> None:
    """Download prebuilt indexes from a (private) Hugging Face dataset, if configured.

    The dataset holds one folder per document, exactly as written to data/indexes/<doc_id>/.
    """
    s = get_settings()
    if not s.hf_data_repo:
        return
    from huggingface_hub import snapshot_download

    s.index_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=s.hf_data_repo,
        repo_type="dataset",
        local_dir=str(s.index_dir),
        token=s.hf_token or None,
        allow_patterns=["*/info.json", "*/chunks.jsonl", "*/dense.faiss", "*/source.pdf", "*/answers/*.json"],
    )
    log.info("pulled prebuilt indexes from %s: %s", s.hf_data_repo, [i.filename for i in DocIndex.list_infos()])


def _bootstrap() -> None:
    """Warm models and auto-ingest any PDFs dropped in data/ (handy for local demos)."""
    from .index.models import warmup

    try:
        warmup()
    except Exception:
        log.exception("model warm-up failed")
    for pdf in sorted(get_settings().data_dir.glob("*.pdf")):
        try:
            pipeline.ingest(pdf)
            log.info("indexed %s", pdf.name)
        except Exception:
            log.exception("auto-ingest failed for %s", pdf)


@app.on_event("startup")
def _startup() -> None:
    # Pulled before serving, so the first page load already lists the demo manual.
    try:
        _pull_prebuilt_indexes()
    except Exception:
        log.exception("could not pull prebuilt indexes")
    threading.Thread(target=_bootstrap, daemon=True).start()
    threading.Thread(target=_cleanup_loop, daemon=True).start()


def _cleanup_loop() -> None:
    while True:
        time.sleep(600)
        try:
            _purge_expired()
        except Exception:
            log.exception("upload cleanup failed")


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
