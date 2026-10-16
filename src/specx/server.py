"""FastAPI app: JSON API + the static web UI.

Run locally:  uvicorn specx.server:app --reload   (from the repo root, with PYTHONPATH=src)
"""

from __future__ import annotations

import logging
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
from .ingest.ocr import ocr_available
from .query.llm import llm_available

log = logging.getLogger("specx")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parents[2] / "web" / "static"

app = FastAPI(title="SpecX", version=__version__)

# --------------------------------------------------------------------------- jobs
_jobs: dict[str, dict] = {}
_ingest_lock = threading.Lock()  # one ingestion at a time keeps memory predictable


def _run_ingest(job_id: str, path: Path, filename: str) -> None:
    job = _jobs[job_id]

    def progress(stage: str, frac: float) -> None:
        job.update(stage=stage, progress=round(frac, 3))

    job.update(status="queued", stage="queued")
    with _ingest_lock:
        job.update(status="running", started=time.time())
        try:
            info = pipeline.ingest(path, filename=filename, progress=progress)
            job.update(status="done", stage="done", progress=1.0, doc=info.model_dump())
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
        "reranker": s.use_reranker,
        "allow_upload": s.allow_upload,
        "default_doc": s.default_doc,
        "default_query": s.default_query,
    }


@app.get("/api/documents")
def documents():
    """Indexed manuals, with the configured default manual first, then by filename."""
    default = get_settings().default_doc
    infos = sorted(
        DocIndex.list_infos(),
        key=lambda i: (default not in (i.filename, i.doc_id), i.filename.lower()),
    )
    return [i.model_dump() for i in infos]


@app.post("/api/documents")
async def upload(request: Request, background: BackgroundTasks, file: UploadFile = File(...)):
    _rate_limit(request, cost=3)
    s = get_settings()
    if not s.allow_upload:
        raise HTTPException(403, "Uploads are disabled on this demo.")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(suffix=".pdf", dir=s.upload_dir)[1])
    size = 0
    with open(tmp, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > s.max_upload_mb * (1 << 20):
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"File is larger than {s.max_upload_mb} MB.")
            out.write(chunk)
    try:
        with pymupdf.open(tmp) as d:
            if d.page_count == 0:
                raise ValueError
    except Exception:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "That file could not be opened as a PDF.")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"id": job_id, "filename": file.filename, "status": "queued", "stage": "queued", "progress": 0.0}
    background.add_task(_run_ingest, job_id, tmp, file.filename)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Unknown job")
    return _jobs[job_id]


@app.post("/api/query")
def run_query(body: QueryIn, request: Request):
    _rate_limit(request)
    if not DocIndex.exists(body.doc_id):
        raise HTTPException(404, "Document not found. Upload it again.")
    try:
        result = pipeline.query(body.doc_id, body.query.strip(), top_k=body.top_k, offline=body.offline)
    except Exception as e:
        log.exception("query failed")
        raise HTTPException(502, f"Extraction failed: {e}")
    return JSONResponse(result.model_dump())


@app.get("/api/documents/{doc_id}/pages/{page}.png")
def page_image(doc_id: str, page: int, hl: list[str] | None = Query(None), dpi: int = 110):
    """Render one source page, highlighting the given strings (value, quote) if found."""
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
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/documents/{doc_id}/source.pdf")
def source(doc_id: str):
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


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
