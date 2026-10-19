"""Uploads on a public demo: private per visitor, capped, deletable, expiring."""

from __future__ import annotations

import time

import pymupdf
import pytest
from fastapi.testclient import TestClient

from specx import config, pipeline
from specx.index.store import DocIndex

from conftest import make_manual

ALICE = {"X-SpecX-Visitor": "alice-0000-1111"}
BOB = {"X-SpecX-Visitor": "bob-2222-3333-4444"}


@pytest.fixture
def client(env, manual_pdf):
    import specx.server as server

    pipeline.ingest(manual_pdf)            # the shared demo manual
    server._hits.clear()
    server._jobs.clear()
    server.app.router.on_startup.clear()
    with TestClient(server.app) as c:
        yield c


def _pdf_bytes(tmp_path, pages: int | None = None, name: str = "u.pdf") -> bytes:
    path = tmp_path / name
    if pages is None:
        make_manual(path)
    else:
        parts = ["caliper bolt", "wheel nut", "drain plug", "axle shaft nut", "hub bolt", "shield bolt",
                 "anchor bracket bolt", "flow bolt", "tie rod nut", "ball joint nut", "sensor bolt", "link nut"]
        doc = pymupdf.open()
        for i in range(pages):
            doc.new_page().insert_text(
                (60, 120), f"Tighten the {parts[i % len(parts)]} to {20 + 5 * i} Nm during step {i + 1}.", fontsize=10)
        doc.save(str(path))
    return path.read_bytes()


def _upload(client, headers, data: bytes, name: str = "my-manual.pdf"):
    r = client.post("/api/documents", headers=headers, files={"file": (name, data, "application/pdf")})
    if r.status_code == 200:  # background ingestion runs right after the response in TestClient
        job = client.get(f"/api/jobs/{r.json()['job_id']}", headers=headers).json()
        assert job["status"] == "done", job
        return r, job["doc"]
    return r, None


def _names(client, headers):
    return [d["filename"] for d in client.get("/api/documents", headers=headers).json()]


def test_upload_is_visible_only_to_its_uploader(client, tmp_path):
    data = _pdf_bytes(tmp_path, pages=3)
    _, doc = _upload(client, ALICE, data)
    assert doc["kind"] == "upload" and doc["expires_at"]
    assert _names(client, ALICE) == ["test-manual.pdf", "my-manual.pdf"]
    assert _names(client, BOB) == ["test-manual.pdf"]
    assert _names(client, {}) == ["test-manual.pdf"]


def test_other_visitors_get_404_everywhere(client, tmp_path):
    _, doc = _upload(client, ALICE, _pdf_bytes(tmp_path, pages=3))
    d = doc["doc_id"]
    assert client.post("/api/query", headers=BOB, json={"doc_id": d, "query": "torque"}).status_code == 404
    assert client.get(f"/api/documents/{d}/pages/1.png", params={"v": BOB["X-SpecX-Visitor"]}).status_code == 404
    assert client.get(f"/api/documents/{d}/source.pdf").status_code == 404
    assert client.delete(f"/api/documents/{d}", headers=BOB).status_code == 404
    # ...while the uploader can use it, including via the ?v= form used by <img> links
    assert client.post("/api/query", headers=ALICE, json={"doc_id": d, "query": "torque"}).status_code == 200
    assert client.get(f"/api/documents/{d}/pages/1.png", params={"v": ALICE["X-SpecX-Visitor"]}).status_code == 200


def test_other_visitors_ids_are_never_returned(client, tmp_path):
    _upload(client, ALICE, _pdf_bytes(tmp_path, pages=3))
    for d in client.get("/api/documents", headers=ALICE).json():
        assert "owners" not in d and "uploaded_at" not in d


def test_job_status_is_private_too(client, tmp_path):
    r = client.post("/api/documents", headers=ALICE, files={"file": ("a.pdf", _pdf_bytes(tmp_path, pages=2))})
    job_id = r.json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}", headers=BOB).status_code == 404


def test_page_cap_rejects_before_indexing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_PAGES", "5")
    config.get_settings.cache_clear()
    r, _ = _upload(client, ALICE, _pdf_bytes(tmp_path, pages=12))
    assert r.status_code == 413
    assert "up to 5 pages" in r.json()["detail"] and "yours: 12 pages" in r.json()["detail"]
    assert _names(client, ALICE) == ["test-manual.pdf"]          # nothing was indexed


def test_size_cap(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    config.get_settings.cache_clear()
    r, _ = _upload(client, ALICE, b"%PDF-1.4\n" + b"0" * (2 << 20))
    assert r.status_code == 413


def test_visitor_id_is_required_to_upload(client, tmp_path):
    r, _ = _upload(client, {}, _pdf_bytes(tmp_path, pages=2))
    assert r.status_code == 400


def test_per_visitor_upload_limit(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOADS_PER_VISITOR", "1")
    config.get_settings.cache_clear()
    _upload(client, ALICE, _pdf_bytes(tmp_path, pages=2, name="a.pdf"), "a.pdf")
    r, _ = _upload(client, ALICE, _pdf_bytes(tmp_path, pages=3, name="b.pdf"), "b.pdf")
    assert r.status_code == 429 and "delete one first" in r.json()["detail"]
    r, _ = _upload(client, BOB, _pdf_bytes(tmp_path, pages=3, name="b.pdf"), "b.pdf")   # Bob has his own quota
    assert r.status_code == 200


def test_same_file_from_two_visitors_is_indexed_once_and_shared_by_them(client, tmp_path):
    data = _pdf_bytes(tmp_path, pages=3)
    _, a = _upload(client, ALICE, data)
    _, b = _upload(client, BOB, data, name="renamed.pdf")
    assert a["doc_id"] == b["doc_id"]
    assert set(DocIndex.read_info(a["doc_id"]).owners) and len(DocIndex.read_info(a["doc_id"]).owners) == 2
    # Alice deleting it does not remove Bob's copy
    assert client.delete(f"/api/documents/{a['doc_id']}", headers=ALICE).status_code == 200
    assert "my-manual.pdf" not in _names(client, ALICE)
    assert "my-manual.pdf" in _names(client, BOB)
    # once the last owner deletes it, the files are gone
    client.delete(f"/api/documents/{a['doc_id']}", headers=BOB)
    assert not DocIndex.exists(a["doc_id"])


def test_reuploading_the_demo_manual_does_not_make_it_private(client, manual_pdf):
    _, doc = _upload(client, ALICE, manual_pdf.read_bytes(), "copy.pdf")
    assert doc["kind"] == "demo"
    assert "test-manual.pdf" in _names(client, BOB)


def test_demo_manual_cannot_be_deleted(client):
    demo = client.get("/api/documents", headers=ALICE).json()[0]["doc_id"]
    assert client.delete(f"/api/documents/{demo}", headers=ALICE).status_code == 403


def test_uploads_expire(client, tmp_path, monkeypatch):
    _, doc = _upload(client, ALICE, _pdf_bytes(tmp_path, pages=2))
    info = DocIndex.read_info(doc["doc_id"])
    info.uploaded_at = time.time() - 7 * 3600                    # older than the 6 h default
    DocIndex.write_info(info)
    assert _names(client, ALICE) == ["test-manual.pdf"]          # listing purges expired uploads
    assert not DocIndex.exists(doc["doc_id"])


def test_pdf_without_text_fails_with_a_clear_message(client, tmp_path):
    path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page().draw_rect((50, 50, 300, 300), fill=(0.5, 0.5, 0.5))   # an "image", no text
    doc.save(str(path))
    r = client.post("/api/documents", headers=ALICE, files={"file": ("scan.pdf", path.read_bytes())})
    job = client.get(f"/api/jobs/{r.json()['job_id']}", headers=ALICE).json()
    assert job["status"] == "error" and "scanned" in job["error"]
    assert _names(client, ALICE) == ["test-manual.pdf"]


def test_health_reports_upload_limits(client):
    h = client.get("/api/health").json()
    assert h["max_upload_pages"] == 150 and h["upload_ttl_hours"] == 6 and h["max_uploads_per_visitor"] == 2
