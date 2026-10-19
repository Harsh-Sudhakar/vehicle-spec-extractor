"""HTTP API tests (FastAPI TestClient) on the generated manual."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from specx import pipeline


@pytest.fixture
def client(env, manual_pdf):
    import specx.server as server

    pipeline.ingest(manual_pdf)
    server._hits.clear()
    # Skip model warm-up / prebuilt-index pulls in tests.
    server.app.router.on_startup.clear()
    with TestClient(server.app) as c:
        yield c


def test_health_reports_configuration(client):
    h = client.get("/api/health").json()
    assert h["llm"] is False and h["reranker"] is False
    assert h["default_doc"] == "test-manual.pdf"


def test_default_manual_is_listed_first(client, tmp_path):
    import shutil

    other = tmp_path / "another-manual.pdf"
    shutil.copy(pipeline.source_pdf(client.get("/api/documents").json()[0]["doc_id"]), other)
    with open(other, "ab") as f:      # change the bytes so it hashes to a different doc_id
        f.write(b"\n% copy\n")
    pipeline.ingest(other)
    names = [d["filename"] for d in client.get("/api/documents").json()]
    assert names[0] == "test-manual.pdf" and "another-manual.pdf" in names


def test_query_returns_specs_with_provenance(client):
    doc_id = client.get("/api/documents").json()[0]["doc_id"]
    r = client.post("/api/query", json={"doc_id": doc_id, "query": "brake caliper flow bolt torque"}).json()
    assert r["mode"] == "offline"
    assert any(s["value"] == "35" and s["page"] == 2 for s in r["specs"])
    assert r["chunks"] and "timings_ms" in r


def test_unknown_document_is_404(client):
    assert client.post("/api/query", json={"doc_id": "nope", "query": "torque"}).status_code == 404


def test_query_validation(client):
    doc_id = client.get("/api/documents").json()[0]["doc_id"]
    assert client.post("/api/query", json={"doc_id": doc_id, "query": "x"}).status_code == 422


def test_page_render_and_highlight(client):
    doc_id = client.get("/api/documents").json()[0]["doc_id"]
    r = client.get(f"/api/documents/{doc_id}/pages/2.png", params={"hl": "Brake caliper flow bolt"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    assert client.get(f"/api/documents/{doc_id}/pages/99.png").status_code == 404


def test_non_pdf_upload_is_rejected(client):
    r = client.post("/api/documents", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 400


def test_uploads_can_be_disabled(client, monkeypatch):
    from specx import config

    monkeypatch.setenv("ALLOW_UPLOAD", "false")
    config.get_settings.cache_clear()
    r = client.post("/api/documents", files={"file": ("m.pdf", b"%PDF-1.4")})
    assert r.status_code == 403
    assert client.get("/api/health").json()["allow_upload"] is False


def test_rate_limit(client, monkeypatch):
    from specx import config

    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    config.get_settings.cache_clear()
    doc_id = client.get("/api/documents").json()[0]["doc_id"]
    codes = [client.post("/api/query", json={"doc_id": doc_id, "query": "brake torque"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
