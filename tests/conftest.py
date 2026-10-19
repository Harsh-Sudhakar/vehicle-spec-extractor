"""Shared fixtures: a synthetic service manual, isolated settings, offline models.

No copyrighted manual, model download or network call is needed: the PDF is generated
here, the embedder is replaced by a deterministic bag-of-words hash, and the LLM is mocked
in the tests that need it.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pymupdf
import pytest

HEADER = "2014 Truck Workshop Manual"
FOOTER = "Printed 2014-03-01"


def _grid(page: pymupdf.Page, x0: float, y0: float, widths: list[float], rows: list[list[str]], h: float = 18) -> None:
    """Draw a ruled table (find_tables detects it from the lines) and fill the cells."""
    x_edges = [x0]
    for w in widths:
        x_edges.append(x_edges[-1] + w)
    y_edges = [y0 + i * h for i in range(len(rows) + 1)]
    for x in x_edges:
        page.draw_line((x, y_edges[0]), (x, y_edges[-1]))
    for y in y_edges:
        page.draw_line((x_edges[0], y), (x_edges[-1], y))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((x_edges[c] + 4, y_edges[r] + 13), cell, fontsize=9,
                             fontname="hebo" if r == 0 else "helv")


def _chrome(page: pymupdf.Page, n: int) -> None:
    page.insert_text((60, 30), HEADER, fontsize=8)
    page.insert_text((60, 825), f"{FOOTER}   Page {n}", fontsize=8)


def make_manual(path) -> None:
    doc = pymupdf.open()

    # p1: table of contents (should be detected and skipped)
    p = doc.new_page()
    _chrome(p, 1)
    for i, t in enumerate(["Front Disc Brake", "Rear Disc Brake", "Suspension", "Wheels and Tires",
                           "Driveshaft", "Hydraulic Brake Actuation", "Parking Brake", "ABS", "Steering"]):
        p.insert_text((60, 90 + 20 * i), f"{t} ........................ {3 + i}", fontsize=10)

    # p2: section label, heading, ruled torque table
    p = doc.new_page()
    _chrome(p, 2)
    p.insert_text((60, 80), "SECTION 206-03: Front Disc Brake", fontsize=10)
    p.insert_text((60, 110), "Torque Specifications", fontsize=14, fontname="hebo")
    _grid(p, 60, 130, [230, 60, 60], [
        ["Description", "Nm", "lb-ft"],
        ["Brake caliper flow bolt", "35", "26"],
        ["Brake caliper guide pin bolts", "37", "27"],
        ["Brake caliper anchor plate bolts", "250", "184"],
    ])

    # p3: procedure with a numbered step and its indented torque bullet
    p = doc.new_page()
    _chrome(p, 3)
    p.insert_text((60, 80), "SECTION 204-01A: Front Suspension", fontsize=10)
    p.insert_text((60, 110), "Stabilizer Bar Installation", fontsize=14, fontname="hebo")
    p.insert_text((60, 140), "11. Install the 2 new stabilizer bar link nuts.", fontsize=10)
    p.insert_text((80, 156), "• Tighten the nuts to 80 Nm (59 lb-ft).", fontsize=10)
    p.insert_text((60, 180), "12. Install the wheel and tire assembly.", fontsize=10)

    # p4: prose with a fluid specification code
    p = doc.new_page()
    _chrome(p, 4)
    p.insert_text((60, 110), "Brake Fluid", fontsize=14, fontname="hebo")
    p.insert_text((60, 140), "Fill the reservoir with DOT 3 brake fluid meeting specification WSS-M6C62-A.", fontsize=10)

    # p5: more prose so header/footer repeat on enough pages to be learned as boilerplate
    p = doc.new_page()
    _chrome(p, 5)
    p.insert_text((60, 110), "Wheels and Tires", fontsize=14, fontname="hebo")
    p.insert_text((60, 140), "Set the tire pressures to the values on the Vehicle Certification label.", fontsize=10)

    doc.save(str(path))


@pytest.fixture
def manual_pdf(tmp_path):
    path = tmp_path / "test-manual.pdf"
    make_manual(path)
    return path


def fake_vec(text: str, dim: int = 256) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated settings (fresh data dir, no reranker, no LLM key) + deterministic embeddings."""
    from specx import config, pipeline
    from specx.index import store

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("USE_RERANKER", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HF_DATA_REPO", "")
    monkeypatch.setenv("DEFAULT_DOC", "test-manual.pdf")
    config.get_settings.cache_clear()
    monkeypatch.setattr(store, "embed_passages", lambda texts: np.vstack([fake_vec(t) for t in texts]))
    monkeypatch.setattr(store, "embed_query", lambda q: fake_vec(q)[None, :])
    pipeline._cache.clear()
    pipeline._answers.clear()
    yield config.get_settings()
    config.get_settings.cache_clear()
    pipeline._cache.clear()
    pipeline._answers.clear()
