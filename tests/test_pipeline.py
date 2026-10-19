"""Integration tests on a generated service manual: ingestion, retrieval, extraction.

The LLM is mocked, so these check the code around it: what reaches the model, and what
is accepted back.
"""

from __future__ import annotations

import pytest

from specx import pipeline
from specx.index.store import DocIndex
from specx.ingest.chunk import build_chunks
from specx.query import extract as extract_mod
from specx.query.prompts import build_user_prompt
from specx.query.retrieve import retrieve

from conftest import FOOTER, HEADER


# ------------------------------------------------------------------ ingestion
@pytest.fixture
def chunks(manual_pdf):
    chunks, stats = build_chunks(str(manual_pdf), "doc")
    return chunks, stats


def test_toc_page_is_skipped(chunks):
    chunks, stats = chunks
    assert stats.toc_pages == 1
    assert all(c.page != 1 for c in chunks)
    assert not any("........" in c.text for c in chunks)


def test_running_header_and_footer_are_removed(chunks):
    chunks, _ = chunks
    text = "\n".join(c.text for c in chunks)
    assert HEADER not in text and FOOTER not in text and "Page 3" not in text


def test_torque_table_becomes_rows_with_column_names(chunks):
    chunks, _ = chunks
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1
    rows = tables[0].text.splitlines()
    assert rows[0] == "Description: Brake caliper flow bolt | Nm: 35 | lb-ft: 26"
    assert "Nm: 250 | lb-ft: 184" in rows[2]
    assert tables[0].page == 2 and tables[0].has_spec


def test_section_label_and_heading_form_the_context_path(chunks):
    chunks, _ = chunks
    table = next(c for c in chunks if c.chunk_type == "table")
    assert table.heading_path == "SECTION 206-03: Front Disc Brake › Torque Specifications"
    assert table.section == "SECTION 206-03: Front Disc Brake"


def test_numbered_step_stays_with_its_torque(chunks):
    chunks, _ = chunks
    step = next(c for c in chunks if "stabilizer bar link nuts" in c.text)
    assert "Tighten the nuts to 80 Nm (59 lb-ft)" in step.text
    assert step.heading_path.endswith("Stabilizer Bar Installation")


def test_table_cells_do_not_leak_into_prose(chunks):
    chunks, _ = chunks
    prose = "\n".join(c.text for c in chunks if c.chunk_type == "text")
    assert "Brake caliper flow bolt" not in prose


# ------------------------------------------------------------------ index + retrieval
@pytest.fixture
def doc_id(env, manual_pdf):
    return pipeline.ingest(manual_pdf).doc_id


def test_ingest_persists_and_reloads(doc_id):
    info = DocIndex.load(doc_id).info
    assert info.filename == "test-manual.pdf"
    assert info.table_chunks == 1 and info.skipped_pages["toc_or_index"] == 1
    assert pipeline.source_pdf(doc_id).exists()


def test_reingest_same_file_is_a_cache_hit(doc_id, manual_pdf):
    assert pipeline.ingest(manual_pdf).doc_id == doc_id


def test_keyword_search_finds_exact_spec_code(doc_id):
    index = pipeline.load_index(doc_id)
    hits = retrieve(index, "WSS-M6C62-A", final_k=3)
    assert "WSS-M6C62-A" in hits[0].chunk.text
    assert hits[0].sparse_rank == 0


def test_retrieval_puts_the_torque_table_first(doc_id):
    hits = retrieve(pipeline.load_index(doc_id), "torque for brake caliper bolts", final_k=3)
    assert hits[0].chunk.chunk_type == "table"


def test_prompt_labels_excerpts_with_page_and_context(doc_id):
    hits = retrieve(pipeline.load_index(doc_id), "brake caliper bolt torque", final_k=2)
    user, labels = build_user_prompt("brake caliper bolt torque", hits)
    assert user.startswith("Question: brake caliper bolt torque")
    assert "[C1] (table rows · page 2 · SECTION 206-03" in user
    assert labels["C1"] == hits[0].chunk.id


# ------------------------------------------------------------------ extraction (mocked LLM)
def _mock_llm(monkeypatch, response: dict) -> None:
    monkeypatch.setattr(extract_mod, "llm_available", lambda: True)
    monkeypatch.setattr(extract_mod, "chat_json", lambda system, user: response)


def test_llm_value_is_grounded_and_enriched(doc_id, monkeypatch):
    _mock_llm(monkeypatch, {"specs": [{
        "component": "Brake caliper flow bolt", "spec_type": "torque", "value": "35 Nm", "unit": "Nm",
        "alt_value": "26", "alt_unit": "lb-ft", "chunk_id": "C1",
        "source_quote": "Brake caliper flow bolt | Nm: 35 | lb-ft: 26"}]})
    r = pipeline.query(doc_id, "torque for brake caliper bolts", use_cache=False)
    assert r.mode == "llm" and not r.rejected
    s = r.specs[0]
    assert (s.value, s.unit, s.spec_type) == ("35", "Nm", "Torque")     # unit split out, type normalised
    assert (s.page, s.source, s.confidence) == (2, "table", 1.0)        # provenance filled from the chunk
    assert s.section == "SECTION 206-03: Front Disc Brake"


def test_invented_value_is_rejected(doc_id, monkeypatch):
    _mock_llm(monkeypatch, {"specs": [
        {"component": "Brake caliper flow bolt", "spec_type": "Torque", "value": "45", "unit": "Nm", "chunk_id": "C1"},
        {"component": "Brake caliper flow bolt", "spec_type": "Torque", "value": "35", "unit": "Nm", "chunk_id": "C1"},
    ]})
    r = pipeline.query(doc_id, "torque for brake caliper bolts", use_cache=False)
    assert [s.value for s in r.specs] == ["35"]
    assert len(r.rejected) == 1 and "not found" in r.rejected[0]["reason"]


def test_citation_to_missing_excerpt_is_rejected(doc_id, monkeypatch):
    _mock_llm(monkeypatch, {"specs": [
        {"component": "Brake caliper flow bolt", "spec_type": "Torque", "value": "35", "unit": "Nm", "chunk_id": "C99"}]})
    r = pipeline.query(doc_id, "torque for brake caliper bolts", use_cache=False)
    assert r.specs == [] and r.rejected[0]["reason"] == "cited excerpt does not exist"


def test_duplicate_specs_are_merged(doc_id, monkeypatch):
    item = {"component": "Brake caliper flow bolt", "spec_type": "Torque", "value": "35", "unit": "Nm", "chunk_id": "C1"}
    _mock_llm(monkeypatch, {"specs": [item, dict(item)]})
    assert len(pipeline.query(doc_id, "caliper flow bolt torque", use_cache=False).specs) == 1


def test_grounded_reference_note_is_kept(doc_id, monkeypatch):
    hits = retrieve(pipeline.load_index(doc_id), "tire pressure", final_k=8)
    label = next(f"C{i}" for i, h in enumerate(hits, 1) if "Vehicle Certification" in h.chunk.text)
    _mock_llm(monkeypatch, {"specs": [], "note_chunk_id": label,
                            "note": "Set the tire pressures to the values on the Vehicle Certification label."})
    r = pipeline.query(doc_id, "tire pressure", use_cache=False)
    assert r.specs == [] and r.note and r.note["page"] == 5


def test_ungrounded_note_is_dropped(doc_id, monkeypatch):
    _mock_llm(monkeypatch, {"specs": [], "note_chunk_id": "C1",
                            "note": "Refer to the dealer website for the recommended inflation values."})
    assert pipeline.query(doc_id, "tire pressure", use_cache=False).note is None


def test_answers_are_cached_per_normalised_question(doc_id, monkeypatch):
    calls = []
    _mock_llm(monkeypatch, {"specs": []})
    monkeypatch.setattr(extract_mod, "chat_json", lambda s, u: calls.append(1) or {"specs": []})
    pipeline.query(doc_id, "Brake  caliper torque")
    r = pipeline.query(doc_id, "brake caliper TORQUE")
    assert len(calls) == 1 and r.cached


def test_offline_mode_without_key_still_extracts_table_values(doc_id):
    r = pipeline.query(doc_id, "brake caliper flow bolt torque", use_cache=False)
    assert r.mode == "offline"
    assert any(s.value == "35" and s.unit == "Nm" and s.alt_value == "26" for s in r.specs)
