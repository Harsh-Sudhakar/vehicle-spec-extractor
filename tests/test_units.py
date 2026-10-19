"""Fast unit tests: no models, no network, no PDFs needed."""

from specx.ingest.chunk import has_spec, serialise_table
from specx.ingest.clean import join_lines
from specx.ingest.parse import Table, _clean_cell
from specx.index.store import tokenize
from specx.query.postprocess import dedupe, ground, normalise, normalise_spec_type, value_grounded
from specx.query.retrieve import rrf
from specx.schema import Chunk, Spec


def chunk(text: str, **kw) -> Chunk:
    return Chunk(id="d-r1", doc_id="d", page=636, page_end=636, chunk_type=kw.get("t", "table"), text=text,
                 heading_path=kw.get("path", "SECTION 206-03: Front Disc Brake › Torque Specifications"),
                 section="SECTION 206-03: Front Disc Brake")


# ------------------------------------------------------------------ ingestion
def test_torque_table_keeps_column_meaning():
    t = Table(rows=[["Description", "Nm", "lb-ft", "lb-in"],
                    ["Brake caliper flow bolt", "35", "26", "—"],
                    ["Brake disc shield bolts", "18", "—", "159"]], bbox=(0, 0, 1, 1))
    _, rows = serialise_table(t)
    assert rows[0] == "Description: Brake caliper flow bolt | Nm: 35 | lb-ft: 26 | lb-in: —"
    assert "lb-in: 159" in rows[1]


def test_category_rows_and_stacked_subtables():
    t = Table(rows=[["Item", "Specification", ""],
                    ["Brake Disc", "", ""],
                    ["Brake disc minimum thickness", "32 mm (1.259 in)", ""],
                    ["", "", ""],
                    ["Variant", "Item", "Capacity"],
                    ["All", "Fuel tank", "45 (9.89)"]], bbox=(0, 0, 1, 1))
    _, rows = serialise_table(t)
    # the category row is carried down onto the rows beneath it
    assert rows[0] == "Brake Disc › Item: Brake disc minimum thickness | Specification: 32 mm (1.259 in)"
    # after a blank row, a header-like row starts a new sub-table with its own column names
    assert rows[1].endswith("Variant: All | Item: Fuel tank | Capacity: 45 (9.89)")


def test_cell_cleaning_removes_symbol_bullets_and_hyphenation():
    assert _clean_cell("z Loose wheel nut") == "• Loose wheel nut"
    assert _clean_cell("Transmis-\nsion fluid") == "Transmission fluid"


def test_join_lines_repairs_hyphenation():
    assert join_lines(["Install the anti-", "lock sensor", "bolts."]) == "Install the antilock sensor bolts."


def test_has_spec_detects_values_and_codes():
    assert has_spec("Tighten the nuts to 55 Nm (41 lb-ft).")
    assert has_spec("Brake fluid WSS-M6C57-A2")
    assert has_spec("Fill to 1,420 ml")
    assert not has_spec("Remove the wheel and tire assembly.")


def test_tokenizer_keeps_compound_codes():
    toks = tokenize("Brake fluid WSS-M6C57-A2 and 175/65 R14")
    assert "wss-m6c57-a2" in toks and "m6c57" in toks and "175/65" in toks


# ------------------------------------------------------------------ retrieval
def test_rrf_rewards_agreement():
    scores = rrf([[1, 2, 3], [3, 1, 4]])
    assert max(scores, key=scores.get) == 1
    assert scores[3] > scores[2]


# ------------------------------------------------------------------ post-processing
def test_grounding_accepts_literal_value_and_rejects_invented_one():
    c = chunk("Description: Brake caliper flow bolt | Nm: 35 | lb-ft: 26 | lb-in: —")
    good = Spec(component="Brake caliper flow bolt", spec_type="Torque", value="35", unit="Nm",
                source_quote="Brake caliper flow bolt | Nm: 35")
    bad = Spec(component="Brake caliper flow bolt", spec_type="Torque", value="40", unit="Nm")
    assert ground(good, c)[0] and good.page == 636 and good.confidence == 1.0
    assert not ground(bad, c)[0]


def test_value_grounding_handles_separators_and_tolerances():
    assert value_grounded("1420", "Fill Capacity: 1,420 ml (3 pt)")
    assert value_grounded("0.30 ± 0.03", "Valve clearance exhaust*: 0.30(±0.03) mm")
    assert not value_grounded("0.40", "Valve clearance exhaust*: 0.30(±0.03) mm")


def test_normalise_splits_units_and_canonicalises():
    s = normalise(Spec(component="Brake fluid", spec_type="fill capacity", value="1,420 ml", unit="ml",
                       alt_value="3 pt", alt_unit="pt"))
    assert (s.value, s.unit, s.alt_value, s.spec_type) == ("1,420", "ml", "3", "Capacity")
    s = normalise(Spec(component="x", spec_type="torque", value="35 N·m", unit=None))
    assert (s.value, s.unit) == ("35", "Nm")
    s = normalise(Spec(component="Oil filter", spec_type="Part Number", value="EFL 600", unit=None))
    assert (s.value, s.unit) == ("EFL 600", None)


def test_spec_type_rules():
    assert normalise_spec_type("Tightening torque") == "Torque"
    assert normalise_spec_type("Inflation pressure") == "Pressure"
    assert normalise_spec_type("Minimum thickness") == "Thickness"


def test_dedupe_prefers_table_source():
    a = Spec(component="Lower ball joint nut", spec_type="Torque", value="175", unit="Nm", source="text", confidence=1.0)
    b = Spec(component="Lower ball joint nut", spec_type="Torque", value="175", unit="Nm", source="table", confidence=0.8)
    out = dedupe([a, b])
    assert len(out) == 1 and out[0].source == "table"


# ------------------------------------------------------------------ hosting
def test_cpu_quota_from_cgroup_files(tmp_path):
    from specx.index.models import cpu_quota

    (tmp_path / "cpu.max").write_text("10000 100000\n")          # Render free: 0.1 CPU (cgroup v2)
    assert cpu_quota(tmp_path) == 0.1
    (tmp_path / "cpu.max").write_text("max 100000\n")            # no limit
    assert cpu_quota(tmp_path) is None
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("50000")        # cgroup v1: 0.5 CPU
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert cpu_quota(v1) == 0.5
    assert cpu_quota(tmp_path / "missing") is None               # e.g. macOS
