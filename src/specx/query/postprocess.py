"""Validation and clean-up of extracted specs.

* **Grounding** - a value must literally appear in the excerpt it cites, otherwise the
  entry is rejected. This is the main guard against hallucinated numbers.
* **Normalisation** - units and spec types are mapped to canonical spellings.
* **Dedupe** - the same spec often appears in a spec table and in a procedure step.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from ..schema import Chunk, Spec

_UNIT_MAP = {
    "nm": "Nm", "n·m": "Nm", "n.m": "Nm", "n-m": "Nm", "newton meters": "Nm", "newton metres": "Nm",
    "lb-ft": "lb-ft", "lb.ft": "lb-ft", "lbft": "lb-ft", "ft-lb": "lb-ft", "ft.lb": "lb-ft", "ft lb": "lb-ft", "lb ft": "lb-ft",
    "lb-in": "lb-in", "lb.in": "lb-in", "in-lb": "lb-in", "in.lb": "lb-in", "lb in": "lb-in",
    "kgf·m": "kgf·m", "kgf.m": "kgf·m", "kg-m": "kgf·m", "kgm": "kgf·m",
    "l": "L", "litre": "L", "litres": "L", "liter": "L", "liters": "L", "ltr": "L",
    "ml": "ml", "millilitre": "ml", "milliliter": "ml",
    "gms": "g", "gm": "g", "grams": "g", "gram": "g",
    "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
    "millimetre": "mm", "millimeter": "mm", "millimetres": "mm", "millimeters": "mm",
    "inch": "in", "inches": "in", '"': "in",
    "pt": "pt", "pint": "pt", "pints": "pt", "qt": "qt", "quart": "qt", "quarts": "qt",
    "gallons": "gal", "gallon": "gal",
    "kpa": "kPa", "mpa": "MPa", "psi": "psi", "bar": "bar",
    "degrees": "°", "deg": "°", "degree": "°",
    "rpm": "rpm", "r/min": "rpm",
    "ohm": "Ω", "ohms": "Ω", "v": "V", "volts": "V", "a": "A", "amps": "A",
}

_TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Torque", ("torque", "tighten")),
    ("Capacity", ("capacity", "fill", "volume", "quantity")),
    ("Pressure", ("pressure", "inflation")),
    ("Clearance", ("clearance", "gap", "play", "runout", "end float", "backlash")),
    ("Thickness", ("thickness", "wear")),
    ("Part Number", ("part number", "part no", "part #", "filter", "plug type", "bulb")),
    ("Fluid Specification", ("fluid spec", "specification", "material", "grade", "viscosity", "oil type", "fluid type")),
    ("Weight", ("weight", "mass", "load")),
    ("Dimension", ("dimension", "length", "width", "height", "diameter", "size", "wheelbase", "track")),
    ("Speed", ("speed", "rpm")),
    ("Electrical", ("voltage", "current", "resistance", "wattage", "power")),
    ("Temperature", ("temperature",)),
    ("Interval", ("interval", "every", "service")),
]
_CANONICAL = {name.lower(): name for name, _ in _TYPE_RULES} | {"other": "Other"}


def normalise_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.strip()
    return _UNIT_MAP.get(u.lower(), u)


def normalise_spec_type(spec_type: str) -> str:
    t = spec_type.strip()
    if t.lower() in _CANONICAL:
        return _CANONICAL[t.lower()]
    low = t.lower()
    for name, keys in _TYPE_RULES:
        if any(k in low for k in keys):
            return name
    return t[:1].upper() + t[1:]


def _squash(text: str) -> str:
    """Lower-case and drop whitespace / thousands separators / brackets for literal matching."""
    text = text.lower().replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"[\s()\[\]]", "", text)


def value_grounded(value: str, chunk_text: str) -> bool:
    v = _squash(value)
    if not v:
        return False
    if v in _squash(chunk_text):
        return True
    # Ranges/tolerances may be re-spelled ("0.30±0.03" vs "0.30 (±0.03)"): all numbers must appear.
    nums = re.findall(r"\d+(?:\.\d+)?", value)
    hay = _squash(chunk_text)
    return bool(nums) and all(n in hay for n in nums)


def ground(spec: Spec, chunk: Chunk | None) -> tuple[bool, str]:
    """Return (accepted, reason). Also fills page/section/source and a confidence score."""
    if chunk is None:
        return False, "cited excerpt does not exist"
    if not value_grounded(spec.value, chunk.text):
        return False, f"value '{spec.value}' not found in cited excerpt"
    hay = f"{chunk.heading_path} {chunk.text}".lower()
    comp_ok = fuzz.partial_ratio(spec.component.lower(), hay) >= 70 or fuzz.token_set_ratio(spec.component.lower(), hay) >= 70
    quote_ok = bool(spec.source_quote) and fuzz.partial_ratio(_squash(spec.source_quote), _squash(chunk.text)) >= 90
    spec.confidence = round(0.5 + 0.3 * comp_ok + 0.2 * quote_ok, 2)
    spec.page = chunk.page
    spec.section = chunk.section or None
    spec.source = "ocr" if chunk.source == "ocr" else chunk.chunk_type
    spec.chunk_id = chunk.id
    return True, "ok"


def normalise(spec: Spec) -> Spec:
    spec.unit = normalise_unit(spec.unit)
    spec.alt_unit = normalise_unit(spec.alt_unit)
    spec.spec_type = normalise_spec_type(spec.spec_type)
    spec.component = re.sub(r"\s+", " ", spec.component).strip(" .:-")
    spec.value = re.sub(r"\s+", " ", spec.value).strip()
    spec.value, spec.unit = _split_unit(spec.value, spec.unit)
    if spec.alt_value:
        spec.alt_value, spec.alt_unit = _split_unit(spec.alt_value.strip(), spec.alt_unit)
    return spec


_NUMERIC_WITH_UNIT = re.compile(r"([\d.,]+(?:\s*(?:±|to|-|–)\s*[\d.,]+)?)\s*([A-Za-z°Ω·/\-]{1,8})")


def _split_unit(value: str, unit: str | None) -> tuple[str, str | None]:
    """'1,420 ml' + 'ml' -> ('1,420', 'ml'); '35 Nm' + None -> ('35', 'Nm'). Codes are left alone."""
    m = _NUMERIC_WITH_UNIT.fullmatch(value)
    if not m:
        return value, unit
    found = normalise_unit(m.group(2))
    if unit is None or (found or "").lower() == (unit or "").lower():
        return m.group(1).strip(), unit or found
    return value, unit


def dedupe(specs: list[Spec]) -> list[Spec]:
    best: dict[tuple, Spec] = {}
    for s in specs:
        key = (
            re.sub(r"[^a-z0-9]", "", s.component.lower()),
            s.spec_type,
            _squash(s.value),
            (s.unit or "").lower(),
            (s.variant or "").lower(),
            (s.condition or "").lower(),
        )
        cur = best.get(key)
        rank = {"table": 2, "text": 1, "ocr": 0}
        if cur is None or (rank[s.source], s.confidence or 0) > (rank[cur.source], cur.confidence or 0):
            best[key] = s
    return list(best.values())
