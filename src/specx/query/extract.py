"""LLM extraction over retrieved chunks, with a rule-based offline fallback.

The LLM path is the real system. The offline path exists so the pipeline still runs
(and can be compared against) without an API key: it reads unit columns from table
rows and "N Nm (M lb-ft)" patterns from prose, but it cannot resolve context the way
the LLM does.
"""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError
from rapidfuzz import fuzz

from ..config import get_settings
from ..ingest.chunk import SPEC_VALUE
from ..schema import RetrievedChunk, Spec
from .llm import LLMUnavailable, chat_json, llm_available
from .postprocess import dedupe, ground, normalise, normalise_spec_type, normalise_unit
from .prompts import SYSTEM, build_user_prompt

log = logging.getLogger(__name__)


def extract_specs(
    question: str, hits: list[RetrievedChunk], offline: bool = False
) -> tuple[list[Spec], list[dict], str | None, str, dict | None]:
    """Return (specs, rejected, model_name, mode, note)."""
    if not hits:
        return [], [], None, "offline" if offline else "llm", None
    if offline or not llm_available():
        return offline_extract(question, hits), [], None, "offline", None
    try:
        specs, rejected, note = _llm_extract(question, hits)
        return specs, rejected, get_settings().openrouter_model, "llm", note
    except LLMUnavailable:
        return offline_extract(question, hits), [], None, "offline", None


_REFERENCE = re.compile(r"\b(refer|see|label|placard|section|consult|indicated on|shown on|specified on)\b", re.I)
_NOTE_STOP = frozenset("the a an and or of to for in on at is are be by with this that these from as it its your their not".split())


def _note(data: dict, labels: dict[str, str], by_id: dict) -> dict | None:
    """A 'the manual says to look elsewhere' note. Kept only if it is a real reference and is
    grounded: most of its content words must come from the excerpt it cites."""
    text = re.sub(r"\s*\[?C\d+\]?", "", str(data.get("note") or "")).strip()
    label = str(data.get("note_chunk_id") or "").strip().strip("[]")
    chunk = by_id.get(labels.get(label, label))
    if not text or chunk is None or not _REFERENCE.search(text):
        return None
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _NOTE_STOP]
    hay = chunk.text.lower()
    if not words or sum(w in hay for w in words) / len(words) < 0.6:
        return None
    return {"text": text[:300], "page": chunk.page, "chunk_id": chunk.id}


def _llm_extract(question: str, hits: list[RetrievedChunk]) -> tuple[list[Spec], list[dict], dict | None]:
    user, labels = build_user_prompt(question, hits)
    data = chat_json(SYSTEM, user)
    raw = data.get("specs", []) if isinstance(data, dict) else []
    by_id = {h.chunk.id: h.chunk for h in hits}

    accepted: list[Spec] = []
    rejected: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("chunk_id") or "").strip().strip("[]")
        item["chunk_id"] = labels.get(label, label)
        item.pop("source", None)
        try:
            spec = normalise(Spec.model_validate(item))
        except ValidationError as e:
            rejected.append({"item": item, "reason": f"schema: {e.errors()[0]['msg']}"})
            continue
        ok, reason = ground(spec, by_id.get(spec.chunk_id or ""))
        if ok:
            accepted.append(spec)
        else:
            rejected.append({"item": item, "reason": reason})
    specs = dedupe(accepted)
    note = None if specs or not isinstance(data, dict) else _note(data, labels, by_id)
    return specs, rejected, note


# --------------------------------------------------------------------------- offline
_UNIT_HEADER = re.compile(r"^(?:N[·.\-]?m|Nm|lb[\s.\-]?ft|lb[\s.\-]?in|mm|in|L|ml|bar|psi|kPa|kg|V)$", re.I)
_PROSE_VALUE = re.compile(
    r"(?P<v>\d[\d.,]*)\s?(?P<u>Nm|N·m|mm|ml|L|bar|psi|kPa|kg)\b(?:\s*\((?P<av>\d[\d.,]*)\s?(?P<au>lb-ft|lb-in|in|qt|pt|psi|lb)\))?"
)


def _relevant(question: str, text: str) -> float:
    return fuzz.token_set_ratio(question.lower(), text.lower())


def offline_extract(question: str, hits: list[RetrievedChunk]) -> list[Spec]:
    specs: list[Spec] = []
    for h in hits:
        c = h.chunk
        if c.chunk_type == "table":
            for line in c.text.splitlines():
                cells = [p.split(": ", 1) for p in line.split(" | ") if ": " in p]
                if len(cells) < 2:
                    continue
                component = cells[0][1]
                if " › " in cells[0][0]:
                    component = f"{cells[0][0].split(' › ')[0]} {component}"
                if _relevant(question, f"{c.heading_path} {component}") < 45:
                    continue
                unit_cells = [(h_, v) for h_, v in cells[1:] if _UNIT_HEADER.match(h_.strip()) and re.search(r"\d", v)]
                stype = normalise_spec_type(c.heading_path.split(" › ")[-1] if c.heading_path else "Other")
                if unit_cells:
                    (u1, v1), *rest = unit_cells
                    specs.append(Spec(component=component, spec_type=stype, value=v1, unit=normalise_unit(u1),
                                      alt_value=rest[0][1] if rest else None,
                                      alt_unit=normalise_unit(rest[0][0]) if rest else None,
                                      page=c.page, section=c.section or None, chunk_id=c.id, source="table",
                                      source_quote=line[:160], confidence=0.6))
                else:
                    for header, v in cells[1:]:
                        m = SPEC_VALUE.search(v)
                        if m:
                            num = re.match(r"[\d.,]+", m.group(0))
                            unit = m.group(0)[num.end():].strip() if num else None
                            specs.append(Spec(component=component, spec_type=normalise_spec_type(header),
                                              value=num.group(0) if num else m.group(0), unit=normalise_unit(unit),
                                              page=c.page, section=c.section or None, chunk_id=c.id, source="table",
                                              source_quote=line[:160], confidence=0.5))
        else:
            for sentence in re.split(r"(?<=[.!?])\s+|\n", c.text):
                m = _PROSE_VALUE.search(sentence)
                if not m or _relevant(question, f"{c.heading_path} {sentence}") < 45:
                    continue
                specs.append(Spec(component=sentence[: m.start()].strip(" .:-")[-80:] or c.heading_path,
                                  spec_type="Torque" if m.group("u").lower() in ("nm", "n·m") else "Other",
                                  value=m.group("v"), unit=normalise_unit(m.group("u")),
                                  alt_value=m.group("av"), alt_unit=normalise_unit(m.group("au")),
                                  page=c.page, section=c.section or None, chunk_id=c.id, source="text",
                                  source_quote=sentence[:160], confidence=0.4))
    return dedupe(specs)
