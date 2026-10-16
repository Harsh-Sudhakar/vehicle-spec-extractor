"""Turn a PDF into retrievable chunks.

Two chunk types:

* **table** - 1-3 table rows serialised as ``Header: value | Header: value`` with the
  table title and heading path attached, so a row like ``37 | 27 | —`` keeps the
  meaning of every number.
* **text** - heading-bounded prose, packed to ~200 words. Numbered procedure steps are
  never split, so "Install the stabilizer bar link nuts" stays with "Tighten to 80 Nm".
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pymupdf

from ..schema import Chunk
from .clean import DocCleaner, join_lines
from .parse import Line, PageContent, Table, parse_page
from .structure import HeadingStyle, HeadingTracker

TARGET_WORDS = 200
MAX_WORDS = 320
TABLE_ROWS_PER_CHUNK = 3

_UNITS = (
    r"N[·.\-]?m|Nm|kN|lb[\s.\-]?ft|ft[\s.\-]?lb|lb[\s.\-]?in|in[\s.\-]?lb|kgf?[\s.\-]?m|kg|g|gms?|"
    r"mm|cm|m|in|inch(?:es)?|ft|L|l|ml|mL|litres?|liters?|qt|pt|gal|oz|cc|"
    r"bar|psi|kPa|MPa|Pa|V|A|mA|W|kW|Ω|ohms?|°C|°F|rpm|km/h|mph|Hz|%|deg|°"
)
SPEC_VALUE = re.compile(rf"\d[\d.,]*\s?(?:±\s?\d[\d.,]*\s?)?(?:{_UNITS})\b", re.I)
SPEC_CODE = re.compile(r"\b[A-Z]{2,4}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)+\b")  # e.g. WSS-M6C57-A2
_STEP_START = re.compile(r"^(?:\d{1,2}[.)]\s|step\s+\d+\b)", re.I)
_BULLET = re.compile(r"^[•\-–▪■●○]\s*")


def has_spec(text: str) -> bool:
    return bool(SPEC_VALUE.search(text) or SPEC_CODE.search(text))


# --------------------------------------------------------------------------- tables
def _is_category_row(row: list[str]) -> bool:
    filled = [c for c in row if c]
    return len(filled) == 1 and bool(row[0]) and len(row) > 1


def _value_like(cell: str) -> bool:
    """True if a cell is essentially a measurement ("32 mm (1.259 in)", "35", "1,420 ml")."""
    rest = SPEC_VALUE.sub("", cell)
    rest = re.sub(r"[\d\s.,()±/\-–—:@x×]+", "", rest)
    return len(rest) < 3


def _header_like(row: list[str]) -> bool:
    filled = [c for c in row if c]
    return len(filled) >= 2 and not any(_value_like(c) for c in filled) and all(len(c) <= 60 for c in filled)


def serialise_table(table: Table) -> tuple[str | None, list[str]]:
    """Return (title, serialised data rows).

    Handles the shapes seen in real manuals: spanning title rows, category rows that
    apply to the rows beneath them, and stacked sub-tables that restart with their own
    header row after a blank or category row.
    """
    rows = [list(r) for r in table.rows]
    title = table.title
    # Leading spanning rows ("TECHNICAL SPECIFICATIONS Vehicle fluids") are titles, not data.
    while rows and _is_category_row(rows[0]) and len(rows) > 2:
        title = f"{title} › {rows[0][0]}" if title else rows[0][0]
        rows.pop(0)
    header = [h or f"col{i+1}" for i, h in enumerate(rows[0])]
    out: list[str] = []
    category = ""
    boundary = False  # previous row was blank or a category row -> a new header may follow
    for row in rows[1:]:
        if not any(row):
            boundary = True
            continue
        if _is_category_row(row):
            if re.fullmatch(r"(?i)(warnings?|cautions?|notes?|important)[:!]?", row[0].strip()):
                boundary = True
                continue
            if len(row[0]) <= 70:
                category = row[0]
            else:  # a long note inside the table: keep it as its own line
                out.append(row[0])
            boundary = True
            continue
        if boundary and _header_like(row):
            header = [h or f"col{i+1}" for i, h in enumerate(row)]
            boundary = False
            continue
        boundary = False
        if [c for c in row if c] == [h for h in header if not h.startswith("col")]:
            continue  # repeated header row
        cells = []
        for i, cell in enumerate(row):
            if not cell:
                continue
            name = header[i] if i < len(header) and header[i] else f"col{i+1}"
            cells.append(f"{name}: {cell}")
        line = " | ".join(cells)
        if category:
            line = f"{category} › {line}"
        out.append(line)
    return title, out


# --------------------------------------------------------------------------- prose
@dataclass
class _Unit:
    text: str
    page: int
    is_step: bool = False
    x0: float = 0.0


@dataclass
class _Section:
    path: str
    section: str
    units: list[_Unit] = field(default_factory=list)


def _lines_to_units(lines: list[Line], page: int) -> list[_Unit]:
    """Group lines into paragraphs (by block), attaching indented bullets to the step above."""
    units: list[_Unit] = []
    buf: list[Line] = []

    def flush():
        if not buf:
            return
        text = join_lines([_BULLET.sub("• ", l.text) for l in buf])
        first = buf[0]
        is_step = bool(_STEP_START.match(first.text))
        prev = units[-1] if units else None
        # A bullet / indented continuation belongs to the numbered step before it.
        if prev and prev.is_step and not is_step and (first.text.startswith("•") or first.x0 > prev.x0 + 8):
            if len(prev.text.split()) + len(text.split()) <= MAX_WORDS:
                prev.text = f"{prev.text}\n{text}"
                buf.clear()
                return
        units.append(_Unit(text=text, page=page, is_step=is_step, x0=first.x0))
        buf.clear()

    for ln in lines:
        if buf and (ln.block != buf[-1].block or _STEP_START.match(ln.text)):
            flush()
        buf.append(ln)
    flush()
    return units


def _pack(section: _Section, doc_id: str, counter: list[int]) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[_Unit] = []
    words = 0

    def emit():
        nonlocal cur, words
        if not cur:
            return
        text = "\n".join(u.text for u in cur).strip()
        if len(text) >= 25:
            counter[0] += 1
            chunks.append(
                Chunk(
                    id=f"{doc_id}-t{counter[0]:05d}",
                    doc_id=doc_id,
                    page=cur[0].page,
                    page_end=cur[-1].page,
                    chunk_type="text",
                    heading_path=section.path,
                    section=section.section,
                    text=text,
                    has_spec=has_spec(text),
                )
            )
        # one-unit overlap when the last unit is short, to keep local context
        last = cur[-1]
        cur = [last] if len(last.text.split()) <= 50 and len(cur) > 1 else []
        words = sum(len(u.text.split()) for u in cur)

    for unit in section.units:
        n = len(unit.text.split())
        if n > MAX_WORDS:  # very long paragraph: split by sentences
            emit()
            sentences = re.split(r"(?<=[.!?])\s+", unit.text)
            piece: list[str] = []
            for s in sentences:
                piece.append(s)
                if len(" ".join(piece).split()) >= TARGET_WORDS:
                    cur, words = [_Unit(" ".join(piece), unit.page)], TARGET_WORDS
                    emit()
                    cur, piece = [], []
            if piece:
                cur, words = [_Unit(" ".join(piece), unit.page)], len(" ".join(piece).split())
            continue
        if words + n > TARGET_WORDS and cur:
            emit()
        cur.append(unit)
        words += n
    # Skip a remainder that is only the overlap unit already ending the previous chunk.
    if cur and not (chunks and len(cur) == 1 and chunks[-1].text.endswith(cur[0].text)):
        emit()
    return chunks


# --------------------------------------------------------------------------- driver
@dataclass
class BuildStats:
    pages: int = 0
    toc_pages: int = 0
    image_pages: int = 0
    blank_pages: int = 0
    ocr_pages: int = 0
    has_outline: bool = False


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def build_chunks(
    pdf_path: str,
    doc_id: str,
    enable_ocr: bool = False,
    progress: Callable[[str, float], None] | None = None,
) -> tuple[list[Chunk], BuildStats]:
    progress = progress or (lambda stage, frac: None)
    doc = pymupdf.open(pdf_path)
    stats = BuildStats(pages=len(doc))

    # Pass 1: parse every page (tables first, prose outside tables).
    pages: list[PageContent] = []
    for i, page in enumerate(doc):
        pages.append(parse_page(page))
        if i % 10 == 0:
            progress("parse", (i + 1) / len(doc))

    # Pass 2: learn document-level boilerplate and heading styles.
    progress("clean", 0.0)
    cleaner = DocCleaner(pages)
    tracker = HeadingTracker.from_toc(doc.get_toc())
    stats.has_outline = tracker.uses_outline

    ocr_texts: dict[int, str] = {}
    if enable_ocr:
        from .ocr import ocr_page

        for p in pages:
            if cleaner.is_image_only(p):
                text = ocr_page(doc[p.number - 1])
                if text:
                    ocr_texts[p.number] = text
        stats.ocr_pages = len(ocr_texts)

    cleaned: dict[int, list[Line]] = {}
    for p in pages:
        if cleaner.is_toc_or_index(p):
            stats.toc_pages += 1
            continue
        if cleaner.is_image_only(p):
            stats.image_pages += 1
            continue
        if cleaner.is_blank(p):
            stats.blank_pages += 1
            continue
        cleaned[p.number] = cleaner.clean_lines(p)
    style = HeadingStyle.learn([l for ls in cleaned.values() for l in ls])

    # Pass 3: walk in reading order, tracking headings, emitting chunks.
    chunks: list[Chunk] = []
    counter = [0]
    current = _Section(path="", section="")

    def close_section():
        nonlocal current
        if current.units:
            chunks.extend(_pack(current, doc_id, counter))
        current = _Section(path=tracker.path, section=tracker.top_section)

    for p in pages:
        if p.number in ocr_texts:
            close_section()
            tracker.start_page(p.number, [])
            for para in re.split(r"\n\s*\n", ocr_texts[p.number]):
                para = " ".join(para.split())
                if len(para) < 25:
                    continue
                counter[0] += 1
                chunks.append(
                    Chunk(
                        id=f"{doc_id}-o{counter[0]:05d}", doc_id=doc_id, page=p.number, page_end=p.number,
                        chunk_type="text", heading_path=tracker.path, section=tracker.top_section,
                        text=para, has_spec=has_spec(para), source="ocr",
                    )
                )
            continue
        if p.number not in cleaned:
            continue
        lines = cleaned[p.number]
        tracker.start_page(p.number, [l.text for l in lines])
        if tracker.path != current.path:
            close_section()

        body: list[Line] = []

        def flush_body():
            if body:
                current.units.extend(_lines_to_units(body, p.number))
                body.clear()

        mid = p.width / 2
        left_cols = [l.x0 for l in lines if l.x0 < mid]
        right_cols = [l.x0 for l in lines if l.x0 >= mid - 10]
        for ln in lines:
            col = left_cols if ln.x0 < mid else right_cols
            left_x = min(col) if col else None
            label = cleaner.section_label(ln.text)
            if label:
                flush_body()
                tracker.set_section(label)
                if tracker.path != current.path:
                    close_section()
                continue
            if tracker.match_outline_line(ln.text):
                flush_body()
                close_section()
                continue
            level = style.level(ln, left_x)
            if level is not None:
                flush_body()
                tracker.enter_inferred(level, ln.text.strip())
                close_section()
                continue
            body.append(ln)
        flush_body()

        # Tables on this page, with the heading context in force for the page.
        for table in p.tables:
            title, rows = serialise_table(table)
            if not rows:
                continue
            context = tracker.path
            if title and title.lower() not in context.lower():
                context = f"{context} › {title}" if context else title
            for start in range(0, len(rows), TABLE_ROWS_PER_CHUNK):
                group = rows[start : start + TABLE_ROWS_PER_CHUNK]
                text = "\n".join(group)
                counter[0] += 1
                chunks.append(
                    Chunk(
                        id=f"{doc_id}-r{counter[0]:05d}",
                        doc_id=doc_id,
                        page=p.number,
                        page_end=p.number,
                        chunk_type="table",
                        heading_path=context,
                        section=tracker.top_section,
                        text=text,
                        has_spec=has_spec(text) or any(ch.isdigit() for ch in text),
                    )
                )
        if p.number % 10 == 0:
            progress("chunk", p.number / len(pages))
    close_section()
    progress("chunk", 1.0)
    return chunks, stats
