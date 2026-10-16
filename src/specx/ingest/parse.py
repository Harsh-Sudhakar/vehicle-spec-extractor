"""Layout-aware page parsing with PyMuPDF.

Tables are extracted first with ``page.find_tables()``. Prose lines are then taken
only from outside the table areas, so table cells never leak into the text stream
in scrambled order (a real problem in typeset manuals, where ``get_text()`` can emit
a value before its label).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

# Fonts whose glyphs are bullets / symbols that decode to stray letters (e.g. "z").
_SYMBOL_FONT = re.compile(r"dingbat|symbol|wingding", re.I)


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    x0: float
    y0: float
    x1: float
    y1: float
    block: int


@dataclass
class Table:
    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    title: str | None = None


@dataclass
class PageContent:
    number: int  # 1-based
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    text_chars: int = 0
    image_count: int = 0


def _clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    value = value.replace("­", "")
    value = re.sub(r"-\n(?=[a-z])", "", value)  # de-hyphenate wrapped words
    # Symbol-font bullets lose their font in table text and surface as a lone "z"/"l"/"n".
    value = re.sub(r"(?:(?<=^)|(?<=\s))[zln]\s+(?=[A-Z])", "• ", value)
    value = re.sub(r"\s*\n\s*", " ", value)
    return re.sub(r"\s{2,}", " ", value).strip()


def _inside(x0: float, y0: float, x1: float, y1: float, box: tuple[float, float, float, float]) -> bool:
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = box
    return bx0 - 1 <= cx <= bx1 + 1 and by0 - 1 <= cy <= by1 + 1


def _extract_tables(page: pymupdf.Page) -> list[Table]:
    try:
        found = page.find_tables()
    except Exception:  # pragma: no cover - defensive: malformed vector graphics
        return []
    tables: list[Table] = []
    for t in found.tables:
        rows = [[_clean_cell(c) for c in row] for row in t.extract()]
        # Blank rows are kept: they often separate stacked sub-tables.
        while rows and not any(rows[0]):
            rows.pop(0)
        filled = [r for r in rows if any(r)]
        # A real table needs at least two rows and two populated columns somewhere.
        if len(filled) < 2 or max(sum(1 for c in r if c) for r in filled) < 2:
            continue
        tables.append(Table(rows=rows, bbox=tuple(t.bbox)))
    return tables


def _reading_order(blocks: list[dict], width: float) -> dict[int, int]:
    """Column-aware block order.

    Blocks are cut into horizontal bands at full-width blocks; inside a band the left
    column is read before the right one. Single-column pages reduce to top-to-bottom.
    """
    mid = width / 2
    blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    two_col = any(b["bbox"][0] > mid - 10 for b in blocks) and any(b["bbox"][2] < mid + 10 for b in blocks)
    order: list[dict] = []
    band: list[dict] = []

    def flush():
        band.sort(key=lambda b: (0 if b["bbox"][0] < mid - 10 else 1, b["bbox"][1], b["bbox"][0]))
        order.extend(band)
        band.clear()

    for b in blocks:
        x0, _, x1, _ = b["bbox"]
        full_width = not two_col or (x0 < mid - 10 and x1 > mid + 30)
        if full_width:
            flush()
            order.append(b)
        else:
            band.append(b)
    flush()
    return {b["number"]: i for i, b in enumerate(order)}


def _extract_lines(page: pymupdf.Page, tables: list[Table]) -> list[Line]:
    data = page.get_text("dict")
    text_blocks = [b for b in data["blocks"] if b.get("type") == 0]
    rank = _reading_order(text_blocks, page.rect.width)
    out: list[Line] = []
    for block in text_blocks:
        for ln in block["lines"]:
            parts: list[str] = []
            sizes: list[float] = []
            bold_chars = total_chars = 0
            spans = [s for s in ln["spans"]]
            body_fonts = [s.get("font", "") for s in spans if len(s["text"].strip()) > 2]
            for idx, span in enumerate(spans):
                text = span["text"]
                if not text.strip():
                    parts.append(text)
                    continue
                font = span.get("font", "")
                # Bullet glyphs from symbol fonts decode to letters ("z", "l", "n"). Catch them by
                # font name, or as a lone leading character in a font the rest of the line doesn't use.
                is_bullet = len(text.strip()) <= 2 and (
                    _SYMBOL_FONT.search(font) or (not "".join(parts).strip() and body_fonts and font not in body_fonts)
                )
                if is_bullet:
                    parts.append("• ")
                    continue
                parts.append(text)
                n = len(text.strip())
                total_chars += n
                sizes.append(span["size"])
                if span["flags"] & 16 or "bold" in span.get("font", "").lower():
                    bold_chars += n
            text = "".join(parts).strip()
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            if any(_inside(x0, y0, x1, y1, t.bbox) for t in tables):
                continue
            out.append(
                Line(
                    text=text,
                    size=round(max(sizes) if sizes else 0, 1),
                    bold=total_chars > 0 and bold_chars / total_chars > 0.6,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    block=rank[block["number"]],
                )
            )
    return _merge_same_row(out)


def _merge_same_row(lines: list[Line]) -> list[Line]:
    """Join fragments on the same baseline in the same block, e.g. a step number "9." and its text."""
    merged: list[Line] = []
    for ln in sorted(lines, key=lambda l: (l.block, round(l.y0), l.x0)):
        prev = merged[-1] if merged else None
        # Only join close fragments; a wide gap means separate items (e.g. left/right header).
        if (
            prev
            and prev.block == ln.block
            and abs(prev.y0 - ln.y0) < 2.5
            and ln.x0 >= prev.x0
            and ln.x0 - prev.x1 < 25
        ):
            prev.text = f"{prev.text} {ln.text}"
            prev.x1 = max(prev.x1, ln.x1)
            prev.bold = prev.bold and ln.bold
            prev.size = max(prev.size, ln.size)
        else:
            merged.append(ln)
    return merged


def _attach_titles(tables: list[Table], lines: list[Line], page_height: float) -> None:
    """Use the nearest short line directly above a table as its title (never a running header)."""
    top_band = page_height * 0.075
    for t in tables:
        x0, y0, x1, _ = t.bbox
        above = [
            l
            for l in lines
            if l.y1 <= y0 + 2
            and y0 - l.y1 < 28
            and l.y0 > top_band
            and l.x1 > x0 - 5
            and l.x0 < x1
            and len(l.text) < 90
        ]
        if above:
            t.title = max(above, key=lambda l: l.y1).text


def parse_page(page: pymupdf.Page) -> PageContent:
    tables = _extract_tables(page)
    lines = _extract_lines(page, tables)
    _attach_titles(tables, lines, page.rect.height)
    text = page.get_text()
    return PageContent(
        number=page.number + 1,
        width=page.rect.width,
        height=page.rect.height,
        lines=lines,
        tables=tables,
        text_chars=len(text.strip()),
        image_count=len(page.get_images()),
    )
