"""Document-level cleaning that adapts to each PDF instead of hard-coding one layout.

* Running headers/footers are found statistically: a line (with digits masked) that
  repeats on many pages, or that repeats in the top/bottom margin, is boilerplate.
* Table-of-contents and index pages are detected by dot-leader / trailing page-number
  lines and excluded, because they keyword-match almost every query.
* Pattern noise: file URLs, figure codes (E120276), lone callout letters, page numbers.
"""

from __future__ import annotations

import re
from collections import Counter

from .parse import Line, PageContent

_DIGITS = re.compile(r"\d+")
_URL = re.compile(r"(file|https?)://\S+", re.I)
_FIG_CODE = re.compile(r"^[A-Z]{1,2}\d{4,7}$")
_CALLOUT = re.compile(r"^(?:[A-Z]|\d{1,2}|[a-z])$")
_PAGE_NO = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/|sur|von|de)\s*\d{1,4})?$", re.I)
_TOC_LINE = re.compile(r"(?:\.{3,}|…{2,}|\. \. \.)\s*\d{1,4}\s*$|,\s*\d{1,4}\s*$")
_SECTION_LABEL = re.compile(r"^(?:SECTION|Section|CHAPTER|Chapter|GROUP|Group)\s+[\w.\-]+\s*[:\-–—]\s*\S.*")

MARGIN = 0.075  # fraction of page height treated as header/footer band


def _norm(text: str) -> str:
    return _DIGITS.sub("#", text.strip().lower())


def _in_margin(line: Line, page: PageContent) -> bool:
    return line.y1 < page.height * MARGIN or line.y0 > page.height * (1 - MARGIN)


class DocCleaner:
    """Learns boilerplate from the whole document, then cleans each page."""

    def __init__(self, pages: list[PageContent]):
        self.n_pages = len(pages)
        anywhere: Counter[str] = Counter()
        margin: Counter[str] = Counter()
        for p in pages:
            seen = {_norm(l.text) for l in p.lines}
            anywhere.update(seen)
            margin.update({_norm(l.text) for l in p.lines if _in_margin(l, p)})
        threshold = max(4, int(0.3 * self.n_pages))
        self.boilerplate = {t for t, c in anywhere.items() if c >= threshold}
        self.margin_boilerplate = {t for t, c in margin.items() if c >= 3}
        self.boilerplate.discard("")

    # ---- page classification -------------------------------------------------
    @staticmethod
    def is_toc_or_index(page: PageContent) -> bool:
        lines = [l.text for l in page.lines if len(l.text) > 3]
        if len(lines) < 8:
            return False
        hits = sum(1 for t in lines if _TOC_LINE.search(t))
        return hits / len(lines) >= 0.35

    @staticmethod
    def is_image_only(page: PageContent) -> bool:
        return page.text_chars < 50 and page.image_count > 0

    @staticmethod
    def is_blank(page: PageContent) -> bool:
        return page.text_chars < 50 and page.image_count == 0 and not page.tables

    # ---- line cleaning -------------------------------------------------------
    def is_noise(self, line: Line, page: PageContent) -> bool:
        text = line.text.strip()
        norm = _norm(text)
        if norm in self.boilerplate:
            return True
        if _in_margin(line, page) and (norm in self.margin_boilerplate or _PAGE_NO.match(text)):
            return True
        if _URL.search(text) and len(_URL.sub("", text).strip()) < 4:
            return True
        if _FIG_CODE.match(text) or _CALLOUT.match(text):
            return True
        if _PAGE_NO.match(text) and _in_margin(line, page):
            return True
        return False

    def clean_lines(self, page: PageContent) -> list[Line]:
        kept = []
        for line in page.lines:
            if self.is_noise(line, page):
                continue
            line.text = _URL.sub("", line.text).strip()
            if line.text:
                kept.append(line)
        return kept

    @staticmethod
    def section_label(text: str) -> str | None:
        m = _SECTION_LABEL.match(text.strip())
        return m.group(0).strip() if m else None


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines into a paragraph, repairing end-of-line hyphenation."""
    out = ""
    for t in lines:
        t = t.strip()
        if not out:
            out = t
        elif out.endswith("-") and t[:1].islower():
            out = out[:-1] + t
        else:
            out = f"{out} {t}"
    return re.sub(r"\s{2,}", " ", out).strip()
