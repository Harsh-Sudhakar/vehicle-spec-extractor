"""Heading detection and heading-path tracking.

Two strategies, chosen per document:

1. **Outline** - if the PDF has bookmarks, they are the most reliable structure.
2. **Inferred** - otherwise headings are short lines that are larger than body text,
   or bold at body size. Font size alone is not enough: in some manuals the running
   footer is the largest text on the page.

A section label pattern ("SECTION 206-03: Front Disc Brake") is also tracked and
carried forward across pages, since many workshop manuals print it only on the first
page of a procedure.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .parse import Line

_STEP_START = re.compile(r"^(?:\d{1,2}[.)]\s|step\s+\d+|•)", re.I)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


@dataclass
class HeadingStyle:
    body_size: float
    levels: dict[float, int]  # size -> level for larger-than-body headings

    @classmethod
    def learn(cls, lines: list[Line]) -> "HeadingStyle":
        sizes: Counter[float] = Counter()
        for l in lines:
            sizes[l.size] += len(l.text)
        body = sizes.most_common(1)[0][0] if sizes else 10.0
        big = sorted({l.size for l in lines if l.size >= body * 1.15 and cls._shape_ok(l.text)}, reverse=True)
        levels = {s: i + 1 for i, s in enumerate(big[:3])}
        return cls(body_size=body, levels=levels)

    @staticmethod
    def _shape_ok(text: str) -> bool:
        t = text.strip()
        return (
            3 <= len(t) <= 80
            and len(t.split()) <= 12
            and any(c.isalpha() for c in t)
            and not t.endswith((".", ",", ";"))
            and not _STEP_START.match(t)
            and not t.lower().startswith(("note", "warning", "caution", "important"))
        )

    def level(self, line: Line, left_x: float | None = None) -> int | None:
        if not self._shape_ok(line.text):
            return None
        if line.size in self.levels:
            return self.levels[line.size]
        # Bold body-size text is a heading only when it starts at the text margin;
        # indented bold lines are usually table headers or emphasis.
        indented = left_x is not None and line.x0 > left_x + 40
        if line.bold and line.size >= self.body_size * 0.95 and not indented:
            return len(self.levels) + 1
        return None


@dataclass
class HeadingTracker:
    """Keeps the current heading path while walking the document in order."""

    outline: dict[int, list[tuple[int, str]]] = field(default_factory=dict)  # page -> [(level, title)]
    stack: list[tuple[int, str]] = field(default_factory=list)
    section: str = ""
    outline_depth: int = 0

    @classmethod
    def from_toc(cls, toc: list[list]) -> "HeadingTracker":
        outline: dict[int, list[tuple[int, str]]] = {}
        depth = 0
        for level, title, page, *_ in toc:
            title = re.sub(r"\s+", " ", str(title)).strip()
            if page > 0 and title:
                outline.setdefault(page, []).append((level, title))
                depth = max(depth, level)
        return cls(outline=outline, outline_depth=depth)

    @property
    def uses_outline(self) -> bool:
        return sum(len(v) for v in self.outline.values()) >= 5

    def _push(self, level: int, title: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))

    def start_page(self, page: int, line_texts: list[str]) -> None:
        """Entering a page. Outline entries are applied when their title line is met;
        entries whose title never appears on the page are applied up front."""
        self.flush_pending()
        present = {_norm_title(t) for t in line_texts}
        pending = []
        for level, title in self.outline.get(page, []):
            n = _norm_title(title)
            if any(p == n or (len(p) > 6 and n.startswith(p)) for p in present):
                pending.append((level, title))
            else:
                self._push(level, title)
        self._pending = pending

    def match_outline_line(self, text: str) -> bool:
        """If `text` is the title of an outline entry on this page, enter it."""
        n = _norm_title(text)
        for i, (level, title) in enumerate(getattr(self, "_pending", [])):
            if n and (n == _norm_title(title) or (len(n) > 6 and _norm_title(title).startswith(n))):
                for lv, tt in self._pending[: i + 1]:
                    self._push(lv, tt)
                del self._pending[: i + 1]
                return True
        return False

    def flush_pending(self) -> None:
        for lv, tt in getattr(self, "_pending", []):
            self._push(lv, tt)
        self._pending = []

    def enter_inferred(self, level: int, title: str) -> None:
        if self.uses_outline:
            level = self.outline_depth + level  # inferred headings nest under the outline
        self._push(level, title)

    def set_section(self, label: str) -> None:
        if label != self.section:
            self.section = label
            # A new section resets inferred headings beneath it.
            self.stack = [s for s in self.stack if self.uses_outline and s[0] <= self.outline_depth]

    @property
    def path(self) -> str:
        parts = [self.section] if self.section else []
        seen = {_norm_title(p) for p in parts}
        for _, title in self.stack:
            n = _norm_title(title)
            if n not in seen:
                parts.append(title)
                seen.add(n)
        return " › ".join(parts)

    @property
    def top_section(self) -> str:
        if self.section:
            return self.section
        return self.stack[0][1] if self.stack else ""
