"""Data models shared across the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ChunkType = Literal["table", "text"]
Source = Literal["table", "text", "ocr"]


class Block(BaseModel):
    """A cleaned unit of page content before chunking."""

    page: int  # 1-based
    kind: Literal["heading", "text", "table"]
    text: str
    level: int = 0  # heading level (1 = top)
    rows: list[list[str]] | None = None  # table rows (header first) when kind == "table"
    title: str | None = None  # table caption, if found
    source: Source = "text"


class Chunk(BaseModel):
    """A retrievable piece of the manual with its context."""

    id: str
    doc_id: str
    page: int
    page_end: int
    chunk_type: ChunkType
    heading_path: str = ""
    section: str = ""
    text: str
    has_spec: bool = False
    source: Source = "text"

    def embed_text(self) -> str:
        """Text that is embedded and BM25-indexed: context path + body."""
        prefix = f"[{self.heading_path}] " if self.heading_path else ""
        return prefix + self.text


class Spec(BaseModel):
    """One extracted specification. The first four fields are the assignment's required output."""

    component: str
    spec_type: str
    value: str
    unit: str | None = None
    variant: str | None = None
    alt_value: str | None = None
    alt_unit: str | None = None
    condition: str | None = None
    section: str | None = None
    page: int | None = None
    source_quote: str | None = None
    chunk_id: str | None = None
    source: Source = "text"
    confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("component", "spec_type", "value")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("unit", "variant", "alt_value", "alt_unit", "condition", mode="before")
    @classmethod
    def empty_to_none(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return None if v in ("", "-", "—", "null", "None", "N/A", "n/a") else v

    def required_fields(self) -> dict:
        """The assignment's 4-field output. Variant/condition are folded into the component
        so that e.g. laden vs. unladen tyre pressures stay distinguishable."""
        qualifiers = [q for q in (self.variant, self.condition) if q and q.lower() not in self.component.lower()]
        component = f"{self.component} ({', '.join(qualifiers)})" if qualifiers else self.component
        return {"component": component, "spec_type": self.spec_type, "value": self.value, "unit": self.unit or ""}


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class QueryResult(BaseModel):
    query: str
    specs: list[Spec]
    chunks: list[RetrievedChunk]
    rejected: list[dict] = []
    timings_ms: dict[str, float] = {}
    model: str | None = None
    mode: Literal["llm", "offline"] = "llm"


class DocInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    table_chunks: int
    text_chunks: int
    ocr_pages: int = 0
    skipped_pages: dict[str, int] = {}
    has_outline: bool = False
    title: str | None = None
    ingest_ms: float = 0
