"""Command line interface.

    specx ingest data/manual.pdf
    specx query  data/manual.pdf "torque for brake caliper bolts" --format csv --out specs.csv
    specx list
    specx serve --port 8000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

REQUIRED = ["component", "spec_type", "value", "unit"]
FULL = REQUIRED + ["alt_value", "alt_unit", "variant", "condition", "section", "page", "source", "confidence", "source_quote"]


def _resolve_doc(ref: str) -> str:
    from .index.store import DocIndex
    from .ingest.chunk import file_hash
    from .pipeline import ingest

    p = Path(ref)
    if p.suffix.lower() == ".pdf" and p.exists():
        doc_id = file_hash(str(p))
        if not DocIndex.exists(doc_id):
            print(f"Indexing {p.name} (first run only)...", file=sys.stderr)
            ingest(p, progress=_progress)
        return doc_id
    if DocIndex.exists(ref):
        return ref
    raise SystemExit(f"Unknown document: {ref} (pass a PDF path or a doc_id from `specx list`)")


def _progress(stage: str, frac: float) -> None:
    print(f"\r  {stage:<6} {frac*100:5.1f}%", end="", file=sys.stderr, flush=True)
    if stage == "done":
        print(file=sys.stderr)


def _rows(specs, minimal: bool) -> list[dict]:
    if minimal:
        return [s.required_fields() for s in specs]
    return [{c: (getattr(s, c) if getattr(s, c) is not None else ("" if c == "unit" else None)) for c in FULL} for s in specs]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="specx", description="Extract vehicle specifications from service manuals.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ingest", help="parse, chunk and index a PDF")
    a.add_argument("pdf")
    a.add_argument("--force", action="store_true", help="re-index even if cached")
    a.add_argument("--ocr", action="store_true", help="OCR image-only pages (bonus; needs tesseract)")

    q = sub.add_parser("query", help="ask a spec question")
    q.add_argument("doc", help="PDF path or doc_id")
    q.add_argument("question")
    q.add_argument("--format", choices=["json", "csv", "table"], default="json")
    q.add_argument("--minimal", action="store_true", help="only the 4 required fields")
    q.add_argument("--out", help="write to a file instead of stdout")
    q.add_argument("--offline", action="store_true", help="rule-based extractor, no LLM call")
    q.add_argument("--show-context", action="store_true", help="print retrieved chunks to stderr")

    sub.add_parser("list", help="list indexed documents")
    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)

    if args.cmd == "ingest":
        from .pipeline import ingest

        info = ingest(args.pdf, progress=_progress, force=args.force, enable_ocr=args.ocr or None)
        print(info.model_dump_json(indent=2))

    elif args.cmd == "list":
        from .index.store import DocIndex

        for i in DocIndex.list_infos():
            print(f"{i.doc_id}  {i.pages:>4} pp  {i.chunks:>5} chunks  {i.filename}")

    elif args.cmd == "query":
        from .pipeline import query

        result = query(_resolve_doc(args.doc), args.question, offline=args.offline)
        if args.show_context:
            for n, h in enumerate(result.chunks, 1):
                print(f"--- [C{n}] p.{h.chunk.page} score={h.score:.2f} {h.chunk.heading_path}\n{h.chunk.text}\n", file=sys.stderr)
        rows = _rows(result.specs, args.minimal)
        buf = io.StringIO()
        if args.format == "json":
            json.dump(rows, buf, indent=2, ensure_ascii=False)
            buf.write("\n")
        elif args.format == "csv":
            w = csv.DictWriter(buf, fieldnames=REQUIRED if args.minimal else FULL)
            w.writeheader()
            w.writerows(rows)
        else:
            for r in rows:
                alt = f" ({r.get('alt_value')} {r.get('alt_unit')})" if r.get("alt_value") else ""
                page = f"  p.{r['page']}" if r.get("page") else ""
                print(f"{r['component']:<55} {r['spec_type']:<20} {r['value']} {r['unit']}{alt}{page}", file=buf)
        text = buf.getvalue()
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {len(rows)} specs to {args.out}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        if result.rejected:
            print(f"({len(result.rejected)} candidate(s) rejected by grounding check)", file=sys.stderr)

    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("specx.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
