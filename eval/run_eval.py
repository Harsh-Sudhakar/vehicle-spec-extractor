"""Evaluate retrieval and extraction against a hand-checked gold set.

    python eval/run_eval.py --doc f150=data/f150-workshop-manual.pdf --doc figo=data/figo-owners-manual.pdf
    python eval/run_eval.py ... --offline          # rule-based baseline, no LLM

Metrics
  retrieval hit@k   gold page appears among the retrieved chunks
  item recall       expected values found in the extracted specs
  query accuracy    every expected value found (or, for "any", at least one; for "none", empty output)
  negatives         questions with no answer in the manual correctly return []
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specx.index.store import DocIndex  # noqa: E402
from specx.ingest.chunk import file_hash  # noqa: E402
from specx.pipeline import ingest, query  # noqa: E402


def norm(v: str) -> str:
    v = (v or "").lower().replace(",", "")
    return re.sub(r"[\s()]", "", v)


def found(expect: dict, specs: list) -> bool:
    ev, eu = norm(expect["value"]), (expect.get("unit") or "").lower()
    for s in specs:
        vals = [norm(s.value), norm(s.alt_value or "")]
        # exact, or a prefix that ends at a number boundary ("0.30" matches "0.30±0.03", "5" never matches "45")
        if any(v and re.match(re.escape(ev) + r"(?![\d.])", v) for v in vals):
            units = {(s.unit or "").lower(), (s.alt_unit or "").lower()}
            if not eu or eu in units:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", required=True, help="alias=path.pdf")
    ap.add_argument("--gold", default=str(Path(__file__).with_name("gold.jsonl")))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=None, help="write a markdown report")
    args = ap.parse_args()

    docs = {}
    for d in args.doc:
        alias, path = d.split("=", 1)
        doc_id = file_hash(path)
        if not DocIndex.exists(doc_id):
            print(f"indexing {path} ...", file=sys.stderr)
            ingest(path)
        docs[alias] = doc_id

    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    rows, latencies = [], []
    hit_k = items = items_found = q_ok = neg_total = neg_ok = 0
    for g in gold:
        if g["doc"] not in docs:
            continue
        t = time.perf_counter()
        r = query(docs[g["doc"]], g["query"], offline=args.offline, use_cache=False)
        latencies.append(time.perf_counter() - t)
        pages = {h.chunk.page for h in r.chunks}
        if g["match"] == "none":
            neg_total += 1
            ok = len(r.specs) == 0
            neg_ok += ok
            q_ok += ok
            rows.append((g, "–", "–", ok, len(r.specs)))
            continue
        hit = g["page"] in pages
        hit_k += hit
        flags = [found(e, r.specs) for e in g["expect"]]
        if g["match"] == "any":
            items += 1
            items_found += any(flags)
            ok = any(flags)
        else:
            items += len(flags)
            items_found += sum(flags)
            ok = all(flags)
        q_ok += ok
        rows.append((g, "✓" if hit else "✗", f"{sum(flags)}/{len(flags)}", ok, len(r.specs)))

    pos = len([g for g in gold if g["doc"] in docs and g["match"] != "none"])
    total = len(rows)
    mode = "offline rule-based" if args.offline else "LLM"
    lines = [
        f"### Evaluation ({mode})",
        "",
        "| # | doc | query | kind | hit@k | values | correct | #specs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, (g, hit, vals, ok, n) in enumerate(rows, 1):
        lines.append(f"| {i} | {g['doc']} | {g['query']} | {g['kind']} | {hit} | {vals} | {'✅' if ok else '❌'} | {n} |")
    lines += [
        "",
        f"- **Retrieval hit@k:** {hit_k}/{pos} ({hit_k / max(pos, 1):.0%})",
        f"- **Item recall:** {items_found}/{items} ({items_found / max(items, 1):.0%})",
        f"- **Query accuracy:** {q_ok}/{total} ({q_ok / max(total, 1):.0%})",
        f"- **Negatives (correct empty answer):** {neg_ok}/{neg_total}",
        f"- **Median latency:** {statistics.median(latencies):.1f}s",
    ]
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
