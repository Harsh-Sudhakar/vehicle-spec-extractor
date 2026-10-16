# SpecX — Vehicle Specification Extraction from Service Manuals

SpecX extracts **vehicle specifications** (torque values, fluid capacities, pressures, clearances, wear limits, part numbers, fluid specs, …) from **automotive service manual PDFs** using **retrieval + an LLM**, and returns structured **JSON / CSV**:

```json
[
  { "component": "Brake caliper flow bolt", "spec_type": "Torque", "value": "35", "unit": "Nm" }
]
```

Every answer is **grounded**: it carries the page it came from, the passage the model quoted, and a check that the value really appears in that passage. If the manual doesn't contain the answer, SpecX returns `[]` instead of guessing. When the manual instead points elsewhere (the F-150 workshop manual says tyre pressures are on the *Vehicle Certification label*), it shows that reference, with its page.

It works on **any** text-based manual, not just one layout. It has been tested on two very different PDFs:

| Manual | Pages | Layout | Structure source |
|---|---|---|---|
| 2014 Ford F-150 Workshop Manual (provided sample) | 852 | HTML-printed workshop manual, ruled torque tables, numbered procedures, no bookmarks | inferred headings + `SECTION xxx` labels |
| 2012 Ford Figo Owner's Manual | 114 | typeset, two-column, stacked spec tables per engine variant, TOC/index pages, image-only pages | PDF outline (bookmarks) |

---

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # add OPENROUTER_API_KEY
```

**Manuals:** copy PDFs into `data/`. They are git-ignored (copyrighted) and indexed automatically when the server starts. The UI opens on `data/f150-workshop-manual.pdf` (change with `DEFAULT_DOC`); any other PDF can be uploaded from the UI.

```bash
cp "/path/to/sample-service-manual.pdf" data/f150-workshop-manual.pdf
```

**Web app** (upload a PDF, ask questions, export CSV/JSON, inspect evidence):

```bash
specx serve --port 8000         # → http://localhost:8000
```

**CLI:**

```bash
specx ingest data/manual.pdf
specx query data/manual.pdf "torque for brake caliper bolts"                 # JSON
specx query data/manual.pdf "engine oil capacity" --format csv --out oil.csv
specx query data/manual.pdf "tyre pressure" --minimal                        # only the 4 required fields
specx query data/manual.pdf "tyre pressure" --show-context                   # print retrieved chunks
```

**Notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb) walks through each pipeline stage on the sample manual.

**Tests / evaluation:**

```bash
pytest -q
python eval/run_eval.py --doc f150=data/f150-workshop-manual.pdf --doc figo=data/figo-owners-manual.pdf
```

Without an API key the app still runs, using a rule-based **offline extractor** (table unit-columns + regex). It serves as a baseline and a fallback; the LLM path is the real system.

---

## Architecture

```
                         INGESTION (once per PDF, cached by file hash)
┌────────┐  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  PDF   │─▶│ 1 Parse          │─▶│ 2 Clean           │─▶│ 3 Structure     │─▶│ 4 Chunk         │─▶│ 5 Index          │
│        │  │ tables FIRST     │  │ learned headers / │  │ PDF outline, or │  │ table rows +    │  │ BGE embeddings   │
│        │  │ (find_tables),   │  │ footers, TOC &    │  │ inferred bold / │  │ heading-bounded │  │ → FAISS          │
│        │  │ prose outside    │  │ index pages,      │  │ large headings, │  │ prose, steps    │  │ BM25 (keywords)  │
│        │  │ table areas,     │  │ figure codes,     │  │ SECTION labels  │  │ never split     │  │                  │
│        │  │ column-aware     │  │ symbol bullets    │  │ → heading path  │  │                 │  │                  │
└────────┘  └──────────────────┘  └───────────────────┘  └─────────────────┘  └─────────────────┘  └──────────────────┘

                         QUERY
"torque for brake caliper bolts"
   │
   ├─▶ dense top-25 (BGE) ─┐
   │                       ├─▶ Reciprocal Rank Fusion ─▶ cross-encoder rerank (+ small priors for
   └─▶ BM25 top-25 ────────┘                              chunks with values / table rows) ─▶ top-8
                                                                    │
                                     LLM (OpenRouter, JSON mode, temperature 0, strict prompt)
                                                                    │
                       Pydantic schema ─▶ grounding check ─▶ unit / type normalisation ─▶ dedupe
                                                                    │
                                                       JSON / CSV / web UI (+ evidence)
```

### Repository layout

```
src/specx/
  ingest/parse.py       PyMuPDF: tables first, prose outside tables, column-aware reading order
  ingest/clean.py       statistical header/footer removal, TOC/index detection, noise patterns
  ingest/structure.py   outline vs. inferred headings, section labels, heading-path tracking
  ingest/chunk.py       table-row + prose chunkers, document walk
  ingest/ocr.py         optional OCR for image-only pages (bonus, off by default)
  index/models.py       fastembed (ONNX) embedder + cross-encoder reranker
  index/store.py        FAISS + BM25 per document, persisted under data/indexes/<doc_id>/
  query/retrieve.py     hybrid retrieval, RRF, reranking
  query/prompts.py      extraction prompt
  query/llm.py          OpenRouter client (OpenAI-compatible) in JSON mode
  query/extract.py      LLM extraction + offline fallback
  query/postprocess.py  grounding, normalisation, dedupe
  pipeline.py           ingest() / query() API
  server.py             FastAPI app (API + UI)
  cli.py                `specx` command
web/static/             UI (vanilla HTML/CSS/JS, no build step)
eval/                   gold set + evaluation script
tests/                  unit tests (no models / network needed)
```

---

## Design decisions

### 1. PDF parsing — tables first
Specs live mostly in tables. `page.get_text()` linearises tables cell by cell, and in typeset manuals it can even emit a value **before** its label (`2.06 (30)` / `Front`). So each page is parsed in two passes:

1. `page.find_tables()` extracts tables as rows of cells.
2. Prose lines are taken from `get_text("dict")` **only outside the table bounding boxes**, so cells never leak into the text stream in scrambled order.

Reading order is column-aware (left column before right, bands split at full-width blocks), and fragments on the same baseline are joined only when they are close together. Step numbers stay with their text, while a left header and a right header on the same line stay apart.

### 2. Cleaning — learned per document, not hard-coded
Nothing is specific to one manual:

* **Running headers/footers:** with digits masked (`Page # sur #`), a line repeating on ≥30% of pages, or ≥3 times in the top/bottom margin band, is boilerplate.
* **TOC and index pages** (dot leaders / trailing page numbers on ≥35% of lines) are dropped. They keyword-match nearly every query and would otherwise win retrieval.
* **Noise patterns:** `file:///` URLs, figure codes (`E120276`), lone callout letters, page numbers.
* **Symbol-font bullets** that decode to letters (`z Tighten the nuts…`) are detected by font (and in table text by pattern) and turned back into bullets.
* **Image-only and blank pages** are skipped (text-only extraction, per the assignment).

### 3. Structure — outline if present, inferred otherwise
* If the PDF has bookmarks, they give a reliable heading path (`Maintenance › Technical Specifications`).
* Otherwise headings are short lines larger than body text, or bold at body size **and** starting at the text margin. Font size alone fails: in the F-150 manual the running footer is the largest text on the page, and indented bold lines are table headers.
* `SECTION 206-03: Front Disc Brake`-style labels are carried forward across pages, because workshop manuals print them only on a procedure's first page.

### 4. Chunking — two chunk types
**Table rows** (1–3 rows per chunk) are serialised with their column headers, title and heading path:

```
[SECTION 206-03: Front Disc Brake › Torque Specifications]
Description: Brake caliper guide pin bolts | Nm: 37 | lb-ft: 27 | lb-in: —
```
```
[Maintenance › Technical Specifications]
Engine specifications › Engine type: Spark plug gap | 1.2L Duratec: 1.3mm | 1.4L Duratorq- TDCi: -
```
Category rows (`Brake Disc`) are carried down as prefixes. Stacked sub-tables that restart with their own header after a blank or category row get their own column names. A bare `37` therefore always keeps its meaning (Nm, guide pin bolts, front disc brake).

**Prose** is grouped into heading-bounded chunks of about 200 words with a one-paragraph overlap. **Numbered steps are never split**, and indented bullets attach to the step above, so the component is not separated from its torque:

```
12. Position the stabilizer bar, install the 2 stabilizer bar brackets and the 4 new stabilizer bar bracket nuts.
• Tighten the nuts to 55 Nm (41 lb-ft).
```

Every chunk is embedded with its heading path as a prefix, and flagged `has_spec` when it contains a value+unit or a spec code.

### 5. Retrieval — hybrid + rerank
* **Dense** (`BAAI/bge-small-en-v1.5`) handles paraphrase: *"how much oil does the diesel take"*.
* **BM25** handles exact tokens that embeddings blur: `WSS-M6C57-A2`, `EFL 600`, `Nm`. The tokenizer keeps compound codes and also splits them.
* **Reciprocal Rank Fusion** merges the two lists without needing to calibrate their scores.
* A **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) re-scores the top 30 by reading query and chunk together, with small priors for chunks that contain values and for table rows. The top 8 go to the LLM.

Both models run through **fastembed (ONNX)**: the same weights as Sentence-Transformers, without a PyTorch dependency, which keeps the Docker image small.

### 6. LLM extraction — strict and grounded
The prompt ([`query/prompts.py`](src/specx/query/prompts.py)) asks for JSON only, at temperature 0, and requires the model to:
* use only the excerpts, and return `[]` if nothing answers the question;
* emit one object per value, putting dual units in `value/unit` + `alt_value/alt_unit` and multi-stage torques in separate objects;
* resolve references like *"the nuts"* from the step or heading;
* copy values verbatim (ranges, tolerances, codes) and record the `variant` (engine / drivetrain);
* cite the excerpt id and a short verbatim `source_quote`.

### 7. Post-processing — trust, but verify
* **Pydantic** validates each object against the schema.
* **Grounding check:** the value must literally appear in the excerpt it cites (after normalising whitespace, thousands separators and brackets). Entries that fail are **rejected** and shown in the UI. A confidence score also checks that the component name and quote match the excerpt.
* **Normalisation:** `N·m → Nm`, `litres → L`, `gms → g`, units split out of values (`"1,420 ml"` → `1,420` + `ml`), spec types mapped to canonical names.
* **Dedupe** across table and procedure mentions, preferring table rows.
* **Nothing found:** returns `[]`. If the cited text points elsewhere ("set to the pressure on the VC label"), that reference is shown, and only if it is worded from a retrieved passage.
* **Answer cache:** results are cached per (document, normalised question), in memory and on disk next to the index, and cleared on re-index. Repeated questions cost no LLM call.

### Output schema
The four required fields come first; the rest are additions (dropped with `--minimal` or the UI toggle).

| field | example |
|---|---|
| `component` | `Brake caliper guide pin bolts` |
| `spec_type` | `Torque` |
| `value` | `37` (string: also `1040 to 1090`, `0.30 ± 0.03`, `EFL 600`) |
| `unit` | `Nm` |
| `alt_value`, `alt_unit` | `27`, `lb-ft` |
| `variant`, `condition` | `1.4L Duratorq-TDCi`, `Laden` |
| `section`, `page` | `SECTION 206-03: Front Disc Brake`, `636` |
| `source`, `confidence`, `source_quote` | `table`, `1.0`, `Brake caliper guide pin bolts \| Nm: 37` |

---

## Evaluation

The gold set ([`eval/gold.jsonl`](eval/gold.jsonl)) has 21 questions checked by hand against the PDF pages. It covers torque, capacity, pressure, thickness, clearance, weight ranges, part numbers and fluid specs, answers found only in procedure prose, an ambiguous component that appears in several sections, and a **negative** question (the Figo owner's manual has no caliper torque, so the correct answer is `[]`).

| Metric | LLM pipeline | Offline rule-based baseline |
|---|---|---|
| Retrieval hit@8 (gold page retrieved) | **20/20 (100%)** | 20/20 (100%) |
| Item recall (expected values extracted) | **22/22 (100%)** | 14/22 (64%) |
| Query accuracy | **21/21 (100%)** | 13/21 (62%) |
| Negative question → `[]` | **1/1** | 1/1 |
| Median latency | 6.2 s | 2.2 s |

Both columns share the same retrieval, so the gap is in extraction. The rule-based baseline handles the F-150's regular `Nm | lb-ft | lb-in` tables but fails on all of the Figo's irregular tables: variant columns, units in table titles, codes and ranges. The LLM handles both without per-manual code. Full per-question results: [`eval/results_llm.md`](eval/results_llm.md), [`eval/results_offline.md`](eval/results_offline.md).

The set is small (21 questions over 2 manuals), so treat it as a regression check rather than a benchmark. Latency was measured on a heavily loaded laptop CPU and is dominated by the LLM call.

---

## Tools used

| Stage | Tool | Why |
|---|---|---|
| PDF parsing | **PyMuPDF** | fast; gives fonts, positions, `find_tables()` and the outline |
| Embeddings | **BGE-small-en-v1.5** via fastembed | strong retrieval quality for its size, local, free |
| Reranking | **ms-marco-MiniLM-L-6-v2** cross-encoder | fixes most first-stage ranking errors, cheaply |
| Vector store | **FAISS** (`IndexFlatIP`) | exact search; a manual is a few thousand vectors |
| Keyword search | **rank-bm25** | exact part numbers and codes |
| LLM | **OpenRouter** (`deepseek/deepseek-v4-flash` by default, any JSON-capable model) | OpenAI-compatible, cheap, model is one env var |
| Validation | **Pydantic v2** | schema enforcement |
| API / UI | **FastAPI** + vanilla JS | no build step, one container |
| OCR (bonus) | **Tesseract** | optional, for image-only pages |

**Why not LangChain?** The quality-critical parts here (table-first parsing, learned cleaning, structure-aware chunking, grounding) are custom in any case: LangChain's PDF loaders flatten tables and its generic splitters cut procedure steps apart. What remains (an embedding call, FAISS, BM25, RRF, one LLM call) is about a hundred lines of plain Python that are easier to read, debug and explain.

---

## Web UI

* **Manuals library:** upload a PDF and watch Parse → Clean → Chunk → Embed progress. Each manual shows its structure source (outline or inferred) and how many TOC, image-only and blank pages were skipped.
* **Query console** with a **pipeline trace** showing real timings for retrieve / rerank / extract and how many candidates passed the grounding check.
* **Results** as a readable Sheet or as the raw JSON output, a "4 required fields" toggle, and CSV / JSON export.
* **Evidence panel:** for any spec, the value, the model's quote, the cited passage with the value marked, and the **rendered source page with the match highlighted**.
* **Retrieved context:** the top chunks with fused score and their dense / BM25 ranks, so retrieval can be inspected directly.

---

## Deployment

The repo ships a `Dockerfile` (FastAPI + UI, embedding models downloaded at build time, listens on `$PORT`) and a [`render.yaml`](render.yaml) Blueprint.

**The live demo runs on Render's free plan:**

* **Manual data stays private.** Manuals are copyrighted, so they are not in git. The prebuilt F-150 index (chunks, FAISS index, source PDF) lives in a *private* Hugging Face dataset and is downloaded at start-up (`HF_DATA_REPO` + a read-only `HF_TOKEN`). The 852-page manual is never re-indexed on the server.
* **Fits in 512 MB.** The cross-encoder reranker is switched off (`USE_RERANKER=false`), and retrieval falls back to the RRF order. The eval still scores 21/21 that way ([`eval/results_llm_no_rerank.md`](eval/results_llm_no_rerank.md)). Peak memory measured with this configuration is about 350 MB. Uploads are off (`ALLOW_UPLOAD=false`) because indexing a large manual needs more CPU/RAM than the free plan has.
* **Cold starts.** The free instance sleeps after 15 minutes idle, so the first visit takes about a minute to wake.

To deploy your own: *Render → New → Blueprint →* select this repo, then enter `OPENROUTER_API_KEY` and `HF_TOKEN` when prompted. Any other Docker host works the same way. With 2 GB+ RAM, leave the reranker and uploads on.

Configuration (env vars, see `.env.example`): `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `USE_RERANKER`, `ALLOW_UPLOAD`, `HF_DATA_REPO`, `HF_TOKEN`, `DEFAULT_DOC`, `DEFAULT_QUERY`, `ENABLE_OCR`, `MAX_UPLOAD_MB`, `RATE_LIMIT_PER_MIN`, `DATA_DIR`. Keys are read from the environment or a local `.env`, which is git-ignored. Never commit them.

> Locally, any PDF placed in `data/` is indexed automatically when the server starts.

---

## Limitations

* **Text only by design.** Specs that exist only inside diagrams or exploded views are not extracted. Image-only pages are skipped unless OCR is enabled, and OCR reads text but does not rebuild complex graphical grids (e.g. a tick-mark maintenance schedule).
* **Borderless tables** that `find_tables()` misses are handled as prose. The LLM usually still reads them, but column meaning is weaker.
* **Very broad questions** ("all engine specs") can exceed the top-8 context.
* **Same component in several sections** (e.g. brake disc shield bolts: 17 Nm in *204-01A Front Suspension*, 18 Nm in *206-03 Front Disc Brake*) returns all matches with section and page, which is correct but needs the reader to pick the right one.
* Ingestion of an 850-page manual takes a few minutes on CPU, almost all of it embedding. It runs once per file and is cached by content hash.

## Ideas for improvement

1. **Layout model for tables** (Docling / Table Transformer) for borderless and multi-page tables.
2. **Vision-LLM pass on figures** to read torque callouts from exploded-view diagrams, as an opt-in mode next to OCR.
3. **Query understanding:** expand synonyms ("lug nut" ↔ "wheel nut"), detect variant hints ("diesel" → 1.4L Duratorq) and filter on metadata before ranking.
4. **Unit conversion & validation:** convert to a requested unit system and flag implausible values (e.g. torque in mm).
5. **Larger gold set** across more manufacturers, with automatic regression runs in CI.
6. **Incremental / parallel ingestion** (page-level multiprocessing, quantised embeddings) for very large manuals.
7. **Semantic answer cache:** reuse answers for paraphrased questions, not just identical ones.
