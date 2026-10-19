/* SpecX front-end — no framework, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const slug = (s) => String(s || "other").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  const REQUIRED = ["component", "spec_type", "value", "unit"];
  const FULL = [...REQUIRED, "alt_value", "alt_unit", "variant", "condition", "section", "page", "source", "confidence", "source_quote"];

  const state = {
    health: null,
    docs: [],
    doc: null,
    result: null,
    selected: -1,
    view: "cards",
    requiredOnly: false,
    busy: false,
  };

  // ----------------------------------------------------------------- visitor id
  // Random per-browser id; the server shows each visitor only their own uploads.
  const VISITOR = (() => {
    let v = store.get("specx-visitor");
    if (!v || !/^[A-Za-z0-9-]{8,64}$/.test(v)) {
      v = (crypto.randomUUID && crypto.randomUUID()) || `v-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
      store.set("specx-visitor", v);
    }
    return v;
  })();
  const withVisitor = (url) => `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(VISITOR)}`;

  // ----------------------------------------------------------------- utils
  async function api(path, opts = {}) {
    const res = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), "X-SpecX-Visitor": VISITOR } });
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
      throw new Error(msg);
    }
    return res.json();
  }

  let toastTimer;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 2800);
  }

  const fmtMs = (ms) => (ms == null ? "" : ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`);
  const fmtInt = (n) => Number(n || 0).toLocaleString();

  function download(name, text, type) {
    const blob = new Blob([text], { type });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  // ----------------------------------------------------------------- theme
  function applyTheme(t) {
    if (t) document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  applyTheme(store.get("specx-theme"));
  $("#themeBtn").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    applyTheme(next);
    store.set("specx-theme", next);
  });

  // ----------------------------------------------------------------- health
  async function loadHealth() {
    const pill = $("#llmStatus");
    try {
      const h = await api("/api/health");
      state.health = h;
      pill.classList.toggle("on", h.llm);
      pill.classList.toggle("off", !h.llm);
      $(".label", pill).textContent = h.llm ? h.model : "offline extractor (no API key)";
      $("#traceModel").textContent = h.llm ? h.model.split("/").pop() : "rule-based";
      $("#embedModel").textContent = h.embed_model.split("/").pop();
      $("#rerankModel").textContent = h.rerank_model.split("/").pop();
      $("#maxMb").textContent = h.max_upload_pages ? `${h.max_upload_pages} pages / ${h.max_upload_mb} MB` : `${h.max_upload_mb} MB`;
      $("#dropzone").hidden = h.allow_upload === false;
      if (h.reranker === false) $('#trace li[data-step="rerank"] span').textContent = "off (RRF order)";
    } catch {
      pill.classList.add("off");
      $(".label", pill).textContent = "server unreachable";
    }
  }

  // ----------------------------------------------------------------- library
  async function loadDocs(selectId) {
    try {
      state.docs = await api("/api/documents");
    } catch (e) {
      toast(`Could not load manuals: ${e.message}`);
      state.docs = [];
    }
    renderDocs();
    // explicit choice > current selection > configured default manual > first in list
    const want = selectId || state.doc?.doc_id || state.health?.default_doc;
    const pick = state.docs.find((d) => d.doc_id === want || d.filename === want) || state.docs[0];
    if (pick) selectDoc(pick.doc_id);
  }

  function renderDocs() {
    const ul = $("#docs");
    $("#docsEmpty").hidden = state.docs.length > 0;
    ul.innerHTML = state.docs.map((d) => {
      const skipped = Object.values(d.skipped_pages || {}).reduce((a, b) => a + b, 0);
      return `<li><button class="doc ${state.doc?.doc_id === d.doc_id ? "active" : ""}" data-id="${esc(d.doc_id)}" title="${esc(d.filename)}">
        <span class="doc-name">${esc(d.filename)}</span>
        ${d.title ? `<span class="doc-title">${esc(d.title)}</span>` : ""}
        <span class="doc-meta">
          <span><b>${fmtInt(d.pages)}</b> pp</span>
          <span><b>${fmtInt(d.table_chunks)}</b> table</span>
          <span><b>${fmtInt(d.text_chunks)}</b> text</span>
          ${d.has_outline ? `<span class="doc-badge">outline</span>` : `<span class="doc-badge">inferred headings</span>`}
          ${skipped ? `<span class="doc-badge" title="TOC/index, image-only or blank pages">${skipped} skipped</span>` : ""}
        </span>
        <span class="doc-owner">${d.kind === "upload"
          ? `<span class="own mine">Your upload</span>${d.expires_at ? `<span>deleted ${esc(untilText(d.expires_at))}</span>` : ""}`
          : `<span class="own">Demo manual</span>`}</span>
      </button>${d.kind === "upload" ? `<button class="doc-del" data-del="${esc(d.doc_id)}" title="Delete this upload" aria-label="Delete ${esc(d.filename)}">✕</button>` : ""}</li>`;
    }).join("");
  }

  function untilText(iso) {
    const mins = Math.max(0, Math.round((new Date(iso) - Date.now()) / 60000));
    return mins >= 90 ? `in ${Math.round(mins / 60)} h` : mins > 0 ? `in ${mins} min` : "soon";
  }

  $("#docs").addEventListener("click", async (e) => {
    const del = e.target.closest("[data-del]");
    if (del) {
      const d = state.docs.find((x) => x.doc_id === del.dataset.del);
      if (!d || !confirm(`Delete your upload "${d.filename}"?`)) return;
      try {
        await api(`/api/documents/${encodeURIComponent(d.doc_id)}`, { method: "DELETE" });
        if (state.doc?.doc_id === d.doc_id) state.doc = null;
        toast(`Deleted ${d.filename}`);
        loadDocs();
      } catch (err) { toast(err.message); }
      return;
    }
    const b = e.target.closest(".doc");
    if (b) selectDoc(b.dataset.id);
  });

  function selectDoc(id) {
    state.doc = state.docs.find((d) => d.doc_id === id) || null;
    $$(".doc").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
    const label = $("#activeDoc");
    label.textContent = state.doc ? state.doc.filename : "no manual selected";
    label.classList.toggle("set", !!state.doc);
  }

  // ----------------------------------------------------------------- upload
  const dz = $("#dropzone");
  $("#fileInput").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });

  const STAGE_ORDER = ["parse", "clean", "chunk", "embed"];
  const STAGE_SPAN = { queued: [0, 0], parse: [0, 0.4], clean: [0.4, 0.45], chunk: [0.45, 0.55], embed: [0.55, 0.99], done: [1, 1] };
  const STAGE_NOTE = {
    queued: "Waiting for the indexer…",
    parse: "Reading tables first, then prose outside them.",
    clean: "Learning running headers, footers and TOC pages.",
    chunk: "Building table-row and heading-bounded chunks.",
    embed: "Embedding chunks and building FAISS + BM25 indexes (a few minutes per 100 pages on the free demo server).",
    done: "Indexed.",
  };

  async function upload(file) {
    if (!/\.pdf$/i.test(file.name)) return toast("Please choose a PDF file.");
    const box = $("#ingest");
    box.hidden = false;
    box.classList.remove("error");
    $("#ingestFile").textContent = file.name;
    setStage("queued", 0);
    $("#ingestNote").textContent = "Uploading…";
    const fd = new FormData();
    fd.append("file", file);
    try {
      const { job_id, pages } = await api("/api/documents", { method: "POST", body: fd });
      if (pages) $("#ingestFile").textContent = `${file.name} · ${pages} pp`;
      pollJob(job_id);
    } catch (e) {
      box.classList.add("error");
      $("#ingestNote").textContent = e.message;
    } finally {
      $("#fileInput").value = "";
    }
  }

  function setStage(stage, frac) {
    const [a, b] = STAGE_SPAN[stage] || [0, 0];
    const pct = Math.round((a + (b - a) * (frac || 0)) * 100);
    $("#ingestPct").textContent = `${pct}%`;
    const idx = STAGE_ORDER.indexOf(stage);
    $$("#stages li").forEach((li, i) => {
      li.classList.toggle("done", stage === "done" || (idx >= 0 && i < idx));
      li.classList.toggle("on", i === idx);
    });
    $("#ingestNote").textContent = STAGE_NOTE[stage] || "";
  }

  function pollJob(id) {
    const tick = async () => {
      try {
        const j = await api(`/api/jobs/${id}`);
        if (j.status === "error") {
          $("#ingest").classList.add("error");
          $("#ingestNote").textContent = `Failed: ${j.error}`;
          return;
        }
        setStage(j.stage, j.progress);
        if (j.status === "done") {
          toast(`Indexed ${j.doc.filename} · ${fmtInt(j.doc.chunks)} chunks`);
          await loadDocs(j.doc.doc_id);
          setTimeout(() => ($("#ingest").hidden = true), 1800);
          return;
        }
      } catch (e) {
        $("#ingestNote").textContent = `Lost contact: ${e.message}`;
      }
      setTimeout(tick, 700);
    };
    tick();
  }

  // ----------------------------------------------------------------- query
  $("#queryForm").addEventListener("submit", (e) => { e.preventDefault(); runQuery(); });

  const trace = {
    reset() { $$("#trace li").forEach((li) => { li.className = ""; $("i", li).textContent = ""; }); },
    set(step, cls, txt) {
      const li = $(`#trace li[data-step="${step}"]`);
      li.className = cls;
      if (txt != null) $("i", li).textContent = txt;
    },
  };
  let traceTimers = [];

  async function runQuery() {
    const q = $("#queryInput").value.trim();
    if (!state.doc) return toast("Upload or select a manual first.");
    if (q.length < 2) return $("#queryInput").focus();
    if (state.busy) return;
    state.busy = true;
    $("#runBtn").disabled = true;
    closeEvidence();

    // optimistic stage animation; real timings replace it when the response lands
    trace.reset();
    trace.set("retrieve", "run");
    traceTimers.forEach(clearTimeout);
    traceTimers = [
      setTimeout(() => { trace.set("retrieve", "done"); trace.set("rerank", "run"); }, 450),
      setTimeout(() => { trace.set("rerank", "done"); trace.set("extract", "run"); }, 900),
    ];

    $("#emptyState").hidden = true;
    $("#sheet").hidden = false;
    $("#sheetQuery").textContent = q;
    $("#specCount").textContent = "…";
    $("#notice").hidden = true;
    $("#viewCards").innerHTML = `<div class="skel"><div></div><div></div><div></div></div>`;
    showView("cards", true);

    try {
      const r = await api("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: state.doc.doc_id, query: q }),
      });
      traceTimers.forEach(clearTimeout);
      const t = r.timings_ms || {};
      trace.set("retrieve", "done", fmtMs(t.retrieve));
      trace.set("rerank", "done", state.health?.reranker === false ? "–" : fmtMs(t.rerank));
      trace.set("extract", "done", r.cached ? "cached" : fmtMs(t.extract));
      trace.set("validate", "done", `${r.specs.length}/${r.specs.length + r.rejected.length}`);
      state.result = r;
      state.selected = -1;
      renderResult();
    } catch (e) {
      traceTimers.forEach(clearTimeout);
      $$("#trace li.run").forEach((li) => (li.className = "fail"));
      $("#specCount").textContent = "0";
      $("#viewCards").innerHTML = `<div class="none"><strong>Extraction failed</strong><p>${esc(e.message)}</p></div>`;
    } finally {
      state.busy = false;
      $("#runBtn").disabled = false;
    }
  }

  // ----------------------------------------------------------------- render
  function rows() {
    const specs = state.result?.specs || [];
    if (state.requiredOnly) {
      // 4-field output: fold variant / condition into the component so rows stay distinguishable
      return specs.map((s) => {
        const q = [s.variant, s.condition].filter((x) => x && !s.component.toLowerCase().includes(x.toLowerCase()));
        return { component: q.length ? `${s.component} (${q.join(", ")})` : s.component, spec_type: s.spec_type, value: s.value, unit: s.unit || "" };
      });
    }
    return specs.map((s) => Object.fromEntries(FULL.map((c) => [c, s[c] ?? (c === "unit" ? "" : null)])));
  }

  function renderResult() {
    const r = state.result;
    $("#specCount").textContent = r.specs.length;

    const notice = $("#notice");
    const parts = [];
    if (r.mode === "offline") parts.push("Offline rule-based extractor in use (no LLM key configured) — table rows and simple patterns only.");
    if (r.rejected.length) {
      parts.push(`<details><summary>${r.rejected.length} candidate${r.rejected.length > 1 ? "s" : ""} rejected by the grounding check</summary><ul>${
        r.rejected.map((x) => `<li>${esc(x.item?.component || "?")} = ${esc(x.item?.value || "?")} — ${esc(x.reason)}</li>`).join("")
      }</ul></details>`);
    }
    notice.innerHTML = parts.join(" ");
    notice.hidden = parts.length === 0;
    notice.classList.toggle("muted", r.mode !== "offline" && !!r.rejected.length);

    renderCards();
    renderJson();
    renderContext();
  }

  function valueBlock(s) {
    const long = String(s.value).length > 12;
    return `<span class="v ${long ? "long" : ""}">${esc(s.value)}</span>${s.unit ? `<span class="u">${esc(s.unit)}</span>` : ""}
      ${s.alt_value ? `<span class="alt">= ${esc(s.alt_value)} ${esc(s.alt_unit || "")}</span>` : ""}`;
  }

  function confBars(c) {
    const n = c == null ? 0 : c >= 0.95 ? 3 : c >= 0.75 ? 2 : 1;
    return `<span class="conf" title="Grounding confidence ${c ?? "n/a"}">${[0, 1, 2].map((i) => `<i class="${i < n ? "on" : ""}"></i>`).join("")}</span>`;
  }

  function renderCards() {
    const r = state.result;
    const box = $("#viewCards");
    if (!r.specs.length) {
      const name = esc(state.doc?.filename || "this manual");
      const others = state.docs.filter((d) => d.doc_id !== state.doc?.doc_id);
      const note = r.note
        ? `<div class="ref-note"><span class="ref-tag">Manual says</span><p>${esc(r.note.text)}</p>
             <a href="${withVisitor(`/api/documents/${esc(state.doc.doc_id)}/source.pdf`)}#page=${r.note.page}" target="_blank" rel="noopener">p.${r.note.page} ↗</a></div>`
        : "";
      box.innerHTML = `<div class="none">
        <strong>No value for this in <span class="mono">${name}</span></strong>
        <p>None of the retrieved passages state it, so nothing was invented.${r.note ? " The manual points elsewhere:" : " Try different wording, or check the retrieved context below."}</p>
        ${note}
        ${others.length ? `<p class="try-other">Search another manual: ${others.map((d) => `<button type="button" class="tool" data-switch="${esc(d.doc_id)}">${esc(d.filename)}</button>`).join(" ")}</p>` : ""}
      </div>`;
      return;
    }
    box.innerHTML = r.specs.map((s, i) => `
      <article class="spec ${i === state.selected ? "sel" : ""}" data-i="${i}" tabindex="0">
        <div class="spec-idx">${String(i + 1).padStart(2, "0")}</div>
        <div class="spec-main">
          <div class="spec-comp">${esc(s.component)}</div>
          <div class="spec-tags">
            <span class="t type ${slug(s.spec_type)}">${esc(s.spec_type)}</span>
            ${s.variant ? `<span class="t plain">${esc(s.variant)}</span>` : ""}
            ${s.condition ? `<span class="t plain">${esc(s.condition)}</span>` : ""}
            ${s.section ? `<span class="t plain" title="${esc(s.section)}">${esc(s.section.length > 42 ? s.section.slice(0, 40) + "…" : s.section)}</span>` : ""}
          </div>
        </div>
        <div class="spec-val">${valueBlock(s)}</div>
        <div class="spec-src">
          <button class="page-btn" type="button" tabindex="-1">p.${esc(s.page ?? "?")}</button>
          ${confBars(s.confidence)}
        </div>
      </article>`).join("");
  }

  function renderJson() {
    $("#viewJson").textContent = JSON.stringify(rows(), null, 2);
  }

  function renderContext() {
    const chunks = state.result.chunks || [];
    $("#ctxCount").textContent = `· top ${chunks.length}`;
    $("#ctxList").innerHTML = chunks.map((h, i) => {
      const c = h.chunk;
      return `<li class="ctx" data-i="${i}">
        <div class="ctx-head">
          <span class="score">#${i + 1} · ${h.score.toFixed(2)}</span>
          <span class="rk" title="Rank in dense (embedding) search">D ${h.dense_rank != null ? h.dense_rank + 1 : "–"}</span>
          <span class="rk" title="Rank in BM25 keyword search">S ${h.sparse_rank != null ? h.sparse_rank + 1 : "–"}</span>
          <span>${c.chunk_type === "table" ? "table rows" : c.source === "ocr" ? "ocr text" : "text"}</span>
          <span>p.${c.page}${c.page_end !== c.page ? "–" + c.page_end : ""}</span>
        </div>
        ${c.heading_path ? `<div class="ctx-path">${esc(c.heading_path)}</div>` : ""}
        <pre class="ctx-text">${esc(c.text)}</pre>
      </li>`;
    }).join("");
  }

  $("#ctxList").addEventListener("click", (e) => e.target.closest(".ctx")?.classList.toggle("open"));

  // views
  function showView(v, silent) {
    if (v !== "json") v = "cards"; // also maps a remembered, now-removed "table" view
    state.view = v;
    $$("#viewSeg button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
    $("#viewCards").hidden = v !== "cards";
    $("#viewJson").hidden = v !== "json";
    if (!silent) store.set("specx-view", v);
  }
  $("#viewSeg").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-view]");
    if (b) showView(b.dataset.view);
  });
  $("#requiredOnly").addEventListener("change", (e) => {
    state.requiredOnly = e.target.checked;
    if (state.result) renderJson();
  });

  // exports
  function csv() {
    const data = rows();
    const cols = state.requiredOnly ? REQUIRED : FULL;
    const cell = (v) => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    return [cols.join(","), ...data.map((r) => cols.map((c) => cell(r[c])).join(","))].join("\n");
  }
  const fileStem = () => `specs-${slug($("#sheetQuery").textContent).slice(0, 40) || "query"}`;
  $("#exportCsv").addEventListener("click", () => state.result && download(`${fileStem()}.csv`, csv(), "text/csv"));
  $("#exportJson").addEventListener("click", () => state.result && download(`${fileStem()}.json`, JSON.stringify(rows(), null, 2), "application/json"));
  $("#copyJson").addEventListener("click", async () => {
    if (!state.result) return;
    try { await navigator.clipboard.writeText(JSON.stringify(rows(), null, 2)); toast("JSON copied"); }
    catch { toast("Clipboard unavailable"); }
  });

  // ----------------------------------------------------------------- evidence
  function onPick(e) {
    const sw = e.target.closest("[data-switch]");
    if (sw) { selectDoc(sw.dataset.switch); runQuery(); return; }
    const el = e.target.closest("[data-i]");
    if (el) openEvidence(Number(el.dataset.i));
  }
  $("#viewCards").addEventListener("click", onPick);
  $("#viewCards").addEventListener("keydown", (e) => { if (e.key === "Enter") onPick(e); });

  function markValue(text, value) {
    const t = esc(text);
    if (!value) return t;
    const v = esc(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return t.replace(new RegExp(`(^|[^\\w.])(${v})(?![\\w])`, "g"), "$1<mark>$2</mark>");
  }

  function openEvidence(i) {
    const r = state.result;
    const s = r?.specs[i];
    if (!s) return;
    state.selected = i;
    $$(".spec").forEach((el) => el.classList.toggle("sel", Number(el.dataset.i) === i));
    const hit = r.chunks.find((h) => h.chunk.id === s.chunk_id);
    const c = hit?.chunk;
    const docId = state.doc.doc_id;
    const hl = [s.source_quote, s.component, String(s.value).length >= 3 ? s.value : null]
      .filter(Boolean).map((x) => `hl=${encodeURIComponent(String(x).slice(0, 120))}`).join("&");
    const img = s.page ? withVisitor(`/api/documents/${docId}/pages/${s.page}.png?${hl}`) : null;

    $("#evBody").innerHTML = `
      <div>
        <p class="ev-comp">${esc(s.component)}</p>
        <div class="spec-tags" style="margin-top:8px">
          <span class="t type ${slug(s.spec_type)}">${esc(s.spec_type)}</span>
          ${s.variant ? `<span class="t plain">${esc(s.variant)}</span>` : ""}
          ${s.condition ? `<span class="t plain">${esc(s.condition)}</span>` : ""}
        </div>
      </div>
      <div class="ev-value"><span class="v">${esc(s.value)}</span>${s.unit ? `<span class="u">${esc(s.unit)}</span>` : ""}
        ${s.alt_value ? `<span class="alt">= ${esc(s.alt_value)} ${esc(s.alt_unit || "")}</span>` : ""}</div>
      <dl class="kv">
        <dt>Page</dt><dd>${esc(s.page ?? "–")}</dd>
        <dt>Section</dt><dd>${esc(s.section || c?.heading_path || "–")}</dd>
        <dt>Source</dt><dd>${esc(s.source)}${c ? ` · ${esc(c.id)}` : ""}</dd>
        <dt>Grounding</dt><dd>${s.confidence != null ? `${Math.round(s.confidence * 100)}% ${confBars(s.confidence)}` : "–"}</dd>
      </dl>
      ${s.source_quote ? `<p class="sub">Quoted by the model</p><blockquote class="quote">${markValue(s.source_quote, s.value)}</blockquote>` : ""}
      ${c ? `<p class="sub">Cited passage</p><blockquote class="quote">${markValue(c.text, s.value)}</blockquote>` : ""}
      ${img ? `<p class="sub">Source page</p><div class="page-frame loading"><span class="pf-label">PAGE ${esc(s.page)}</span><img alt="Manual page ${esc(s.page)} with the specification highlighted" src="${img}"></div>` : ""}
      <div class="ev-links">
        ${s.page ? `<a href="${withVisitor(`/api/documents/${docId}/source.pdf`)}#page=${s.page}" target="_blank" rel="noopener">Open PDF at p.${esc(s.page)} ↗</a>` : ""}
      </div>`;
    const imgEl = $("#evBody img");
    if (imgEl) {
      imgEl.addEventListener("load", () => imgEl.parentElement.classList.remove("loading"));
      imgEl.addEventListener("error", () => { imgEl.parentElement.classList.remove("loading"); imgEl.replaceWith(Object.assign(document.createElement("p"), { className: "ev-empty", textContent: "Page preview unavailable." })); });
    }
    const ev = $("#evidence");
    ev.classList.add("open");
    ev.setAttribute("aria-hidden", "false");
  }

  function closeEvidence() {
    const ev = $("#evidence");
    ev.classList.remove("open");
    ev.setAttribute("aria-hidden", "true");
    state.selected = -1;
    $$(".spec.sel").forEach((el) => el.classList.remove("sel"));
  }
  $("#evClose").addEventListener("click", closeEvidence);

  // ----------------------------------------------------------------- keys
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeEvidence();
    if (e.key === "/" && document.activeElement !== $("#queryInput")) { e.preventDefault(); $("#queryInput").focus(); }
    if (state.result && (e.key === "ArrowDown" || e.key === "ArrowUp") && $("#evidence").classList.contains("open")) {
      e.preventDefault();
      const n = state.result.specs.length;
      openEvidence((state.selected + (e.key === "ArrowDown" ? 1 : -1) + n) % n);
    }
  });

  // ----------------------------------------------------------------- boot
  const savedView = store.get("specx-view");
  if (savedView) showView(savedView, true);

  // Open on the default manual with the default question already answered (served from
  // the server's answer cache after the first run, so it costs no LLM call per visit).
  (async () => {
    await loadHealth();
    await loadDocs();
    const q = state.health?.default_query;
    if (q && state.doc) {
      $("#queryInput").value = q;
      runQuery();
    }
  })();
})();
