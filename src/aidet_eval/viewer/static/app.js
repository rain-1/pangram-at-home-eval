// aidet_eval local viewer. Dataset text is untrusted: it is only ever inserted via textContent.
// Offsets are Unicode code points (Python str indices): text is split with Array.from(), never by UTF-16 index.
"use strict";

const S = { runs: [], run: null, summary: null, summaries: {}, model: null, protocol: null, op: "human_fpr_1pct", hasFull: false };
const $ = (id) => document.getElementById(id);

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : String(v));
  }
  for (const k of kids.flat()) {
    if (k === null || k === undefined || k === false) continue;
    e.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return e;
}
function throttled(fn) { // run at most once per animation frame (keeps hover cheap)
  let pending = null;
  return (ev) => { if (pending) { pending = ev; return; } pending = ev; requestAnimationFrame(() => { const e = pending; pending = null; fn(e); }); };
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
async function api(path, params) {
  const u = new URL(path, location.origin);
  for (const [k, v] of Object.entries(params || {})) if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, v);
  const r = await fetch(u);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function select(options, value, onchange, attrs) {
  const s = el("select", attrs || {});
  for (const o of options) {
    const [v, label] = Array.isArray(o) ? o : [o, o];
    const opt = el("option", { value: v }, label);
    if (v === value) opt.selected = true;
    s.appendChild(opt);
  }
  s.addEventListener("change", () => onchange(s.value));
  return s;
}
function table(headers, rows) {
  return el("div", { class: "tablewrap" },
    el("table", {}, el("thead", {}, el("tr", {}, headers.map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c) => (c instanceof Node && c.tagName === "TD") ? c : el("td", {}, c)))))));
}
const pct = (x, d = 1) => (x === null || x === undefined) ? "n/a" : (100 * x).toFixed(x > 0 && x < 0.01 ? Math.max(d, 2) : d) + "%";
const pnum = (v) => v.toFixed(v > 0 && v < 1 ? 2 : 1);

// Metric cell: value as percentage with counts and CI; unavailable values show their reason, never 0.
function metricTd(m) {
  if (!m) return el("td", { class: "na", title: "No materialised cell: the requested population is empty (see shortfalls under Run provenance) or the slice was not predeclared" }, "no data (empty cell / shortfall)");
  if (m.value === null || m.value === undefined) return el("td", { class: "na", title: m.reason || "" }, `n/a (${m.reason || m.status})`);
  const ci = m.ci;
  let txt;
  if (m.unit === "proportion") {
    txt = `${pnum(m.percent)}% (${m.numerator}/${m.denominator})`;
    if (ci) txt += ` [${pnum(100 * ci.lower)}, ${pnum(100 * ci.upper)}]`;
  } else {
    txt = m.value.toFixed(3) + (ci ? ` [${ci.lower.toFixed(3)}, ${ci.upper.toFixed(3)}]` : "");
  }
  const td = el("td", { class: "num", title: ci ? `${ci.method}; groups=${ci.n_groups}; ${ci.assumptions}` : "" }, txt);
  if (m.status === "provisional") td.appendChild(el("span", { class: "badge prov" }, "provisional"));
  if (ci && ci.degenerate_all_zero) td.appendChild(el("span", { class: "badge", title: "all bootstrap replicates were zero; not proof of a 0% rate" }, "degenerate CI"));
  if (m.failure_bounds) td.appendChild(el("div", { class: "muted" }, `bounds incl. failures: ${pct(m.failure_bounds.lower)}–${pct(m.failure_bounds.upper)}`));
  if (m.group_level_any_error_upper_95) td.appendChild(el("div", { class: "muted", title: m.group_level_any_error_upper_95.note }, `group-level 95% upper: ${pct(m.group_level_any_error_upper_95.value, 2)}`));
  return td;
}

// ------------------------------------------------------------------ data access
function cells() { return S.summary ? S.summary.cells : []; }
function sliceKey(sl) { return JSON.stringify(Object.keys(sl).sort().reduce((o, k) => (o[k] = sl[k], o), {})); }
function findCell(panel, slice, task, op) {
  const key = sliceKey({ ...slice, view: slice.view || "all" });
  return cells().find((c) => c.model_id === S.model && c.protocol_id === S.protocol && c.panel_id === panel &&
    (c.task_id || null) === (task || null) && c.operating_point_id === (op === undefined ? S.op : op) && sliceKey(c.slice) === key);
}

async function loadRun(name) {
  S.run = name;
  const info = S.runs.find((r) => r.run === name);
  S.hasFull = !!(info && info.has_full);
  if (!S.summaries[name]) S.summaries[name] = await api("/api/summary", { run: name });
  S.summary = S.summaries[name];
  const models = S.summary.models.map((m) => m.model_id);
  const protos = S.summary.protocols.map((p) => p.protocol_id);
  if (!models.includes(S.model)) S.model = models[0];
  if (!protos.includes(S.protocol)) S.protocol = protos.find((p) => p === "shared_windows_v1") || protos[0];
  renderBanner();
  renderAll();
}

function renderBanner() {
  const b = clear($("banner"));
  const s = S.summary;
  const msgs = [];
  if (s.data_origin !== "measured") msgs.push(["bad", `data_origin = ${s.data_origin}: NOT a measured baseline result`]);
  if (s.status !== "complete") msgs.push(["bad", `Run status: ${s.status}. Some results are partial, provisional or unavailable.`]);
  const prov = s.operating_points.filter((o) => o.status !== "ok");
  if (prov.length) msgs.push(["", `${prov.length} operating point(s) not ok: ${prov.map((o) => o.calibration_id + "=" + o.status).join(", ")}`]);

  for (const [cls, m] of msgs) b.appendChild(el("div", { class: cls }, m));
}

function selectors(onchange) {
  const ops = [...new Set(S.summary.operating_points.map((o) => o.operating_point_id))];
  return el("div", { class: "controls" },
    el("label", {}, "Model ", select(S.summary.models.map((m) => m.model_id), S.model, (v) => { S.model = v; onchange(); })),
    el("label", {}, "Protocol ", select(S.summary.protocols.map((p) => p.protocol_id), S.protocol, (v) => { S.protocol = v; onchange(); })),
    el("label", {}, "Operating point ", select(ops, S.op, (v) => { S.op = v; onchange(); })));
}

// ------------------------------------------------------------------ scorecard (any number of detectors)
const PANEL_NAMES = {
  audited_human_core: "Human writing (pre-LLM, 5 domains)", student_human: "Student essays (human)",
  modern_human_instructions: "Instruction responses (human, No Robots)", benchmark_source_matched_human_references: "Human references from AI benchmarks",
  fully_ai_core: "Fully AI-written", fully_ai_noncore: "Fully AI-written (plot summaries)",
  controlled_replacement: "AI sentences inserted into human text", controlled_replacement_noncore: "AI sentences inserted (plot summaries)",
  heterogeneous_block_replacement: "AI blocks inserted into human text", heterogeneous_matched_human_controls: "Originals of the block-insertion texts (human)",
  aitdna_real_cowriting: "Real human+AI co-writing sessions", beemo_matched_editing: "AI answers and expert-edited AI answers",
  lamp_human_edited_ai: "AI fiction drafts and writer-edited versions", editlens_in_family_reference: "EditLens test set (baselines' own family)",
  damasha_tagged_mixed_unknown_mechanism: "DAMASHA mixtures (mechanism unknown)", calibration_human: "Calibration humans",
};
const pname = (p) => PANEL_NAMES[p] || p;
const PROTO_NAMES = { shared_windows_v1: "Overlapping windows, highest score (primary)", native_truncate_v1: "Official model behaviour (first 512/1024 tokens)" };
const SMALL_N = 30;
const short = (m) => m.replace(/^open_pangram_/, "").replace(/_s(\d)_/, " s$1 ").replace(/_/g, " ");
const allModels = () => S.summary.models.map((m) => m.model_id);
// colour follows the detector (its position in the run's full list), never its rank among the visible ones
const mclass = (m) => `m${(allModels().indexOf(m) % 8) + 1}`;
function models() {
  const all = allModels();
  if (!S._shown || !S._shownRun || S._shownRun !== S.run) { S._shown = new Set(all); S._shownRun = S.run; }
  return all.filter((m) => S._shown.has(m));
}
const swatch = (m) => el("i", { class: `sw ${mclass(m)}` });

function cellFor(model, panel, slice, metric) {
  const key = sliceKey({ ...(slice || {}), view: "all" });
  const c = cells().find((c) => c.model_id === model && c.protocol_id === S.protocol && c.panel_id === panel && !c.task_id &&
    c.operating_point_id === S.op && sliceKey(c.slice) === key);
  return c ? c.metrics[metric] : null;
}
// 8 discrete colour steps (0 = best ... 7 = worst); bins chosen where the decisions are
const BINS = {
  fpr: [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25],   // wrongly flagged: rate <= edge -> step index
  recall: [0.95, 0.90, 0.80, 0.65, 0.50, 0.35, 0.20],   // caught: rate >= edge -> step index
};
const STEP_NAMES = ["dark green", "green", "light green", "off-white", "light orange", "orange", "red", "dark red"];
function colourStep(rate, kind) {
  if (rate === null || rate === undefined || !kind) return null;
  const e = BINS[kind];
  for (let i = 0; i < e.length; i++) if (kind === "fpr" ? rate <= e[i] : rate >= e[i]) return i;
  return 7;
}
function tint(td, step) {
  if (step === null) return;
  td.classList.add("cstep", `c${step}`);
}
// row-relative colouring: compare the detectors within one row (5 distinct hues, best -> worst)
const REL_NAMES = ["best in row", "near best", "middle", "behind", "worst in row"];
function colourRow(tds) {
  if ((S.colourMode || "row") !== "row") return;
  const vals = tds.filter((td) => td.dataset && td.dataset.rate !== undefined && td.dataset.kind).map((td) => {
    const r = parseFloat(td.dataset.rate), k = td.dataset.kind;
    return { td, k, v: k === "fpr" ? Math.log10(Math.max(r, 0.0005)) : r };  // FPR on a log scale
  });
  if (vals.length < 2) return;
  const k = vals[0].k;
  const best = k === "fpr" ? Math.min(...vals.map((x) => x.v)) : Math.max(...vals.map((x) => x.v));
  const worst = k === "fpr" ? Math.max(...vals.map((x) => x.v)) : Math.min(...vals.map((x) => x.v));
  const minSpread = k === "fpr" ? Math.log10(2) : 0.05;  // differences smaller than this stay close in colour
  const span = Math.max(Math.abs(worst - best), minSpread);
  for (const x of vals) {
    const t = Math.abs(x.v - best) / span;
    const step = t < 0.15 ? 0 : t < 0.4 ? 1 : t < 0.6 ? 2 : t < 0.85 ? 3 : 4;
    x.td.classList.add("rstep", `r${step}`);
    x.td.title += ` · ${REL_NAMES[step]}`;
  }
}
function colourKey() {
  if ((S.colourMode || "row") === "row")
    return el("div", { class: "colourkey" }, el("div", { class: "ckrow" }, el("span", { class: "cklabel" }, "Colour compares detectors within each row:"),
      ...REL_NAMES.map((n, i) => el("span", { class: `ckcell rstep r${i}` }, n)),
      el("span", { class: "muted ckhint" }, "(false alarms compared on a log scale; differences under 5 pp, or under 2× for false alarms, stay close in colour)")));
  const fmt = (v) => `${pnum(100 * v)}%`;
  const row = (kind, label) => el("div", { class: "ckrow" }, el("span", { class: "cklabel" }, label),
    ...STEP_NAMES.map((n, i) => { const e = BINS[kind];
      const txt = kind === "fpr" ? (i === 0 ? `≤ ${fmt(e[0])}` : i === 7 ? `> ${fmt(e[6])}` : `≤ ${fmt(e[i])}`)
                                 : (i === 0 ? `≥ ${fmt(e[0])}` : i === 7 ? `< ${fmt(e[6])}` : `≥ ${fmt(e[i])}`);
      return el("span", { class: `ckcell cstep c${i}`, title: n }, txt); }));
  return el("div", { class: "colourkey" }, row("fpr", "Wrongly flagged (lower is better)"), row("recall", "Caught (higher is better)"));
}
function rateTd(rate, k, n, ci, reason, kind, compact) {
  const td = el("td", { class: "num" + (compact ? " compact" : "") });
  if (rate === null || rate === undefined || !n) { td.appendChild(el("span", { class: "na", title: reason || "no documents of this kind here" }, "—")); return td; }
  const small = n < SMALL_N;
  if (small) td.classList.add("lown");
  const degenerate = ci && ci[0] === ci[1];
  td.title = `${k} of ${n}` + (ci ? (degenerate ? " · degenerate interval (every resample gave the same value; not certainty)" : ` · 95% interval ${pnum(100 * ci[0])}–${pnum(100 * ci[1])}%`) : "") + (small ? ` · only ${n} documents: too few to trust` : "");
  const ciTxt = !ci || compact ? "" : degenerate ? " · no interval*" : ` · ${pnum(100 * ci[0])}–${pnum(100 * ci[1])}`;
  td.append(el("div", { class: "big" }, `${pnum(100 * rate)}%`), el("div", { class: "sub" }, `${k}/${n}` + ciTxt));
  if (small) td.appendChild(el("div", { class: "sub smalln" }, "small n"));
  else {
    td.dataset.rate = String(rate); td.dataset.kind = kind || "";
    if ((S.colourMode || "row") === "absolute") tint(td, colourStep(rate, kind));
  }
  return td;
}
const kindOf = (metric) => (metric === "human_fpr" || (metric || "").endsWith("_fpr")) ? "fpr" : "recall";
const metricCellTd = (m, metric) => m && m.value !== null && m.value !== undefined
  ? rateTd(m.value, m.numerator, m.denominator, m.ci ? [m.ci.lower, m.ci.upper] : null, null, kindOf(metric))
  : rateTd(null, 0, 0, null, m ? (m.reason || m.status) : "no data");
// rows = things, columns = visible detectors
function detTable(firstHeader, rowLabels, cellFn, title, note) {
  const ms = models();
  const t = el("table", { class: "clean dettable" },
    el("thead", {}, el("tr", {}, el("th", {}, firstHeader), ...ms.map((m) => el("th", { class: "num det", title: m }, swatch(m), short(m))))),
    el("tbody", {}, rowLabels.map(([key, label]) => { const tds = ms.map((m) => cellFn(m, key)); colourRow(tds);
      return el("tr", {}, el("td", {}, label), ...tds); })));
  return el("div", { class: "detblock" }, title ? el("h3", {}, title) : null, note ? el("p", { class: "muted small" }, note) : null, el("div", { class: "tablewrap" }, t));
}

function renderScorecard() {
  const root = clear($("scorecard"));
  const ops = [...new Set(S.summary.operating_points.map((o) => o.operating_point_id))];
  const opLabel = (o) => o === "human_fpr_1pct" ? "1% of humans flagged (strict)" : o === "human_fpr_5pct" ? "5% of humans flagged" : o;
  const run = S.runs.find((r) => r.run === S.run) || {};
  if (run.profile && run.profile !== "full") root.appendChild(el("div", { class: "explore" }, el("b", { class: "tag" }, "SMOKE TEST RUN: "),
    `only ${run.n_evaluation} documents — a software check, not a result. Pick a full run above.`));
  models();
  root.appendChild(el("div", { class: "controls" },
    el("label", {}, "How documents are read ", select(S.summary.protocols.map((p) => [p.protocol_id, PROTO_NAMES[p.protocol_id] || p.protocol_id]), S.protocol, (v) => { S.protocol = v; renderScorecard(); })),
    el("label", {}, "False-alarm budget ", select(ops.map((o) => [o, opLabel(o)]), S.op, (v) => { S.op = v; renderScorecard(); })),
    el("label", {}, "Cell colour ", select([["row", "Compare detectors in each row"], ["absolute", "Absolute quality bands"]], S.colourMode || "row", (v) => { S.colourMode = v; renderScorecard(); }))));
  root.appendChild(el("div", { class: "controls detpicker" }, el("span", { class: "muted" }, "Detectors:"),
    ...allModels().map((m) => el("label", { class: "chk", title: (S.summary.models.find((x) => x.model_id === m) || {}).display_name || m },
      el("input", { type: "checkbox", checked: S._shown.has(m), onchange: (e) => { e.target.checked ? S._shown.add(m) : S._shown.delete(m); renderScorecard(); } }),
      swatch(m), short(m)))));
  const thr = models().map((m) => S.summary.operating_points.find((o) => o.model_id === m && o.protocol_id === S.protocol && o.operating_point_id === S.op)).filter(Boolean);
  root.appendChild(el("p", { class: "lede" }, "Each detector gets its own threshold, set so that ", el("b", {}, opLabel(S.op).split(" (")[0]),
    ` of ${thr[0] ? thr[0].n_scored.toLocaleString() : "?"} held-out human calibration documents are flagged. Evaluation: ${run.n_evaluation ? run.n_evaluation.toLocaleString() : "?"} documents. `,
    el("span", { class: "muted" }, "Thresholds: ", ...thr.flatMap((o) => [swatch(o.model_id), `${short(o.model_id)} > ${o.threshold.toPrecision(3)}  `]))));

  root.appendChild(colourKey());
  const heads = [["audited_human_core|human_fpr", "Human writing wrongly flagged"], ["student_human|human_fpr", "Student essays wrongly flagged"],
    ["fully_ai_core|fully_ai_recall", "Fully AI texts caught"], ["controlled_replacement|mixed_recall", "AI sentences inserted: caught"],
    ["heterogeneous_block_replacement|mixed_recall", "AI blocks inserted: caught"], ["aitdna_real_cowriting|mixed_recall", "Real co-writing caught"]];
  root.appendChild(detTable("Headline", heads, (m, key) => { const [p, met] = key.split("|"); return metricCellTd(cellFor(m, p, {}, met), met); }, "Headline"));

  const doms = [["research", "Research"], ["news", "News"], ["reviews", "Reviews"], ["expository", "Encyclopedia / reports"], ["creative", "Fiction"]];
  const grid = el("div", { class: "metricgrid" });
  for (const [p, met, title] of [["audited_human_core", "human_fpr", "Human writing wrongly flagged"], ["fully_ai_core", "fully_ai_recall", "Fully AI caught"],
    ["controlled_replacement", "mixed_recall", "AI sentences inserted: caught"], ["heterogeneous_block_replacement", "mixed_recall", "AI blocks inserted: caught"]])
    grid.appendChild(detTable("Domain", doms, (m, d) => metricCellTd(cellFor(m, p, { domain: d }, met), met), title));
  root.append(el("h2", {}, "By domain"), grid);
  root.appendChild(el("ul", { class: "muted notes" },
    el("li", {}, "Colour: see the key at the top (switch between “compare detectors in each row” and absolute bands with “Cell colour”). Cells with fewer than 30 documents are greyed (“small n”) and not coloured."),
    el("li", {}, "Small grey numbers: count and 95% interval. “no interval*”: every resample gave the same value (clustered data) — not certainty."),
    el("li", {}, "AI blocks inserted come from generate-heterogeneous: in-distribution for any detector trained on that release, and its Encyclopedia/reports cell is Federal Reserve Beige Books (genre confound)."),
    el("li", {}, "Scores are AI-likelihood/extent signals, not calibrated probabilities.")));

  root.appendChild(section("Hard human genres (stress test)", true, () => stressSection()));
  root.appendChild(section("What if the false-alarm budget were different?", true, () => whatifSection()));
  root.appendChild(section("Break down by category", true, () => categoryBreakdown()));
  root.appendChild(section("How well each detector separates AI from human (ROC / AUROC)", true, () => rocSection()));
  root.appendChild(section("Every panel", false, () => {
    const panels = [...new Set(cells().filter((c) => c.protocol_id === S.protocol && !c.task_id).map((c) => c.panel_id))].filter((p) => !p.startsWith("_")).sort();
    const g = el("div", { class: "metricgrid" });
    for (const [met, title] of [["human_fpr", "Humans wrongly flagged"], ["fully_ai_recall", "Fully AI caught"], ["mixed_recall", "Mixed caught"]]) {
      const rows = panels.filter((p) => models().some((m) => { const x = cellFor(m, p, {}, met); return x && x.value !== null; })).map((p) => [p, pname(p)]);
      g.appendChild(detTable("Panel", rows, (m, p) => metricCellTd(cellFor(m, p, {}, met), met), title));
    }
    return g;
  }));
  root.appendChild(section("Coverage and speed", false, () => table(["detector", "reading", "documents", "scored", "errors"],
    S.summary.coverage.map((c) => [c.model_id, PROTO_NAMES[c.protocol_id] || c.protocol_id, c.expected, c.ok || 0, c.error || 0]))));
  root.appendChild(el("p", { class: "muted" }, "Downloads: ", el("a", { href: `/api/download?run=${encodeURIComponent(S.run)}&name=results.summary.cells.csv` }, "all cells (CSV)"), " · ",
    el("a", { href: `/api/download?run=${encodeURIComponent(S.run)}&name=report.md` }, "Markdown report")));
}
function section(title, open, build) {
  const d = el("details", open ? { open: true } : {}, el("summary", {}, title));
  let built = false;
  const fill = () => { if (!built) { built = true; d.appendChild(build()); } };
  if (open) fill(); else d.addEventListener("toggle", fill);
  return d;
}

function stressSection() {
  const box = el("div", {});
  if (!S.hasFull) { box.appendChild(el("p", { class: "na" }, "Needs results.full.json.")); return box; }
  box.appendChild(el("p", { class: "muted" }, "Human writing (all published before modern LLMs) from genres that trip detectors. Same official thresholds. These are not part of the headline human-writing panel; read them as stress tests."));
  const out = el("div", {});
  box.appendChild(out);
  api("/api/stress", { run: S.run, protocol: S.protocol, op: S.op }).then((r) => {
    out.appendChild(detTable("Human genre", r.rows.map((x, i) => [i, x.genre]), (m, i) => { const v = r.rows[i].by_model[m];
      return v ? rateTd(v.rate, v.k, v.n, v.ci, null, "fpr") : rateTd(null, 0, 0); }));
    out.appendChild(el("p", { class: "muted small" }, r.interval + "."));
  }).catch((e) => out.appendChild(el("p", { class: "na" }, String(e))));
  return box;
}

function whatifSection() {
  const box = el("div", {});
  if (!S.hasFull) { box.appendChild(el("p", { class: "na" }, "Needs results.full.json.")); return box; }
  box.appendChild(el("p", { class: "muted" }, "Each row sets every detector’s threshold so that the given share of the held-out human calibration documents would be flagged. 1% and 5% match the official operating points; the others are exploratory. Hover a cell for its 95% interval."));
  const out = el("div", { class: "metricgrid" });
  box.appendChild(out);
  api("/api/whatif", { run: S.run, protocol: S.protocol }).then((r) => {
    const budgets = r.budgets.map((b) => [b, `${pnum(100 * b)}%`]);
    for (const [k, title] of [["human_fpr", "Human writing wrongly flagged"], ["student_fpr", "Students wrongly flagged"], ["fully_ai", "Fully AI caught"],
      ["sentences", "AI sentences caught"], ["blocks", "AI blocks caught"], ["cowriting", "Co-writing caught"]])
      out.appendChild(detTable("Budget", budgets, (m, b) => { const row = r.rows.find((x) => x.model === m && x.budget === b); const v = row && row[k];
        return v ? rateTd(v.rate, v.k, v.n, v.ci, null, k.endsWith("_fpr") ? "fpr" : "recall", true) : rateTd(null, 0, 0); }, title));
  }).catch((e) => out.appendChild(el("p", { class: "na" }, String(e))));
  return box;
}

const BY_NAMES = [["subdomain", "Category"], ["dataset", "Source dataset"], ["generator", "Generator"], ["mechanism", "How AI was mixed in"],
  ["length_bin", "Length"], ["frac_bin", "Share of text written by AI"], ["domain", "Domain"], ["provenance_tier", "Human provenance"]];
function categoryBreakdown() {
  const box = el("div", {});
  if (!S.hasFull) { box.appendChild(el("p", { class: "na" }, "Needs results.full.json.")); return box; }
  const st = S._bd || (S._bd = { by: "subdomain", panel: "", domain: "", cls: "human" });
  const panels = [...new Set(cells().map((c) => c.panel_id))].filter((p) => !p.startsWith("task:") && !p.startsWith("_") && p !== "calibration_human").sort();
  const doms = [...new Set(cells().map((c) => c.slice.domain).filter(Boolean))].sort();
  const out = el("div", {});
  box.append(el("p", { class: "muted" }, "Same official thresholds, split by any property."),
    el("div", { class: "controls" },
      el("label", {}, "Show ", select([["human", "Humans wrongly flagged"], ["ai", "Fully AI caught"], ["mixed", "Mixed caught"]], st.cls, (v) => { st.cls = v; load(); })),
      el("label", {}, "split by ", select(BY_NAMES, st.by, (v) => { st.by = v; load(); })),
      el("label", {}, "Panel ", select([["", "All evaluation documents"], ...panels.map((p) => [p, pname(p)])], st.panel, (v) => { st.panel = v; load(); })),
      el("label", {}, "Domain ", select([["", "All domains"], ...doms.map((d) => [d, d])], st.domain, (v) => { st.domain = v; load(); }))), out);
  async function load() {
    clear(out).appendChild(el("span", { class: "muted" }, "loading…"));
    const ms = models();
    const res = Object.fromEntries(await Promise.all(ms.map(async (m) => [m, await api("/api/breakdown", { run: S.run, model: m, protocol: S.protocol, op: S.op, by: st.by, panel: st.panel, domain: st.domain })])));
    const first = res[ms[0]];
    const cats = first.rows.filter((r) => r[st.cls] && r[st.cls].scored).map((r) => [r.category, r.category]);
    const label = BY_NAMES.find((b) => b[0] === st.by)[1];
    clear(out).append(cats.length ? detTable(label, cats, (m, cat) => { const row = res[m].rows.find((r) => r.category === cat); const a = row && row[st.cls];
      return a && a.scored ? rateTd(a.rate, a.flagged, a.scored, a.ci, null, st.cls === "human" ? "fpr" : "recall") : rateTd(null, 0, 0); })
      : el("p", { class: "na" }, "No documents of this kind for this selection."), el("p", { class: "muted small" }, first.interval + "."));
  }
  load().catch((e) => clear(out).appendChild(el("p", { class: "na" }, String(e))));
  return box;
}

// ------------------------------------------------------------------ ROC small multiples + AUROC dot chart
function rocSection() {
  const box = el("div", {});
  if (!S.hasFull) { box.appendChild(el("p", { class: "na" }, "Needs results.full.json.")); return box; }
  const st = S._roc || (S._roc = { task: "fully", humans: "audited", by: "domain", log: true });
  const out = el("div", {});
  box.append(el("p", { class: "muted" }, "ROC: for every possible threshold, the share of AI texts caught (up) against the share of human texts wrongly flagged (right). Up-and-left is better; the diagonal is chance. The dot marks the official threshold. AUROC = area under the curve (1.0 = perfect)."),
    el("div", { class: "controls" },
      el("label", {}, "AI texts ", select([["fully", "Fully AI only"], ["any", "Fully AI + mixed"]], st.task, (v) => { st.task = v; load(); })),
      el("label", {}, "Humans ", select([["audited", "Pre-LLM + student humans"], ["all", "All humans (incl. benchmark references)"]], st.humans, (v) => { st.humans = v; load(); })),
      el("label", {}, "One chart per ", select([["domain", "Domain"], ["generator", "Generator (AI side)"], ["dataset", "Source dataset"]], st.by, (v) => { st.by = v; load(); })),
      el("label", {}, el("input", { type: "checkbox", checked: st.log, onchange: (e) => { st.log = e.target.checked; load(); } }), " log scale for false alarms (focus on the low-false-alarm region)")), out);
  async function load() {
    clear(out).appendChild(el("span", { class: "muted" }, "computing curves…"));
    const visible = models();
    const rocModels = visible.slice(0, 3);
    const res = await Promise.all(rocModels.map((m) => api("/api/roc", { run: S.run, model: m, protocol: S.protocol, op: S.op, task: st.task, humans: st.humans, by: st.by })));
    const groups = [...new Set(res.flatMap((r) => r.curves.map((c) => c.group)))];
    if (st.by === "generator") groups.sort();
    clear(out);
    out.appendChild(el("div", { class: "legendrow" }, ...rocModels.map((m) => el("span", { class: "key" }, swatch(m), short(m))),
      el("span", { class: "key muted" }, "dot = official threshold · grey diagonal = chance"),
      visible.length > 3 ? el("span", { class: "key prov" }, `Showing the first 3 of ${visible.length} ticked detectors (lines stay readable with ≤ 3); untick some above to compare others.`) : null));
    out.appendChild(aurocDots(groups, rocModels, res));
    const grid = el("div", { class: "rocgrid" });
    for (const g of groups) grid.appendChild(rocPanel(g, rocModels, res.map((r) => r.curves.find((c) => c.group === g)), st.log));
    out.appendChild(grid);
  }
  load().catch((e) => clear(out).appendChild(el("p", { class: "na" }, String(e))));
  return box;
}

function rocPanel(title, models, curves, log) {
  const NS = "http://www.w3.org/2000/svg", W = 280, H = 250, L = 40, R = 18, T = 26, B = 34;
  const xmin = 1e-3;
  const x = (f) => log ? L + (Math.log10(Math.max(f, xmin)) - Math.log10(xmin)) / -Math.log10(xmin) * (W - L - R) : L + f * (W - L - R);
  const y = (t) => T + (1 - t) * (H - T - B);
  const wrap = el("div", { class: "chartwrap rocpanel" });
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `ROC curves for ${title}`);
  const mk = (tag, at, cls) => { const n = document.createElementNS(NS, tag); for (const k in at) n.setAttribute(k, at[k]); if (cls) n.setAttribute("class", cls); svg.appendChild(n); return n; };
  const c0 = curves.find(Boolean) || {};
  mk("text", { x: L, y: 14 }, "lab").textContent = `${title}`;
  mk("text", { x: W - R, y: 14, "text-anchor": "end" }).textContent = `AI ${c0.n_pos ?? "–"} · human ${c0.n_neg ?? "–"}`;
  for (const t of [0, 0.5, 1]) { mk("line", { x1: L, x2: W - R, y1: y(t), y2: y(t) }, "gridl"); mk("text", { x: L - 4, y: y(t) + 4, "text-anchor": "end" }).textContent = `${t * 100}%`; }
  const xt = log ? [0.001, 0.01, 0.1, 1] : [0, 0.5, 1];
  for (const f of xt) { mk("line", { x1: x(f), x2: x(f), y1: T, y2: H - B }, "gridl"); mk("text", { x: x(f), y: H - B + 14, "text-anchor": "middle" }).textContent = `${f * 100 >= 1 ? f * 100 : (f * 100).toFixed(1)}%`; }
  mk("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle" }).textContent = "humans wrongly flagged";
  // chance line
  let dch = ""; for (let i = 0; i <= 60; i++) { const f = log ? Math.pow(10, Math.log10(xmin) * (1 - i / 60)) : i / 60; dch += (i ? "L" : "M") + x(f).toFixed(1) + "," + y(f).toFixed(1); }
  mk("path", { d: dch }, "chance");
  curves.forEach((c, i) => {
    if (!c) return;
    let d = "";
    c.fpr.forEach((f, k) => { if (log && f < xmin && k < c.fpr.length - 1 && c.fpr[k + 1] < xmin) return; d += (d ? "L" : "M") + x(f).toFixed(1) + "," + y(c.tpr[k]).toFixed(1); });
    mk("path", { d }, `curve ${mclass(models[i])}`);
  });
  curves.forEach((c, i) => { if (c && c.operating_point) mk("circle", { cx: x(c.operating_point.fpr), cy: y(c.operating_point.tpr), r: 4.5 }, `opdot ${mclass(models[i])}`); });
  const tip = el("div", { class: "tip" });
  const hit = mk("rect", { x: L, y: T, width: W - L - R, height: H - T - B }, "hit");
  hit.addEventListener("mousemove", throttled((ev) => {
    clear(tip).append(el("div", {}, el("b", {}, title)), ...curves.map((c, i) => c ? el("div", {}, swatch(models[i]),
      `${models[i]}: AUROC ${c.auroc.toFixed(3)}` + (c.operating_point ? ` · at threshold: ${pct(c.operating_point.tpr)} caught, ${pct(c.operating_point.fpr)} humans flagged` : "")) : null));
    const b = wrap.getBoundingClientRect();
    tip.style.display = "block"; tip.style.left = Math.min(ev.clientX - b.left + 10, b.width - 250) + "px"; tip.style.top = (ev.clientY - b.top + 10) + "px";
  }));
  hit.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  wrap.append(svg, tip);
  return wrap;
}

function aurocDots(groups, models, res) {
  const NS = "http://www.w3.org/2000/svg", rowH = 26, L = 190, R = 120, T = 22, W = 800;
  const H = T + groups.length * rowH + 26;
  const vals = res.flatMap((r) => r.curves.map((c) => c.auroc));
  if (!groups.length || !vals.length) return el("p", { class: "na" }, "No group has both human and AI texts for this choice (e.g. humans have no generator). Try “Domain”, or “All humans”.");
  const lo = Math.min(0.95, Math.max(0.5, Math.floor(Math.min(...vals) * 20) / 20));
  const x = (v) => L + (v - lo) / (1 - lo) * (W - L - R);
  const wrap = el("div", { class: "chartwrap" });
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "AUROC per group for each detector");
  const mk = (tag, at, cls) => { const n = document.createElementNS(NS, tag); for (const k in at) n.setAttribute(k, at[k]); if (cls) n.setAttribute("class", cls); svg.appendChild(n); return n; };
  mk("text", { x: L, y: 12 }, "lab").textContent = "AUROC (1.0 = perfect separation)";
  mk("text", { x: W - R + 10, y: 12 }).textContent = models.map((m) => short(m)).join("  |  ");
  for (let i = 0; i <= 4; i++) { const v = lo + (1 - lo) * i / 4; mk("line", { x1: x(v), x2: x(v), y1: T, y2: H - 22 }, "gridl"); mk("text", { x: x(v), y: H - 6, "text-anchor": "middle" }).textContent = v.toFixed(2); }
  groups.forEach((g, i) => {
    const yc = T + i * rowH + rowH / 2;
    mk("text", { x: L - 10, y: yc + 4, "text-anchor": "end" }, "lab").textContent = g.length > 26 ? g.slice(0, 25) + "…" : g;
    const pts = res.map((r) => (r.curves.find((c) => c.group === g) || {}).auroc).filter((v) => v !== undefined);
    if (pts.length >= 2) mk("line", { x1: x(Math.min(...pts)), x2: x(Math.max(...pts)), y1: yc, y2: yc }, "connector");
    res.forEach((r, j) => { const c = r.curves.find((q) => q.group === g); if (!c) return;
      const dot = mk("circle", { cx: x(c.auroc), cy: yc, r: 5 }, `dot ${mclass(models[j])}`);
      const t = document.createElementNS(NS, "title"); t.textContent = `${models[j]}: ${c.auroc.toFixed(3)} (AI ${c.n_pos}, human ${c.n_neg})`; dot.appendChild(t); });
    mk("text", { x: W - R + 10, y: yc + 4 }).textContent = res.map((r) => { const c = r.curves.find((q) => q.group === g); return c ? c.auroc.toFixed(3) : "–"; }).join("  |  ");
  });
  wrap.appendChild(svg);
  return wrap;
}

// ------------------------------------------------------------------ compare
function renderCompare() {
  const root = clear($("compare"));
  const st = S._cmp || (S._cmp = { lrun: S.run, rrun: S.run, lmodel: null, rmodel: null });
  const runNames = S.runs.filter((r) => r.has_summary).map((r) => r.run);
  const ctr = el("div", { class: "controls" });
  root.appendChild(ctr);
  const body = el("div", {});
  root.appendChild(body);
  (async () => {
    for (const r of [st.lrun, st.rrun]) if (!S.summaries[r]) S.summaries[r] = await api("/api/summary", { run: r });
    const L = S.summaries[st.lrun], R = S.summaries[st.rrun];
    const lm = L.models.map((m) => m.model_id), rm = R.models.map((m) => m.model_id);
    st.lmodel = lm.includes(st.lmodel) ? st.lmodel : lm[0];
    st.rmodel = rm.includes(st.rmodel) ? st.rmodel : (rm.find((m) => m !== st.lmodel) || rm[0]);
    clear(ctr).append(
      el("label", {}, "Left run ", select(runNames, st.lrun, (v) => { st.lrun = v; renderCompare(); })),
      el("label", {}, "model ", select(lm, st.lmodel, (v) => { st.lmodel = v; renderCompare(); })),
      el("label", {}, "Right run ", select(runNames, st.rrun, (v) => { st.rrun = v; renderCompare(); })),
      el("label", {}, "model ", select(rm, st.rmodel, (v) => { st.rmodel = v; renderCompare(); })),
      el("label", {}, "protocol ", select(L.protocols.map((p) => p.protocol_id), S.protocol, (v) => { S.protocol = v; renderCompare(); })),
      el("label", {}, "op ", select([...new Set(L.operating_points.map((o) => o.operating_point_id))], S.op, (v) => { S.op = v; renderCompare(); })));
    // compatibility
    const warn = [];
    if (L.benchmark.benchmark_hash !== R.benchmark.benchmark_hash) warn.push("Different benchmark versions (text/gold/selection differ): marginal estimates are not directly comparable.");
    if (L.benchmark.calibration_membership_sha256 !== R.benchmark.calibration_membership_sha256) warn.push("Different calibration membership.");
    const pl = L.protocols.find((p) => p.protocol_id === S.protocol), pr = R.protocols.find((p) => p.protocol_id === S.protocol);
    if (!pl || !pr || pl.fingerprint !== pr.fingerprint) warn.push("Protocol/window/preprocessing fingerprints differ or protocol missing.");
    const ml = L.models.find((m) => m.model_id === st.lmodel), mr = R.models.find((m) => m.model_id === st.rmodel);
    if (JSON.stringify(ml.precision) !== JSON.stringify(mr.precision)) warn.push(`Precision/quantisation differ: ${JSON.stringify(ml.precision)} vs ${JSON.stringify(mr.precision)}`);
    clear(body);
    for (const w of warn) body.appendChild(el("div", { class: "explore" }, el("b", { class: "tag" }, "Compatibility: "), w));
    const same = st.lrun === st.rrun;
    if (same) {
      const comps = (L.comparisons || []).filter((c) => c.protocol_id === S.protocol && c.operating_point_id === S.op &&
        ((c.left.model_id === st.lmodel && c.right.model_id === st.rmodel) || (c.left.model_id === st.rmodel && c.right.model_id === st.lmodel)));
      body.appendChild(el("h2", {}, "Paired comparison on common scored examples (same run)"));
      body.appendChild(table(["panel", "slice", "metric", "left", "right", "Δ (pp, left−right)", "95% paired CI (pp)", "n common"],
        comps.map((c) => { const flip = c.left.model_id !== st.lmodel; const d = c.difference_pp === null ? null : (flip ? -c.difference_pp : c.difference_pp);
          const lo = c.uncertainty ? (flip ? -c.uncertainty.upper : c.uncertainty.lower) : null, hi = c.uncertainty ? (flip ? -c.uncertainty.lower : c.uncertainty.upper) : null;
          return [c.task_or_panel, JSON.stringify(c.slice), c.metric, pct(flip ? c.right_estimate : c.left_estimate), pct(flip ? c.left_estimate : c.right_estimate),
            d === null ? "n/a" : d.toFixed(2), lo === null ? "n/a" : `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`, c.n_common_scored]; })));
    } else {
      body.appendChild(el("p", { class: "explore" }, "Different runs: showing marginal estimates side by side. Differences are NOT paired and two marginal intervals do not establish significance; load both full files and rerun evaluation together for paired estimates."));
    }
    body.appendChild(el("h2", {}, "Side by side (marginal estimates on each run's full supported population)"));
    const keyed = (Sm, model) => { const m = new Map(); for (const c of Sm.cells) if (c.model_id === model && c.protocol_id === S.protocol && c.operating_point_id === S.op && !c.task_id) m.set(c.panel_id + "|" + sliceKey(c.slice), c); return m; };
    const A = keyed(L, st.lmodel), Bm = keyed(R, st.rmodel);
    const rows = [];
    const showAll = !!st.all;
    for (const [k, a] of A) {
      if (!(a.slice.view === "all" && Object.keys(a.slice).length <= 2)) continue;
      const sk = Object.keys(a.slice).filter((x) => x !== "view");
      if (!showAll && !(sk.length === 0 || (sk.length === 1 && sk[0] === "domain"))) continue;
      const b = Bm.get(k);
      for (const name of ["human_fpr", "fully_ai_recall", "mixed_recall"]) {
        const ma = a.metrics[name], mb = b ? b.metrics[name] : null;
        if ((!ma || ma.value === null) && (!mb || mb.value === null)) continue;
        const diff = ma && mb && ma.value !== null && mb.value !== null ? (100 * (ma.value - mb.value)).toFixed(2) : "n/a";
        const mismatch = b && a.example_ids_hash !== b.example_ids_hash ? "populations differ" : "";
        rows.push([a.panel_id, JSON.stringify(Object.fromEntries(Object.entries(a.slice).filter(([x]) => x !== "view"))), name, metricTd(ma), metricTd(mb), diff, mismatch]);
      }
    }
    body.appendChild(el("label", {}, el("input", { type: "checkbox", checked: showAll, onchange: (e) => { st.all = e.target.checked; renderCompare(); } }), " show every slice (long)"));
    body.appendChild(table(["panel", "slice", "metric", "left", "right", "Δ pp (unpaired)", "note"], rows));
  })().catch((e) => body.appendChild(el("p", { class: "na" }, String(e))));
}

// ------------------------------------------------------------------ errors
function renderErrors() {
  const root = clear($("errors"));
  if (!S.hasFull) { root.appendChild(el("p", { class: "na" }, "The error browser needs results.full.json for this run (summary-only).")); return; }
  root.appendChild(selectors(renderErrors));
  const f = S._ef || (S._ef = { outcome: "false_positive", panel: "", dataset: "", domain: "", authorship: "", mechanism: "", generator: "", frac_bin: "", length_bin: "", provenance_tier: "", role: "evaluation", model2: "" });
  const panels = [...new Set(cells().map((c) => c.panel_id))].filter((p) => !p.startsWith("task:")).sort();
  const otherModels = S.summary.models.map((m) => m.model_id).filter((m) => m !== S.model);
  const inp = (k) => el("input", { value: f[k], placeholder: k, size: 14, onchange: (e) => { f[k] = e.target.value.trim(); load(); } });
  root.appendChild(el("div", { class: "controls" },
    el("label", {}, "Outcome ", select([["false_positive", "human false accusations"], ["false_negative", "missed AI involvement"], ["disagreement", "model disagreement"], ["error", "inference errors"], ["", "all"]], f.outcome, (v) => { f.outcome = v; load(); })),
    el("label", {}, "vs model ", select([["", "-"], ...otherModels.map((m) => [m, m])], f.model2, (v) => { f.model2 = v; load(); })),
    el("label", {}, "Panel ", select([["", "(any)"], ...panels.map((p) => [p, p])], f.panel, (v) => { f.panel = v; load(); })),
    el("label", {}, "Role ", select([["evaluation", "evaluation"], ["calibration", "calibration"], ["", "any"]], f.role, (v) => { f.role = v; load(); })),
    inp("dataset"), inp("domain"), inp("authorship"), inp("mechanism"), inp("generator"), inp("frac_bin"), inp("length_bin"), inp("provenance_tier")));
  const out = el("div", {});
  root.appendChild(out);
  async function load() {
    clear(out).appendChild(el("span", { class: "muted" }, "loading…"));
    const r = await api("/api/examples", { run: S.run, model: S.model, protocol: S.protocol, op: S.op, ...f });
    clear(out);
    if (f.outcome === "false_positive") out.appendChild(el("div", { class: "explore" }, el("b", { class: "tag" }, "Human false accusations: "), `${r.rows.length}${r.rows.length >= r.limit ? "+" : ""} human-written documents flagged at ${S.op} (sorted by score).`));
    out.appendChild(table(["example", "role", "dataset", "domain", "authorship", "mechanism", "generator", "AI frac", "length", "provenance", "status", "score", "decisions"],
      r.rows.map((x) => [el("td", {}, el("span", { class: "clickable", onclick: () => { S._inspect = x.example_id; activate("inspect"); } }, x.example_id)),
        x.role, x.dataset, x.domain, el("td", { class: x.authorship === "human" && f.outcome === "false_positive" ? "fp" : "" }, x.authorship), x.mechanism || "", x.generator || "",
        x.frac_bin || "", `${x.length_bin} (${x.word_count}w)`, x.provenance_tier || "", x.status || "missing", x.score === null ? "n/a" : x.score.toFixed(4),
        el("td", { class: "muted" }, x.decisions || "") ])));
  }
  load().catch((e) => clear(out).appendChild(el("p", { class: "na" }, String(e))));
}

// ------------------------------------------------------------------ inspect
function renderInspect() {
  const root = clear($("inspect"));
  if (!S.hasFull) { root.appendChild(el("p", { class: "na" }, "Full-text and span inspection needs results.full.json (summary-only run).")); return; }
  const idIn = el("input", { value: S._inspect || "", size: 60, placeholder: "example_id" });
  root.appendChild(el("div", { class: "controls" }, el("label", {}, "Example ", idIn), el("button", { onclick: () => { S._inspect = idIn.value.trim(); renderInspect(); } }, "Open")));
  if (!S._inspect) { root.appendChild(el("p", { class: "muted" }, "Pick an example from the error browser or paste an example_id.")); return; }
  const out = el("div", {});
  root.appendChild(out);
  api("/api/example", { run: S.run, id: S._inspect }).then(({ example: e, predictions: P, error }) => {
    if (error) { out.appendChild(el("p", { class: "na" }, error)); return; }
    const t = e.truth;
    out.appendChild(el("div", { class: "grid2" },
      table(["field", "value"], [["example_id", e.example_id], ["dataset / split / row", `${e.source.dataset_key} / ${e.source.original_split} / ${e.source.original_row_id}`],
        ["role / panels", `${e.role} / ${e.panel_ids.join(", ")}`], ["domain / subdomain", `${e.domain} / ${e.subdomain}`],
        ["authorship (basis)", `${t.authorship} (${t.authorship_basis})`], ["AI involvement / mechanism", `${t.ai_involvement} / ${t.mix_mechanism || "-"}`],
        ["generator", t.generator || "-"], ["gold spans", t.ai_char_spans === null ? "unavailable (null)" : `${t.ai_char_spans.length} span(s); annotation ${t.annotation_completeness} (${t.annotation_provenance})`],
        ["AI fraction (chars)", t.ai_fraction_char === null ? "n/a" : pct(t.ai_fraction_char, 2)],
        ["source / leakage / independence group", `${e.source_group_id} / ${e.leakage_group_id} / ${e.independence_group_id}`],
        ["human provenance", `${e.human_provenance.category || "-"} (${e.human_provenance.tier || "-"})`], ["words / code points", `${e.word_count} / ${e.char_count}`],
        ["training exposure", JSON.stringify(e.training_exposure)], ["artefact flags", e.artefact_flags.join(", ") || "-"]]),
      table(["model", "protocol", "status", "score", "decisions", "coverage"], P.map((p) => [p.model_id, p.protocol_id, p.status, p.score === null ? "n/a" : p.score.toFixed(4),
        p.operating_point_decisions.map((d) => `${d.operating_point_id}: ${d.flag_ai === null ? "n/a" : d.flag_ai ? "FLAG" : "pass"}`).join("; "),
        p.protocol_id === "native_truncate_v1" ? `truncated=${p.coverage.truncated}; tokens ${p.coverage.n_tokens_used}/${p.coverage.n_tokens_full}` : `${p.coverage.n_windows} window(s)`]))));
    // layer controls
    const lay = S._layers || (S._layers = { gold: true, windows: true, deleted: true, unread: true, model: P[0] ? P[0].model_id : null });
    const models = [...new Set(P.map((p) => p.model_id))];
    const ctr = el("div", { class: "controls legend" },
      ...["gold", "windows", "deleted", "unread"].map((k) => el("label", {}, el("input", { type: "checkbox", checked: lay[k], onchange: (ev) => { lay[k] = ev.target.checked; draw(); } }), " ", k)),
      el("label", {}, "model ", select(models, lay.model, (v) => { lay.model = v; draw(); })),
      el("span", { class: "l-ai" }, "gold AI"), el("span", { class: "l-win" }, "scored window boundary (contextual, not a span prediction)"),
      el("span", { class: "l-del" }, "deleted by model preprocessing"), el("span", { class: "l-unread" }, "unread under native truncation"));
    const view = el("div", { class: "textview" });
    const winTable = el("div", {});
    out.append(el("h2", {}, "Canonical text"), ctr, view, el("p", { class: "muted" }, "Native span predictions: unavailable for both baselines (document/window scorers)."), winTable);
    function draw() {
      const cps = Array.from(e.text || "");  // code points, matching Python str indices
      const n = cps.length;
      const layers = [];
      if (lay.gold && t.ai_char_spans) for (const [a, b] of t.ai_char_spans) layers.push([a, b, "l-ai", "gold AI span"]);
      const pw = P.find((p) => p.model_id === lay.model && p.protocol_id === "shared_windows_v1");
      const pn = P.find((p) => p.model_id === lay.model && p.protocol_id === "native_truncate_v1");
      if (lay.windows && pw && pw.windows) for (const w of pw.windows) layers.push([w.canonical_end - 1, w.canonical_end, "l-win", `end of ${w.window_id}: score ${w.score === null ? "n/a" : w.score.toFixed(4)}`]);
      if (lay.deleted && pw && pw.coverage.preprocessing_deleted_canonical) for (const [a, b] of pw.coverage.preprocessing_deleted_canonical) layers.push([a, b, "l-del", "deleted by EditLens preprocessing in a window"]);
      if (lay.unread && pn && pn.coverage.first_unread_canonical !== null && pn.coverage.first_unread_canonical !== undefined) layers.push([pn.coverage.first_unread_canonical, n, "l-unread", "not read under native truncation"]);
      const cuts = new Set([0, n]);
      for (const [a, b] of layers) { cuts.add(Math.max(0, Math.min(n, a))); cuts.add(Math.max(0, Math.min(n, b))); }
      const pts = [...cuts].sort((x, y) => x - y);
      clear(view);
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        const act = layers.filter(([x, y]) => x <= a && b <= y);
        const span = el("span", { class: act.map((l) => l[2]).join(" "), title: act.map((l) => l[3]).join("; ") });
        span.textContent = cps.slice(a, b).join("");
        view.appendChild(span);
      }
      clear(winTable);
      if (pw && pw.windows) winTable.appendChild(table(["window", "canonical [start,end)", "words", "status", "score", "bucket probs", "tokens"],
        pw.windows.map((w) => [w.window_id, `[${w.canonical_start}, ${w.canonical_end})`, w.n_words, w.status, w.score === null ? "n/a" : w.score.toFixed(4),
          w.raw_outputs ? w.raw_outputs.bucket_probabilities.map((x) => x.toFixed(3)).join(" ") : "n/a", w.n_tokens_model_visible ?? "n/a"])));
    }
    draw();
  }).catch((err) => out.appendChild(el("p", { class: "na" }, String(err))));
}

// ------------------------------------------------------------------ provenance
function renderProvenance() {
  const root = clear($("provenance"));
  const s = S.summary, b = s.benchmark, r = s.run;
  root.appendChild(el("h2", {}, "Benchmark"));
  root.appendChild(table(["field", "value"], [["id / version / profile", `${b.benchmark_id} / ${b.benchmark_version} / ${b.profile}`], ["purpose", `${b.purpose} — ${b.release_role}`],
    ["benchmark hash", b.benchmark_hash], ["calibration membership", b.calibration_membership_sha256], ["window manifest", b.fingerprints.window_manifest_sha256],
    ["examples (eval / calib)", `${b.realised_counts.examples} (${b.realised_counts.evaluation} / ${b.realised_counts.calibration})`],
    ["evaluation leakage / independence groups", `${b.realised_counts.evaluation_leakage_groups} / ${b.realised_counts.evaluation_independence_groups}`], ["offsets", b.offset_convention]]));
  root.appendChild(el("h3", {}, "Shortfalls (requested but unavailable cells)"));
  root.appendChild(table(["panel", "cell", "requested", "eligible", "selected", "missing"], b.shortfalls.map((c) => [c.panel, JSON.stringify(Object.fromEntries(Object.entries(c).filter(([k]) => k.startsWith("cell_")))), c.requested, c.eligible, c.selected, c.missing])));
  root.appendChild(el("h2", {}, "Run"));
  root.appendChild(table(["field", "value"], [["run_id / status", `${s.run_id} / ${s.status}`], ["code", JSON.stringify(r.code)], ["hardware", JSON.stringify(r.hardware && r.hardware.device)],
    ["packages", Object.entries(r.packages).map(([k, v]) => `${k}=${v}`).join(", ")], ["loader parity", JSON.stringify(r.loader_parity && r.loader_parity.models.map((m) => ({ model: m.model_id, pass: m.pass, max_score_diff: m.max_abs_score_diff_single_vs_upstream })))],
    ["tests", JSON.stringify(r.tests)], ["full result sha256", s.full_result_sha256]]));
  root.appendChild(el("h2", {}, "Models"));
  root.appendChild(table(["model", "checkpoint@rev", "base@rev", "loader / head", "precision", "K", "score semantics", "status"],
    s.models.map((m) => [m.model_id, `${m.checkpoint}@${(m.checkpoint_revision || "").slice(0, 10)}`, `${m.base_model}@${(m.base_revision || "").slice(0, 10)}`, `${m.loader} / ${m.head}`, JSON.stringify(m.precision), m.n_buckets, m.score_semantics, m.status])));
  root.appendChild(el("h2", {}, "Operating points"));
  root.appendChild(table(["calibration_id", "threshold (exact)", "target", "achieved", "ties", "n", "status", "reasons"],
    s.operating_points.map((o) => [o.calibration_id, String(o.threshold), pct(o.target_fpr), `${pct(o.actual_fpr, 2)} (${o.false_positives}/${o.n_scored})`, o.tied_scores_at_threshold, o.n_scored, o.status, (o.reasons || []).join("; ")])));
  root.appendChild(el("h2", {}, "Protocols"));
  root.appendChild(table(["protocol", "kind", "fingerprint", "aggregation", "truncation", "localisation"], s.protocols.map((p) => [p.protocol_id, p.implementation_kind, p.fingerprint.slice(0, 16), p.document_aggregation, p.truncation, p.localisation_mode])));
  root.appendChild(el("h2", {}, "Source acquisition (all catalogue entries)"));
  root.appendChild(table(["source", "status", "files", "observed rows", "reported", "eligibility", "rights"], s.source_status.map((x) => [x.source_id, x.acquisition_status,
    `${x.download_completeness.n_files_ok}/${x.download_completeness.n_files_expected}`, x.observed_rows, JSON.stringify(x.reported_rows), x.eligibility_status, x.rights_restrictions])));
  root.appendChild(el("h2", {}, "Validation"));
  root.appendChild(table(["invariant", "ok", "detail"], s.extensions.validation.semantic_checks.map((c) => [c.invariant, String(c.ok), c.detail ? JSON.stringify(c.detail) : ""])));
  root.appendChild(el("h2", {}, "Warnings and limitations"));
  root.appendChild(el("ul", {}, s.warnings.map((w) => el("li", {}, el("b", {}, w.code + ": "), w.message))));
}


// ------------------------------------------------------------------ calibration sensitivity (exploratory)
function renderCalibration() {
  const root = clear($("calibration"));
  root.appendChild(el("div", { class: "explore" }, el("b", { class: "tag" }, "EXPLORATORY: "),
    "per-dataset thresholds use the official order-statistic rule on subsets of the same calibration documents. They never replace the frozen operating points."));
  const out = el("div", {});
  root.appendChild(out);
  api("/api/sensitivity", { run: S.run }).then((d) => {
    const st = S._cal || (S._cal = { protocol: "shared_windows_v1", op: "human_fpr_1pct", variant: "leave_out:no_robots" });
    const rs = d.results.filter((r) => r.protocol_id === st.protocol && r.operating_point_id === st.op);
    const models = rs.map((r) => r.model_id);
    const variants = Object.keys(rs[0].variants);
    clear(out);
    out.appendChild(el("div", { class: "controls" },
      el("label", {}, "Protocol ", select([...new Set(d.results.map((r) => r.protocol_id))], st.protocol, (v) => { st.protocol = v; renderCalibration(); })),
      el("label", {}, "Target FPR ", select([["human_fpr_1pct", "1%"], ["human_fpr_5pct", "5%"]], st.op, (v) => { st.op = v; renderCalibration(); }))));
    out.appendChild(el("h2", {}, `Threshold calibrated on each dataset alone (target ${pct(rs[0].target_fpr, 0)} FPR), with 95% bootstrap interval`));
    out.appendChild(el("div", {}, ...rs.map((r, i) => el("span", { class: "key" }, el("i", { class: `s${i + 1}` }), `${r.model_id} (pooled official threshold ${r.variants.pooled_official_rule.toFixed(3)}, dashed)`))));
    out.appendChild(dotChart(rs));
    // table view (accessibility + exact values)
    const datasets = rs[0].per_dataset.map((p) => p.dataset);
    out.appendChild(el("h3", {}, "Table view"));
    out.appendChild(table(["dataset", "domain", "n", "k", ...models.flatMap((m) => [`${m} threshold [95% CI]`, `${m}: FPR here at pooled threshold`])],
      datasets.map((ds) => { const p0 = rs[0].per_dataset.find((p) => p.dataset === ds);
        return [ds, p0.domain, p0.n, p0.k, ...rs.flatMap((r) => { const p = r.per_dataset.find((x) => x.dataset === ds);
          return [`${p.threshold.toFixed(3)} [${p.ci_lower.toFixed(3)}, ${p.ci_upper.toFixed(3)}]`, pct(p.pooled_threshold_fpr_on_this_dataset)]; })]; })));
    out.appendChild(table(["model", "min", "median", "max", "separated datasets (interval overlaps no other dataset's)"],
      rs.map((r) => [r.model_id, r.summary.min.toFixed(3), r.summary.median.toFixed(3), r.summary.max.toFixed(3), r.summary.outliers.join(", ") || "none"])));
    // consequences of threshold variants on the evaluation panels
    out.appendChild(el("h2", {}, "What each calibration choice would do on the evaluation panels"));
    out.appendChild(el("p", { class: "muted" }, "Equal-weight pooling equals the pooled rule whenever every calibration dataset has the same n (500 here)."));
    const metricsShown = [["fully_ai_recall:all_core", "fully-AI recall"], ["replacement_recall:all_core", "replacement recall"], ["aitdna_mixed_recall", "AITDNA mixed recall"], ["block_replacement_recall", "block-replacement recall"],
      ["instruction_human_fpr", "instruction human FPR"], ["student_human_fpr", "student FPR"], ["audited_human_fpr:news", "news FPR"], ["audited_human_fpr:reviews", "reviews FPR"],
      ["audited_human_fpr:research", "research FPR"], ["audited_human_fpr:expository", "expository FPR"], ["audited_human_fpr:creative", "creative FPR"]];
    for (const r of rs) {
      out.appendChild(el("h3", {}, r.model_id));
      out.appendChild(table(["calibration variant", "threshold", ...metricsShown.map((m) => m[1])],
        variants.map((v) => { const c = r.consequences[v];
          return [v, c.threshold.toFixed(3), ...metricsShown.map(([k]) => c[k] && c[k].value !== null ? `${pct(c[k].value)} (${c[k].numerator}/${c[k].denominator})` : "n/a")]; })));
    }
    out.appendChild(el("h3", {}, "Caveats"));
    out.appendChild(el("ul", {}, d.statistical_caveats.map((c) => el("li", {}, c))));
  }).catch((e) => out.appendChild(el("p", { class: "na" }, String(e))));
}

function dotChart(rs) {
  const NS = "http://www.w3.org/2000/svg";
  const datasets = rs[0].per_dataset.map((p) => p.dataset);
  const W = 820, L = 140, R = 24, T = 16, rowH = 34, H = T + datasets.length * rowH + 34;
  const x = (v) => L + v * (W - L - R);
  const wrap = el("div", { class: "chartwrap" });
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Per-dataset calibration thresholds with bootstrap intervals for each model");
  const mk = (tag, at, cls) => { const n = document.createElementNS(NS, tag); for (const k in at) n.setAttribute(k, at[k]); if (cls) n.setAttribute("class", cls); svg.appendChild(n); return n; };
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    mk("line", { x1: x(t), x2: x(t), y1: T, y2: H - 28 }, "gridl");
    mk("text", { x: x(t), y: H - 10, "text-anchor": "middle" }).textContent = t.toFixed(2);
  }
  mk("line", { x1: L, x2: W - R, y1: H - 28, y2: H - 28 }, "ax");
  const tip = el("div", { class: "tip" });
  datasets.forEach((ds, i) => {
    const yc = T + i * rowH + rowH / 2;
    const lab = mk("text", { x: L - 10, y: yc + 4, "text-anchor": "end" }, "lab");
    lab.textContent = ds;
    rs.forEach((r, j) => {
      const p = r.per_dataset.find((q) => q.dataset === ds);
      const y = yc + (j === 0 ? -6 : 6);
      const cls = `s${j + 1}`;
      mk("line", { x1: x(p.ci_lower), x2: x(p.ci_upper), y1: y, y2: y }, `whisk ${cls}`);
      mk("circle", { cx: x(p.threshold), cy: y, r: 5 }, `dot ${cls}`);
      const hit = mk("rect", { x: Math.min(x(p.ci_lower), x(p.threshold) - 8), y: y - 7, width: Math.max(16, x(p.ci_upper) - x(p.ci_lower) + 16), height: 14 }, "hit");
      hit.addEventListener("mousemove", throttled((ev) => {
        clear(tip).append(el("div", {}, el("b", {}, `${ds} · ${r.model_id}`)),
          el("div", {}, `threshold ${p.threshold.toFixed(3)}  [${p.ci_lower.toFixed(3)}, ${p.ci_upper.toFixed(3)}]`),
          el("div", {}, `n=${p.n}, k=${p.k}; FPR here at pooled threshold: ${pct(p.pooled_threshold_fpr_on_this_dataset)}`));
        const b = wrap.getBoundingClientRect();
        tip.style.display = "block";
        tip.style.left = Math.min(ev.clientX - b.left + 12, b.width - 260) + "px";
        tip.style.top = (ev.clientY - b.top + 12) + "px";
      }));
      hit.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    });
  });
  rs.forEach((r, j) => {
    const t = r.variants.pooled_official_rule;
    mk("line", { x1: x(t), x2: x(t), y1: T, y2: H - 28 }, `rule s${j + 1}`).setAttribute("fill", "none");
  });
  mk("text", { x: L, y: H - 10, "text-anchor": "start" }).textContent = "";
  wrap.append(svg, tip);
  return wrap;
}

// ------------------------------------------------------------------ boot
const renderers = { scorecard: renderScorecard, calibration: renderCalibration, compare: renderCompare, errors: renderErrors, inspect: renderInspect, provenance: renderProvenance };
function activate(tab) {
  for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("active", b.dataset.tab === tab);
  for (const s of document.querySelectorAll(".tab")) s.classList.toggle("active", s.id === tab);
  S.tab = tab;
  renderers[tab]();
}
function renderAll() { activate(S.tab || "scorecard"); }
for (const b of document.querySelectorAll("#tabs button")) b.addEventListener("click", () => activate(b.dataset.tab));

(async () => {
  S.runs = await api("/api/runs");
  const sel = $("run");
  for (const r of S.runs) sel.appendChild(el("option", { value: r.run },
    (r.profile && r.profile !== "full" ? "SMOKE TEST (tiny) · " : "") + (r.benchmark_id || r.run) +
    (r.n_evaluation ? ` · ${r.n_evaluation.toLocaleString()} eval docs` : "") + (r.status && r.status !== "complete" ? ` · ${r.status}` : "") + (r.has_full ? "" : " · summary only")));
  sel.addEventListener("change", () => loadRun(sel.value));
  // hash routing: #<tab>, #inspect=<example_id>, optional &run=<run>
  const h = new URLSearchParams(location.hash.slice(1).replace(/^([a-z]+)(=([^&]*))?/, (m, t, _e, v) => `tab=${t}` + (v ? `&id=${v}` : "")));
  if (h.get("tab") && renderers[h.get("tab")]) S.tab = h.get("tab");
  if (h.get("id")) S._inspect = decodeURIComponent(h.get("id"));
  const pref = S.runs.find((r) => r.run === h.get("run")) || S.runs[0];  // server orders: newest full measured run first
  if (pref) { sel.value = pref.run; await loadRun(pref.run); }
})().catch((e) => document.body.appendChild(el("p", { class: "na" }, "Failed to load: " + e)));
