"use strict";
// Sumizome dashboard. Vanilla JS, hand-rolled inline SVG charts, fully offline.
// No em-dashes in this project.

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const j = async (u) => (await fetch(u)).json();
const post = (u, b) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) }).then(r => r.json());
const del = (u, b) => fetch(u, { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) }).then(r => r.json());
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtTs = (t) => t ? new Date(t * 1000).toLocaleString() : "never";
const hhmm = (t) => t ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";

function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove("show"), 2600); }

// nav icons (stroke SVG)
const ICONS = {
  "i-grid": "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
  "i-play": "M6 4l14 8-14 8z",
  "i-table": "M3 5h18v14H3zM3 10h18M9 5v14",
  "i-shield": "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z",
  "i-chart": "M4 20V10M10 20V4M16 20v-7M22 20H2",
  "i-log": "M5 4h14v16H5zM8 8h8M8 12h8M8 16h5",
  "i-loop": "M3 12a9 9 0 0 1 15-6.7L21 8M21 3v5h-5M21 12a9 9 0 0 1-15 6.7L3 16M3 21v-5h5",
  "i-cpu": "M6 6h12v12H6zM9 9h6v6H9M2 9h2M2 14h2M20 9h2M20 14h2M9 2v2M14 2v2M9 20v2M14 20v2",
  "i-gear": "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M19 12l2-1-2-4-2 1a7 7 0 0 0-2-1l-1-2H8L7 5a7 7 0 0 0-2 1L3 5 1 9l2 1v2l-2 1 2 4 2-1a7 7 0 0 0 2 1l1 2h4l1-2a7 7 0 0 0 2-1l2 1 2-4-2-1z",
  "i-pulse": "M3 12h4l3 8 4-16 3 8h4",
};
function svgIcon(name) { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${ICONS[name]}"/></svg>`; }
$$("nav a").forEach(a => { a.innerHTML = a.textContent.replace(/\{(i-[a-z]+)\}/, (_, n) => svgIcon(n)); });

// ---------------------------------------------------------------- charts
const CH = { w: 480, h: 190, pad: 30 };
function svgBars(values, labels, opts = {}) {
  const w = opts.width || CH.w, h = opts.height || CH.h, pad = 28;
  const max = Math.max(1, ...values), n = values.length;
  const bw = (w - pad * 2) / Math.max(1, n), colors = opts.colors || values.map(() => "var(--accent)");
  let out = "";
  values.forEach((v, i) => {
    const bh = (h - pad * 2) * (v / max), x = pad + i * bw, y = h - pad - bh;
    out += `<rect x="${x + 2}" y="${y}" width="${Math.max(1, bw - 4)}" height="${bh}" rx="3" fill="${colors[i]}"><title>${esc(labels[i])}: ${v}</title></rect>`;
    if (n <= 26) out += `<text x="${x + bw / 2}" y="${h - pad + 13}" text-anchor="middle">${esc(labels[i])}</text>`;
    if (v > 0 && bw > 15) out += `<text x="${x + bw / 2}" y="${y - 4}" text-anchor="middle" fill="var(--txt)" font-size="10">${v}</text>`;
  });
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" width="100%">${out}</svg>`;
}
function svgLines(xlabels, series, opts = {}) {
  const w = opts.width || CH.w, h = opts.height || CH.h, pad = 32, n = xlabels.length;
  if (!n) return `<div class="muted">no data yet</div>`;
  const max = opts.max != null ? opts.max : Math.max(1, ...series.flatMap(s => s.data));
  const xp = i => pad + (n === 1 ? (w - pad * 2) / 2 : i * (w - pad * 2) / (n - 1));
  const yp = v => h - pad - (h - pad * 2) * (v / max);
  let out = "";
  for (let g = 0; g <= 2; g++) { const yy = pad + g * (h - pad * 2) / 2; out += `<line x1="${pad}" y1="${yy}" x2="${w - pad}" y2="${yy}" stroke="var(--line)"/><text x="4" y="${yy + 3}">${Math.round(max - g * max / 2)}</text>`; }
  series.forEach(s => {
    let d = ""; s.data.forEach((v, i) => d += (i ? "L" : "M") + xp(i).toFixed(1) + " " + yp(v).toFixed(1) + " ");
    out += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    s.data.forEach((v, i) => out += `<circle cx="${xp(i)}" cy="${yp(v)}" r="2.5" fill="${s.color}"><title>${esc(xlabels[i])}: ${v}</title></circle>`);
  });
  const step = Math.ceil(n / 6);
  xlabels.forEach((l, i) => { if (i % step === 0) out += `<text x="${xp(i)}" y="${h - pad + 14}" text-anchor="middle">${esc(l.slice(5))}</text>`; });
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" width="100%">${out}</svg>`;
}
function stackedBar(parts) {
  const total = parts.reduce((a, p) => a + p.v, 0) || 1;
  return `<div style="display:flex;border-radius:9px;overflow:hidden;border:1px solid var(--line);height:30px">` +
    parts.map(p => `<div title="${esc(p.label)}: ${p.v}" style="width:${100 * p.v / total}%;background:${p.color}"></div>`).join("") + `</div>`;
}
function heatKeyboard(rows, heat) {
  const max = Math.max(1, ...Object.values(heat));
  return rows.map((row, ri) => `<div class="kbd-row r${ri + 1}">` + [...row].map(ch => {
    const v = heat[ch] || 0, t = v / max;
    const bg = t > 0 ? `rgba(255,${Math.round(120 - t * 95)},${Math.round(85 - t * 65)},${(0.15 + t * 0.85).toFixed(2)})` : "var(--panel2)";
    return `<div class="key" style="background:${bg}" title="${ch}: ${v} errors">${ch}${v ? `<span class="n">${v}</span>` : ""}</div>`;
  }).join("") + `</div>`).join("");
}

// ---------------------------------------------------------------- sections
const loaders = {
  async overview() {
    const d = await j("/api/overview"), L = d.layers;
    $("#ovCards").innerHTML = [
      ["accent", d.corrections_today, "corrections today"],
      ["accent", L.total, "corrections all time"],
      ["good", (d.memory.dictionary || 0).toLocaleString(), "dictionary typos (instant)"],
      ["mem", d.memory.pairs, "learned from you"],
      ["matcher", d.whitelist_size, "whitelisted words"],
      ["llm", L.llm, "LLM recoveries"],
    ].map(([c, v, l]) => `<div class="card"><div class="val ${c}" style="font-size:${String(v).length > 6 ? 20 : 27}px">${v}</div><div class="lbl">${l}</div></div>`).join("");
    $("#ovLayerBar").innerHTML = stackedBar([
      { label: "Layer 1 memory", v: L.memory, color: "var(--mem)" },
      { label: "Layer 2 matcher", v: L.matcher, color: "var(--matcher)" },
      { label: "Layer 3 LLM", v: L.llm, color: "var(--llm)" }]);
    const g = $("#ovGauge"); g.style.setProperty("--v", L.without_llm_pct);
    $("#ovGaugeVal").textContent = L.without_llm_pct + "%";
    $("#ovGaugeNote").innerHTML = `<b style="color:var(--txt)">${L.memory + L.matcher}</b> of <b style="color:var(--txt)">${L.total}</b> corrections were deterministic, no GPU needed. Higher is faster and more personal.`;
  },

  async playground() { /* rendered on demand via runPlayground */ },

  async typos() {
    const rows = await j("/api/typos");
    window._typos = rows; window._typoSort = window._typoSort || { k: "count", d: -1 };
    $("#typoCount").textContent = rows.length + " mappings";
    renderTypos();
  },

  async whitelist() {
    const w = await j("/api/whitelist");
    $("#wlCount").textContent = w.length + " words";
    $("#wlList").innerHTML = w.length ? w.map(x => `<span class="w">${esc(x)}<span class="tag-del" data-w="${esc(x)}">x</span></span>`).join("") : `<span class="muted">empty for now</span>`;
    $$("#wlList .tag-del").forEach(b => b.onclick = async () => { await del("/api/whitelist", { word: b.dataset.w }); loaders.whitelist(); });
  },

  async insights() {
    const d = await j("/api/insights");
    $("#heatmap").innerHTML = heatKeyboard(d.keyboard_rows, d.key_heat);
    const tm = d.top_mangles;
    $("#topMangles").innerHTML = tm.length ? svgBars(tm.map(m => m.count), tm.map(m => m.mangled), { colors: tm.map(() => "var(--matcher)"), height: 210 }) : `<div class="muted">no mangles learned yet</div>`;
    const ch = d.confidence_hist;
    $("#confChart").innerHTML = svgBars(ch.counts, ch.labels, { colors: ch.labels.map(l => parseFloat(l) >= 0.5 ? "var(--good)" : "var(--warn)"), height: 210 });
    const lp = d.length_profile, lk = Object.keys(lp);
    $("#lenChart").innerHTML = lk.length ? svgBars(lk.map(k => lp[k]), lk.map(k => k + "c"), { colors: lk.map(() => "var(--accent)") }) : `<div class="muted">no data yet</div>`;
    const hp = d.hour_profile;
    $("#hourChart").innerHTML = svgBars(hp.counts, hp.hours.map(String), { colors: hp.hours.map(() => "var(--warn)"), height: 170 });
    const ot = d.over_time;
    $("#timeChart").innerHTML = ot.days.length ? svgLines(ot.days, [{ data: ot.memory, color: "var(--mem)" }, { data: ot.matcher, color: "var(--matcher)" }, { data: ot.llm, color: "var(--llm)" }], { width: 960 }) : `<div class="muted">no data yet</div>`;
  },

  async learning() {
    const d = await j("/api/learning");
    $("#learnCards").innerHTML = [
      ["mem", d.pairs, "learned pairs"],
      ["matcher", d.whitelist_size, "whitelist size"],
      ["good", d.layers.without_llm_pct + "%", "without LLM"],
      ["llm", fmtTs(d.last_compaction), "last compaction"],
    ].map(([c, v, l]) => `<div class="card"><div class="val ${c}" style="font-size:${String(v).length > 8 ? 14 : 27}px">${v}</div><div class="lbl">${l}</div></div>`).join("");
    const ot = d.over_time;
    $("#noLlmChart").innerHTML = ot.days.length ? svgLines(ot.days, [{ data: ot.no_llm_share, color: "var(--good)" }], { max: 100, width: 960 }) : `<div class="muted">no data yet, keep typing</div>`;
  },

  async model() {
    const d = await j("/api/model");
    $("#promptBox").textContent = d.prompt;
    const sel = $("#modelSelect");
    sel.innerHTML = (d.installed.length ? d.installed : [d.active]).map(m => `<option ${m === d.active ? "selected" : ""}>${esc(m)}</option>`).join("");
    sel.onchange = async () => { await post("/api/model", { model: sel.value }); toast("Layer 3 model set to " + sel.value); refreshHeader(); };
    renderBench(d.last_benchmark);
  },

  async settings() {
    const s = await j("/api/settings");
    const rng = (k, min, max, step) => `<div class="field"><label>${labelFor(k)}</label><div class="rangewrap"><input type="range" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${s[k]}"><span class="rval">${s[k]}</span></div><span class="hint">${hintFor(k)}</span></div>`;
    const num = (k) => `<div class="field"><label>${labelFor(k)}</label><input type="number" data-k="${k}" value="${esc(s[k])}"><span class="hint">${hintFor(k)}</span></div>`;
    const txt = (k) => `<div class="field"><label>${labelFor(k)}</label><input type="text" data-k="${k}" value="${esc(s[k])}"><span class="hint">${hintFor(k)}</span></div>`;
    const bool = (k) => `<div class="field"><label>${labelFor(k)}</label><label class="muted"><input type="checkbox" data-k="${k}" ${s[k] ? "checked" : ""} style="width:auto"> enabled</label><span class="hint">${hintFor(k)}</span></div>`;
    $("#settingsForm").innerHTML = [
      bool("stage2_enabled"), bool("llm_only"),
      rng("layer2_apply_confidence", 0.5, 0.95, 0.01),
      rng("layer1_apply_confidence", 0.2, 0.9, 0.01), num("min_word_len"),
      num("stage2_idle_ms"), num("stage2_max_drift_chars"),
      num("undo_window_s"), txt("ollama_url"),
      txt("dashboard_hostname"), num("dashboard_port"),
    ].join("");
    $$("#settingsForm input[type=range]").forEach(r => r.oninput = () => r.nextElementSibling.textContent = r.value);
  },

  async rawlog() {
    const d = await j("/api/rawlog");
    $("#corrTable").innerHTML = `<tr><th>time</th><th>from</th><th>to</th><th>layer</th></tr>` +
      d.corrections.slice().reverse().map(c => `<tr><td class="muted">${hhmm(c.ts)}</td><td class="m">${esc(c.original)}</td><td>${c.undone ? '<s class="muted">' + esc(c.corrected) + '</s>' : esc(c.corrected)}</td><td><span class="chip ${c.stage}">${esc(c.stage)}</span></td></tr>`).join("");
    $("#rawTable").innerHTML = `<tr><th>time</th><th>sentence</th></tr>` +
      d.raw.map(r => `<tr><td class="muted">${hhmm(r.ts)}</td><td>${esc(r.text)}</td></tr>`).join("");
  },

  async health() {
    const d = await j("/api/health");
    $("#healthCards").innerHTML = [
      [d.engine_enabled ? "good" : "bad", d.engine_enabled ? "ON" : "OFF", "engine"],
      ["accent", esc(d.model), "active model"],
      ["matcher", d.layer3_queue, "Layer 3 queue depth"],
      [d.stage2_enabled ? "good" : "dim", d.stage2_enabled ? "on" : "off", "Layer 3 enabled"],
      [d.layer3_last_error ? "bad" : "good", d.layer3_last_error ? "error" : "ok", "Layer 3 status"],
    ].map(([c, v, l]) => `<div class="card"><div class="val ${c}" style="font-size:${String(v).length > 9 ? 14 : 27}px">${v}</div><div class="lbl">${l}</div></div>`).join("");
  },
};

const LABELS = { stage2_enabled: "Layer 3 (context LLM)", llm_only: "LLM-only mode", layer2_apply_confidence: "Layer 2 apply threshold", layer1_apply_confidence: "Layer 1 apply threshold", min_word_len: "Minimum word length", stage2_idle_ms: "Idle pause before LLM apply (ms)", stage2_max_drift_chars: "Max cursor drift (chars)", undo_window_s: "Undo window (seconds)", ollama_url: "Ollama URL", dashboard_hostname: "Dashboard hostname", dashboard_port: "Dashboard port" };
const HINTS = { llm_only: "skip the fast memory + keyboard layers; the LLM fixes every word (slower, more context-aware)", layer2_apply_confidence: "higher means Layer 2 defers more to the LLM", layer1_apply_confidence: "confidence a learned pair needs to auto-apply", min_word_len: "shorter words are never corrected", stage2_idle_ms: "how long you must pause before an LLM fix lands", stage2_max_drift_chars: "drop the LLM fix if you typed past this", undo_window_s: "backspace within this to undo and whitelist", ollama_url: "use 127.0.0.1, not localhost", dashboard_hostname: "run setup_hosts.py to map it", dashboard_port: "80 for a bare hostname, else a fallback" };
const labelFor = k => LABELS[k] || k;
const hintFor = k => HINTS[k] || "";

function renderTypos() {
  const filt = ($("#typoSearch").value || "").toLowerCase();
  const sort = window._typoSort;
  let rows = (window._typos || []).filter(r => !filt || r.mangled.includes(filt) || r.intended.includes(filt));
  rows.sort((a, b) => { const x = a[sort.k], y = b[sort.k]; return (x < y ? -1 : x > y ? 1 : 0) * sort.d; });
  const conf = c => `<span style="color:${c >= 0.5 ? "var(--good)" : "var(--warn)"}">${c.toFixed(2)}</span>`;
  const th = (k, t) => `<th data-k="${k}">${t}${sort.k === k ? (sort.d < 0 ? " v" : " ^") : ""}</th>`;
  $("#typoTable").innerHTML = `<tr>${th("mangled", "mangled")}${th("intended", "intended")}${th("count", "count")}${th("confidence", "conf")}${th("source", "source")}<th></th></tr>` +
    rows.map(r => `<tr><td class="m">${esc(r.mangled)}</td><td>${esc(r.intended)}</td><td>${r.count}</td><td>${conf(r.confidence)}</td><td><span class="chip">${esc(r.source)}</span></td><td><span class="tag-del" data-m="${esc(r.mangled)}" data-i="${esc(r.intended)}">delete</span></td></tr>`).join("");
  $$("#typoTable th[data-k]").forEach(h => h.onclick = () => { const k = h.dataset.k; window._typoSort = { k, d: sort.k === k ? -sort.d : (k === "count" || k === "confidence" ? -1 : 1) }; renderTypos(); });
  $$("#typoTable .tag-del").forEach(b => b.onclick = async () => { await del("/api/typos", { mangled: b.dataset.m, intended: b.dataset.i }); loaders.typos(); });
}

function renderBench(b) {
  if (!b || !b.length) { $("#benchResults").innerHTML = `<div class="muted">no benchmark yet. Run one above.</div>`; return; }
  $("#benchResults").innerHTML = `<div class="tablewrap"><table class="data"><tr><th>model</th><th>recovery</th><th>mash</th><th>passthru</th><th>discipline</th><th>eval p50</th><th>gate</th></tr>` +
    b.filter(s => s.ok).sort((a, c) => (c.qualifies - a.qualifies) || (c.recovery_acc - a.recovery_acc)).map(s => `<tr><td class="m">${esc(s.model)}</td><td>${(s.recovery_acc * 100).toFixed(0)}%</td><td>${s.mash_acc != null ? (s.mash_acc * 100).toFixed(0) + "%" : "-"}</td><td>${(s.passthrough_rate * 100).toFixed(0)}%</td><td>${(s.discipline_rate * 100).toFixed(0)}%</td><td>${s.eval_ms_p50.toFixed(0)}ms</td><td>${s.qualifies ? '<span style="color:var(--good)">PASS</span>' : '<span class="muted">no</span>'}</td></tr>`).join("") + `</table></div>`;
}

async function runPlayground() {
  const sentence = $("#pgInput").value;
  $("#pgMsg").textContent = "routing..."; $("#pgFinal").style.display = "none";
  const t0 = performance.now();
  const d = await post("/api/test", { sentence, use_llm: $("#pgLlm").checked });
  $("#pgMsg").textContent = `${Math.round(performance.now() - t0)} ms`;
  $("#pgOut").innerHTML = d.words.map(w => {
    if (w.kind === "sep") return esc(w.text);
    if (w.kind === "apply") return `<span class="pg-word apply ${w.layer === "matcher" ? "matcherL" : w.layer === "llm" ? "llmL" : ""}" title="${w.layer} ${w.confidence}"><s class="muted">${esc(w.text)}</s> <span class="fix">${esc(w.intended)}</span></span>`;
    if (w.kind === "defer") { const fix = d.llm[w.text]; return `<span class="pg-word ${fix && fix !== w.text ? "apply llmL" : "defer"}" title="deferred to Layer 3">${fix && fix !== w.text ? `<s class="muted">${esc(w.text)}</s> <span class="fix">${esc(fix)}</span>` : esc(w.text)}</span>`; }
    return `<span class="pg-word" title="passthrough">${esc(w.text)}</span>`;
  }).join("");
  let final = d.words.map(w => (w.kind === "apply" || (w.kind === "defer" && d.llm[w.text])) ? (w.intended || d.llm[w.text]) : w.text).join("");
  $("#pgFinal").innerHTML = `<span class="muted">result:</span> ${esc(final)}`;
  $("#pgFinal").style.display = "block";
}

function exportData() {
  Promise.all([j("/api/typos"), j("/api/whitelist")]).then(([typos, whitelist]) => {
    const blob = new Blob([JSON.stringify({ typos, whitelist }, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "sumizome-data.json"; a.click();
    toast("exported " + typos.length + " pairs and " + whitelist.length + " words");
  });
}
async function importData(file) {
  const data = JSON.parse(await file.text());
  for (const t of (data.typos || [])) await post("/api/typos", { mangled: t.mangled, intended: t.intended });
  for (const w of (data.whitelist || [])) await post("/api/whitelist", { word: w });
  toast("imported"); loaders.typos();
}

// ---------------------------------------------------------------- header + nav
async function refreshHeader() {
  try {
    const d = await j("/api/overview");
    const ep = $("#enginePill");
    ep.className = "pill click " + (d.engine_enabled ? "on" : "off");
    ep.querySelector("span:last-child").textContent = d.engine_enabled ? "engine on" : "engine off";
    $("#modelPill").querySelector("span:last-child").textContent = d.model;
  } catch (e) { /* server starting */ }
}
function show(name) {
  $$("nav a").forEach(a => a.classList.toggle("active", a.dataset.s === name));
  $$("main section").forEach(s => s.classList.toggle("active", s.id === name));
  if (loaders[name]) loaders[name]();
  window._active = name;
}
$$("nav a").forEach(a => a.onclick = () => show(a.dataset.s));
$("#enginePill").onclick = async () => { await post("/api/toggle"); refreshHeader(); if (window._active === "health") loaders.health(); };
$("#modelPill").onclick = () => show("model");
$("#typoSearch").oninput = renderTypos;
$("#pgRun").onclick = runPlayground;
$("#addTypo").onclick = async () => { const m = $("#newMangled").value.trim(), i = $("#newIntended").value.trim(); if (m && i) { await post("/api/typos", { mangled: m, intended: i }); $("#newMangled").value = $("#newIntended").value = ""; loaders.typos(); toast("added " + m + " to " + i); } };
$("#addWord").onclick = async () => { const w = $("#newWord").value.trim(); if (w) { await post("/api/whitelist", { word: w }); $("#newWord").value = ""; loaders.whitelist(); } };
$("#exportBtn").onclick = exportData;
$("#importBtn").onclick = () => $("#importFile").click();
$("#importFile").onchange = (e) => e.target.files[0] && importData(e.target.files[0]);
$("#compactBtn").onclick = async () => { $("#compactMsg").textContent = "running..."; const r = await post("/api/compact"); $("#compactMsg").textContent = `promoted ${r.promoted} pairs, whitelisted ${r.whitelisted} words`; loaders.learning(); };
$("#benchBtn").onclick = async () => { const r = await post("/api/benchmark"); $("#benchMsg").textContent = r.ok ? `benchmarking ${r.started}, refresh in a minute` : r.error; };
$("#saveSettings").onclick = async () => {
  const body = {};
  $$("#settingsForm [data-k]").forEach(el => { body[el.dataset.k] = el.type === "checkbox" ? el.checked : el.value; });
  await post("/api/settings", body); $("#settingsMsg").textContent = "saved"; toast("settings saved"); refreshHeader();
};

async function checkVersion() {
  try {
    const v = await j("/api/version");
    const b = $("#updateBanner");
    if (v.available) {
      b.textContent = `Update available: v${v.latest} (you have v${v.current}) - click to open the repo`;
      b.href = v.url; b.style.display = "block";
    } else { b.style.display = "none"; }
  } catch (e) { /* offline */ }
}

refreshHeader();
checkVersion();
show("overview");
setInterval(() => { refreshHeader(); const a = window._active; if (["overview", "health", "rawlog"].includes(a) && loaders[a]) loaders[a](); }, 6000);
