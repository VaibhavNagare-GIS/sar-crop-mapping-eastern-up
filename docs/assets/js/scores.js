import { loadECharts } from "./echarts-loader.js";

const section = document.querySelector("#scores");
const $ = (id) => document.querySelector(id);
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const base = location.origin + location.pathname.replace(/[^/]*$/, "");
const REGIONS = { test: ["Study block", "Fields held out from training, inside the study block."], near: ["Nearby", "A new area the model never saw."], far: ["Far", "A new area, farther away, that the model never saw."] };
const LEVELS = { smoothed_field_vote: "Field", pixel_smoothed: "Pixel, smoothed", pixel_raw: "Pixel, raw" };
const METRICS = [["f1", "Macro-F1"], ["bal", "Balanced accuracy"], ["oa", "Overall accuracy"], ["auc", "AUC"]];
const state = { region: "test", level: "smoothed_field_vote" };
const results = {};
let matrix;

const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
const f2 = (v) => (v == null ? "n/a" : v.toFixed(2)), f3 = (v) => (v == null ? "n/a" : v.toFixed(3));
const chips = (box, name, items) => {
  for (const [k, v] of items) {
    const label = el("label", "", "chip"), input = el("input"), span = el("span", v);
    Object.assign(input, { type: "radio", name, value: k, checked: state[name] === k });
    label.append(input, span);
    box.append(label);
  }
};

function count(node, to) {
  const from = parseFloat(node.dataset.v) || 0;
  node.dataset.v = to ?? 0;
  if (reduce || to == null) { node.textContent = f3(to); return; }
  const t0 = performance.now();
  const step = (t) => { const k = Math.min(1, (t - t0) / 700), e = 1 - (1 - k) ** 3; node.textContent = f3(from + (to - from) * e); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

function summary(r) {
  const m = r[state.level], box = $("#scores-big");
  if (!box.children.length) for (const [k, name] of [["f1", "Macro-F1"], ["oa", "Overall accuracy"], ["auc", "AUC"]]) {
    const d = el("div", "", "big");
    d.append(el("span", "0.000", "big__v"), el("span", name, "big__k"), el("span", "", "big__ci"));
    d.dataset.k = k;
    box.append(d);
  }
  for (const d of box.children) {
    const k = d.dataset.k, ci = m.ci[k] || [null, null];
    count(d.firstChild, m[k]);
    d.lastChild.textContent = m[k] == null ? "" : `95% range ${f3(ci[0])} to ${f3(ci[1])}`;
  }
  const [lo, hi] = m.ci.oa || [null, null], b = r.baseline_oa;
  $("#scores-verdict").textContent = lo == null || hi == null ? `Overall accuracy could not be compared with always guessing the biggest class (${f3(b)}).`
    : lo > b ? `Overall accuracy beats always guessing the biggest class (${f3(b)}).`
    : hi < b ? `Overall accuracy is below always guessing the biggest class (${f3(b)}), so read macro-F1 and AUC instead.`
    : `Overall accuracy is not clearly different from always guessing the biggest class (${f3(b)}).`;
}

function bars(r) {
  const m = r[state.level], box = $("#scores-bars");
  if (!box.children.length) for (const [k, name] of METRICS) {
    const row = el("div", "", "bar"), track = el("div", "", "bar__track");
    track.append(el("i", "", "bar__range"), el("b", "", "bar__dot"));
    if (k === "oa") track.append(el("u", "", "bar__base"));
    row.append(el("span", name, "bar__name"), track, el("span", "", "bar__val"));
    row.dataset.k = k;
    box.append(row);
  }
  for (const row of box.children) {
    const k = row.dataset.k, [lo, hi] = m.ci[k] || [m[k], m[k]], q = row.children[1];
    row.hidden = m[k] == null;
    if (m[k] == null) continue;
    q.children[0].style.cssText = `left:${lo * 100}%;width:${(hi - lo) * 100}%`;
    q.children[1].style.left = `${m[k] * 100}%`;
    if (k === "oa") q.children[2].style.left = `${r.baseline_oa * 100}%`;
    row.lastChild.textContent = f2(m[k]);
  }
}

function classTable(r) {
  const body = $("#scores-class tbody"), pc = r[state.level].per_class;
  body.replaceChildren();
  for (const [name, v] of Object.entries(pc).filter(([n]) => r.classes_present.includes(n))) {
    const tr = el("tr");
    tr.append(el("th", name), ...[v.PA, v.UA, v.F1, v.AUC].map((x) => el("td", f2(x))));
    tr.firstChild.scope = "row";
    body.append(tr);
  }
}

function notes(r) {
  const list = $("#scores-notes");
  list.replaceChildren();
  const add = (t) => list.append(el("li", t));
  add(`${r.n_fields.toLocaleString("en-GB")} fields and ${r.n_pixels.toLocaleString("en-GB")} pixels were scored. ${REGIONS[state.region][1]}`);
  if (r.classes_present.length < 3) add(`Only ${r.classes_present.length} classes occur here (${r.classes_present.join(" and ")}), so this score is not comparable with areas that have all three.`);
  if (r.second_look) add("This test set had already been scored once earlier in the project, so treat this as a second look.");
}

async function drawMatrix() {
  const csv = await (await fetch(`${base}data/results/confusion_final_${state.region}_${state.level}.csv`)).text();
  const rows = csv.trim().split(/\r?\n/).map((l) => l.split(","));
  const cols = rows[0].slice(1), names = rows.slice(1).map((r) => r[0]), n = rows.slice(1).map((r) => r.slice(1).map(Number));
  const cells = [];
  n.forEach((r, y) => { const tot = r.reduce((a, b) => a + b, 0) || 1; r.forEach((c, x) => { const p = +(c / tot * 100).toFixed(1); cells.push({ value: [x, y, p, c], label: { color: p > 55 ? "#0e1d47" : "#f6faf2" } }); }); });
  matrix.setOption({
    animation: !reduce, textStyle: { fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif", color: "#f6faf2" },
    title: { text: "Survey says", left: 4, top: 0, textStyle: { color: "#f6faf2", fontSize: 14, fontWeight: "normal" } },
    grid: { left: 8, right: 8, top: 34, bottom: 56, containLabel: true },
    xAxis: { type: "category", data: cols, name: "Model said", nameLocation: "middle", nameGap: 34, nameTextStyle: { color: "#f6faf2", fontSize: 14 }, axisLabel: { color: "#f6faf2", fontSize: 13 }, axisLine: { show: false }, axisTick: { show: false }, splitArea: { show: false } },
    yAxis: { type: "category", data: names, inverse: true, axisLabel: { color: "#f6faf2", fontSize: 13 }, axisLine: { show: false }, axisTick: { show: false } },
    visualMap: { show: false, min: 0, max: 100, dimension: 2, inRange: { color: ["#1c3a73", "#2f6f9f", "#6bb5a8", "#ffe39a"] } },
    tooltip: { confine: true, backgroundColor: "#fff", textStyle: { color: "#1b3a33", fontSize: 14 }, formatter: (p) => `Survey says <b>${names[p.value[1]]}</b>, model said <b>${cols[p.value[0]]}</b>: ${p.value[3]} (${p.value[2]}% of that row)` },
    series: [{ type: "heatmap", data: cells, itemStyle: { borderColor: "#0e1d47", borderWidth: 3, borderRadius: 4 }, label: { show: true, fontSize: 14, fontWeight: 700, formatter: (p) => `${p.value[3]}\n${Math.round(p.value[2])}%` }, emphasis: { itemStyle: { borderColor: "#f6faf2" } } }],
  }, true);
}

async function render() {
  const r = results[state.region];
  summary(r); bars(r); classTable(r); notes(r);
  if (matrix) await drawMatrix().catch((e) => console.error(e));
}

async function start() {
  try {
    for (const k of Object.keys(REGIONS)) results[k] = await (await fetch(`${base}data/results/final_${k}.json`)).json();
    chips($("#scores-region"), "region", Object.entries(REGIONS).map(([k, v]) => [k, v[0]]));
    chips($("#scores-level"), "level", Object.entries(LEVELS));
    section.querySelector(".scores__controls").addEventListener("change", (e) => { state[e.target.name] = e.target.value; render(); });
    const echarts = await loadECharts();
    matrix = echarts.init($("#scores-matrix"), null, { renderer: "canvas" });
    new ResizeObserver(() => matrix.resize()).observe($("#scores-matrix"));
    await render();
  } catch (err) {
    console.error(err);
    $("#scores-verdict").textContent = "The scores could not load. Reload the page to try again.";
  }
}

if (section) new IntersectionObserver((entries, io) => { if (entries[0].isIntersecting) { io.disconnect(); start(); } }, { rootMargin: "300px" }).observe(section);
