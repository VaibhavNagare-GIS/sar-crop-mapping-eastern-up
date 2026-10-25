import { loadECharts } from "./echarts-loader.js";

const section = document.querySelector("#voices");
const holder = document.querySelector("#voices-chart");
const readout = document.querySelector("#voices-read");
const cropBox = document.querySelector("#voices-crops");
const channelBox = document.querySelector("#voices-channels");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const base = location.origin + location.pathname.replace(/[^/]*$/, "");
const COLORS = { Wheat: "#009e73", Mustard: "#e0a800", Lentil: "#56b4e9", Sugarcane: "#d55e00", "No crop/Fallow": "#6b7c76" };
const FALLBACK = ["#cc79a7", "#0072b2", "#999999"];
const CHANNEL = { VV: "VV", VH: "VH", VHVV: "VH minus VV" };
const state = { channel: "VV", crops: new Set() };
let data, chart;

const colour = (c) => COLORS[c] || FALLBACK[[...Object.keys(data.series.VV)].indexOf(c) % FALLBACK.length];
const day = (iso, add = 0) => new Date(Date.parse(`${iso}T00:00:00Z`) + add * 864e5).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
const range = (i) => `${day(data.windows[i])} to ${day(data.windows[i], data.window_days - 1)}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);
const db = (v) => `${v.toFixed(1)} dB`;

function describe(i) {
  const s = data.series[state.channel], rows = [...state.crops].map((c) => [c, s[c].med[i]]).sort((a, b) => b[1] - a[1]);
  if (rows.length < 2) return rows.length ? `${range(i)}: ${rows[0][0]} is at ${db(rows[0][1])}.` : "Pick at least one crop.";
  const [hi, lo] = [rows[0], rows.at(-1)];
  return `${range(i)}: ${hi[0]} is highest at ${db(hi[1])}, ${lo[0]} is lowest at ${db(lo[1])}.`;
}

const isNarrow = () => holder.clientWidth < 560;
const layout = (n) => ({ xAxis: { axisLabel: { interval: n ? 2 : 1, rotate: n ? 35 : 0 } }, grid: { bottom: n ? 56 : 44 } });

function option() {
  const s = data.series[state.channel], narrow = isNarrow(), series = [];
  for (const c of state.crops) {
    const d = s[c], col = colour(c);
    series.push({ name: c, type: "line", data: d.med, symbol: "none", z: 3, lineStyle: { width: 3, color: col, type: c.includes("Fallow") ? "dashed" : "solid" }, itemStyle: { color: col }, emphasis: { disabled: true } });
    series.push({ name: `${c} low`, type: "line", data: d.lo, stack: c, stackStrategy: "all", symbol: "none", lineStyle: { opacity: 0 }, silent: true, z: 1 });
    series.push({ name: `${c} band`, type: "line", data: d.hi.map((h, i) => +(h - d.lo[i]).toFixed(2)), stack: c, stackStrategy: "all", symbol: "none", lineStyle: { opacity: 0 }, areaStyle: { color: col, opacity: 0.14 }, silent: true, z: 1 });
  }
  return {
    animation: !reduce, animationDuration: 1700, animationEasing: "cubicOut", textStyle: { fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif", color: "#1b3a33" },
    grid: { left: 52, right: 16, top: 28, bottom: narrow ? 56 : 44 },
    xAxis: { type: "category", boundaryGap: false, data: data.windows.map((w) => day(w)), axisLine: { lineStyle: { color: "#1b3a33" } }, axisLabel: { color: "#36544c", fontSize: 13, interval: narrow ? 2 : 1, rotate: narrow ? 35 : 0 } },
    yAxis: { type: "value", scale: true, name: "dB", nameTextStyle: { color: "#36544c", fontSize: 13 }, axisLabel: { color: "#36544c", fontSize: 13 }, splitLine: { lineStyle: { color: "rgba(27,58,51,.12)" } } },
    tooltip: { trigger: "axis", confine: true, backgroundColor: "#fff", borderColor: "rgba(27,58,51,.2)", textStyle: { color: "#1b3a33", fontSize: 14 }, axisPointer: { lineStyle: { color: "#1b3a33", width: 1.5 } },
      formatter: (p) => { const i = p[0].dataIndex; return `<b>${esc(range(i))}</b>` + [...state.crops].map((c) => `<div><span style="display:inline-block;width:.7em;height:.7em;border-radius:50%;background:${colour(c)};margin-right:.4em"></span>${esc(c)}: ${db(s[c].med[i])}</div>`).join(""); } },
    series,
  };
}

function draw() {
  chart.setOption(option(), true);
  readout.textContent = "Hover or tap the chart to compare a date.";
}

function controls() {
  const crops = Object.keys(data.series.VV);
  crops.forEach((c) => state.crops.add(c));
  for (const c of crops) {
    const label = document.createElement("label"), input = document.createElement("input"), span = document.createElement("span"), sw = document.createElement("i");
    label.className = "chip"; input.type = "checkbox"; input.value = c; input.checked = true; sw.className = "swatch"; sw.style.setProperty("--c", colour(c));
    span.append(sw, document.createTextNode(c)); label.append(input, span); cropBox.append(label);
  }
  for (const [k, v] of Object.entries(CHANNEL)) {
    const label = document.createElement("label"), input = document.createElement("input"), span = document.createElement("span");
    label.className = "chip"; input.type = "radio"; input.name = "channel"; input.value = k; input.checked = k === state.channel; span.textContent = v; label.append(input, span); channelBox.append(label);
  }
  section.querySelector(".voices__side").addEventListener("change", (e) => {
    if (e.target.name === "channel") state.channel = e.target.value;
    else if (e.target.checked) state.crops.add(e.target.value); else if (state.crops.size > 1) state.crops.delete(e.target.value); else e.target.checked = true;
    draw();
  });
}

async function start() {
  try {
    data = await (await fetch(`${base}data/signatures.json`)).json();
    const echarts = await loadECharts();
    holder.replaceChildren();
    const node = document.createElement("div");
    holder.append(node);
    chart = echarts.init(node, null, { renderer: "canvas" });
    controls();
    chart.on("updateAxisPointer", (e) => { const i = e.axesInfo?.[0]?.value; if (Number.isInteger(i)) readout.textContent = describe(i); });
    let wasNarrow = isNarrow();
    new ResizeObserver(() => { chart.resize(); const now = isNarrow(); if (now !== wasNarrow) { wasNarrow = now; chart.setOption(layout(now)); } }).observe(holder);
    draw();
  } catch (err) {
    console.error(err);
    holder.textContent = "The chart could not load. Reload the page to try again.";
  }
}

if (section) new IntersectionObserver((entries, io) => { if (entries[0].isIntersecting) { io.disconnect(); start(); } }, { threshold: 0.25 }).observe(holder);
