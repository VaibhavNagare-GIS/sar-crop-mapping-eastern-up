const section = document.querySelector("#map");
const canvas = document.querySelector("#map-canvas");
const form = document.querySelector(".mapbox__panel");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const base = location.origin + location.pathname.replace(/[^/]*$/, "");

const REGIONS = {
  study: { tiles: { class: "class", masked: "class_masked", conf: "conf" }, fields: ["fields_train", "fields_test"] },
  near: { tiles: { class: "class_near", conf: "conf_near" }, fields: ["fields_near"] },
  far: { tiles: { class: "class_far", conf: "conf_far" }, fields: ["fields_far"] },
};
const RIGHT = "#0072b2", WRONG = "#cc79a7", NEUTRAL = "#6b7c76";
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
let map, meta, ml, popupOnMap, ready = false;
const state = { region: "study", layer: "class", basemap: "map" };

class BasemapControl {
  onAdd() {
    this.el = el("div", "", "maplibregl-ctrl basemap");
    this.el.setAttribute("role", "group");
    this.el.setAttribute("aria-label", "Basemap");
    for (const [key, label] of [["map", "Map"], ["sat", "Satellite"]]) {
      const b = el("button", label);
      b.type = "button";
      b.dataset.key = key;
      b.addEventListener("click", () => { state.basemap = key; show(); });
      this.el.append(b);
    }
    this.sync();
    return this.el;
  }
  sync() { for (const b of this.el.children) b.setAttribute("aria-pressed", String(b.dataset.key === state.basemap)); }
  onRemove() { this.el.remove(); }
}
const basemapControl = new BasemapControl();

const el = (tag, text, cls) => { const n = document.createElement(tag); if (text) n.textContent = text; if (cls) n.className = cls; return n; };
const rows = (box, list) => { for (const [k, v] of list) { const p = el("p"); p.append(el("span", k, "pop__k"), document.createTextNode(v)); box.append(p); } };
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

function legend() {
  const box = document.querySelector("#legend");
  box.replaceChildren();
  if (state.layer === "conf") {
    const r = meta.confidence_ramp, bar = el("div", "", "legend__ramp"), lab = el("div", "", "legend__scale");
    bar.style.background = `linear-gradient(to right, ${r.colors.join(", ")})`;
    lab.append(el("span", `${Math.round(r.low * 100)}%`), el("span", "100%"));
    box.append(el("p", "Confidence in the winning class", "legend__title"), bar, lab);
    return;
  }
  box.append(el("p", "Predicted class", "legend__title"));
  for (const c of Object.values(meta.classes)) {
    const row = el("div", "", "legend__row"), sw = el("span", "", "legend__swatch");
    sw.style.background = c.color;
    row.append(sw, el("span", c.name));
    box.append(row);
  }
}

function fieldLayers() {
  return REGIONS[state.region].fields.filter((n) => meta.fields[n]);
}

function show() {
  const r = REGIONS[state.region];
  const wanted = r.tiles[state.layer] || r.tiles.class;
  for (const name of Object.keys(meta.tiles)) map.setLayoutProperty(`l_${name}`, "visibility", name === wanted ? "visible" : "none");
  const on = form.elements.fields.checked ? fieldLayers() : [];
  for (const name of Object.keys(meta.fields)) for (const p of ["ff", "fc", "fl"]) map.setLayoutProperty(`${p}_${name}`, "visibility", on.includes(name) ? "visible" : "none");
  form.querySelector('input[value="masked"]').disabled = !r.tiles.masked;
  const sat = state.basemap === "sat";
  basemapControl.sync();
  document.querySelector("#basemap-note").hidden = !sat;
  map.setLayoutProperty("sat", "visibility", sat ? "visible" : "none");
  map.setLayoutProperty("osm", "visibility", sat ? "none" : "visible");
  document.querySelector("#map-fields-note").hidden = !form.elements.fields.checked;
  legend();
}

function fit(duration) {
  const b = meta.tiles[REGIONS[state.region].tiles.class].bounds;
  map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 40, duration: reduce ? 0 : duration, essential: false });
}

function open(e, box) {
  popupOnMap?.remove();
  popupOnMap = new ml.Popup({ maxWidth: "280px" }).setLngLat(e.lngLat).setDOMContent(box).addTo(map);
}

function fieldPopup(e, f) {
  const p = f.properties, box = el("div", "", "pop");
  box.append(el("h3", p.crop));
  const list = [["Model said", p.pred ? `${p.pred}${p.conf != null ? ` (${Math.round(p.conf * 100)}% sure)` : ""}` : "No map pixels here"]];
  if (f.layer.id.endsWith("train")) list.push(["Status", "Training field: the model saw it, so it is not a test"]);
  else if (p.ok === 1) list.push(["Against the survey", "Right"]);
  else if (p.ok === 0) list.push(["Against the survey", `Wrong. The survey says ${p.target}`]);
  else list.push(["Against the survey", "Not scored: the model has no class for this crop"]);
  list.push(["Pixels", String(p.px)]);
  rows(box, list);
  open(e, box);
}

// Reads one pixel of a tile PNG at the tile's top zoom: returns [r, g, b] or null where the tile is empty.
async function pixelAt(name, ll) {
  const z = meta.tiles[name].maxzoom, n = 2 ** z, s = Math.sin((ll.lat * Math.PI) / 180);
  const x = ((ll.lng + 180) / 360) * n, y = (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * n;
  const tx = Math.floor(x), ty = Math.floor(y);
  const res = await fetch(`${base}tiles/${name}/${z}/${tx}/${ty}.png`);
  if (!res.ok) return null;
  const bmp = await createImageBitmap(await res.blob());
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(Math.floor((x - tx) * 256), Math.floor((y - ty) * 256), 1, 1).data;
  return d[3] ? [d[0], d[1], d[2]] : null;
}

const nearest = (rgb, palette) => {
  let best = null, bd = 1e9;
  for (const p of palette) { const d = p.rgb.reduce((s, v, i) => s + (v - rgb[i]) ** 2, 0); if (d < bd) { bd = d; best = p; } }
  return bd < 900 ? best : null;
};

function confidenceOf(rgb) {
  const r = meta.confidence_ramp, stops = r.colors.map(hex), low = Math.round(r.low * 255), pal = [];
  for (let i = low; i < 256; i++) {
    const t = ((i - low) / (255 - low)) * (stops.length - 1), k = Math.min(Math.floor(t), stops.length - 2), f = t - k;
    pal.push({ rgb: stops[k].map((v, j) => v * (1 - f) + stops[k + 1][j] * f), v: i / 255 });
  }
  return nearest(rgb, pal)?.v ?? null;
}

async function pixelPopup(e) {
  const t = REGIONS[state.region].tiles, classes = Object.values(meta.classes).map((c) => ({ rgb: hex(c.color), name: c.name }));
  const [base0, masked, conf] = await Promise.all([pixelAt(t.class, e.lngLat), state.layer === "masked" ? pixelAt(t.masked, e.lngLat) : null, pixelAt(t.conf, e.lngLat)]);
  const box = el("div", "", "pop"), cls = base0 && nearest(base0, classes);
  if (!cls) { box.append(el("h3", "No prediction here")); rows(box, [["Why", "This spot is outside the mapped areas"]]); return open(e, box); }
  const hidden = state.layer === "masked" && !masked, c = conf && confidenceOf(conf);
  box.append(el("h3", cls.name));
  rows(box, [["Confidence", c != null ? `${Math.round(c * 100)}%` : "Not available"], ...(hidden ? [["Farmland mask", "Hidden: the land-cover map says this is not farmland"]] : [])]);
  open(e, box);
}

function build() {
  map.addControl(basemapControl, "top-left");
  map.addControl(new ml.NavigationControl({ showCompass: false }), "top-right");
  map.addControl(new ml.ScaleControl({ unit: "metric" }), "bottom-left");
  for (const [name, t] of Object.entries(meta.tiles)) {
    map.addSource(`t_${name}`, { type: "raster", tiles: [`${base}tiles/${name}/{z}/{x}/{y}.png`], tileSize: 256, minzoom: t.minzoom, maxzoom: t.maxzoom, bounds: t.bounds });
    map.addLayer({ id: `l_${name}`, type: "raster", source: `t_${name}`, layout: { visibility: "none" }, paint: { "raster-resampling": "nearest", "raster-opacity": 0.85, "raster-fade-duration": 200 } });
  }
  for (const name of Object.keys(meta.fields)) {
    map.addSource(`f_${name}`, { type: "geojson", data: `${base}data/${name}.geojson` });
    const colour = name === "fields_train" ? NEUTRAL : ["case", ["==", ["get", "ok"], 1], RIGHT, ["==", ["get", "ok"], 0], WRONG, NEUTRAL];
    const widths = (a, b, c) => ["interpolate", ["linear"], ["zoom"], 9, a, 14, b, 17, c];
    const lay = { source: `f_${name}`, minzoom: 9, layout: { visibility: "none" } };
    map.addLayer({ ...lay, id: `ff_${name}`, type: "fill", paint: { "fill-color": colour, "fill-opacity": 0.01 } });
    map.addLayer({ ...lay, id: `fc_${name}`, type: "line", paint: { "line-color": "#ffffff", "line-width": widths(2, 3.6, 5), "line-opacity": 0.85 } });
    map.addLayer({ ...lay, id: `fl_${name}`, type: "line", paint: { "line-color": colour, "line-width": widths(0.6, 1.8, 3.2) } });
    map.on("mouseenter", `ff_${name}`, () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", `ff_${name}`, () => (map.getCanvas().style.cursor = ""));
  }
  map.on("click", (e) => {
    const hit = form.elements.fields.checked ? map.queryRenderedFeatures(e.point, { layers: fieldLayers().map((n) => `ff_${n}`) }) : [];
    if (hit.length) fieldPopup(e, hit[0]);
    else pixelPopup(e).catch((err) => console.error(err));
  });
  show();
}

form.addEventListener("input", (e) => {
  if (!ready) return;
  const n = e.target.name;
  if (n === "region") { state.region = e.target.value; if (!REGIONS[state.region].tiles[state.layer]) state.layer = "class"; form.elements.layer.value = state.layer; popupOnMap?.remove(); show(); fit(1600); }
  else if (n === "layer") { state.layer = e.target.value; show(); }
  else if (n === "fields") show();
  else if (n === "opacity") for (const name of Object.keys(meta.tiles)) map.setPaintProperty(`l_${name}`, "raster-opacity", e.target.value / 100);
});
form.addEventListener("submit", (e) => e.preventDefault());

async function init() {
  try {
    meta = await (await fetch(`${base}data/meta.json`)).json();
    meta.fields = {};
    await Promise.all(["fields_train", "fields_test", "fields_near", "fields_far"].map(async (n) => { if ((await fetch(`${base}data/${n}.geojson`, { method: "HEAD" })).ok) meta.fields[n] = true; }));
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = `${base}assets/vendor/maplibre/maplibre-gl.css`;
    document.head.append(css);
    ml = await import("../vendor/maplibre/maplibre-gl.mjs");
    canvas.textContent = "";
    const b = meta.tiles.class.bounds;
    map = new ml.Map({
      container: canvas, attributionControl: { compact: true, customAttribution: "Contains modified Copernicus Sentinel data 2021-2022" },
      center: [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], zoom: 6.2, maxZoom: 18, dragRotate: false, pitchWithRotate: false, cooperativeGestures: true,
      style: { version: 8, sources: {
          osm: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, maxzoom: 19, attribution: "&copy; OpenStreetMap contributors" },
          sat: { type: "raster", tiles: [ESRI], tileSize: 256, maxzoom: 16, attribution: 'Tiles &copy; <a href="https://www.esri.com" target="_blank" rel="noopener">Esri</a>, Maxar, Earthstar Geographics, and the GIS User Community' } },
        layers: [{ id: "bg", type: "background", paint: { "background-color": "#e8efe4" } },
          { id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.6, "raster-brightness-max": 0.96 } },
          { id: "sat", type: "raster", source: "sat", layout: { visibility: "none" } }] },
    });
    map.touchZoomRotate.disableRotation();
    map.on("error", (e) => { if (e.error?.status !== 404) console.error(e.error); });  // sparse tiles: a missing tile is empty, not an error
    map.on("load", () => { build(); ready = true; });
    new IntersectionObserver((entries, io) => { if (entries[0].isIntersecting) { io.disconnect(); const go = () => fit(2600); ready ? go() : map.once("load", go); } }, { threshold: 0.35 }).observe(canvas);
  } catch (err) {
    console.error(err);
    canvas.textContent = "The map could not load. Check your connection and reload the page.";
    canvas.classList.add("is-error");
  }
}

new IntersectionObserver((entries, io) => { if (entries[0].isIntersecting) { io.disconnect(); init(); } }, { rootMargin: "600px" }).observe(section);
