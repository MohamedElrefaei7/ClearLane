// ClearLane map: MapLibre basemap + deck.gl H3HexagonLayer, reading the M9 API.
// Everything is precomputed server-side; this file only fetches and draws.

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const LANE_COLORS = { protected: "#2a78d6", curbside: "#eb6834", buffered: "#1baf7a", conventional: "#9ec5f4" };
const RAMP_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"];
const STYLES = {
  light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
};

const state = { how: 8, layer: "predicted", playing: false, selected: null, meta: null, grid: null };
const slotCache = new Map();
// ?theme=light|dark forces a theme (sets data-theme before anything renders).
const themeParam = new URLSearchParams(location.search).get("theme");
if (themeParam === "light" || themeParam === "dark") document.documentElement.dataset.theme = themeParam;
// Theme: an explicit data-theme on <html> wins; otherwise follow the system setting (same rule as the CSS).
const dark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};
// Sequential ramp: light→dark on a light surface; flipped on dark so low values recede.
const ramp = () => (dark() ? [...RAMP_LIGHT].reverse() : RAMP_LIGHT);
const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function binOf(v, edges) {
  let b = 0;
  while (b < edges.length && v >= edges[b]) b++;
  return b; // 0..edges.length
}

function fmt(v) {
  if (v === 0) return "0";
  if (v < 0.001) return v.toExponential(1);
  if (v < 0.1) return v.toFixed(3);
  return v.toFixed(2);
}

function hourLabel(h) {
  const ampm = h < 12 ? "AM" : "PM";
  return `${h % 12 === 0 ? 12 : h % 12}:00 ${ampm}`;
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function slot(how, layer) {
  const key = `${layer}:${how}`;
  if (!slotCache.has(key)) {
    slotCache.set(key, api(`/api/slot?how=${how}&layer=${layer}`).then((d) => {
      const m = new Map(d.cells.map((c) => [c.cell, c.value]));
      return { values: m, data: d };
    }));
  }
  return slotCache.get(key);
}

// ---------------------------------------------------------------- map + layers

const map = new maplibregl.Map({
  container: "map",
  style: dark() ? STYLES.dark : STYLES.light,
  center: [-73.95, 40.71],
  zoom: 10.4,
  attributionControl: { compact: true },
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
const overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
map.addControl(overlay);

const tip = document.getElementById("tip");

async function render() {
  const { values } = await slot(state.how, state.layer);
  const edges = state.meta.scale[state.layer];
  const colors = ramp().map(hex2rgb);
  const grey = hex2rgb(cssVar("--nolane"));
  const sel = state.selected;
  overlay.setProps({
    layers: [
      new deck.H3HexagonLayer({
        id: "nolane",
        data: state.grid.off_network,
        getHexagon: (d) => d,
        getFillColor: [...grey, dark() ? 110 : 120],
        stroked: false,
        extruded: false,
        pickable: false,
      }),
      new deck.H3HexagonLayer({
        id: "risk",
        data: state.grid.network,
        getHexagon: (d) => d,
        getFillColor: (d) => [...colors[binOf(values.get(d) ?? 0, edges)], 215],
        getLineColor: (d) => (d === sel ? hex2rgb(cssVar("--text")) : [0, 0, 0, 0]),
        getLineWidth: (d) => (d === sel ? 3 : 0),
        lineWidthUnits: "pixels",
        stroked: true,
        extruded: false,
        pickable: true,
        updateTriggers: { getFillColor: [state.how, state.layer, dark()], getLineColor: [sel], getLineWidth: [sel] },
        onHover: ({ object, x, y }) => {
          if (!object) { tip.style.display = "none"; return; }
          const v = values.get(object) ?? 0;
          tip.innerHTML = `<strong>${fmt(v)}</strong> expected reports / week<br>${DAYS[Math.floor(state.how / 24)]} ${hourLabel(state.how % 24)} · ${layerName(state.layer)}`;
          tip.style.left = `${x + 12}px`;
          tip.style.top = `${y + 12}px`;
          tip.style.display = "block";
        },
        onClick: ({ object }) => object && selectCell(object),
      }),
    ],
  });
  if (state.selected) drawSparkRule();
}

function layerName(l) {
  return { predicted: "Predicted", adjusted: "Reporting-adjusted (heuristic)", historical: "Historical" }[l];
}

// ---------------------------------------------------------------- controls

function setSlot(how) {
  state.how = ((how % 168) + 168) % 168;
  const d = Math.floor(state.how / 24), h = state.how % 24;
  document.querySelectorAll("#days button").forEach((b, i) => b.setAttribute("aria-pressed", String(i === d)));
  document.getElementById("hour").value = h;
  document.getElementById("hourLabel").textContent = hourLabel(h);
  render();
  // Prefetch the next slot so play-week stays smooth.
  slot((state.how + 1) % 168, state.layer);
}

function buildControls() {
  const days = document.getElementById("days");
  DAYS.forEach((name, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = name;
    b.addEventListener("click", () => setSlot(i * 24 + (state.how % 24)));
    days.appendChild(b);
  });
  document.getElementById("hour").addEventListener("input", (e) => setSlot(Math.floor(state.how / 24) * 24 + Number(e.target.value)));
  document.querySelectorAll('input[name="layer"]').forEach((r) =>
    r.addEventListener("change", (e) => {
      state.layer = e.target.value;
      updateLegend();
      render();
      if (state.selected) selectCell(state.selected);
    }));
  let timer = null;
  const play = document.getElementById("play");
  play.addEventListener("click", () => {
    state.playing = !state.playing;
    play.setAttribute("aria-pressed", String(state.playing));
    play.classList.toggle("on", state.playing);
    play.textContent = state.playing ? "❚❚ Pause" : "▶ Play week";
    clearInterval(timer);
    if (state.playing) timer = setInterval(() => setSlot(state.how + 1), 450);
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type === "range") return;
    if (e.key === "ArrowRight") setSlot(state.how + 1);
    if (e.key === "ArrowLeft") setSlot(state.how - 1);
  });
  const about = document.getElementById("about");
  document.getElementById("aboutBtn").addEventListener("click", () => about.showModal());
}

function updateLegend() {
  const edges = state.meta.scale[state.layer];
  const r = document.getElementById("ramp");
  r.innerHTML = "";
  ramp().forEach((c, i) => {
    const s = document.createElement("span");
    s.style.background = c;
    const lo = i === 0 ? 0 : edges[i - 1];
    s.title = i === edges.length ? `≥ ${fmt(lo)}` : `${fmt(lo)} – ${fmt(edges[i])}`;
    r.appendChild(s);
  });
  document.getElementById("legendNote").textContent =
    state.layer === "adjusted"
      ? "Reporting-adjusted (heuristic): predicted ÷ the area's overall 311 reporting index."
      : "Expected reports per week at the selected hour. Predicted and historical share one scale.";
}

// ---------------------------------------------------------------- detail panel

async function selectCell(cell) {
  state.selected = cell;
  render();
  const d = await api(`/api/cell/${cell}`);
  state.detail = d;
  const lanes = Object.entries(d.lane_share).filter(([, v]) => v > 0);
  const histHours = d.historical_hours.reduce((a, b) => a + b, 0);
  const panel = document.getElementById("panel");
  panel.innerHTML = `
    <h2>Cell detail</h2>
    <div class="muted">${d.borough} · <code>${d.cell}</code></div>
    <div class="lanes">
      <div class="muted" style="margin-bottom:4px">Lanes (${Math.round(Object.values(d.lane_metres).reduce((a, b) => a + b, 0))} m)</div>
      <div class="lanebar">${lanes.map(([k, v]) => `<div style="width:${v * 100}%;background:${LANE_COLORS[k]}" title="${k} ${Math.round(v * 100)}%"></div>`).join("")}</div>
      <div class="lanelegend">${lanes.map(([k, v]) => `<span><span class="sw" style="background:${LANE_COLORS[k]}"></span>${k} ${Math.round(v * 100)}%</span>`).join("")}</div>
    </div>
    <dl class="kv">
      <dt>Predicted</dt><dd>${fmt(d.weekly.predicted)} / wk</dd>
      <dt>Observed (last 12 months)</dt><dd>${histHours ? fmt(d.weekly.historical) + " / wk" : "no history"}</dd>
      <dt>Reporting-adjusted*</dt><dd>${fmt(d.weekly.adjusted)} / wk</dd>
      <dt>311 reporting index</dt><dd>${d.propensity_index.toFixed(2)}×</dd>
    </dl>
    <div class="muted" style="margin-bottom:4px">Reports per week, by hour across the week</div>
    <svg id="spark" role="img" aria-label="Expected and observed reports for each of the 168 hours of the week"></svg>
    <div class="sparklegend"><span><i style="background:var(--accent)"></i>Predicted</span><span><i style="background:var(--series-hist)"></i>Observed (12 mo)</span></div>
    <p class="muted" style="margin-top:12px">All values are <strong>reported</strong> obstruction (311), not observed obstruction.</p>`;
  drawSpark(d);
}

function drawSpark(d) {
  const svg = document.getElementById("spark");
  const W = svg.clientWidth || 300, H = 120, pad = { l: 4, r: 4, t: 8, b: 18 };
  const pred = d.series.predicted, hist = d.series.historical;
  const ymax = Math.max(...pred, ...hist, 1e-9);
  const x = (i) => pad.l + (i / 167) * (W - pad.l - pad.r);
  const y = (v) => H - pad.b - (v / ymax) * (H - pad.t - pad.b);
  const path = (s) => s.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const grid = cssVar("--border"), ink = cssVar("--text-2");
  let ticks = "";
  for (let k = 0; k < 7; k++) {
    const xi = x(k * 24);
    ticks += `<line x1="${xi}" x2="${xi}" y1="${pad.t}" y2="${H - pad.b}" stroke="${grid}" stroke-width="1"/>`;
    ticks += `<text x="${xi + 3}" y="${H - 4}" font-size="10" fill="${ink}">${DAYS[k]}</text>`;
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `${ticks}
    <path d="${path(hist)}" fill="none" stroke="${cssVar("--series-hist")}" stroke-width="1.5" opacity="0.8"/>
    <path d="${path(pred)}" fill="none" stroke="${cssVar("--accent")}" stroke-width="2"/>
    <line id="sparkRule" y1="${pad.t}" y2="${H - pad.b}" stroke="${cssVar("--text")}" stroke-width="1" stroke-dasharray="2,2"/>
    <rect id="sparkHit" x="0" y="0" width="${W}" height="${H}" fill="transparent"/>`;
  svg._x = x;
  drawSparkRule();
  const hit = svg.querySelector("#sparkHit");
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(167, Math.round(((e.clientX - r.left - pad.l) / (W - pad.l - pad.r)) * 167)));
    const rect = document.getElementById("map").getBoundingClientRect();
    tip.innerHTML = `${DAYS[Math.floor(i / 24)]} ${hourLabel(i % 24)}<br>Predicted <strong>${fmt(pred[i])}</strong> · Observed <strong>${fmt(hist[i])}</strong>`;
    tip.style.left = `${Math.min(e.clientX - rect.left - 160, rect.width - 220)}px`;
    tip.style.top = `${e.clientY - rect.top - 50}px`;
    tip.style.display = "block";
  });
  hit.addEventListener("mouseleave", () => (tip.style.display = "none"));
  hit.addEventListener("click", (e) => {
    const r = svg.getBoundingClientRect();
    setSlot(Math.round(((e.clientX - r.left - pad.l) / (W - pad.l - pad.r)) * 167));
  });
}

function drawSparkRule() {
  const svg = document.getElementById("spark");
  const rule = svg && svg.querySelector("#sparkRule");
  if (rule && svg._x) {
    const xi = svg._x(state.how);
    rule.setAttribute("x1", xi);
    rule.setAttribute("x2", xi);
  }
}

// ---------------------------------------------------------------- boot

(async function boot() {
  const status = document.getElementById("status");
  try {
    [state.meta, state.grid] = await Promise.all([api("/api/meta"), api("/api/grid")]);
    status.textContent = `Model month ${state.meta.month} · ${state.meta.cells.toLocaleString()} cells on the network`;
    document.getElementById("aboutMeta").textContent =
      `Serving month ${state.meta.month}; weekly level factor ${state.meta.recalibration.factor.toFixed(3)} from reports ${state.meta.recalibration.window.join(" to ")}; generated ${state.meta.generated_at}. ${state.meta.caveat}`;
    buildControls();
    updateLegend();
    // Draw as soon as the data is here (the hexes don't need the basemap), and again once the
    // basemap finishes loading so deck.gl picks up any camera change made while the style loaded.
    setSlot(state.how);
    map.once("load", () => { map.triggerRepaint(); render(); });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      map.setStyle(dark() ? STYLES.dark : STYLES.light);
      updateLegend();
      render();
      if (state.detail) drawSpark(state.detail);
    });
  } catch (err) {
    status.textContent = `Could not load data: ${err.message}`;
  }
})();
