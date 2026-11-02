// autozoom editor: vanilla JS. Mirrors autozoom/style.py (layout) and
// autozoom/camera.py (spring camera) so the preview matches the render.
"use strict";

const $ = (id) => document.getElementById(id);
const S = {
  project: null, plan: null, info: null, sel: -1, mode: "preview",
  cam: null, camFps: 30, layout: null, saveTimer: null,
  style: { background: "sunset", padding: 8, radius: 16, shadow: true, frame: "none", aspect: "source", motion_blur: true },
};
const video = $("video");

// ---------------------------------------------------------------- utils
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const fmt = (t) => { t = Math.max(0, t || 0); const m = Math.floor(t / 60); return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`; };
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

// ---------------------------------------------------------------- layout (port of style.compute_layout)
const ASPECTS = { "source": null, "16:9": [16, 9], "4:3": [4, 3], "1:1": [1, 1], "4:5": [4, 5], "9:16": [9, 16] };
const even = (v) => { v = Math.round(v); return v - (v % 2); };
function computeLayout(srcW, srcH, st) {
  let ow = even(srcW), oh = even(srcH);
  const a = ASPECTS[st.aspect];
  if (a) { const sh = Math.min(srcW, srcH), r = a[0] / a[1]; if (r >= 1) { ow = even(sh * r); oh = even(sh); } else { ow = even(sh); oh = even(sh / r); } }
  const short = Math.min(ow, oh), unit = short / 1080;
  const pad = Math.round(st.padding / 100 * short);
  const bar = st.frame !== "none" ? Math.round(38 * unit) : 0;
  const aw = Math.max(16, ow - 2 * pad), ah = Math.max(16, oh - 2 * pad - bar);
  const ra = srcW / srcH, rc = aw / ah;
  const crop = Math.max(ra / rc, rc / ra) > 1.5;
  let cw, ch, bw, bh;
  if (crop) { cw = aw; ch = ah; if (cw / ch < ra) { bh = srcH; bw = srcH * cw / ch; } else { bw = srcW; bh = srcW * ch / cw; } }
  else { const s = Math.min(aw / srcW, ah / srcH); cw = Math.round(srcW * s); ch = Math.round(srcH * s); bw = srcW; bh = srcH; }
  const wx = Math.floor((ow - cw) / 2), wy = Math.floor((oh - ch - bar) / 2);
  const radius = Math.min(Math.round(st.radius * unit), Math.floor(cw / 2), Math.floor((ch + bar) / 2));
  return { ow, oh, unit, bar, cw, ch, bw, bh, wx, wy, radius, crop };
}

// ---------------------------------------------------------------- camera (port of camera.compute_camera)
function segmentAt(t) { for (const s of S.plan.segments) if (s.start <= t && t < s.end) return s; return null; }
function centerAt(s, t) {
  const p = s.path; if (!p || !p.length) return [s.center_x, s.center_y];
  if (t <= p[0][0]) return [p[0][1], p[0][2]];
  if (t >= p[p.length - 1][0]) return [p[p.length - 1][1], p[p.length - 1][2]];
  for (let i = 0; i < p.length - 1; i++) { const a = p[i], b = p[i + 1]; if (a[0] <= t && t <= b[0]) { const k = (t - a[0]) / Math.max(b[0] - a[0], 1e-9); return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]; } }
  return [s.center_x, s.center_y];
}
function focusAt(t) {
  const f = S.plan.focus; if (!f || !f.length) return null;
  if (t <= f[0][0]) return [f[0][1], f[0][2]];
  let lo = 0, hi = f.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (f[m][0] <= t) lo = m; else hi = m; }
  const a = f[lo], b = f[hi]; if (t >= b[0]) return [b[1], b[2]];
  const k = (t - a[0]) / Math.max(b[0] - a[0], 1e-9); return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}
function springStep(x, v, target, om, dt) {
  const e = x - target, w = Math.exp(-om * dt), tmp = (v + om * e) * dt;
  return [target + (e + tmp) * w, (v - om * tmp) * w];
}
function computeCamera() {
  const W = S.info.width, H = S.info.height, L = S.layout, fps = S.camFps, dt = 1 / fps;
  const n = Math.max(1, Math.ceil(S.info.duration * fps) + 1);
  const bw = L.bw, bh = L.bh, cropK = Math.min(bw / W, bh / H), follow = L.crop;
  const cl = (x, y, z) => { const hw = bw / z / 2, hh = bh / z / 2; return [clamp(x, hw, W - hw), clamp(y, hh, H - hh)]; };
  const target = (t) => {
    const s = segmentAt(t);
    if (s) { const c = centerAt(s, t); return [c[0] * W, c[1] * H, Math.max(1, s.zoom * cropK), true]; }
    if (follow) { const f = focusAt(t); if (f) return [f[0] * W, f[1] * H, 1, false]; }
    return [W / 2, H / 2, 1, false];
  };
  const cx = new Float64Array(n), cy = new Float64Array(n), zz = new Float64Array(n);
  let [tx, ty, tz] = target(0); let [x, y] = cl(tx, ty, tz); let lz = Math.log(tz), vx = 0, vy = 0, vz = 0;
  for (let i = 0; i < n; i++) {
    const t = i * dt; let inSeg; [tx, ty, tz, inSeg] = target(t); [tx, ty] = cl(tx, ty, tz);
    const om = (inSeg || !follow) ? 5.5 : 3.0;
    if (i > 0) { [x, vx] = springStep(x, vx, tx, om, dt); [y, vy] = springStep(y, vy, ty, om, dt); [lz, vz] = springStep(lz, vz, Math.log(tz), 5.5, dt); }
    const z = clamp(Math.exp(lz), 1, 16); const c = cl(x, y, z); cx[i] = c[0]; cy[i] = c[1]; zz[i] = z;
  }
  S.cam = { cx, cy, zz, n };
}
function viewAt(t) {
  const i = clamp(Math.round(t * S.camFps), 0, S.cam.n - 1), z = S.cam.zz[i];
  const w = S.layout.bw / z, h = S.layout.bh / z;
  return { x: S.cam.cx[i] - w / 2, y: S.cam.cy[i] - h / 2, w, h };
}

// ---------------------------------------------------------------- stage
function bgCss(spec) {
  const p = S.project.presets[spec];
  if (p) {
    const st = p.stops;
    return `radial-gradient(circle at 18% 28%, ${st[0]}aa 0, transparent 45%), radial-gradient(circle at 85% 75%, ${st[st.length - 1]}99 0, transparent 45%), linear-gradient(${p.angle}deg, ${st.join(", ")})`;
  }
  if (/^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.test(spec)) return spec.startsWith("#") ? spec : "#" + spec;
  return "repeating-linear-gradient(45deg,#333 0 10px,#2a2a2a 10px 20px)"; // image: shown at render time
}
function layoutStage() {
  const W = S.info.width, H = S.info.height;
  S.layout = computeLayout(W, H, S.style);
  const L = S.layout, stage = $("stage"), canvas = $("canvas"), win = $("win"), vp = $("viewport"), tb = $("titlebar");
  const sw = stage.clientWidth - 32, sh = stage.clientHeight - 32;
  if (S.mode === "full") {
    const k = Math.min(sw / W, sh / H);
    Object.assign(canvas.style, { width: W * k + "px", height: H * k + "px", background: "#000" });
    Object.assign(win.style, { left: 0, top: 0, width: W * k + "px", height: H * k + "px", borderRadius: 0, boxShadow: "none" });
    tb.style.display = "none"; vp.style.top = 0; vp.style.height = H * k + "px";
    S.k = k;
  } else {
    const k = Math.min(sw / L.ow, sh / L.oh);
    Object.assign(canvas.style, { width: L.ow * k + "px", height: L.oh * k + "px", background: bgCss(S.style.background) });
    Object.assign(win.style, { left: L.wx * k + "px", top: L.wy * k + "px", width: L.cw * k + "px", height: (L.ch + L.bar) * k + "px",
      borderRadius: L.radius * k + "px",
      boxShadow: S.style.shadow ? `0 ${18 * L.unit * k}px ${60 * L.unit * k}px rgba(0,0,0,.45), 0 ${4 * L.unit * k}px ${12 * L.unit * k}px rgba(0,0,0,.25)` : "none" });
    tb.style.display = L.bar ? "flex" : "none";
    tb.style.height = L.bar * k + "px";
    tb.style.background = S.style.frame === "dark" ? "linear-gradient(#3a3a3c,#2c2c2e)" : "linear-gradient(#f2f2f4,#e4e4e7)";
    tb.style.paddingLeft = 13.5 * L.unit * k + "px"; tb.style.gap = 7 * L.unit * k + "px";
    for (const d of tb.children) { d.style.width = d.style.height = 13 * L.unit * k + "px"; }
    vp.style.top = L.bar * k + "px"; vp.style.height = L.ch * k + "px";
    S.k = k;
  }
  video.style.width = W + "px"; video.style.height = H + "px";
  computeCamera();
  draw();
}
function draw() {
  const t = video.currentTime || 0, W = S.info.width, H = S.info.height, rect = $("viewRect");
  if (S.mode === "full") {
    const s = S.k; video.style.transform = `scale(${s})`;
    const seg = S.sel >= 0 ? S.plan.segments[S.sel] : null;
    let v;
    if (seg) { const c = centerAt(seg, clamp(t, seg.start, seg.end)); const z = Math.max(1, seg.zoom * Math.min(S.layout.bw / W, S.layout.bh / H)); const w = S.layout.bw / z, h = S.layout.bh / z; v = { x: clamp(c[0] * W - w / 2, 0, W - w), y: clamp(c[1] * H - h / 2, 0, H - h), w, h }; }
    else v = viewAt(t);
    rect.classList.remove("hidden");
    Object.assign(rect.style, { left: v.x * s + "px", top: v.y * s + "px", width: v.w * s + "px", height: v.h * s + "px" });
  } else {
    rect.classList.add("hidden");
    const v = viewAt(t), s = S.layout.cw * S.k / v.w;
    video.style.transform = `translate(${-v.x * s}px, ${-v.y * s}px) scale(${s})`;
  }
  $("time").textContent = `${fmt(t)} / ${fmt(S.info.duration)}`;
  $("playhead").style.left = (t / S.info.duration) * $("timeline").clientWidth + "px";
  $("play").innerHTML = video.paused ? "&#9654;" : "&#10074;&#10074;";
}
function loop() { draw(); requestAnimationFrame(loop); }

// ---------------------------------------------------------------- timeline
function renderRuler() {
  const r = $("ruler"), d = S.info.duration, w = r.clientWidth; r.innerHTML = "";
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]; const step = steps.find((s) => w / (d / s) >= 60) || 600;
  for (let t = 0; t <= d + 1e-6; t += step) {
    const x = t / d * w; const s = document.createElement("span"); s.style.left = x + "px"; s.textContent = fmt(t).replace(/\.0$/, "");
    const b = document.createElement("b"); b.style.left = x + "px"; r.append(s, b);
  }
}
function renderSegments() {
  const tr = $("track"), d = S.info.duration, w = tr.clientWidth; tr.innerHTML = "";
  S.plan.segments.forEach((s, i) => {
    const el = document.createElement("div"); el.className = "seg" + (i === S.sel ? " sel" : "");
    el.style.left = s.start / d * w + "px"; el.style.width = Math.max(6, (s.end - s.start) / d * w) + "px";
    el.innerHTML = `<div class="h l"></div><span>${s.zoom.toFixed(1)}x ${s.kind && s.kind !== "manual" ? "· " + s.kind : ""}</span><div class="h r"></div>`;
    el.addEventListener("pointerdown", (e) => startDrag(e, i, e.target.classList.contains("l") ? "l" : e.target.classList.contains("r") ? "r" : "m"));
    tr.appendChild(el);
  });
  renderInspector();
}
function startDrag(e, i, mode) {
  e.stopPropagation(); e.preventDefault(); select(i);
  const s = S.plan.segments[i], d = S.info.duration, w = $("track").clientWidth, x0 = e.clientX;
  const orig = { start: s.start, end: s.end, path: s.path ? s.path.map((p) => p.slice()) : null };
  const prev = S.plan.segments[i - 1], next = S.plan.segments[i + 1];
  const lo = prev ? prev.end : 0, hi = next ? next.start : d, MIN = 0.3;
  let moved = false;
  const onMove = (ev) => {
    const dt = (ev.clientX - x0) / w * d; if (Math.abs(ev.clientX - x0) > 2) moved = true;
    if (mode === "m") {
      const len = orig.end - orig.start; const ns = clamp(orig.start + dt, lo, hi - len);
      s.start = ns; s.end = ns + len;
      if (orig.path) s.path = orig.path.map((p) => [p[0] + (ns - orig.start), p[1], p[2]]);
    } else if (mode === "l") s.start = clamp(orig.start + dt, lo, s.end - MIN);
    else s.end = clamp(orig.end + dt, s.start + MIN, hi);
    renderSegments(); computeCamera();
  };
  const onUp = () => {
    window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp);
    if (moved) changed(); else { video.currentTime = s.start + 0.01; }
  };
  window.addEventListener("pointermove", onMove); window.addEventListener("pointerup", onUp);
}
function addSegment(t) {
  const d = S.info.duration, segs = S.plan.segments;
  let start = clamp(t, 0, d - 0.3), end = Math.min(d, start + 2.5);
  for (const s of segs) {
    if (s.start <= start && start < s.end) return select(segs.indexOf(s));
    if (s.start > start) { end = Math.min(end, s.start); break; }
  }
  if (end - start < 0.3) return;
  segs.push({ start, end, zoom: Math.min(2, S.plan.max_zoom || 2), center_x: 0.5, center_y: 0.5, kind: "manual" });
  segs.sort((a, b) => a.start - b.start);
  select(segs.findIndex((s) => s.start === start)); changed();
}
function select(i) { S.sel = i; renderSegments(); draw(); }
function renderInspector() {
  const s = S.sel >= 0 ? S.plan.segments[S.sel] : null;
  $("segEmpty").classList.toggle("hidden", !!s); $("segForm").classList.toggle("hidden", !s);
  if (!s) return;
  $("segStart").value = s.start.toFixed(2); $("segEnd").value = s.end.toFixed(2);
  $("segZoom").value = s.zoom; $("segZoomVal").textContent = s.zoom.toFixed(2) + "x";
  $("segCenter").textContent = `${s.center_x.toFixed(3)}, ${s.center_y.toFixed(3)}`;
  $("segPath").textContent = s.path ? `follows the action (${s.path.length} keyframes)` : "fixed center";
}
function changed() {
  computeCamera(); renderSegments(); draw();
  $("saveState").textContent = "Saving…";
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(async () => {
    try { await api("/api/plan", S.plan); $("saveState").textContent = "Saved"; }
    catch (e) { $("saveState").textContent = "Save failed: " + e.message; }
  }, 350);
}

// ---------------------------------------------------------------- events
function bind() {
  $("play").onclick = () => (video.paused ? video.play() : video.pause());
  $("addZoom").onclick = () => addSegment(video.currentTime);
  $("modePreview").onclick = () => { S.mode = "preview"; $("modePreview").classList.add("on"); $("modeFull").classList.remove("on"); layoutStage(); };
  $("modeFull").onclick = () => { S.mode = "full"; $("modeFull").classList.add("on"); $("modePreview").classList.remove("on"); layoutStage(); };
  const seekFromEvent = (e) => { const r = $("timeline").getBoundingClientRect(); video.currentTime = clamp((e.clientX - r.left) / r.width, 0, 1) * S.info.duration; };
  $("ruler").addEventListener("pointerdown", seekFromEvent);
  $("film").addEventListener("pointerdown", seekFromEvent);
  $("track").addEventListener("pointerdown", (e) => { if (e.target === $("track")) { select(-1); seekFromEvent(e); } });
  $("track").addEventListener("dblclick", (e) => { if (e.target !== $("track")) return; const r = $("track").getBoundingClientRect(); addSegment((e.clientX - r.left) / r.width * S.info.duration); });
  $("viewport").addEventListener("click", (e) => {
    const r = $("viewport").getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    const W = S.info.width, H = S.info.height; let sx, sy;
    if (S.mode === "full") { sx = px / S.k; sy = py / S.k; }
    else { const v = viewAt(video.currentTime), s = S.layout.cw * S.k / v.w; sx = v.x + px / s; sy = v.y + py / s; }
    let i = S.sel;
    if (i < 0) { const s = segmentAt(video.currentTime); i = s ? S.plan.segments.indexOf(s) : -1; }
    if (i < 0) { addSegment(video.currentTime); i = S.sel; if (i < 0) return; }
    const seg = S.plan.segments[i]; seg.center_x = clamp(sx / W, 0, 1); seg.center_y = clamp(sy / H, 0, 1); seg.path = null;
    if (seg.kind !== "manual") seg.kind = "manual";
    S.sel = i; changed();
  });
  $("segStart").onchange = () => { const s = S.plan.segments[S.sel]; s.start = clamp(+$("segStart").value, 0, s.end - 0.3); changed(); };
  $("segEnd").onchange = () => { const s = S.plan.segments[S.sel]; s.end = clamp(+$("segEnd").value, s.start + 0.3, S.info.duration); changed(); };
  $("segZoom").oninput = () => { const s = S.plan.segments[S.sel]; s.zoom = +$("segZoom").value; changed(); };
  $("segDelete").onclick = () => { S.plan.segments.splice(S.sel, 1); S.sel = -1; changed(); };
  // style
  const sw = $("swatches");
  for (const [name, p] of Object.entries(S.project.presets)) {
    const b = document.createElement("button"); b.title = name; b.dataset.name = name;
    b.style.background = `linear-gradient(${p.angle}deg, ${p.stops.join(", ")})`;
    b.onclick = () => { S.style.background = name; $("bgHex").value = ""; styleChanged(); };
    sw.appendChild(b);
  }
  $("bgHex").onchange = () => { if ($("bgHex").value.trim()) { S.style.background = $("bgHex").value.trim(); styleChanged(); } };
  for (const id of ["padding", "radius"]) $(id).oninput = () => { S.style[id] = +$(id).value; styleChanged(); };
  $("shadow").onchange = () => { S.style.shadow = $("shadow").checked; styleChanged(); };
  $("motionBlur").onchange = () => { S.style.motion_blur = $("motionBlur").checked; };
  $("frame").onchange = () => { S.style.frame = $("frame").value; styleChanged(); };
  for (const a of S.project.aspects) { const o = document.createElement("option"); o.value = a; o.textContent = a === "source" ? "Same as source" : a; $("aspect").appendChild(o); }
  $("aspect").onchange = () => { S.style.aspect = $("aspect").value; styleChanged(); };
  $("reanalyze").onclick = async () => {
    if (!confirm("Replace all zooms with a fresh automatic analysis?")) return;
    $("reanalyze").disabled = true; $("saveState").textContent = "Analyzing…";
    try { const j = await api("/api/analyze", { max_zoom: +$("maxZoom").value }); S.plan = j.plan; S.sel = -1; computeCamera(); renderSegments(); $("saveState").textContent = "Saved"; }
    catch (e) { $("saveState").textContent = e.message; }
    $("reanalyze").disabled = false;
  };
  $("renderBtn").onclick = startRender;
  window.addEventListener("resize", () => { layoutStage(); renderRuler(); renderSegments(); });
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    else if ((e.key === "Delete" || e.key === "Backspace") && S.sel >= 0) { S.plan.segments.splice(S.sel, 1); S.sel = -1; changed(); }
    else if (e.key === "z" || e.key === "Z") addSegment(video.currentTime);
    else if (e.key === "ArrowLeft") video.currentTime = Math.max(0, video.currentTime - (e.shiftKey ? 1 : 1 / 30));
    else if (e.key === "ArrowRight") video.currentTime = Math.min(S.info.duration, video.currentTime + (e.shiftKey ? 1 : 1 / 30));
    else if (e.key === "Escape") select(-1);
  });
}
function styleChanged() {
  for (const b of $("swatches").children) b.classList.toggle("on", b.dataset.name === S.style.background);
  $("paddingVal").textContent = S.style.padding + "%"; $("radiusVal").textContent = S.style.radius + "px";
  layoutStage();
}

// ---------------------------------------------------------------- render
async function startRender() {
  const style = Object.assign({}, S.style, { padding: S.style.padding + "%" });
  $("renderBtn").disabled = true; $("renderDone").classList.add("hidden");
  try { await api("/api/render", { plan: S.plan, style, preview: $("draft").checked }); }
  catch (e) { $("renderStatus").textContent = e.message; $("renderBtn").disabled = false; return; }
  poll();
}
async function poll() {
  let j; try { j = await api("/api/render"); } catch (e) { setTimeout(poll, 1000); return; }
  const pct = j.total ? (100 * j.done / j.total) : 0;
  $("progressBar").style.width = pct.toFixed(1) + "%";
  if (j.state === "running") {
    $("renderStatus").textContent = `Rendering… ${pct.toFixed(0)}%  ·  ${j.fps} fps  ·  ETA ${j.eta == null ? "…" : fmt(j.eta)}`;
    setTimeout(poll, 400); return;
  }
  $("renderBtn").disabled = false;
  if (j.state === "done") {
    $("renderStatus").textContent = `Done in ${j.seconds}s (${j.fps} fps).`;
    $("renderDone").classList.remove("hidden"); $("outPath").textContent = j.output;
    $("openOutput").href = "/output?t=" + Date.now();
  } else if (j.state === "error" || j.state === "cancelled") $("renderStatus").textContent = `Render ${j.state}: ${j.error || ""}`;
}

// ---------------------------------------------------------------- boot
(async function init() {
  S.project = await api("/api/project");
  S.plan = S.project.plan; S.info = S.project.info;
  $("filename").textContent = S.project.name;
  $("maxZoom").value = S.plan.max_zoom || 2;
  $("film").style.backgroundImage = "url(/filmstrip.jpg)";
  bind(); styleChanged(); renderRuler(); renderSegments();
  video.addEventListener("loadedmetadata", () => layoutStage());
  requestAnimationFrame(loop);
  window.__autozoomReady = true;
})();
