// eyes.js - camera -> server (flyvis eyes + MaleCNS brain) -> pictures of what each layer "sees".
//
// Two loops run side by side:
//   eye loop   (~30/s): grab a grey 320x180 frame, POST /eye, get back 7 hex maps x 2 eyes, draw them
//   brain loop (~15/s): POST /frame like the Terrarium does, get spikes + looming/escape counts
//
// Hex maps: each eye is 721 "hexals" (ommatidia). flyvis gives each one a (u, v) hex coordinate;
// on screen x = v, y = u + v/2. flyvis images have x = FORWARD, so the right eye is flipped back
// to match the camera picture (fly's right eye = right half of the frame, it's facing you).

"use strict";

const $ = s => document.querySelector(s);
const CAP_W = 320, CAP_H = 180;               // what we send: small and grey, the eyes are only 31 hexals wide
const DIR_RGB = [[255, 107, 107], [108, 195, 255], [159, 232, 112], [245, 181, 61]]; // a, b, c, d
const VISION_INFO = {
  LPLC2: { label: "Looming detectors", cells: "LPLC2", colour: "#ff6b6b" },
  LC4:   { label: "Looming detectors", cells: "LC4",   colour: "#ff8f6b" },
  LPLC1: { label: "Collision / objects", cells: "LPLC1", colour: "#c79bff" },
  LC11:  { label: "Small moving objects", cells: "LC11", colour: "#6cc3ff" },
};
const ACTION_INFO = {
  escape:   { label: "Escape jump",   cells: "DNp01 giant fiber", colour: "#ff6b6b", max: 300 },
  backward: { label: "Walk backward", cells: "MDN",               colour: "#c79bff", max: 150 },
  turnL:    { label: "Turn left",     cells: "DNa01/02 L",        colour: "#d8dee6", max: 150 },
  turnR:    { label: "Turn right",    cells: "DNa01/02 R",        colour: "#d8dee6", max: 150 },
};

let meta = null, eyeMeta = null, brainView = null, labelCtx = null;
let source = "camera", video = null, stream = null;
const capture = document.createElement("canvas");
capture.width = CAP_W; capture.height = CAP_H;
const capCtx = capture.getContext("2d", { willReadFrequently: true });

// ------------------------------------------------------------------ helpers
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sleep = ms => new Promise(r => setTimeout(r, ms));
function fit(c) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
}
let simMs = 0;
function log(html) {
  const li = document.createElement("li");
  li.innerHTML = `<span>${(simMs / 1000).toFixed(1)}s</span>${html}`;
  $("#log").prepend(li);
  while ($("#log").children.length > 40) $("#log").lastChild.remove();
}

// ------------------------------------------------------------------ sources
async function startCamera() {
  const msg = $("#cam-msg");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 }, audio: false });
    video = document.createElement("video");
    video.muted = true; video.playsInline = true; video.srcObject = stream;
    await video.play();
    msg.hidden = true;
  } catch (err) {
    video = null;
    msg.innerHTML = `<div>No camera: ${String(err.message || err)}<br>Allow camera access, or try a demo source.<br><button id="retry-cam">Try camera again</button></div>`;
    msg.hidden = false;
    $("#retry-cam").addEventListener("click", startCamera);
  }
}
function setSource(s) {
  source = s;
  for (const [id, key] of [["#src-camera", "camera"], ["#src-loom", "loom"], ["#src-bars", "bars"]]) $(id).classList.toggle("on", key === s);
  if (s === "camera" && !video) startCamera();
  if (s !== "camera") $("#cam-msg").hidden = true;
}
$("#src-camera").addEventListener("click", () => setSource("camera"));
$("#src-loom").addEventListener("click", () => setSource("loom"));
$("#src-bars").addEventListener("click", () => setSource("bars"));

// draw the current source into the 320x180 capture canvas (and return false if there's nothing yet)
function drawSource(t) {
  const g = capCtx;
  if (source === "camera") {
    if (!video || video.readyState < 2) return false;
    // cover-crop the camera into 16:9
    const vw = video.videoWidth, vh = video.videoHeight, s = Math.max(CAP_W / vw, CAP_H / vh);
    g.drawImage(video, (CAP_W - vw * s) / 2, (CAP_H - vh * s) / 2, vw * s, vh * s);
    return true;
  }
  g.fillStyle = "#c8c8c8"; g.fillRect(0, 0, CAP_W, CAP_H);
  g.fillStyle = "#161616";
  if (source === "loom") {
    // a ball rushing at the fly every 2.5 s, alternating eyes: size grows like 1/distance
    const cycle = 2.5, k = Math.floor(t / cycle), p = (t % cycle) / cycle;
    const cx = k % 2 ? CAP_W * 0.72 : CAP_W * 0.28, cy = CAP_H * 0.5;
    if (p < 0.6) {
      const dist = 1 - p / 0.6;                         // 1 = far, 0 = hits
      const r = clamp(6 / (dist + 0.04), 4, 220);
      g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fill();
    }
  } else if (source === "bars") {
    const cycle = 2, k = Math.floor(t / cycle), p = (t % cycle) / cycle;
    if (k % 4 < 2) { const x = k % 4 === 0 ? p * CAP_W : (1 - p) * CAP_W; g.fillRect(x - 12, 0, 24, CAP_H); }
    else { const y = k % 4 === 2 ? p * CAP_H : (1 - p) * CAP_H; g.fillRect(0, y - 10, CAP_W, 20); }
  }
  return true;
}

// ------------------------------------------------------------------ hex drawing
let hexXY = null, hexBox = null;
function prepHex() {
  hexXY = eyeMeta.hex.map(([u, v]) => [v, u + v / 2]);
  const xs = hexXY.map(p => p[0]), ys = hexXY.map(p => p[1]);
  hexBox = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}
// colour for one hexal of one layer
function colourFor(layer, val) {
  if (layer === "R1") return `rgb(${val},${val},${val})`;
  if (layer === "T4" || layer === "T5") {
    const s = (val & 63) / 63;
    if (s < 0.04) return "#07090b";
    const c = DIR_RGB[val >> 6];
    return `rgb(${Math.round(c[0] * s)},${Math.round(c[1] * s)},${Math.round(c[2] * s)})`;
  }
  const d = (val - 128) / 127;                       // -1 .. 1: less / more active than on a grey screen
  if (d >= 0) return `rgb(${Math.round(20 + 235 * d)},${Math.round(20 + 161 * d)},${Math.round(24 + 37 * d)})`;
  return `rgb(${Math.round(20 - 0 * d)},${Math.round(20 - 175 * d)},${Math.round(24 - 231 * d)})`;
}
function drawEyes(canvas, layer, values) {
  fit(canvas);
  const g = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, n = hexXY.length;
  g.fillStyle = "#07090b"; g.fillRect(0, 0, W, H);
  const bw = hexBox.x1 - hexBox.x0 + 1, bh = hexBox.y1 - hexBox.y0 + 1;
  const scale = Math.min((W / 2 - 8) / bw, (H - 8) / bh);
  const r = scale * 0.52;
  for (let e = 0; e < 2; e++) {
    const cx = W / 4 + (e * W) / 2, cy = H / 2;
    for (let i = 0; i < n; i++) {
      const [x, y] = hexXY[i];
      const hx = (e === 0 ? x : -x) - (hexBox.x0 + hexBox.x1) / 2 * (e === 0 ? 1 : -1);
      g.fillStyle = colourFor(layer, values[e * n + i]);
      g.beginPath();
      g.arc(cx + hx * scale, cy + (y - (hexBox.y0 + hexBox.y1) / 2) * scale, r, 0, Math.PI * 2);
      g.fill();
    }
  }
}

// ------------------------------------------------------------------ eye loop
let fpsCount = 0, fpsT = performance.now();
async function eyeLoop() {
  const n = eyeMeta.hex.length;
  const layerCanvas = {};
  for (const L of eyeMeta.layers.slice(1)) {
    const card = document.createElement("div");
    card.className = "layer";
    card.innerHTML = `<canvas></canvas><b>${L.name}</b><small>${L.desc}</small>`;
    $("#layers").appendChild(card);
    layerCanvas[L.id] = card.querySelector("canvas");
  }
  const cam = $("#cam"), camCtx = cam.getContext("2d");
  const t0 = performance.now();
  for (;;) {
    const started = performance.now();
    const t = (started - t0) / 1000;
    if (!drawSource(t)) { await sleep(100); continue; }
    fit(cam);
    camCtx.drawImage(capture, 0, 0, cam.width, cam.height);
    // grey bytes: the eyes only use luminance
    const px = capCtx.getImageData(0, 0, CAP_W, CAP_H).data;
    const body = new Uint8Array(4 + CAP_W * CAP_H);
    new DataView(body.buffer).setUint16(0, CAP_W, true);
    new DataView(body.buffer).setUint16(2, CAP_H, true);
    for (let i = 0, j = 4; i < px.length; i += 4, j++) body[j] = (px[i] * 77 + px[i + 1] * 150 + px[i + 2] * 29) >> 8;
    try {
      const res = await fetch("/eye", { method: "POST", body });
      const buf = await res.arrayBuffer();
      const hlen = new DataView(buf).getUint32(0, true);
      if (hlen === 0) throw new Error("server has no eyes");
      let off = 4 + hlen;
      for (const L of eyeMeta.layers) {
        const vals = new Uint8Array(buf, off, 2 * n);
        off += 2 * n;
        drawEyes(L.id === "R1" ? $("#fly-view") : layerCanvas[L.id], L.id, vals);
      }
      fpsCount++;
    } catch (err) {
      $("#offline").hidden = false;
      await sleep(1500);
      continue;
    }
    const now = performance.now();
    if (now - fpsT > 1000) { $("#st-fps").textContent = (fpsCount * 1000 / (now - fpsT)).toFixed(0); fpsCount = 0; fpsT = now; }
    await sleep(Math.max(0, 33 - (now - started)));
  }
}

// ------------------------------------------------------------------ brain loop
const hz = {};
const rows = {};
function buildMeters() {
  const el = $("#vision-meters");
  const add = (key, info) => {
    const row = document.createElement("div");
    row.className = "meter";
    row.innerHTML = `<div class="name">${info.label} <code>${info.cells}</code></div>
      <div class="bar"><div class="fill" style="background:${info.colour}"></div></div><div class="val">-</div>`;
    el.appendChild(row);
    rows[key] = { row, fill: row.querySelector(".fill"), val: row.querySelector(".val"), max: info.max || 100 };
    hz[key] = 0;
  };
  for (const [k, info] of Object.entries(VISION_INFO)) if (eyeMeta.vision[k]) add(k, info);
  for (const [k, info] of Object.entries(ACTION_INFO)) add(k, info);
}
let lastSim = 0, lastJump = 0;
async function brainLoop() {
  for (;;) {
    try {
      const res = await fetch("/frame", { method: "POST", body: JSON.stringify({ senses: {} }) });
      const buf = await res.arrayBuffer();
      const hlen = new DataView(buf).getUint32(0, true);
      const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
      const off = 4 + hlen;
      brainView && brainView.addSpikes(new Int32Array(buf, off, head.active), new Uint8Array(buf, off + head.active * 4, head.active));
      simMs = head.simMs;
      const dSec = Math.max(0.001, (simMs - lastSim) / 1000);
      lastSim = simMs;
      const counts = { ...(head.vision || {}), ...head.acts };
      const sizes = { ...eyeMeta.vision };
      for (const k of Object.keys(ACTION_INFO)) sizes[k] = meta.groups[k].length || 1;
      for (const k of Object.keys(rows)) {
        const target = (counts[k] || 0) / sizes[k] / dSec;
        hz[k] += (target - hz[k]) * 0.5;
        rows[k].fill.style.width = clamp(hz[k] / rows[k].max * 100, 0, 100) + "%";
        rows[k].val.textContent = hz[k] >= 0.5 ? hz[k].toFixed(0) + " Hz" : "-";
        rows[k].row.classList.toggle("hot", hz[k] >= rows[k].max * 0.2);
      }
      if (hz.escape > 100 && simMs - lastJump > 800) {
        lastJump = simMs;
        const j = $("#jump");
        j.classList.add("fire");
        setTimeout(() => j.classList.remove("fire"), 60);
        log(`<b>giant fiber fired: JUMP!</b> (${hz.escape.toFixed(0)} Hz)`);
      }
      $("#st-rate").textContent = head.simRate.toFixed(2) + "x";
      $("#st-active").textContent = head.active.toLocaleString();
    } catch (err) { await sleep(1000); }
    await sleep(66);
  }
}

// ------------------------------------------------------------------ 3D brain
let lastT = performance.now();
function drawBrain(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  if (brainView) { fit(labelCtx.canvas); brainView.frame(dt, labelCtx, []); }
  requestAnimationFrame(drawBrain);
}

async function boot() {
  meta = await (await fetch("/meta.json")).json();
  eyeMeta = await (await fetch("/eye_meta.json")).json();
  if (!meta.eyes || eyeMeta.error) {
    $("#offline-msg").innerHTML = `${eyeMeta.error || "The server started without eyes."}<br><br>The eyes need the Python 3.12 environment: run <code>play.bat</code>, or <code>.venv-eye\\Scripts\\python server.py</code>.`;
    $("#offline").hidden = false;
    return;
  }
  prepHex();
  buildMeters();
  try {
    brainView = new BrainView($("#brain"), await (await fetch("/geometry.bin")).arrayBuffer(), meta);
    const lc = document.createElement("canvas");
    lc.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none";
    $("#brain").parentElement.appendChild(lc);
    labelCtx = lc.getContext("2d");
    $("#legend").innerHTML = BRAIN_PALETTE.map(p => `<div><b style="background:rgb(${p.rgb.map(v => Math.round(v * 255))})"></b>${p.key}</div>`).join("");
    requestAnimationFrame(drawBrain);
  } catch (err) { console.error(err); }
  await fetch("/reset", { method: "POST", body: "{}" });
  const demo = new URLSearchParams(location.search).get("source");
  setSource(demo === "loom" || demo === "bars" ? demo : "camera");
  window.__eyesBoot = true;
  brainLoop();
  eyeLoop();
}
boot();
