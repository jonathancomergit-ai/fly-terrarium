// app.js - the terrarium: world, fly body, senses in, commands out.
//
// Loop, ~30 times a second:
//   1. work out what the fly senses right now (touching sugar? wind on its antennae? a shadow growing?)
//   2. POST those rates to the Python brain server
//   3. get back which of the 165k neurons fired, and how hard the command neurons are firing
//   4. the body obeys the commands (jump, eat, groom, back up, turn), the brain view glows
//
// The body is a cartoon. Every decision to jump, eat, groom or back up comes from real wiring.

"use strict";

// ------------------------------------------------------------------ labels + colours
const SENSE_INFO = {
  sugar:   { label: "Sugar taste",   cells: "LB3 GRNs",  colour: "#f5b53d" },
  bitter:  { label: "Bitter taste",  cells: "LB1 GRNs",  colour: "#7ee0a1" },
  wind:    { label: "Wind",          cells: "JO-C/E",    colour: "#6cc3ff" },
  sound:   { label: "Courtship song", cells: "JO-A/B",   colour: "#c79bff" },
  looming: { label: "Looming shadow", cells: "LPLC2/LC4", colour: "#ff6b6b" },
};
const ACTION_INFO = {
  escape:   { label: "Escape jump",  cells: "DNp01",   colour: "#ff6b6b", on: 100, verb: "giant fiber fired: JUMP!" },
  feed:     { label: "Extend proboscis", cells: "MN9", colour: "#f5b53d", on: 30, verb: "MN9 fired: proboscis out, eating" },
  groom:    { label: "Groom antennae", cells: "aDN1/2", colour: "#6cc3ff", on: 30, verb: "aDN fired: grooming antennae" },
  backward: { label: "Walk backward", cells: "MDN",    colour: "#c79bff", on: 15, verb: "moonwalker fired: backing up" },
  forward:  { label: "Walk forward",  cells: "P9/BDN2/oDN1", colour: "#9fe870", on: 15, verb: "forward-walking DNs fired" },
  turnL:    { label: "Turn left",     cells: "DNa01/02 L", colour: "#d8dee6", on: 20, verb: "left turning DNs fired" },
  turnR:    { label: "Turn right",    cells: "DNa01/02 R", colour: "#d8dee6", on: 20, verb: "right turning DNs fired" },
};
const MAX_HZ = { sense: 200, action: 300 };

// ------------------------------------------------------------------ DOM
const $ = s => document.querySelector(s);
const arena = $("#arena");
const actx = arena.getContext("2d");
const W = 1000, H = 700; // world units; the canvas is scaled to fit

let tool = "sugar";
let paused = false;
let silence = [];
let silenceDirty = true;
let speed = 1;

// ------------------------------------------------------------------ world state
const world = {
  foods: [],     // {x, y, r, kind: "sugar"|"bitter", amount 0..1}
  fans: [],      // {x, y, angle}
  speakers: [],  // {x, y, t}
  swats: [],     // {x, y, t}  a hand coming down: shadow grows for SWAT_TIME, then lands
};
const SWAT_TIME = 0.9;
const FLY_SCALE = 1.7; // drawing size only. A real fly would be ~3 px in a room this big

const fly = {
  x: W / 2, y: H / 2, heading: -Math.PI / 2, speed: 0,
  legPhase: 0, mode: "walk", modeTime: 0,
  jump: null, squished: 0, cooldown: 0, proboscis: 0, groomPhase: 0,
  wanderTurn: 0,
};

const senses = { sugar: 0, bitter: 0, wind: 0, sound: 0, looming: 0 };
const hz = { forward: 0, turnL: 0, turnR: 0, backward: 0, escape: 0, feed: 0, groom: 0 };
let meta = null, brainView = null, labelCtx = null;

// ------------------------------------------------------------------ helpers
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const dist = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);
const angDiff = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
function headPos() { return [fly.x + Math.cos(fly.heading) * 26, fly.y + Math.sin(fly.heading) * 26]; }

const logEl = $("#log");
function log(html) {
  const li = document.createElement("li");
  li.innerHTML = `<span>${(simMs / 1000).toFixed(1)}s</span>${html}`;
  logEl.prepend(li);
  while (logEl.children.length > 60) logEl.lastChild.remove();
}

// ------------------------------------------------------------------ input
document.querySelectorAll("#tools button").forEach(b => b.addEventListener("click", () => setTool(b.dataset.tool)));
function setTool(t) {
  tool = t;
  document.querySelectorAll("#tools button").forEach(b => b.classList.toggle("on", b.dataset.tool === t));
}
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const t = { 1: "sugar", 2: "bitter", 3: "fan", 4: "speaker", 5: "swat" }[e.key];
  if (t) setTool(t);
  if (e.key === " ") { e.preventDefault(); togglePause(); }
});

function toWorld(e) {
  const r = arena.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width * W, (e.clientY - r.top) / r.height * H];
}
let fanDrag = null;
arena.addEventListener("contextmenu", e => e.preventDefault());
arena.addEventListener("pointerdown", e => {
  const [x, y] = toWorld(e);
  if (e.button === 2) return removeNear(x, y);
  if (tool === "sugar" || tool === "bitter") {
    world.foods.push({ x, y, r: 26, kind: tool, amount: 1 });
    log(`dropped <b>${tool}</b>`);
  } else if (tool === "fan") {
    // fan faces the fly by default; drag to aim it somewhere else
    const f = { x, y, angle: Math.atan2(fly.y - y, fly.x - x) };
    world.fans.push(f);
    fanDrag = f;
    arena.setPointerCapture(e.pointerId);
  } else if (tool === "speaker") {
    world.speakers.push({ x, y, t: 0 });
    log(`a speaker starts playing <b>courtship song</b>`);
  } else if (tool === "swat") {
    world.swats.push({ x, y, t: 0 });
  }
});
arena.addEventListener("pointermove", e => {
  if (!fanDrag) return;
  const [x, y] = toWorld(e);
  if (dist(x, y, fanDrag.x, fanDrag.y) > 12) fanDrag.angle = Math.atan2(y - fanDrag.y, x - fanDrag.x);
});
arena.addEventListener("pointerup", () => (fanDrag = null));

function removeNear(x, y) {
  for (const list of [world.foods, world.fans, world.speakers]) {
    const i = list.findIndex(o => dist(o.x, o.y, x, y) < 40);
    if (i >= 0) { list.splice(i, 1); return; }
  }
}

$("#speed").addEventListener("input", e => { speed = +e.target.value; $("#speed-v").textContent = speed.toFixed(2) + "x"; });
$("#pause").addEventListener("click", togglePause);
function togglePause() { paused = !paused; $("#pause").textContent = paused ? "Resume" : "Pause"; }
$("#reset").addEventListener("click", () => { fetch("/reset", { method: "POST", body: "{}" }); log("brain reset to rest"); });
$("#clear").addEventListener("click", () => { world.foods = []; world.fans = []; world.speakers = []; world.swats = []; });
document.querySelectorAll("[data-silence]").forEach(cb => cb.addEventListener("change", () => {
  silence = [...document.querySelectorAll("[data-silence]:checked")].map(c => c.dataset.silence);
  silenceDirty = true;
  const info = ACTION_INFO[cb.dataset.silence];
  log(cb.checked ? `silenced <b>${info.cells}</b>. The fly can no longer ${info.label.toLowerCase()}`
                 : `<b>${info.cells}</b> restored`);
}));

// ------------------------------------------------------------------ meters
function buildMeters(el, keys, info) {
  const rows = {};
  for (const k of keys) {
    const row = document.createElement("div");
    row.className = "meter";
    row.innerHTML = `<div class="name">${info[k].label} <code>${info[k].cells}</code></div>
      <div class="bar"><div class="fill" style="background:${info[k].colour}"></div></div><div class="val">0</div>`;
    el.appendChild(row);
    rows[k] = { row, fill: row.querySelector(".fill"), val: row.querySelector(".val") };
  }
  return rows;
}
const senseRows = buildMeters($("#sense-meters"), Object.keys(SENSE_INFO), SENSE_INFO);
const actionRows = buildMeters($("#action-meters"), Object.keys(ACTION_INFO), ACTION_INFO);
function updateMeters() {
  for (const [k, r] of Object.entries(senseRows)) {
    r.fill.style.width = clamp(senses[k] / MAX_HZ.sense * 100, 0, 100) + "%";
    r.val.textContent = senses[k] ? Math.round(senses[k]) + " Hz" : "-";
    r.row.classList.toggle("hot", senses[k] > 0);
  }
  for (const [k, r] of Object.entries(actionRows)) {
    r.fill.style.width = clamp(hz[k] / MAX_HZ.action * 100, 0, 100) + "%";
    r.val.textContent = hz[k] >= 0.5 ? hz[k].toFixed(0) + " Hz" : "-";
    r.row.classList.toggle("hot", hz[k] >= ACTION_INFO[k].on);
  }
}

// ------------------------------------------------------------------ senses (world -> brain)
function sense(dt) {
  const [hx, hy] = headPos();
  senses.sugar = senses.bitter = 0;
  // Taste is contact: taste neurons on the proboscis/legs only fire when touching the drop
  for (const f of world.foods) {
    if (dist(hx, hy, f.x, f.y) < f.r * Math.sqrt(f.amount) + 8) senses[f.kind] = Math.max(senses[f.kind], 150 * (0.4 + 0.6 * f.amount));
  }
  // Wind: fans blow in a 70-degree cone, fading with distance
  let wind = 0;
  for (const f of world.fans) {
    const d = dist(fly.x, fly.y, f.x, f.y);
    const off = Math.abs(angDiff(Math.atan2(fly.y - f.y, fly.x - f.x), f.angle));
    if (d < 520 && off < 0.62) wind += (1 - d / 520) * (1 - off / 0.62);
  }
  senses.wind = clamp(wind, 0, 1) * 180;
  // Sound: courtship song carries in all directions a short way
  let snd = 0;
  for (const s of world.speakers) { const d = dist(fly.x, fly.y, s.x, s.y); if (d < 380) snd = Math.max(snd, 1 - d / 380); }
  senses.sound = snd * 160;
  // Looming: a hand's shadow that grows fast and is close = LPLC2/LC4 go wild
  let loom = 0;
  for (const s of world.swats) {
    const p = s.t / SWAT_TIME;
    const radius = 20 + 150 * p * p;
    const d = dist(fly.x, fly.y, s.x, s.y);
    const growth = 2 * p; // expansion speed is what looming detectors care about
    // how much of the eye the shadow covers ~ (size / distance)^2, and it only counts while it's growing
    const cover = clamp(radius / Math.max(d, 30), 0, 1.6);
    loom = Math.max(loom, cover * cover * growth);
  }
  senses.looming = clamp(loom, 0, 1) * 200;
}

// ------------------------------------------------------------------ brain link
let simMs = 0, lastSimMs = 0, inflight = false, online = true, runaway = 0;
const prevHot = {};
async function tickBrain() {
  if (inflight) return;
  inflight = true;
  const body = { senses: paused ? {} : senses, speed, paused };
  if (silenceDirty) { body.silence = silence; silenceDirty = false; }
  try {
    const res = await fetch("/frame", { method: "POST", body: JSON.stringify(body) });
    const buf = await res.arrayBuffer();
    const dv = new DataView(buf);
    const hlen = dv.getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    const off = 4 + hlen;
    const idx = new Int32Array(buf, off, head.active);
    const vals = new Uint8Array(buf, off + head.active * 4, head.active);
    brainView && brainView.addSpikes(idx, vals);

    simMs = head.simMs;
    const dSim = Math.max(1, simMs - lastSimMs);
    lastSimMs = simMs;
    for (const k of Object.keys(hz)) {
      const n = meta.groups[k].length || 1;
      const target = head.acts[k] / n / (dSim / 1000);
      hz[k] += (target - hz[k]) * 0.45; // light smoothing so meters don't strobe
      const hot = hz[k] >= ACTION_INFO[k].on;
      if (hot && !prevHot[k] && k !== "turnL" && k !== "turnR") log(`<b>${ACTION_INFO[k].verb}</b> (${hz[k].toFixed(0)} Hz)`);
      prevHot[k] = hot;
    }
    $("#st-clock").textContent = (simMs / 1000).toFixed(1) + " s";
    $("#st-rate").textContent = head.simRate.toFixed(2) + "x";
    $("#st-active").textContent = head.active.toLocaleString();
    if (!online) { online = true; $("#offline").hidden = true; }
    // runaway check: a calm brain here uses a few thousand neurons. Tens of thousands for
    // seconds with nothing touching the fly means a loop has locked on (model limit, not biology)
    const quiet = Object.values(senses).every((v, i, a) => v <= 30);
    runaway = head.active > 12000 && quiet ? runaway + 1 : 0;
    if (runaway === 90) log("<b>runaway loop:</b> a circuit is stuck firing on its own. Press Reset brain");
  } catch (err) {
    if (online) { online = false; $("#offline").hidden = false; }
    await new Promise(r => setTimeout(r, 1000)); // don't hammer a stopped server
  }
  inflight = false;
}

// ------------------------------------------------------------------ body (brain -> world)
function body(dt) {
  fly.cooldown -= dt;
  fly.modeTime += dt;

  if (fly.squished > 0) {
    fly.squished -= dt;
    if (fly.squished <= 0) { Object.assign(fly, { x: W / 2, y: H / 2, mode: "walk", speed: 0 }); log("a new fly wanders in"); }
    return;
  }
  if (fly.jump) {
    const j = fly.jump;
    j.t += dt;
    const p = Math.min(1, j.t / j.dur);
    fly.x = j.x0 + (j.x1 - j.x0) * p;
    fly.y = j.y0 + (j.y1 - j.y0) * p;
    if (p >= 1) fly.jump = null;
    return;
  }

  const onFood = world.foods.find(f => { const [hx, hy] = headPos(); return dist(hx, hy, f.x, f.y) < f.r * Math.sqrt(f.amount) + 8; });
  let mode = "walk";
  if (hz.escape >= ACTION_INFO.escape.on && fly.cooldown <= 0) mode = "jump";
  else if (hz.feed >= ACTION_INFO.feed.on && onFood) mode = "feed";
  else if (hz.groom >= ACTION_INFO.groom.on) mode = "groom";
  else if (hz.backward >= ACTION_INFO.backward.on) mode = "back";

  if (mode === "jump") {
    // jump away from the nearest looming shadow (or anywhere, if the brain fired on its own)
    let ax = Math.random() - 0.5, ay = Math.random() - 0.5;
    const s = world.swats[world.swats.length - 1];
    if (s) { ax = fly.x - s.x; ay = fly.y - s.y; }
    const a = Math.atan2(ay, ax) + (Math.random() - 0.5) * 0.8, d = 190 + Math.random() * 60;
    fly.jump = { t: 0, dur: 0.22, x0: fly.x, y0: fly.y, x1: clamp(fly.x + Math.cos(a) * d, 30, W - 30), y1: clamp(fly.y + Math.sin(a) * d, 30, H - 30) };
    fly.heading = a;
    fly.cooldown = 0.8;
    return;
  }

  if (mode !== fly.mode) { fly.mode = mode; fly.modeTime = 0; }
  fly.proboscis += ((mode === "feed" ? 1 : 0) - fly.proboscis) * Math.min(1, dt * 10);

  let targetSpeed;
  if (mode === "feed") {
    targetSpeed = 0;
    onFood.amount -= dt * 0.035;
    if (onFood.amount <= 0.05) { world.foods.splice(world.foods.indexOf(onFood), 1); log(`the fly finished the ${onFood.kind} drop`); }
  } else if (mode === "groom") {
    targetSpeed = 0;
    fly.groomPhase += dt * 14;
  } else if (mode === "back") {
    targetSpeed = -55;
  } else {
    // autopilot wander (the nerve cord's own walking rhythm), nudged by brain commands
    fly.wanderTurn += (Math.random() - 0.5) * dt * 6;
    fly.wanderTurn *= Math.pow(0.3, dt);
    targetSpeed = 48 + hz.forward * 0.6;
  }
  // turning DNs steer in every mode where the fly is moving
  const steer = (hz.turnR - hz.turnL) * 0.03;
  if (mode === "walk" || mode === "back") fly.heading += (fly.wanderTurn + steer) * dt;

  // stay inside the terrarium: steer toward the middle near the walls
  const margin = 70;
  if (fly.x < margin || fly.x > W - margin || fly.y < margin || fly.y > H - margin) {
    const toC = Math.atan2(H / 2 - fly.y, W / 2 - fly.x);
    fly.heading += clamp(angDiff(toC, fly.heading), -1, 1) * dt * 2.5;
  }
  fly.speed += (targetSpeed - fly.speed) * Math.min(1, dt * 6);
  fly.x = clamp(fly.x + Math.cos(fly.heading) * fly.speed * dt, 20, W - 20);
  fly.y = clamp(fly.y + Math.sin(fly.heading) * fly.speed * dt, 20, H - 20);
  fly.legPhase += fly.speed * dt * 0.28;
}

function updateSwats(dt) {
  for (let i = world.swats.length - 1; i >= 0; i--) {
    const s = world.swats[i];
    s.t += dt;
    if (s.t >= SWAT_TIME && !s.landed) {
      s.landed = true;
      if (!fly.jump && fly.squished <= 0 && dist(fly.x, fly.y, s.x, s.y) < 75) {
        fly.squished = 2.5;
        log(silence.includes("escape") ? "<b>SPLAT.</b> With the giant fiber silenced, it never jumped" : "<b>SPLAT.</b> Too slow this time");
      } else if (dist(fly.x, fly.y, s.x, s.y) < 260) {
        log("swat missed");
      }
    }
    if (s.t > SWAT_TIME + 0.6) world.swats.splice(i, 1);
  }
  for (const sp of world.speakers) sp.t += dt;
}

// ------------------------------------------------------------------ drawing the arena
function fitCanvas(c) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
}

function drawArena(time) {
  fitCanvas(arena);
  const g = actx;
  g.setTransform(arena.width / W, 0, 0, arena.height / H, 0, 0);
  // floor: warm dark with a faint hex pattern, like looking down into a glass tank
  const bg = g.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, 650);
  bg.addColorStop(0, "#1d1a16"); bg.addColorStop(1, "#0d0c0b");
  g.fillStyle = bg; g.fillRect(0, 0, W, H);
  g.strokeStyle = "rgba(255,240,210,0.035)"; g.lineWidth = 1;
  for (let y = 0; y < H + 40; y += 34) for (let x = (y / 34) % 2 ? 20 : 0; x < W + 40; x += 40) {
    g.beginPath();
    for (let k = 0; k < 6; k++) { const a = k * Math.PI / 3 + Math.PI / 6; g.lineTo(x + Math.cos(a) * 20, y + Math.sin(a) * 20); }
    g.closePath(); g.stroke();
  }

  // wind streaks
  for (const f of world.fans) {
    for (let k = 0; k < 14; k++) {
      const p = ((time * 0.6 + k / 14) % 1);
      const spread = ((k * 7919) % 100) / 100 - 0.5;
      const a = f.angle + spread * 1.1;
      const d = 40 + p * 480;
      const x = f.x + Math.cos(a) * d, y = f.y + Math.sin(a) * d;
      g.strokeStyle = `rgba(108,195,255,${0.28 * (1 - p)})`; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x - Math.cos(a) * 26, y - Math.sin(a) * 26); g.stroke();
    }
    g.save(); g.translate(f.x, f.y); g.rotate(f.angle);
    g.fillStyle = "#26303a"; g.strokeStyle = "#6cc3ff"; g.lineWidth = 2;
    g.beginPath(); g.roundRect(-22, -20, 20, 40, 5); g.fill(); g.stroke();
    g.save(); g.translate(2, 0); g.scale(0.35, 1); g.rotate(time * 20);
    g.fillStyle = "#6cc3ff";
    for (let b = 0; b < 3; b++) { g.rotate(Math.PI * 2 / 3); g.beginPath(); g.ellipse(0, -9, 5, 9, 0, 0, Math.PI * 2); g.fill(); }
    g.restore(); g.restore();
  }

  // song rings
  for (const s of world.speakers) {
    for (let k = 0; k < 3; k++) {
      const p = (s.t * 0.7 + k / 3) % 1;
      g.strokeStyle = `rgba(199,155,255,${0.35 * (1 - p)})`; g.lineWidth = 2;
      g.beginPath(); g.arc(s.x, s.y, 16 + p * 360, 0, Math.PI * 2); g.stroke();
    }
    g.fillStyle = "#c79bff"; g.font = "26px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText("🔊", s.x, s.y);
  }

  // food drops
  for (const f of world.foods) {
    const r = f.r * Math.sqrt(f.amount);
    const col = f.kind === "sugar" ? [245, 181, 61] : [126, 224, 161];
    const grad = g.createRadialGradient(f.x - r * 0.3, f.y - r * 0.3, 2, f.x, f.y, r);
    grad.addColorStop(0, `rgba(255,255,255,0.8)`);
    grad.addColorStop(0.35, `rgba(${col},0.75)`);
    grad.addColorStop(1, `rgba(${col},0.25)`);
    g.fillStyle = grad;
    g.beginPath(); g.arc(f.x, f.y, r, 0, Math.PI * 2); g.fill();
  }

  // swat shadows (under the fly) and the hand landing
  for (const s of world.swats) {
    const p = Math.min(1, s.t / SWAT_TIME);
    const radius = 20 + 150 * p * p;
    g.fillStyle = `rgba(0,0,0,${0.15 + 0.5 * p})`;
    g.beginPath(); g.ellipse(s.x, s.y, radius, radius * 0.85, 0, 0, Math.PI * 2); g.fill();
    if (s.t >= SWAT_TIME) {
      const q = (s.t - SWAT_TIME) / 0.6;
      g.globalAlpha = 1 - q;
      g.font = "150px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText("🖐", s.x, s.y);
      g.globalAlpha = 1;
    }
  }

  drawFly(g, time);

  // sense halo: little coloured ring around the fly for whatever it's sensing
  let ring = 0;
  for (const k of ["sugar", "bitter", "wind", "sound", "looming"]) {
    if (senses[k] > 5 && fly.squished <= 0) {
      g.strokeStyle = SENSE_INFO[k].colour; g.globalAlpha = clamp(senses[k] / 150, 0.2, 0.8); g.lineWidth = 2;
      g.beginPath(); g.arc(fly.x, fly.y, 52 + ring * 7, 0, Math.PI * 2); g.stroke();
      ring++;
    }
  }
  g.globalAlpha = 1;
}

function drawFly(g, time) {
  g.save();
  g.translate(fly.x, fly.y);
  if (fly.squished > 0) {
    g.globalAlpha = clamp(fly.squished, 0, 1);
    g.fillStyle = "#3a2a1c";
    for (let k = 0; k < 9; k++) { const a = k * 2.4; g.beginPath(); g.arc(Math.cos(a) * 12, Math.sin(a) * 9, 7 - k * 0.5, 0, Math.PI * 2); g.fill(); }
    g.restore();
    return;
  }
  const lift = fly.jump ? Math.sin(Math.PI * fly.jump.t / fly.jump.dur) : 0;
  g.scale(FLY_SCALE, FLY_SCALE);
  if (lift) { // shadow on the floor while airborne
    g.fillStyle = "rgba(0,0,0,0.35)";
    g.beginPath(); g.ellipse(10, 14, 16, 9, fly.heading, 0, Math.PI * 2); g.fill();
  }
  g.scale(1 + lift * 0.5, 1 + lift * 0.5);
  g.rotate(fly.heading + Math.PI / 2); // draw pointing "up" (-y)

  // legs: three pairs, tripod gait (alternate sets swing together)
  g.strokeStyle = "#2b2016"; g.lineWidth = 1.6; g.lineCap = "round";
  const grooming = fly.mode === "groom";
  for (const side of [-1, 1]) {
    for (let k = 0; k < 3; k++) {
      const phase = fly.legPhase + (k % 2 === (side > 0 ? 0 : 1) ? 0 : Math.PI);
      let swing = Math.sin(phase) * 3.5;
      const ay = -4 + k * 4.5;
      let fx = side * 13, fy = ay + (k - 1) * 9 + swing;
      if (grooming && k === 0) { fx = side * (3 + Math.sin(fly.groomPhase) * 2); fy = -17 + Math.cos(fly.groomPhase) * 2; }
      g.beginPath(); g.moveTo(side * 3, ay); g.lineTo(side * 9, ay + (k - 1) * 3 - 2); g.lineTo(fx, fy); g.stroke();
    }
  }
  // wings (folded back over the abdomen, buzz when jumping)
  const buzz = fly.jump ? Math.sin(time * 90) * 0.5 : 0;
  g.fillStyle = "rgba(210,225,235,0.28)"; g.strokeStyle = "rgba(210,225,235,0.45)"; g.lineWidth = 0.8;
  for (const side of [-1, 1]) {
    g.save(); g.translate(side * 2, 0); g.rotate(side * (0.28 + buzz));
    g.beginPath(); g.ellipse(side * 4, 11, 5.5, 13, 0, 0, Math.PI * 2); g.fill(); g.stroke();
    g.restore();
  }
  // abdomen with stripes
  g.fillStyle = "#b8894e";
  g.beginPath(); g.ellipse(0, 9, 6, 9.5, 0, 0, Math.PI * 2); g.fill();
  g.strokeStyle = "rgba(60,35,15,0.8)"; g.lineWidth = 1.6;
  for (let k = 0; k < 3; k++) { g.beginPath(); g.ellipse(0, 9 + k * 3, 5.6 - k * 0.6, 1.2, 0, 0, Math.PI); g.stroke(); }
  // thorax
  g.fillStyle = "#8f6a3c"; g.beginPath(); g.ellipse(0, -3, 5.5, 6, 0, 0, Math.PI * 2); g.fill();
  // proboscis
  if (fly.proboscis > 0.05) {
    g.strokeStyle = "#5a3d22"; g.lineWidth = 2;
    g.beginPath(); g.moveTo(0, -13); g.lineTo(0, -13 - fly.proboscis * 8); g.stroke();
  }
  // head + big red eyes
  g.fillStyle = "#9c7646"; g.beginPath(); g.ellipse(0, -11, 5, 4, 0, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#c2261d";
  for (const side of [-1, 1]) { g.beginPath(); g.ellipse(side * 4, -11, 2.8, 3.6, side * 0.3, 0, Math.PI * 2); g.fill(); }
  // antennae (flutter in wind)
  const flutter = senses.wind > 5 ? Math.sin(time * 40) * 0.25 : 0;
  g.strokeStyle = "#3b2a18"; g.lineWidth = 1.2;
  for (const side of [-1, 1]) {
    g.beginPath(); g.moveTo(side * 1.3, -14); g.lineTo(side * (2.4 + flutter), -17.5); g.stroke();
  }
  g.restore();
}

// ------------------------------------------------------------------ brain panel + legend
function buildLegend() {
  $("#legend").innerHTML = BRAIN_PALETTE.map(p =>
    `<div><b style="background:rgb(${p.rgb.map(v => Math.round(v * 255))})"></b>${p.key}</div>`).join("");
}

// ------------------------------------------------------------------ main loop
let lastT = performance.now(), brainClock = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  const time = now / 1000;
  if (!paused) {
    sense(dt);
    body(dt);
    updateSwats(dt);
  }
  brainClock += dt;
  if (brainClock > 1 / 30) { brainClock = 0; tickBrain(); }
  drawArena(time);
  if (brainView) {
    const labels = [];
    for (const [k, info] of Object.entries(ACTION_INFO)) {
      if (hz[k] > 2) labels.push({ group: k, text: `${info.cells} ${hz[k].toFixed(0)}Hz`, colour: info.colour, alpha: clamp(hz[k] / info.on, 0.25, 1) });
    }
    for (const [k, info] of Object.entries(SENSE_INFO)) {
      if (senses[k] > 5) labels.push({ group: k, text: info.cells, colour: info.colour, alpha: 0.7 });
    }
    fitCanvas(labelCtx.canvas);
    brainView.frame(dt, labelCtx, labels);
  }
  updateMeters();
  requestAnimationFrame(frame);
}

async function boot() {
  try {
    meta = await (await fetch("/meta.json")).json();
    await fetch("/reset", { method: "POST", body: "{}" }); // fresh page = fresh, calm brain
    const geo = await (await fetch("/geometry.bin")).arrayBuffer();
    const canvas = $("#brain");
    brainView = new BrainView(canvas, geo, meta);
    const lc = document.createElement("canvas");
    lc.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none";
    canvas.parentElement.appendChild(lc);
    labelCtx = lc.getContext("2d");
    buildLegend();
    // a starter drop of sugar just ahead of the fly, so something happens right away
    world.foods.push({ x: W / 2, y: H / 2 - 120, r: 26, kind: "sugar", amount: 1 });
    log("the fly wakes up. Try: swat near it, then silence the giant fiber and swat again");
    // ?demo=feed|bitter|wind|swat sets up a scene (handy for screenshots and showing people)
    const demo = new URLSearchParams(location.search).get("demo");
    if (demo) {
      fly.heading = -Math.PI / 2;
      const [hx, hy] = headPos();
      if (demo === "feed" || demo === "bitter") { world.foods = [{ x: hx, y: hy - 6, r: 30, kind: demo === "feed" ? "sugar" : "bitter", amount: 1 }]; }
      if (demo === "wind") world.fans.push({ x: fly.x - 220, y: fly.y, angle: 0 });
      if (demo === "swat") setInterval(() => world.swats.push({ x: fly.x + 10, y: fly.y + 10, t: 0 }), 2500);
    }
  } catch (err) {
    console.error(err);
    const box = $("#offline");
    if (meta) box.querySelector("div").innerHTML = `<h2>The 3D brain couldn't start</h2><p>${String(err.message || err)}</p><p>WebGL may be off in this browser. The fly still works.</p>`;
    box.hidden = false;
    box.addEventListener("click", () => (box.hidden = true));
  }
  window.__flyBoot = true;
  requestAnimationFrame(frame);
}
boot();
