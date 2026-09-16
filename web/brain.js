// brain.js - draws all 165k neurons as glowing points with raw WebGL (no libraries).
//
// Every neuron is one point at its cell body. Each point has a "heat" number:
// a spike bumps heat up, and heat fades every frame, so firing neurons flash and glow out.
// Python analogy: heat is a numpy array we decay with `heat *= 0.86` each frame, then
// upload to the GPU in one go (bufferSubData) instead of drawing points one by one.

"use strict";

// superclass name -> colour group (the annotations' coarse "what part of the nervous system")
const BRAIN_PALETTE = [
  { key: "optic lobe", test: s => s.startsWith("ol_") || s.startsWith("visual"), rgb: [0.25, 0.45, 1.0] },
  { key: "sensory", test: s => s.includes("sensory"), rgb: [1.0, 0.72, 0.2] },
  { key: "central brain", test: s => s.startsWith("cb_"), rgb: [0.62, 0.38, 1.0] },
  { key: "nerve cord (VNC)", test: s => s.startsWith("vnc_"), rgb: [0.15, 0.85, 0.75] },
  { key: "descending / ascending", test: s => s.includes("descending") || s.includes("ascending"), rgb: [1.0, 0.35, 0.6] },
  { key: "other", test: () => true, rgb: [0.5, 0.52, 0.56] },
];

class BrainView {
  constructor(canvas, geometry, meta) {
    this.canvas = canvas;
    this.n = meta.n;
    const gl = (this.gl = canvas.getContext("webgl", { antialias: false, alpha: false, premultipliedAlpha: false }));
    if (!gl) throw new Error("WebGL not available");

    this.pos = new Float32Array(geometry, 0, this.n * 3);
    const cls = new Uint8Array(geometry, this.n * 3 * 4, this.n); // after n*3 float32s = n*12 bytes
    // map each superclass id to a palette colour, once
    const clsColour = meta.superclasses.map(s => BRAIN_PALETTE.findIndex(p => p.test(s)));
    const colour = new Float32Array(this.n * 3);
    for (let i = 0; i < this.n; i++) colour.set(BRAIN_PALETTE[clsColour[cls[i]]].rgb, i * 3);
    this.heat = new Float32Array(this.n);

    const vs = `
      attribute vec3 aPos; attribute vec3 aCol; attribute float aHeat;
      uniform mat4 uMVP; uniform float uSize;
      varying vec3 vCol; varying float vHeat;
      void main() {
        gl_Position = uMVP * vec4(aPos, 1.0);
        vCol = aCol; vHeat = aHeat;
        gl_PointSize = uSize * (1.0 + aHeat * 2.2) / gl_Position.w;
      }`;
    const fs = `
      precision mediump float;
      varying vec3 vCol; varying float vHeat;
      void main() {
        vec2 d = gl_PointCoord - 0.5;
        float r = dot(d, d) * 4.0;
        if (r > 1.0) discard;
        float soft = 1.0 - r;
        // resting neurons: faint tinted dust. Firing: tint -> white-hot.
        vec3 c = mix(vCol, vec3(1.0, 0.95, 0.85), clamp(vHeat * 0.8, 0.0, 1.0));
        float a = (0.085 + vHeat * 0.9) * soft;
        gl_FragColor = vec4(c * a, 1.0);
      }`;
    const prog = (this.prog = gl.createProgram());
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
      gl.attachShader(prog, sh);
    }
    gl.linkProgram(prog);
    gl.useProgram(prog);

    const attr = (nameIn, data, size, usage) => {
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, usage);
      const loc = gl.getAttribLocation(prog, nameIn);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      return buf;
    };
    attr("aPos", this.pos, 3, gl.STATIC_DRAW);
    attr("aCol", colour, 3, gl.STATIC_DRAW);
    this.heatBuf = attr("aHeat", this.heat, 1, gl.DYNAMIC_DRAW);
    this.uMVP = gl.getUniformLocation(prog, "uMVP");
    this.uSize = gl.getUniformLocation(prog, "uSize");

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE); // additive: overlapping glow adds up like light

    // camera
    this.yaw = 0.6; this.pitch = 0.12; this.dist = 3.1; this.auto = true;
    this.labels = [];
    this._bindMouse();

    // centroid of each named group, for the floating labels
    this.groupCentre = {};
    for (const [g, ids] of Object.entries(meta.groups)) {
      if (!ids.length) continue;
      const c = [0, 0, 0];
      for (const i of ids) for (let k = 0; k < 3; k++) c[k] += this.pos[i * 3 + k] / ids.length;
      this.groupCentre[g] = c;
    }
  }

  _bindMouse() {
    let drag = null;
    this.canvas.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; this.auto = false; this.canvas.setPointerCapture(e.pointerId); });
    this.canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      this.yaw += (e.clientX - drag[0]) * 0.008;
      this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + (e.clientY - drag[1]) * 0.008));
      drag = [e.clientX, e.clientY];
    });
    this.canvas.addEventListener("pointerup", () => { drag = null; clearTimeout(this._resume); this._resume = setTimeout(() => (this.auto = true), 4000); });
    this.canvas.addEventListener("wheel", e => { e.preventDefault(); this.dist = Math.max(1.2, Math.min(7, this.dist * (1 + e.deltaY * 0.001))); }, { passive: false });
  }

  addSpikes(idx, vals) {
    const h = this.heat;
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j];
      h[i] = Math.min(1.5, h[i] + 0.35 + vals[j] * 0.15);
    }
  }

  _matrix() {
    const cw = this.canvas.width, ch = this.canvas.height;
    const f = 1 / Math.tan(0.35), aspect = cw / ch;
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    // column-major 4x4: rotate yaw (around y), then pitch (around x), push back by dist, perspective
    const r = [cy, sp * sy, -cp * sy, 0, 0, cp, sp, 0, sy, -sp * cy, cp * cy, 0, 0, 0, -this.dist, 1];
    const p = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, -1.02, -1, 0, 0, -0.2, 0];
    const m = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let rr = 0; rr < 4; rr++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += p[k * 4 + rr] * r[c * 4 + k];
      m[c * 4 + rr] = s;
    }
    return m;
  }

  project(m, v) {
    const x = m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12];
    const y = m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13];
    const w = m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15];
    return [(x / w * 0.5 + 0.5) * this.canvas.width, (1 - (y / w * 0.5 + 0.5)) * this.canvas.height];
  }

  frame(dt, overlayCtx, activeLabels) {
    const gl = this.gl, c = this.canvas, dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(c.clientWidth * dpr), hgt = Math.round(c.clientHeight * dpr);
    if (c.width !== w || c.height !== hgt) { c.width = w; c.height = hgt; }
    if (this.auto) this.yaw += dt * 0.18;

    const decay = Math.pow(0.08, dt); // heat falls to 8% per second
    const h = this.heat;
    for (let i = 0; i < h.length; i++) if (h[i] > 0.002) h[i] *= decay; else h[i] = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.heatBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, h);

    gl.viewport(0, 0, w, hgt);
    gl.clearColor(0.043, 0.051, 0.063, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const m = this._matrix();
    gl.uniformMatrix4fv(this.uMVP, false, m);
    gl.uniform1f(this.uSize, 4.2 * dpr * (hgt / 700));
    gl.drawArrays(gl.POINTS, 0, this.n);

    // floating labels for the command neurons that are firing right now
    if (overlayCtx) {
      const o = overlayCtx;
      o.clearRect(0, 0, o.canvas.width, o.canvas.height);
      o.font = `${11 * dpr}px ui-monospace, Consolas, monospace`;
      for (const lab of activeLabels) {
        const ctr = this.groupCentre[lab.group];
        if (!ctr) continue;
        const [x, y] = this.project(m, ctr);
        o.globalAlpha = lab.alpha;
        o.strokeStyle = o.fillStyle = lab.colour;
        o.beginPath(); o.arc(x, y, 5 * dpr, 0, Math.PI * 2); o.stroke();
        o.beginPath(); o.moveTo(x + 5 * dpr, y); o.lineTo(x + 22 * dpr, y - 12 * dpr); o.stroke();
        o.fillText(lab.text, x + 25 * dpr, y - 14 * dpr);
      }
      o.globalAlpha = 1;
    }
  }
}

window.BrainView = BrainView;
window.BRAIN_PALETTE = BRAIN_PALETTE;
