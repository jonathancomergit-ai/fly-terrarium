"""
server.py - runs the fly brain on the GPU and serves the terrarium page.

    python server.py            then open http://127.0.0.1:8793

How the pieces talk (like a game client + game server on one machine):
  browser  --POST /frame {senses, silence}-->  server
  browser  <-- binary: which neurons fired + motor rates --  server
The brain runs in its own thread at real time (1 ms of fly = 1 ms of wall clock).
The browser owns the world (fly body, sugar, fans, swatter) and asks ~30x a second.

Only uses the standard library for the web part (no Flask), so nothing else to install.
"""
import json
import struct
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from sim import ACTIONS, SENSES, FlyBrain

HERE = Path(__file__).parent
WEB = HERE / "web"
PORT = 8793

# Already running (e.g. Play pressed twice in the Dev HUD)? Just show the page and quit,
# instead of loading a second brain onto the GPU and failing on the busy port.
try:
    import urllib.request
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/meta.json", timeout=1) as r:
        if b'"actions"' in r.read(300000):
            print("the terrarium is already running - opening it")
            webbrowser.open(f"http://127.0.0.1:{PORT}")
            sys.exit(0)
except OSError:
    pass  # nothing there (or something else is): carry on and start normally

print("loading connectome onto the GPU...")
brain = FlyBrain()
lock = threading.Lock()
state = {"speed": 1.0, "paused": False, "sim_rate": 0.0, "last_frame": time.time()}

# ---------------------------------------------------------------- static brain geometry (sent once)
pos = brain.pos.astype(np.float32)
centre = (pos.min(0) + pos.max(0)) / 2
scale = (pos.max(0) - pos.min(0)).max() / 2
# EM axes: x = left/right, y = top->bottom depth, z = front of brain -> end of nerve cord.
# Browser wants x right, y up, z toward viewer: brain on top, nerve cord hanging below.
norm = (pos - centre) / scale
xyz = np.stack([norm[:, 0], -norm[:, 2], norm[:, 1]], axis=1).astype(np.float32)
meta_json = json.loads((HERE / "data" / "neurons.json").read_text())
geometry = xyz.tobytes() + np.asarray(meta_json["superclass"], np.uint8).tobytes()
meta = json.dumps({
    "n": brain.N,
    "superclasses": meta_json["superclasses"],
    "groups": meta_json["groups"],
    "senses": SENSES,
    "actions": ACTIONS,
}).encode()


# ---------------------------------------------------------------- brain thread
def brain_loop():
    """Keep fly-time in step with wall-time. Runs 1.8 ms blocks until caught up."""
    sim_origin = brain.sim_ms
    wall_origin = time.perf_counter()
    rate_t, rate_sim = time.perf_counter(), brain.sim_ms
    while True:
        if state["paused"] or time.time() - state["last_frame"] > 5:
            # nobody watching (or paused): idle, and re-anchor the clock so we don't sprint to catch up
            time.sleep(0.05)
            sim_origin, wall_origin = brain.sim_ms, time.perf_counter()
            continue
        target = sim_origin + (time.perf_counter() - wall_origin) * 1000 * state["speed"]
        if brain.sim_ms < target:
            behind = target - brain.sim_ms
            if behind > 200:  # GPU can't keep up at this speed: drop the debt instead of spiralling
                sim_origin, wall_origin = brain.sim_ms, time.perf_counter()
            with lock:
                brain.run_ms(7.2)  # 4 blocks
        else:
            time.sleep(0.001)
        now = time.perf_counter()
        if now - rate_t > 0.5:
            state["sim_rate"] = (brain.sim_ms - rate_sim) / ((now - rate_t) * 1000)
            rate_t, rate_sim = now, brain.sim_ms


# ---------------------------------------------------------------- HTTP
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(WEB), **k)

    def log_message(self, *a):  # keep the console quiet
        pass

    def send_bytes(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/geometry.bin":
            return self.send_bytes(geometry, "application/octet-stream")
        if self.path == "/meta.json":
            return self.send_bytes(meta, "application/json")
        return super().do_GET()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
        req = json.loads(body or b"{}")
        state["last_frame"] = time.time()

        if self.path == "/reset":
            with lock:
                brain.reset()
            return self.send_bytes(b"{}", "application/json")

        if self.path != "/frame":
            self.send_error(404)
            return

        if "speed" in req:
            state["speed"] = max(0.05, min(2.0, float(req["speed"])))
        if "paused" in req:
            state["paused"] = bool(req["paused"])
        with lock:
            senses = req.get("senses", {})
            rounded = {k: round(float(senses.get(k, 0)), 1) for k in SENSES}
            if rounded != brain.rates:
                brain.set_rates(**rounded)
            if "silence" in req:  # list of group names to lesion
                alive = torch.ones(brain.N, dtype=torch.bool, device=brain.dev)
                for gname in req["silence"]:
                    if gname in brain.groups:
                        alive[brain.groups[gname]] = False
                brain.alive.copy_(alive)
            idx, vals, acts = brain.take_spikes()
            sim_ms = brain.sim_ms

        head = json.dumps({
            "acts": acts, "simMs": sim_ms, "active": int(len(idx)),
            "simRate": round(state["sim_rate"], 2),
        }).encode()
        head += b" " * (-(len(head) + 4) % 4)  # pad so the int32 array starts on a 4-byte boundary
        out = struct.pack("<I", len(head)) + head + idx.astype("<i4").tobytes() + vals.tobytes()
        self.send_bytes(out, "application/octet-stream")


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets two servers share a port silently. Refuse instead.
    allow_reuse_address = False
    daemon_threads = True


if __name__ == "__main__":
    print("warming up (recording the CUDA graph)...")
    brain.run_ms(20)
    brain.reset()
    threading.Thread(target=brain_loop, daemon=True).start()
    try:
        srv = Server(("127.0.0.1", PORT), Handler)
    except OSError:
        sys.exit(f"port {PORT} is already in use - is the terrarium already running?")
    url = f"http://127.0.0.1:{PORT}"
    print(f"fly terrarium running at {url}   (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
