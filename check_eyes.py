r"""
check_eyes.py - headless check of the whole vision chain (no browser, no camera).

    .venv-eye\Scripts\python check_eyes.py

Synthetic movies -> flyvis eyes -> MaleCNS brain; prints Hz per neuron for the motion detectors,
looming detectors, giant fiber (GF), moonwalker (MDN) and turning DNs. Expect looming >> receding > 0,
and ~0 for grey, a still disc and a small dot. Close anything heavy on the GPU first.
"""
import time, json, numpy as np, torch
from sim import FlyBrain
from eyes import FlyEyes
brain = FlyBrain()
typ = np.array(json.load(open("data/neurons.json"))["type"])
E = np.load("data/eyes.npz")
eyes = FlyEyes()
idx = eyes.build_bridge(typ, E["eye_all"], E["fwd_all"], E["up_all"], brain.dev)
brain.make_inputs(idx)
print("bridged", len(idx), "MaleCNS neurons over", len(eyes.bridge_types), "types")
G = {k: torch.from_numpy(np.where(np.isin(typ, v))[0]).to(brain.dev) for k, v in
     {"T4": ["T4a","T4b","T4c","T4d"], "LPLC2": ["LPLC2"], "LC4": ["LC4"], "LPLC1": ["LPLC1"], "LC11": ["LC11"], "GF": ["DNp01"], "MDN": ["MDN"], "DNa": ["DNa01","DNa02"]}.items()}
H, W = 180, 320
yy, xx = np.mgrid[0:H, 0:W]
def run(label, frame_fn, secs=1.5):
    brain.reset(); brain.take_spikes()
    with torch.no_grad():
        eyes.state = eyes.net.steady_state(1.0, 0.01, batch_size=2, value=0.5)
        eyes.running = eyes.state.nodes.activity.clone()
    for i in range(100):                      # 1 s looking at the first frame, so onset isn't a flash
        eyes.step(frame_fn(0.0))
        brain.set_extra_rates(idx, eyes.bridge_rates())
        brain.run_ms(10)
    brain.take_spikes()
    tot = {k: 0 for k in G}
    t0 = time.time()
    n = int(secs / 0.01)
    for i in range(n):
        eyes.step(frame_fn(i / n))
        brain.set_extra_rates(idx, eyes.bridge_rates())
        brain.run_ms(10)
        c = brain.spike_count
        for k, g in G.items(): tot[k] += int(c[g].sum())
        brain.spike_count.zero_()
    print(f"{label:22s} " + " ".join(f"{k}={tot[k]/secs/len(G[k]):.1f}" for k in G) + f"  ({secs/(time.time()-t0):.2f}x rt)")
grey = lambda p: np.full((H, W), 128, np.uint8)
disc = lambda r: np.where(np.hypot(xx - 240, yy - 90) < r, 20, 200).astype(np.uint8)
run("grey", grey)
run("looming (right eye)", lambda p: disc(3 + 150 * p ** 3))
run("receding", lambda p: disc(3 + 150 * (1 - p) ** 3))
run("static big disc", lambda p: disc(80))
run("dark bar sweep", lambda p: np.where(np.abs(xx - p * W) < 15, 20, 200).astype(np.uint8))
run("small dot moving", lambda p: np.where(np.hypot(xx - (180 + 120 * p), yy - 90) < 6, 20, 200).astype(np.uint8))
