"""
sim.py - the whole MaleCNS fly brain as spiking neurons on the GPU.

Model: leaky integrate-and-fire (LIF), same equations and numbers as
Shiu et al. 2024, "A Drosophila computational brain model reveals sensorimotor processing" (Nature).
Every neuron is just two numbers:
    v  membrane voltage (mV). Leaks back to rest, fires a spike when it crosses threshold.
    g  synaptic input (mV). Jumps when an upstream neuron spikes, then decays.
Every step (dt = 0.1 ms):
    v += dt * (g - (v - V_REST)) / TAU_M
    g -= dt * g / TAU_SYN
    spike if v > V_TH  ->  v = V_RESET, silent for T_REF
    each spike reaches its targets DELAY ms later, adding  W_SYN * synapse_count * (+1 or -1)

The wiring (who connects to whom, how many synapses, excite or inhibit) comes 100%
from the connectome. Nothing is trained. That's the fun part.

Speed trick: most neurons are silent most of the time, so instead of multiplying the
whole 25M-edge matrix every step we only walk the outgoing edges of neurons that spiked
(CSR slices), like updating only the dirty sprites in a game.

Try it headless:   python sim.py        (runs the sanity experiments and prints rates)
"""
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent

# ---- Shiu et al. 2024 parameters
DT = 0.1          # ms per step
V_REST = -52.0    # mV
V_RESET = -52.0
V_TH = -45.0
TAU_M = 20.0      # ms
TAU_SYN = 5.0     # ms
T_REF = 2.2       # ms refractory
DELAY = 1.8       # ms synaptic delay
# Shiu used 0.275 mV per synapse for FlyWire. The MaleCNS counts synapses with a different detector
# and also includes the optic lobes and nerve cord, so at 0.275 whole circuits locked into
# seizure-like loops. Sweeping 0.3 - 0.6x: at 0.5x every reflex below still works and the brain
# calms down afterwards. (see NOTES.md "Calibration")
W_SYN = 0.275 * 0.5  # mV per synapse
W_INPUT = 68.75   # mV per external Poisson spike (strong enough to make a sensory neuron fire)

SENSES = ["sugar", "bitter", "wind", "sound", "looming", "light"]
ACTIONS = ["forward", "turnL", "turnR", "backward", "escape", "feed", "groom"]


class FlyBrain:
    def __init__(self, path=HERE / "data" / "brain.npz", device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        d = np.load(path)
        self.N = N = len(d["ids"])
        dev = self.dev
        self.pos = d["pos"]
        self.indptr = torch.from_numpy(d["indptr"]).to(dev)
        self.post = torch.from_numpy(d["post"].astype(np.int64)).to(dev)
        sign = torch.from_numpy(d["sign"]).to(dev)
        # weight already multiplied by the sign of the PRESYNAPTIC neuron's transmitter
        pre = torch.repeat_interleave(torch.arange(N, device=dev), self.indptr[1:] - self.indptr[:-1])
        self.wsig = torch.from_numpy(d["w"]).to(dev) * sign[pre] * W_SYN
        del pre
        self.groups = {k[2:]: torch.from_numpy(d[k].astype(np.int64)).to(dev) for k in d.files if k.startswith("g_")}

        self.v = torch.full((N,), V_REST, device=dev)
        self.g = torch.zeros(N, device=dev)
        self.D = D = int(round(DELAY / DT))                # 18 steps
        self.R = int(round(T_REF / DT))                    # 22 steps
        self.ref = torch.zeros(N, dtype=torch.int32, device=dev)  # refractory steps left
        self.ring = torch.zeros((D, N), device=dev)        # delayed input, one row per step of the block
        self.S = torch.zeros((D, N), dtype=torch.bool, device=dev)  # who spiked at each step of the block
        self.input_p = torch.zeros(N, device=dev)          # per-neuron chance of an external spike per step
        self.bias = torch.zeros(N, device=dev)             # mV of steady drive (unused by default)
        self.sense_p = torch.zeros(N, device=dev)          # input from the named senses (sugar, wind...)
        self.extra_p = torch.zeros(N, device=dev)          # input from the eyes (flyvis bridge)
        self.alive = torch.ones(N, dtype=torch.bool, device=dev)    # False = lesioned (silenced)
        self.spike_count = torch.zeros(N, dtype=torch.int32, device=dev)  # spikes since last take_spikes()
        self.rates = {k: 0.0 for k in SENSES}              # Hz, set by the world via set_rates()
        self.sim_ms = 0.0
        self.graph = None

    # ------------------------------------------------------------------ inputs
    def set_rates(self, **rates):
        """Sense name -> Poisson rate in Hz. e.g. set_rates(sugar=150, wind=0)"""
        self.rates.update({k: float(v) for k, v in rates.items() if k in SENSES})
        p = torch.zeros(self.N, device=self.dev)
        for k, hz in self.rates.items():
            if hz > 0:
                p[self.groups[k]] = hz * DT / 1000.0
        self.sense_p.copy_(p)
        self.input_p.copy_(self.sense_p + self.extra_p)   # copy_ (not =) so the CUDA graph sees it

    def set_extra_rates(self, idx, hz):
        """Poisson drive (Hz) for arbitrary neurons, e.g. the visual neurons fed by flyvis."""
        self.extra_p.zero_()
        self.extra_p[idx] = hz * DT / 1000.0
        self.input_p.copy_(self.sense_p + self.extra_p)

    def make_inputs(self, idx):
        """Cut the synapses INTO these neurons: they're now driven from outside (like sensory neurons)."""
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        mask[idx] = True
        # rebuild the CSR without those edges: a dead edge still costs GPU time every spike
        pre = torch.repeat_interleave(torch.arange(self.N, device=self.dev), self.indptr[1:] - self.indptr[:-1])
        keep = ~mask[self.post]
        self.post, self.wsig = self.post[keep], self.wsig[keep]
        counts = torch.bincount(pre[keep], minlength=self.N)
        self.indptr = torch.zeros(self.N + 1, dtype=self.indptr.dtype, device=self.dev)
        self.indptr[1:] = torch.cumsum(counts, 0)
        return int((~keep).sum())

    def reset(self):
        self.v.fill_(V_REST); self.g.zero_(); self.ring.zero_(); self.ref.zero_()
        self.spike_count.zero_()

    # ------------------------------------------------------------------ the core
    def _block(self):
        """D steps of pure per-neuron math. No spike can reach anyone within D steps
        (that's the synaptic delay), so these steps never need the wiring at all."""
        for k in range(self.D):
            self.g += self.ring[k]
            self.ring[k].zero_()
            self.g += (torch.rand_like(self.v) < self.input_p) * W_INPUT
            free = self.ref <= 0
            self.v += free * (DT * (self.g + self.bias - (self.v - V_REST)) / TAU_M)
            self.g -= DT * self.g / TAU_SYN
            spk = (self.v > V_TH) & self.alive
            self.S[k] = spk
            self.v.masked_fill_(spk, V_RESET)
            self.ref.sub_(1).masked_fill_(spk, self.R)
            self.spike_count += spk

    def _propagate(self):
        """Deliver every spike from the block: spike at step k lands in ring[k] of the NEXT block,
        i.e. exactly D steps (1.8 ms) later. Only the outgoing edges of neurons that fired are touched."""
        ks, idx = self.S.nonzero(as_tuple=True)
        if idx.numel() == 0:
            return
        starts = self.indptr[idx]
        counts = self.indptr[idx + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        # CSR slice expansion: edge ids for all fired neurons, without a Python loop
        offs = torch.repeat_interleave(starts - (torch.cumsum(counts, 0) - counts), counts, output_size=total)
        eidx = torch.arange(total, device=self.dev) + offs
        row = torch.repeat_interleave(ks, counts, output_size=total)
        self.ring.view(-1).index_add_(0, row * self.N + self.post[eidx], self.wsig[eidx])

    def run_ms(self, ms):
        """Advance the brain by (about) ms milliseconds, in whole 1.8 ms blocks."""
        if self.dev.type == "cuda" and self.graph is None:
            # CUDA graph = record the D-step block once, then replay it as ONE GPU call.
            # Like caching a compiled function instead of re-interpreting it every frame.
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._block()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._block()
            self.reset()
        for _ in range(max(1, round(ms / (self.D * DT)))):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._block()
            self._propagate()
            self.sim_ms += self.D * DT

    # ------------------------------------------------------------------ outputs
    def take_spikes(self):
        """(neuron indices that fired, their spike counts, {action: total spikes}) since last call."""
        c = self.spike_count
        acts = {k: int(c[self.groups[k]].sum()) for k in ACTIONS}
        idx = c.nonzero().squeeze(1)
        vals = c[idx].clamp(max=255).to(torch.uint8)
        c.zero_()
        return idx.to(torch.int32).cpu().numpy(), vals.cpu().numpy(), acts


def experiment(brain, label, seconds=1.0, **rates):
    brain.reset()
    brain.set_rates(**{k: rates.get(k, 0) for k in SENSES})
    brain.take_spikes()
    torch.cuda.synchronize()
    t0 = time.time()
    brain.run_ms(seconds * 1000)
    idx, vals, acts = brain.take_spikes()
    wall = time.time() - t0
    hz = {k: round(v / seconds / max(1, len(brain.groups[k])), 1) for k, v in acts.items()}
    print(f"{label:16s} active={len(idx):6d}  speed={seconds/wall:5.2f}x realtime  Hz/neuron {hz}")


if __name__ == "__main__":
    t0 = time.time()
    b = FlyBrain()
    print(f"loaded {b.N} neurons, {len(b.post):,} edges on {b.dev} in {time.time()-t0:.1f}s")
    experiment(b, "nothing")
    experiment(b, "sugar 150Hz", sugar=150)
    experiment(b, "bitter 150Hz", bitter=150)
    experiment(b, "sugar + bitter", sugar=150, bitter=150)
    experiment(b, "wind 150Hz", wind=150)
    experiment(b, "sound 150Hz", sound=150)
    experiment(b, "looming 150Hz", looming=150)
    experiment(b, "light 50Hz", light=50)
