"""
eyes.py - the fly's eyes: webcam frames -> flyvis optic lobe -> drive for the MaleCNS brain.

Why a second model: early fly vision is ANALOG. Photoreceptors and lamina cells don't spike,
they slide up and down smoothly, and the spiking brain model can't reproduce that (tested:
everything past the photoreceptors went silent). flyvis (Lappalainen et al. 2024, Nature, MIT)
is a connectome-built analog model of the optic lobe, trained so its T4/T5 cells became real
motion detectors. It runs both eyes here:

    camera frame --split--> left half / right half (mirrored)  --BoxEye--> 721 hex "pixels" per eye
      --flyvis (45k analog neurons per eye, dt 10 ms)--> activity of 65 cell types per column

Then every flyvis cell type that also exists in MaleCNS (Tm, Mi, T4/T5, TmY, ...) drives those
MaleCNS neurons as Poisson spikes, matched by position in the eye. From there the spiking MaleCNS
brain takes over: lobula plate and lobula -> looming detectors -> giant fiber -> jump.

Image orientation (checked with moving bars: flyvis T4a prefers motion toward -x):
    flyvis image x = FORWARD, y = DOWN, for both eyes, so T4a = front-to-back, matching the
    MaleCNS names (a front-to-back, b back-to-front, c up, d down). The fly faces the camera,
    so the camera's left half is the fly's left eye; the right eye's half is mirrored.

Needs Python 3.12 + flyvis (see .venv-eye in README). Without it, the Eyes page is disabled.
"""
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
os.environ.setdefault("FLYVIS_ROOT_DIR", str(HERE / "data" / "flyvis"))

import flyvis  # noqa: E402
from flyvis.datasets.rendering import BoxEye  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DT = 0.01          # s per flyvis step
GAIN = float(os.environ.get("FLY_GAIN", 90))    # Hz of MaleCNS drive per unit of activity above the running baseline
ADAPT_S = float(os.environ.get("FLY_ADAPT", 0.3))  # s: how fast the bridge's baseline follows each cell (only CHANGE drives the brain)
MAX_HZ = 300.0
EXTENT, SPACING = 15, 13   # BoxEye defaults the pretrained models were trained with (721 hexals)

# cell types shown on the Eyes page, in pipeline order
SHOW = [
    ("R1", "Photoreceptors", "what light reaches the eye (R1)"),
    ("L1", "Lamina L1", "ON pathway start: brightening"),
    ("L2", "Lamina L2", "OFF pathway start: darkening"),
    ("Mi1", "Medulla Mi1", "ON edges"),
    ("Tm9", "Medulla Tm9", "OFF edges"),
    ("T4", "Motion T4 (ON)", "colour = direction of moving bright edges"),
    ("T5", "Motion T5 (OFF)", "colour = direction of moving dark edges"),
]


class FlyEyes:
    def __init__(self, checkpoint="flow/0000/000"):
        self.dev = flyvis.device
        self.net = flyvis.NetworkView(checkpoint).init_network()
        self.net.eval()
        ct = self.net.connectome
        self.types = np.array([t.decode() for t in ct.nodes.type[:]])
        u, v = ct.nodes.u[:], ct.nodes.v[:]
        self.u, self.v = u, v
        self.box = BoxEye(extent=EXTENT, kernel_size=SPACING)
        self.frame_px = int(self.box.min_frame_size.max())            # 391
        self.params = self.net._param_api()
        with torch.no_grad():
            self.state = self.net.steady_state(1.0, DT, batch_size=2, value=0.5)
            self.baseline = self.state.nodes.activity.clone()          # (2, nodes) grey-screen activity
        self.activity = self.baseline.clone()
        self.running = self.baseline.clone()                           # slow average of each node, for the bridge
        # nodes of each shown type, in one shared hexal order (sorted by u, v)
        self.type_nodes = {}
        real = [s[0] for s in SHOW if s[0] in set(self.types)] + [f"T{k}{d}" for k in "45" for d in "abcd"]
        for t in real:                                                 # ("T4"/"T5" are display groups, not types)
            idx = np.where(self.types == t)[0]
            idx = idx[np.lexsort((v[idx], u[idx]))]
            self.type_nodes[t] = torch.from_numpy(idx).to(self.dev)
        r1 = self.type_nodes["R1"].cpu().numpy()
        self.hex_uv = np.stack([u[r1], v[r1]], 1)                      # 721 hexals, same order for every type
        self.input_luminance = torch.zeros(2, len(self.hex_uv), device=self.dev)

    # ------------------------------------------------------------------ frames in
    def step(self, gray, steps=1):
        """gray: HxW uint8 numpy (whole camera frame, NOT mirrored). Advances flyvis `steps` x 10 ms."""
        # flyvis makes new tensors on torch's DEFAULT device, and that default is per-thread. The web
        # server answers each request on a fresh thread, so pin the device here or tensors land on the CPU.
        with torch.device(self.dev):
            self._step(gray, steps)

    def _step(self, gray, steps):
        g = torch.from_numpy(gray).to(self.dev, torch.float32)[None, None] / 255.0
        h, w = gray.shape
        left, right = g[..., : w // 2], torch.flip(g[..., w - w // 2:], dims=[-1])
        eyes = torch.cat([left, right], 0)                               # (2, 1, h, w/2)
        eyes = F.interpolate(eyes, size=(self.frame_px, self.frame_px), mode="bilinear", align_corners=False)
        with torch.no_grad():
            x = self.box(eyes[:, 0][:, None], ftype="mean")              # (2, 1, 1, hexals)
            self.input_luminance = x[:, 0, 0]
            stim = self.net.stimulus
            stim.zero(2, 1)
            stim.add_input(x)
            xt = stim()[:, 0]
            for _ in range(steps):
                self.state = self.net._next_state(self.params, self.state, xt, DT)
            self.activity = self.state.nodes.activity
            self.running += (self.activity - self.running) * min(1.0, steps * DT / ADAPT_S)

    # ------------------------------------------------------------------ out: pictures
    def layer_maps(self):
        """uint8 arrays for the page: per shown layer, per eye, one value per hexal."""
        out = {}
        a, base = self.activity, self.baseline
        lum = (self.input_luminance.clamp(0, 1) * 255).to(torch.uint8)
        out["R1"] = lum
        for t in ["L1", "L2", "Mi1", "Tm9"]:
            d = a[:, self.type_nodes[t]] - base[:, self.type_nodes[t]]
            s = d.abs().amax().clamp(min=0.05)
            out[t] = ((d / s) * 127 + 128).clamp(0, 255).to(torch.uint8)  # 128 = no change from grey
        for k in "45":
            # strongest direction per hexal + its strength, so the page can colour by direction
            stack = torch.stack([(a[:, self.type_nodes[f"T{k}{d}"]] - base[:, self.type_nodes[f"T{k}{d}"]]).clamp(min=0)
                                 for d in "abcd"], 0)                     # (4, 2, hexals)
            strength, which = stack.max(0)
            out[f"T{k}"] = (which.to(torch.uint8) << 6) | (strength / 2.5 * 63).clamp(0, 63).to(torch.uint8)
        return {k: v.cpu().numpy() for k, v in out.items()}

    # ------------------------------------------------------------------ out: drive for MaleCNS
    def build_bridge(self, male_types, eye_all, fwd_all, up_all, device):
        """Match MaleCNS neurons to flyvis nodes of the same type at the nearest hexal."""
        hex_xy = np.stack([SPACING * self.hex_uv[:, 1],                   # x = d * v
                           SPACING * (self.hex_uv[:, 0] + self.hex_uv[:, 1] / 2)], 1)  # y = d * (u + v/2)
        r = np.abs(hex_xy).max(0)
        shared = sorted(set(self.types) & set(male_types))
        neurons, nodes = [], []
        for e in (0, 1):
            m = eye_all == e
            f0, f1 = np.nanmin(fwd_all[m]), np.nanmax(fwd_all[m])
            u0, u1 = np.nanmin(up_all[m]), np.nanmax(up_all[m])
            for t in shared:
                ids = np.where((male_types == t) & m & ~np.isnan(fwd_all))[0]
                fv = np.where(self.types == t)[0]
                if not len(ids) or not len(fv):
                    continue
                fv_xy = np.stack([SPACING * self.v[fv], SPACING * (self.u[fv] + self.v[fv] / 2)], 1)
                # eye position -> flyvis image: x = forward, y = down, stretched to the hex patch
                px = ((fwd_all[ids] - f0) / (f1 - f0) * 2 - 1) * r[0]
                py = -((up_all[ids] - u0) / (u1 - u0) * 2 - 1) * r[1]
                d2 = (px[:, None] - fv_xy[None, :, 0]) ** 2 + (py[:, None] - fv_xy[None, :, 1]) ** 2
                neurons.append(ids)
                nodes.append(e * len(self.types) + fv[d2.argmin(1)])      # flatten (eye, node)
        self.bridge_neurons = torch.from_numpy(np.concatenate(neurons)).to(device)
        self.bridge_nodes = torch.from_numpy(np.concatenate(nodes)).to(self.dev)
        self.bridge_types = shared
        return self.bridge_neurons

    def bridge_rates(self):
        """Hz for each bridged MaleCNS neuron: how far its flyvis twin is above grey baseline."""
        d = (self.activity - self.running).reshape(-1)[self.bridge_nodes]
        return (d.clamp(min=0) * GAIN).clamp(max=MAX_HZ)
