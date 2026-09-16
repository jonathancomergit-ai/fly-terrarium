"""
prep_eyes.py - build the fly's eye map: where in its visual field every visual neuron "looks".

Run after prep.py:   python prep_eyes.py      -> data/eyes.npz + data/eyes.json

How the map is made (all from the MaleCNS data, nothing hand-placed):
  1. Janelia assigned 12 columnar cell types (L1-L5, Mi1, Mi4, Mi9, Tm1, Tm2, Tm9, Tm20, C3, T1)
     to optic lobe columns with two hex coordinates. One column = one ommatidium (one "pixel").
  2. Lattice: neighbouring columns are the offsets (+-1,0), (0,+-1), +-(1,1) - checked by counting
     which offsets carry the most synapses between columns. That's a hex grid with 120 deg axes.
  3. Orientation: the dorsal-rim photoreceptors (R7d/R8d, only found at the top of real eyes) sit at
     high hex1+hex2, so that's UP. hex1-hex2 points toward the front of the brain (antennal lobe
     side), so that's FORWARD.
        up      = (hex1 + hex2) / 2
        forward = (hex1 - hex2) * sqrt(3) / 2        (unit = one column spacing)
  4. Neurons without a column get one from their wiring, like working out someone's neighbourhood
     from who they talk to:
       photoreceptors  -> the column of the lamina/medulla cells they send to
       everyone else   -> synapse-weighted average position of their INPUTS, repeated a few rounds
     That weighted average is the neuron's receptive field centre; the spread is its size.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
V = "male-cns-v1.0"

ann = pd.read_feather(RAW / f"body-annotations-{V}-minconf-0.5.feather")
ann = ann[ann.status == "Traced"].reset_index(drop=True)
d = np.load(HERE / "data" / "brain.npz")
assert (d["ids"] == ann.bodyId.to_numpy()).all(), "brain.npz is out of date - run prep.py first"
N = len(ann)
indptr, post, w = d["indptr"], d["post"].astype(np.int64), d["w"].astype(np.float64)
pre = np.repeat(np.arange(N), np.diff(indptr))

typ = ann["type"].fillna("").to_numpy().astype(str)
sup = ann["superclass"].fillna("").to_numpy().astype(str)
side_s = ann["somaSide"].fillna(ann["rootSide"]).fillna("").to_numpy()
h1, h2 = ann.assignedOlHex1.to_numpy(), ann.assignedOlHex2.to_numpy()

# positions: fwd, up, and eye (0 = left, 1 = right). NaN = not placed yet.
fwd = (h1 - h2) * np.sqrt(3) / 2
up = (h1 + h2) / 2
eye = np.where(side_s == "R", 1.0, np.where(side_s == "L", 0.0, np.nan))
placed = ~np.isnan(h1) & ~np.isnan(eye)
fwd[~placed] = np.nan; up[~placed] = np.nan
spread = np.where(placed, 0.0, np.nan)
print(f"{placed.sum()} neurons with a Janelia column")

# ---------------------------------------------------------------- columns (the "pixels")
col_keys = sorted({(int(e), int(a), int(b)) for e, a, b in zip(eye[placed], h1[placed], h2[placed])})
col_index = {k: i for i, k in enumerate(col_keys)}
cols = np.array([[e, (a - b) * np.sqrt(3) / 2, (a + b) / 2] for e, a, b in col_keys], np.float32)
print(f"{len(cols)} columns ({int((cols[:, 0] == 0).sum())} left eye, {int((cols[:, 0] == 1).sum())} right eye)")
neuron_col = np.full(N, -1, np.int32)
for i in np.where(placed)[0]:
    neuron_col[i] = col_index[(int(eye[i]), int(h1[i]), int(h2[i]))]

# ---------------------------------------------------------------- photoreceptors: column of their targets
photo_types = ["R1-R6", "R7p", "R7y", "R7d", "R7_unclear", "R8p", "R8y", "R8d", "R8_unclear", "R7R8_unclear"]
photo = np.where(np.isin(typ, photo_types))[0]
is_photo = np.zeros(N, bool); is_photo[photo] = True
e = is_photo[pre] & (neuron_col[post] >= 0)
# synapses per (photoreceptor, target column); keep each photoreceptor's strongest column
df =pd.DataFrame({"p": pre[e], "c": neuron_col[post[e]], "w": w[e]}).groupby(["p", "c"]).w.sum().reset_index()
df = df.sort_values("w", ascending=False).drop_duplicates("p")
for p_, c in zip(df.p, df.c):
    neuron_col[p_] = c
    eye[p_], fwd[p_], up[p_] = cols[c]
    spread[p_] = 0.0
photo_col = neuron_col[photo]
print(f"{len(photo)} photoreceptors, {int((photo_col >= 0).sum())} placed in a column")

# ---------------------------------------------------------------- everyone else in the visual system
visual = np.isin(sup, ["ol_intrinsic", "visual_projection", "ol_sensory"]) | np.char.startswith(typ, "LPLC") \
    | np.char.startswith(typ, "LC") | np.char.startswith(typ, "LT")
for rnd in range(4):
    todo = visual & np.isnan(fwd)
    e = todo[post] & ~np.isnan(fwd[pre])
    if not e.any():
        break
    q, p_, ww = post[e], pre[e], w[e]
    newly = 0
    fw = np.zeros(N); uw = np.zeros(N); ws = np.zeros(N); f2 = np.zeros(N); u2 = np.zeros(N)
    # each neuron sits in the eye most of its input comes from
    right = np.bincount(q, weights=ww * (eye[p_] == 1), minlength=N)
    total = np.bincount(q, weights=ww, minlength=N)
    eye_q = (right[q] * 2 > total[q]).astype(float)
    same = eye[p_] == eye_q
    q, p_, ww = q[same], p_[same], ww[same]
    ws = np.bincount(q, weights=ww, minlength=N)
    fw = np.bincount(q, weights=ww * fwd[p_], minlength=N)
    uw = np.bincount(q, weights=ww * up[p_], minlength=N)
    f2 = np.bincount(q, weights=ww * fwd[p_] ** 2, minlength=N)
    u2 = np.bincount(q, weights=ww * up[p_] ** 2, minlength=N)
    ok = todo & (ws >= 3)
    fwd[ok] = fw[ok] / ws[ok]
    up[ok] = uw[ok] / ws[ok]
    eye[ok] = (right[ok] * 2 > total[ok]).astype(float)
    spread[ok] = np.sqrt(np.maximum(0, f2[ok] / ws[ok] - fwd[ok] ** 2 + u2[ok] / ws[ok] - up[ok] ** 2))
    print(f"  round {rnd + 1}: placed {int(ok.sum())} more")

# ---------------------------------------------------------------- display layers
LAYERS = [
    ("photoreceptors", "Photoreceptors", "what light hits the eye", photo_types),
    ("lamina", "Lamina", "L1-L5: contrast, split into ON and OFF", ["L1", "L2", "L3", "L4", "L5"]),
    ("on", "Medulla ON", "Mi1, Tm3, Mi4, Mi9: brightening edges", ["Mi1", "Tm3", "Mi4", "Mi9"]),
    ("off", "Medulla OFF", "Tm1, Tm2, Tm4, Tm9: darkening edges", ["Tm1", "Tm2", "Tm4", "Tm9"]),
    ("motion", "Motion (T4/T5)", "direction detectors, colour = direction",
     ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]),
    ("features", "Looming & objects", "LC4, LPLC2 (looming), LC11 (small objects), LPLC1",
     ["LC4", "LPLC2", "LC11", "LPLC1"]),
]
layer_of = np.full(N, -1, np.int8)
for li, (_, _, _, types) in enumerate(LAYERS):
    layer_of[np.isin(typ, types) & ~np.isnan(fwd)] = li
# motion direction per T4/T5 subtype: a = front-to-back, b = back-to-front, c = upward, d = downward
direction = np.full(N, -1, np.int8)
for k, letter in enumerate("abcd"):
    direction[np.isin(typ, [f"T4{letter}", f"T5{letter}"])] = k

show = np.where(layer_of >= 0)[0]
print("layer sizes:", {LAYERS[li][0]: int((layer_of == li).sum()) for li in range(len(LAYERS))})

np.savez(HERE / "data" / "eyes.npz",
         cols=cols, photo=photo.astype(np.int32), photo_col=photo_col.astype(np.int32),
         show=show.astype(np.int32),
         # every neuron's eye position (NaN = not a visual neuron); used to wire flyvis into the brain
         eye_all=eye.astype(np.float32), fwd_all=fwd.astype(np.float32), up_all=up.astype(np.float32))
json.dump({
    "cols": np.round(cols, 3).tolist(),                       # [eye, fwd, up] per column
    "layers": [{"id": a, "name": b, "desc": c} for a, b, c, _ in LAYERS],
    "show": show.tolist(),                                     # neuron indices drawn in the layer maps
    "eye": eye[show].astype(int).tolist(),
    "fwd": np.round(fwd[show], 2).tolist(),
    "up": np.round(up[show], 2).tolist(),
    "spread": np.round(np.nan_to_num(spread[show]), 2).tolist(),
    "layer": layer_of[show].tolist(),
    "dir": direction[show].tolist(),
    "type": typ[show].tolist(),
}, open(HERE / "data" / "eyes.json", "w"))
print("wrote data/eyes.npz and data/eyes.json")
