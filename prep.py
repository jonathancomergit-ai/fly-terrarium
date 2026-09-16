"""
prep.py - turn the raw MaleCNS v1.0 download into one compact file the sim can load fast.

Run once (takes a minute or two):   python prep.py

Input  (data/raw/, from male-cns.janelia.org/download):
  body-annotations-...feather     who each neuron is (type, class, soma position)
  body-neurotransmitters-...feather  what chemical each neuron releases (decides + or - synapse)
  connectome-weights-...feather   every (pre, post, synapse count) edge, ~1 GB

Output (data/):
  brain.npz    neurons, positions, signed edge list in CSR order (grouped by presynaptic neuron)
  neurons.json small table for the browser: type/class per neuron + named groups we stimulate/read

Python analogy: this is the "load level from disk and bake it into a numpy array" step
you'd do before a game starts, so the game loop never touches pandas.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data"
V = "male-cns-v1.0"

URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = [f"body-annotations-{V}-minconf-0.5.feather",     # 14 MB
         f"body-neurotransmitters-{V}.feather",           # 43 MB
         f"connectome-weights-{V}-minconf-0.5.feather"]   # 1.05 GB


def download_missing():
    """First run: fetch the three public MaleCNS files (CC-BY 4.0, HHMI Janelia + Google)."""
    import urllib.request
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists():
            continue
        print(f"downloading {name} ...")
        tmp = dest.with_suffix(".part")

        def progress(blocks, bsize, total):
            if total > 0:
                print(f"\r  {min(100, blocks * bsize * 100 // total):3d}%", end="", flush=True)
        urllib.request.urlretrieve(URL + name, tmp, progress)
        tmp.rename(dest)  # only a finished download gets the real name
        print()


download_missing()
t0 = time.time()

# ---------------------------------------------------------------- neurons
ann = pd.read_feather(RAW / f"body-annotations-{V}-minconf-0.5.feather")
# "Traced" = a real, proofread neuron. The rest are glia, orphan fragments, etc.
ann = ann[ann.status == "Traced"].reset_index(drop=True)
ids = ann.bodyId.to_numpy(np.int64)
N = len(ids)
typ = ann["type"].fillna("").to_numpy()
cls = ann["class"].fillna("").to_numpy()
sup = ann["superclass"].fillna("").to_numpy()
print(f"{N} traced neurons")

# Soma position (8 nm voxels). About 15% have no soma (e.g. sensory axons whose cell
# body is out in the antenna/leg) - those get placed later at the centre of their synapses.
soma = np.full((N, 3), np.nan, np.float32)
has = ann.somaLocation.notna().to_numpy()
soma[has] = np.stack(ann.somaLocation[has].to_numpy()).astype(np.float32)

# ---------------------------------------------------------------- neurotransmitter -> sign
nt = pd.read_feather(RAW / f"body-neurotransmitters-{V}.feather", columns=["body", "consensus_nt", "predicted_nt"])
nt = nt.set_index("body").reindex(ids)
chem = nt.consensus_nt.where(nt.consensus_nt.notna() & (nt.consensus_nt != "unclear"), nt.predicted_nt)
chem = chem.fillna("unclear").to_numpy()
# Like Shiu et al. 2024 (Nature): acetylcholine excites, GABA and glutamate inhibit.
# Histamine is the photoreceptor transmitter and inhibits its targets, so it's -1 too.
# Dopamine, serotonin and octopamine are NEUROMODULATORS: they change synapses and moods over
# seconds, they don't fire their targets. Counting them as fast excitation made a
# Kenyon cell -> dopamine neuron -> Kenyon cell loop that locked on forever, so they get 0.
sign = np.where(np.isin(chem, ["gaba", "glutamate", "histamine"]), -1.0, 1.0).astype(np.float32)
sign[np.isin(chem, ["dopamine", "serotonin", "octopamine"])] = 0.0
print("transmitters:", pd.Series(chem).value_counts().to_dict())

# ---------------------------------------------------------------- edges
# Read the 1 GB file one record batch at a time and keep only traced->traced edges.
# pc.is_in is the vectorised "x in set" - thousands of times faster than a Python loop.
idset = pa.array(ids)
pre_l, post_l, w_l = [], [], []
with pa.memory_map(str(RAW / f"connectome-weights-{V}-minconf-0.5.feather")) as src:
    rdr = ipc.open_file(src)
    for b in range(rdr.num_record_batches):
        batch = rdr.get_batch(b)
        keep = pc.and_(pc.is_in(batch["body_pre"], idset), pc.is_in(batch["body_post"], idset))
        batch = batch.filter(keep)
        pre_l.append(batch["body_pre"].to_numpy())
        post_l.append(batch["body_post"].to_numpy())
        w_l.append(batch["weight"].to_numpy())
pre_b = np.concatenate(pre_l); post_b = np.concatenate(post_l); w = np.concatenate(w_l).astype(np.float32)
del pre_l, post_l, w_l

# bodyId -> row index (0..N-1). searchsorted on sorted ids is a fast dict lookup for arrays.
order = np.argsort(ids)
pre = order[np.searchsorted(ids, pre_b, sorter=order)].astype(np.int32)
post = order[np.searchsorted(ids, post_b, sorter=order)].astype(np.int32)
self_loop = pre == post
pre, post, w = pre[~self_loop], post[~self_loop], w[~self_loop]
print(f"{len(pre):,} connections, {int(w.sum()):,} synapses  ({time.time()-t0:.0f}s)")

# Neurons with no soma: park them at the mean position of the neurons they talk to.
nos = np.where(np.isnan(soma[:, 0]))[0]
if len(nos):
    fill = np.zeros((N, 3)); cnt = np.zeros(N)
    good = ~np.isnan(soma[post, 0])
    np.add.at(fill, pre[good], soma[post[good]]); np.add.at(cnt, pre[good], 1)
    good = ~np.isnan(soma[pre, 0])
    np.add.at(fill, post[good], soma[pre[good]]); np.add.at(cnt, post[good], 1)
    ok = cnt[nos] > 0
    soma[nos[ok]] = (fill[nos[ok]] / cnt[nos[ok], None]).astype(np.float32)
    soma[np.isnan(soma)] = np.nanmean(soma, axis=0)[np.where(np.isnan(soma))[1]]

# ---- edges the simple model can't handle (each one found by watching the sim seize up):
#  * Kenyon cell -> Kenyon cell: mostly axon-to-axon contacts in the mushroom body. Treated as normal
#    excitatory synapses they turn 2,000 KCs into a self-exciting blob firing flat out.
#  * anything -> a sensory neuron: synapses onto sensory axon terminals are mainly presynaptic
#    gain control. In this model sensory neurons are driven by the WORLD only.
#  * modulator (sign 0) edges: no fast effect, so skip them for speed.
is_kc = np.char.startswith(typ.astype(str), "KC")
is_sensory = np.char.find(sup.astype(str), "sensory") >= 0
drop = (is_kc[pre] & is_kc[post]) | is_sensory[post] | (sign[pre] == 0)
print(f"dropping {drop.sum():,} edges (KC-KC {int((is_kc[pre] & is_kc[post]).sum()):,}, into sensory {int(is_sensory[post].sum()):,}, modulatory {int((sign[pre] == 0).sum()):,})")
pre, post, w = pre[~drop], post[~drop], w[~drop]

# CSR order: sort edges by presynaptic neuron, so "everything neuron i sends to" is
# the slice indptr[i]:indptr[i+1]. That's what lets the GPU propagate only the spikes.
o = np.argsort(pre, kind="stable")
pre, post, w = pre[o], post[o], w[o]
indptr = np.zeros(N + 1, np.int64)
np.cumsum(np.bincount(pre, minlength=N), out=indptr[1:])

# ---------------------------------------------------------------- named groups


def of_type(*names, prefix=False):
    if prefix:
        return np.where(np.any([np.char.startswith(typ.astype(str), n) for n in names], axis=0))[0]
    return np.where(np.isin(typ, names))[0]


# The labellar taste neurons aren't labelled sugar/bitter in this release, so we read it from
# the wiring: tracing 3 synapses back from MN9 (the proboscis-extension "eat" motor neuron),
# LB3* cells push it ON and LB1* cells push it OFF - the classic sugar vs bitter split.
# The sim checks this: driving "sugar" should make MN9 fire (Shiu et al. 2024 saw the same).
sugar_grn = of_type("LB3a", "LB3b", "LB3c", "LB3d")
bitter_grn = of_type("LB1a", "LB1b", "LB1c", "LB1d", "LB1e")

groups = {
    # ---- senses (we drive these with Poisson spikes)
    "sugar":    sugar_grn,
    "bitter":   bitter_grn,
    "wind":     of_type("JO-CL", "JO-CM", "JO-CA1", "JO-CA2", "JO-ED1", "JO-ED2_a", "JO-ED2_b",
                        "JO-ED2_c", "JO-EV1", "JO-EV2", "JO-EV3", "JO-EV5", "JO-EV6"),  # JO-C/E: static deflection (wind, gravity)
    "sound":    of_type("JO-A-unclear", "JO-A1", "JO-A2", "JO-A3", "JO-A4",
                        "JO-B-unclear", "JO-B1_a", "JO-B1_b", "JO-B1_c", "JO-B2", "JO-B3", "JO-B4_a", "JO-B4_b"),  # JO-A/B: vibration
    "looming":  of_type("LPLC2", "LC4"),
    "light":    of_type("R1-R6"),
    # ---- actions (we read these out)
    "escape":   of_type("DNp01"),                   # giant fiber -> jump
    "backward": of_type("MDN"),                     # moonwalker
    "feed":     of_type("MN9"),                     # proboscis extension motor neuron
    "groom":    of_type("DNg62", "DNge078"),        # aDN1, aDN2 antennal grooming
    "turnL":    np.array([i for i in of_type("DNa01", "DNa02") if ann.somaSide[i] == "L"]),
    "turnR":    np.array([i for i in of_type("DNa01", "DNa02") if ann.somaSide[i] == "R"]),
    "forward":  of_type("DNp09", "DNg100", "DNg97"),  # P9-like forward walking, BDN2, oDN1
}
for k, v in groups.items():
    print(f"  {k:9s} {len(v):5d}  e.g. {sorted(set(typ[v]))[:6]}")

np.savez(OUT / "brain.npz", ids=ids, pos=soma, sign=sign, indptr=indptr, post=post, w=w,
         **{f"g_{k}": np.asarray(v, np.int32) for k, v in groups.items()})

# Browser-side table: compact class code per neuron for colouring the 3D brain.
sup_names = sorted(set(sup))
json.dump({
    "n": N,
    "superclasses": sup_names,
    "superclass": [sup_names.index(s) for s in sup],
    "type": typ.tolist(),
    "groups": {k: np.asarray(v).tolist() for k, v in groups.items()},
}, open(OUT / "neurons.json", "w"))
print(f"done in {time.time()-t0:.0f}s")
