# Notes

## Model
Leaky integrate-and-fire (LIF), with the equations and constants from Shiu et al. 2024 (Nature),
"A Drosophila computational brain model reveals sensorimotor processing".
dt 0.1 ms, 1.8 ms synaptic delay. Sensory neurons get Poisson spikes from the world.

Speed: synapses have a 1.8 ms delay, so no spike can reach anyone within 18 steps. `sim.py` runs
18 steps of per-neuron maths as one captured CUDA graph, then delivers all spikes from that block
at once, walking only the outgoing edges of neurons that fired (CSR slices). That's ~1.7x
real time on the 5090 while busy, and 2.5x idle.

## Calibration (2026-09-16), or why the brain had seizures at first
With Shiu's numbers unchanged, the full male CNS locked into self-sustaining loops: ~20k neurons
firing flat out forever after one sip of sugar. Found and fixed in `prep.py` / `sim.py`, in order:
1. **Dopamine/serotonin/octopamine = 0 fast effect.** They're neuromodulators. As excitation they
   formed a Kenyon cell -> dopamine neuron -> Kenyon cell loop.
2. **Drop KC -> KC edges** (643k). These are mostly axo-axonic contacts in the mushroom body,
   and as excitatory synapses they made the KCs a self-exciting blob.
3. **No synaptic input onto sensory neurons** (486k edges). Sensory axons synapse on each other.
   In this model they're driven by the world only.
4. **W_SYN x 0.5.** A sweep over 0.3-0.6x: at 0.5 every reflex works and activity dies back
   down afterwards. At 0.6, sugar sometimes relights the KCs. At 0.3, sugar and bitter stop working.
   Tried and rejected: spike-frequency adaptation (it also tires the inhibitory neurons, which made things worse).

Still there: after strong wind, a small optic lobe loop (Lawf2 / Mi18) hums for a while, and after
looming the central complex heading ring (EPG / Delta7 / PEN) keeps firing. The heading ring
holding activity is actually what it does in real flies. The page logs a warning and offers
Reset if >12k neurons fire with nothing touching the fly.

## Groups (data/neurons.json -> groups)
Sugar = LB3a-d and bitter = LB1a-e. Neither is labelled in the release, so they were picked from
the wiring (3 hops back from MN9: LB3 excites, LB1 inhibits) and confirmed in simulation.

## Ideas for next time
- Olfaction: ORNs are all typed (ORN_VA6 etc). Food smell -> fly steers toward sugar by itself.
- Vision for real: drive R1-R6 from a rendered 1D view of the arena, so turning comes from the optic lobes.
- Male vs female: same terrarium, FlyWire v783 brain. Courtship song should matter more for a female.
- Learning: switch dopamine back on as a plasticity signal on KC -> MBON synapses (odour + bitter = avoid).

## Eyes (2026-09-16)
**Eye map (`prep_eyes.py`).** Janelia put 12 columnar types into ~890 hex columns per eye (`assignedOlHex1/2`).
- Lattice: the offsets that carry the most synapses between columns are (+-1,0), (0,+-1), +-(1,1), so it's a
  120-degree hex grid.
- Up = hex1+hex2, because the dorsal-rim R7d/R8d sit at high values.
- Forward = hex1-hex2, because that points toward the antennal lobe side of the brain (low EM z; KCs and the VNC are high z).
- Photoreceptors take the column of their lamina targets. Everything else takes the synapse-weighted
  mean position of its inputs (4 rounds), which places ~75k visual neurons.

**Why flyvis.** With spiking LIF, the visual system was dead past the photoreceptors. Photoreceptors are
histaminergic (inhibitory), and inhibiting a silent neuron does nothing. A near-threshold bias on
L/Mi/Tm cells made them fire, but nothing moved with the image and T4/T5 stayed silent. Real
lamina/medulla neurons are graded. flyvis (Python <= 3.12, so `.venv-eye`) is an analog model with
trained parameters, and its T4/T5 really are direction selective.

**Bridge (`eyes.py`).**
- Camera frame: the left half goes to the left eye and the right half is mirrored for the right eye.
  flyvis image x = forward, y = down. Checked with bars through the server: rightward motion reads
  back->front in the left eye and front->back in the right eye, and up reads as up in both.
- Every flyvis type that also exists in MaleCNS (49 types, 63k neurons) drives those MaleCNS neurons
  at the nearest hexal. Rate = 90 Hz x (activity - running average over 0.3 s), clipped at 0.
  Using a running average instead of the grey-screen baseline made the bridge carry CHANGE only;
  before that, a still dark disc fired the giant fiber.
- The bridged neurons become inputs: their incoming synapses are cut and the CSR is rebuilt
  (4.8M edges), because otherwise spikes into them cost GPU time and do nothing.
- Results: approaching disc GF 41 Hz / LPLC2 1.4; receding GF 19 / LPLC2 0; still image and small dot 0.
  Receding still drives LC4 and the GF somewhat, which isn't ideal.

**Gotchas hit:**
- datamate (flyvis's storage) deleted an open HDF5 file, which Windows refuses. `setup_eyes.py` patches it.
- `steady_state()` resets the stimulus buffer, so call it BEFORE `stimulus.add_input`.
- torch's default device is per-THREAD, and flyvis relies on it. The HTTP server answers on new threads,
  so `FlyEyes.step` runs inside `with torch.device(...)`.
- flyvis dropped to 0.2-0.4x real time while a game held 60% of the GPU. It's ~0.9x on an idle GPU.
