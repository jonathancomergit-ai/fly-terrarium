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
