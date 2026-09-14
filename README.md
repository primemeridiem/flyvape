# FlyVape

**Watch it:** https://primemeridiem.github.io/flyvape/

This repository started as a copy of fruitflydev/flycoinrh. Its original README is kept as
[README_flycoinrh.md](README_flycoinrh.md).

A whole fruit fly connectome on a tethered nicotine rig. The fly smells an
e-liquid, gets nicotine and a dopamine reward on every puff, and keeps going
until its neurons die. You watch all 165,122 neurons fire and die in 3D.

Built on [fruitflydev/flycoinrh](https://github.com/fruitflydev/flycoinrh), the
leaky integrate-and-fire simulation of the FlyEM male CNS connectome. Every
file from that repository is here unchanged. The new files are `vape.py`,
`vape_rig.py`, `vape_server.py`, `web/vape3d.html`, `test_vape.py`,
`fetch_data.sh`, `tools/` and `docs/`, which is the GitHub Pages site.

No animal is involved. It is a simulation.

## Run it

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python numpy==2.4.2 scipy==1.17.1 pandas==3.0.1 \
    pyarrow==23.0.1 fastapi==0.128.0 "uvicorn[standard]==0.40.0" pytest

./fetch_data.sh                      # ~565 MB from the public FlyEM bucket
.venv/bin/python build_graph.py      # -> build/graph.npz, 165,122 neurons, 10,228,000 edges
.venv/bin/python -m pytest -q test_vape.py

.venv/bin/python vape_server.py      # live 3D at http://localhost:4670, records the run
.venv/bin/python vape_rig.py         # the same run without a browser
```

When a run ends, its replay is written to `build/vape_replay/`. To watch it
without the rig, serve `web/vape3d.html` as `index.html` with that folder next
to it as `replay/`.

Every setting in `vape_rig.Config` is a command-line flag, for example
`--toxicity 1 --escalate 0.1 --dose 0`. A dose of 0 is the sham control.

The README's connectome URLs are out of date. `fetch_data.sh` uses the live
`flat-connectome/` path and the traced-only weights table (508 MB instead of
1.05 GB), which gives the same graph.

## What happens in a run

1. **Baseline.** Four drug-free rig cycles. The first is warm-up. The other
   three record each neuron's highest smoothed firing rate, which becomes its
   safe ceiling.
2. **Rig cycle, repeated.** Air for 8 windows, a puff for 4, wash-out for 12.
   One window is 24 ms of brain time. During the puff, the e-liquid smell
   drives the olfactory receptor neurons and nicotine is dosed. At the end of
   the puff, reward dopamine reaches the mushroom body.
3. **Damage.** A neuron that fires above its ceiling while nicotine is bound
   at its receptors takes damage. At damage 1 it dies. Its synapses are
   removed and it never fires again.
4. **Sniff test.** Every 5 puffs, the fly smells the e-liquid and a control
   odour from a fixed rest state. The mushroom body's output synapses say
   whether the e-liquid leans toward approach.
5. **Stop.** The run stops when all 10 walking descending neurons are dead,
   half the brain is dead, or brain activity collapses. It also stops at a
   window limit.

## Measured and chosen

**Measured**
- The connectome: every neuron, synapse and transmitter sign.
- Soma positions, which the 3D view uses.
- Fly receptor responses to the vape's compounds, from DoOR 2.0.
- Which dopamine neurons reach which mushroom-body outputs.

**Chosen**
- The nicotine model: extra drive in proportion to a neuron's acetylcholine
  inputs.
- The damage rule and its constants.
- The e-liquid recipe.
- Reward dopamine on every puff.
- The dose rising 25% every 10 puffs.
- Compressed time.
- The 3D rig. The fly's body, tether, air-supported ball and box-mod vape are
  drawn around the measured neurons. The ball turns with the walking neurons'
  output, and the vape's screen counts puffs.

The docstrings in `vape.py` and `vape_rig.py` say which is which, line by line.

## What calibration showed

These checks are why the model looks the way it does:

- **A fly barely smells nicotine.** In DoOR 2.0, nicotine drives one receptor
  (Or19a) at 0.05 above spontaneous firing. The smell of a vape, to a fly, is
  its flavour (ethyl vanillin) and acetaldehyde from heated propylene glycol.
- **Judging death window by window kills neurons with no drug at all.** A sham
  run with zero nicotine killed 1,402 neurons in 150 windows. The deaths
  disinhibited their targets and cascaded. Smoothing the rate over 20 windows
  was not enough either, since the simulated brain wanders and some quiet
  cells switch on by themselves. So damage is gated on nicotine at the
  neuron's receptors, and a sham run cannot kill anything.
- **Without tolerance, dying plateaus.** At a fixed dose, the nicotine level
  settles near 2.8 and neurons die at a steady 1 to 2 per window. At that
  rate a run never reaches a stop condition, so the dose escalates.
- **The learning readout is noise.** Across 60 puffs at three toxicities, the
  sniff test swung between about -0.08 and +0.06 with no trend, while 23,993
  of 44,042 KC-to-MBON synapses were weakened. Learning in this simulator is
  partly global (see `calibration.py`), and dying Kenyon cells confound the
  readout. The page shows the numbers as they are.

## The recorded run

Run on 2026-09-14 with the defaults: toxicity 2, dose rising 25% every 10
puffs, seed 7.

- **Replay online:** https://claude.ai/code/artifact/5f3eb638-7a3a-427c-85c4-81bc67e36972
- **Archive:** `build/run_2026-09-14_half/`. It holds the raw recording
  (`vape_run/`), the local replay (`vape_replay/`) and the log.
- **Web copy:** `vape_replay_web/` is the copy that was published. The
  artifact host does not serve raw binary files, so every `.bin` is stored as
  base64 text (`.b64.txt`). `replay.json` sets `index.encoding` to `base64` so
  the page knows which to load.
- **Overwrite warning:** a new run deletes and rewrites `build/vape_run/` and
  `build/vape_replay/`. Keep the archive.

| | |
|---|---|
| stopped because | half the brain was dead |
| puffs | 192 |
| windows | 4,692 (112.6 s of brain time, 20 min on an M4 CPU) |
| first death | after 3 puffs |
| neurons dead | 82,575 of 165,122 |
| Kenyon cells dead | 3,956 of 4,064 |
| walking neurons alive at the end | 2 of 10 (both DNa02 steering cells) |
| final dose per puff | 24.3, up from 0.35 |

Dead by class:

| class | dead | of |
|---|---|---|
| optic lobe intrinsic | 45,061 | 89,390 |
| central brain intrinsic | 19,676 | 32,160 |
| visual projection | 7,466 | 9,201 |
| nerve cord intrinsic | 6,411 | 13,151 |
| ascending | 1,134 | 1,846 |
| descending | 820 | 1,314 |
| nerve cord motor | 379 | 708 |

**The fly did not learn to want the vape.** The e-liquid and the control
smell moved together, from a leaning of -0.22 before the first puff to +0.88
by puff 90. The gap between them stayed inside ±0.07 the whole time. That
drift is the avoidance side of the mushroom body dying, not a preference.
From about puff 120, the few Kenyon cells still firing had no output synapses
left, and the sniff test returned no reading.

## Credits

- **Connectome:** FlyEM male CNS v1.0, © HHMI Janelia FlyEM, Cambridge
  Connectomics Group and Google Research. CC-BY.
- **Odour responses:** DoOR 2.0, Münch & Galizia 2016, *Scientific Reports*
  6, 21841. CC BY-SA 4.0.
- **Simulator, mushroom body and plume code:** fruitflydev/flycoinrh. MIT.
- **Fruit fly body:** NeuroMechFly v2 by the Ramdya lab at EPFL, from
  flygym (github.com/NeLy-EPFL/flygym), Apache-2.0. Its licence is kept as
  `build/models/fly_nmf_LICENSE.txt`. `tools/prepare_nmf.py` converts the
  simplified meshes, rigging and neutral pose into `build/models/fly_nmf.glb`.
  That file has 69 segments and 66,342 triangles, with a node for every leg
  segment. The page places the fly by its eyes over the brain's optic lobes.
  It then presses the AIR, PUFF and WASH buttons by bending each leg's
  coxa-femur joint. Its wings beat during puffs, slow through the wash and fold
  in clean air. How fast they beat follows dopamine: a burst after each reward
  that fades over about a second, on top of a rate that climbs from 5 to about
  18 strokes a second as rewards add up. The on-screen rate is slowed down to
  be visible, and the simulated neurons do not drive the wings. flygym's joint axes are pitch about y, roll about z and yaw
  about x. Using the textbook axes instead puts the hind feet above the body.
  Without the file, the page falls back to the drawn fly.

The fly stands on a three-button pad, not a ball. Its front legs sit on AIR, its
middle legs on PUFF and its hind legs on WASH. The pair over the current rig
phase presses its button down and lights it. This shows the rig's schedule:
the simulated fly does not choose when to vape.

The rig is drawn at real sizes, taking the fly as 3 mm with its wings folded:
- a 6 mm foam ball, used for scale though no longer drawn
- an 8 mm tether pin
- a nozzle 1.5 mm from the face
- a 3 mm tube
- a 130 mm box-mod vape

The page has no view, size or colour menus. It always shows a crimson box mod
0.8 times the fly's length, standing beside it, from the side. The page uses Big Shoulders Display for the title and big numbers, Public Sans for text and Martian Mono for readouts.

The rig is sized from the drawn fly, which is taken as 3 mm from the front of
the head to the tip of the abdomen. The fly is then drawn 2.5 times larger
than life against the rig (`FLY_SCALE` in the page) so it is easy to see. The
camera opens on a side view across the rig and does not auto-rotate.
