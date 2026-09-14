"""
The vape: what a tethered fly smells in it, what nicotine does to the brain,
and how a neuron dies of too much excitation.

Nothing in the rest of this project models a drug or a death. Everything in
this file that is not a lookup into the connectome or DoOR is a modelling
choice, and it is labelled CHOSEN.

MEASURED
  * The connectome: every neuron, every synapse, and which ones are
    cholinergic (build_graph.py gives acetylcholine sign +1 and every other
    transmitter a sign <= 0, so the positive weights ARE the ACh synapses).
  * How fly olfactory receptors respond to the compounds in an e-liquid and
    its vapour: DoOR 2.0 (Muench and Galizia 2016, CC BY-SA 4.0), via
    olfaction.Door.profile. The finding that shapes this file: nicotine
    itself barely registers (Or19a, 0.05 above spontaneous firing). The
    carriers propylene glycol and glycerol are near zero too. What a fly nose
    does pick up is the flavour (ethyl vanillin, Or71a 0.36) and acetaldehyde,
    which heated propylene glycol gives off (30+ receptors).
  * Soma positions for the 3D view (body-annotations somaLocation), for
    140,024 of the 165,122 neurons.

CHOSEN
  * The e-liquid recipe weights, ELIQUID below.
  * Nicotine as extra Poisson drive onto each neuron in proportion to how many
    cholinergic synapses it receives. Nicotine is an agonist at nicotinic
    acetylcholine receptors, and acetylcholine is the fly brain's main fast
    excitatory transmitter, so incoming ACh synapse count stands in for
    receptor density. Real receptor density is not in the connectome.
  * Nicotine kinetics: a dose per puff and an exponential clearance with a
    half-life in rig windows.
  * Excitotoxicity: each neuron accumulates damage while it fires above its
    own pre-drug ceiling, slowly repairs, and dies at damage 1. A real neuron
    dies of calcium overload through mechanisms this simulator does not have.
  * Death: a dead neuron's outgoing synapses are zeroed and it never fires
    again. Nothing regrows.
  * Neurons with no soma in the imaged volume (sensory cells whose bodies sit
    in the antenna, the eye or the legs) are drawn at the mean soma position
    of the neurons they synapse onto. Their dots are placed, not measured.
"""
import json
from pathlib import Path

import numpy as np

import olfaction

ROOT = Path(__file__).parent
ANNOTATIONS = ROOT / "data" / "body-annotations.feather"

# DoOR odorant name -> weight in the smell of one puff. CHOSEN. The names are
# DoOR's own (door/odor.csv). Nicotine is in the list so nobody has to wonder
# whether it was left out: its receptor response is almost nothing.
ELIQUID = {
    "ethyl vanillin": 1.0,       # vanilla flavour
    "acetaldehyde": 0.6,         # thermal breakdown of propylene glycol
    "propylene glycol": 1.0,     # carrier
    "glycerol": 1.0,             # carrier
    "menthol": 0.5,              # cooling flavour
    "l-(-)-nicotine": 1.0,       # the drug; nearly odourless to a fly
}
# The control odour for the learning test, the same one plume_learning_design.md uses.
CONTROL_ODOUR = "2,3-butanedione"

BIG_REFR = np.int32(2 ** 30)     # refractory steps that outlast any run (59 h of brain time)


# ---------------------------------------------------------------------------
# smell
# ---------------------------------------------------------------------------
def blend_profile(door, recipe):
    """Per-glomerulus response of a mixture: the strongest weighted response wins (olfaction.py's rule)."""
    out, missing = {}, []
    for name, weight in recipe.items():
        key = door.key_of.get(name.lower())
        if key is None:
            missing.append(name)
            continue
        for g, v in door.profile(key).items():
            out[g] = max(out.get(g, 0.0), float(v) * float(weight))
    if missing:
        raise KeyError(f"not in DoOR: {missing}")
    return {g: v for g, v in out.items() if v >= olfaction.MIN_RESPONSE}


def odour_drive(nose, profile, concentration=1.0):
    """A FlyBrain drive dict for a profile at a concentration in [0, 1]."""
    c = float(np.clip(concentration, 0.0, 1.0))
    return nose.drive({"profile": {g: v * c for g, v in profile.items()}})


# ---------------------------------------------------------------------------
# nicotine
# ---------------------------------------------------------------------------
def ach_input(fb):
    """
    Incoming cholinergic synapse count per neuron. MEASURED.

    W's column j lists neuron j's targets; a positive weight is an ACh
    synapse of n contacts at 0.275 mV each.
    """
    w = np.asarray(fb.wdata, dtype=np.float64)
    pos = w > 0
    tgt = np.asarray(fb.indices)[pos]
    return np.bincount(tgt, weights=w[pos] / 0.275, minlength=fb.n)


class Nicotine:
    """
    Body level of nicotine and the drive it puts on the brain. CHOSEN throughout.

    level      dimensionless; one full puff adds `dose`
    half_life  in rig windows
    max_hz     extra Poisson rate at level 1 on the most ACh-innervated neurons
    """

    def __init__(self, fb, dose=0.35, half_life=90.0, max_hz=12.0, pct=99.0):
        self.dose = float(dose)
        self.half_life = float(half_life)
        self.max_hz = float(max_hz)
        self.level = 0.0
        self.total = 0.0
        ach = ach_input(fb)
        top = np.percentile(ach[ach > 0], pct) if (ach > 0).any() else 1.0
        self.receptor = np.clip(ach / max(top, 1e-9), 0.0, 1.0).astype(np.float32)
        self.idx = np.flatnonzero(self.receptor > 0)
        self._key = tuple(self.idx.tolist())

    def puff(self, fraction=1.0):
        amount = self.dose * float(fraction)
        self.level += amount
        self.total += amount
        return amount

    def tick(self, windows=1.0):
        if self.half_life > 0:
            self.level *= 0.5 ** (float(windows) / self.half_life)

    def drive(self):
        """{neurons: per-neuron Hz}, or {} when there is no nicotine to speak of."""
        if self.level < 1e-4 or not len(self.idx):
            return {}
        return {self._key: self.receptor[self.idx] * (self.level * self.max_hz)}


# ---------------------------------------------------------------------------
# excitotoxic death
# ---------------------------------------------------------------------------
class Excitotoxicity:
    """
    Damage accrues while a neuron's sustained rate sits above its own ceiling,
    and at 1 it dies. CHOSEN.

    The rate that counts is smoothed over windows (an exponential average with
    time constant `tau` windows). One 24 ms window holds only a few spikes per
    neuron, so its rate is quantised at 41.7 Hz per spike and jumps with every
    chance spike. MEASURED 2026-09-14 on this brain: judged per window, a
    sham run with no nicotine at all killed 1,402 neurons in 150 windows, and
    those deaths disinhibited their targets into a cascade. Excitotoxicity is
    a sustained load, so the smoothed rate is the honest input.

    ceiling_i = baseline_max_i * margin + floor_hz, where baseline_max_i is the
    highest smoothed rate neuron i reached during the drug-free baseline
    (after a warm-up that is not counted, since every run starts from rest).
    A busy neuron's normal busyness is safe; the cost falls on what the drug
    adds.

    The load is gated by nicotine at the neuron's own receptors: over-firing
    only damages a cell while nicotine is bound there (exposure = body level x
    receptor proxy). Nicotinic receptors pass calcium, and calcium entry is
    the route by which nicotine is thought to push neurons toward
    excitotoxicity. MEASURED 2026-09-14: without the gate, even with smoothing,
    a 6-cycle baseline and a 1.5x / +30 Hz ceiling, a sham run with no
    nicotine killed 93 neurons in 266 windows, because this simulated brain
    wanders: some cells that were quiet all through the baseline switch on
    later by themselves. With the gate, no nicotine means no death, and a
    neuron that wanders while nicotine is on board is the one that pays.

    damage += kappa * max(0, smoothed - ceiling) * exposure - repair, per window, floored at 0.
    """

    def __init__(self, n, kappa=2e-4, repair=5e-4, margin=1.25, floor_hz=8.0, tau=20.0):
        self.n = int(n)
        self.kappa = float(kappa)
        self.repair = float(repair)
        self.margin = float(margin)
        self.floor_hz = float(floor_hz)
        self.alpha = 1.0 / max(1.0, float(tau))
        self.ema = None
        self.baseline_max = np.zeros(self.n, dtype=np.float32)
        self.baseline_windows = 0
        self.damage = np.zeros(self.n, dtype=np.float32)
        self.dead = np.zeros(self.n, dtype=bool)
        self.death_window = np.full(self.n, -1, dtype=np.int32)
        self.excess = np.zeros(self.n, dtype=np.float32)

    def smooth(self, rate_hz):
        r = np.asarray(rate_hz, dtype=np.float32)
        if self.ema is None:
            self.ema = r.copy()
        else:
            self.ema += self.alpha * (r - self.ema)
        return self.ema

    def observe_baseline(self, rate_hz, warm=False):
        ema = self.smooth(rate_hz)
        if not warm:
            np.maximum(self.baseline_max, ema, out=self.baseline_max)
            self.baseline_windows += 1

    @property
    def ceiling(self):
        return self.baseline_max * self.margin + self.floor_hz

    def update(self, rate_hz, window, exposure=None):
        """
        Accrue damage for one window and return the indices that died in it.

        exposure: per-neuron nicotine at the receptors (level x receptor proxy).
        None counts over-firing alone, which is only for tests of the rule.
        """
        ema = self.smooth(rate_hz)
        excess = np.maximum(0.0, ema - self.ceiling)
        excess[self.dead] = 0.0
        self.excess = excess
        load = excess if exposure is None else excess * np.asarray(exposure, dtype=np.float32)
        self.damage += self.kappa * load
        self.damage -= self.repair
        np.maximum(self.damage, 0.0, out=self.damage)
        self.damage[self.dead] = 1.0
        new = np.flatnonzero((self.damage >= 1.0) & ~self.dead)
        if len(new):
            self.dead[new] = True
            self.death_window[new] = int(window)
        return new


def rest_state(n, seed, v_rest=-52.0):
    """A '_state' dict for FlyBrain.run: every neuron at rest, no refractory time, a seeded rng."""
    rng = np.random.default_rng(seed)
    return {"v": np.full(n, v_rest, dtype=np.float32),
            "refr": np.zeros(n, dtype=np.int32),
            "rng": rng.bit_generator.state}


def silence(state, dead_mask):
    """Hold every dead neuron refractory in a run state, so it cannot fire in the next run."""
    if dead_mask.any():
        state["refr"][dead_mask] = BIG_REFR
        state["v"][dead_mask] = -52.0
    return state


def kill(fb, idx, mb=None):
    """
    Remove neurons from the circuit: zero every outgoing synapse.

    The mushroom body keeps its own copy of the KC->MBON baseline weights and
    writes base * gain back into the simulation on every apply(), so a dead
    Kenyon cell would come back to life through it. Its base is zeroed too, and
    so is the base of every synapse onto a dead MBON, so the synaptic readout
    (calibration.syn_drive) stops counting output a dead cell cannot deliver.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if not len(idx):
        return 0
    indptr = np.asarray(fb.indptr)
    zeroed = 0
    for i in idx:
        a, b = int(indptr[i]), int(indptr[i + 1])
        if b > a:
            fb.wdata[a:b] = 0.0
            zeroed += b - a
    if mb is not None and len(mb.pos):
        gone = np.isin(mb.pre, idx) | np.isin(mb.post, idx)
        if gone.any():
            mb.base[gone] = 0.0
            fb.wdata[mb.pos[gone]] = 0.0
    return zeroed


# ---------------------------------------------------------------------------
# anatomy for the 3D view
# ---------------------------------------------------------------------------
def _xyz(v):
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
        return float(v[0]), float(v[1]), float(v[2])
    return None


def anatomy(fb, annotations=ANNOTATIONS):
    """
    Neuron positions and labels for drawing.

    returns dict:
      xyz        (n, 3) float32, centred and scaled to about [-1, 1]
      placed     (n,) uint8: 0 soma measured, 1 toward-soma point measured,
                 2 placed at partner centroid, 3 no position (hidden)
      superclass list of names; sc (n,) uint8 codes into it
      extent     scale used, in the annotation's voxel units
    """
    import pandas as pd
    a = pd.read_feather(annotations).drop_duplicates("bodyId").set_index("bodyId")
    n = fb.n
    xyz = np.full((n, 3), np.nan, dtype=np.float64)
    placed = np.full(n, 3, dtype=np.uint8)
    for col, code in (("somaLocation", 0), ("tosomaLocation", 1)):
        if col not in a.columns:
            continue
        vals = a[col].reindex(fb.bodies).to_numpy()
        for i, v in enumerate(vals):
            if placed[i] != 3:
                continue
            p = _xyz(v)
            if p is not None:
                xyz[i] = p
                placed[i] = code

    # Sensory neurons have no soma inside the volume. Put each at the mean
    # position of the neurons it synapses onto (every sign, weighted by
    # contacts); two passes reach a cell whose partners were placed this way.
    import scipy.sparse as sp
    W = fb.W if hasattr(fb, "W") else sp.csc_matrix((fb.wdata, fb.indices, fb.indptr), shape=(n, n))
    A = abs(W).tocsc()                         # column j = targets of j
    for _ in range(2):
        have = ~np.isnan(xyz[:, 0])
        need = np.flatnonzero(~have)
        if not len(need):
            break
        known = np.where(have[:, None], xyz, 0.0)
        wsum = A.T @ have.astype(np.float64)   # per presynaptic neuron: weight onto placed cells
        pos = A.T @ known                      # weighted sum of their positions
        ok = need[wsum[need] > 0]
        xyz[ok] = pos[ok] / wsum[ok, None]
        placed[ok] = 2

    have = ~np.isnan(xyz[:, 0])
    centre = np.nanmedian(xyz[have], axis=0) if have.any() else np.zeros(3)
    span = np.nanmax(np.abs(xyz[have] - centre)) if have.any() else 1.0
    out = ((xyz - centre) / span).astype(np.float32)
    names, sc = np.unique(np.asarray(fb.superclass).astype(str), return_inverse=True)
    return {"xyz": out, "placed": placed, "superclass": [str(s) or "unlabelled" for s in names],
            "sc": sc.astype(np.uint8), "extent": float(span), "centre": [float(c) for c in centre]}


def groups(fb, motor=None, mb=None):
    """Index groups the page can highlight."""
    g = {
        "KC": fb.where(type_re=r"^KC"),
        "MBON": fb.where(type_re=r"^MBON"),
        "ORN": fb.where(type_re=r"^ORN_"),
        "JO": fb.where(type_re=r"^JO-"),
        "DN": fb.where(superclass="descending_neuron"),
        "DAN": fb.where(type_re=r"^(PAM|PPL1)"),
    }
    labels = []
    if motor is not None:
        names = {"steer_L": "DNa02 L", "steer_R": "DNa02 R", "fwd_L": "DNa01 L", "fwd_R": "DNa01 R",
                 "back": "MDN", "stop": "DNp09"}
        g["walk"] = np.concatenate([np.asarray(v, dtype=np.int64) for v in motor.values()])
        labels = [names.get(k, k) for k, v in motor.items() for _ in range(len(v))]
    out = {k: [int(i) for i in np.asarray(v)] for k, v in g.items()}
    out["walk_labels"] = labels
    return out


def write_anatomy(fb, out_dir, motor=None):
    """neurons.bin (float32 xyz), neurons_meta.bin (uint8 superclass, uint8 placed), neurons.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    an = anatomy(fb)
    xyz = np.nan_to_num(an["xyz"], nan=0.0).astype("<f4")
    (out / "neurons.bin").write_bytes(xyz.tobytes())
    (out / "neurons_meta.bin").write_bytes(np.stack([an["sc"], an["placed"]], 1).astype(np.uint8).tobytes())
    meta = {"n": int(fb.n), "superclass": an["superclass"], "groups": groups(fb, motor),
            "placed_counts": {k: int((an["placed"] == v).sum()) for k, v in
                              (("soma", 0), ("tosoma", 1), ("partner_centroid", 2), ("hidden", 3))}}
    (out / "neurons.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta
