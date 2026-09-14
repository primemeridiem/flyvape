"""
The tethered vape rig: a whole connectome fly, held still, puffed on a
schedule, rewarded with dopamine at the end of every puff, dosed with nicotine
until its neurons start to die.

One rig cycle, repeated:
  air   background drive only (clean air on the antenna, receptor neurons idling)
  puff  the e-liquid smell ramps on and nicotine is dosed; at the last window
        of the puff, reward dopamine reaches the mushroom body while the
        Kenyon cells that smelled the puff are still eligible
  wash  background again while nicotine clears

Before the first puff the rig runs drug-free baseline cycles (same smell, no
nicotine, no dopamine) and records every neuron's highest rate. That is the
ceiling excitotoxic damage is measured against (vape.Excitotoxicity).

Every few puffs a sniff test asks what the mushroom body has learned: from a
fixed rest state and seed, one sniff of the e-liquid and one of a control
odour, no nicotine, no reward, read off the KC->MBON output synapses exactly
the way the backroom reads a look (calibration.syn_drive, calibration.leaning).
If learning is working, the e-liquid leans toward approach more than the
control does, and more after training than before.

The run stops at the first of:
  motor   every walking descending neuron is dead (the fly can no longer walk)
  half    half of all neurons are dead
  silent  whole-brain spike rate stays under 5% of baseline
  limit   --max-windows

MEASURED and CHOSEN are listed in vape.py. Additionally CHOSEN here: the
dose rising 25% every 10 puffs as tolerance builds (MEASURED 2026-09-14:
without escalation the level plateaus near 2.8 and neurons die at a steady
1-2 per window, so a run never reaches a stop condition), the cycle lengths, one rig window counting as one rig second for nicotine
clearance, damage and memory decay (brain time per window is 24 ms, so time
is compressed), the background rates, the reward amount, and pairing every
puff with reward dopamine.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RUN_DIR = ROOT / "build" / "vape_run"
REPLAY_DIR = ROOT / "build" / "vape_replay"


@dataclass
class Config:
    seed: int = 7
    sim_steps: int = 120            # 120 x 0.2 ms = 24 ms of brain per window
    air: int = 8                    # windows per phase
    puff: int = 4
    wash: int = 12
    dose: float = 0.35              # nicotine level added per puff
    half_life: float = 90.0         # nicotine clearance, rig windows
    nic_max_hz: float = 12.0        # extra drive at level 1 on the most ACh-innervated cells
    toxicity: float = 2.0           # multiplies kappa
    kappa: float = 1e-4             # damage per (Hz above ceiling x nicotine exposure), per window
    repair: float = 5e-4            # damage repaired per window
    margin: float = 1.5             # ceiling = baseline max (smoothed) * margin + floor
    floor_hz: float = 20.0
    escalate: float = 0.25          # tolerance: the dose grows by this fraction ...
    escalate_every: int = 10        # ... every this many puffs
    tau: float = 20.0               # windows the damage rate is smoothed over
    orn_hz: float = 6.0             # background, every receptor neuron
    jo_hz: float = 30.0             # background, JO-E wind cells (the tether's airflow)
    odour_hz: float = 200.0         # receptor ceiling (Hallem and Carlson 2006)
    baseline_cycles: int = 4        # the first is warm-up and does not set the ceiling
    test_every: int = 5             # puffs between sniff tests
    test_windows: int = 2
    reward: float = 1.0
    mb_half_life_h: float = 1.0     # memory half-life in rig hours (3,600 windows)
    stop_dead_frac: float = 0.5
    collapse_frac: float = 0.05
    collapse_windows: int = 25
    max_windows: int = 9000
    risk_every: int = 4             # windows between damage snapshots on the stream
    damage_every: int = 10          # windows between damage snapshots on disk


def merge(*drives):
    """Merge FlyBrain drive dicts. A key in two of them gets both inputs (two Poisson chances)."""
    out = {}
    for d in drives:
        for k, v in d.items():
            if k in out:
                raise ValueError("drive key collision")
            out[k] = v
    return out


class Recorder:
    """Every window to disk, raw: packed spike bits, periodic damage, one JSON line of telemetry."""

    def __init__(self, out_dir, fb, motor, cfg):
        import vape
        self.dir = Path(out_dir)
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.meta = vape.write_anatomy(fb, self.dir, motor)
        (self.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
        self.n = fb.n
        self.nbytes = (fb.n + 7) // 8
        self.act = open(self.dir / "activity.raw", "wb")
        self.dmg = open(self.dir / "damage.raw", "wb")
        self.tl = open(self.dir / "timeline.jsonl", "w", encoding="utf-8")
        self.windows = 0
        self.damage_windows = []

    def add(self, header, fired, damage, damage_every):
        mask = np.zeros(self.n, dtype=bool)
        mask[fired] = True
        self.act.write(np.packbits(mask).tobytes())
        if header["w"] % damage_every == 0:
            self.dmg.write(np.clip(damage * 255.0, 0, 255).astype(np.uint8).tobytes())
            self.damage_windows.append(int(header["w"]))
        self.tl.write(json.dumps(header, separators=(",", ":")) + "\n")
        self.windows += 1

    def finish(self, tox, summary):
        for f in (self.act, self.dmg, self.tl):
            f.close()
        (self.dir / "death_window.bin").write_bytes(tox.death_window.astype("<i4").tobytes())
        (self.dir / "damage_windows.json").write_text(json.dumps(self.damage_windows), encoding="utf-8")
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


class Rig:
    def __init__(self, fb, cfg=None, on_frame=None, record_dir=None, log=print):
        import calibration
        import olfaction
        import plume_fly
        import vape
        from mushroom import MushroomBody

        self.fb = fb
        self.cfg = cfg = cfg or Config()
        self.on_frame = on_frame
        self.log = log or (lambda *a, **k: None)
        self.n = fb.n
        self.secs = cfg.sim_steps * fb.p.dt / 1000.0

        self.gains = calibration.gains_for(fb, calibration.CHOSEN)
        self.door = olfaction.Door(ROOT)
        self.nose = olfaction.Nose(fb, max_hz=cfg.odour_hz, door=self.door)
        self.eliquid = vape.blend_profile(self.door, vape.ELIQUID)
        self.control = vape.blend_profile(self.door, {vape.CONTROL_ODOUR: 1.0})
        self.nic = vape.Nicotine(fb, dose=cfg.dose, half_life=cfg.half_life, max_hz=cfg.nic_max_hz)
        self.tox = vape.Excitotoxicity(fb.n, kappa=cfg.kappa * cfg.toxicity, repair=cfg.repair,
                                       margin=cfg.margin, floor_hz=cfg.floor_hz, tau=cfg.tau)

        self.motor = plume_fly.motor_groups(fb, cfg.sim_steps)
        self.walk = np.concatenate([np.asarray(v, dtype=np.int64) for v in self.motor.values()])
        self.kc = np.asarray(fb.where(type_re=r"^KC"), dtype=np.int64)
        orn = np.asarray(fb.where(type_re=r"^ORN_"), dtype=np.int64)
        jo = np.asarray(fb.where(type_re=plume_fly.JO_DRIVEN_RE), dtype=np.int64)
        # background keys are copies of the index lists as tuples; the odour
        # drive's ORN keys are per glomerulus, so no key is shared
        self.background = {tuple(orn.tolist()): cfg.orn_hz, tuple(jo.tolist()): cfg.jo_hz}
        self.sc_names, self.sc_code = np.unique(np.asarray(fb.superclass).astype(str), return_inverse=True)

        self.window = 0
        self.clock = [0.0]
        self.record_dir = Path(record_dir) if record_dir else None
        store = (self.record_dir or RUN_DIR) / "mb_gains.vape.npz"
        if store.exists():
            store.unlink()                         # a fresh fly every run
        self.mb = MushroomBody(fb, calibration=calibration.CHOSEN, store=store,
                               half_life_h=cfg.mb_half_life_h, clock=lambda: self.clock[0])
        self.mb.apply()
        self.state = vape.rest_state(fb.n, cfg.seed)
        self.puffs = 0
        self.cycle = 0
        self.tests = []
        self.baseline_sps = None
        self.low_run = 0
        self.first_death = None
        self.stop = None
        self.recorder = None
        self.t_start = time.time()
        self.stopping = False                     # set True from outside to end early

    # ---- one window ------------------------------------------------------
    def run_window(self, drive, phase, learn, damage, warm=False):
        import vape
        cfg = self.cfg
        t0 = time.time()
        vape.silence(self.state, self.tox.dead)
        r = self.fb.run(drive, steps=cfg.sim_steps, gains=self.gains, seed=0,
                        spike_log=True, state=self.state)
        self.state = r["_state"]
        log = [s for s in r["_spikes"] if len(s)]
        counts = (np.bincount(np.concatenate(log), minlength=self.n) if log
                  else np.zeros(self.n, dtype=np.int64))
        rate = counts / self.secs
        fired = np.flatnonzero(counts)

        new_dead = np.array([], dtype=np.int64)
        if damage:
            exposure = self.nic.receptor * np.float32(self.nic.level)
            new_dead = self.tox.update(rate, self.window, exposure)
            if len(new_dead):
                vape.kill(self.fb, new_dead, self.mb)
                if self.first_death is None:
                    self.first_death = {"window": self.window, "puffs": self.puffs,
                                        "nicotine_total": round(self.nic.total, 3)}
        else:
            self.tox.observe_baseline(rate, warm=warm)

        if learn:
            self.clock[0] = float(self.window)
            self.mb.observe(fired)
            self.mb.forget()

        import plume_fly
        hz = {k: float(rate[np.asarray(v, dtype=np.int64)].mean()) if len(v) else 0.0
              for k, v in self.motor.items()}
        turn, speed, _ = plume_fly.PlumeFly.motor_from_rates(hz)
        dead_n = int(self.tox.dead.sum())
        header = {
            "w": self.window, "phase": phase, "cycle": self.cycle, "puffs": self.puffs,
            "nic": round(self.nic.level, 4), "nic_total": round(self.nic.total, 3),
            "dose": round(self.nic.dose, 4),
            "fired": int(len(fired)), "sps": round(float(r["_spikes_per_sec"]), 1),
            "mv": round(float(r["_mean_mv"]), 3),
            "turn": round(float(np.clip(turn, -1, 1)), 4), "speed": round(float(speed), 4),
            "motor": {k: round(v, 1) for k, v in hz.items()},
            "walk_alive": int((~self.tox.dead[self.walk]).sum()), "walk_total": int(len(self.walk)),
            "dead": dead_n, "dead_frac": round(dead_n / self.n, 5), "new_dead": int(len(new_dead)),
            "dmg_max": round(float(self.tox.damage[~self.tox.dead].max()) if dead_n < self.n else 1.0, 4),
            "at_risk": int(((self.tox.damage >= 0.5) & ~self.tox.dead).sum()),
            "over": int((self.tox.excess > 0).sum()) if damage else 0,
            "excess_p99": round(float(np.percentile(self.tox.excess, 99.9)), 2) if damage else 0.0,
            "mb": {k: self.mb.stats().get(k) for k in ("depressed", "mean_gain", "min_gain", "rewards")},
            "sec": round(time.time() - t0, 3),
            "baseline_sps": self.baseline_sps,
            "first_death": self.first_death,
        }
        return header, fired, new_dead

    def emit(self, header, fired, new_dead):
        cfg = self.cfg
        if len(new_dead):
            counts = np.bincount(self.sc_code[new_dead], minlength=len(self.sc_names))
            header["new_dead_by_sc"] = {str(self.sc_names[i]) or "unlabelled": int(c)
                                        for i, c in enumerate(counts) if c}
        risk_idx = risk_lvl = None
        if self.window % cfg.risk_every == 0:
            d = self.tox.damage
            risk_idx = np.flatnonzero((d >= 0.05) & ~self.tox.dead).astype(np.uint32)
            risk_lvl = np.clip(d[risk_idx] * 255, 0, 255).astype(np.uint8)
        if self.recorder is not None:
            self.recorder.add(header, fired, self.tox.damage, cfg.damage_every)
        if self.on_frame is not None:
            self.on_frame(header, fired.astype(np.uint32), new_dead.astype(np.uint32), risk_idx, risk_lvl)
        if self.window % 10 == 0 or len(new_dead) or header.get("dopamine") is not None:
            self.log(f"w{header['w']:>6} {header['phase']:<9} puffs {header['puffs']:>4}  "
                     f"nic {header['nic']:.3f}  fired {header['fired']:>6}  sps {header['sps']:>10.0f}  "
                     f"dead {header['dead']:>6} (+{header['new_dead']})  dmg {header['dmg_max']:.3f}  "
                     f"over {header.get('over', 0):>6} x99.9 {header.get('excess_p99', 0):>6.1f}  "
                     f"walk {header['walk_alive']}/{header['walk_total']}  {header['sec']:.2f}s")
        self.window += 1

    # ---- the learning test ----------------------------------------------
    def sniff_test(self):
        import calibration
        import vape
        cfg = self.cfg
        out = {"window": self.window, "puffs": self.puffs, "dead": int(self.tox.dead.sum())}
        for name, prof in (("eliquid", self.eliquid), ("control", self.control)):
            st = vape.rest_state(self.n, cfg.seed + 9001)
            vape.silence(st, self.tox.dead)
            drive = merge(self.background, vape.odour_drive(self.nose, prof, 1.0))
            fired = []
            for _ in range(cfg.test_windows):
                vape.silence(st, self.tox.dead)
                r = self.fb.run(drive, steps=cfg.sim_steps, gains=self.gains, seed=0, state=st)
                st = r["_state"]
                fired.append(r["_fired"])
            fired = np.unique(np.concatenate(fired)) if fired else np.array([], dtype=np.int64)
            a, v = calibration.syn_drive(self.mb, fired)
            out[name] = {"approach": round(a, 3), "avoid": round(v, 3),
                         "leaning": round(calibration.leaning((a, v)), 5),
                         "kc_fired": int(np.isin(self.kc, fired).sum())}
        # an empty reading (no fired Kenyon cell has an output synapse left) is
        # not a score of 0; calibration.has_reading says the same for the room
        readable = all(calibration.has_reading((out[k]["approach"], out[k]["avoid"]))
                       for k in ("eliquid", "control"))
        out["readable"] = readable
        out["relative"] = round(out["eliquid"]["leaning"] - out["control"]["leaning"], 5) if readable else None
        self.tests.append(out)
        self.log(f"   sniff test @ {self.puffs} puffs: e-liquid leaning {out['eliquid']['leaning']:+.4f}  "
                 f"control {out['control']['leaning']:+.4f}  relative {out['relative'] if out['relative'] is not None else 'n/a'}  "
                 f"KCs {out['eliquid']['kc_fired']}/{out['control']['kc_fired']}")
        return out

    # ---- the stop rule -----------------------------------------------------
    def check_stop(self, header):
        cfg = self.cfg
        if len(self.walk) and self.tox.dead[self.walk].all():
            return "motor"
        if header["dead_frac"] >= cfg.stop_dead_frac:
            return "half"
        if self.baseline_sps:
            self.low_run = self.low_run + 1 if header["sps"] < cfg.collapse_frac * self.baseline_sps else 0
            if self.low_run >= cfg.collapse_windows:
                return "silent"
        if self.window >= cfg.max_windows:
            return "limit"
        if self.stopping:
            return "stopped"
        return None

    # ---- the session ------------------------------------------------------
    def cycle_windows(self, baseline):
        """One air, puff, wash cycle; yields (phase, drive, is_last_puff_window)."""
        import vape
        cfg = self.cfg
        tag = "base-" if baseline else ""
        for _ in range(cfg.air):
            yield tag + "air", False
        for k in range(cfg.puff):
            yield tag + "puff", k
        for _ in range(cfg.wash):
            yield tag + "wash", False

    def drive_for(self, phase, k, baseline):
        import vape
        cfg = self.cfg
        drives = [self.background]
        if phase.endswith("puff"):
            conc = min(1.0, (k + 1) / 2.0)          # two-window ramp
            drives.append(vape.odour_drive(self.nose, self.eliquid, conc))
            if not baseline:
                if k == 0:
                    self.puffs += 1
                    if cfg.escalate and self.puffs % cfg.escalate_every == 0:
                        self.nic.dose *= 1.0 + cfg.escalate
                self.nic.puff(1.0 / cfg.puff)
        if not baseline:
            drives.append(self.nic.drive())
        return merge(*drives)

    def run(self, record=True):
        cfg = self.cfg
        if record:
            self.recorder = Recorder(self.record_dir or RUN_DIR, self.fb, self.motor, cfg)
        self.log(f"e-liquid smell on {len(self.eliquid)} glomeruli; control on {len(self.control)}; "
                 f"nicotine drives {len(self.nic.idx):,} neurons; walking DNs {len(self.walk)}; "
                 f"KC->MBON synapses {len(self.mb.pos):,}")

        # baseline: no nicotine, no dopamine, no damage
        sps = []
        for c in range(cfg.baseline_cycles):
            self.cycle = -cfg.baseline_cycles + c
            for phase, k in self.cycle_windows(baseline=True):
                drive = self.drive_for(phase, k, baseline=True)
                header, fired, new_dead = self.run_window(drive, phase, learn=False, damage=False,
                                                          warm=(c == 0 and cfg.baseline_cycles > 1))
                if c > 0 or cfg.baseline_cycles == 1:
                    sps.append(header["sps"])
                self.emit(header, fired, new_dead)
        self.baseline_sps = float(np.mean(sps)) if sps else None
        self.log(f"baseline: {self.baseline_sps:,.0f} spikes/s mean; ceiling median "
                 f"{float(np.median(self.tox.ceiling)):.1f} Hz, max {float(self.tox.ceiling.max()):.0f} Hz")
        test = self.sniff_test()

        self.cycle = 0
        while self.stop is None:
            for phase, k in self.cycle_windows(baseline=False):
                drive = self.drive_for(phase, k, baseline=False)
                header, fired, new_dead = self.run_window(drive, phase, learn=True, damage=True)
                if phase == "puff" and k == cfg.puff - 1:
                    header["dopamine"] = int(self.mb.dopamine(+1, cfg.reward))
                    self.mb.apply()
                    vape_kill_again(self)
                if test is not None:
                    header["test"] = test
                    test = None
                self.nic.tick()
                self.stop = self.check_stop(header)
                header["stop"] = self.stop
                self.emit(header, fired, new_dead)
                if self.stop:
                    break
            if self.stop is None and self.puffs % cfg.test_every == 0:
                test = self.sniff_test()
            self.cycle += 1

        summary = self.summary()
        self.log(json.dumps(summary, indent=1))
        if self.recorder is not None:
            self.recorder.finish(self.tox, summary)
        return summary

    def summary(self):
        dead = self.tox.dead
        by_sc = np.bincount(self.sc_code[dead], minlength=len(self.sc_names))
        tot_sc = np.bincount(self.sc_code, minlength=len(self.sc_names))
        return {
            "stop": self.stop,
            "windows": self.window,
            "brain_seconds": round(self.window * self.secs, 3),
            "wall_seconds": round(time.time() - self.t_start, 1),
            "puffs": self.puffs,
            "nicotine_total": round(self.nic.total, 3),
            "final_dose": round(self.nic.dose, 4),
            "first_death": self.first_death,
            "dead": int(dead.sum()),
            "dead_frac": round(float(dead.mean()), 5),
            "dead_by_superclass": {str(self.sc_names[i]) or "unlabelled":
                                   {"dead": int(by_sc[i]), "of": int(tot_sc[i])}
                                   for i in np.argsort(-by_sc) if by_sc[i]},
            "walking_dn_alive": int((~dead[self.walk]).sum()),
            "kc_dead": int(dead[self.kc].sum()), "kc_total": int(len(self.kc)),
            "baseline_spikes_per_s": self.baseline_sps,
            "sniff_tests": self.tests,
            "mushroom_body": self.mb.stats(),
            "config": asdict(self.cfg),
            "honesty": "Connectome, synapse signs, DoOR receptor responses and soma positions are measured. "
                       "The nicotine model, excitotoxic death, e-liquid recipe, time compression and the "
                       "puff-reward pairing are chosen. No animal was involved.",
        }


def vape_kill_again(rig):
    """mb.apply() rewrites KC->MBON weights; re-zero every synapse of the dead so none come back."""
    import vape
    dead = np.flatnonzero(rig.tox.dead)
    if len(dead):
        gone = np.isin(rig.mb.pre, dead) | np.isin(rig.mb.post, dead)
        if gone.any():
            rig.fb.wdata[rig.mb.pos[gone]] = 0.0


# ---------------------------------------------------------------------------
# replay: downsample the raw recording into files a static page can load
# ---------------------------------------------------------------------------
def build_replay(run_dir=RUN_DIR, out_dir=REPLAY_DIR, max_frames=600, chunk=100, damage_chunk=20,
                 damage_frames=60):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    meta = json.loads((run_dir / "neurons.json").read_text(encoding="utf-8"))
    n = meta["n"]
    nbytes = (n + 7) // 8
    for f in ("neurons.bin", "neurons_meta.bin", "neurons.json", "summary.json", "config.json"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, out_dir / f)

    act = np.memmap(run_dir / "activity.raw", dtype=np.uint8, mode="r")
    W = len(act) // nbytes
    act = act[: W * nbytes].reshape(W, nbytes)
    stride = max(1, math.ceil(W / max_frames))
    K = math.ceil(W / stride)
    frames = np.zeros((K, nbytes), dtype=np.uint8)
    for k in range(K):
        # one real 24 ms window per frame (the last of its span). A union over
        # the span would light most of the brain: 30-50k of 165k fire per window.
        frames[k] = act[min((k + 1) * stride, W) - 1]
    act_files = []
    for c in range(0, K, chunk):
        name = f"act_{c // chunk:03d}.bin"
        (out_dir / name).write_bytes(frames[c:c + chunk].tobytes())
        act_files.append(name)

    death = np.frombuffer((run_dir / "death_window.bin").read_bytes(), dtype="<i4")
    death_k = np.where(death >= 0, death // stride, -1).astype("<i4")
    (out_dir / "death.bin").write_bytes(death_k.tobytes())

    dmg_windows = json.loads((run_dir / "damage_windows.json").read_text(encoding="utf-8"))
    dmg = np.memmap(run_dir / "damage.raw", dtype=np.uint8, mode="r")
    D = min(len(dmg) // n, len(dmg_windows))
    dmg_files, dmg_frames = [], []
    if D:
        dmg = dmg[: D * n].reshape(D, n)
        pick = np.unique(np.linspace(0, D - 1, min(damage_frames, D)).round().astype(int))
        sel = dmg[pick]
        dmg_frames = [int(dmg_windows[i]) // stride for i in pick]
        for c in range(0, len(pick), damage_chunk):
            name = f"dmg_{c // damage_chunk:03d}.bin"
            (out_dir / name).write_bytes(np.ascontiguousarray(sel[c:c + damage_chunk]).tobytes())
            dmg_files.append(name)

    rows = [json.loads(line) for line in (run_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines() if line]
    tl = []
    for k in range(K):
        span = rows[k * stride: (k + 1) * stride]
        if not span:
            break
        last = span[-1]
        tl.append({
            "w": last["w"], "phase": last["phase"], "puffs": last["puffs"], "nic": last["nic"],
            "dead": last["dead"], "dead_frac": last["dead_frac"],
            "sps": round(float(np.mean([s["sps"] for s in span])), 1),
            "fired": int(last["fired"]),
            "turn": round(float(np.mean([s["turn"] for s in span])), 4),
            "speed": round(float(np.mean([s["speed"] for s in span])), 4),
            "walk_alive": last["walk_alive"], "walk_total": last["walk_total"],
            "dmg_max": max(s["dmg_max"] for s in span), "at_risk": last["at_risk"],
            "mb": last["mb"],
            "new_dead": int(sum(s["new_dead"] for s in span)),
            "dopamine": int(sum(s.get("dopamine") or 0 for s in span)),
            "puffing": any(s["phase"].endswith("puff") for s in span),
            "test": next((s["test"] for s in span if s.get("test")), None),
            "stop": last.get("stop"),
        })
    index = {"n": n, "windows": W, "stride": stride, "frames": len(tl), "nbytes": nbytes,
             "act": act_files, "chunk": chunk, "damage": dmg_files, "damage_chunk": damage_chunk,
             "damage_frames": dmg_frames, "brain_ms_per_window": 24.0}
    (out_dir / "replay.json").write_text(json.dumps({"index": index, "timeline": tl}, separators=(",", ":")),
                                         encoding="utf-8")
    size = sum(p.stat().st_size for p in out_dir.iterdir())
    print(f"replay: {W} windows -> {len(tl)} frames (stride {stride}), {size / 1e6:.1f} MB in {out_dir}")
    return index


def load_brain():
    from flysim import FlyBrain
    return FlyBrain()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for f in Config.__dataclass_fields__.values():
        flag = "--" + f.name.replace("_", "-")
        ap.add_argument(flag, type=type(f.default), default=f.default)
    ap.add_argument("--record", default=str(RUN_DIR))
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--replay-only", action="store_true", help="rebuild the replay from an existing recording")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    if args.replay_only:
        build_replay(args.record, REPLAY_DIR)
        return
    cfg = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
    t = time.time()
    fb = load_brain()
    print(f"brain: {fb.n:,} neurons, {len(fb.wdata):,} synapses, loaded in {time.time() - t:.1f}s")
    rig = Rig(fb, cfg, record_dir=None if args.no_record else args.record)
    try:
        rig.run(record=not args.no_record)
    except KeyboardInterrupt:
        rig.stop = "interrupted"
        if rig.recorder is not None:
            rig.recorder.finish(rig.tox, rig.summary())
    if not args.no_record:
        build_replay(args.record, REPLAY_DIR)


if __name__ == "__main__":
    main()
