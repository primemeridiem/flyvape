"""
Tests for the vape model on a tiny real graph run by the real simulator.

The graph: neuron 0 is driven from outside and excites 1 and 2 (ACh);
1 excites 3 (ACh); 4 inhibits 3 (GABA). Neurons 5 and 6 are a Kenyon cell and
an MBON so the mushroom body has one synapse to rewrite.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import vape
from flysim import FlyBrain


def tiny_brain(tmp):
    n = 7
    types = np.array(["S", "A", "B", "C", "I", "KCab", "MBON01"])
    pre = [0, 0, 1, 4, 5]
    post = [1, 2, 3, 3, 6]
    w = np.array([+30, +30, +30, -30, +5], dtype=np.float32) * 0.275
    W = sp.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    path = Path(tmp) / "graph.npz"
    np.savez_compressed(path, data=W.data, indices=W.indices, indptr=W.indptr, shape=W.shape,
                        bodies=np.arange(100, 100 + n), sign=np.zeros(n, np.float32), types=types,
                        superclass=np.array(["s"] * n), subclass=np.array([""] * n),
                        receptor=np.array([""] * n), fru=np.array([""] * n), nt=np.array([""] * n))
    return FlyBrain(path)


class Nicotine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fb = tiny_brain(self.tmp)

    def test_ach_input_counts_only_cholinergic_contacts(self):
        ach = vape.ach_input(self.fb)
        self.assertEqual(ach.round(3).tolist(), [0, 30, 30, 30, 0, 0, 5])

    def test_level_rises_with_puffs_and_halves_each_half_life(self):
        nic = vape.Nicotine(self.fb, dose=0.4, half_life=10, max_hz=20)
        self.assertEqual(nic.drive(), {})
        nic.puff()
        nic.puff()
        self.assertAlmostEqual(nic.level, 0.8)
        nic.tick(10)
        self.assertAlmostEqual(nic.level, 0.4, places=6)
        self.assertAlmostEqual(nic.total, 0.8)
        (key, rates), = nic.drive().items()
        self.assertEqual(list(key), [1, 2, 3, 6])        # only neurons with ACh input
        self.assertTrue(np.all(rates <= 0.4 * 20 + 1e-6))


class Excitotoxicity(unittest.TestCase):
    def test_damage_only_above_own_ceiling_and_death_at_one(self):
        # tau=1: no smoothing, so the arithmetic is the rule itself
        tox = vape.Excitotoxicity(3, kappa=0.01, repair=0.0, margin=1.0, floor_hz=10.0, tau=1)
        tox.observe_baseline(np.array([100.0, 0.0, 0.0]))
        self.assertEqual(tox.ceiling.tolist(), [110.0, 10.0, 10.0])
        died = tox.update(np.array([110.0, 60.0, 5.0]), window=0)   # 50 Hz over on neuron 1
        self.assertEqual(len(died), 0)
        self.assertAlmostEqual(float(tox.damage[0]), 0.0)
        self.assertAlmostEqual(float(tox.damage[1]), 0.5, places=5)
        died = tox.update(np.array([110.0, 60.0, 5.0]), window=1)
        self.assertEqual(died.tolist(), [1])
        self.assertEqual(tox.death_window.tolist(), [-1, 1, -1])
        self.assertEqual(tox.update(np.array([0.0, 999.0, 0.0]), window=2).tolist(), [])  # dies once

    def test_one_chance_burst_does_not_kill_but_a_sustained_rise_does(self):
        tox = vape.Excitotoxicity(1, kappa=0.01, repair=0.0, margin=1.0, floor_hz=10.0, tau=20)
        tox.observe_baseline(np.array([0.0]), warm=True)
        for _ in range(40):
            tox.observe_baseline(np.array([0.0]))
        self.assertEqual(float(tox.ceiling[0]), 10.0)
        tox.update(np.array([125.0]), 0)                 # three chance spikes in one window
        self.assertLess(float(tox.ema[0]), 10.0)         # smoothed: 6.25 Hz, under the ceiling
        self.assertEqual(float(tox.damage[0]), 0.0)
        w = 1
        while not tox.dead[0] and w < 500:
            tox.update(np.array([60.0]), w)
            w += 1
        self.assertTrue(tox.dead[0])

    def test_warm_up_is_not_counted_in_the_ceiling(self):
        tox = vape.Excitotoxicity(1, margin=1.0, floor_hz=0.0, tau=1)
        tox.observe_baseline(np.array([500.0]), warm=True)
        tox.observe_baseline(np.array([20.0]))
        self.assertEqual(float(tox.ceiling[0]), 20.0)

    def test_no_nicotine_at_the_receptors_means_no_damage(self):
        tox = vape.Excitotoxicity(2, kappa=1.0, repair=0.0, margin=1.0, floor_hz=0.0, tau=1)
        tox.observe_baseline(np.array([0.0, 0.0]))
        tox.update(np.array([500.0, 500.0]), 0, exposure=np.array([0.0, 0.002], dtype=np.float32))
        self.assertEqual(float(tox.damage[0]), 0.0)
        self.assertAlmostEqual(float(tox.damage[1]), 1.0, places=4)

    def test_repair_floors_at_zero(self):
        tox = vape.Excitotoxicity(1, kappa=0.0, repair=0.5)
        tox.update(np.array([0.0]), 0)
        self.assertEqual(float(tox.damage[0]), 0.0)


class Death(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fb = tiny_brain(self.tmp)

    def run_chain(self, dead, k=5, steps=200):
        st = vape.rest_state(self.fb.n, seed=1)
        mask = np.zeros(self.fb.n, dtype=bool)
        mask[dead] = True
        fired = set()
        for _ in range(k):
            vape.silence(st, mask)
            r = self.fb.run({(0, 5): 400.0}, steps=steps, state=st)
            st = r["_state"]
            fired |= set(r["_fired"].tolist())
        return fired, st

    def test_a_dead_neuron_never_fires_again_and_drives_nothing(self):
        fired, _ = self.run_chain(dead=[])
        self.assertTrue({0, 1, 2, 3}.issubset(fired))
        vape.kill(self.fb, [1])
        fired, st = self.run_chain(dead=[1])
        self.assertNotIn(1, fired)
        self.assertNotIn(3, fired)             # its only excitatory input was 1
        self.assertIn(2, fired)                # 0 -> 2 is untouched
        self.assertGreater(int(st["refr"][1]), 10 ** 8)

    def test_kill_zeroes_outgoing_only(self):
        before = self.fb.wdata.copy()
        vape.kill(self.fb, [0])
        a, b = self.fb.indptr[0], self.fb.indptr[1]
        self.assertTrue(np.all(self.fb.wdata[a:b] == 0))
        rest = np.ones(len(before), dtype=bool)
        rest[a:b] = False
        self.assertTrue(np.array_equal(self.fb.wdata[rest], before[rest]))

    def test_mushroom_body_cannot_resurrect_a_dead_kenyon_cell(self):
        from mushroom import MushroomBody
        sides = Path(self.tmp) / "sides.json"
        sides.write_text(json.dumps({"types": {"MBON01": {"side": "PAM"}}}), encoding="utf-8")
        mb = MushroomBody(self.fb, sides=sides, store=Path(self.tmp) / "g.npz", clock=lambda: 0.0)
        self.assertEqual(len(mb.pos), 1)
        mb.observe(np.array([5]))
        vape.kill(self.fb, [5], mb)
        mb.dopamine(+1, 1.0)
        mb.apply()
        self.assertEqual(float(self.fb.wdata[mb.pos[0]]), 0.0)
        self.assertEqual(float(mb.base[0]), 0.0)


class Stop(unittest.TestCase):
    def rig(self):
        from vape_rig import Config

        class R:
            pass
        r = R()
        r.cfg = Config(collapse_windows=3, max_windows=100)
        r.walk = np.array([0, 1])
        r.tox = vape.Excitotoxicity(4)
        r.baseline_sps = 1000.0
        r.low_run = 0
        r.window = 0
        r.stopping = False
        import vape_rig
        r.check = lambda h: vape_rig.Rig.check_stop(r, h)
        return r

    def test_motor(self):
        r = self.rig()
        r.tox.dead[[0, 1]] = True
        self.assertEqual(r.check({"dead_frac": 0.5, "sps": 1000}), "motor")

    def test_half(self):
        r = self.rig()
        self.assertEqual(r.check({"dead_frac": 0.5, "sps": 1000}), "half")
        self.assertIsNone(r.check({"dead_frac": 0.49, "sps": 1000}))

    def test_silent_needs_consecutive_windows(self):
        r = self.rig()
        h = {"dead_frac": 0.0, "sps": 10}
        self.assertIsNone(r.check(h))
        self.assertIsNone(r.check(h))
        self.assertIsNone(r.check({"dead_frac": 0.0, "sps": 999}))   # resets
        self.assertIsNone(r.check(h))
        self.assertIsNone(r.check(h))
        self.assertEqual(r.check(h), "silent")


class Packing(unittest.TestCase):
    def test_packbits_round_trip_matches_page_bit_order(self):
        n = 21
        fired = np.array([0, 7, 8, 20])
        mask = np.zeros(n, dtype=bool)
        mask[fired] = True
        packed = np.packbits(mask)
        # the page reads bit i as (byte[i >> 3] >> (7 - (i & 7))) & 1
        back = [i for i in range(n) if (packed[i >> 3] >> (7 - (i & 7))) & 1]
        self.assertEqual(back, fired.tolist())


if __name__ == "__main__":
    unittest.main()
