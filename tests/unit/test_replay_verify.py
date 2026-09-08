"""WP02 verification replay: offline counts/bound must equal the online record."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticCalculator
from pyraimd2.store import Store
from pyraimd2.store.replay import replay_verification
from pyraimd2.surrogate.base import SurrogatePrediction


class Harmonic:
    def __init__(self, k=0.8):
        self.k = k

    def predict(self, atoms):
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None,
                                   np.full(len(atoms), np.nan))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2, quartic=0.0):
        self.k, self.quartic = k, quartic

    def compute(self, atoms):
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2 + 0.25 * self.quartic * x**4)),
                            -self.k * x - self.quartic * x**3, None, 0.0)


def setup(tmp_path, *, base=0.8, reference=None, position=0.2, **kwargs):
    atoms = Atoms("H", positions=[[position, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(tmp_path / "run.db")
    options = {"force_budget": 0.1, "timestep_fs": 0.1, "check_probability": 1,
               "time_cap_fs": 1.0}
    options.update(kwargs)
    calc = EnergeticCalculator(Harmonic(base), reference or Reference(), store, "run",
                               **options)
    atoms.calc = calc
    return atoms, calc, store


def test_replay_matches_online_with_violation_p1(tmp_path):
    atoms, calc, store = setup(tmp_path, position=0, reference=Reference(quartic=1000),
                               probe_steps=(0.01, 0.02), force_budget=3.0)
    atoms.get_forces()  # eval 0: initial anchor
    atoms.positions[0, 0] = 0.2
    atoms.get_forces()  # eval 1: accepted + checked, violation detected
    atoms.positions[0, 0] = 0.201
    atoms.get_forces()  # eval 2: forced reference route after the violation
    assert calc.verification.accepted_count == calc.verification.detected_count == 1
    replay = replay_verification(store, "run")
    assert replay["enabled"] and replay["matches_online"]
    assert replay["n_evaluations"] == 3
    assert replay["accepted_count"] == 1 and replay["detected_count"] == 1
    assert replay["probability"] == 1
    # p=1: the bound is the exact observed fraction.
    assert replay["bound"] == 1.0 == replay["detected_count"] / replay["accepted_count"]
    assert replay["mismatches"] == []
    recorded = replay["recorded"]
    assert recorded["accepted_count"] == 1 and recorded["detected_count"] == 1


def test_replay_matches_online_clean_acceptances_p1(tmp_path):
    atoms, calc, store = setup(tmp_path, force_budget=0.05)
    atoms.get_forces()
    for x in (0.205, 0.21):
        atoms.positions[0, 0] = x
        atoms.get_forces()  # small moves stay accepted; all checked at p=1
    assert calc.verification.accepted_count == 2
    assert calc.verification.detected_count == 0
    replay = replay_verification(store, "run")
    assert replay["enabled"] and replay["matches_online"]
    assert replay["accepted_count"] == 2 and replay["detected_count"] == 0
    assert replay["bound"] == 0.0


def test_replay_reports_disabled_for_p0(tmp_path):
    atoms, calc, store = setup(tmp_path, check_probability=0)
    atoms.get_forces()
    atoms.positions[0, 0] = 0.21
    atoms.get_forces()
    assert calc.verification is None
    replay = replay_verification(store, "run")
    assert replay["enabled"] is False
    assert replay["matches_online"] and replay["n_evaluations"] == 2
