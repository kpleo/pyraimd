"""§5.4 numeric label cache: exact keys, default-off, static p=1 checks."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticCalculator
from pyraimd2.runtime.events import EVALUATION_COMMITTED, TASK, EventLog
from pyraimd2.runtime.labels import LabelCache, label_key
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction


def _atoms() -> Atoms:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    atoms.set_initial_charges([0.0, 0.0])
    atoms.set_initial_magnetic_moments([0.0, 0.0])
    return atoms


def test_label_key_sensitivity():
    base = label_key(_atoms(), "ref-a", "energy")
    assert base == label_key(_atoms(), "ref-a", "energy")

    def keyed(mutate, reference_id="ref-a", kind="energy",
              properties=("energy", "forces")):
        atoms = _atoms()
        mutate(atoms)
        return label_key(atoms, reference_id, kind, properties)

    assert keyed(lambda a: a.positions.__iadd__(0.001)) != base  # positions
    assert keyed(lambda a: a.set_cell([5, 5, 5])) != base  # cell
    assert keyed(lambda a: a.set_pbc([True, False, False])) != base  # pbc
    assert keyed(lambda a: a.set_initial_charges([1.0, -1.0])) != base  # charges
    assert keyed(lambda a: a.set_initial_magnetic_moments([1.0, 0.0])) != base
    assert keyed(lambda a: a.numbers.__setitem__(1, 2)) != base  # atom identity
    assert keyed(lambda a: None, reference_id="ref-b") != base  # settings
    assert keyed(lambda a: None, kind="free_energy") != base  # convention
    assert keyed(lambda a: None, properties=("energy",)) != base  # properties
    # Isotopes and velocities do not change the electronic label (§5.4).
    assert keyed(lambda a: a.set_masses([2.0, 1.0])) == base
    assert keyed(lambda a: a.set_momenta(np.ones((2, 3)))) == base


def test_label_cache_exact_match_only_and_disabled_states():
    atoms = _atoms()
    result = EngineResult(-1.0, np.zeros((2, 3)), None, 0.0, energy_kind="energy")
    cache = LabelCache("ref-a")
    assert cache.enabled
    cache.put(atoms, result, "label-1")
    assert cache.get(atoms, "energy") == (result, "label-1")
    assert cache.get(atoms, "free_energy") is None  # convention must match
    other_settings = LabelCache("ref-b")
    other_settings.put(atoms, result, "label-2")
    assert cache.get(atoms, "energy")[1] == "label-1"  # no cross-settings hit
    # Unidentifiable backend (no fingerprint): cache disabled by default.
    anonymous = LabelCache(None)
    assert not anonymous.enabled
    anonymous.put(atoms, result, "label-3")
    assert anonymous.get(atoms, "energy") is None
    moved = _atoms()
    moved.positions += 1e-9  # exact matching: no geometric approximation
    assert cache.get(moved, "energy") is None


class Harmonic:
    def __init__(self, k=0.8):
        self.k = k

    def predict(self, atoms):
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None,
                                   np.full(len(atoms), np.nan))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2):
        self.k = k
        self.attempts = 0

    def compute(self, atoms):
        self.attempts += 1
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2)), -self.k * x, None, 0.0)


class FingerprintedReference(Reference):
    @property
    def fingerprint(self):
        return f"analytic-reference:k={self.k}"


def setup_run(tmp_path, log, engine, **kwargs):
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(tmp_path / "run.db")
    model = Harmonic()
    options = {"force_budget": 0.1, "timestep_fs": 0.1, "check_probability": 1,
               "time_cap_fs": 1.0}
    options.update(kwargs)
    calc = EnergeticCalculator(model, engine, store, "run", event_log=log, **options)
    atoms.calc = calc
    return atoms, calc, store, model


def test_static_geometry_p1_cached_checks_count_without_new_scf(tmp_path):
    engine = FingerprintedReference()
    with EventLog(tmp_path) as log:
        atoms, calc, _store, model = setup_run(tmp_path, log, engine)
        callback_calls = []

        def update(observation):
            callback_calls.append(observation.label_id)
            if len(callback_calls) <= 2:
                model.k += 0.1  # first two labels announce a model change
                return None
            return False  # third label: model untouched, no new generation

        calc.on_label = update
        atoms.get_forces()  # eval 0: initial anchor label only (calibration
        # is deferred to the next proposal when on_label is set), gen -> 1
        assert engine.attempts == 1
        atoms.get_forces()  # eval 1: SAME geometry, recalibrated (4 probes),
        # forecast-accepted, checked -> verification served from label cache
        assert engine.attempts == 5  # only the recalibration probes ran
        assert calc.reference_calls["check"] == 0  # zero physical check SCF
        assert calc.verification.accepted_count == 1  # ...but the check counts
        assert calc.verification.detected_count == 0
        events = list(log.iter_events())
        cache_hits = [e for e in events if e.get("type") == TASK
                      and e["status"] == "cache_hit"]
        assert len(cache_hits) == 1
        assert cache_hits[0]["purpose"] == "verification"
        assert cache_hits[0]["label_id"] == "run-label-1"  # eval 0's label
        committed = [e for e in events if e["type"] == EVALUATION_COMMITTED]
        assert len(committed) == 2

        # eval 2: eval 1's cache-hit check still fired the callback (gen -> 2),
        # so the next same-geometry request is ANOTHER new accepted evaluation
        # — and p=1 checks it too, again from cache, again without new SCF.
        atoms.get_forces()
        assert engine.attempts == 9  # one more recalibration probe set
        assert calc.reference_calls["check"] == 0
        assert calc.verification.accepted_count == 2
        events = list(log.iter_events())
        cache_hits = [e for e in events if e.get("type") == TASK
                      and e["status"] == "cache_hit"]
        assert len(cache_hits) == 2
        committed = [e for e in events if e["type"] == EVALUATION_COMMITTED]
        assert len(committed) == 3

        # The third callback returned False (model unchanged): the identical
        # follow-up request is a REPLAY of the committed evaluation — no new
        # draw, no new check count, no new SCF, no new event.
        before = (engine.attempts, calc.verification.accepted_count,
                  len(cache_hits), len(committed))
        atoms.get_forces()
        events = list(log.iter_events())
        committed = [e for e in events if e["type"] == EVALUATION_COMMITTED]
        cache_hits = [e for e in events if e.get("type") == TASK
                      and e["status"] == "cache_hit"]
        assert (engine.attempts, calc.verification.accepted_count,
                len(cache_hits), len(committed)) == before


def test_unidentifiable_backend_gets_no_cross_evaluation_cache(tmp_path):
    engine = Reference()  # no fingerprint declared
    with EventLog(tmp_path) as log:
        atoms, calc, _store, model = setup_run(tmp_path, log, engine)

        def update(observation):
            model.k += 0.1

        calc.on_label = update
        atoms.get_forces()
        atoms.get_forces()  # same geometry, accepted, checked
        assert not calc._label_cache.enabled
        assert calc.reference_calls["check"] == 1  # real SCF for the check
        assert calc.verification.accepted_count == 1
        events = list(log.iter_events())
        assert not [e for e in events if e.get("type") == TASK
                    and e["status"] == "cache_hit"]
