"""Review R3 regression: online updates keep the physical structure, check
internal energy-force consistency, roll back atomically on any failure,
and persist tensor states to disk (F05/F06/F08/F09 counterexamples,
hermetic — no torch).

Counterexamples from INDEPENDENT_REVIEW_20260909.md §R3: cell/PBC/charges/
magmoms were dropped on the way into fine-tuning; the energy-force FD
check probed only the rigid-translation zero mode; a candidate whose
validation crashed or whose artifact could not be saved stayed active
without a publish event; updater states carrying real tensors could not
be persisted with plain json.dumps.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from test_review_r1 import Model, Reference, _direction

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.runtime.events import MODEL_UPDATE, EventLog
from pyraimd2.runtime.models import (
    ModelRegistry,
    artifact_digest,
    dump_state_arrays,
    load_state_arrays,
)
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.switch.base import LabelObservation


def _observation(atoms, label, label_id, k=0.8):
    prediction = SurrogatePrediction(
        0.5 * k * float(np.sum(atoms.positions**2)), -k * atoms.positions,
        None, np.full(len(atoms), np.nan))
    return LabelObservation(step=0, atoms=atoms.copy(), prediction=prediction,
                            label=label, label_id=label_id)


def periodic_atoms(charge=0.2, magmom=1.0):
    atoms = Atoms("H2", positions=[[0.2, 0.1, 0.0], [1.1, 0.9, 0.8]],
                  cell=[3.0, 3.0, 3.0], pbc=True)
    atoms.set_initial_charges([charge, -charge])
    atoms.set_initial_magnetic_moments([magmom, -magmom])
    return atoms


class StructureProbe:
    """E = 1/2 k |x|^2 + 0.1 * cell volume + sum(charges); F = -k x.

    Records every structure it sees.  A positions-only model cannot notice
    dropped cell/PBC/charge fields — this one can (the review explicitly
    asks for a model that depends on the dropped fields).
    """

    def __init__(self, k=0.8):
        self.k = float(k)
        self.finetune_structures = []
        self.predict_structures = []

    @staticmethod
    def _signature(atoms):
        return {
            "cell": np.asarray(atoms.cell.array, dtype=float),
            "pbc": bool(np.all(atoms.pbc)),
            "charges": np.asarray(atoms.get_initial_charges(), dtype=float),
            "magmoms": np.asarray(atoms.get_initial_magnetic_moments(),
                                  dtype=float),
        }

    def predict(self, atoms):
        self.predict_structures.append(self._signature(atoms))
        x = atoms.positions
        volume = float(abs(np.linalg.det(atoms.cell.array)))
        energy = (0.5 * self.k * float(np.sum(x**2)) + 0.1 * volume
                  + float(atoms.get_initial_charges().sum()))
        return SurrogatePrediction(energy, -self.k * x, None,
                                   np.full(len(atoms), np.nan))

    def finetune(self, labels):
        for atoms, _result in labels:
            self.finetune_structures.append(self._signature(atoms))
        return _REPORT

    def state_dict(self):
        return {"k": self.k}

    def load_state_dict(self, state):
        self.k = float(state["k"])


from pyraimd2.surrogate.base import TrainReport

_REPORT = TrainReport(n_labels=1, n_epochs=1, initial_loss=1.0,
                      final_loss=0.1, member_losses=(0.1,), wall_time_s=0.0)


def test_training_guard_and_fd_probes_keep_the_physical_structure():
    model = StructureProbe()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=1, guard_size=1))
    atoms = periodic_atoms()
    atoms.set_constraint(FixAtoms(indices=[0]))
    label = EngineResult(1.0, -1.2 * atoms.positions, None, 0.0,
                         energy_kind="energy", force_consistent=True)
    published = updater(_observation(atoms, label, "label-1"))
    assert published is True
    # Fine-tune, guard evaluation and the FD probes must all see the same
    # physical structure: 3 A cubic cell, PBC, charges, magmoms.
    seen = model.finetune_structures + model.predict_structures
    assert model.finetune_structures, "fine-tune saw no structures"
    for signature in seen:
        np.testing.assert_allclose(signature["cell"], np.diag([3.0] * 3),
                                   rtol=0, atol=1e-15)
        assert signature["pbc"] is True
        np.testing.assert_allclose(signature["charges"], [0.2, -0.2],
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(signature["magmoms"], [1.0, -1.0],
                                   rtol=0, atol=1e-15)
    # The persisted payload itself carries the fields and the label kind.
    payload = updater.state_dict()["guard"][0]
    assert payload["cell"] == [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    assert payload["pbc"] == [True, True, True]
    assert payload["initial_charges"] == [0.2, -0.2]
    assert payload["constraint"]["indices"] == [0]
    assert payload["energy_kind"] == "energy"
    assert payload["force_consistent"] is True


class TranslationInvariantCheat:
    """E = 1/2 k |x0 - x1|^2 (relative coordinates only); after fine-tuning
    the reported forces are 1.1x the gradient of E while E is unchanged —
    the review's F06 cheat.  A rigid-translation FD probe cannot catch it:
    the forces of a translation-invariant potential sum to zero."""

    def __init__(self, k=1.0, force_scale=1.0):
        self.k = float(k)
        self.force_scale = float(force_scale)

    def predict(self, atoms):
        r = atoms.positions[0] - atoms.positions[1]
        energy = 0.5 * self.k * float(np.sum(r**2))
        gradient = self.k * r  # dE/dx0 = +k r, dE/dx1 = -k r
        forces = self.force_scale * np.array([-gradient, gradient])
        return SurrogatePrediction(energy, forces, None,
                                   np.full(len(atoms), np.nan))

    def finetune(self, labels):
        self.force_scale = 1.1  # the cheat: forces drift off the gradient
        return _REPORT

    def state_dict(self):
        return {"k": self.k, "force_scale": self.force_scale}

    def load_state_dict(self, state):
        self.k = float(state["k"])
        self.force_scale = float(state["force_scale"])


def _two_atom_observation(model, label_id):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.3, 0.2, -0.4]])
    r = atoms.positions[0] - atoms.positions[1]
    gradient = model.k * r
    label = EngineResult(0.5 * model.k * float(np.sum(r**2)),
                         np.array([-gradient, gradient]), None, 0.0)
    return _observation(atoms, label, label_id, k=model.k)


def test_fd_check_rejects_force_scaled_translation_invariant_candidate():
    model = TranslationInvariantCheat()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=1, guard_size=1))
    assert updater(_two_atom_observation(model, "label-1")) is False
    assert updater.n_updates == 0
    assert updater.rejections[-1]["reason"] == "energy_force_inconsistent"
    assert model.force_scale == 1.0  # rolled back to the parent state


def test_fd_check_accepts_force_consistent_candidate():
    model = TranslationInvariantCheat()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=1, guard_size=1))
    model.finetune = lambda labels: _REPORT  # honest training: scale stays 1
    assert updater(_two_atom_observation(model, "label-1")) is True
    assert updater.n_updates == 1


class ExplodingCandidate(StructureProbe):
    """Fine-tune succeeds, but the candidate cannot predict a guard
    geometry (the review's validation-crash case)."""

    def __init__(self):
        super().__init__()
        self.explode = False

    def predict(self, atoms):
        if self.explode:
            raise RuntimeError("candidate exploded on a guard geometry")
        return super().predict(atoms)

    def finetune(self, labels):
        self.explode = True
        self.k = 99.0  # candidate mutation that must be rolled back
        return _REPORT

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.explode = bool(state.get("explode", False))

    def state_dict(self):
        return {**super().state_dict(), "explode": self.explode}


def test_candidate_validation_crash_rolls_back_to_parent():
    model = ExplodingCandidate()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=1, guard_size=1))
    atoms = periodic_atoms()
    label = EngineResult(1.0, -1.2 * atoms.positions, None, 0.0)
    assert updater(_observation(atoms, label, "label-1")) is False
    assert model.k == pytest.approx(0.8)  # parent state restored
    assert updater.n_updates == 0
    assert updater.rejections[-1]["reason"] == "validation_failed"
    assert updater.n_consumed == 1


def test_artifact_save_failure_rolls_back_publish(tmp_path):
    updater_model = ArrayStateModel()
    runner, _, _, updater = _world_with_updater(tmp_path / "publish-oserror",
                                                updater_model)

    def fail_publish(model_id, record):
        raise OSError("disk full")

    runner.calc._model_publisher = fail_publish
    with pytest.raises(OSError, match="disk full"):
        runner.run(0)  # the initial label triggers an update attempt
    # The stopped run still satisfies the atomic-publish contract: no
    # generation bump, no commit event, parent model and updater state.
    assert runner.calc._model_generation == 0
    assert updater_model.k == pytest.approx(0.8)
    assert updater.n_updates == 0
    assert len(updater._pending) == 1  # the label is queued, not lost
    committed = [e for e in _events(tmp_path / "publish-oserror")
                 if e["type"] == MODEL_UPDATE]
    assert committed == []


def _world_with_updater(run_dir, model, update_n=1):
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    updater = GuardedUpdater(model, UpdatePolicy(n_label=update_n,
                                                 guard_size=1))
    runner = EnergeticRunner(
        atoms, model, Reference(), Store(run_dir / "trajectory.db"), "run",
        on_label=updater, run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=100, direction=_direction,
        force_budget=0.08, timestep_fs=0.1, time_cap_fs=2.0,
        check_probability=0.0, check_seed=2)
    return runner, model, None, updater


def _events(run_dir):
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


class ArrayStateModel(Model):
    """Harmonic model whose state carries a real array (stand-in for torch
    tensors; the serialization path is identical)."""

    def __init__(self, k=0.8):
        super().__init__(k)
        self.weights = np.linspace(0.0, 1.0, 6).reshape(2, 3)

    def finetune(self, labels):
        num = denom = 0.0
        n = 0
        for n, (atoms, result) in enumerate(labels, start=1):
            x = atoms.positions
            num -= float((np.asarray(result.forces) * x).sum())
            denom += float((x**2).sum())
        self.k = num / denom
        self.weights = self.weights + 0.5  # the state actually advances
        return _REPORT

    def state_dict(self):
        return {"k": self.k, "weights": self.weights.copy()}

    def load_state_dict(self, state):
        self.k = float(state["k"])
        self.weights = np.asarray(state["weights"], dtype=float)


def test_tensor_state_persists_through_artifact_and_replay(tmp_path):
    model = ArrayStateModel()
    runner, *_ = _world_with_updater(tmp_path / "tensor", model)
    runner.run(0)  # publishes an update whose state carries an array
    runner.close()
    run_dir = tmp_path / "tensor"

    registry = ModelRegistry(run_dir)
    model_id = "ArrayStateModel#g1"
    raw = registry.read(model_id, resolve=False)
    assert raw is not None
    arrays = list(_find_placeholders(raw["updater_state"]))
    assert arrays, "updater state was not converted to array placeholders"
    assert (run_dir / "models" / model_id
            / "state_arrays.npz").is_file()
    resolved = registry.read(model_id, resolve=True)
    weights = resolved["updater_state"]["surrogate"]["weights"]
    np.testing.assert_allclose(weights, model.weights, rtol=0, atol=1e-15)

    update_events = [e for e in _events(run_dir) if e["type"] == MODEL_UPDATE]
    assert len(update_events) == 1
    assert update_events[0]["artifact_digest"] == artifact_digest(raw)

    # New-process resume: replay restores the updater state from the
    # artifact's arrays (a stopped run without a clean close).
    resumed_model = ArrayStateModel()
    resumed = EnergeticRunner.resume(
        run_dir, resumed_model, Reference(),
        updater=GuardedUpdater(resumed_model,
                               UpdatePolicy(n_label=1, guard_size=1)),
        direction=_direction, event_log_force=True,
        checkpoint_interval_steps=100)
    np.testing.assert_allclose(resumed_model.weights, model.weights,
                               rtol=0, atol=1e-15)
    resumed.close()


def test_consumed_event_state_arrays_dedupe_and_replay(tmp_path):
    model = ArrayStateModel()
    runner, *_ = _world_with_updater(tmp_path / "consumed", model,
                                     update_n=100)  # no update fires
    runner.run(0)
    runner.run(1)  # two consumptions, identical state: one array file
    runner.close()
    run_dir = tmp_path / "consumed"
    store_dir = run_dir / "models" / "state-arrays"
    files = list(store_dir.glob("*.npz"))
    assert len(files) == 1  # content-addressed: identical arrays stored once

    resumed_model = ArrayStateModel()
    resumed = EnergeticRunner.resume(
        run_dir, resumed_model, Reference(),
        updater=GuardedUpdater(resumed_model,
                               UpdatePolicy(n_label=100, guard_size=1)),
        direction=_direction, checkpoint_interval_steps=100)
    resumed.run(1)
    assert resumed.calc.n_evaluations == 3
    resumed.close()


def _find_placeholders(state):
    if isinstance(state, dict):
        if state.get("__ndarray__"):
            yield state
            return
        for value in state.values():
            yield from _find_placeholders(value)
    elif isinstance(state, list):
        for value in state:
            yield from _find_placeholders(value)


def test_state_array_roundtrip_and_tamper_detection(tmp_path):
    from pyraimd2.runtime.models import content_array_sink, content_array_source

    sink = content_array_sink(tmp_path / "arrays")
    state = {"a": np.ones((2, 2)), "b": [1, "x", np.arange(3.0)], "c": 4}
    converted = dump_state_arrays(state, sink)
    json.dumps(converted)  # the whole point: the converted state is JSON-safe
    source = content_array_source(tmp_path / "arrays")
    restored = load_state_arrays(converted, source)
    np.testing.assert_allclose(restored["a"], state["a"])
    np.testing.assert_allclose(restored["b"][2], state["b"][2])
    assert restored["b"][0] == 1 and restored["c"] == 4

    placeholder = next(_find_placeholders(converted))
    broken = dict(placeholder, sha256="0" * 24)
    with pytest.raises(Exception, match="digest|sha256|tampered"):
        load_state_arrays({"x": broken}, source)
