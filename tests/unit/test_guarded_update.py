"""WP06: guarded model updates, label consumption, rollback, resume (hermetic)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import (
    EnergeticCalculator,
    EnergeticRunner,
    GuardedUpdater,
    LegacyCallbackAdapter,
    UpdatePolicy,
)
from pyraimd2.runtime.events import (
    LABEL_CONSUMED,
    MODEL_UPDATE,
    TASK,
    UPDATE_REJECTED,
    EventLog,
)
from pyraimd2.runtime.models import ModelRegistry
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction, TrainReport
from pyraimd2.switch.base import LabelObservation


class TrainableHarmonic:
    """E = 1/2 k x^2, F = -k x + bias; k fits by least squares on labels."""

    def __init__(self, k=0.8, bias=0.0):
        self.k = float(k)
        self.bias = float(bias)
        self.finetune_calls = 0
        self.fail_on: set[int] = set()
        self.break_consistency_on: set[int] = set()
        self.poison_x_gt: float | None = None  # data-driven guard failure

    @property
    def fingerprint(self):
        # Settings-level lineage; parameters live in the updater state.
        return "trainable-harmonic"

    def predict(self, atoms):
        x = atoms.positions
        return SurrogatePrediction(0.5 * self.k * float(np.sum(x**2)),
                                   -self.k * x + self.bias, None,
                                   np.full(len(atoms), np.nan))

    def finetune(self, labels):
        self.finetune_calls += 1
        if self.finetune_calls in self.fail_on:
            raise RuntimeError("deliberate training failure")
        num = denom = 0.0
        n = 0
        poison = False
        for atoms, result in labels:
            x = atoms.positions
            num -= float((result.forces * x).sum())
            denom += float((x**2).sum())
            n += 1
            poison = poison or (self.poison_x_gt is not None
                                and bool((np.abs(x) > self.poison_x_gt).any()))
        self.k = num / denom
        if poison:
            self.k *= 50.0
        if self.finetune_calls in self.break_consistency_on:
            self.bias = 10.0  # force without an energy term: inconsistent
        return TrainReport(n_labels=n, n_epochs=1, initial_loss=1.0,
                           final_loss=0.1, member_losses=(0.1,), wall_time_s=0.0)

    def state_dict(self):
        return {"k": self.k, "bias": self.bias,
                "finetune_calls": self.finetune_calls}

    def load_state_dict(self, state):
        self.k = float(state["k"])
        self.bias = float(state.get("bias", 0.0))
        self.finetune_calls = int(state.get("finetune_calls", 0))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2, quartic=0.05):
        self.k, self.quartic = k, quartic
        self.attempts = 0

    def compute(self, atoms):
        self.attempts += 1
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2 + 0.25 * self.quartic * x**4)),
                            -self.k * x - self.quartic * x**3, None, 0.0)


POLICY = {"force_budget": 0.08, "timestep_fs": 0.1, "check_probability": 0.5,
          "check_seed": 2, "time_cap_fs": 2.0}


def make_run(run_dir, *, engine=None, update_policy=None, checkpoint_interval=7,
             log_force=False, policy_overrides=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    log = EventLog(run_dir, force=log_force)
    model = TrainableHarmonic()
    engine = Reference() if engine is None else engine
    updater = GuardedUpdater(model, update_policy or UpdatePolicy(n_label=3,
                                                                  guard_size=2))
    policy = dict(POLICY)
    policy.update(policy_overrides or {})
    runner = EnergeticRunner(atoms, model, engine, store, "run",
                             event_log=log, on_label=updater,
                             run_dir=run_dir,
                             checkpoint_interval_steps=checkpoint_interval,
                             **policy)
    return runner, model, engine, updater, store, log


def resume_world(run_dir, *, update_policy=None, force=True, checkpoint_interval=7):
    model = TrainableHarmonic()
    updater = GuardedUpdater(model, update_policy or UpdatePolicy(n_label=3,
                                                                  guard_size=2))
    runner = EnergeticRunner.resume(run_dir, model, Reference(), updater=updater,
                                    event_log_force=force,
                                    checkpoint_interval_steps=checkpoint_interval)
    return runner, model, updater


def rows(store, run_id="run"):
    return sorted(store._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def _observation(label_id, position=0.3, k_ref=1.2):
    atoms = Atoms("H", positions=[[position, 0, 0]])
    label = EngineResult(0.5 * k_ref * position**2,
                         np.array([[-k_ref * position, 0.0, 0.0]]), None, 0.0)
    prediction = SurrogatePrediction(0.5 * 0.8 * position**2,
                                     np.array([[-0.8 * position, 0.0, 0.0]]), None,
                                     np.full(1, np.nan))
    return LabelObservation(0, atoms, prediction, label, label_id=label_id)


# -- acceptance: record-only callbacks never re-probe spuriously ---------------


def test_record_only_callback_triggers_no_unnecessary_reprobes(tmp_path):
    seen = []

    def record_only(observation):
        seen.append(observation.label_id)  # legacy None return: nothing changed

    run_dir = tmp_path / "recorded"
    run_dir.mkdir()
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    log = EventLog(run_dir)
    runner = EnergeticRunner(atoms, TrainableHarmonic(), Reference(), store,
                             "run", event_log=log,
                             on_label=LegacyCallbackAdapter(record_only),
                             run_dir=run_dir, **POLICY)
    runner.run(12)
    events = list(log.iter_events())
    assert not [e for e in events if e["type"] == MODEL_UPDATE]
    assert runner.calc._model_generation == 0
    # Baseline without any callback: identical routing and reference cost.
    base_dir = tmp_path / "baseline"
    base_dir.mkdir()
    base_atoms = Atoms("H", positions=[[0.2, 0, 0]])
    base_atoms.set_velocities([[0.1, 0, 0]])
    base_store = Store(base_dir / "trajectory.db")
    base_runner = EnergeticRunner(base_atoms, TrainableHarmonic(), Reference(),
                                  base_store, "run", **POLICY)
    base_runner.run(12)
    assert runner.calc.reference_calls == base_runner.calc.reference_calls
    assert [r.key_value_pairs["route"] for r in rows(store)] == \
           [r.key_value_pairs["route"] for r in rows(base_store)]
    assert seen  # the callback did observe every label


# -- acceptance: durable label IDs dedup consumption ----------------------------


def test_redelivered_label_never_consumed_or_trained_twice():
    model = TrainableHarmonic()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=2, guard_size=1))
    assert updater(_observation("L1")) is False
    assert updater(_observation("L1")) is False  # redelivery: no-op
    assert updater.n_consumed == 1 and model.finetune_calls == 0
    assert updater(_observation("L2")) is True  # second new label fires
    assert model.finetune_calls == 1 and updater.n_updates == 1
    assert updater(_observation("L2")) is False  # redelivery after firing
    assert updater.n_consumed == 2 and model.finetune_calls == 1
    assert updater.update_record()["label_ids"] == ["L1", "L2"]


def test_full_history_recipe_reuses_history_explicitly():
    model = TrainableHarmonic()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=2, guard_size=1,
                                                 train_on="full_history"))
    updater(_observation("L1"))
    updater(_observation("L1"))  # redelivery never re-consumes
    assert len(updater._history) == 1
    updater(_observation("L2"))
    trained_ids = [item["label_id"] for item in updater._history]
    assert trained_ids == ["L1", "L2"]  # recipe reuse: each exactly once


# -- acceptance: publish path ---------------------------------------------------


def test_publish_produces_artifact_and_reanchors_with_new_generation(tmp_path):
    runner, model, _, _, store, log = make_run(tmp_path / "publish")
    runner.run(12)
    events = list(log.iter_events())
    updates = [e for e in events if e["type"] == MODEL_UPDATE]
    assert updates, "expected at least one published update"
    first = updates[0]
    assert first["generation"] == 1
    assert len(first["label_ids"]) == 3
    artifact = ModelRegistry(tmp_path / "publish").read(first["model_id"])
    assert artifact["parent_model_id"] == "trainable-harmonic#g0"
    assert artifact["generation"] == 1
    assert artifact["label_ids"] == first["label_ids"]
    assert artifact["recipe"]["n_label"] == 3
    assert artifact["training"]["n_labels"] == 3
    assert artifact["updater_state"]["n_updates"] == 1
    # The model actually moved toward the reference curvature.
    assert model.k == pytest.approx(1.2, rel=0.1)
    # After publishing, the next evaluation re-anchors under the new
    # generation — old responses are never reused.
    origin_eval = first["origin_evaluation_id"]
    following = next(r for r in rows(store)
                     if int(r.key_value_pairs["step"]) == origin_eval)
    record = following.data["metadata"]["calibration_after_previous_label"]
    assert record["anchor"]["model_generation"] == 1
    assert record["anchor"]["segment_id"] != 1


def test_stale_pending_rejected_after_model_change(tmp_path):
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(tmp_path / "run.db")
    model = TrainableHarmonic()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=1, guard_size=1))
    calc = EnergeticCalculator(model, Reference(), store, "run",
                               on_label=updater, **POLICY)
    atoms.calc = calc
    atoms.get_forces()  # eval 0 labels -> publishes -> generation 1
    assert calc._model_generation == 1
    calc._evaluation_calls_before = calc.reference_calls.copy()
    pending = calc._freeze(atoms, calc.n_evaluations)
    calc._pending = pending
    calc._model_generation += 1  # another model change arrives
    with pytest.raises(ValueError, match="model generation"):
        atoms.get_forces()


# -- acceptance: rollback paths ---------------------------------------------------


def _run_rejection(tmp_path, name, mutate, expected_reason):
    run_dir = tmp_path / name
    # check_probability=1 makes every accepted evaluation label: exactly one
    # finetune fires inside run(4), so the rollback can be observed in full.
    runner, model, _, updater, _, log = make_run(
        run_dir, policy_overrides={"check_probability": 1})
    mutate(model)
    runner.run(4)
    events = list(log.iter_events())
    rejections = [e for e in events if e["type"] == UPDATE_REJECTED]
    assert len(rejections) == 1
    assert rejections[0]["reason"] == expected_reason
    assert not [e for e in events if e["type"] == MODEL_UPDATE]
    assert runner.calc._model_generation == 0
    assert updater.n_updates == 0 and updater.n_rejected == 1
    # The parent model is back: parameter state equals the known version.
    assert model.state_dict() == {"k": 0.8, "bias": 0.0, "finetune_calls": 0}
    consumed = [e for e in events if e["type"] == LABEL_CONSUMED]
    assert consumed and consumed[-1]["updater_state"]["n_rejected"] == 1
    # The failed attempt stays billed as training cost.
    training = [e for e in events if e["type"] == TASK
                and e["operation"] == "training"]
    assert training and training[0]["status"] == "success"
    return runner, model, updater


def test_rollback_on_force_growth_restores_parent(tmp_path):
    _run_rejection(tmp_path, "growth",
                   lambda model: setattr(model, "poison_x_gt", 0.0),
                   "force_growth")


def test_rollback_on_energy_force_inconsistency_restores_parent(tmp_path):
    _run_rejection(tmp_path, "consistency",
                   lambda model: model.break_consistency_on.add(1),
                   "energy_force_inconsistent")


def test_rollback_on_training_failure_restores_parent(tmp_path):
    _run_rejection(tmp_path, "training",
                   lambda model: model.fail_on.add(1), "training_failed")


# -- acceptance: full resume with a lightweight savable updater ------------------


def test_guarded_updater_resume_matches_continuous(tmp_path):
    continuous, model_c, _, updater_c, store_c, _ = make_run(
        tmp_path / "continuous")
    continuous.run(60)

    stopped, _, _, _, _, _ = make_run(tmp_path / "resumed")
    stopped.run(30)
    stopped.close()
    resumed, model_r, updater_r = resume_world(tmp_path / "resumed",
                                               force=False)
    resumed.run(30)

    def record(store):
        result = []
        for row in rows(store):
            metadata = row.data.get("metadata") or {}
            context = metadata.get("context") or {}
            result.append((int(row.key_value_pairs["step"]),
                           row.key_value_pairs["route"],
                           metadata.get("check_draw"),
                           context.get("model_id"),
                           row.data.get("engine_label_id"),
                           row.toatoms().positions.tolist(),
                           row.toatoms().get_momenta().tolist()))
        return result

    record_c = record(store_c)
    record_r = record(Store(tmp_path / "resumed" / "trajectory.db"))
    assert len(record_c) == len(record_r) == 61
    for a, b in zip(record_c, record_r):
        assert a[:5] == b[:5]
        np.testing.assert_allclose(a[5], b[5], rtol=0, atol=1e-12)
        np.testing.assert_allclose(a[6], b[6], rtol=0, atol=1e-12)
    assert (updater_c.n_consumed, updater_c.n_updates, updater_c.n_rejected) == \
           (updater_r.n_consumed, updater_r.n_updates, updater_r.n_rejected)
    assert model_c.k == model_r.k
    assert updater_c.n_updates > 0  # updates really fired in this window
    resumed.close()
    continuous.close()


def test_guarded_updater_resume_with_rejection_in_window(tmp_path):
    # Data-driven poison: any finetune whose training set contains a
    # configuration past 0.23 A produces a guard-failing candidate. The
    # schedule depends only on the trajectory, so it is identical in the
    # continuous and the resumed world (and advances across rollbacks).
    continuous, model_c, _, updater_c, store_c, _ = make_run(
        tmp_path / "continuous")
    model_c.poison_x_gt = 0.215
    continuous.run(40)

    stopped, model_s, _, _, _, _ = make_run(tmp_path / "resumed")
    model_s.poison_x_gt = 0.215
    stopped.run(14)
    stopped.close()
    resumed, model_r, updater_r = resume_world(tmp_path / "resumed",
                                               force=False)
    model_r.poison_x_gt = 0.215
    resumed.run(26)

    record_c = [(int(r.key_value_pairs["step"]), r.key_value_pairs["route"],
                 (r.data.get("metadata") or {}).get("check_draw"),
                 r.toatoms().positions.tolist())
                for r in rows(store_c)]
    record_r = [(int(r.key_value_pairs["step"]), r.key_value_pairs["route"],
                 (r.data.get("metadata") or {}).get("check_draw"),
                 r.toatoms().positions.tolist())
                for r in rows(Store(tmp_path / "resumed" / "trajectory.db"))]
    assert len(record_c) == len(record_r) == 41
    for a, b in zip(record_c, record_r):
        assert a[:3] == b[:3]
        np.testing.assert_allclose(a[3], b[3], rtol=0, atol=1e-12)
    assert updater_c.n_rejected == updater_r.n_rejected >= 1
    assert updater_c.n_updates == updater_r.n_updates
    assert model_c.k == model_r.k
    resumed.close()
    continuous.close()
