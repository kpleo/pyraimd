"""WP03 resume acceptance: continuous runs vs stop + fresh-process resume.

A "new process" is simulated honestly for same-CPU determinism: brand new
Python objects (surrogate, updater, engine, Store, EventLog) sharing nothing
with the stopped run but the filesystem.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.runtime import ResumeError
from pyraimd2.runtime.events import (
    EVALUATION_COMMITTED,
    MODEL_UPDATE,
    RUN_END,
    TASK,
    EventLog,
    EventLogError,
)
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction


class HarmonicModel:
    """E = 1/2 k |x|^2, F = -k x; k is the updatable parameter."""

    def __init__(self, k=0.8):
        self.k = k

    def predict(self, atoms):
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None,
                                   np.full(len(atoms), np.nan))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2, quartic=0.05):
        self.k, self.quartic = k, quartic
        self.attempts = 0
        self.fail_on = set()

    def compute(self, atoms):
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise EngineError("deliberate reference failure")
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2 + 0.25 * self.quartic * x**4)),
                            -self.k * x - self.quartic * x**3, None, 0.0)


class LeastSquaresUpdater:
    """Deterministic stateful updater: every ``n_label``-th label refits the
    surrogate's k by least squares over the pending queue; the queue position
    is part of the exported state."""

    def __init__(self, model, n_label=3):
        self.model = model
        self.n_label = int(n_label)
        self.n_consumed = 0
        self.n_updates = 0
        self.queue = []

    def __call__(self, observation):
        self.queue.append({"positions": observation.atoms.positions.tolist(),
                           "forces": observation.label.forces.tolist(),
                           "label_id": observation.label_id})
        self.n_consumed += 1
        if self.n_consumed % self.n_label:
            return False  # consumed only; model unchanged
        num = denom = 0.0
        for item in self.queue:
            x = np.asarray(item["positions"])
            forces = np.asarray(item["forces"])
            num -= float((forces * x).sum())
            denom += float((x**2).sum())
        self.queue = []
        self.model.k = num / denom
        self.n_updates += 1
        return True

    def state_dict(self):
        return {"k": float(self.model.k), "n_consumed": self.n_consumed,
                "n_updates": self.n_updates, "n_label": self.n_label,
                "queue": self.queue}

    def load_state_dict(self, state):
        if int(state["n_label"]) != self.n_label:
            raise ValueError("updater recipe mismatch")
        self.model.k = float(state["k"])
        self.n_consumed = int(state["n_consumed"])
        self.n_updates = int(state["n_updates"])
        self.queue = [dict(item) for item in state["queue"]]


POLICY = {"force_budget": 0.08, "timestep_fs": 0.1, "check_probability": 0.5,
          "check_seed": 2, "time_cap_fs": 2.0}


def make_run(run_dir, *, engine=None, n_label=3, checkpoint_interval=7,
             log_force=False, policy_overrides=None):
    """Build a fresh runner world (no shared state with any other world)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    log = EventLog(run_dir, force=log_force)
    model = HarmonicModel()
    engine = Reference() if engine is None else engine
    updater = LeastSquaresUpdater(model, n_label=n_label)
    policy = dict(POLICY)
    policy.update(policy_overrides or {})
    runner = EnergeticRunner(atoms, model, engine, store, "run",
                             event_log=log, on_label=updater,
                             run_dir=run_dir,
                             checkpoint_interval_steps=checkpoint_interval,
                             **policy)
    return runner, model, engine, updater, store, log


def resume_world(run_dir, *, n_label=3, force=True, checkpoint_interval=7):
    """Resume with entirely new objects — the fresh-process resume."""
    model = HarmonicModel()
    engine = Reference()
    updater = LeastSquaresUpdater(model, n_label=n_label)
    runner = EnergeticRunner.resume(run_dir, model, engine, updater=updater,
                                    event_log_force=force,
                                    checkpoint_interval_steps=checkpoint_interval)
    return runner, model, engine, updater


def rows(store, run_id="run"):
    return sorted(store._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def run_record(store):
    """Per-step comparison record: routes, check draws, model chain, arrays."""
    record = []
    for row in rows(store):
        metadata = row.data.get("metadata") or {}
        context = metadata.get("context") or {}
        driving = row.data.get("driving") or {}
        record.append({
            "step": int(row.key_value_pairs["step"]),
            "route": row.key_value_pairs["route"],
            "reason": row.data.get("reason"),
            "check_draw": metadata.get("check_draw"),
            "checked": metadata.get("checked"),
            "accepted": metadata.get("accepted"),
            "violation": metadata.get("violation"),
            "model_id": context.get("model_id"),
            "label_id": row.data.get("engine_label_id"),
            "positions": row.toatoms().positions.tolist(),
            "momenta": row.toatoms().get_momenta().tolist(),
            "driving_energy": driving.get("energy"),
            "driving_forces": driving.get("forces"),
            "accepted_count": (metadata.get("verification") or {}).get("accepted_count"),
            "detected_count": (metadata.get("verification") or {}).get("detected_count"),
        })
    return record


def assert_same_run(record_a, record_b):
    assert len(record_a) == len(record_b) > 0
    for a, b in zip(record_a, record_b):
        for key in ("step", "route", "reason", "check_draw", "checked", "accepted",
                    "violation", "model_id", "label_id", "accepted_count",
                    "detected_count", "driving_energy"):
            assert a[key] == b[key], (key, a["step"], a[key], b[key])
        np.testing.assert_allclose(a["positions"], b["positions"],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(a["momenta"], b["momenta"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(a["driving_forces"], b["driving_forces"],
                                   rtol=0, atol=1e-12)


def test_continuous_100_vs_40_resume_60(tmp_path):
    continuous, model_c, _, updater_c, store_c, _log_c = make_run(
        tmp_path / "continuous")
    summary_c = continuous.run(100)
    assert summary_c.n_evaluations == 101

    stopped, _, _, _, _, _ = make_run(
        tmp_path / "resumed")
    stopped.run(40)
    stopped.close()  # deliberate end of the first process

    resumed, model_r, _, updater_r = resume_world(tmp_path / "resumed",
                                                  force=False)
    summary_r = resumed.run(60)
    assert summary_r.n_evaluations == 60

    record_c = run_record(store_c)
    record_r = run_record(Store(tmp_path / "resumed" / "trajectory.db"))
    assert_same_run(record_c, record_r)
    # Counts, model chain and updater bookkeeping strictly equal.
    assert continuous.calc.n_accepted == resumed.calc.n_accepted
    assert continuous.calc.reference_calls == resumed.calc.reference_calls
    assert continuous.calc.n_calibrations == resumed.calc.n_calibrations
    assert updater_c.n_consumed == updater_r.n_consumed
    assert updater_c.n_updates == updater_r.n_updates
    assert model_c.k == model_r.k
    assert (continuous.calc.verification.accepted_count
            == resumed.calc.verification.accepted_count)
    assert (continuous.calc.verification.detected_count
            == resumed.calc.verification.detected_count)
    # Final complete-step state matches within the fixed tolerance.
    np.testing.assert_allclose(continuous.atoms.positions, resumed.atoms.positions,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(continuous.atoms.get_momenta(),
                               resumed.atoms.get_momenta(), rtol=0, atol=1e-12)
    resumed.close()
    continuous.close()


def _boundary_steps(store):
    """Interruption points: after-anchor, after-accepted-check, model-update."""
    anchor_steps, check_steps = [], []
    for row in rows(store):
        metadata = row.data.get("metadata") or {}
        step = int(row.key_value_pairs["step"])
        record = metadata.get("calibration_after_previous_label") or {}
        if record.get("anchor"):
            anchor_steps.append(step)
        if metadata.get("checked") and row.key_value_pairs["route"] == "ml":
            check_steps.append(step)
    return anchor_steps, check_steps


def _update_boundaries(log):
    return [int(e["origin_evaluation_id"]) - 1
            for e in log.iter_events() if e.get("type") == MODEL_UPDATE]


@pytest.mark.parametrize("kind", ["anchor", "check", "update"])
def test_interruption_points_resume_identically(tmp_path, kind):
    probe, _, _, _, store_probe, log_probe = make_run(tmp_path / "probe")
    probe.run(100)
    anchor_steps, check_steps = _boundary_steps(store_probe)
    update_steps = _update_boundaries(log_probe)
    probe.close()
    choices = {"anchor": anchor_steps, "check": check_steps, "update": update_steps}
    assert choices[kind], f"no {kind} interruption point in the probe run"
    stop_step = choices[kind][len(choices[kind]) // 2]

    interrupted, _, _, _, _, _ = make_run(tmp_path / "interrupted",
                                          checkpoint_interval=1)
    interrupted.run(stop_step + 1)  # stop right after the boundary step
    interrupted.close()
    resumed, _, _, _ = resume_world(tmp_path / "interrupted", force=False,
                                    checkpoint_interval=1)
    resumed.run(100 - (stop_step + 1))
    record_p = run_record(Store(tmp_path / "probe" / "trajectory.db"))
    record_r = run_record(Store(tmp_path / "interrupted" / "trajectory.db"))
    assert_same_run(record_p, record_r)
    resumed.close()


def test_crash_window_replay_without_double_effects(tmp_path):
    # Interval larger than the window: checks and updates committed after the
    # last checkpoint must replay without re-sampling or re-consuming.
    crashed, _, _, _, _, _ = make_run(
        tmp_path / "crash", checkpoint_interval=10)
    crashed.run(40)  # checkpoints at 10/20/30/40
    crashed.run(5)  # window: steps 41..45, no new checkpoint — now "crash"
    del crashed  # abrupt stop: lock leaked, events fsynced per append
    log_reader = EventLog(tmp_path / "crash", force=True)
    n_tasks_before = sum(1 for e in log_reader.iter_events()
                         if e.get("type") == TASK)
    log_reader.close()

    resumed, _, _, updater_r = resume_world(tmp_path / "crash", force=True,
                                            checkpoint_interval=10)
    before = (resumed.calc.n_evaluations, resumed.calc.n_accepted,
              resumed.calc.reference_calls.copy(), updater_r.n_consumed)
    resumed.run(55)
    # The cost ledger is append-only: nothing pre-crash was lost or rewritten.
    log_reader = EventLog(tmp_path / "crash", force=True)
    n_tasks_after = sum(1 for e in log_reader.iter_events()
                        if e.get("type") == TASK)
    log_reader.close()
    assert n_tasks_after > n_tasks_before
    record_c = run_record(Store(tmp_path / "crash" / "trajectory.db"))
    assert len(record_c) == 101
    # Replay applied the window exactly once before continuing.
    assert before[0] == 46 and updater_r.n_consumed > 0
    continuous, _, _, updater_c, store_c, _ = make_run(tmp_path / "continuous")
    continuous.run(100)
    assert_same_run(run_record(store_c), record_c)
    assert updater_c.n_consumed == updater_r.n_consumed
    assert updater_c.n_updates == updater_r.n_updates
    resumed.close()
    continuous.close()


def test_reference_failure_resume_matches_uninterrupted(tmp_path):
    engine = Reference()
    # Choose a call inside the window after the checkpoint at step 7.
    engine.fail_on = {51}
    failed, _, _, _, _, _ = make_run(
        tmp_path / "failure", engine=engine, checkpoint_interval=7)
    with pytest.raises(EngineError, match="deliberate"):
        failed.run(100)
    with pytest.raises(RuntimeError, match="failed"):
        failed.run(1)  # the failed runner never continues half-step states
    del failed

    resumed, _, _, _ = resume_world(tmp_path / "failure", force=True)
    completed = resumed.calc.n_evaluations - 1  # steps done before the crash
    resumed.run(100 - completed)
    continuous, _, _, _, store_c, _ = make_run(tmp_path / "continuous")
    continuous.run(100)
    assert_same_run(run_record(store_c),
                    run_record(Store(tmp_path / "failure" / "trajectory.db")))
    # The failed attempt stays billed; the run's logical events are singular.
    log = EventLog(tmp_path / "failure", force=True)
    events = list(log.iter_events())
    failed_tasks = [e for e in events if e.get("type") == TASK
                    and e.get("status") == "failed"]
    assert failed_tasks and "deliberate" in failed_tasks[0]["error"]
    committed = [e for e in events if e["type"] == EVALUATION_COMMITTED]
    assert len(committed) == len({e["context"]["evaluation_id"]
                                  for e in committed}) == 101
    log.close()
    resumed.close()
    continuous.close()


def test_failed_update_refuses_resume_but_fork_works(tmp_path):
    class FragileUpdater(LeastSquaresUpdater):
        def __call__(self, observation):
            if self.n_consumed >= 2:
                raise RuntimeError("deliberate updater failure")
            return super().__call__(observation)

    run_dir = tmp_path / "update-failure"
    run_dir.mkdir()
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    log = EventLog(run_dir)
    model = HarmonicModel()
    updater = FragileUpdater(model)
    runner = EnergeticRunner(atoms, model, Reference(), store, "run",
                             event_log=log, on_label=updater,
                             run_dir=run_dir, checkpoint_interval_steps=1,
                             **POLICY)
    with pytest.raises(RuntimeError, match="deliberate updater failure"):
        runner.run(100)
    del runner

    with pytest.raises(ResumeError, match="failed model update"):
        resume_world(run_dir, force=True)
    # The escape hatch: fork from the last valid checkpoint, model carried.
    forked_dir = tmp_path / "forked"
    model_f = HarmonicModel()
    updater_f = LeastSquaresUpdater(model_f)
    forked = EnergeticRunner.fork(run_dir, forked_dir, "forked", model_f,
                                  Reference(), updater=updater_f,
                                  checkpoint_interval_steps=1)
    forked.run(5)
    assert forked.calc.n_evaluations == 6
    forked_log = forked.calc._event_log
    events = list(forked_log.iter_events())
    fork_events = [e for e in events if e["type"] == "forked_from"]
    assert fork_events and fork_events[0]["parent_run_id"] == "run"
    forked.close()


def test_truncated_checkpoint_resume_uses_previous(tmp_path):
    stopped, _, _, _, _, _ = make_run(tmp_path / "truncated",
                                      checkpoint_interval=1)
    stopped.run(10)
    stopped.close()
    arrays = tmp_path / "truncated" / "checkpoints" / "11" / "arrays.npz"
    with arrays.open("r+b") as fh:
        fh.truncate(16)  # tear the latest generation
    resumed, _, _, _ = resume_world(tmp_path / "truncated", force=False,
                                    checkpoint_interval=1)
    resumed_events = list(resumed.calc._event_log.iter_events())
    resumed_markers = [e for e in resumed_events if e["type"] == "resumed"]
    assert resumed_markers[0]["checkpoint_generation"] == 10
    assert resumed.calc.n_evaluations == 11  # checkpoint 10 + replayed step 10
    resumed.run(90)
    continuous, _, _, _, store_c, _ = make_run(tmp_path / "continuous")
    continuous.run(100)
    assert_same_run(run_record(store_c),
                    run_record(Store(tmp_path / "truncated" / "trajectory.db")))
    resumed.close()
    continuous.close()


def test_concurrent_resume_rejected_and_sequential_resume_consistent(tmp_path):
    stopped, _, _, _, _, _ = make_run(tmp_path / "dup", checkpoint_interval=5)
    stopped.run(10)
    stopped.close()
    first, _, _, _ = resume_world(tmp_path / "dup", force=False)
    with pytest.raises(EventLogError, match="active writer"):
        resume_world(tmp_path / "dup", force=False)
    first.run(10)
    first.close()  # deliberate end: a later resume continues honestly
    second, _, _, _ = resume_world(tmp_path / "dup", force=False)
    second.run(80)
    continuous, _, _, _, store_c, _ = make_run(tmp_path / "continuous")
    continuous.run(100)
    assert_same_run(run_record(store_c),
                    run_record(Store(tmp_path / "dup" / "trajectory.db")))
    second.close()
    continuous.close()


def test_stop_request_checkpoints_then_continues_identically(tmp_path):
    runner, _, _, _, _, _ = make_run(tmp_path / "stop",
                                     checkpoint_interval=1000)
    runner.run(20)
    runner.request_stop()
    summary = runner.run(1000)  # stops at the next complete step
    assert summary.n_evaluations == 1
    log = EventLog(tmp_path / "stop", force=True)
    run_ends = [e for e in log.iter_events() if e["type"] == RUN_END]
    assert run_ends and run_ends[0]["status"] == "stopped"
    assert (tmp_path / "stop" / "checkpoints" / "latest.json").exists()
    log.close()
    summary = runner.run(79)  # the same live runner may continue cleanly
    assert summary.n_evaluations == 79
    continuous, _, _, _, store_c, _ = make_run(tmp_path / "continuous")
    continuous.run(100)
    assert_same_run(run_record(store_c),
                    run_record(Store(tmp_path / "stop" / "trajectory.db")))
    runner.close()
    continuous.close()


def test_sigint_sets_flag_and_stops_at_boundary(tmp_path):
    import os
    import signal

    runner, _, _, _, _, _ = make_run(tmp_path / "sigint",
                                     checkpoint_interval=1000)
    runner._install_sigint_handler()
    runner.run(20)
    os.kill(os.getpid(), signal.SIGINT)  # the handler only sets the flag
    summary = runner.run(1000)
    assert summary.n_evaluations == 1  # stopped at the next complete step
    assert (tmp_path / "sigint" / "checkpoints" / "latest.json").exists()
    summary = runner.run(79)
    continuous, _, _, _, store_c, _ = make_run(tmp_path / "continuous")
    continuous.run(100)
    assert_same_run(run_record(store_c),
                    run_record(Store(tmp_path / "sigint" / "trajectory.db")))
    runner.close()
    continuous.close()


def test_missing_model_artifact_stops_resume_as_pending(tmp_path):
    stopped, _, _, _, _, log = make_run(tmp_path / "artifact",
                                        checkpoint_interval=5)
    stopped.run(8)
    stopped.close()
    # The checkpoint at step 5 precedes any model update in the window after
    # it; replaying that update needs its artifact.
    updates = [e for e in log.iter_events() if e.get("type") == MODEL_UPDATE
               and int(e["origin_evaluation_id"]) > 5]
    assert updates, "expected a model update after the last checkpoint"
    artifact = (tmp_path / "artifact" / "models"
                / updates[0]["model_id"].replace("/", "_") / "state.json")
    assert artifact.exists()
    artifact.unlink()
    with pytest.raises(ResumeError, match="missing or incomplete"):
        resume_world(tmp_path / "artifact", force=False)


def test_resume_requires_the_updater_for_consuming_runs(tmp_path):
    run_dir = tmp_path / "plain-callback"
    run_dir.mkdir()
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    log = EventLog(run_dir)
    model = HarmonicModel()
    runner = EnergeticRunner(atoms, model, Reference(), store, "run",
                             event_log=log,
                             on_label=lambda observation: None,  # no state export
                             run_dir=run_dir, checkpoint_interval_steps=1,
                             **POLICY)
    runner.run(3)
    runner.close()
    with pytest.raises(ResumeError, match="updater"):
        EnergeticRunner.resume(run_dir, HarmonicModel(), Reference())


def test_verified_probes_reused_after_mid_calibration_crash(tmp_path):
    engine = Reference()
    run_dir = tmp_path / "probe-crash"
    # With check_probability=1 and n_label=1 every evaluation labels and
    # updates, so eval 1's update triggers a recalibration at eval 2.
    run_dir.mkdir()
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    log = EventLog(run_dir)
    model = HarmonicModel()
    updater = LeastSquaresUpdater(model, n_label=1)
    policy = dict(POLICY, check_probability=1)
    runner = EnergeticRunner(atoms, model, engine, store, "run",
                             event_log=log, on_label=updater,
                             run_dir=run_dir, checkpoint_interval_steps=1,
                             **policy)
    runner.run(1)  # evals 0,1: anchor + probes + checked eval 1 -> update
    # eval 2's recalibration probes are calls 7..10; fail the third (call 9).
    engine.fail_on = {9}
    with pytest.raises(EngineError, match="deliberate"):
        runner.run(1)
    del runner  # crash: 2 verified probes persisted, eval 2 never proposed

    resumed_model = HarmonicModel()
    resumed_updater = LeastSquaresUpdater(resumed_model, n_label=1)
    resumed_engine = Reference()
    resumed = EnergeticRunner.resume(run_dir, resumed_model, resumed_engine,
                                     updater=resumed_updater,
                                     event_log_force=True,
                                     checkpoint_interval_steps=1)
    resumed.run(1)  # recalibration: 2 reused + 2 fresh probes, then the step
    # Two verified probes were reused, not re-executed; eval 2's own check
    # (p=1, uncached backend) accounts for the third execution.
    assert resumed_engine.attempts == 3
    row2 = next(r for r in rows(Store(run_dir / "trajectory.db"))
                if int(r.key_value_pairs["step"]) == 1)
    calibration = (row2.data["metadata"]["calibration_after_previous_label"]
                   ["anchor"]["calibration"])
    assert calibration["reused_probes"] == 2
    assert calibration["force_call_count"] == 5
    continuous, _, _, _, store_c, _ = make_run(
        tmp_path / "continuous", n_label=1, checkpoint_interval=1,
        policy_overrides={"check_probability": 1})
    continuous.run(2)
    assert_same_run(run_record(store_c),
                    run_record(Store(run_dir / "trajectory.db")))
    resumed.close()
    continuous.close()
