"""WP02 cost ledger acceptance: controlled traces matched by hand (hermetic)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticCalculator, EnergeticRunner
from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import (
    EVALUATION_COMMITTED,
    EVALUATION_PROPOSED,
    MODEL_UPDATE,
    RUN_END,
    RUN_START,
    RUN_SUMMARY,
    TASK,
    EventLog,
    EventLogError,
)
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction


class Harmonic:
    def __init__(self, k=0.8):
        self.k = k
        self.calls = 0

    def predict(self, atoms):
        self.calls += 1
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None,
                                   np.full(len(atoms), np.nan))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2):
        self.k = k
        self.attempts = 0
        self.fail_on = set()

    def compute(self, atoms):
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise EngineError("deliberate reference failure")
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2)), -self.k * x, None, 0.0)


def setup(tmp_path, event_log, *, base=0.8, reference=None, position=0.2, **kwargs):
    atoms = Atoms("H", positions=[[position, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(tmp_path / "run.db")
    model = Harmonic(base)
    engine = Reference() if reference is None else reference
    options = {"force_budget": 0.1, "timestep_fs": 0.1, "check_probability": 1,
               "time_cap_fs": 1.0}
    options.update(kwargs)
    calc = EnergeticCalculator(model, engine, store, "run", event_log=event_log, **options)
    atoms.calc = calc
    return atoms, calc, store, model, engine


def tasks(events, operation=None, purpose=None, status=None):
    selected = [e for e in events if e.get("type") == TASK]
    if operation is not None:
        selected = [e for e in selected if e["operation"] == operation]
    if purpose is not None:
        selected = [e for e in selected if e["purpose"] == purpose]
    if status is not None:
        selected = [e for e in selected if e["status"] == status]
    return selected


def test_controlled_trace_matches_handcomputed_ledger(tmp_path):
    with EventLog(tmp_path) as log:
        atoms, _calc, store, model, engine = setup(tmp_path, log)
        atoms.get_forces()  # eval 0: initial anchor + 4 calibration probes
        atoms.positions[0, 0] = 0.21
        atoms.get_forces()  # eval 1: forecast-accepted, checked (p=1)
        events = list(log.iter_events())

    # Hand-computed expectations for this exact call sequence.
    assert events[0]["type"] == RUN_START
    assert len(tasks(events, "reference", "anchor")) == 1
    assert len(tasks(events, "reference", "probe")) == 4
    assert len(tasks(events, "reference", "verification")) == 1
    assert len(tasks(events, "inference", "proposal")) == 2
    assert len(tasks(events, "inference", "probe")) == 4
    assert len(tasks(events, "io", "trajectory_append")) == 2
    assert len([e for e in events if e["type"] == EVALUATION_PROPOSED]) == 2
    committed = [e for e in events if e["type"] == EVALUATION_COMMITTED]
    assert [c["context"]["evaluation_id"] for c in committed] == [0, 1]
    assert committed[0]["input_hash"] and committed[1]["route"] == "ml"
    assert committed[1]["checked"] is True
    # Every reference task carries identity, timing and resource fields.
    for event in tasks(events, "reference"):
        assert event["task_id"].startswith("run-task-") and event["attempt"] == 1
        assert event["label_id"].startswith("run-label-")
        assert event["elapsed_s"] >= 0 and event["started_unix"] > 0
        assert event["cpu_cores"] is None and event["gpu"] is None
        assert event["queue_s"] is None and event["source"] == "energetic"
    summary = summarize_tasks(events)
    reference = summary["reference"]
    assert reference["logical_requests"] == 6
    assert reference["successful_executions"] == 6
    assert reference["failed_attempts"] == 0 and reference["cache_hits"] == 0
    assert reference["actual_executions"] == 6 == engine.attempts
    assert summary["counts"] == {"inference": 6, "training": 0, "io": 2}
    assert model.calls == 6  # 1+4 proposals/probes of eval 0, 1 of eval 1
    # Label IDs are durable and distinct per acquired label.
    label_ids = {e["label_id"] for e in tasks(events, "reference")}
    assert len(label_ids) == 6
    # Store rows carry schema version and the durable label ID.
    rows = list(store._db.select(run_id="run"))
    assert all(row.data["schema_version"] == 2 for row in rows)
    assert rows[0].data["engine_label_id"] == committed[0]["label_id"]


def test_failed_attempt_and_probes_enter_total_cost(tmp_path):
    engine = Reference()
    engine.fail_on = {3}  # second probe of eval 0 fails once, then retry runs
    with EventLog(tmp_path) as log:
        atoms, _calc, _, _, _ = setup(tmp_path, log, reference=engine)
        with pytest.raises(EngineError, match="deliberate"):
            atoms.get_forces()
        np.testing.assert_allclose(atoms.get_forces(), [[-0.24, 0, 0]])
        events = list(log.iter_events())
    failed = tasks(events, "reference", status="failed")
    assert len(failed) == 1 and failed[0]["purpose"] == "probe"
    assert failed[0]["elapsed_s"] >= 0 and "deliberate" in failed[0]["error"]
    summary = summarize_tasks(events)
    reference = summary["reference"]
    # 1 anchor + 1 probe ok, 1 probe failed, then 4 probes ok on retry.
    assert reference["successful_executions"] == 6
    assert reference["failed_attempts"] == 1
    assert reference["actual_executions"] == 7 == engine.attempts
    assert reference["logical_requests"] == 7
    probe_bucket = next(b for b in summary["by"]
                        if b["operation"] == "reference" and b["purpose"] == "probe")
    assert probe_bucket["count"] == 6 and probe_bucket["failed"] == 1
    # The retry reused the evaluation identity: exactly one proposed and one
    # committed event, and one io append — no duplicate logical events.
    assert len([e for e in events if e["type"] == EVALUATION_PROPOSED]) == 1
    assert len([e for e in events if e["type"] == EVALUATION_COMMITTED]) == 1
    assert len(tasks(events, "io")) == 1


def test_duplicate_callback_cannot_rewrite_a_logical_event(tmp_path):
    seen_label_ids = []

    with EventLog(tmp_path) as log:
        atoms, calc, _, model, _ = setup(tmp_path, log, check_probability=0)

        def update(observation):
            seen_label_ids.append(observation.label_id)
            model.k = 0.9
            # A callback re-invoked with the same label must not duplicate
            # its logical event: keyed appends are idempotent.
            first = log.append_once(f"label-consumed:{observation.label_id}",
                                    "label_consumed", {"by": "test"})
            again = log.append_once(f"label-consumed:{observation.label_id}",
                                    "label_consumed", {"by": "test"})
            assert first is not None and again is None

        calc.on_label = update
        atoms.get_forces()
        events = list(log.iter_events())
    assert seen_label_ids == ["run-label-1"]
    updates = [e for e in events if e["type"] == MODEL_UPDATE]
    assert len(updates) == 1
    assert updates[0]["origin_label_id"] == "run-label-1"
    assert updates[0]["generation"] == 1
    consumed = [e for e in events if e["type"] == "label_consumed"]
    assert len(consumed) == 1


def test_concurrent_run_start_rejected_by_lock(tmp_path):
    with EventLog(tmp_path) as log:
        atoms = Atoms("H", positions=[[0.2, 0, 0]])
        atoms.set_velocities([[0.1, 0, 0]])
        store = Store(tmp_path / "run.db")
        runner = EnergeticRunner(atoms, Harmonic(), Reference(), store, "run",
                                 force_budget=0.1, timestep_fs=0.1,
                                 check_probability=0, event_log=log)
        # A second writer for the same run directory is rejected outright.
        with pytest.raises(EventLogError, match="active writer"):
            EventLog(tmp_path)
        summary = runner.run(1)
        assert summary.n_evaluations == 2
        events = list(log.iter_events())
    summaries = [e for e in events if e["type"] == RUN_SUMMARY]
    assert len(summaries) == 1 and summaries[0]["wall_time_s"] >= 0
    # The outer wall time is measured directly; leaf tasks aggregate apart.
    assert summarize_tasks(events)["total_elapsed_s"] >= 0


def test_run_failure_recorded_with_reason(tmp_path):
    engine = Reference()
    engine.fail_on = {6}  # eval 0 fine (5 calls); eval 1's check fails
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(tmp_path / "run.db")
    with EventLog(tmp_path) as log:
        runner = EnergeticRunner(atoms, Harmonic(), engine, store, "run",
                                 force_budget=0.1, timestep_fs=0.1,
                                 check_probability=1, time_cap_fs=1.0,
                                 event_log=log)
        with pytest.raises(EngineError, match="deliberate"):
            runner.run(1)
        events = list(log.iter_events())
    failed = tasks(events, "reference", "verification", "failed")
    assert len(failed) == 1
    run_end = [e for e in events if e["type"] == RUN_END]
    assert run_end and run_end[0]["status"] == "failed"
    assert "deliberate" in run_end[0]["reason"]
    summary = summarize_tasks(events)
    assert summary["reference"]["failed_attempts"] == 1
    assert summary["reference"]["successful_executions"] == 5
