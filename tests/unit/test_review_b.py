"""Review B2/B3 consumer-side regression: the attempt sink is explicitly
connected (or the combination refused before launch), new logs are
recognized by an explicit protocol marker, and cache hits bill the access
time (INDEPENDENT_REVIEW_040_20260909 §B2/§B3; shared event semantics per
the agreed convention)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from test_review_r1 import Model, _direction

from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import ATTEMPT, TASK, EventLog, EventLogError
from pyraimd2.store import Store


class RetryingSinkEngine:
    """Accepts request_id and self-reports per-launch attempts through its
    sink (one failed launch, then a successful retry).  The sink starts
    disconnected (``_event_log = None``)."""

    fingerprint = "retrying-sink-engine:test"

    def __init__(self):
        self._event_log = None
        self.launches = 0

    def compute(self, atoms, *, request_id=None):
        self.launches += 1
        log = self._event_log
        assert log is not None, "sink not connected"
        log.append(ATTEMPT, {
            "record": "physical_attempt", "operation": "reference",
            "purpose": "verification", "request_id": request_id,
            "attempt": 1, "status": "failed", "started_unix": 0.0,
            "elapsed_s": 0.0, "returncode": 139, "directory": None,
            "start": "atomic", "source": "fake", "error": "launch hiccup"})
        log.append(ATTEMPT, {
            "record": "physical_attempt", "operation": "reference",
            "purpose": "verification", "request_id": request_id,
            "attempt": 2, "status": "success", "started_unix": 0.0,
            "elapsed_s": 0.0, "returncode": 0, "directory": None,
            "start": "atomic", "source": "fake", "error": None})
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * 1.2 * x**2)), -1.2 * x,
                            None, 0.0)


def _world(run_dir, engine):
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    return EnergeticRunner(
        atoms, Model(), engine, Store(run_dir / "trajectory.db"), "run",
        run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=100, direction=_direction,
        force_budget=0.08, timestep_fs=0.1, time_cap_fs=2.0,
        check_probability=1.0, check_seed=2)


def _events(run_dir):
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_b2_runner_log_connects_the_engine_sink(tmp_path):
    engine = RetryingSinkEngine()
    runner = _world(tmp_path / "b2", engine)
    runner.run(0)  # the initial anchor: one logical request, two launches
    runner.close()
    reference = summarize_tasks(_events(tmp_path / "b2"))["reference"]
    assert reference["logical_requests"] >= 1
    assert reference["actual_executions"] == 2 * reference["logical_requests"]
    assert reference["failed_attempts"] == reference["logical_requests"]
    assert engine.launches == reference["logical_requests"]
    # The sink is disconnected again after the call (no dangling state).
    assert engine._event_log is None


class NoSinkEngine:
    """Accepts request_id but exposes no attempt sink at all."""

    fingerprint = "no-sink-engine:test"

    def __init__(self):
        self.launches = 0

    def compute(self, atoms, *, request_id=None):
        self.launches += 1
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * 1.2 * x**2)), -1.2 * x,
                            None, 0.0)


def test_b2_sinkless_combination_is_refused_before_launch(tmp_path):
    engine = NoSinkEngine()
    runner = _world(tmp_path / "b2-refuse", engine)
    with pytest.raises(EventLogError, match="attempt sink"):
        runner.run(0)
    assert engine.launches == 0  # refused before any process start


def test_b3_new_protocol_log_without_attempts_is_not_legacy():
    # A fresh new-semantics log whose first logical request fails before any
    # launch must not be read with legacy semantics (actual=0, failed=0).
    new_log = [
        {"type": "run_start", "run_id": "r", "attempt_ledger": "physical_attempt_v1"},
        {"type": "task", "task_id": "t1", "operation": "reference",
         "purpose": "verification", "status": "failed", "elapsed_s": 0.0},
    ]
    reference = summarize_tasks(new_log)["reference"]
    assert reference["logical_requests"] == 1
    assert reference["actual_executions"] == 0
    assert reference["failed_attempts"] == 0
    # Same content without the marker and without any attempt event keeps
    # the legacy reading (the failed task was the failed execution).
    legacy = [dict(e) for e in new_log[1:]]
    reference = summarize_tasks(legacy)["reference"]
    assert reference["actual_executions"] == 1
    assert reference["failed_attempts"] == 1


def test_b3_cache_hit_bills_access_time_not_the_old_wall_time(tmp_path):
    from test_review_r4 import _write_config

    from pyraimd2.config import load_config
    from pyraimd2.workflows import md as md_module
    from pyraimd2.workflows import run_workflow

    class SlowReference:
        name = "slow-reference"
        fingerprint = "slow-reference:1"

        def compute(self, atoms):
            return EngineResult(0.0, np.zeros((len(atoms), 3)), None, 5.0)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: SlowReference())
    try:
        # Stationary run: forces are zero, so from step 2 on ASE serves the
        # calculator cache — those logical requests are cache hits.
        stopped = load_config(_write_config(tmp_path / "slow", mode="reference",
                                            steps=3, momenta=[[0.0, 0.0, 0.0]]))
        run_workflow(stopped, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()
    log = [json.loads(line) for line in
           (stopped.run.directory / "events.jsonl").read_text().splitlines()]
    hits = [e for e in log if e["type"] == TASK and e["status"] == "cache_hit"]
    assert hits, "the stationary resume step should hit the ASE cache"
    assert all(e["elapsed_s"] < 1.0 for e in hits)  # the access, not 5.0 s
    successes = [e for e in log if e["type"] == TASK and e["status"] == "success"
                 and e.get("operation") == "reference"]
    assert all(e["elapsed_s"] == 5.0 for e in successes)


def test_b3_failed_first_relax_evaluation_keeps_its_task_id(tmp_path):
    from test_review_r4 import _write_config

    from pyraimd2.config import load_config
    from pyraimd2.workflows import md as md_module
    from pyraimd2.workflows import run_workflow

    class FailingReference:
        name = "failing-reference"
        fingerprint = "failing-reference:1"

        def compute(self, atoms):
            raise EngineError("deliberate relax failure")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: FailingReference())
    try:
        config = load_config(_write_config(tmp_path / "relaxfail",
                                           kind="relax", mode="reference"))
        with pytest.raises(EngineError, match="deliberate"):
            run_workflow(config, verbose=False)
    finally:
        monkey.undo()
    log = [json.loads(line) for line in
           (config.run.directory / "events.jsonl").read_text().splitlines()]
    reference = summarize_tasks(log)["reference"]
    assert reference["logical_requests"] == 1
    assert reference["failed_attempts"] == 1
    attempts = [e for e in log if e["type"] == ATTEMPT]
    tasks = [e for e in log if e["type"] == TASK]
    assert len(attempts) == 1 and len(tasks) == 1
    assert attempts[0]["request_id"] == tasks[0]["task_id"]
