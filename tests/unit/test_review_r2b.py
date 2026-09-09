"""Review R2 regression: attempt-sink identity, temporary connection and
the no-log API (0.4.2 plan §R2)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from test_review_r1 import Model, _direction

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import ATTEMPT, EventLog, EventLogError
from pyraimd2.store import Store


class QeLikeEngine:
    """Mimics the QE engine contract: compute accepts request_id and
    self-reports one attempt per real launch through its sink."""

    fingerprint = "qe-like:test"

    def __init__(self, event_log=None):
        self._event_log = event_log
        self.launches = 0

    def _attempt(self, request_id, attempt, status):
        if self._event_log is not None:
            self._event_log.append(ATTEMPT, {
                "record": "physical_attempt", "operation": "reference",
                "purpose": "verification", "request_id": request_id,
                "attempt": attempt, "status": status, "started_unix": 0.0,
                "elapsed_s": 0.0, "returncode": None, "directory": None,
                "start": "atomic", "source": "fake-qe",
                "error": None if status == "success" else "hiccup"})

    def compute(self, atoms, *, request_id=None):
        self.launches += 1
        self._attempt(request_id, 1, "failed")
        self.launches += 1
        self._attempt(request_id, 2, "success")
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * 1.2 * x**2)), -1.2 * x,
                            None, 0.0)


class BareSinklessEngine:
    """Accepts request_id but has no sink attribute at all (third-party
    no-log API)."""

    fingerprint = "bare-sinkless:test"

    def __init__(self):
        self.launches = 0

    def compute(self, atoms, *, request_id=None):
        self.launches += 1
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * 1.2 * x**2)), -1.2 * x,
                            None, 0.0)


def _runner(run_dir, engine, with_log=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    return EnergeticRunner(
        atoms, Model(), engine, Store(run_dir / "trajectory.db"), "run",
        # Library mode without a log: no run_dir, no checkpointing.
        run_dir=run_dir if with_log else None,
        event_log=EventLog(run_dir) if with_log else None,
        checkpoint_interval_steps=100, direction=_direction,
        force_budget=0.08, timestep_fs=0.1, time_cap_fs=2.0,
        check_probability=1.0, check_seed=2)


def _events(run_dir):
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_r2_runner_log_only_temporarily_connects_and_restores(tmp_path):
    engine = QeLikeEngine()  # engine without its own log
    runner = _runner(tmp_path / "r2-runner", engine)
    runner.run(0)
    runner.close()
    reference = summarize_tasks(_events(tmp_path / "r2-runner"))["reference"]
    assert reference["logical_requests"] >= 1
    assert reference["actual_executions"] == 2 * reference["logical_requests"]
    assert reference["failed_attempts"] == reference["logical_requests"]
    assert engine._event_log is None  # restored after the call


def test_r2_different_logs_is_refused_before_launch(tmp_path):
    engine_dir = tmp_path / "engine-log"
    engine_dir.mkdir()
    engine_log = EventLog(engine_dir, name="engine-events.jsonl")
    engine = QeLikeEngine(event_log=engine_log)
    runner = _runner(tmp_path / "r2-diff", engine)
    with pytest.raises(EventLogError, match="different attempt sink"):
        runner.run(0)
    assert engine.launches == 0
    engine_log.close()


def test_r2_error_also_restores_the_sink(tmp_path):
    class FailingEngine(QeLikeEngine):
        def compute(self, atoms, *, request_id=None):
            self.launches += 1
            self._attempt(request_id, 1, "failed")
            raise RuntimeError("compute blew up after a failed launch")

    engine = FailingEngine()
    runner = _runner(tmp_path / "r2-error", engine)
    with pytest.raises(Exception, match="blew up"):
        runner.run(0)
    assert engine._event_log is None  # restored even on the error path


def test_r2_no_logs_anywhere_computes_normally(tmp_path):
    engine = QeLikeEngine()
    runner = _runner(tmp_path / "r2-nolog", engine, with_log=False)
    runner.run(0)  # computes normally; every call fails then retries
    assert engine.launches >= 2
    assert runner.calc.n_evaluations == 1

    bare = BareSinklessEngine()
    runner = _runner(tmp_path / "r2-bare", bare, with_log=False)
    runner.run(0)  # no request_id is injected, no refusal, no sink needed
    assert bare.launches >= 1
