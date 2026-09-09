"""Review R6 regression: the cost ledger distinguishes logical requests
from physical process executions (hermetic analytic engines).

Counterexamples from INDEPENDENT_REVIEW_20260909.md §R6: one logical
reference request with an internal retry must report logical=1/actual=2/
failed=1; a cache hit and a pre-launch failure are not physical
executions; replay after a crash must not fabricate executions.  Attempt
events follow the locked engine convention (type "attempt",
record="physical_attempt", request_id naming the parent task — see
REVIEW_FIXES_BACKEND_20260909.md).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from ase import Atoms
from test_review_r1 import Model, Reference, _direction, world

from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import ATTEMPT, TASK, EventLog
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store


def _emit(engine, request_id, *, attempt, status, started, error=None):
    engine._event_log.append(ATTEMPT, {
        "record": "physical_attempt", "operation": "reference",
        "purpose": "scf", "request_id": request_id, "attempt": attempt,
        "status": status, "started_unix": started, "elapsed_s": 0.0,
        "returncode": 0 if status == "success" else 1,
        "directory": None, "start": "atomic",
        "source": "fake-engine", "error": error})


class RetryingReference:
    """Self-reporting engine (accepts ``request_id``): the first launch of
    every call fails, the retry succeeds — one logical request, two
    physical attempts."""

    name = "retrying-reference"

    def __init__(self, event_log, k=1.2):
        self._event_log = event_log
        self.k = k

    @property
    def fingerprint(self):
        return f"retrying-reference:k={self.k}"

    def compute(self, atoms, *, request_id=None):
        started = time.time()
        _emit(self, request_id, attempt=1, status="failed", started=started,
              error="EngineError('first launch failed')")
        x = atoms.positions
        _emit(self, request_id, attempt=2, status="success", started=started)
        return EngineResult(float(np.sum(0.5 * self.k * x**2)), -self.k * x,
                            None, 0.0)


class PrecheckReference(Reference):
    """Self-reporting engine that can fail before launching anything."""

    def __init__(self, event_log, k=1.2):
        super().__init__(k)
        self._event_log = event_log
        self.fail_precheck = False

    def compute(self, atoms, *, request_id=None):
        if self.fail_precheck:
            raise EngineError("precheck failed before any launch")
        started = time.time()
        result = super().compute(atoms)
        _emit(self, request_id, attempt=1, status="success", started=started)
        return result


def runner_with(engine_factory, run_dir, *, stationary=False, p=1.0):
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.0 if stationary else 0.2, 0, 0]])
    atoms.set_velocities([[0.0 if stationary else 0.1, 0, 0]])
    log = EventLog(run_dir)
    engine = engine_factory(log)
    runner = EnergeticRunner(
        atoms, Model(), engine, Store(run_dir / "trajectory.db"), "run",
        run_dir=run_dir, event_log=log,
        checkpoint_interval_steps=100, direction=_direction,
        force_budget=0.08, timestep_fs=0.1, time_cap_fs=2.0,
        check_probability=p, check_seed=2)
    return runner, engine


def events(run_dir):
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_internal_retry_counts_logical_one_actual_two(tmp_path):
    run_dir = tmp_path / "retry"
    runner, _ = runner_with(RetryingReference, run_dir)
    runner.run(0)  # every logical reference request retries internally
    runner.close()
    reference = summarize_tasks(events(run_dir))["reference"]
    assert reference["logical_requests"] >= 1
    # One logical request, two physical launches (failed + succeeded).
    assert reference["actual_executions"] == 2 * reference["logical_requests"]
    assert reference["failed_attempts"] == reference["logical_requests"]
    assert reference["successful_executions"] == reference["logical_requests"]
    log = events(run_dir)
    attempts = [e for e in log if e["type"] == ATTEMPT
                and e.get("record") == "physical_attempt"]
    assert {e["status"] for e in attempts} == {"failed", "success"}
    task_ids = {e["task_id"] for e in log if e["type"] == TASK}
    assert all(e["request_id"] in task_ids for e in attempts)


def test_cache_hit_is_not_a_physical_execution(tmp_path):
    runner, _, _, _ = world(tmp_path / "hit", stationary=True, p=1.0)
    runner.run(0)
    runner.run(1)  # same geometry: the check is served from the label cache
    runner.close()
    reference = summarize_tasks(events(tmp_path / "hit"))["reference"]
    assert reference["cache_hits"] == 1
    # Every logical request launched except the one served from cache.
    assert reference["actual_executions"] == \
        reference["logical_requests"] - reference["cache_hits"]
    attempts = [e for e in events(tmp_path / "hit") if e["type"] == ATTEMPT]
    assert len(attempts) == reference["actual_executions"]


def test_prelaunch_failure_is_not_a_physical_execution(tmp_path):
    run_dir = tmp_path / "precheck"
    runner, engine = runner_with(PrecheckReference, run_dir)
    runner.run(0)  # initial reference calls launch successfully
    engine.fail_precheck = True
    with pytest.raises(EngineError, match="precheck"):
        runner.run(1)  # the next reference call fails before any launch
    log = events(run_dir)
    reference = summarize_tasks(log)["reference"]
    successful_tasks = [e for e in log
                        if e["type"] == TASK and e["status"] == "success"
                        and e.get("operation") == "reference"]
    failed_tasks = [e for e in log
                    if e["type"] == TASK and e["status"] == "failed"
                    and e.get("operation") == "reference"]
    assert len(failed_tasks) == 1  # the logical failure stays visible
    assert reference["logical_requests"] == \
        len(successful_tasks) + len(failed_tasks)
    # The failed task never launched: only the successful tasks executed.
    assert reference["actual_executions"] == len(successful_tasks)
    assert reference["failed_attempts"] == 0


def test_simple_engine_gets_one_attempt_per_call(tmp_path):
    runner, _, engine, _ = world(tmp_path / "simple", p=1.0)
    runner.run(0)
    engine.fail_next = True
    with pytest.raises(EngineError, match="deliberate"):
        runner.run(1)  # the checked evaluation's reference call fails
    reference = summarize_tasks(events(tmp_path / "simple"))["reference"]
    # A backend without internal launches records exactly one attempt per
    # call — success or failure — so the logical and physical counts agree.
    assert reference["actual_executions"] == reference["logical_requests"]
    assert reference["failed_attempts"] == 1
    assert reference["successful_executions"] == \
        reference["logical_requests"] - 1


def test_replay_after_crash_fabricates_no_executions(tmp_path):
    stopped, _, _, _ = world(tmp_path / "replay", stationary=True, p=1.0)
    stopped.run(0)
    stopped.run(2)  # window after the only checkpoint (interval 100)
    stopped.close()
    before = [e for e in events(tmp_path / "replay") if e["type"] == ATTEMPT]
    assert before

    from test_review_r1 import resume_world

    resumed, _, _ = resume_world(tmp_path / "replay")
    after = [e for e in events(tmp_path / "replay") if e["type"] == ATTEMPT]
    assert len(after) == len(before)  # replay recomputes nothing
    resumed.close()


def test_plain_reference_run_records_attempts_per_step(tmp_path):
    from test_review_r4 import _write_config

    from pyraimd2.config import load_config
    from pyraimd2.workflows import run_workflow

    config = load_config(_write_config(tmp_path / "plain", mode="reference",
                                       steps=3))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    log = events(result.run_dir)
    tasks = [e for e in log if e["type"] == TASK and e["purpose"] == "md"]
    attempts = [e for e in log if e["type"] == ATTEMPT]
    assert len(tasks) == 4  # initial evaluation + 3 steps
    assert len(attempts) == 4
    assert {e["request_id"] for e in attempts} == \
        {e["task_id"] for e in tasks}
    reference = inspect_run(result.run_dir)["cost"]["reference"]
    assert reference["logical_requests"] == 4
    assert reference["actual_executions"] == 4
    assert reference["failed_attempts"] == 0


def test_nested_physical_io_is_not_double_counted():
    log = [
        {"type": "task", "task_id": "t1", "operation": "reference",
         "purpose": "anchor", "status": "success", "elapsed_s": 5.0},
        {"type": "attempt", "record": "physical_attempt",
         "operation": "reference", "request_id": "t1", "attempt": 1,
         "status": "success", "elapsed_s": 4.0},
        {"type": "task", "task_id": "t1-io-1", "operation": "io",
         "purpose": "density_copy", "record": "physical_io",
         "request_id": "t1", "status": "success", "elapsed_s": 1.0},
    ]
    summary = summarize_tasks(log)
    # The task span and the nested physical-I/O copy are both inside the
    # attempt span: only the leaf attempt time is summed.
    assert summary["total_elapsed_s"] == pytest.approx(4.0)
    assert summary["reference"]["actual_executions"] == 1
    assert summary["counts"]["io"] == 1


def test_legacy_log_without_attempts_keeps_old_semantics():
    legacy = [
        {"type": "task", "task_id": "t1", "operation": "reference",
         "purpose": "anchor", "status": "success", "elapsed_s": 1.0},
        {"type": "task", "task_id": "t2", "operation": "reference",
         "purpose": "verification", "status": "failed", "elapsed_s": 0.5},
        {"type": "task", "task_id": "t3", "operation": "reference",
         "purpose": "verification", "status": "cache_hit", "elapsed_s": 0.0},
    ]
    reference = summarize_tasks(legacy)["reference"]
    assert reference["logical_requests"] == 3
    assert reference["actual_executions"] == 2
    assert reference["failed_attempts"] == 1
    assert reference["cache_hits"] == 1
