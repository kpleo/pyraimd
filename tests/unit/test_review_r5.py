"""Review R5 regression: plain-MD failed logical requests get exactly one
task record with the same id as the attempt's request_id (0.4.2 plan §R5)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_review_r4 import _write_config

from pyraimd2.config import load_config
from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import ATTEMPT, TASK
from pyraimd2.workflows import md as md_module
from pyraimd2.workflows import run_workflow


class FlakyReference:
    """Reference that fails on chosen compute calls (1-based)."""

    name = "flaky-reference"
    fingerprint = "flaky-reference:1"

    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = set(fail_on)

    def compute(self, atoms):
        self.calls += 1
        if self.calls in self.fail_on:
            raise EngineError(f"deliberate failure at call {self.calls}")
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * 1.0 * x**2)), -1.0 * x,
                            None, 0.0)


def _run(tmp_path, engine):
    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: engine)
    try:
        config = load_config(_write_config(tmp_path, mode="reference",
                                           steps=2))
        with pytest.raises(EngineError, match="deliberate"):
            run_workflow(config, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()
    return [json.loads(line) for line in
            (config.run.directory / "events.jsonl").read_text().splitlines()]


def test_r5_initial_failure_is_one_logical_one_attempt(tmp_path):
    events = _run(tmp_path / "r5-first", FlakyReference(fail_on={1}))
    tasks = [e for e in events if e["type"] == TASK]
    attempts = [e for e in events if e["type"] == ATTEMPT]
    assert len(tasks) == 1 and tasks[0]["status"] == "failed"
    assert len(attempts) == 1 and attempts[0]["status"] == "failed"
    assert attempts[0]["request_id"] == tasks[0]["task_id"]
    reference = summarize_tasks(events)["reference"]
    assert (reference["logical_requests"], reference["actual_executions"],
            reference["failed_attempts"]) == (1, 1, 1)
    # The run termination records only the run failure — no extra request.
    assert [e["type"] for e in events].count("task") == 1


def test_r5_second_step_failure_keeps_ids_paired(tmp_path):
    events = _run(tmp_path / "r5-second", FlakyReference(fail_on={2}))
    tasks = [e for e in events if e["type"] == TASK]
    attempts = [e for e in events if e["type"] == ATTEMPT]
    assert len(tasks) == 2
    assert [t["status"] for t in tasks] == ["success", "failed"]
    assert len(attempts) == 2
    # Every attempt pairs with exactly one real logical request by id.
    assert {a["request_id"] for a in attempts} == \
        {t["task_id"] for t in tasks}
    for attempt in attempts:
        assert sum(1 for t in tasks if t["task_id"] == attempt["request_id"]) == 1
    reference = summarize_tasks(events)["reference"]
    assert (reference["logical_requests"], reference["actual_executions"],
            reference["failed_attempts"]) == (2, 2, 1)
