"""Review C3 regression: after an accepted candidate, the success-task log,
state_dict, artifact persist and commit event all live in the same rollback
domain (INDEPENDENT_REVIEW_040_20260909 §C3 — an OSError at the success-task
write used to stop the run with model/generation/counters inconsistent)."""

from __future__ import annotations

import pytest
from test_review_r3 import ArrayStateModel, _events, _world_with_updater

from pyraimd2.runtime.events import MODEL_UPDATE


def _fail_on_training_success(calc, exc):
    original = calc._emit_task

    def fail(*, task_id, attempt, operation, purpose, status, **kwargs):
        if operation == "training" and status == "success":
            raise exc("injected failure at the training-success record")
        return original(task_id=task_id, attempt=attempt, operation=operation,
                        purpose=purpose, status=status, **kwargs)

    calc._emit_task = fail


def _assert_parent_state(runner, model, updater):
    assert runner.calc._model_generation == 0
    assert model.k == pytest.approx(0.8)
    assert updater.n_updates == 0
    assert len(updater._pending) == 1  # the label is queued, not lost
    committed = [e for e in _events(runner.run_dir) if e["type"] == MODEL_UPDATE]
    assert committed == []


def test_training_success_log_failure_rolls_back(tmp_path):
    model = ArrayStateModel()
    runner, _, _, updater = _world_with_updater(tmp_path / "c3-log", model)
    _fail_on_training_success(runner.calc, OSError)
    with pytest.raises(OSError, match="training-success"):
        runner.run(0)
    _assert_parent_state(runner, model, updater)


def test_state_dict_failure_rolls_back(tmp_path):
    model = ArrayStateModel()
    runner, _, _, updater = _world_with_updater(tmp_path / "c3-state", model)
    original = model.state_dict

    def fail_state():
        raise OSError("injected state_dict failure")

    model.state_dict = fail_state
    with pytest.raises(OSError, match="state_dict"):
        runner.run(0)
    model.state_dict = original
    _assert_parent_state(runner, model, updater)


def test_commit_event_failure_rolls_back(tmp_path):
    model = ArrayStateModel()
    runner, _, _, updater = _world_with_updater(tmp_path / "c3-commit", model)
    log = runner.calc._event_log
    original = log.append_once

    def fail_commit(key, event_type, payload):
        if event_type == MODEL_UPDATE:
            raise OSError("injected failure at the update commit")
        return original(key, event_type, payload)

    log.append_once = fail_commit
    with pytest.raises(OSError, match="update commit"):
        runner.run(0)
    _assert_parent_state(runner, model, updater)
