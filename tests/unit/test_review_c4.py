"""Review C4 regression: a recalibration that completed before an
uncommitted proposal (segment, calibration count, anchor and deferred
state) is restored exactly once on resume — not lost, not doubled
(INDEPENDENT_REVIEW_040_20260909 §C4: continuous 3/3 vs resumed 2/2)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from test_review_r1 import Reference, _direction, events
from test_review_r3 import ArrayStateModel

from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.runtime.events import EVALUATION_COMMITTED, EventLog
from pyraimd2.store import Store


def _make_world(run_dir, update_n):
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    model = ArrayStateModel()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=update_n,
                                                 guard_size=1))
    runner = EnergeticRunner(
        atoms, model, Reference(), Store(run_dir / "trajectory.db"), "run",
        on_label=updater, run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=100, direction=_direction,
        force_budget=0.08, timestep_fs=0.1, time_cap_fs=2.0,
        check_probability=1.0, check_seed=2)
    return runner


def _resume(run_dir, update_n):
    model = ArrayStateModel()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=update_n,
                                                 guard_size=1))
    return EnergeticRunner.resume(
        run_dir, model, Reference(), updater=updater, direction=_direction,
        event_log_force=True, checkpoint_interval_steps=100)


def _fail_commit_of(runner, evaluation_id):
    log = runner.calc._event_log
    original = log.append_once

    def fail_commit(key, event_type, payload):
        if (event_type == EVALUATION_COMMITTED
                and payload["context"]["evaluation_id"] == evaluation_id):
            raise RuntimeError("injected crash after the proposal persisted")
        return original(key, event_type, payload)

    log.append_once = fail_commit


def _state(runner):
    calc = runner.calc
    return {"segment": calc._segment,
            "n_calibrations": calc.n_calibrations,
            "generation": calc._model_generation,
            "positions": runner.atoms.positions.copy(),
            "anchor_segment": (None if calc._anchor is None
                               else calc._anchor.segment)}


def _commits(run_dir):
    return [(e["context"]["evaluation_id"], e["route"], e["checked"])
            for e in events(run_dir) if e["type"] == EVALUATION_COMMITTED]


@pytest.mark.parametrize("update_n", (1, 2))
def test_recalibrated_uncommitted_proposal_restores_segment_state(tmp_path,
                                                                  update_n):
    # The model updates once per n labels; the evaluation after an update
    # recalibrates.  Crash it after the (recalibrated) proposal persists but
    # before the commit lands.
    crash_at = update_n
    run_dir = tmp_path / f"crashed-{update_n}"
    runner = _make_world(run_dir, update_n)
    runner.run(0)
    if crash_at > 1:
        runner.run(crash_at - 1)
    _fail_commit_of(runner, crash_at)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(crash_at)
    runner.close()

    resumed = _resume(run_dir, update_n)
    resumed.run(2)
    resumed.close()

    control_dir = tmp_path / f"control-{update_n}"
    control = _make_world(control_dir, update_n)
    control.run(0)
    control.run(crash_at + 1)
    control.close()

    crashed_state, control_state = _state(resumed), _state(control)
    for key in ("segment", "n_calibrations", "generation", "anchor_segment"):
        assert crashed_state[key] == control_state[key], key
    np.testing.assert_allclose(crashed_state["positions"],
                               control_state["positions"], rtol=0, atol=1e-12)
    assert _commits(run_dir) == _commits(control_dir)  # same check stream


@pytest.mark.parametrize("update_n", (1, 2))
def test_second_resume_after_recalibration_commit_counts_once(tmp_path,
                                                              update_n):
    # The recalibration's counters are applied from the rebuilt proposal on
    # the first resume; after the evaluation commits, its metadata carries
    # the calibration — a second resume must land on the same totals.
    crash_at = update_n
    run_dir = tmp_path / f"twice-{update_n}"
    runner = _make_world(run_dir, update_n)
    runner.run(0)
    if crash_at > 1:
        runner.run(crash_at - 1)
    _fail_commit_of(runner, crash_at)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(crash_at)
    runner.close()
    first = _resume(run_dir, update_n)
    first.run(2)
    first.close()
    del first

    second = _resume(run_dir, update_n)
    second.run(1)
    second.close()

    control_dir = tmp_path / f"twice-control-{update_n}"
    control = _make_world(control_dir, update_n)
    control.run(0)
    control.run(crash_at + 2)
    control.close()
    for key in ("segment", "n_calibrations", "generation", "anchor_segment"):
        assert _state(second)[key] == _state(control)[key], key
    assert _commits(run_dir) == _commits(control_dir)
