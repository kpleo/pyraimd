"""Review R6 regression: restored counters apply only to calibrations that
completed BEFORE the proposal — a reference-route calibration completed
later, inside _finish, must not be overwritten by the proposal's older
frozen counters (0.4.2 plan §R6; continuous segments [1,2,3], resumed was
[1,2,2])."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from test_review_r1 import Model, Reference, _direction

from pyraimd2.loop import EnergeticRunner
from pyraimd2.runtime.events import EVALUATION_COMMITTED, EventLog
from pyraimd2.store import Store


def _make_world(run_dir):
    """Single-atom energetic run WITHOUT an updater and with a time cap
    that sends every evaluation after the first to the reference route —
    the review's C4-residual fixture."""
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    runner = EnergeticRunner(
        atoms, Model(), Reference(), Store(run_dir / "trajectory.db"), "run",
        run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=100, direction=_direction,
        force_budget=0.08, timestep_fs=0.1, time_cap_fs=0.001,
        check_probability=0.0, check_seed=2)
    return runner


def _resume(run_dir):
    return EnergeticRunner.resume(
        run_dir, Model(), Reference(), direction=_direction,
        event_log_force=True, checkpoint_interval_steps=100)


def _crash_at_commit(runner, evaluation_id):
    log = runner.calc._event_log
    original = log.append_once

    def fail_commit(key, event_type, payload):
        if (event_type == EVALUATION_COMMITTED
                and payload["context"]["evaluation_id"] == evaluation_id):
            raise RuntimeError("injected crash between db write and commit")
        return original(key, event_type, payload)

    log.append_once = fail_commit


def _calib(runner):
    calc = runner.calc
    return (calc._segment, calc.n_calibrations,
            None if calc._anchor is None else calc._anchor.segment)


def _row_segments(run_dir):
    import json

    store = Store(run_dir / "trajectory.db")
    events = [json.loads(line) for line in
              (run_dir / "events.jsonl").read_text().splitlines()]
    segments = []
    for _event, row in store.iter_committed(events, "run"):
        metadata = row.data.get("metadata") or {}
        anchor = metadata.get("new_anchor") or {}
        if anchor.get("segment_id") is not None:
            segments.append(int(anchor["segment_id"]))
    return segments


def test_r6_reference_route_calibration_is_not_overwritten(tmp_path):
    run_dir = tmp_path / "r6"
    runner = _make_world(run_dir)
    runner.run(0)  # initial evaluation + initial checkpoint
    _crash_at_commit(runner, 1)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(1)
    runner.close()

    resumed = _resume(run_dir)
    resumed.run(1)  # complete the crashed evaluation: calibration completes
    assert _calib(resumed) == (2, 2, 2)
    resumed.run(1)  # next step calibrates again
    assert _calib(resumed) == (3, 3, 3)
    resumed.close()
    assert _row_segments(run_dir) == [1, 2, 3]

    # Resuming again must not re-apply or double the calibration counts:
    # counters and the restored anchor match the continuous run exactly.
    again = _resume(run_dir)
    assert _calib(again) == (3, 3, 3)
    again.close()

    control_dir = tmp_path / "r6-control"
    control = _make_world(control_dir)
    control.run(0)
    control.run(2)
    control.close()
    assert _calib(control) == (3, 3, 3)
    np.testing.assert_allclose(resumed.atoms.positions,
                               control.atoms.positions, rtol=0, atol=1e-12)
    assert _row_segments(control_dir) == [1, 2, 3]
