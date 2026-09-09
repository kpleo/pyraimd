"""Review C1 regression: resume takes positions, driving forces and
re-anchor labels from the same verified committed row — never from an
orphan row at the same step (counterexample in
INDEPENDENT_REVIEW_040_20260909 §C1: boundary forces off by 9.87e-7)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from test_review_r1 import Model, _direction

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.runtime.events import EVALUATION_COMMITTED, EventLog
from pyraimd2.store import Store


class DriftingReference:
    """No fingerprint (label cache off); each physical execution of the same
    geometry returns a slightly different label, so an orphan row and its
    committed re-execution are numerically distinguishable."""

    name = "drifting-reference"
    fingerprint = None

    def __init__(self, k=1.2):
        self.k = k
        self.attempts = 0

    def compute(self, atoms):
        self.attempts += 1
        x = atoms.positions
        drift = 1e-6 * self.attempts
        return EngineResult(float(np.sum(0.5 * self.k * x**2)) + drift,
                            -self.k * x - drift, None, 0.0)


def make_world(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    engine = DriftingReference()
    runner = EnergeticRunner(
        atoms, Model(), engine, Store(run_dir / "trajectory.db"), "run",
        run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=100, direction=None,
        force_budget=0.01, timestep_fs=0.1, time_cap_fs=2.0,
        check_probability=0.0, check_seed=2)
    # No probe directions available: every evaluation takes the reference
    # route, so the driving payload IS the (drifting) engine label — an
    # orphan row and its re-execution are numerically distinguishable.
    runner.calc._directions = lambda atoms: None
    return runner, engine


def fail_commit_of(runner, evaluation_id):
    log = runner.calc._event_log
    original = log.append_once

    def fail_commit(key, event_type, payload):
        if (event_type == EVALUATION_COMMITTED
                and payload["context"]["evaluation_id"] == evaluation_id):
            raise RuntimeError("injected crash between db write and commit")
        return original(key, event_type, payload)

    log.append_once = fail_commit


def resume(run_dir):
    model = Model()
    runner = EnergeticRunner.resume(
        run_dir, model, DriftingReference(), updater=None,
        direction=_direction, event_log_force=True,
        checkpoint_interval_steps=100)
    return runner


def test_resume_boundary_forces_come_from_the_committed_row(tmp_path):
    run_dir = tmp_path / "c1"
    runner, _engine = make_world(run_dir)
    runner.run(0)  # initial evaluation committed (row 1)
    fail_commit_of(runner, 1)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(1)  # orphan row 2 (attempt-2 label); commit lost
    del runner

    resumed1 = resume(run_dir)
    resumed1.run(1)  # re-executes: committed row 3 (attempt-3 label)
    resumed1.close()

    store = Store(run_dir / "trajectory.db")
    step_rows = sorted(store._db.select(run_id="run", step=0),
                       key=lambda r: int(r.id))
    # Row 1 is the initial evaluation (step -1); step 0 holds the orphan
    # row 2 (attempt-2 label) and the committed re-execution row 3.
    assert [int(r.id) for r in step_rows] == [2, 3]
    committed = store.committed_row(resumed1.calc._event_log, "run", 1)
    assert int(committed.id) == 3
    orphan, committed_forces = (np.asarray(r.data["driving"]["forces"])
                                for r in (step_rows[0], step_rows[1]))
    assert not np.allclose(orphan, committed_forces, rtol=0, atol=1e-12)

    # The resume boundary must rebuild from the committed row only.
    resumed2 = resume(run_dir)
    boundary_forces = np.asarray(resumed2.calc.results["forces"], dtype=float)
    np.testing.assert_allclose(boundary_forces, committed_forces,
                               rtol=0, atol=1e-15)
    assert not np.allclose(boundary_forces, orphan, rtol=0, atol=1e-12)
    np.testing.assert_allclose(resumed2.atoms.positions,
                               committed.toatoms().positions, rtol=0, atol=1e-15)
    resumed2.close()
