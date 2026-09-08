"""WP07 FixAtoms in the energetic runner and unwrapped PBC coordinates."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from pyraimd2.energetics import residual_work
from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.workflows.export import frame_from_row


class Harmonic:
    def __init__(self, k=0.8):
        self.k = k

    def predict(self, atoms):
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None,
                                   np.full(len(atoms), np.nan))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2):
        self.k = k

    def compute(self, atoms):
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2)), -self.k * x,
                            None, 0.0)


POLICY = {"force_budget": 0.08, "timestep_fs": 0.1, "check_probability": 0,
          "time_cap_fs": 5.0}


def _cluster(fixed=(0, 1)):
    atoms = Atoms("H4", positions=[[0.20, 0.0, 0.0], [0.24, 0.0, 0.0],
                                   [0.21, 0.02, 0.0], [0.19, -0.02, 0.0]])
    velocities = np.zeros((4, 3))
    velocities[2:, 0] = [0.05, -0.05]
    atoms.set_velocities(velocities)
    atoms.set_constraint(FixAtoms(indices=list(fixed)))
    return atoms


def _runner(tmp_path, atoms, name="run", **overrides):
    store = Store(tmp_path / f"{name}.db")
    options = dict(POLICY)
    options.update(overrides)
    runner = EnergeticRunner(atoms, Harmonic(), Reference(), store, name,
                             **options)
    return runner, store


def rows(store, run_id="run"):
    return sorted(store._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def test_fixed_layer_never_moves_and_driving_is_projected(tmp_path):
    atoms = _cluster()
    initial = atoms.positions.copy()
    runner, store = _runner(tmp_path, atoms)
    summary = runner.run(10)
    assert summary.n_evaluations == 11
    for row in rows(store):
        frame = row.toatoms()
        np.testing.assert_array_equal(frame.positions[:2], initial[:2])
        metadata = row.data["metadata"]
        record = metadata.get("constraint")
        assert record is not None and record["indices"] == [0, 1]
        assert record["n_free"] == 2 and record["n_fixed"] == 2
        # raw physical forces kept; the driving payload is projected
        assert record["raw_forces_eV_A"][0] != [0.0, 0.0, 0.0]
        driving = np.asarray(row.data["driving"]["forces"], dtype=float)
        np.testing.assert_array_equal(driving[:2], np.zeros((2, 3)))
        assert record.get("max_fixed_displacement_A", 0.0) == 0.0
    # the free atoms really moved
    final = rows(store)[-1].toatoms()
    assert not np.allclose(final.positions[2:], initial[2:])


def test_force_metric_selects_what_the_budget_controls(tmp_path):
    # Reference couples fixed atom 0 to both free atoms; the surrogate has
    # no coupling.  The fixed-DOF residual therefore grows at twice the
    # free-atom rate (gamma*(d2+d3) vs gamma*d_i), so a budget between the
    # two admits the forecast in active mode but trips in all-atom mode.
    class Surrogate:
        def __init__(self, k=1.2):
            self.k = k

        def predict(self, atoms):
            x = atoms.positions
            return SurrogatePrediction(0.5 * self.k * float(np.sum(x**2)),
                                       -self.k * x, None,
                                       np.full(len(atoms), np.nan))

    class Reference:
        name = "coupled-reference"

        def __init__(self, k=1.2, gamma=0.3):
            self.k, self.gamma = k, gamma

        def compute(self, atoms):
            x = atoms.positions
            forces = -self.k * x.copy()
            energy = 0.5 * self.k * float(np.sum(x**2))
            for free in (2, 3):
                bond = x[0] - x[free]
                forces[0] -= self.gamma * bond
                forces[free] += self.gamma * bond
                energy += 0.5 * self.gamma * float(np.sum(bond**2))
            return EngineResult(float(energy), forces, None, 0.0)

    def build(metric):
        atoms = Atoms("H4", positions=[[0.30, 0.0, 0.0], [0.34, 0.0, 0.0],
                                       [0.20, 0.0, 0.0], [0.24, 0.0, 0.0]])
        velocities = np.zeros((4, 3))
        velocities[2:, 0] = [5.0, 5.0]
        atoms.set_velocities(velocities)
        atoms.set_constraint(FixAtoms(indices=[0, 1]))
        store = Store(tmp_path / f"{metric}.db")
        runner = EnergeticRunner(atoms, Surrogate(), Reference(), store,
                                 f"metric-{metric}", force_budget=0.02,
                                 timestep_fs=0.1, check_probability=1,
                                 time_cap_fs=5.0, force_metric=metric)
        return runner, store

    runner_active, store_active = build("active_dofs_max_atom")
    summary_active = runner_active.run(3)
    runner_all, store_all = build("all_atoms_max_atom")
    runner_all.run(3)

    # active: the free-coordinate error (≈0.015) stays inside the 0.02
    # budget — accepted, checked, no violation.
    assert summary_active.n_accepted >= 1
    active_rows = rows(store_active, "metric-active_dofs_max_atom")
    assert any(row.key_value_pairs["route"] == "ml" for row in active_rows)
    accepted = next(row for row in active_rows
                    if row.key_value_pairs["route"] == "ml")
    observed = accepted.data["metadata"]["observed"]
    assert observed["force_metric"] == "active_dofs_max_atom"
    assert observed["max_force_error_eV_A"] == pytest.approx(0.015, abs=2e-3)
    assert observed["force_budget_exceeded"] is False
    assert len(observed["residual_eV_A"]) == 4  # raw diagnostic, all atoms

    # all-atom: the fixed-DOF residual (≈0.03) trips the same budget at the
    # forecast — every step after the anchor is a reference.
    all_rows = rows(store_all, "metric-all_atoms_max_atom")
    assert [row.key_value_pairs["route"] for row in all_rows] == ["dft"] * len(all_rows)
    observed_all = all_rows[-1].data["metadata"]["observed"]
    assert observed_all["force_metric"] == "all_atoms_max_atom"
    assert observed_all["max_force_error_eV_A"] == pytest.approx(0.03, abs=5e-3)
    assert observed_all["force_budget_exceeded"] is True


def test_removing_the_constraint_restores_unprojected_behavior(tmp_path):
    atoms = _cluster()
    atoms.set_constraint()  # drop the FixAtoms set
    runner, store = _runner(tmp_path, atoms)
    runner.run(3)
    for row in rows(store):
        assert row.data["metadata"].get("constraint") is None
        driving = np.asarray(row.data["driving"]["forces"], dtype=float)
        assert np.abs(driving[:2]).sum() > 0  # no projection anywhere


def test_pbc_unwrapped_displacement_and_work_stay_continuous(tmp_path):
    # One atom drifting across the cell boundary: positions pass the edge
    # continuously (never wrapped mid-run), and the endpoint work follows
    # the same continuous displacement — no minimum-image jump.
    atoms = Atoms("H", positions=[[2.98, 0.5, 0.5]], cell=[3.0, 3.0, 3.0],
                  pbc=[True, True, True])
    atoms.set_velocities([[1.0, 0.0, 0.0]])
    runner, store = _runner(tmp_path, atoms)
    runner.run(6)
    positions = [row.toatoms().positions[0, 0] for row in rows(store)]
    assert positions[-1] > 3.0  # crossed the boundary, unwrapped
    deltas = np.diff(positions)
    assert (deltas > 0).all() and deltas.max() < 0.2  # continuous drift
    # residual work of the same physical step across the boundary equals
    # the intra-cell one (continuous displacement, no -2.97 jump)
    x0 = np.array([[2.99, 0.5, 0.5]])
    x1 = np.array([[3.02, 0.5, 0.5]])
    correction = np.zeros((1, 3))
    work_across = residual_work(x0, x1, 0.0, 0.0, 0.0, 0.0, correction)
    work_inside = residual_work(x0 - 3.0, x1 - 3.0, 0.0, 0.0, 0.0, 0.0,
                                correction)
    assert work_across == pytest.approx(work_inside)
    # export may still hand out wrapped frames on request
    wrapped = frame_from_row(rows(store)[-1], "run", force_source="driving",
                             wrap=True)
    assert 0.0 <= wrapped.positions[0, 0] < 3.0
    assert wrapped.info["coordinates"] == "wrapped"
    unwrapped = frame_from_row(rows(store)[-1], "run", force_source="driving")
    assert unwrapped.info["coordinates"] == "unwrapped"
    assert unwrapped.positions[0, 0] > 3.0
