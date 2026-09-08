"""Small analytic checks of routing, work, causal updates and failure handling."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticCalculator, EnergeticRunner
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction


class Harmonic:
    def __init__(self, k=0.8):
        self.k = k

    def predict(self, atoms):
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None, np.full(len(atoms), np.nan))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2, quartic=0.0):
        self.k, self.quartic = k, quartic
        self.attempts = 0
        self.fail_on = set()

    def compute(self, atoms):
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise EngineError("deliberate reference failure")
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2 + 0.25 * self.quartic * x**4)),
                            -self.k * x - self.quartic * x**3, None, 0.0)


def setup(tmp_path, *, base=0.8, reference=None, position=0.2, **kwargs):
    atoms = Atoms("H", positions=[[position, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(tmp_path / "run.db")
    model = Harmonic(base)
    engine = Reference() if reference is None else reference
    options = dict(force_budget=0.1, timestep_fs=0.1, check_probability=1, time_cap_fs=1.0)
    options.update(kwargs)
    calc = EnergeticCalculator(model, engine, store, "run", **options)
    atoms.calc = calc
    return atoms, calc, store, model, engine


def rows(store):
    return list(store._db.select(run_id="run"))


@pytest.mark.parametrize("base", [0.8, 1.6])
def test_signed_work_and_frozen_corrected_driving_label(tmp_path, base):
    atoms, calc, store, model, engine = setup(tmp_path, base=base)
    initial = atoms.get_forces()
    np.testing.assert_allclose(initial, [[-0.24, 0, 0]])
    assert calc.reference_calls == {"anchor": 1, "probe": 4, "check": 0}
    calibration = rows(store)[0].data["metadata"]["new_anchor"]["calibration"]
    assert calibration["force_call_count"] == 5
    assert calibration["responses"][0]["coefficient"] == pytest.approx(1 / (1.2 - base))
    atoms.positions[0, 0] = 0.21
    driving = atoms.get_forces()
    expected = -base * 0.21 + (base - 1.2) * 0.2
    assert driving[0, 0] == pytest.approx(expected)
    row = rows(store)[1]
    assert row.route == "ml" and row.data["engine"] is not None
    assert row.data["surrogate"]["forces"][0][0] == pytest.approx(-base * 0.21)
    np.testing.assert_array_equal(store.driving_label("run", 0)[1], driving)
    metadata = row.data["metadata"]
    work = 0.5 * (1.2 - base) * 0.01**2
    assert metadata["observed"]["endpoint_work_eV"] == pytest.approx(work)
    assert metadata["forecasts"][0]["predicted_work_eV"] == pytest.approx(work)
    assert metadata["observed"]["observed_coefficient_A2_eV"] == pytest.approx(1 / (1.2 - base))
    assert metadata["verification"]["accepted_count"] == 1
    assert metadata["verification"]["detected_count"] == 0
    assert atoms.get_potential_energy() == pytest.approx(row.data["engine"]["energy"] - work)
    assert calc.n_reference == engine.attempts == 6


@pytest.mark.parametrize("position,reason", [(0.6, "budget"), (0.21, "time")])
def test_failed_forecast_references_and_records_old_segment_work(tmp_path, position, reason):
    atoms, calc, store, _, _ = setup(tmp_path, force_budget=0.05,
                                    time_cap_fs=0.05 if reason == "time" else 1)
    atoms.get_forces()
    atoms.positions[0, 0] = position
    np.testing.assert_allclose(atoms.get_forces(), [[-1.2 * position, 0, 0]])
    metadata = rows(store)[1].data["metadata"]
    assert rows(store)[1].route == "dft"
    assert not metadata["accepted"]
    assert metadata["segment_id"] == metadata["observed"]["segment_id"] == 1
    assert metadata["new_anchor"]["segment_id"] == 2
    assert metadata["observed"]["endpoint_work_eV"] == pytest.approx(0.2 * (position - 0.2)**2)
    assert calc.reference_calls == {"anchor": 2, "probe": 8, "check": 0}


def test_empirical_miss_checks_frozen_force_then_references_next_step(tmp_path):
    atoms, calc, store, _, _ = setup(tmp_path, position=0, reference=Reference(quartic=1000),
                                    probe_steps=(0.01, 0.02), force_budget=3.0)
    atoms.get_forces()
    atoms.positions[0, 0] = 0.2
    frozen = atoms.get_forces()
    assert frozen[0, 0] == pytest.approx(-0.16)
    row = rows(store)[1]
    assert row.route == "ml"
    assert row.data["engine"]["forces"][0][0] == pytest.approx(-8.24)
    assert row.data["metadata"]["violation"] is True
    assert calc.verification.accepted_count == calc.verification.detected_count == 1
    atoms.positions[0, 0] = 0.201
    atoms.get_forces()
    assert rows(store)[2].route == "dft"
    assert rows(store)[2].data["reason"] == "previous_independent_check_violation"
    assert calc.n_accepted == calc.n_violations == 1


def test_callback_updates_then_calibrates_new_model_without_relabeling_origin(tmp_path):
    atoms, calc, store, model, engine = setup(tmp_path, check_probability=0)
    calls = []

    def update(observation):
        calls.append(observation.step)
        # The stored driving arrays already exist and are independently owned.
        assert len(rows(store)) == 1
        observation.label.forces[:] = 999
        observation.prediction.forces[:] = 888
        model.k = 0.9

    calc.on_label = update
    initial = atoms.get_forces()
    np.testing.assert_allclose(initial, [[-0.24, 0, 0]])
    assert engine.attempts == 1  # callback precedes all calibration probes
    atoms.positions[0, 0] = 0.21
    np.testing.assert_allclose(atoms.get_forces(), [[-0.249, 0, 0]])
    assert engine.attempts == 5  # old reference label + four new-model probes
    assert rows(store)[1].route == "ml"
    record = rows(store)[1].data["metadata"]["calibration_after_previous_label"]
    assert record["reference_origin_label_reused"] is True
    assert record["anchor"]["calibration"]["responses"][0]["coefficient"] == pytest.approx(1 / 0.3)
    np.testing.assert_allclose(store.driving_label("run", -1)[1], initial)
    assert calls == [-1]


def test_checked_callback_invalidates_model_before_next_forecast(tmp_path):
    atoms, calc, store, model, engine = setup(tmp_path)

    def update(observation):
        if observation.step == 0:
            model.k = 0.9
            return True
        return False

    calc.on_label = update
    atoms.get_forces()
    atoms.positions[0, 0] = 0.21
    first = atoms.get_forces()
    assert first[0, 0] == pytest.approx(-0.248)
    atoms.positions[0, 0] = 0.22
    second = atoms.get_forces()
    assert second[0, 0] == pytest.approx(-0.261)
    assert calc.verification.accepted_count == 2
    assert engine.attempts == 11  # one origin, eight probes, two checks
    assert rows(store)[2].data["metadata"]["calibration_after_previous_label"] is not None


def test_failed_probe_retry_counts_successes_and_never_caches_invalid_results(tmp_path):
    engine = Reference()
    engine.fail_on = {3}
    atoms, calc, store, _, _ = setup(tmp_path, reference=engine)
    with pytest.raises(EngineError, match="deliberate"):
        atoms.get_forces()
    assert calc.results == {} and rows(store) == []
    assert calc.n_reference == 2 and calc.n_evaluations == 0
    np.testing.assert_allclose(atoms.get_forces(), [[-0.24, 0, 0]])
    assert calc.reference_calls == {"anchor": 1, "probe": 5, "check": 0}
    assert rows(store)[0].data["metadata"]["reference_calls_this_evaluation"]["probe"] == 5


def test_failed_check_retries_same_decision_without_redrawing(tmp_path):
    atoms, calc, store, _, engine = setup(tmp_path, check_probability=0.5, check_seed=2)
    atoms.get_forces()
    engine.fail_on.add(6)
    atoms.positions[0, 0] = 0.21
    with pytest.raises(EngineError):
        atoms.get_forces()
    original_draw = calc._pending.draw
    assert calc._pending.checked and calc.results == {}
    assert calc.verification.accepted_count == 0 and len(rows(store)) == 1
    force = atoms.get_forces()
    assert force[0, 0] == pytest.approx(-0.248)
    assert rows(store)[1].data["metadata"]["check_draw"] == original_draw
    assert calc.verification.accepted_count == 1


@pytest.mark.parametrize("mutation", ["cell", "identity", "mass", "nonfinite"])
def test_invalid_atomic_changes_rejected(tmp_path, mutation):
    atoms, calc, store, _, _ = setup(tmp_path)
    atoms.get_forces()
    if mutation == "cell":
        atoms.set_cell([2, 2, 2])
    elif mutation == "identity":
        atoms.numbers[0] = 2
    elif mutation == "mass":
        atoms.set_masses([2.0])
        atoms.positions[0, 0] += 0.01  # ASE's standard cache ignores masses
    else:
        atoms.positions[0, 0] = np.nan
    with pytest.raises(ValueError):
        atoms.get_forces()
    assert calc.results == {} and len(rows(store)) == 1


def test_nonfinite_reference_cannot_be_success_cached(tmp_path):
    class BrokenReference(Reference):
        def compute(self, atoms):
            if self.attempts == 0:
                self.attempts += 1
                return EngineResult(float("nan"), np.zeros((1, 3)), None, 0.0)
            return super().compute(atoms)

    atoms, calc, store, _, _ = setup(tmp_path, reference=BrokenReference())
    with pytest.raises(EngineError, match="finite"):
        atoms.get_forces()
    assert calc.n_reference == 0 and rows(store) == []
    np.testing.assert_allclose(atoms.get_forces(), [[-0.24, 0, 0]])
    assert calc.n_reference == 5


def test_multiple_directions_use_full_configuration_norm_and_cost(tmp_path):
    atoms, calc, store, _, _ = setup(
        tmp_path, direction=lambda atoms: np.array([[[2, 0, 0]], [[0, 3, 0]]]))
    atoms.get_forces()
    calibration = rows(store)[0].data["metadata"]["new_anchor"]["calibration"]
    assert calibration["force_call_count"] == calc.n_reference == 9
    for response in calibration["responses"]:
        assert np.linalg.norm(response["direction"]) == pytest.approx(1)


def test_zero_motion_advances_clock_with_reference_only(tmp_path):
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.set_velocities([[0, 0, 0]])
    store = Store(tmp_path / "run.db")
    runner = EnergeticRunner(atoms, Harmonic(), Reference(), store, "run",
                             force_budget=0.1, timestep_fs=0.1, check_probability=0)
    summary = runner.run(3)
    assert summary.n_evaluations == summary.n_reference == summary.n_anchor == 4
    assert summary.n_probe == summary.n_accepted == 0
    assert [row.step for row in rows(store)] == [-1, 0, 1, 2]
    np.testing.assert_allclose([row.data["metadata"]["time_fs"] for row in rows(store)],
                               [0, 0.1, 0.2, 0.3])
    assert runner.run(2).n_evaluations == 2


def test_zero_velocity_can_start_moving_and_calibrate(tmp_path):
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0, 0, 0]])
    store = Store(tmp_path / "run.db")
    runner = EnergeticRunner(atoms, Harmonic(), Reference(), store, "run",
                             force_budget=0.1, timestep_fs=0.1, check_probability=0)
    summary = runner.run(2)
    assert rows(store)[0].data["metadata"]["new_anchor"] is None
    assert rows(store)[1].data["metadata"]["new_anchor"] is not None
    assert summary.n_accepted == 1 and summary.n_probe == 4


@pytest.mark.parametrize("update_to_exact", [False, True])
def test_zero_response_keeps_reference_route_without_inventing_coefficient(tmp_path, update_to_exact):
    atoms, calc, store, model, _ = setup(tmp_path, base=0.8 if update_to_exact else 1.2)
    if update_to_exact:
        def update(observation):
            model.k = 1.2
        calc.on_label = update
    atoms.get_forces()
    atoms.positions[0, 0] = 0.21
    np.testing.assert_allclose(atoms.get_forces(), [[-0.252, 0, 0]])
    assert [row.route for row in rows(store)] == ["dft", "dft"]
    assert calc.n_accepted == calc.n_calibrations == 0
    assert all(row.data["metadata"]["new_anchor"] is None for row in rows(store))


def test_zero_time_cap_rejected_before_any_reference_call(tmp_path):
    with pytest.raises(ValueError, match="time_cap_fs"):
        setup(tmp_path, time_cap_fs=0)


def test_explicit_restart_rejected_without_changing_legacy_runner(tmp_path):
    atoms, _, store, model, engine = setup(tmp_path)
    atoms.get_forces()
    with pytest.raises(ValueError, match="restart"):
        EnergeticCalculator(model, engine, store, "run", force_budget=0.1)
    # Resume exists but requires a valid complete-step checkpoint: a bare
    # Store trajectory is not one and is refused, not silently degraded.
    from pyraimd2.runtime import ResumeError

    with pytest.raises(ResumeError, match="checkpoint"):
        EnergeticRunner.resume(tmp_path / "no-such-run", model, engine)


def test_callback_failure_cannot_repeat_a_stored_event(tmp_path):
    atoms, calc, store, _, _ = setup(tmp_path)

    def fail(observation):
        raise RuntimeError("update failed")

    calc.on_label = fail
    with pytest.raises(RuntimeError, match="update failed"):
        atoms.get_forces()
    assert len(rows(store)) == 1 and calc.results == {}
    with pytest.raises(RuntimeError, match="callback failed"):
        atoms.get_forces()
    assert len(rows(store)) == 1


def test_mass_change_rejected_even_when_results_cached(tmp_path):
    """Masses are not part of ASE's cache-invalidation state: changing them
    alone must still be caught before a cached property is served."""
    atoms, _, _, _, _ = setup(tmp_path)
    atoms.get_forces()  # caches energy/forces for this state
    atoms.set_masses([2.0])
    with pytest.raises(ValueError, match="mass"):
        atoms.get_forces()


def test_constraint_added_after_caching_rejected(tmp_path):
    """Adding a constraint must not silently reuse cached unconstrained
    forces; the energetic calculator only supports unconstrained dynamics."""
    from ase.constraints import FixAtoms

    atoms, _, _, _, _ = setup(tmp_path)
    atoms.get_forces()
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="unconstrained"):
        atoms.get_forces()
