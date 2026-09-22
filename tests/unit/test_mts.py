"""Unit and numerics tests for the experimental fixed-model MTS kernel
(pyraimd2.loop.mts) on analytic doubles — no real engines ever launch.

The two-body harmonic model is the fixed analytic case: N=2,
masses 28.085 amu, r0=(2.35,0,0) A, K=diag(1,4,16) eV/A^2,
U = 1/2 q^T K q with q = x2 - x1 - r0.  The fast double scales the
reference by c (energy and forces together, so metadata stays
consistent).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms, units

from pyraimd2.engines.base import EngineCapabilities, EngineResult
from pyraimd2.loop.mts import MtsError, run_mts
from pyraimd2.surrogate.base import SurrogateCapabilities, SurrogatePrediction

MASS = 28.085
R0 = np.array([2.35, 0.0, 0.0])
K_DIAG = np.array([1.0, 4.0, 16.0])


def harmonic_ef(positions, scale=1.0):
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    q = pos[1] - pos[0] - R0
    energy = 0.5 * float(K_DIAG @ (q ** 2))
    fq = K_DIAG * q
    forces = np.array([fq, -fq])
    return scale * energy, scale * forces


class ReferenceDouble:
    """Engine-protocol double over the analytic reference (self-counting)."""

    capabilities = EngineCapabilities(
        energy_kind="energy", force_consistent=True,
        forces_conservative=True, stress_available=False)

    def __init__(self, *, scale=1.0, fail_at_call=None):
        self.scale = scale
        self.calls = 0
        self.fail_at_call = fail_at_call
        self.positions_seen = []

    def compute(self, atoms):
        self.calls += 1
        if self.fail_at_call is not None and self.calls == self.fail_at_call:
            raise RuntimeError("injected reference failure")
        self.positions_seen.append(np.array(atoms.positions, dtype=float))
        t0 = 0.0
        e, f = harmonic_ef(atoms.positions, self.scale)
        return EngineResult(energy=e, forces=f, stress=None,
                            wall_time_s=t0, energy_kind="energy",
                            force_consistent=True)


class SurrogateDouble:
    """Surrogate-protocol double (c x reference), self-counting."""

    capabilities = SurrogateCapabilities(
        energy_kind="energy", force_consistent=True,
        forces_conservative=True, stress_available=False,
        uncertainty_available=False)

    def __init__(self, *, c=1.0, fail_at_call=None):
        self.c = c
        self.calls = 0
        self.fail_at_call = fail_at_call
        self.positions_seen = []

    def predict(self, atoms):
        self.calls += 1
        if self.fail_at_call is not None and self.calls == self.fail_at_call:
            raise RuntimeError("injected surrogate failure")
        self.positions_seen.append(np.array(atoms.positions, dtype=float))
        e, f = harmonic_ef(atoms.positions, self.c)
        return SurrogatePrediction(
            energy=e, forces=f, stress=None,
            uncertainty=np.full(len(atoms), np.nan),
            energy_kind="energy", force_consistent=True)


def demo_atoms(v0=(0.001, -0.0015, 0.002)):
    q0 = np.array([0.03, -0.02, 0.01])
    x1 = -(R0 + q0) / 2
    x2 = +(R0 + q0) / 2
    atoms = Atoms("Si2", positions=[x1, x2], masses=[MASS, MASS],
                  pbc=False)
    v = np.array([[-v0[0], -v0[1], -v0[2]], list(v0)]) / units.fs
    atoms.set_momenta(v * MASS)
    return atoms


def plain_vv(x, p, forces_fn, h_fs, n_steps, masses):
    """Independent reference velocity-Verlet (control implementation)."""
    h = h_fs * units.fs
    inv_m = 1.0 / np.asarray(masses, dtype=float)[:, None]
    f = np.asarray(forces_fn(x), dtype=float)
    for _ in range(n_steps):
        p = p + 0.5 * h * f
        x = x + h * p * inv_m
        f = np.asarray(forces_fn(x), dtype=float)
        p = p + 0.5 * h * f
    return x, p


# --- input validation -------------------------------------------------------


def test_rejects_empty_structure():
    atoms = Atoms("", positions=np.zeros((0, 3)))
    with pytest.raises(MtsError, match="non-empty"):
        run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


def test_rejects_bad_masses():
    atoms = demo_atoms()
    atoms.set_masses([MASS, 0.0])
    with pytest.raises(MtsError, match="positive masses"):
        run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


def test_rejects_missing_momenta():
    atoms = demo_atoms()
    del atoms.arrays["momenta"]
    with pytest.raises(MtsError, match="existing momenta"):
        run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


def test_rejects_nonfinite_momenta():
    atoms = demo_atoms()
    bad = atoms.get_momenta()
    bad[0, 0] = np.nan
    atoms.set_momenta(bad)
    with pytest.raises(MtsError, match="finite momenta"):
        run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


@pytest.mark.parametrize("h", [0.0, -0.5, np.nan])
def test_rejects_bad_inner_timestep(h):
    with pytest.raises(MtsError, match="inner_timestep_fs"):
        run_mts(demo_atoms(), ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=h, outer_ratio=1, n_outer_steps=1)


@pytest.mark.parametrize("m", [True, 0, 1.5])
def test_rejects_bad_outer_ratio(m):
    with pytest.raises(MtsError, match="outer_ratio"):
        run_mts(demo_atoms(), ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=m, n_outer_steps=1)


@pytest.mark.parametrize("n", [-1, True, 2.5])
def test_rejects_bad_n_outer_steps(n):
    with pytest.raises(MtsError, match="n_outer_steps"):
        run_mts(demo_atoms(), ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=n)


def test_rejects_constraints(tmp_path):
    from ase.constraints import FixAtoms
    atoms = demo_atoms()
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(MtsError, match="constraints"):
        run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


def test_rejects_same_object_backends():
    atoms = demo_atoms()
    shared = ReferenceDouble()
    with pytest.raises(MtsError, match="distinct"):
        run_mts(atoms, shared, shared, inner_timestep_fs=1.0,
                outer_ratio=1, n_outer_steps=1)


class UndeclaredDouble:
    def compute(self, atoms):
        raise AssertionError("must never be evaluated")


def test_rejects_undeclared_capabilities():
    with pytest.raises(MtsError, match="force_consistent"):
        run_mts(demo_atoms(), UndeclaredDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


class NonConservativeDouble(ReferenceDouble):
    capabilities = EngineCapabilities(
        energy_kind="energy", force_consistent=True,
        forces_conservative=False, stress_available=False)


def test_rejects_non_conservative_force():
    with pytest.raises(MtsError, match="forces_conservative"):
        run_mts(demo_atoms(), NonConservativeDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


def test_caller_atoms_never_mutated():
    atoms = demo_atoms()
    atoms.calc = "sentinel-calculator"
    x0 = atoms.positions.copy()
    p0 = atoms.get_momenta().copy()
    run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
            inner_timestep_fs=1.0, outer_ratio=2, n_outer_steps=4)
    assert np.array_equal(atoms.positions, x0)
    assert np.array_equal(atoms.get_momenta(), p0)
    assert atoms.calc == "sentinel-calculator"


# --- exact call accounting --------------------------------------------------


@pytest.mark.parametrize("m,n", [(1, 4), (2, 8), (4, 3)])
def test_exact_call_counts(m, n):
    ref, fast = ReferenceDouble(), SurrogateDouble()
    result = run_mts(demo_atoms(), ref, fast, inner_timestep_fs=1.0,
                     outer_ratio=m, n_outer_steps=n)
    assert ref.calls == n + 1 == result.reference_calls
    assert fast.calls == m * n + 1 == result.surrogate_calls
    assert len(result.boundaries) == n + 1


def test_zero_outer_steps_is_a_zero_call_noop():
    atoms = demo_atoms()
    ref, fast = ReferenceDouble(), SurrogateDouble()
    result = run_mts(atoms, ref, fast, inner_timestep_fs=1.0,
                     outer_ratio=4, n_outer_steps=0)
    assert ref.calls == 0 and fast.calls == 0
    assert result.boundaries == ()
    assert np.array_equal(result.final_positions_A, atoms.positions)
    assert np.array_equal(result.final_momenta_ase, atoms.get_momenta())


def test_no_backend_call_on_input_or_capability_errors():
    ref, fast = ReferenceDouble(), SurrogateDouble()
    with pytest.raises(MtsError):
        run_mts(demo_atoms(), ref, fast, inner_timestep_fs=0.0,
                outer_ratio=1, n_outer_steps=1)
    assert ref.calls == 0 and fast.calls == 0


# --- result-contract checks -------------------------------------------------


class BadShapeDouble(ReferenceDouble):
    def compute(self, atoms):
        self.calls += 1
        return EngineResult(energy=0.0, forces=np.zeros((3, 3)),
                            stress=None, wall_time_s=0.0,
                            energy_kind="energy", force_consistent=True)


def test_bad_force_shape_is_an_error():
    with pytest.raises(MtsError, match="shape"):
        run_mts(demo_atoms(), BadShapeDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


class NanDouble(ReferenceDouble):
    def compute(self, atoms):
        self.calls += 1
        return EngineResult(energy=np.nan,
                            forces=np.zeros((len(atoms), 3)),
                            stress=None, wall_time_s=0.0,
                            energy_kind="energy", force_consistent=True)


def test_nan_energy_is_an_error():
    with pytest.raises(MtsError, match="non-finite"):
        run_mts(demo_atoms(), NanDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


class ConflictMetaDouble(ReferenceDouble):
    def compute(self, atoms):
        self.calls += 1
        e, f = harmonic_ef(atoms.positions)
        return EngineResult(energy=e, forces=f, stress=None,
                            wall_time_s=0.0, energy_kind="energy",
                            force_consistent=False)


def test_result_metadata_conflict_is_an_error():
    with pytest.raises(MtsError, match="conflicts"):
        run_mts(demo_atoms(), ConflictMetaDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)


def test_endpoint_failure_keeps_prefix_and_propagates():
    atoms = demo_atoms()
    x0, p0 = atoms.positions.copy(), atoms.get_momenta().copy()
    ref = ReferenceDouble(fail_at_call=3)   # initial + 2 endpoints succeed
    kept = []
    with pytest.raises(RuntimeError, match="injected reference failure"):
        run_mts(atoms, ref, SurrogateDouble(), inner_timestep_fs=1.0,
                outer_ratio=2, n_outer_steps=8,
                boundary_callback=kept.append)
    # one complete outer step committed; the second never published
    assert len(kept) == 2            # initial + 1 completed outer step
    assert [b.outer_index for b in kept] == [0, 1]
    assert np.array_equal(atoms.positions, x0)
    assert np.array_equal(atoms.get_momenta(), p0)
    assert ref.calls == 3            # spent calls stay spent


def test_boundary_snapshots_are_independent():
    result = run_mts(demo_atoms(), ReferenceDouble(), SurrogateDouble(),
                     inner_timestep_fs=1.0, outer_ratio=2, n_outer_steps=4)
    first = result.boundaries[0]
    assert np.array_equal(first.positions_A, demo_atoms().positions)
    xs = [b.positions_A for b in result.boundaries]
    assert not any(x is xs[0] for x in xs[1:])
    assert len({x.ctypes.data for x in xs}) == len(xs)  # no shared buffers


# --- numerics on the analytic model -----------------------------------------


def test_m1_equals_reference_verlet_any_c():
    for c in (1.0, 0.9, 0.2):
        atoms = demo_atoms()
        result = run_mts(atoms, ReferenceDouble(), SurrogateDouble(c=c),
                         inner_timestep_fs=1.0, outer_ratio=1,
                         n_outer_steps=16)
        x_ref, p_ref = plain_vv(
            np.array(atoms.positions, float),
            np.array(atoms.get_momenta(), float),
            lambda x: harmonic_ef(x)[1], 1.0, 16, [MASS, MASS])
        assert np.allclose(result.final_positions_A, x_ref,
                           atol=1e-10, rtol=1e-10)
        assert np.allclose(result.final_momenta_ase, p_ref,
                           atol=1e-10, rtol=1e-10)


def test_c1_boundaries_equal_same_inner_step_verlet():
    atoms = demo_atoms()
    for m in (1, 2, 4):
        result = run_mts(atoms, ReferenceDouble(), SurrogateDouble(c=1.0),
                         inner_timestep_fs=0.5, outer_ratio=m,
                         n_outer_steps=8)
        x = np.array(atoms.positions, float)
        p = np.array(atoms.get_momenta(), float)
        for k in range(8):
            x, p = plain_vv(x, p, lambda xx: harmonic_ef(xx)[1], 0.5,
                            m, [MASS, MASS])
            b = result.boundaries[k + 1]
            assert np.allclose(b.positions_A, x, atol=1e-10, rtol=1e-10)
            assert np.allclose(b.momenta_ase, p, atol=1e-10, rtol=1e-10)


def test_time_reversal_returns_to_start():
    atoms = demo_atoms()
    first = run_mts(atoms, ReferenceDouble(), SurrogateDouble(c=0.9),
                    inner_timestep_fs=1.0, outer_ratio=4,
                    n_outer_steps=32)
    back = demo_atoms()
    back.positions = first.final_positions_A.copy()
    back.set_momenta(-first.final_momenta_ase)
    second = run_mts(back, ReferenceDouble(), SurrogateDouble(c=0.9),
                     inner_timestep_fs=1.0, outer_ratio=4,
                     n_outer_steps=32)
    assert np.allclose(second.final_positions_A, atoms.positions,
                       atol=1e-10)
    assert np.allclose(second.final_momenta_ase, -atoms.get_momenta(),
                       atol=1e-10)


def test_total_momentum_and_com_conservation():
    atoms = demo_atoms()
    p_total0 = atoms.get_momenta().sum(axis=0)
    com0 = (atoms.positions * MASS).sum(axis=0) / (2 * MASS)
    result = run_mts(atoms, ReferenceDouble(), SurrogateDouble(c=0.9),
                     inner_timestep_fs=0.25, outer_ratio=4,
                     n_outer_steps=16)
    for b in result.boundaries:
        assert np.allclose(b.momenta_ase.sum(axis=0), p_total0,
                           atol=1e-12)
        com = (b.positions_A * MASS).sum(axis=0) / (2 * MASS)
        v_com = p_total0 / (2 * MASS)          # zero in this fixture
        expected = com0 + v_com * b.time_fs * units.fs
        assert np.allclose(com, expected, atol=1e-10)
    assert np.all(np.isfinite(result.final_positions_A))
    assert np.all(np.isfinite(result.final_momenta_ase))


# ---------------------------------------------------------------------------
# Ledger regressions: request association, unique task identity, the
# run_start protocol marker and label validity as part of the logical
# task.

from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import EventLog


class SelfReportingDouble(ReferenceDouble):
    """Engine double that self-reports one physical attempt per launch
    (the QE-style request_id protocol), with injectable pre-launch
    failure and internal retry."""

    def __init__(self, *, fail_before_launch=False, retry_once=False):
        super().__init__()
        self.attempt_sink = None
        self.seen_ids = []
        self.launched = 0
        self.fail_before_launch = fail_before_launch
        self.retry_once = retry_once

    def compute(self, atoms, *, request_id=None):
        self.seen_ids.append(request_id)
        if self.fail_before_launch:
            raise RuntimeError("pre-launch failure: no physical execution")
        self.launched += 1
        sink = self.attempt_sink
        assert sink is not None
        n = self.launched
        if self.retry_once and n == 2:
            sink.append("attempt", {
                "record": "physical_attempt", "operation": "reference",
                "purpose": "test", "request_id": request_id, "attempt": 1,
                "status": "failed", "started_unix": 0.0,
                "elapsed_s": 0.01, "returncode": 1,
                "directory": f"call-{n}a", "source": "test"})
        sink.append("attempt", {
            "record": "physical_attempt", "operation": "reference",
            "purpose": "test", "request_id": request_id, "attempt": 1,
            "status": "success", "started_unix": 0.0, "elapsed_s": 0.01,
            "returncode": 0, "directory": f"call-{n}b", "source": "test"})
        return super().compute(atoms)


def _events_of(path):
    return [json.loads(l) for l in
            (path / "events.jsonl").read_text().splitlines()]


def test_self_reporting_backend_gets_request_id_counted_once(tmp_path):
    log = EventLog(tmp_path)
    ref = SelfReportingDouble()
    run_mts(demo_atoms(), ref, SurrogateDouble(), inner_timestep_fs=1.0,
            outer_ratio=2, n_outer_steps=1, event_log=log)
    log.close()
    assert ref.seen_ids and all(i is not None for i in ref.seen_ids)
    summary = summarize_tasks(_events_of(tmp_path))
    assert summary["reference"]["logical_requests"] == 2
    assert summary["reference"]["actual_executions"] == 2  # not 4


def test_prelaunch_failure_counts_zero_executions(tmp_path):
    log = EventLog(tmp_path)
    ref = SelfReportingDouble(fail_before_launch=True)
    with pytest.raises(RuntimeError, match="pre-launch"):
        run_mts(demo_atoms(), ref, SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1,
                event_log=log)
    log.close()
    events = _events_of(tmp_path)
    # the protocol marker precedes the first evaluation/failure
    assert events[0]["type"] == "run_start"
    assert events[0]["attempt_ledger"] == "physical_attempt_v1"
    summary = summarize_tasks(events)
    assert summary["reference"]["logical_requests"] == 1
    assert summary["reference"]["actual_executions"] == 0  # not 1


def test_repeated_default_calls_keep_unique_task_ids(tmp_path):
    log = EventLog(tmp_path)
    for _ in range(2):
        run_mts(demo_atoms(), ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=2, n_outer_steps=1,
                event_log=log)
    log.close()
    events = _events_of(tmp_path)
    task_ids = [e["task_id"] for e in events if e["type"] == "task"]
    assert len(task_ids) == len(set(task_ids))
    summary = summarize_tasks(events)
    assert summary["reference"]["actual_executions"] == 4  # not 8
    assert summary["reference"]["logical_requests"] == 4


def test_self_reporting_internal_retry_counts_each_launch(tmp_path):
    log = EventLog(tmp_path)
    ref = SelfReportingDouble(retry_once=True)
    run_mts(demo_atoms(), ref, SurrogateDouble(), inner_timestep_fs=1.0,
            outer_ratio=2, n_outer_steps=1, event_log=log)
    log.close()
    summary = summarize_tasks(_events_of(tmp_path))
    # 2 logical requests (initial + one endpoint); the second request
    # self-reported a failed launch then a successful one
    assert summary["reference"]["logical_requests"] == 2
    assert summary["reference"]["actual_executions"] == 3
    assert summary["reference"]["failed_attempts"] == 1
    assert summary["reference"]["successful_executions"] == 2


def test_invalid_label_fails_the_task_but_keeps_the_attempt(tmp_path):
    log = EventLog(tmp_path)
    with pytest.raises(MtsError, match="non-finite"):
        run_mts(demo_atoms(), NanDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1,
                event_log=log)
    log.close()
    events = _events_of(tmp_path)
    task = next(e for e in events if e["type"] == "task")
    assert task["status"] == "failed"
    assert "non-finite" in task["error"]
    attempt = next(e for e in events if e["type"] == "attempt")
    assert attempt["status"] == "success"   # the launch really happened
    summary = summarize_tasks(events)
    assert summary["reference"]["logical_requests"] == 1
    assert summary["reference"]["actual_executions"] == 1
    assert summary["reference"]["successful_executions"] == 1


# --- segment chaining (workflow resume machinery) --------------------------


def test_segment_chaining_matches_continuous_bit_for_bit():
    from pyraimd2.loop.mts import MtsLabels  # noqa: F401  (public type)
    atoms = demo_atoms()
    full = run_mts(atoms, ReferenceDouble(), SurrogateDouble(c=0.9),
                   inner_timestep_fs=1.0, outer_ratio=2, n_outer_steps=8)
    ref1, fast1 = ReferenceDouble(), SurrogateDouble(c=0.9)
    seg1 = run_mts(atoms, ref1, fast1, inner_timestep_fs=1.0,
                   outer_ratio=2, n_outer_steps=4)
    assert seg1.final_labels is not None
    back = demo_atoms()
    back.positions = seg1.final_positions_A.copy()
    back.set_momenta(seg1.final_momenta_ase.copy())
    ref2, fast2 = ReferenceDouble(), SurrogateDouble(c=0.9)
    seg2 = run_mts(back, ref2, fast2, inner_timestep_fs=1.0, outer_ratio=2,
                   n_outer_steps=4, initial_labels=seg1.final_labels,
                   task_counter_start=seg1.task_counter_end)
    # bit-identical trajectory across the segment boundary
    assert np.array_equal(seg2.final_positions_A, full.final_positions_A)
    assert np.array_equal(seg2.final_momenta_ase, full.final_momenta_ase)
    for b_full, b_seg in zip(full.boundaries[4:], seg2.boundaries):
        assert np.array_equal(b_full.positions_A, b_seg.positions_A)
        assert np.array_equal(b_full.momenta_ase, b_seg.momenta_ase)
    # counts add without a duplicated initial evaluation
    assert ref1.calls + ref2.calls == full.reference_calls == 9
    assert fast1.calls + fast2.calls == full.surrogate_calls == 17
    assert ref2.calls == 4 and fast2.calls == 8
    assert seg2.task_counter_end == seg1.task_counter_end + 12


def test_initial_labels_reject_bad_shape_and_zero_steps():
    from pyraimd2.loop.mts import MtsLabels
    atoms = demo_atoms()
    good = run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
                   inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1)
    labels = good.final_labels
    bad = MtsLabels(labels.U_ref_eV, np.zeros((3, 3)),
                    labels.U_fast_eV, labels.F_fast_eV_A)
    with pytest.raises(MtsError, match="shapes"):
        run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=1,
                initial_labels=bad)
    with pytest.raises(MtsError, match="zero"):
        run_mts(atoms, ReferenceDouble(), SurrogateDouble(),
                inner_timestep_fs=1.0, outer_ratio=1, n_outer_steps=0,
                initial_labels=labels)
