"""Offline trajectory comparison (`compare_runs` / `pyramid compare`).

Two layers of coverage, all offline:

- hand-built run directories (store rows + event log written directly)
  with HAND-COMPUTED expected qRMS/vRMS/H answers on two different time
  grids — the per-atom (/N) normalization and the ASE velocity unit
  conversion (p/m * ase.units.fs) are guarded by explicit arithmetic;
- a real Pyramid flow on the builtin harmonic backends: a plain
  reference-only NVE run vs a fixed-model MTS run from the same initial
  structure, including an incomplete tail (a committed evaluation without
  its step boundary) and a byte-level check that comparison modifies
  nothing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.io import write as ase_write

from pyraimd2.cli import main as cli_main
from pyraimd2.config import load_config
from pyraimd2.engines.base import EngineResult
from pyraimd2.runtime.context import EvaluationContext, EvaluationPhase
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.workflows import (
    CompareError,
    compare_runs,
    run_workflow,
)

# ---------------------------------------------------------------------------
# hand-built run directories (plain-nve-shaped records, no MD executed)

NUMBERS = [1, 8, 26]                    # three different species
MASSES = np.array([1.0, 2.0, 4.0])      # distinct masses on purpose
P0 = np.array([[0.00, 0.00, 0.00],
               [1.00, 0.20, 0.30],
               [0.10, 0.90, 0.50]])


def _write_run(run_dir: Path, run_id: str, times_fs, positions, *,
               momenta=None, masses=MASSES, numbers=NUMBERS,
               cell=None, pbc=False, driver="plain-nve",
               energies=None, energy_kind="energy",
               force_consistent=True, reference_labels=True) -> Path:
    """One minimal but complete run directory: run_start, one committed
    evaluation (+ step_completed boundary) per state.

    ``positions``/``momenta`` are per-state (n_states, N, 3) arrays;
    ``energies`` are the per-state reference labels.  ``masses`` is one
    (N,) array for the whole run or a per-state list of (N,) arrays.
    Nothing here is a shortcut around the real readers: the records are
    exactly what the plain NVE driver commits.
    """
    run_dir.mkdir(parents=True)
    n_atoms = len(numbers)
    events: list[dict] = []

    def emit(event_type: str, payload: dict) -> None:
        events.append({"seq": len(events) + 1, "type": event_type,
                       **payload})

    if (isinstance(masses, (list, tuple))
            and len(masses) == len(times_fs)
            and np.asarray(masses[0]).shape == (n_atoms,)):
        masses_per_state = [np.asarray(m, dtype=float) for m in masses]
    else:
        masses_per_state = [np.asarray(masses, dtype=float)] * len(times_fs)

    emit("run_start", {
        "run_id": run_id, "schema_version": 2, "event_schema_version": 1,
        "attempt_ledger": "physical-v1", "software_version": "test",
        "reference_id": "hand-built-reference", "model_id": "hand-built",
        "workflow": {"driver": driver, "mode": "reference",
                     "integrator": {"algorithm": "verlet", "ensemble": "nve",
                                    "timestep_fs": 1.0}},
        "streams": None, "policy": None})
    with Store(run_dir / "trajectory.db") as store:
        for index, time_fs in enumerate(times_fs):
            step = index - 1
            atoms = Atoms(numbers=numbers,
                          positions=np.asarray(positions[index], float),
                          masses=masses_per_state[index],
                          cell=(np.zeros((3, 3)) if cell is None else cell),
                          pbc=pbc)
            if momenta is not None:
                atoms.set_momenta(np.asarray(momenta[index], float))
            ctx = EvaluationContext(
                run_id=run_id, step_id=step, evaluation_id=index,
                phase=(EvaluationPhase.INITIAL if index == 0
                       else EvaluationPhase.MD_STEP),
                physical_time_fs=float(time_fs), model_id="hand-built")
            engine = None
            if reference_labels and energies is not None:
                engine = EngineResult(
                    energy=float(energies[index]),
                    forces=np.zeros((n_atoms, 3)), stress=None,
                    wall_time_s=0.0, energy_kind=energy_kind,
                    force_consistent=force_consistent)
            surrogate = None
            if engine is None:
                surrogate = SurrogatePrediction(
                    energy=0.0, forces=np.zeros((n_atoms, 3)), stress=None,
                    uncertainty=np.full(n_atoms, np.nan))
            row_id = store.append(
                run_id, step, atoms, "dft" if engine is not None else "ml",
                surrogate=surrogate, engine=engine, reason="md",
                metadata={"context": ctx.as_dict(), "accepted": True,
                          "checked": False, "constraint": None},
                driving=engine,
                label_id=(f"{run_id}-label-{index}"
                          if engine is not None else None))
            emit("evaluation_committed", {
                "run_id": run_id, "context": ctx.as_dict(),
                "route": "dft" if engine is not None else "ml",
                "checked": False, "verification": None,
                "row_id": int(row_id),
                "row_digest": store.row_digest(store.row_by_id(row_id))})
            if step >= 0:
                emit("step_completed", {
                    "run_id": run_id, "step_id": step,
                    "physical_time_fs": float(time_fs),
                    "integrator": {"algorithm": "verlet", "ensemble": "nve",
                                   "timestep_fs": 1.0}})
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n")
    return run_dir


# --- the hand-computed analytic pair ----------------------------------------
#
# Reference grid 0,1,2,3 fs; candidate grid 0,2 fs (matched: 0 and 2 fs).
# Positions drift linearly in time: dq(t) = t * D with D chosen so the
# per-atom (/N) and per-component (/3N) normalizations differ by a
# factor sqrt(3) — an implementation dividing by 3N fails these numbers.
# Momenta are constant, so the velocity difference is constant; energies
# are linear in time with deliberately different zeros per run.

A_REF = np.array([[0.001, 0.0, 0.0],
                  [0.0, 0.002, 0.0],
                  [0.0, 0.0, 0.001]])
D_DQ = np.array([[0.01, 0.0, 0.0],
                 [0.0, -0.02, 0.0],
                 [0.0, 0.0, 0.03]])
P_REF_MOMENTA = np.array([[0.1, 0.0, 0.0],
                          [0.0, 0.2, 0.0],
                          [0.0, 0.0, -0.1]])
DP = np.array([[0.01, 0.0, 0.0],
               [0.0, 0.02, 0.0],
               [0.0, 0.0, 0.03]])
REF_TIMES = [0.0, 1.0, 2.0, 3.0]
CAND_TIMES = [0.0, 2.0]
REF_ENERGIES = [1.0 + 0.5 * t for t in REF_TIMES]
CAND_ENERGIES = [2.0 - 0.25 * t for t in CAND_TIMES]

# hand-computed expectations
QRMS_PER_FS = float(np.sqrt((D_DQ**2).sum() / 3))          # /N, not /3N
DV = DP / MASSES[:, None] * units.fs
VRMS = float(np.sqrt((DV**2).sum() / 3))
KE_REF = float((P_REF_MOMENTA**2 / (2 * MASSES[:, None])).sum())
KE_CAND = float(((P_REF_MOMENTA + DP)**2
                 / (2 * MASSES[:, None])).sum())


def _analytic_pair(tmp_path: Path) -> tuple[Path, Path]:
    ref_positions = [P0 + t * A_REF for t in REF_TIMES]
    cand_positions = [P0 + t * (A_REF + D_DQ) for t in CAND_TIMES]
    reference = _write_run(
        tmp_path / "ref", "ref-run", REF_TIMES, ref_positions,
        momenta=[P_REF_MOMENTA] * len(REF_TIMES), energies=REF_ENERGIES)
    candidate = _write_run(
        tmp_path / "cand", "cand-run", CAND_TIMES, cand_positions,
        # the same-initial-state contract: identical momenta at t = 0,
        # diverging afterwards
        momenta=[P_REF_MOMENTA, P_REF_MOMENTA + DP],
        energies=CAND_ENERGIES)
    return reference, candidate


def test_hand_computed_metrics_on_two_time_grids(tmp_path):
    reference, candidate = _analytic_pair(tmp_path)
    report = compare_runs(reference, candidate)

    assert report["criteria_status"] == "not_requested"
    assert report["velocity_available"] is True
    axis = report["time_axis"]
    assert axis["matched_points"] == 2
    assert axis["start_fs"] == 0.0 and axis["end_fs"] == 2.0
    assert axis["reference_complete_states"] == 4
    assert axis["candidate_complete_states"] == 2

    position = report["position_rms_A"]
    assert position["times_fs"] == [0.0, 2.0]
    np.testing.assert_allclose(
        position["per_time"], [0.0 * QRMS_PER_FS, 2.0 * QRMS_PER_FS],
        rtol=1e-12, atol=0.0)
    # explicit per-atom normalization guard: /3N would give sqrt(1/3) of it
    assert position["max"] == pytest.approx(2.0 * QRMS_PER_FS, rel=1e-12)
    assert not np.isclose(position["max"], 2.0 * np.sqrt((D_DQ**2).sum()
                                                         / 9))

    velocity = report["velocity_rms_A_fs"]
    np.testing.assert_allclose(velocity["per_time"], [0.0, VRMS],
                               rtol=1e-12, atol=0.0)
    assert velocity["max"] == pytest.approx(VRMS, rel=1e-12)

    # Hamiltonian: per-trajectory, zeroed at each run's own start
    ref_h = report["hamiltonian"]["reference"]
    assert ref_h["status"] == "available"
    np.testing.assert_allclose(ref_h["times_fs"], REF_TIMES)
    np.testing.assert_allclose(
        ref_h["hamiltonian_eV"], [e + KE_REF for e in REF_ENERGIES])
    np.testing.assert_allclose(
        ref_h["drift_per_atom_eV"], [0.5 * t / 3 for t in REF_TIMES])
    assert ref_h["max_abs_drift_per_atom_eV"] == pytest.approx(0.5)
    cand_h = report["hamiltonian"]["candidate"]
    assert cand_h["status"] == "available"
    np.testing.assert_allclose(
        cand_h["hamiltonian_eV"],
        [CAND_ENERGIES[0] + KE_REF, CAND_ENERGIES[1] + KE_CAND])
    # drift: (E(t)-E(0) + KE(t)-KE(0)) / N, hand-computed
    dke = KE_CAND - KE_REF
    np.testing.assert_allclose(
        cand_h["drift_per_atom_eV"], [0.0, (-0.25 * 2.0 + dke) / 3])
    assert cand_h["max_abs_drift_per_atom_eV"] == pytest.approx(
        abs((-0.5 + dke) / 3))

    # serializable as one strict JSON object (no NaN/Inf anywhere)
    json.dumps(report, allow_nan=False)


def test_reference_may_carry_denser_points(tmp_path):
    # candidate grid is a strict subset; reference extras are simply unused
    reference, candidate = _analytic_pair(tmp_path)
    report = compare_runs(reference, candidate)
    assert report["coverage"]["reference"]["complete_states"] == 4
    assert report["coverage"]["candidate"]["complete_states"] == 2
    assert report["time_axis"]["matched_points"] == 2


# --- refusal and degradation paths -------------------------------------------


def test_missing_momenta_position_only_and_velocity_threshold_refused(
        tmp_path):
    reference, _ = _analytic_pair(tmp_path)
    cand_positions = [P0 + t * (A_REF + D_DQ) for t in CAND_TIMES]
    no_momenta = _write_run(tmp_path / "cand-nop", "cand-nop", CAND_TIMES,
                            cand_positions, momenta=None,
                            energies=CAND_ENERGIES)
    report = compare_runs(reference, no_momenta)
    # positions and static structure are still verified and reported
    assert report["initial_state"]["positions_match"] is True
    assert report["initial_state"]["initial_momenta_match"] == "unavailable"
    assert report["velocity_available"] is False
    assert report["velocity_rms_A_fs"]["status"] == "unavailable"
    assert report["position_rms_A"]["max"] == pytest.approx(
        2.0 * QRMS_PER_FS, rel=1e-12)
    # an explicit velocity criterion cannot be evaluated -> non-zero exit
    with pytest.raises(CompareError) as excinfo:
        compare_runs(reference, no_momenta, max_velocity_rms_A_fs=1.0)
    assert excinfo.value.reason == "missing_information"


def test_zero_momenta_array_is_real_data_not_a_default(tmp_path):
    # explicitly RECORDED zero momenta are real data (stored array present),
    # unlike a missing momenta array that ASE would read back as zeros
    zeros = [np.zeros((3, 3)), np.zeros((3, 3))]
    reference = _write_run(tmp_path / "ref-still", "ref-still", CAND_TIMES,
                           [P0, P0 + 2 * A_REF], momenta=zeros,
                           energies=CAND_ENERGIES)
    candidate = _write_run(tmp_path / "cand-still", "cand-still", CAND_TIMES,
                           [P0, P0 + 2 * (A_REF + D_DQ)], momenta=zeros,
                           energies=CAND_ENERGIES)
    report = compare_runs(reference, candidate)
    assert report["velocity_available"] is True
    assert report["initial_state"]["initial_momenta_match"] is True
    assert report["velocity_rms_A_fs"]["max"] == 0.0
    assert report["position_rms_A"]["max"] == pytest.approx(
        2.0 * QRMS_PER_FS, rel=1e-12)


def test_missing_reference_energy_labels_mark_hamiltonian_unavailable(
        tmp_path):
    reference, _ = _analytic_pair(tmp_path)
    cand_positions = [P0 + t * (A_REF + D_DQ) for t in CAND_TIMES]
    unlabeled = _write_run(tmp_path / "cand-nolab", "cand-nolab",
                           CAND_TIMES, cand_positions,
                           momenta=[P_REF_MOMENTA, P_REF_MOMENTA + DP],
                           reference_labels=False)
    report = compare_runs(reference, unlabeled)
    block = report["hamiltonian"]["candidate"]
    assert block["status"] == "unavailable"
    assert "reference energy label" in block["reason"]
    assert block["coverage"] == {"complete_states": 2,
                                 "reference_energy_labels": 0}
    # the reference side stays available; position/velocity metrics stand
    assert report["hamiltonian"]["reference"]["status"] == "available"


def test_wrong_energy_convention_marks_hamiltonian_unavailable(tmp_path):
    reference, _ = _analytic_pair(tmp_path)
    cand_positions = [P0 + t * (A_REF + D_DQ) for t in CAND_TIMES]
    free_energy = _write_run(tmp_path / "cand-fe", "cand-fe", CAND_TIMES,
                             cand_positions,
                             momenta=[P_REF_MOMENTA, P_REF_MOMENTA + DP],
                             energies=CAND_ENERGIES,
                             energy_kind="free_energy")
    report = compare_runs(reference, free_energy)
    block = report["hamiltonian"]["candidate"]
    assert block["status"] == "unavailable"
    assert "free_energy" in block["reason"]


def test_unmatched_candidate_time_refused(tmp_path):
    reference, _ = _analytic_pair(tmp_path)
    shifted = _write_run(tmp_path / "cand-off", "cand-off", [0.0, 1.5],
                         [P0, P0 + 1.5 * A_REF],
                         momenta=[P_REF_MOMENTA] * 2,
                         energies=[1.0, 1.5])
    with pytest.raises(CompareError, match="no reference time point") as e:
        compare_runs(reference, shifted)
    assert e.value.reason == "incompatible_inputs"


def test_duplicate_time_axis_refused(tmp_path):
    reference, _ = _analytic_pair(tmp_path)
    # two committed states at the same physical time: ambiguous axis
    duplicate = _write_run(tmp_path / "cand-dup", "cand-dup", [0.0, 2.0, 2.0],
                           [P0, P0 + A_REF, P0 + 2 * A_REF],
                           momenta=[P_REF_MOMENTA] * 3,
                           energies=[1.0, 1.5, 2.0])
    with pytest.raises(CompareError, match="strictly increasing") as e:
        compare_runs(reference, duplicate)
    assert e.value.reason == "incompatible_inputs"


def test_atom_count_and_cell_mismatch_refused(tmp_path):
    reference, _ = _analytic_pair(tmp_path)
    fewer = _write_run(tmp_path / "fewer", "fewer", CAND_TIMES,
                       [p[:2] for p in [P0, P0 + 2 * (A_REF + D_DQ)]],
                       momenta=[P_REF_MOMENTA[:2]] * 2,
                       masses=MASSES[:2], numbers=NUMBERS[:2],
                       energies=[1.0, 1.5])
    with pytest.raises(CompareError, match="atom counts differ") as e:
        compare_runs(reference, fewer)
    assert e.value.reason == "incompatible_inputs"

    box = np.diag([10.0, 10.0, 10.0])
    boxed = _write_run(tmp_path / "boxed", "boxed", CAND_TIMES,
                       [P0, P0 + 2 * (A_REF + D_DQ)],
                       momenta=[P_REF_MOMENTA + DP] * 2,
                       cell=box, pbc=True, energies=CAND_ENERGIES)
    with pytest.raises(CompareError, match="pbc differs"):
        compare_runs(reference, boxed)
    reference_box = _write_run(tmp_path / "ref-box", "ref-box", REF_TIMES,
                               [P0 + t * A_REF for t in REF_TIMES],
                               momenta=[P_REF_MOMENTA] * 4,
                               cell=np.diag([10.0, 10.5, 10.0]), pbc=True,
                               energies=REF_ENERGIES)
    with pytest.raises(CompareError, match="cells differ"):
        compare_runs(reference_box, boxed)


def test_different_initial_state_refused(tmp_path):
    reference, _ = _analytic_pair(tmp_path)
    displaced = _write_run(tmp_path / "cand-x0", "cand-x0", CAND_TIMES,
                           [P0 + 1e-6, P0 + 1e-6 + 2 * A_REF],
                           momenta=[P_REF_MOMENTA] * 2,
                           energies=CAND_ENERGIES)
    with pytest.raises(CompareError, match="initial positions differ"):
        compare_runs(reference, displaced)


def test_out_of_scope_drivers_refused(tmp_path):
    reference, _ = _analytic_pair(tmp_path)
    for driver, kind in (("plain-nvt", "NVT"),
                         ("relax", "relaxation"),
                         ("singlepoint", "single-point")):
        run = _write_run(tmp_path / f"cand-{driver}", f"cand-{driver}",
                         CAND_TIMES, [P0, P0 + 2 * A_REF],
                         momenta=[P_REF_MOMENTA] * 2, energies=[1.0, 1.5],
                         driver=driver)
        with pytest.raises(CompareError, match="out of scope") as e:
            compare_runs(reference, run)
        assert e.value.reason == "unsupported_scope", kind


def test_missing_run_pieces_refused(tmp_path):
    reference, candidate = _analytic_pair(tmp_path)
    with pytest.raises(CompareError, match="not found") as e:
        compare_runs(tmp_path / "no-such-dir", candidate)
    assert e.value.reason == "missing_information"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CompareError, match="no trajectory.db"):
        compare_runs(reference, empty)
    # a missing database is never created as a side effect
    assert not list(empty.iterdir())


def test_threshold_boundary_exactly_at_vs_just_over(tmp_path):
    reference, candidate = _analytic_pair(tmp_path)
    # the observed max, bit-identical, as threshold: passes (<= semantics)
    observed = compare_runs(reference, candidate)["position_rms_A"]["max"]
    assert observed > 0.0
    report = compare_runs(reference, candidate, max_position_rms_A=observed)
    assert report["criteria_status"] == "passed"
    assert report["criteria"]["max_position_rms_A"]["passed"] is True
    just_under = np.nextafter(observed, 0.0)
    report = compare_runs(reference, candidate,
                          max_position_rms_A=just_under)
    assert report["criteria_status"] == "failed"
    assert report["criteria"]["max_position_rms_A"]["passed"] is False
    # velocity threshold exactly at the observed max passes too
    observed_v = compare_runs(reference,
                              candidate)["velocity_rms_A_fs"]["max"]
    assert observed_v == pytest.approx(VRMS, rel=1e-12)
    report = compare_runs(reference, candidate,
                          max_velocity_rms_A_fs=observed_v)
    assert report["criteria"]["max_velocity_rms_A_fs"]["passed"] is True
    with pytest.raises(CompareError, match="finite"):
        compare_runs(reference, candidate, max_position_rms_A=-1.0)


def test_comparison_never_touches_backend_factories(tmp_path, monkeypatch):
    reference, candidate = _analytic_pair(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("compare must never construct a backend")

    from pyraimd2.backends import registry
    from pyraimd2.workflows import setup
    monkeypatch.setattr(registry, "create_backend", forbidden)
    monkeypatch.setattr(registry, "backend_factory", forbidden)
    monkeypatch.setattr(setup, "build_backends", forbidden)
    monkeypatch.setattr(setup, "create_configured_backend", forbidden)
    report = compare_runs(reference, candidate)
    assert report["time_axis"]["matched_points"] == 2
    for module in ("mace", "torch", "pyscf"):
        assert module not in sys.modules


# ---------------------------------------------------------------------------
# real Pyramid flow: plain reference NVE vs fixed-model MTS, same start

MASS = 28.085


def _structure(path: Path) -> None:
    a = np.array([0.9, 0.9, 0.9])
    d = np.array([0.03, -0.02, 0.01])
    atoms = Atoms("Si2", positions=[a - d, a + d], masses=[MASS, MASS],
                  pbc=False)
    v0 = np.array([0.001, -0.0015, 0.002]) / units.fs
    atoms.set_momenta(np.array([-v0, v0]) * MASS)
    ase_write(path, atoms)


def _real_pair(tmp_path: Path) -> tuple[Path, Path]:
    _structure(tmp_path / "structure.extxyz")
    (tmp_path / "ref.toml").write_text("""schema_version = 1
[run]
id = "compare-ref"
directory = "run-ref"
[task]
kind = "md"
mode = "reference"
[structure]
file = "structure.extxyz"
[dynamics]
ensemble = "nve"
integrator = "verlet"
timestep_fs = 1.0
steps = 16
[checkpoint]
interval_steps = 8
[reference]
backend = "harmonic-reference"
k = 1.0
r0 = 0.9
""")
    (tmp_path / "mts.toml").write_text("""schema_version = 1
[run]
id = "compare-mts"
directory = "run-mts"
[task]
kind = "md"
mode = "mts"
[structure]
file = "structure.extxyz"
[dynamics]
ensemble = "nve"
integrator = "respa"
timestep_fs = 1.0
steps = 16
outer_ratio = 4
[checkpoint]
interval_steps = 8
[reference]
backend = "harmonic-reference"
k = 1.0
r0 = 0.9
[surrogate]
backend = "harmonic-surrogate"
k = 0.9
r0 = 0.9
bias = 0.0
""")
    run_workflow(load_config(tmp_path / "ref.toml"), verbose=False,
                 handle_sigint=False)
    run_workflow(load_config(tmp_path / "mts.toml"), verbose=False,
                 handle_sigint=False)
    return tmp_path / "run-ref", tmp_path / "run-mts"


def _file_digests(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_real_runs_compare_and_leave_directories_untouched(tmp_path):
    reference, candidate = _real_pair(tmp_path)
    before = {str(run): _file_digests(run) for run in (reference, candidate)}
    report = compare_runs(reference, candidate)
    after = {str(run): _file_digests(run) for run in (reference, candidate)}
    assert before == after  # read-only: DB, events and checkpoints identical

    assert report["reference_run"]["driver"] == "plain-nve"
    assert report["candidate_run"]["driver"] == "mts-nve-respa"
    assert report["initial_state"]["positions_match"] is True
    assert report["initial_state"]["initial_momenta_match"] is True
    axis = report["time_axis"]
    assert axis["matched_points"] == 5            # 0, 4, 8, 12, 16 fs
    assert axis["reference_complete_states"] == 17
    assert axis["candidate_complete_states"] == 5
    # the MTS surrogate (k=0.9) differs from the reference (k=1.0): the
    # trajectories genuinely diverge, and both Hamiltonians are available
    assert report["position_rms_A"]["max"] > 0.0
    assert report["velocity_rms_A_fs"]["max"] > 0.0
    assert report["hamiltonian"]["reference"]["status"] == "available"
    assert report["hamiltonian"]["candidate"]["status"] == "available"
    assert report["criteria_status"] == "not_requested"


def test_incomplete_tail_is_coverage_not_a_trajectory_point(tmp_path):
    reference, candidate = _real_pair(tmp_path)
    # an interrupted run on a COPY: committed outer boundary whose
    # step_completed record never landed (crash between the two)
    truncated = tmp_path / "copy-mts"
    shutil.copytree(candidate, truncated)
    events_path = truncated / "events.jsonl"
    lines = events_path.read_text().splitlines()
    items = [json.loads(line) for line in lines]
    boundary = next(i for i, e in enumerate(items)
                    if e.get("type") == "step_completed"
                    and e.get("step_id") == 16)
    events_path.write_text("\n".join(lines[:boundary]) + "\n")

    report = compare_runs(reference, truncated)
    axis = report["time_axis"]
    assert axis["matched_points"] == 4            # 16 fs tail not a point
    assert axis["end_fs"] == 12.0
    coverage = report["coverage"]["candidate"]
    assert coverage["committed_evaluations"] == 5
    assert coverage["complete_states"] == 4
    assert coverage["incomplete_tail_evaluations"] == 1
    assert coverage["incomplete_tail_step_ids"] == [16]
    # the truncated copy keeps its committed rows; the original is untouched
    assert (truncated / "trajectory.db").is_file()


# ---------------------------------------------------------------------------
# CLI


def test_cli_compare_human_json_and_exit_codes(tmp_path, capsys):
    reference, candidate = _real_pair(tmp_path)
    code = cli_main(["compare", str(reference), str(candidate)])
    out = capsys.readouterr().out
    assert code == 0
    assert "compare: reference compare-ref (plain-nve)" in out
    assert "criteria          : not requested" in out

    code = cli_main(["compare", str(reference), str(candidate), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["criteria_status"] == "not_requested"
    assert report["time_axis"]["matched_points"] == 5

    observed = report["position_rms_A"]["max"]
    code = cli_main(["compare", str(reference), str(candidate),
                     "--max-position-rms", f"{observed:.17e}", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["criteria_status"] == "passed"
    code = cli_main(["compare", str(reference), str(candidate),
                     "--max-position-rms", f"{observed / 2:.17e}"])
    assert code == 1                      # requested criterion exceeded
    assert "FAIL" in capsys.readouterr().out


def test_cli_compare_errors_are_structured(tmp_path, capsys):
    reference, _ = _real_pair(tmp_path)
    nvt = _write_run(tmp_path / "nvt", "nvt", [0.0, 4.0],
                     [P0[:2], P0[:2]], masses=[MASS, MASS],
                     numbers=[14, 14], driver="plain-nvt",
                     energies=[0.0, 0.0])
    code = cli_main(["compare", str(reference), str(nvt), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["ok"] is False
    assert payload["error"]["reason"] == "unsupported_scope"
    assert "NVT" in payload["error"]["message"]

    code = cli_main(["compare", str(reference),
                     str(tmp_path / "no-such-run")])
    assert code == 1                      # missing required information
    assert "error: compare:" in capsys.readouterr().err

    # a velocity criterion against momenta-less records: checked failure
    p0 = np.array([[0.87, 0.92, 0.89], [0.93, 0.88, 0.91]])  # _structure()
    nop = _write_run(tmp_path / "nop", "nop", [0.0, 4.0],
                     [p0, p0 + 0.001], momenta=None, masses=[MASS, MASS],
                     numbers=[14, 14], energies=[0.0, 0.0])
    code = cli_main(["compare", str(reference), str(nop),
                     "--max-velocity-rms", "1.0", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["error"]["reason"] == "missing_information"


# ---------------------------------------------------------------------------
# read-only guarantee and structured read-failure conversion


def _tree_state(root: Path) -> dict[str, str]:
    """relpath -> sha256 for every file; a NEW sidecar (-wal/-shm/-journal)
    shows up as a new key, so dict equality proves the tree byte-identical."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _broken_db_run(tmp_path: Path, source: Path, name: str,
                   content: bytes) -> Path:
    """A run directory with the candidate's event log but a broken db."""
    run = tmp_path / name
    run.mkdir()
    (run / "trajectory.db").write_bytes(content)
    (run / "events.jsonl").write_bytes((source / "events.jsonl").read_bytes())
    return run


def test_zero_byte_database_refused_without_touching(tmp_path, capsys):
    reference, candidate = _analytic_pair(tmp_path)
    broken = _broken_db_run(tmp_path, candidate, "empty-db", b"")
    before = _tree_state(broken)
    with pytest.raises(CompareError, match="missing or unreadable") as e:
        compare_runs(reference, broken)
    assert e.value.reason == "missing_information"
    # never grown, never side-carred: the tree is byte-identical
    assert _tree_state(broken) == before
    assert (broken / "trajectory.db").stat().st_size == 0
    assert sorted(p.name for p in broken.iterdir()) == ["events.jsonl",
                                                       "trajectory.db"]

    code = cli_main(["compare", str(reference), str(broken), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)   # exactly one parseable JSON object
    assert code == 1
    assert payload["ok"] is False
    assert payload["error"]["reason"] == "missing_information"
    assert captured.err == ""
    assert _tree_state(broken) == before


def test_damaged_database_refused_without_touching(tmp_path, capsys):
    reference, candidate = _analytic_pair(tmp_path)
    broken = _broken_db_run(tmp_path, candidate, "damaged-db",
                            b"truncated database fixture")
    before = _tree_state(broken)
    with pytest.raises(CompareError, match="missing or unreadable") as e:
        compare_runs(reference, broken)
    assert e.value.reason == "missing_information"
    assert _tree_state(broken) == before

    code = cli_main(["compare", str(reference), str(broken), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["ok"] is False
    assert payload["error"]["reason"] == "missing_information"
    assert captured.err == ""
    assert _tree_state(broken) == before


def test_malformed_interior_event_refused_as_structured_json(tmp_path,
                                                             capsys):
    reference, candidate = _analytic_pair(tmp_path)
    broken = tmp_path / "bad-events"
    broken.mkdir()
    (broken / "trajectory.db").write_bytes(
        (candidate / "trajectory.db").read_bytes())
    lines = (candidate / "events.jsonl").read_text().splitlines()
    lines.insert(1, "{invalid json")          # an interior line, not a tail
    (broken / "events.jsonl").write_text("\n".join(lines) + "\n")
    before = _tree_state(broken)
    with pytest.raises(CompareError, match="missing or unreadable") as e:
        compare_runs(reference, broken)
    assert e.value.reason == "missing_information"
    assert _tree_state(broken) == before

    code = cli_main(["compare", str(reference), str(broken), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["ok"] is False
    assert payload["error"]["reason"] == "missing_information"
    assert captured.err == ""


def test_valid_compare_leaves_no_sidecar_files(tmp_path):
    reference, candidate = _analytic_pair(tmp_path)
    before = {str(run): _tree_state(run) for run in (reference, candidate)}
    report = compare_runs(reference, candidate)
    assert report["time_axis"]["matched_points"] == 2
    assert {str(run): _tree_state(run)
            for run in (reference, candidate)} == before
    for run in (reference, candidate):
        assert not [p for p in run.rglob("*")
                    if p.suffix in ("-wal", "-shm", "-journal")
                    or p.name.endswith(("-wal", "-shm", "-journal"))]


# --- per-frame mass validation ------------------------------------------------


def test_mass_change_after_initial_refused(tmp_path):
    # the reviewer's case: the last committed state carries doubled masses;
    # velocities/H must never be computed from the first frame's masses
    reference, _ = _analytic_pair(tmp_path)
    changed = _write_run(tmp_path / "cand-mass", "cand-mass", CAND_TIMES,
                         [P0, P0 + 2 * (A_REF + D_DQ)],
                         momenta=[P_REF_MOMENTA, P_REF_MOMENTA + DP],
                         masses=[MASSES, MASSES * 2],
                         energies=CAND_ENERGIES)
    with pytest.raises(CompareError, match="masses") as e:
        compare_runs(reference, changed, max_velocity_rms_A_fs=1.0)
    assert e.value.reason == "incompatible_inputs"


@pytest.mark.parametrize("bad_masses", [np.zeros(3),
                                        np.full(3, np.nan),
                                        np.full(3, -1.0)],
                         ids=["zero", "nan", "negative"])
def test_invalid_masses_refused(tmp_path, bad_masses):
    reference, _ = _analytic_pair(tmp_path)
    broken = _write_run(tmp_path / "cand-bad-mass", "cand-bad-mass",
                        CAND_TIMES, [P0, P0 + 2 * (A_REF + D_DQ)],
                        momenta=[P_REF_MOMENTA, P_REF_MOMENTA + DP],
                        masses=[MASSES, bad_masses],
                        energies=CAND_ENERGIES)
    with pytest.raises(CompareError, match="invalid masses") as e:
        compare_runs(reference, broken)
    assert e.value.reason == "missing_information"


# --- WAL / sidecar-log databases are refused before any connection -----------


def test_wal_journal_database_refused_without_touching(tmp_path, capsys):
    """A cleanly closed WAL database (header marks WAL, no sidecars left):
    a read-only SQLite connection would still create -wal/-shm files, so
    the comparison refuses BEFORE connecting."""
    reference, candidate = _analytic_pair(tmp_path)
    db = candidate / "trajectory.db"
    connection = sqlite3.connect(db)
    with connection:
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        assert mode == "wal"
    connection.close()   # clean close checkpoints and removes sidecars
    assert sorted(p.name for p in candidate.iterdir()) == ["events.jsonl",
                                                           "trajectory.db"]
    before = _tree_state(candidate)

    with pytest.raises(CompareError, match="WAL journal mode") as e:
        compare_runs(reference, candidate)
    assert e.value.reason == "unsupported_scope"

    code = cli_main(["compare", str(reference), str(candidate), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)   # exactly one parseable JSON object
    assert code == 2                     # out-of-scope input: usage exit
    assert payload["ok"] is False
    assert payload["error"]["reason"] == "unsupported_scope"
    assert captured.err == ""
    # the source directory is byte-identical: no -wal/-shm appeared
    assert _tree_state(candidate) == before
    assert sorted(p.name for p in candidate.iterdir()) == ["events.jsonl",
                                                           "trajectory.db"]


def test_wal_sidecar_database_refused_and_stale_main_db_not_read(tmp_path):
    """A committed-but-not-checkpointed WAL transaction: the sidecar log
    holds state the main file lacks — refused, never read as current."""
    reference, candidate = _analytic_pair(tmp_path)
    db = candidate / "trajectory.db"
    writer = sqlite3.connect(db)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        # commit a marker transaction and keep the writer open: the commit
        # lives in the -wal sidecar, the main db file is stale
        writer.execute("CREATE TABLE wal_marker (value INTEGER)")
        writer.execute("INSERT INTO wal_marker VALUES (1)")
        writer.commit()
        assert (candidate / "trajectory.db-wal").exists()
        assert (candidate / "trajectory.db-shm").exists()
        # the main file is genuinely stale: an immutable reader (which
        # ignores the WAL) cannot see the committed marker table
        probe = sqlite3.connect(
            f"file:{db}?immutable=1", uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError,
                               match="no such table"):
                probe.execute("SELECT COUNT(*) FROM wal_marker").fetchone()
        finally:
            probe.close()
        before = _tree_state(candidate)

        with pytest.raises(CompareError, match="sidecar log") as e:
            compare_runs(reference, candidate)
        assert e.value.reason == "unsupported_scope"
        # nothing checkpointed, converted or deleted; nothing was read
        # from the stale main file as if it were current
        assert _tree_state(candidate) == before
        assert (candidate / "trajectory.db-wal").exists()
    finally:
        writer.close()
