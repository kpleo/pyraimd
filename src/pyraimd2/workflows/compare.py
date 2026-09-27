"""Offline accuracy comparison of two completed NVE MD runs.

``compare_runs`` answers one narrow question: given two ALREADY-COMPLETED
run directories that started from the same initial state, how far apart
are the two trajectories at identical physical times?  The typical pair is
a reference-driven NVE run and a cheaper candidate (for example a
fixed-model MTS run), but any two supported runs compare the same way.

Scope of this first version (everything outside it is refused with a
structured error, never silently approximated):

- fixed-cell, same-atom-order, same-initial-state deterministic NVE
  trajectories, written by the ``plain-nve`` or ``mts-nve-respa`` drivers;
- fully offline: no backend is constructed, no optional model package is
  imported, no predict/compute ever runs, and the run directories, their
  databases, event logs and checkpoints are opened read-only — the
  trajectory database through ``Store(path, read_only=True)``, whose
  file-level pre-check (before any connection) refuses WAL-mode or
  sidecar-log databases, so compared directories stay byte-identical; an
  empty, damaged or schema-less database is refused up front, never
  initialized as a side effect;
- only committed STEP_COMPLETED complete states are trajectory points,
  selected through the same verified read paths as ``pyramid export``
  (``frames_for_run`` over the commit-bound row view, with the store's
  complete-step momentum reconstruction); a committed tail evaluation
  without its boundary is reported in the coverage counts, never compared.

Time matching is exact set intersection at absolute tolerance 1e-9 fs
(rtol 0): every complete candidate time point must find a UNIQUE reference
time point.  There is no interpolation, no snapping to a coarser grid, and
step/row ids are never used as time — times come only from the runs'
recorded ``physical_time_fs`` context.

Metrics are pointwise in the stored continuous coordinates (no minimum
image, no alignment, no unwrapping, no reordering): the per-atom-normalized
position RMS ``sqrt(sum_{i,xyz} dq^2 / N)`` in angstrom and, when both runs
carry real recorded momenta, the velocity RMS in angstrom/fs with
``v = p / m * ase.units.fs``.  Per-trajectory Hamiltonian drift
``(H(t) - H(0)) / N`` is attached only when every complete state of that
trajectory carries a reference energy label declaring
``energy_kind="energy"`` with ``force_consistent=True`` and real
complete-step momenta; otherwise the whole metric is marked unavailable
with the reason and coverage counts (a partial drift curve would claim
full-run coverage it does not have).

The report is descriptive: it never concludes "reliable"/"recommended",
never estimates speed from call counts, and applies thresholds only when
the caller explicitly passes them.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import units

from pyraimd2.runtime.inspect import _read_events
from pyraimd2.store import Store, UnsupportedJournalError
from pyraimd2.workflows.export import (
    ExportError,
    completed_step_ids,
    frames_for_run,
)

COMPARE_SCHEMA_VERSION = 1

#: drivers whose records this version can compare honestly (fixed-cell NVE)
SUPPORTED_DRIVERS = ("plain-nve", "mts-nve-respa")

#: absolute time-match tolerance (rtol is exactly 0 — no relative slack)
TIME_TOLERANCE_FS = 1e-9
POSITION_TOLERANCE_A = 1e-10
CELL_TOLERANCE_A = 1e-10
MASS_TOLERANCE_AMU = 1e-12
#: initial momenta are compared in ASE momentum units (mass*velocity) with
#: the same numerical rounding tolerance as positions
MOMENTUM_TOLERANCE_ASE = 1e-10

#: momenta provenances that are real complete-step values (never an ASE
#: default all-zero array masquerading as velocities)
_REAL_MOMENTA_SOURCES = (
    "complete_step_recorded",
    "complete_step_reconstructed",
    "initial_evaluation_record",
)


class CompareError(RuntimeError):
    """The comparison cannot be produced honestly from the run directories.

    ``reason`` is the machine-readable failure class the CLI maps to an
    exit code and the JSON report carries verbatim:

    - ``"unsupported_scope"``: the input is a kind of run this version
      refuses on principle (NVT, relax, single-point, adaptive drivers,
      missing momenta conventions it cannot verify, ...);
    - ``"incompatible_inputs"``: the two runs do not describe the same
      physical system/initial state/time axis (atom count or order, cell,
      masses, initial state, unmatched or ambiguous times, ...);
    - ``"missing_information"``: a run exists but required records are
      absent or unusable (no event log, no committed complete states, no
      recorded times, non-finite data, a requested velocity criterion
      without real momenta);
    - ``"usage"``: the request itself is malformed (a negative or
      non-finite threshold).
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = str(reason)


@dataclass
class _RunData:
    """One run's verified complete-step trajectory plus provenance."""

    run_dir: Path
    run_id: str
    driver: str
    times_fs: np.ndarray
    positions: np.ndarray            # (n_states, n_atoms, 3), continuous
    momenta: np.ndarray | None       # same shape, or None when not real
    masses: np.ndarray
    numbers: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    # per-state reference energy label facts (Hamiltonian precondition)
    reference_energies: list[float | None] = field(default_factory=list)
    energy_kinds: list[str] = field(default_factory=list)
    force_consistent: list[bool | None] = field(default_factory=list)
    momenta_sources: list[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)


def _check_threshold(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise CompareError(
            f"{name} must be a finite value >= 0, got {value!r}",
            reason="usage")
    return value


def _load_run(run_dir: str | Path, *, role: str) -> _RunData:
    """Read one run's committed complete-step states, read-only."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise CompareError(
            f"{role} run directory not found: {run_dir}; pass a directory "
            "written by `pyramid run` (it contains trajectory.db and "
            "events.jsonl)", reason="missing_information")
    db_path = run_dir / "trajectory.db"
    if not db_path.is_file():
        # checked BEFORE opening the store: comparison never creates a
        # missing database as a side effect
        raise CompareError(
            f"no trajectory.db in {run_dir}; the {role} run has no "
            "trajectory database to compare", reason="missing_information")
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise CompareError(
            f"no events.jsonl in {run_dir}; without the event log the "
            "committed complete-step states cannot be identified, so this "
            "run cannot be compared", reason="missing_information")
    try:
        events = _read_events(events_path)
        start = next((e for e in events if e.get("type") == "run_start"),
                     None)
        if start is None:
            raise CompareError(
                f"{run_dir}: the event log has no run_start record; the "
                "run's driver and identity are unknown",
                reason="missing_information")
        driver = (start.get("workflow") or {}).get("driver")
        if driver not in SUPPORTED_DRIVERS:
            known = {"plain-nvt": "fixed-cell NVT (Langevin)",
                     "relax": "a relaxation",
                     "singlepoint": "a single-point evaluation"}
            kind = known.get(driver, f"driver {driver!r}" if driver
                             else "an adaptive (policy-driven) run")
            raise CompareError(
                f"{role} run {run_dir} is {kind}; `compare` covers only "
                "fixed-cell deterministic NVE trajectories written by the "
                f"{list(SUPPORTED_DRIVERS)} drivers in this version — NVT, "
                "variable-cell, relax, single-point and adaptive runs are "
                "out of scope", reason="unsupported_scope")
        run_id = str(start["run_id"])
        with Store(db_path, read_only=True) as store:
            committed = list(store.iter_committed(events, run_id))
            complete_steps = completed_step_ids(run_dir)
            frames = frames_for_run(store, run_dir, run_id,
                                    force_source="reference")
    except (CompareError, ExportError):
        raise
    except UnsupportedJournalError as error:
        # a storage format outside this version's supported scope: the
        # run's records are intact, but reading them could not guarantee
        # an unchanged source directory
        raise CompareError(str(error), reason="unsupported_scope") from error
    except (OSError, sqlite3.DatabaseError, KeyError, json.JSONDecodeError,
            RuntimeError) as error:
        # the known read failures at this boundary: file I/O (OSError,
        # including unreadable/old ASE database formats), damaged or
        # uninitialized SQLite files (sqlite3.DatabaseError, StoreError),
        # corrupt event-log records (EventLogError, JSONDecodeError) and
        # committed rows the store can no longer resolve (KeyError
        # 'no match', the store's RuntimeErrors) — all mean the run's
        # records are missing or unusable, never that the inputs mismatch
        raise CompareError(
            f"the {role} run records in {run_dir} are missing or "
            f"unreadable ({type(error).__name__}: {error})",
            reason="missing_information") from error
    incomplete_tail = sorted(
        int(row.key_value_pairs["step"]) for _event, row in committed
        if int(row.key_value_pairs["step"]) >= 0
        and int(row.key_value_pairs["step"]) not in complete_steps)
    coverage = {"committed_evaluations": len(committed),
                "complete_states": len(frames),
                # a committed evaluation without its STEP_COMPLETED boundary
                # (a crash/failure mid-step) is a cost record, not a
                # trajectory point — reported, never compared
                "incomplete_tail_evaluations": len(incomplete_tail),
                "incomplete_tail_step_ids": incomplete_tail}
    if not frames:
        raise CompareError(
            f"{role} run {run_id!r} has no committed complete states to "
            "compare", reason="missing_information")
    if frames[0].info.get("integration_phase") != "initial_evaluation":
        raise CompareError(
            f"{role} run {run_id!r}: the first committed state is not the "
            "initial evaluation; the same-initial-state check has no "
            "anchor", reason="missing_information")
    n_atoms = len(frames[0])
    numbers = np.asarray(frames[0].numbers)
    masses: np.ndarray | None = None   # validated per frame below
    cell = np.asarray(frames[0].cell.array, dtype=float)
    pbc = np.asarray(frames[0].pbc, dtype=bool)

    times: list[float] = []
    positions: list[np.ndarray] = []
    momenta: list[np.ndarray] = []
    momenta_real = True
    energies: list[float | None] = []
    kinds: list[str] = []
    consistent: list[bool | None] = []
    sources: list[str] = []
    for frame_index, frame in enumerate(frames):
        time_fs = float(frame.info.get("physical_time_fs", np.nan))
        if not np.isfinite(time_fs):
            raise CompareError(
                f"{role} run {run_id!r}: a committed state carries no "
                "finite physical_time_fs record; times are never inferred "
                "from step or row numbers", reason="missing_information")
        if len(frame) != n_atoms or not np.array_equal(frame.numbers,
                                                      numbers):
            raise CompareError(
                f"{role} run {run_id!r}: atom count or species order "
                "changes inside the trajectory; only fixed-composition "
                "runs are in scope", reason="unsupported_scope")
        frame_masses = np.asarray(frame.get_masses(), dtype=float)
        if (frame_masses.shape != (n_atoms,)
                or not np.isfinite(frame_masses).all()
                or (frame_masses <= 0).any()):
            # never "repair": velocities and H would silently use them
            raise CompareError(
                f"{role} run {run_id!r}: the committed state at index "
                f"{frame_index} carries invalid masses (shape "
                f"{frame_masses.shape}, values must be finite and "
                "strictly positive); the record is unusable",
                reason="missing_information")
        if masses is None:
            masses = frame_masses
        elif not np.allclose(frame_masses, masses, rtol=0,
                             atol=MASS_TOLERANCE_AMU):
            raise CompareError(
                f"{role} run {run_id!r}: the masses at the committed "
                f"state at index {frame_index} differ from the initial "
                f"frame's by more than {MASS_TOLERANCE_AMU:g} amu; only "
                "fixed-mass trajectories are in scope",
                reason="incompatible_inputs")
        if not np.allclose(frame.cell.array, cell, rtol=0,
                           atol=CELL_TOLERANCE_A):
            raise CompareError(
                f"{role} run {run_id!r}: the cell changes inside the "
                "trajectory; only fixed-cell runs are in scope",
                reason="unsupported_scope")
        if not np.array_equal(np.asarray(frame.pbc, dtype=bool), pbc):
            raise CompareError(
                f"{role} run {run_id!r}: pbc changes inside the "
                "trajectory", reason="unsupported_scope")
        xyz = np.asarray(frame.positions, dtype=float)
        if not np.isfinite(xyz).all():
            raise CompareError(
                f"{role} run {run_id!r}: non-finite positions in a "
                "committed state", reason="missing_information")
        source = str(frame.info.get("momenta_source", ""))
        has_real_momenta = ("momenta" in frame.arrays
                            and source in _REAL_MOMENTA_SOURCES)
        if has_real_momenta:
            p = np.asarray(frame.get_momenta(), dtype=float)
            if not np.isfinite(p).all():
                raise CompareError(
                    f"{role} run {run_id!r}: non-finite momenta in a "
                    "committed state", reason="missing_information")
            momenta.append(p)
        else:
            momenta_real = False
        times.append(time_fs)
        positions.append(xyz)
        energies.append(float(frame.info["energy"])
                        if frame.info.get("forces_available") else None)
        kinds.append(str(frame.info.get("energy_kind", "unknown")))
        consistent.append(frame.info.get("force_consistent"))
        sources.append(source)

    times_array = np.asarray(times, dtype=float)
    if len(times_array) > 1 and not (np.diff(times_array) > 0).all():
        raise CompareError(
            f"{role} run {run_id!r}: the committed physical-time axis is "
            "not strictly increasing (duplicate or out-of-order time "
            "points); an ambiguous time axis cannot be matched",
            reason="incompatible_inputs")
    return _RunData(
        run_dir=run_dir, run_id=run_id, driver=str(driver),
        times_fs=times_array, positions=np.asarray(positions, dtype=float),
        momenta=(np.asarray(momenta, dtype=float)
                 if momenta_real and len(momenta) == len(times) else None),
        masses=masses, numbers=numbers, cell=cell, pbc=pbc,
        reference_energies=energies, energy_kinds=kinds,
        force_consistent=consistent, momenta_sources=sources,
        coverage=coverage)


def _check_compatible(reference: _RunData, candidate: _RunData) -> dict:
    """Same system, same initial state — or a structured refusal."""
    if len(reference.numbers) != len(candidate.numbers):
        raise CompareError(
            f"atom counts differ ({len(reference.numbers)} vs "
            f"{len(candidate.numbers)}); trajectory comparison requires "
            "the same atoms in the same order", reason="incompatible_inputs")
    if not np.array_equal(reference.numbers, candidate.numbers):
        raise CompareError(
            "the species order differs between the two runs; this "
            "comparison never reorders atoms", reason="incompatible_inputs")
    if not np.array_equal(reference.pbc, candidate.pbc):
        raise CompareError(
            f"pbc differs ({reference.pbc.tolist()} vs "
            f"{candidate.pbc.tolist()})", reason="incompatible_inputs")
    if not np.allclose(reference.cell, candidate.cell, rtol=0,
                       atol=CELL_TOLERANCE_A):
        raise CompareError(
            "the cells differ by more than "
            f"{CELL_TOLERANCE_A:g} A per element",
            reason="incompatible_inputs")
    if not np.allclose(reference.masses, candidate.masses, rtol=0,
                       atol=MASS_TOLERANCE_AMU):
        raise CompareError(
            f"the masses differ by more than {MASS_TOLERANCE_AMU:g} amu",
            reason="incompatible_inputs")
    initial = {"time_fs": float(reference.times_fs[0])}
    if abs(float(candidate.times_fs[0])
           - float(reference.times_fs[0])) > TIME_TOLERANCE_FS:
        raise CompareError(
            f"the initial physical times differ ({reference.times_fs[0]} "
            f"vs {candidate.times_fs[0]} fs); the two runs did not start "
            "from the same initial state", reason="incompatible_inputs")
    dq0 = candidate.positions[0] - reference.positions[0]
    max_dq0 = float(np.abs(dq0).max())
    initial["max_position_difference_A"] = max_dq0
    initial["positions_match"] = bool(max_dq0 <= POSITION_TOLERANCE_A)
    if not initial["positions_match"]:
        raise CompareError(
            f"the initial positions differ by up to {max_dq0:.3e} A "
            f"(tolerance {POSITION_TOLERANCE_A:g} A); the two runs did "
            "not start from the same initial state",
            reason="incompatible_inputs")
    if reference.momenta is None or candidate.momenta is None:
        # positions and static structure are still verified above; the
        # full initial state (momenta included) is honestly not verified
        initial["initial_momenta_match"] = "unavailable"
        initial["max_momentum_difference_ase"] = None
    else:
        dp0 = candidate.momenta[0] - reference.momenta[0]
        max_dp0 = float(np.abs(dp0).max())
        initial["max_momentum_difference_ase"] = max_dp0
        initial["initial_momenta_match"] = bool(max_dp0
                                                <= MOMENTUM_TOLERANCE_ASE)
        if not initial["initial_momenta_match"]:
            raise CompareError(
                f"the initial momenta differ by up to {max_dp0:.3e} "
                f"(ASE momentum units; tolerance "
                f"{MOMENTUM_TOLERANCE_ASE:g}); the two runs did not "
                "start from the same initial state",
                reason="incompatible_inputs")
    return initial


def _match_times(reference: _RunData, candidate: _RunData) -> np.ndarray:
    """Reference row index for every candidate time point (unique, exact).

    Every complete candidate point must match exactly one reference point
    within TIME_TOLERANCE_FS (rtol 0).  The reference may carry extra
    denser points; the candidate may not carry unmatched ones.
    """
    indices: list[int] = []
    for time_fs in candidate.times_fs:
        matches = np.nonzero(
            np.abs(reference.times_fs - time_fs) <= TIME_TOLERANCE_FS)[0]
        if len(matches) == 0:
            raise CompareError(
                f"candidate run {candidate.run_id!r} has a complete state "
                f"at t = {time_fs} fs with no reference time point within "
                f"{TIME_TOLERANCE_FS:g} fs; times are never interpolated "
                "or snapped to a coarser grid", reason="incompatible_inputs")
        if len(matches) > 1:
            raise CompareError(
                f"candidate time t = {time_fs} fs matches several "
                f"reference points of run {reference.run_id!r}; the "
                "reference time axis is ambiguous",
                reason="incompatible_inputs")
        indices.append(int(matches[0]))
    if len(indices) < 2:
        raise CompareError(
            "fewer than 2 matched time points; a trajectory comparison "
            "needs at least a start and one later point",
            reason="incompatible_inputs")
    return np.asarray(indices, dtype=int)


def _rms_per_time(delta: np.ndarray, n_atoms: int) -> np.ndarray:
    """sqrt(sum_{i,xyz} delta^2 / N) per time point — per-ATOM normalization."""
    return np.sqrt((delta**2).sum(axis=(1, 2)) / n_atoms)


def _velocities(momenta: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """ASE momenta -> angstrom/fs (v = p/m * ase.units.fs)."""
    return momenta / masses[None, :, None] * units.fs


def _hamiltonian_block(run: _RunData) -> dict:
    """Per-trajectory Hamiltonian drift, or an honest unavailable marker.

    H(t) = E_reference(t) + sum_i p_i^2/(2 m_i) over the run's own complete
    states, zeroed at the run's own start.  Attached only when EVERY
    complete state carries a reference energy label declaring
    energy_kind="energy" with force_consistent=True and real complete-step
    momenta — a partial curve would claim full-run drift it does not have.
    """
    n_states = len(run.times_fs)
    n_labeled = sum(energy is not None for energy in run.reference_energies)
    coverage = {"complete_states": n_states,
                "reference_energy_labels": n_labeled}
    if run.momenta is None:
        return {"status": "unavailable", "coverage": coverage,
                "reason": "the run has no real recorded complete-step "
                          "momenta; the kinetic energy is unknown"}
    if n_labeled != n_states:
        return {"status": "unavailable", "coverage": coverage,
                "reason": f"only {n_labeled} of {n_states} complete "
                          "states carry a reference energy label"}
    bad = [kind != "energy" or consistent is not True
           for kind, consistent in zip(run.energy_kinds,
                                       run.force_consistent)]
    if any(bad):
        first = next(i for i, flag in enumerate(bad) if flag)
        return {"status": "unavailable", "coverage": coverage,
                "reason": "the reference energy labels do not all declare "
                          "energy_kind=\"energy\" with "
                          f"force_consistent=true (state {first} has "
                          f"energy_kind={run.energy_kinds[first]!r}, "
                          f"force_consistent={run.force_consistent[first]!r});"
                          " other conventions are never substituted for a "
                          "reference energy"}
    energies = np.asarray(run.reference_energies, dtype=float)
    kinetic = (run.momenta**2
               / (2.0 * run.masses[None, :, None])).sum(axis=(1, 2))
    total = energies + kinetic
    if not np.isfinite(total).all():
        return {"status": "unavailable", "coverage": coverage,
                "reason": "non-finite energy or momentum data in the "
                          "committed states"}
    drift_per_atom = (total - total[0]) / len(run.numbers)
    return {"status": "available", "coverage": coverage,
            "times_fs": run.times_fs.tolist(),
            "hamiltonian_eV": total.tolist(),
            "drift_per_atom_eV": drift_per_atom.tolist(),
            "max_abs_drift_per_atom_eV":
                float(np.abs(drift_per_atom).max())}


def compare_runs(reference_run: str | Path, candidate_run: str | Path, *,
                 max_position_rms_A: float | None = None,
                 max_velocity_rms_A_fs: float | None = None) -> dict:
    """Compare two completed runs' trajectories at identical physical times.

    ``reference_run`` and ``candidate_run`` are run directories written by
    ``pyramid run`` (fixed-cell NVE, ``plain-nve`` or ``mts-nve-respa``
    drivers) that started from the same initial state.  The returned report
    (``schema_version`` 1) carries the matched time window, per-time and
    whole-window-max position/velocity RMS arrays, coverage counts
    (including incomplete tail evaluations, which are never trajectory
    points), per-trajectory Hamiltonian drift when its preconditions hold,
    and ``criteria_status``: ``"not_requested"`` by default, or
    ``"passed"``/``"failed"`` per the explicitly passed thresholds.

    Raises :class:`CompareError` (with a machine-readable ``reason``) for
    out-of-scope inputs, incompatible runs and missing required
    information; nothing is ever written and no backend is constructed.
    """
    if max_position_rms_A is not None:
        max_position_rms_A = _check_threshold(max_position_rms_A,
                                              "max_position_rms_A")
    if max_velocity_rms_A_fs is not None:
        max_velocity_rms_A_fs = _check_threshold(max_velocity_rms_A_fs,
                                                 "max_velocity_rms_A_fs")
    reference = _load_run(reference_run, role="reference")
    candidate = _load_run(candidate_run, role="candidate")
    initial = _check_compatible(reference, candidate)
    matched = _match_times(reference, candidate)
    matched_times = candidate.times_fs.tolist()
    n_atoms = len(reference.numbers)

    dq = candidate.positions - reference.positions[matched]
    position_rms = _rms_per_time(dq, n_atoms)
    velocity_available = (reference.momenta is not None
                          and candidate.momenta is not None)
    if velocity_available:
        dv = (_velocities(candidate.momenta, candidate.masses)
              - _velocities(reference.momenta[matched], reference.masses))
        velocity_rms = _rms_per_time(dv, n_atoms)
    elif max_velocity_rms_A_fs is not None:
        missing = [run.run_id for run in (reference, candidate)
                   if run.momenta is None]
        raise CompareError(
            "a velocity RMS criterion was requested, but run(s) "
            f"{missing} have no real recorded momenta; velocity metrics "
            "are unavailable (positions were still verified)",
            reason="missing_information")
    else:
        velocity_rms = None

    criteria: dict = {}
    requested = False
    all_passed = True
    if max_position_rms_A is not None:
        requested = True
        observed = float(position_rms.max())
        passed = bool(observed <= max_position_rms_A)
        all_passed = all_passed and passed
        criteria["max_position_rms_A"] = {
            "threshold_A": max_position_rms_A,
            "observed_max_A": observed, "passed": passed}
    if max_velocity_rms_A_fs is not None:
        requested = True
        observed = float(velocity_rms.max())
        passed = bool(observed <= max_velocity_rms_A_fs)
        all_passed = all_passed and passed
        criteria["max_velocity_rms_A_fs"] = {
            "threshold_A_fs": max_velocity_rms_A_fs,
            "observed_max_A_fs": observed, "passed": passed}

    return {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "statement": (
            "same-initial-state pointwise comparison of two completed "
            "fixed-cell NVE trajectories at identical physical times; "
            "stored continuous coordinates are subtracted directly (no "
            "minimum image, no alignment, no interpolation, no atom "
            "reordering).  This is a descriptive accuracy report, not an "
            "evaluation platform: it draws no reliability or "
            "recommendation conclusion and estimates no speed."),
        "reference_run": {"run_dir": str(reference.run_dir),
                          "run_id": reference.run_id,
                          "driver": reference.driver},
        "candidate_run": {"run_dir": str(candidate.run_dir),
                          "run_id": candidate.run_id,
                          "driver": candidate.driver},
        "structure": {"n_atoms": n_atoms,
                      "pbc": reference.pbc.tolist(),
                      "species_order_match": True,
                      "masses_match": True,
                      "cell_match": True},
        "initial_state": initial,
        "time_axis": {
            "match_tolerance_fs": TIME_TOLERANCE_FS,
            "interpolation": "none",
            "start_fs": matched_times[0],
            "end_fs": matched_times[-1],
            "matched_points": len(matched_times),
            "reference_complete_states":
                int(reference.coverage["complete_states"]),
            "candidate_complete_states":
                int(candidate.coverage["complete_states"]),
        },
        "coverage": {"reference": reference.coverage,
                     "candidate": candidate.coverage},
        "velocity_available": velocity_available,
        "position_rms_A": {"times_fs": matched_times,
                           "per_time": position_rms.tolist(),
                           "max": float(position_rms.max())},
        "velocity_rms_A_fs": (
            {"times_fs": matched_times,
             "per_time": velocity_rms.tolist(),
             "max": float(velocity_rms.max())}
            if velocity_available else
            {"status": "unavailable",
             "reason": "at least one run has no real recorded momenta; "
                       "an ASE default all-zero array is never treated as "
                       "velocities"}),
        "hamiltonian": {
            "reference": _hamiltonian_block(reference),
            "candidate": _hamiltonian_block(candidate),
            "note": ("descriptive only; each trajectory is zeroed at its "
                     "own run's start and absolute H values are never "
                     "compared across different energy zeros; this "
                     "version defines no energy-drift threshold"),
        },
        "criteria": criteria,
        "criteria_status": ("not_requested" if not requested
                            else "passed" if all_passed else "failed"),
    }


def format_comparison(report: dict) -> str:
    """Human-readable rendering of :func:`compare_runs`' report."""
    reference = report["reference_run"]
    candidate = report["candidate_run"]
    initial = report["initial_state"]
    axis = report["time_axis"]
    lines = [
        (f"compare: reference {reference['run_id']} ({reference['driver']})"
         f" vs candidate {candidate['run_id']} ({candidate['driver']})"),
        ("  scope             : same-initial-state pointwise comparison of "
         "fixed-cell NVE trajectories (stored continuous coordinates; no "
         "alignment, no interpolation)"),
        (f"  initial state     : t = {initial['time_fs']} fs, positions "
         f"match (max |dq| = {initial['max_position_difference_A']:.3e} A)"),
    ]
    if initial["initial_momenta_match"] == "unavailable":
        lines.append("  initial momenta   : unavailable (at least one run "
                     "has no recorded momenta; the full initial state was "
                     "NOT verified)")
    else:
        lines.append(f"  initial momenta   : match (max |dp| = "
                     f"{initial['max_momentum_difference_ase']:.3e} ASE "
                     "units)")
    lines.append(
        f"  matched times     : {axis['matched_points']} points, "
        f"{axis['start_fs']} .. {axis['end_fs']} fs (reference complete "
        f"states {axis['reference_complete_states']}, candidate "
        f"{axis['candidate_complete_states']})")
    for role in ("reference", "candidate"):
        coverage = report["coverage"][role]
        tail = coverage["incomplete_tail_evaluations"]
        tail_note = (" (reported, never trajectory points)" if tail
                     else "")
        lines.append(
            f"  coverage {role:9s}: {coverage['complete_states']} complete "
            f"states of {coverage['committed_evaluations']} committed "
            f"evaluations; {tail} incomplete tail evaluation(s)"
            + tail_note)
    position = report["position_rms_A"]
    lines.append(f"  position rms      : max {position['max']:.6e} A over "
                 "the matched window (per-atom normalization; per-time "
                 "array in the JSON report)")
    velocity = report["velocity_rms_A_fs"]
    if report["velocity_available"]:
        lines.append(f"  velocity rms      : max {velocity['max']:.6e} "
                     "A/fs over the matched window")
    else:
        lines.append(f"  velocity rms      : unavailable ({velocity['reason']})")
    for role in ("reference", "candidate"):
        block = report["hamiltonian"][role]
        if block["status"] == "available":
            lines.append(
                f"  H drift {role:9s}: max |(H(t)-H(0))|/N = "
                f"{block['max_abs_drift_per_atom_eV']:.6e} eV/atom over "
                f"{len(block['times_fs'])} states (descriptive; per-run "
                "energy zero)")
        else:
            lines.append(f"  H drift {role:9s}: unavailable — "
                         f"{block['reason']}")
    status = report["criteria_status"]
    if status == "not_requested":
        lines.append("  criteria          : not requested — metrics "
                     "reported only, no pass/fail conclusion")
    else:
        for name, criterion in report["criteria"].items():
            threshold = next(value for key, value in criterion.items()
                             if key.startswith("threshold"))
            observed = next(value for key, value in criterion.items()
                            if key.startswith("observed"))
            lines.append(f"  criterion {name}: observed max {observed:.6e} "
                         f"vs threshold {threshold:.6e} — "
                         + ("PASS" if criterion["passed"] else "FAIL"))
        lines.append(f"  criteria          : {status}")
    return "\n".join(lines)
