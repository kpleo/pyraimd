"""M3A-5 acceptance: fixed-model adaptive NVT on analytic potentials.

A. Same conservative potential in reference and base: direct ASE Langevin,
   plain NVT and adaptive NVT agree bit-for-bit on positions, complete
   momenta, the actual force-request configurations and physical time; the
   bath stream is independent of the check stream (M3A-4).
B. Anisotropic harmonic reference/base with slightly different Hessians:
   the decision sees the realized displacement (random transverse part
   included), segment residual work closes two ways, and a forced re-anchor
   closes the old segment with the old correction.
C. Cross-process resume: continuous vs split runs agree on the complete
   state, the ledger and the independent call counts, with true subprocess
   exits in both commit windows and a sentinel proving the recovery phase
   never touches a live backend.

All runs are runner-level (the config-level ensemble=nvt + mode=adaptive
gate opens only after this acceptance passes) on a few atoms and a few
tens of steps; zero real DFT budget.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from test_nvt import events, write_nvt

from pyraimd2.backends.harmonic import HarmonicReference, HarmonicSurrogate
from pyraimd2.config import load_config
from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.loop.energetic import _EnergeticLangevin
from pyraimd2.loop.integrators import IntegratorSpec, derive_stream_seed
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.workflows import run_workflow

K, R0 = 1.0, 0.9
DT, TEMP, FRICTION = 0.5, 300.0, 0.01
POSITIONS = [[0.85, 0.9, 0.9], [0.95, 0.9, 0.9]]
MOMENTA = [[0.05, 0.02, 0.0], [-0.03, 0.01, 0.0]]
USER_THERMOSTAT_SEED = 123
BATH_SEED = derive_stream_seed(USER_THERMOSTAT_SEED, "thermostat")


def _spec(seed=BATH_SEED, temperature=TEMP):
    return IntegratorSpec(algorithm="langevin", ensemble="nvt",
                          timestep_fs=DT, temperature_K=temperature,
                          friction_per_fs=FRICTION, thermostat_seed=seed)


class _AseHarmonic(Calculator):
    """The builtin harmonic law as a plain ASE calculator, logging every
    force-request configuration."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, log):
        super().__init__()
        self.log = log

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        dr = self.atoms.positions - R0
        self.log.append(self.atoms.positions.copy())
        self.results = {"energy": 0.5 * K * float((dr**2).sum()),
                        "forces": -K * dr}


class _LoggingSurrogate:
    """HarmonicSurrogate (bias=0: the same potential as the reference)
    logging every prediction's configuration."""

    def __init__(self, log, bias=0.0):
        self.inner = HarmonicSurrogate(k=K, r0=R0, bias=bias)
        self.log = log

    @property
    def fingerprint(self):
        return self.inner.fingerprint

    @property
    def capabilities(self):
        return self.inner.capabilities

    def predict(self, atoms):
        self.log.append(atoms.positions.copy())
        return self.inner.predict(atoms)


def _atoms(positions=POSITIONS, momenta=MOMENTA, masses=None, fixed=None):
    atoms = Atoms("H" * len(positions), positions=positions)
    if masses is not None:
        atoms.set_masses(masses)
    atoms.set_momenta(np.asarray(momenta, dtype=float))
    if fixed:
        atoms.set_constraint(FixAtoms(indices=list(fixed)))
    return atoms


def _direct_ase(*, steps, masses=None, fixed=None, seed=BATH_SEED,
                temperature=TEMP, positions=POSITIONS, momenta=MOMENTA):
    """ASE's own Langevin on the same law/stream; returns per-boundary
    positions/momenta and the force-request configurations."""
    log: list[np.ndarray] = []
    atoms = _atoms(positions, momenta, masses, fixed)
    atoms.calc = _AseHarmonic(log)
    dyn = Langevin(atoms, DT * units.fs, temperature_K=temperature,
                   friction=FRICTION / units.fs, fixcm=False,
                   rng=np.random.default_rng(seed))
    atoms.get_forces()  # prime the cache like the drivers' boundary read
    out = [(atoms.positions.copy(), atoms.get_momenta().copy())]
    for _ in range(steps):
        dyn.step()
        out.append((atoms.positions.copy(), atoms.get_momenta().copy()))
    return out, log


def _adaptive(run_dir, *, steps, check_probability=0.25, check_seed=7,
              surrogate=None, engine=None, spec=None, masses=None,
              fixed=None, positions=POSITIONS, momenta=MOMENTA,
              time_cap_fs=100.0, force_budget=0.5, temperature=TEMP,
              transverse_cap=0.1, checkpoint_interval=4):
    run_dir.mkdir(parents=True)
    atoms = _atoms(positions, momenta, masses, fixed)
    runner = EnergeticRunner(
        atoms,
        surrogate if surrogate is not None else HarmonicSurrogate(k=K, r0=R0, bias=0.0),
        engine if engine is not None else HarmonicReference(k=K, r0=R0),
        Store(run_dir / "trajectory.db"), "run", run_dir=run_dir,
        event_log=EventLog(run_dir),
        checkpoint_interval_steps=checkpoint_interval,
        force_budget=force_budget, timestep_fs=DT, time_cap_fs=time_cap_fs,
        transverse_cap=transverse_cap,
        check_probability=check_probability, check_seed=check_seed,
        integrator_spec=spec if spec is not None else _spec(temperature=temperature))
    runner.run(steps)
    runner.close()
    return run_dir


def _rows(run_dir, run_id="run"):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def _complete_boundaries(run_dir, *, fixed=None):
    """The complete boundary (positions, full momenta) of every committed
    evaluation: row 0 carries its own; later boundaries complete from the
    commit's recorded bath increments, exactly as resume does."""
    store = Store(run_dir / "trajectory.db")
    evs = events(run_dir)
    rows = _rows(run_dir)
    out = []
    dyn = None
    for index, row in enumerate(rows):
        atoms = row.toatoms()
        if index == 0:
            out.append((atoms.positions.copy(), atoms.get_momenta().copy()))
            continue
        commit = next(e for e in evs if e["type"] == "evaluation_committed"
                      and int((e.get("context") or {})
                              ["evaluation_id"]) == index)
        bath = commit["bath_step"]
        if dyn is None:
            template = _atoms(positions=atoms.positions,
                              momenta=atoms.get_momenta(),
                              masses=atoms.get_masses())
            dyn = _EnergeticLangevin(
                template, DT * units.fs, temperature_K=TEMP,
                friction=FRICTION / units.fs, fixcm=False,
                rng=np.random.default_rng(0))
        _, driving_forces = store.driving_label_for_row(row)
        projection = None
        if fixed:
            from pyraimd2.loop.constraints import FixAtomsProjection

            projection = FixAtomsProjection(len(atoms), list(fixed))
        momenta = dyn.complete_momenta(
            rows[index - 1].toatoms().positions, atoms.positions,
            np.array(bath["rnd_pos"]), np.array(bath["rnd_vel"]),
            driving_forces, projection)
        out.append((atoms.positions.copy(), momenta))
    return out


def _assert_subsequence(log, sequence):
    """Every boundary configuration is requested in order; extra entries
    are the anchor's off-trajectory probes."""
    cursor = 0
    for expected in sequence:
        while cursor < len(log) and not np.array_equal(log[cursor], expected):
            cursor += 1
        assert cursor < len(log), \
            f"a force-request configuration was never requested: {expected}"
        cursor += 1


# --- A. same potential: wiring and state -------------------------------------


def test_a_three_way_bit_identical(tmp_path):
    steps = 12
    direct, direct_log = _direct_ase(steps=steps)

    plain = load_config(write_nvt(
        tmp_path / "plain", steps=steps, thermostat_seed=USER_THERMOSTAT_SEED,
        momenta=MOMENTA, checkpoint_interval=4))
    run_workflow(plain, verbose=False, handle_sigint=False)

    predict_log: list[np.ndarray] = []
    adaptive_dir = _adaptive(tmp_path / "adaptive", steps=steps,
                             surrogate=_LoggingSurrogate(predict_log))
    # The plain driver derives the bath seed by role from the config value;
    # the adaptive spec above carries that same effective seed.
    assert plain.run.directory != adaptive_dir

    plain_rows = _rows(plain.run.directory, "nvt-demo")
    adaptive_rows = _rows(adaptive_dir)
    assert len(plain_rows) == len(adaptive_rows) == steps + 1

    # Positions of every committed evaluation, three ways, bit-identical.
    for index, (direct_positions, _) in enumerate(direct):
        np.testing.assert_array_equal(
            plain_rows[index].toatoms().positions, direct_positions)
        np.testing.assert_array_equal(
            adaptive_rows[index].toatoms().positions, direct_positions)

    # The actual force-request configurations: every boundary was requested
    # in order from the surrogate (off-trajectory anchor probes aside) and
    # from the direct ASE calculator.
    _assert_subsequence(predict_log, [positions for positions, _ in direct])
    assert len(direct_log) == steps + 1  # priming + one per step
    for index, (positions, _) in enumerate(direct):
        np.testing.assert_array_equal(direct_log[index], positions)

    # Complete momenta: plain rows carry them directly; the adaptive
    # boundaries complete from the committed bath increments.
    adaptive_boundaries = _complete_boundaries(adaptive_dir)
    for index, (_, direct_momenta) in enumerate(direct):
        np.testing.assert_array_equal(
            plain_rows[index].toatoms().get_momenta(), direct_momenta)
        positions, momenta = adaptive_boundaries[index]
        np.testing.assert_array_equal(positions, direct[index][0])
        np.testing.assert_array_equal(momenta, direct_momenta)

    # With an identically zero residual the probe response is degenerate —
    # the honest safe fallback keeps every evaluation on the reference
    # route (no anchor is ever extrapolated from an unmeasurable
    # correction), which is also why the three-way comparison is exact.
    proposals = [e for e in events(adaptive_dir)
                 if e["type"] == "evaluation_proposed"]
    assert not any(p["accepted"] for p in proposals)
    assert {p["reason"] for p in proposals} <= {
        "initial_reference", "direction_unavailable_reference"}

    # Physical time: every step record names its boundary time.
    for run_dir in (plain.run.directory, adaptive_dir):
        for event in events(run_dir):
            if event["type"] == "step_completed":
                assert event["physical_time_fs"] == pytest.approx(
                    (event["step_id"] + 1) * DT)
                assert event["digest_format"] == "boundary-v2"


def test_a_check_frequency_never_touches_the_bath_stream(tmp_path):
    # Same raw user seed for bath and check: the role-derived streams
    # differ, and the RUN_START record says so (M3A-4).
    runs_start = next(e for e in events(
        _adaptive(tmp_path / "streams", steps=1, check_probability=0.0,
                  check_seed=USER_THERMOSTAT_SEED)) if e["type"] == "run_start")
    streams = runs_start["streams"]
    assert streams["scheme"] == "role-derive-v1"
    assert streams["thermostat_seed"] == derive_stream_seed(
        USER_THERMOSTAT_SEED, "thermostat")
    assert streams["check_seed"] == derive_stream_seed(
        USER_THERMOSTAT_SEED, "verification")
    assert streams["thermostat_seed"] != streams["check_seed"]

    # base == reference: no check can violate, so different check rates
    # must produce bit-identical trajectories and bath streams.  (With an
    # identically zero residual the response is degenerate, so both runs
    # stay on the reference route — the stream isolation claim is exact.)
    checked = _adaptive(tmp_path / "checked", steps=12, check_probability=0.5,
                        check_seed=7)
    unchecked = _adaptive(tmp_path / "unchecked", steps=12,
                          check_probability=0.0)
    rows_a, rows_b = _rows(checked), _rows(unchecked)
    assert len(rows_a) == len(rows_b)
    for row_a, row_b in zip(rows_a, rows_b):
        np.testing.assert_array_equal(row_a.toatoms().positions,
                                      row_b.toatoms().positions)
        np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                      row_b.toatoms().get_momenta())
    baths_a = [e["thermostat_rng"]["state"]["state"] for e in events(checked)
               if e["type"] == "step_completed"]
    baths_b = [e["thermostat_rng"]["state"]["state"] for e in events(unchecked)
               if e["type"] == "step_completed"]
    assert baths_a == baths_b

    # A small but nonzero residual makes the surrogate route admissible:
    # checks then fire on accepted evaluations, bill reference calls
    # exactly, and — none violating — leave the trajectory bit-identical.
    # (transverse_cap=1 admits the bath's random transverse displacement —
    # the envelope still counts it; admitting is the point of this wiring
    # test.  The default cap's honest refusal behavior is acceptance B.)
    def biased():
        return HarmonicSurrogate(k=K, r0=R0, bias=1e-3)
    on = _adaptive(tmp_path / "bias-on", steps=12, check_probability=0.6,
                   check_seed=11, surrogate=biased(), force_budget=0.5,
                   transverse_cap=1.0)
    off = _adaptive(tmp_path / "bias-off", steps=12, check_probability=0.0,
                    surrogate=biased(), force_budget=0.5, transverse_cap=1.0)
    accepted_on = [e for e in events(on) if e["type"] == "evaluation_proposed"
                   and e["accepted"]]
    assert accepted_on and any(e["checked"] for e in accepted_on)
    assert not [e for e in events(on)
                if e["type"] == "evaluation_committed" and e["violation"]]
    rows_on, rows_off = _rows(on), _rows(off)
    assert len(rows_on) == len(rows_off)
    for row_on, row_off in zip(rows_on, rows_off):
        np.testing.assert_array_equal(row_on.toatoms().positions,
                                      row_off.toatoms().positions)
        np.testing.assert_array_equal(row_on.toatoms().get_momenta(),
                                      row_off.toatoms().get_momenta())
    reference_on = [e for e in events(on) if e["type"] == "task"
                    and e.get("operation") == "reference"
                    and e.get("purpose") == "verification"]
    n_checks = sum(1 for e in accepted_on if e["checked"])
    assert len(reference_on) == n_checks > 0  # every check billed, once
    reference_off = [e for e in events(off) if e["type"] == "task"
                     and e.get("operation") == "reference"
                     and e.get("purpose") == "verification"]
    assert not reference_off


def test_a_different_masses_and_fixatoms_match_direct_ase(tmp_path):
    steps = 8
    masses = [1.0, 2.0]
    momenta = [[0.0, 0.0, 0.0], [-0.04, 0.02, 0.01]]
    direct, _ = _direct_ase(steps=steps, masses=masses, fixed=[0],
                            momenta=momenta)
    adaptive_dir = _adaptive(tmp_path / "fixed", steps=steps, masses=masses,
                             fixed=[0], momenta=momenta)
    boundaries = _complete_boundaries(adaptive_dir, fixed=[0])
    assert len(boundaries) == len(direct) == steps + 1
    for (positions, momenta_ad), (positions_d, momenta_d) in zip(boundaries,
                                                                 direct):
        np.testing.assert_array_equal(positions, positions_d)
        np.testing.assert_array_equal(momenta_ad, momenta_d)
        np.testing.assert_array_equal(positions[0], POSITIONS[0])  # fixed
        assert np.all(momenta_ad[0] == 0.0)


def test_a_zero_displacement_falls_back_safely(tmp_path):
    # At the minimum with zero momenta and a zero-temperature bath the
    # realized displacement is exactly zero: no probe direction exists, so
    # every evaluation stays on the reference route with the reason
    # recorded — the degenerate-direction fallback, never a guessed anchor.
    run_dir = _adaptive(tmp_path / "degenerate", steps=4, temperature=0.0,
                        spec=_spec(temperature=0.0),
                        positions=[[R0, R0, R0]],
                        momenta=[[0.0, 0.0, 0.0]],
                        check_probability=0.0)
    rows = _rows(run_dir)
    assert len(rows) == 5
    for row in rows:
        np.testing.assert_array_equal(row.toatoms().positions[0],
                                      np.array([R0, R0, R0]))
    proposals = [e for e in events(run_dir) if e["type"] == "evaluation_proposed"]
    assert [bool(p["accepted"]) for p in proposals] == [False] * 5
    assert {p["reason"] for p in proposals} <= {
        "initial_reference", "direction_unavailable_reference"}
    summary = next(e for e in events(run_dir) if e["type"] == "run_summary")
    assert summary["n_reference"] == 5  # every evaluation: one anchor call
    # the step records still bind the complete (stationary) boundary
    for event in events(run_dir):
        if event["type"] == "step_completed":
            assert event["digest_format"] == "boundary-v2"
            assert event["boundary_digest"]


# --- B. analytic residual: decisions and residual work -----------------------

_B_CENTER = np.array([R0, R0, R0])
_B_K_REF = np.array([1.0, 1.62, 0.4])
_B_K_BASE = np.array([1.0, 1.5, 0.4])


def _aniso_energy(k3, positions):
    dr = np.asarray(positions, dtype=float) - _B_CENTER
    return 0.5 * float((k3 * dr**2).sum())


def _aniso_forces(k3, positions):
    return -(k3 * (np.asarray(positions, dtype=float) - _B_CENTER))


class _AnisoReference:
    """Engine: U = 1/2 sum k3 (x - center)^2 per Cartesian component."""

    name = "aniso-reference"

    def __init__(self, k3):
        self.k3 = np.asarray(k3, dtype=float)
        self.attempts = 0

    @property
    def fingerprint(self):
        import hashlib

        return ("aniso-reference:"
                + hashlib.sha256(self.k3.tobytes()).hexdigest()[:12])

    def compute(self, atoms):
        self.attempts += 1
        return EngineResult(_aniso_energy(self.k3, atoms.positions),
                            _aniso_forces(self.k3, atoms.positions), None, 0.0)


class _AnisoModel:
    """Surrogate with a slightly different Hessian (y only)."""

    def __init__(self, k3):
        self.k3 = np.asarray(k3, dtype=float)

    @property
    def fingerprint(self):
        import hashlib

        return ("aniso-model:"
                + hashlib.sha256(self.k3.tobytes()).hexdigest()[:12])

    @property
    def capabilities(self):
        return None

    def predict(self, atoms):
        return SurrogatePrediction(_aniso_energy(self.k3, atoms.positions),
                                   _aniso_forces(self.k3, atoms.positions),
                                   None, np.full(len(atoms), np.nan))


def _b_world(tmp_path, **overrides):
    engine = _AnisoReference(_B_K_REF)
    options = {"steps": 14, "surrogate": _AnisoModel(_B_K_BASE),
               "engine": engine, "time_cap_fs": 1.0, "force_budget": 0.02,
               "check_probability": 1.0, "check_seed": 3,
               "checkpoint_interval": 4, "temperature": 100.0,
               "spec": _spec(temperature=100.0),
               "momenta": [[0.1, 0.04, 0.0], [-0.06, 0.02, 0.0]],
               "positions": [[0.82, 0.9, 0.9], [0.98, 0.9, 0.9]]}
    options.update(overrides)
    run_dir = _adaptive(tmp_path, **options)
    return run_dir, engine


def _anchors(run_dir):
    """Segment anchor records keyed by segment id, from the row metadata."""
    out = {}
    for row in _rows(run_dir):
        record = (row.data.get("metadata") or {}).get("new_anchor")
        if record is not None:
            out[int(record["segment_id"])] = record
    return out


def _commit_by_evaluation(run_dir):
    return {int((e.get("context") or {})["evaluation_id"]): e
            for e in events(run_dir) if e["type"] == "evaluation_committed"}


def test_b_decision_sees_the_realized_displacement(tmp_path):
    from pyraimd2.loop.energetic import _response_from_dict

    run_dir, _ = _b_world(tmp_path / "b")
    proposals = [e for e in events(run_dir)
                 if e["type"] == "evaluation_proposed"]
    # the deterministic seed admits at least one accepted prefix and shows
    # refusals with re-anchoring (segments advance)
    assert any(p["accepted"] for p in proposals)
    assert any(not p["accepted"] for p in proposals)
    segments = {p.get("segment_id") for p in proposals if p.get("segment_id")}
    assert len(segments) >= 2

    admitted_with_transverse = 0
    for proposal in proposals:
        anchor = proposal.get("anchor_record")
        if anchor is None:
            continue
        index = int(proposal["context"]["evaluation_id"])
        displacement = (np.array(proposal["positions_A"], dtype=float)
                        - np.array(anchor["positions_A"], dtype=float))
        responses = [_response_from_dict(r)
                     for r in anchor["calibration"]["responses"]]
        assert len(responses) == len(proposal["forecasts"])
        elapsed = (index - int(anchor["evaluation_index"])) * DT
        for response, recorded in zip(responses, proposal["forecasts"]):
            forecast = response.forecast(
                displacement, elapsed, force_budget=0.02,
                numerical_floor=0.0, time_cap_fs=1.0, transverse_cap=0.1)
            # the recorded decision inputs recompute exactly from the
            # actual, realized displacement — random transverse part included
            assert forecast.linear_error == pytest.approx(
                recorded["linear_error_eV_A"], rel=0, abs=1e-15)
            assert forecast.envelope == pytest.approx(
                recorded["envelope_eV_A"], rel=0, abs=1e-15)
            assert forecast.predicted_work == pytest.approx(
                recorded["predicted_work_eV"], rel=0, abs=1e-15)
            assert forecast.transverse_fraction == pytest.approx(
                recorded["transverse_fraction"], rel=0, abs=1e-15)
            assert forecast.admitted == recorded["admitted"]
        if (proposal["accepted"] and proposal["forecasts"]
                and proposal["forecasts"][0]["transverse_fraction"] > 0.01):
            admitted_with_transverse += 1
    # a deterministic-seed case where the bath displacement is not collinear
    # with the boundary velocity: the admitted decision consumed the
    # transverse (random) component — the old Verlet extrapolation bug
    # class.
    assert admitted_with_transverse >= 1


def test_b_actual_step_is_not_the_verlet_extrapolation(tmp_path):
    # For accepted steps the actual displacement differs materially from the
    # deterministic velocity-Verlet extrapolation of the previous boundary
    # (the bath increment is in the scheduled configuration).
    run_dir, _ = _b_world(tmp_path / "b")
    boundaries = _complete_boundaries(run_dir)
    store = Store(run_dir / "trajectory.db")
    proposals = {int((e.get("context") or {})["evaluation_id"]): e
                 for e in events(run_dir)
                 if e["type"] == "evaluation_proposed"}
    deviations = []
    for index in range(2, len(boundaries)):
        if not proposals[index]["accepted"]:
            continue
        positions_prev, momenta_prev = boundaries[index - 1]
        row_prev = _rows(run_dir)[index - 1]
        _, forces_prev = store.driving_label_for_row(row_prev)
        masses = row_prev.toatoms().get_masses()[:, None]
        verlet = positions_prev + DT * units.fs * (
            momenta_prev + 0.5 * DT * units.fs * forces_prev) / masses
        actual = boundaries[index][0]
        deviations.append(float(np.max(np.abs(actual - verlet))))
    assert deviations and max(deviations) > 1e-4  # bath-scale, not rounding


def test_b_residual_work_closes_two_ways_per_segment(tmp_path):
    from pyraimd2.energetics.work import integrate_residual_work, residual_work

    run_dir, _ = _b_world(tmp_path / "b")
    rows = _rows(run_dir)
    anchors = _anchors(run_dir)
    commits = _commit_by_evaluation(run_dir)
    checked = [e for e in commits.values() if e.get("observed") is not None]
    assert checked, "expected at least one label-carrying evaluation"
    boundaries = _complete_boundaries(run_dir)
    masses = rows[0].toatoms().get_masses()[:, None]

    def kinetic(boundary):
        return float((0.5 * boundaries[boundary][1]**2 / masses).sum())

    def anchor_energy(correction, anchor_positions, pos):
        return (_aniso_energy(_B_K_BASE, pos)
                - float((correction * (pos - anchor_positions)).sum()))

    for commit in checked:
        index = int(commit["context"]["evaluation_id"])
        segment = int(commit["segment_id"])
        anchor = anchors[segment]
        anchor_index = int(anchor["evaluation_index"])
        anchor_positions = np.array(anchor["positions_A"], dtype=float)
        correction = np.array(anchor["correction_eV_A"], dtype=float)
        positions = rows[index].toatoms().positions
        # endpoint identity, recomputed analytically from the potentials:
        # W_R,s = ΔU_ref − ΔU_anchor,s (residual_work's own convention)
        endpoint = residual_work(
            anchor_positions, positions,
            _aniso_energy(_B_K_BASE, anchor_positions),
            _aniso_energy(_B_K_BASE, positions),
            float(anchor["reference_energy_eV"]),
            float(rows[index].data["engine"]["energy"]),
            correction)
        assert endpoint == pytest.approx(
            commit["observed"]["endpoint_work_eV"], rel=0, abs=1e-12)
        # trapezoidal path integral of the independent force residual
        # R_s = F_base + c_s − F_ref along the recorded segment states;
        # harmonic forces are linear, so the segment quadrature is exact.
        path = [rows[j].toatoms().positions
                for j in range(anchor_index, index + 1)]
        residuals = [_aniso_forces(_B_K_BASE, p) + correction
                     - _aniso_forces(_B_K_REF, p) for p in path]
        trapezoidal = float(integrate_residual_work(
            np.array(path), np.array(residuals))[-1].sum())
        assert trapezoidal == pytest.approx(endpoint, rel=0, abs=1e-12)

        # ΔH_ref = W_R,s + ΔH_anchor,s — an algebra identity on the
        # recorded numbers, checked with the complete boundary momenta.
        delta_h_anchor = (kinetic(index)
                          + anchor_energy(correction, anchor_positions,
                                          positions)
                          - kinetic(anchor_index)
                          - anchor_energy(correction, anchor_positions,
                                          anchor_positions))
        delta_h_ref = (kinetic(index)
                       + float(rows[index].data["engine"]["energy"])
                       - kinetic(anchor_index)
                       - float(anchor["reference_energy_eV"]))
        assert delta_h_ref == pytest.approx(endpoint + delta_h_anchor,
                                            rel=0, abs=1e-12)
        # predicted work and reference-endpoint work stay distinct fields
        proposal = next(e for e in events(run_dir)
                        if e["type"] == "evaluation_proposed"
                        and int(e["context"]["evaluation_id"]) == index)
        if proposal["forecasts"]:
            assert "predicted_work_eV" in proposal["forecasts"][0]
            assert "endpoint_work_eV" in commit["observed"]


def test_b_forced_reanchor_closes_old_segment_with_old_correction(tmp_path):
    from pyraimd2.energetics.work import residual_work

    run_dir, _engine = _b_world(tmp_path / "b")
    rows = _rows(run_dir)
    anchors = _anchors(run_dir)
    commits = _commit_by_evaluation(run_dir)
    # the time cap forces re-anchoring: several segments, each with its own
    # correction (the anchors sit at different positions, and the residual
    # is position-dependent in y)
    assert len(anchors) >= 2
    corrections = {segment: np.array(record["correction_eV_A"], dtype=float)
                   for segment, record in anchors.items()}
    for first, second in zip(sorted(corrections), sorted(corrections)[1:]):
        assert not np.allclose(corrections[first], corrections[second],
                               rtol=0, atol=1e-15)

    for commit in commits.values():
        if commit.get("observed") is None:
            continue
        index = int(commit["context"]["evaluation_id"])
        segment = int(commit["segment_id"])
        anchor = anchors[segment]
        anchor_positions = np.array(anchor["positions_A"], dtype=float)
        positions = rows[index].toatoms().positions
        base = _aniso_energy
        own = residual_work(
            anchor_positions, positions, base(_B_K_BASE, anchor_positions),
            base(_B_K_BASE, positions), float(anchor["reference_energy_eV"]),
            float(rows[index].data["engine"]["energy"]),
            corrections[segment])
        # the recorded work used THIS segment's correction …
        assert own == pytest.approx(commit["observed"]["endpoint_work_eV"],
                                    rel=0, abs=1e-12)
        # … and a neighboring segment's correction would give a materially
        # different number — cross-segment work is never interchangeable.
        others = [s for s in corrections if s != segment]
        if others:
            foreign = residual_work(
                anchor_positions, positions, base(_B_K_BASE, anchor_positions),
                base(_B_K_BASE, positions),
                float(anchor["reference_energy_eV"]),
                float(rows[index].data["engine"]["energy"]),
                corrections[others[0]])
            assert foreign != pytest.approx(own, rel=0, abs=1e-12)


def test_b_call_counts_and_error_records(tmp_path):
    run_dir, engine = _b_world(tmp_path / "b")
    evs = events(run_dir)
    reference_tasks = [e for e in evs if e["type"] == "task"
                       and e.get("operation") == "reference"]
    # every real engine execution is ledgered exactly once; the toy engine
    # has no cache, so tasks never hit the label cache here
    assert engine.attempts == len(reference_tasks)
    assert all(t["status"] == "success" for t in reference_tasks)
    by_purpose = {}
    for task in reference_tasks:
        by_purpose[task["purpose"]] = by_purpose.get(task["purpose"], 0) + 1
    proposals = [e for e in evs if e["type"] == "evaluation_proposed"]
    # anchor-route calls bill as "anchor" (initial) and "refusal"; probes:
    # 4 per calibration that produced an anchor; checks: one per checked
    # accept
    n_refused = sum(1 for p in proposals
                    if not p["accepted"] and p["reason"] != "initial_reference")
    n_checked = sum(1 for p in proposals if p["accepted"] and p["checked"])
    anchors_in_rows = _anchors(run_dir)
    assert by_purpose.get("anchor", 0) == 1  # the initial evaluation
    assert by_purpose.get("refusal", 0) == n_refused
    assert by_purpose.get("probe", 0) == 4 * len(anchors_in_rows)
    assert by_purpose.get("verification", 0) == n_checked
    assert engine.attempts == 1 + n_refused + 4 * len(anchors_in_rows) + n_checked
    # predicted vs measured error are both recorded per checked evaluation
    for commit in _commit_by_evaluation(run_dir).values():
        if commit.get("observed") is not None and commit["checked"]:
            observed = commit["observed"]
            assert observed["force_budget_exceeded"] == (
                observed["max_force_error_eV_A"] > 0.02)


# --- C. cross-process resume: real costs -------------------------------------

_C_CHILD = '''
import os, sys
from pathlib import Path

import numpy as np
from ase import Atoms

from pyraimd2.backends.harmonic import HarmonicReference, HarmonicSurrogate
from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.loop.integrators import IntegratorSpec, derive_stream_seed
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store

run_dir = Path(sys.argv[1])
steps = int(sys.argv[2])
kill_key = os.environ.get("KILL_KEY")
kill_after = os.environ.get("KILL_AFTER") == "1"
count_file = Path(os.environ["COUNT_FILE"])
drift_scale = float(os.environ.get("DRIFT", "0.0"))


class CountingReference:
    name = "counting-harmonic-reference"

    def __init__(self):
        self.inner = HarmonicReference(k=1.0, r0=0.9)

    @property
    def fingerprint(self):
        return self.inner.fingerprint

    @property
    def capabilities(self):
        return self.inner.capabilities

    def compute(self, atoms):
        n = int(count_file.read_text()) + 1 if count_file.exists() else 1
        count_file.write_text(str(n))
        label = self.inner.compute(atoms)
        drift = drift_scale * n
        return EngineResult(label.energy + drift, label.forces + drift,
                            None, 0.0)


if kill_key:
    original = EventLog.append_once

    def patched(self, key, event_type, payload):
        if key == kill_key and not kill_after:
            os._exit(73)
        result = original(self, key, event_type, payload)
        if key == kill_key and kill_after:
            os._exit(73)
        return result

    EventLog.append_once = patched

atoms = Atoms("H2", positions=[[0.85, 0.9, 0.9], [0.95, 0.9, 0.9]])
atoms.set_momenta([[0.05, 0.02, 0.0], [-0.03, 0.01, 0.0]])
spec = IntegratorSpec(algorithm="langevin", ensemble="nvt", timestep_fs=0.5,
                      temperature_K=300.0, friction_per_fs=0.01,
                      thermostat_seed=derive_stream_seed(123, "thermostat"))
run_dir.mkdir(parents=True)
runner = EnergeticRunner(
    atoms, HarmonicSurrogate(k=1.0, r0=0.9, bias=1e-3), CountingReference(),
    Store(run_dir / "trajectory.db"), "run", run_dir=run_dir,
    event_log=EventLog(run_dir), checkpoint_interval_steps=4,
    force_budget=0.5, timestep_fs=0.5, time_cap_fs=100.0, transverse_cap=1.0,
    check_probability=0.5, check_seed=7, integrator_spec=spec)
runner.run(steps)
runner.close()
'''


class _CountingReference:
    """Parent-side twin of the child's engine: same law, same fingerprint,
    every real execution counted in a shared file, optional per-call drift
    so a silent recomputation is numerically distinguishable."""

    name = "counting-harmonic-reference"

    def __init__(self, count_file, *, drift_scale=0.0, armed=False):
        self.inner = HarmonicReference(k=K, r0=R0)
        self.count_file = Path(count_file)
        self.drift_scale = drift_scale
        self.armed = armed

    @property
    def fingerprint(self):
        return self.inner.fingerprint

    @property
    def capabilities(self):
        return self.inner.capabilities

    @property
    def n_calls(self):
        return int(self.count_file.read_text()) \
            if self.count_file.exists() else 0

    def compute(self, atoms):
        if self.armed:
            raise AssertionError("the recovery phase touched a live backend")
        n = self.n_calls + 1
        self.count_file.write_text(str(n))
        label = self.inner.compute(atoms)
        drift = self.drift_scale * n
        return EngineResult(label.energy + drift, label.forces + drift,
                            None, 0.0)


def _c_control(tmp_path, *, drift_scale=0.0):
    """Continuous 20-step control run, in-process, its own counter."""
    count_file = tmp_path / "control.calls"
    run_dir = tmp_path / "control"
    run_dir.mkdir(parents=True)
    atoms = _atoms()
    runner = EnergeticRunner(
        atoms, HarmonicSurrogate(k=K, r0=R0, bias=1e-3),
        _CountingReference(count_file, drift_scale=drift_scale),
        Store(run_dir / "trajectory.db"), "run", run_dir=run_dir,
        event_log=EventLog(run_dir), checkpoint_interval_steps=4,
        force_budget=0.5, timestep_fs=DT, time_cap_fs=100.0,
        transverse_cap=1.0, check_probability=0.5, check_seed=7,
        integrator_spec=_spec())
    runner.run(20)
    runner.close()
    return run_dir, int(count_file.read_text())


def _c_child(tmp_path, *, steps, kill_key=None, kill_after=False,
             drift_scale=0.0):
    import subprocess
    import sys
    import textwrap

    run_dir = tmp_path / "crashed"
    count_file = tmp_path / "crashed.calls"
    child = tmp_path / "child.py"
    tmp_path.mkdir(parents=True, exist_ok=True)
    child.write_text(textwrap.dedent(_C_CHILD))
    env = dict(os.environ, COUNT_FILE=str(count_file),
               DRIFT=str(drift_scale),
               PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    if kill_key is not None:
        env["KILL_KEY"] = kill_key
        env["KILL_AFTER"] = "1" if kill_after else "0"
    result = subprocess.run([sys.executable, str(child),
                             str(run_dir), str(steps)],
                            env=env, capture_output=True, text=True,
                            check=False)
    expected = 73 if kill_key is not None else 0
    assert result.returncode == expected, result.stderr[-500:]
    return run_dir, count_file


def _c_resume_and_compare(crash_dir, count_file, control_dir, *,
                          remaining, expected_extra_calls=0,
                          expected_extra_tasks=None,
                          missing_step_ids=()):
    """Resume with an armed sentinel (recovery must make zero live calls),
    then continue and compare the complete state against the control."""
    from pyraimd2.runtime.checkpoint import CheckpointManager
    from pyraimd2.workflows.export import export_run

    calls_before = int(count_file.read_text())
    sentinel = _CountingReference(count_file, armed=True)
    runner = EnergeticRunner.resume(
        crash_dir, HarmonicSurrogate(k=K, r0=R0, bias=1e-3), sentinel,
        event_log_force=True, checkpoint_interval_steps=4)
    # recovery (checkpoint + replay + boundary reconstruction) is over:
    # not one live reference execution happened, and no label was re-read
    # as a new physical execution
    assert sentinel.n_calls == calls_before
    sentinel.armed = False
    runner.run(remaining)
    runner.close()

    rows_a, rows_b = _rows(crash_dir), _rows(control_dir)
    assert len(rows_a) == len(rows_b) == 21
    for row_a, row_b in zip(rows_a, rows_b):
        np.testing.assert_array_equal(row_a.toatoms().positions,
                                      row_b.toatoms().positions)
        np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                      row_b.toatoms().get_momenta())
    boundaries_a = _complete_boundaries(crash_dir)
    boundaries_b = _complete_boundaries(control_dir)
    for (pos_a, mom_a), (pos_b, mom_b) in zip(boundaries_a, boundaries_b):
        np.testing.assert_array_equal(pos_a, pos_b)
        np.testing.assert_array_equal(mom_a, mom_b)

    # bath stream, check stream, decisions and anchor segments
    def key_events(run_dir):
        steps = {e["step_id"]: e for e in events(run_dir)
                 if e["type"] == "step_completed"}
        proposals = [(bool(e["accepted"]), e["reason"], e.get("check_draw"))
                     for e in events(run_dir)
                     if e["type"] == "evaluation_proposed"]
        return steps, proposals

    steps_a, proposals_a = key_events(crash_dir)
    steps_b, proposals_b = key_events(control_dir)
    assert set(steps_b) - set(steps_a) == set(missing_step_ids)
    for step_id, event in steps_a.items():
        control_event = steps_b[step_id]
        assert event["state_digest"] == control_event["state_digest"]
        assert event["boundary_digest"] == control_event["boundary_digest"]
        assert event["thermostat_rng"] == control_event["thermostat_rng"]
        assert event["segment_id"] == control_event["segment_id"]
    assert proposals_a == proposals_b
    anchors_a = {s: (r["evaluation_index"], r["correction_eV_A"])
                 for s, r in _anchors(crash_dir).items()}
    anchors_b = {s: (r["evaluation_index"], r["correction_eV_A"])
                 for s, r in _anchors(control_dir).items()}
    assert anchors_a.keys() == anchors_b.keys()
    for segment in anchors_a:
        assert anchors_a[segment][0] == anchors_b[segment][0]
        np.testing.assert_array_equal(np.array(anchors_a[segment][1]),
                                      np.array(anchors_b[segment][1]))

    # the cost ledger: identical tasks, plus exactly the expected retries
    # (a retried evaluation bills its inference and reference tasks again;
    # the counter file tracks reference executions only)
    if expected_extra_tasks is None:
        expected_extra_tasks = expected_extra_calls
    def ledger(run_dir):
        return [(e.get("operation"), e.get("purpose"), e.get("status"),
                 e.get("evaluation_id"))
                for e in events(run_dir) if e["type"] == "task"
                and e.get("operation") in ("reference", "inference")]

    ledger_a, ledger_b = ledger(crash_dir), ledger(control_dir)
    assert len(ledger_a) == len(ledger_b) + expected_extra_tasks
    control_calls = len([t for t in ledger_b if t[0] == "reference"])
    crashed_calls = int(count_file.read_text())
    assert crashed_calls == control_calls + expected_extra_calls

    # trajectory db, manual export, inspect and the final checkpoint all
    # name the same complete state.  Export frames carry the reconstructed
    # complete-step momenta (rows stay the mid-step record), so frames are
    # compared against the completed boundaries at the writer's text
    # precision, and the crashed run's export is bit-identical to the
    # control's (same writer path).
    from ase.io import read as ase_read

    report = export_run(crash_dir, force=True)
    frames = ase_read(report["output"], index=":")
    assert len(frames) == len(rows_a)
    control_report = export_run(control_dir, force=True)
    control_frames = ase_read(control_report["output"], index=":")
    for frame, control_frame, row, (pos, mom) in zip(
            frames, control_frames, rows_a, boundaries_a):
        np.testing.assert_array_equal(frame.positions, control_frame.positions)
        np.testing.assert_array_equal(frame.get_momenta(),
                                      control_frame.get_momenta())
        np.testing.assert_allclose(frame.positions, row.toatoms().positions,
                                   rtol=0, atol=1e-7)
        np.testing.assert_allclose(frame.positions, pos, rtol=0, atol=1e-7)
        np.testing.assert_allclose(frame.get_momenta(), mom,
                                   rtol=0, atol=1e-7)
    sources = {f.info.get("momenta_source") for f in frames}
    assert sources == {"initial_evaluation_record",
                       "complete_step_reconstructed"}
    from pyraimd2.runtime.inspect import inspect_run

    status = inspect_run(crash_dir)
    assert status["n_evaluations"] == 21
    checkpoint = CheckpointManager(crash_dir).read_latest_valid()
    final_positions, final_momenta = boundaries_a[-1]
    np.testing.assert_array_equal(checkpoint.arrays["positions"],
                                  final_positions)
    np.testing.assert_array_equal(checkpoint.arrays["momenta"],
                                  final_momenta)


def test_c_clean_split_20_equals_7_plus_13_in_a_new_process(tmp_path):
    control_dir, _ = _c_control(tmp_path / "ctl", drift_scale=1e-6)
    crash_dir, count_file = _c_child(tmp_path / "split", steps=7,
                                     drift_scale=1e-6)
    _c_resume_and_compare(crash_dir, count_file, control_dir, remaining=13)


def test_c_committed_label_uncommitted_step_recovers_without_redraw(tmp_path):
    # os._exit between the evaluation commit and the step commit: the
    # pending stochastic step's label is committed, the bath increments are
    # recorded, and the boundary completes without any redraw or recompute.
    # Resume heals the missing step record (the plain-driver window), so no
    # step id stays missing and the healed record's digests equal the
    # control's (checked inside the shared comparison).
    control_dir, _ = _c_control(tmp_path / "ctl", drift_scale=1e-6)
    crash_dir, count_file = _c_child(tmp_path / "w", steps=20,
                                     kill_key="step:run:6", drift_scale=1e-6)
    _c_resume_and_compare(crash_dir, count_file, control_dir, remaining=13)
    healed = [e for e in events(crash_dir) if e["type"] == "step_completed"
              and e["step_id"] == 6]
    control_step6 = next(e for e in events(control_dir)
                         if e["type"] == "step_completed" and e["step_id"] == 6)
    assert len(healed) == 1
    assert healed[0]["boundary_digest"] == control_step6["boundary_digest"]
    assert healed[0]["thermostat_rng"] == control_step6["thermostat_rng"]


def test_c_completed_step_before_next_checkpoint_recovers(tmp_path):
    # os._exit right after the step-6 commit: the next checkpoint (step 8)
    # never existed; resume replays evaluations 5-6 past checkpoint 4.
    control_dir, _ = _c_control(tmp_path / "ctl", drift_scale=1e-6)
    crash_dir, count_file = _c_child(tmp_path / "w", steps=20,
                                     kill_key="step:run:5", kill_after=True,
                                     drift_scale=1e-6)
    _c_resume_and_compare(crash_dir, count_file, control_dir, remaining=14)


def test_c_probe_interruption_reuses_verified_probes(tmp_path):
    # Crash inside the evaluation-1 calibration: probes (0,+h) and (0,-h)
    # are committed (event persisted); probe (1,+h) was computed but its
    # record never landed — a spent, uncommitted cost that stays on the
    # ledger.  Resume reuses the two verified probes, re-executes the two
    # uncommitted ones, and retries the evaluation's own uncommitted
    # anchor label: exactly two extra billed calls against the control.
    control_dir, _ = _c_control(tmp_path / "ctl")
    crash_dir, count_file = _c_child(tmp_path / "w", steps=20,
                                     kill_key="probe:run:1:0:1:1")
    _c_resume_and_compare(crash_dir, count_file, control_dir, remaining=20,
                          expected_extra_calls=2, expected_extra_tasks=3)
    reuse = [e for e in events(crash_dir) if e["type"] == "probe_completed"
             and int(e["evaluation_id"]) == 1]
    assert len(reuse) == 4
    probe_calls = [e for e in events(crash_dir)
                   if e["type"] == "task" and e.get("purpose") == "probe"
                   and e.get("operation") == "reference"]
    # 3 executed before the crash (one never committed) + 2 after resume;
    # the 2 reused probes execute nothing and bill nothing.
    assert len(probe_calls) == 5


# --- user interface: the released config path ---------------------------------


def test_template_harmonic_adaptive_nvt_runs_and_resumes(tmp_path):
    from pyraimd2.workflows import resume_workflow
    from pyraimd2.workflows.templates import write_template

    config_path = write_template("harmonic-adaptive-nvt", tmp_path / "demo")
    config = load_config(config_path)
    assert config.task.mode == "adaptive"
    assert config.dynamics.ensemble == "nvt"
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 20

    # the manifest and the RUN_START event record the same stream identity
    manifest = json.loads(
        (config.run.directory / "manifest.json").read_text())
    start = next(e for e in events(config.run.directory)
                 if e["type"] == "run_start")
    assert manifest["streams"] == {
        "scheme": "role-derive-v1",
        "velocity_seed": derive_stream_seed(7, "velocity"),
        "thermostat_seed": derive_stream_seed(123, "thermostat"),
        "check_seed": derive_stream_seed(19, "verification"),
    }
    assert manifest["streams"]["thermostat_seed"] == start["streams"][
        "thermostat_seed"]
    assert manifest["streams"]["check_seed"] == start["streams"]["check_seed"]
    assert manifest["dynamics"]["ensemble"] == "nvt"
    resolved = json.loads(
        (config.run.directory / "resolved_config.json").read_text())
    assert resolved["dynamics"]["integrator"] == "langevin"
    assert resolved["dynamics"]["friction_per_fs"] == 0.01

    # resume continues the same run to the same state as a continuous one
    resumed = resume_workflow(config.run.directory, 4, verbose=False,
                              handle_sigint=False)
    assert resumed.steps_completed == 24
    control_path = write_template("harmonic-adaptive-nvt",
                                  tmp_path / "control")
    control = load_config(control_path)
    control_text = (tmp_path / "control" / "run.toml").read_text()
    (tmp_path / "control" / "run.toml").write_text(
        control_text.replace("steps = 20", "steps = 24"))
    control = load_config(tmp_path / "control" / "run.toml")
    run_workflow(control, verbose=False, handle_sigint=False)
    rows_a = _rows(config.run.directory, "harmonic-adaptive-nvt-demo")
    rows_b = _rows(control.run.directory, "harmonic-adaptive-nvt-demo")
    assert len(rows_a) == len(rows_b) == 25
    for row_a, row_b in zip(rows_a, rows_b):
        np.testing.assert_array_equal(row_a.toatoms().positions,
                                      row_b.toatoms().positions)
        np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                      row_b.toatoms().get_momenta())
