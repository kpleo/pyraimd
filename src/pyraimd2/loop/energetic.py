"""Energetic force-error gating for fixed-cell, unconstrained ASE dynamics.

The decision concerns the *current discrete force evaluation*. Two-scale
reference probes supply empirical response coefficients, not a certificate
for the next velocity-Verlet interval. A frozen base potential and its
constant anchor correction define each segment. Reference checks measure
the accepted corrected force without replacing that force retrospectively.

Coordinates must be unwrapped and atom order, cell and masses must remain
fixed. EnergeticRunner maintains the physical evaluation clock, including
steps with unchanged positions. EnergeticCalculator can also be used directly:
each new uncached configuration then advances its clock by ``timestep_fs``.
Neither interface supports restoring an energetic run from disk yet.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

from pyraimd2.energetics import (
    DegenerateResponseError,
    DirectionalResponse,
    IndependentCheckBound,
    estimate_responses,
    residual_work,
)
from pyraimd2.engines.base import Engine, EngineError, EngineResult
from pyraimd2.store.store import Store
from pyraimd2.surrogate.base import Surrogate, SurrogatePrediction
from pyraimd2.switch.base import LabelObservation

Direction = Callable[[Atoms], np.ndarray]
LabelCallback = Callable[[LabelObservation], bool | None]


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _same_state(first: Atoms, second: Atoms, *, momenta: bool = False) -> bool:
    fields = ["numbers", "positions", "cell", "pbc"]
    equal = all(np.array_equal(getattr(first, key), getattr(second, key)) for key in fields)
    equal = equal and np.array_equal(first.get_masses(), second.get_masses())
    equal = equal and np.array_equal(first.get_initial_charges(), second.get_initial_charges())
    equal = equal and np.array_equal(first.get_initial_magnetic_moments(),
                                   second.get_initial_magnetic_moments())
    return equal and (not momenta or np.array_equal(first.get_momenta(), second.get_momenta()))


def _label_arrays(value: object, n_atoms: int) -> tuple[float, np.ndarray, np.ndarray | None]:
    energy = float(value.energy)
    forces = np.array(value.forces, dtype=float, copy=True)
    if not math.isfinite(energy) or forces.shape != (n_atoms, 3) or not np.isfinite(forces).all():
        raise ValueError("energy and (N, 3) forces must be finite")
    stress = None if value.stress is None else np.array(value.stress, dtype=float, copy=True)
    if stress is not None and (stress.shape != (6,) or not np.isfinite(stress).all()):
        raise ValueError("stress must be None or a finite (6,) array")
    return energy, forces, stress


@dataclass
class _Anchor:
    segment: int
    index: int
    positions: np.ndarray
    prediction: SurrogatePrediction
    label: EngineResult
    correction: np.ndarray
    responses: tuple[DirectionalResponse, ...]
    open_prefix: list[bool]
    calibration: dict


@dataclass
class _Pending:
    """A frozen decision survives a failed reference call without resampling."""

    atoms: Atoms
    index: int
    prediction: SurrogatePrediction
    anchor: _Anchor | None
    accepted: bool
    reason: str
    forecasts: list[dict]
    selected_direction: int | None
    open_prefix: list[bool]
    energy: float | None
    forces: np.ndarray | None
    checked: bool
    draw: float | None
    calls_before: dict
    label: EngineResult | None = None
    new_anchor: _Anchor | None = None
    calibration_done: bool = False
    probe_records: list[dict] = field(default_factory=list)


@dataclass
class _CalibrationOrigin:
    atoms: Atoms
    index: int
    label: EngineResult


@dataclass(frozen=True)
class EnergeticRunSummary:
    """Segment costs. Reference calls include anchors, probes and checks."""

    n_steps: int
    n_evaluations: int
    n_accepted: int
    n_reference: int
    n_anchor: int
    n_probe: int
    n_checks: int
    n_violations: int
    n_calibrations: int
    accepted_fraction: float
    verification: dict | None
    wall_time_s: float


class EnergeticCalculator(Calculator):
    """Predict, gate, drive, independently check, and log each evaluation.

    ``force_budget`` and ``numerical_floor`` are in eV/angstrom;
    ``probe_steps`` are two increasing positive distances in angstrom;
    ``time_cap_fs`` bounds elapsed time since calibration. ``direction``
    returns an (N, 3) or (D, N, 3) array; each nonzero direction is normalized
    in the full 3N-dimensional Euclidean norm. The default uses velocity.
    A zero velocity uses a reference force without directional probes.

    ``check_probability=0`` explicitly disables independent verification.
    Otherwise a dedicated RNG draws only after the decision and corrected
    force have been frozen. Every accepted evaluation enters the denominator.

    ``on_label`` runs after the force is frozen and the row is stored, never
    between calibration probes. It receives an uncorrected base prediction.
    On a reference route the updated model is calibrated before the next
    decision, reusing the valid reference origin label. On a checked accepted
    route, return exactly ``False`` to declare that the model was not changed;
    every other return similarly replaces the old anchor with a calibration
    of the updated model at the checked origin. A detected violation always
    forces the next evaluation to the reference route.
    Only this callback may update the model while a calculation is running.
    A callback failure stops the run; in-process retry of a stored callback
    failure and checkpoint restart are deliberately unsupported.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        surrogate: Surrogate,
        engine: Engine,
        store: Store,
        run_id: str,
        *,
        force_budget: float,
        timestep_fs: float = 0.5,
        probe_steps: tuple[float, float] = (0.02, 0.04),
        numerical_floor: float = 0.0,
        time_cap_fs: float = 1.0,
        transverse_cap: float = 0.1,
        check_probability: float = 0.05,
        check_seed: int = 0,
        failure_probability: float = 0.05,
        tilt: float = math.log(2.0),
        direction: Direction | None = None,
        on_label: LabelCallback | None = None,
    ) -> None:
        super().__init__()
        self.force_budget = _positive(force_budget, "force_budget")
        self.timestep_fs = _positive(timestep_fs, "timestep_fs")
        self.numerical_floor = _positive(numerical_floor, "numerical_floor", zero=True)
        self.time_cap_fs = _positive(time_cap_fs, "time_cap_fs")
        self.transverse_cap = _positive(transverse_cap, "transverse_cap", zero=True)
        if self.transverse_cap > 1:
            raise ValueError("transverse_cap must be in [0, 1]")
        self.probe_steps = np.asarray(probe_steps, dtype=float)
        if (self.probe_steps.shape != (2,) or not np.isfinite(self.probe_steps).all()
                or not 0 < self.probe_steps[0] < self.probe_steps[1]):
            raise ValueError("probe_steps must contain two increasing positive distances")
        if not math.isfinite(check_probability) or not 0 <= check_probability <= 1:
            raise ValueError("check_probability must be in [0, 1]")
        if not isinstance(check_seed, (int, np.integer)) or isinstance(check_seed, bool):
            raise ValueError("check_seed must be an integer")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        # Reject accidental append/restart rather than silently losing the
        # reference anchor, RNG history or the independent bound's counts.
        if next(store._db.select(run_id=run_id), None) is not None:
            raise ValueError("run_id already exists; energetic restart is not implemented")
        self.surrogate, self.engine, self.store, self.run_id = surrogate, engine, store, run_id
        self.direction, self.on_label = direction, on_label
        self.check_probability, self.check_seed = float(check_probability), int(check_seed)
        self._rng = np.random.default_rng(check_seed)
        self.verification = (IndependentCheckBound(check_probability, failure_probability, tilt)
                             if check_probability > 0 else None)
        self.step = -1
        self.n_evaluations = self.n_accepted = self.n_violations = self.n_calibrations = 0
        self.reference_calls = {"anchor": 0, "probe": 0, "check": 0}
        self._anchor: _Anchor | None = None
        self._segment = 0
        self._identity: Atoms | None = None
        self._pending: _Pending | None = None
        self._scheduled_index: int | None = None
        self._expected_positions: np.ndarray | None = None
        self._callback_failed = False
        self._next_reason = "initial_reference"
        self._deferred_origin: _CalibrationOrigin | None = None
        self._deferred_record: dict | None = None
        self._evaluation_calls_before: dict | None = None

    @property
    def n_reference(self) -> int:
        """Actual successful reference calls, including off-trajectory probes."""
        return sum(self.reference_calls.values())

    def _validate_atoms(self, atoms: Atoms) -> None:
        if not len(atoms) or atoms.constraints:
            raise ValueError("energetic dynamics requires nonempty, unconstrained Atoms")
        for array in (atoms.positions, atoms.cell.array, atoms.get_momenta(), atoms.get_masses(),
                      atoms.get_initial_charges(), atoms.get_initial_magnetic_moments()):
            if not np.isfinite(array).all():
                raise ValueError("atomic state must be finite")
        if np.any(atoms.get_masses() <= 0):
            raise ValueError("atomic masses must be positive")
        if self._identity is None:
            self._identity = atoms.copy()
        else:
            identity = self._identity.copy()
            identity.positions = atoms.positions.copy()
            if not _same_state(identity, atoms):
                raise ValueError("atom identity/order, mass, charge, PBC and fixed cell must not change")
        if self._expected_positions is not None and not np.allclose(
            atoms.positions, self._expected_positions, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("positions do not match the scheduled unwrapped Verlet step")

    def _predict(self, atoms: Atoms) -> SurrogatePrediction:
        work = atoms.copy()
        prediction = self.surrogate.predict(work)
        if not _same_state(atoms, work):
            raise ValueError("surrogate.predict mutated its atomic input")
        energy, forces, stress = _label_arrays(prediction, len(atoms))
        uncertainty = np.array(prediction.uncertainty, dtype=float, copy=True)
        if (uncertainty.shape != (len(atoms),) or np.isinf(uncertainty).any()
                or np.any(uncertainty < 0)):
            raise ValueError("uncertainty must be a nonnegative (N,) array or NaN")
        return SurrogatePrediction(energy, forces, stress, uncertainty)

    def _reference(self, atoms: Atoms, purpose: str) -> EngineResult:
        work = atoms.copy()
        try:
            result = self.engine.compute(work)
            if not _same_state(atoms, work):
                raise ValueError("engine.compute mutated its atomic input")
            energy, forces, stress = _label_arrays(result, len(atoms))
            wall = _positive(result.wall_time_s, "reference wall_time_s", zero=True)
        except EngineError:
            raise
        except Exception as error:
            raise EngineError(f"invalid {purpose} reference evaluation: {error}") from error
        self.reference_calls[purpose] += 1
        return EngineResult(energy, forces, stress, wall)

    def _directions(self, atoms: Atoms) -> np.ndarray | None:
        raw = atoms.get_velocities() if self.direction is None else self.direction(atoms.copy())
        if raw is None:
            return None
        directions = np.array(raw, dtype=float, copy=True)
        if directions.shape == (len(atoms), 3):
            directions = directions[None, ...]
        if (directions.ndim != 3 or directions.shape[1:] != (len(atoms), 3)
                or not len(directions) or not np.isfinite(directions).all()):
            raise ValueError("direction must return finite (N, 3) or (D, N, 3) coordinates")
        lengths = np.linalg.norm(directions.reshape(len(directions), -1), axis=1)
        if np.all(lengths == 0):
            return None
        if np.any(lengths == 0):
            raise ValueError("supplied directions cannot mix zero and nonzero vectors")
        return directions / lengths[:, None, None]

    def _calibrate(self, pending: _Pending) -> _Anchor | None:
        directions = self._directions(pending.atoms)
        if directions is None:
            return None
        correction = pending.label.forces - pending.prediction.forces
        shape = (len(directions), 2, len(pending.atoms), 3)
        plus_d, minus_d, plus_r, minus_r = [np.empty(shape) for _ in range(4)]
        records = []
        for d, direction in enumerate(directions):
            for h_index, h in enumerate(self.probe_steps):
                for sign, displacements, residuals in ((1, plus_d, plus_r), (-1, minus_d, minus_r)):
                    probe = pending.atoms.copy()
                    probe.positions += sign * h * direction
                    prediction = self._predict(probe)
                    label = self._reference(probe, "probe")
                    displacement = probe.positions - pending.atoms.positions
                    displacements[d, h_index] = displacement
                    residuals[d, h_index] = prediction.forces + correction - label.forces
                    records.append({"direction": d, "step_A": float(h), "sign": sign,
                                    "displacement_A": displacement.tolist(),
                                    "base_energy_eV": prediction.energy,
                                    "reference_energy_eV": label.energy,
                                    "base_forces_eV_A": prediction.forces.tolist(),
                                    "reference_forces_eV_A": label.forces.tolist()})
        pending.probe_records = records
        try:
            responses = estimate_responses(directions, self.probe_steps, plus_d, minus_d, plus_r, minus_r)
        except DegenerateResponseError:
            # C_rw is undefined, so keep using reference forces. A zero
            # finite-probe derivative does not establish a global bound.
            return None
        calibration = {"probe_steps_A": self.probe_steps.tolist(), "probes": records,
                       "responses": [response.as_dict() for response in responses],
                       "force_call_count": 1 + 4 * len(directions),
                       "type": "two_scale_empirical_reference_probes"}
        self._segment += 1
        self.n_calibrations += 1
        return _Anchor(self._segment, pending.index, pending.atoms.positions.copy(),
                       pending.prediction, pending.label, correction, responses,
                       [True] * len(responses), calibration)

    @staticmethod
    def _anchor_record(anchor: _Anchor | None) -> dict | None:
        if anchor is None:
            return None
        return {"segment_id": anchor.segment, "evaluation_index": anchor.index,
                "positions_A": anchor.positions.tolist(),
                "base_energy_eV": anchor.prediction.energy,
                "reference_energy_eV": anchor.label.energy,
                "correction_eV_A": anchor.correction.tolist(),
                "calibration": anchor.calibration}

    def _prepare_updated_model(self) -> None:
        """Calibrate only after a stored label's update, before a new proposal."""
        origin = self._deferred_origin
        if origin is None:
            return
        prediction = self._predict(origin.atoms)
        pending = _Pending(origin.atoms, origin.index, prediction, None, False,
                           "model_update_calibration", [], None, [], None, None,
                           False, None, self.reference_calls.copy(), label=origin.label)
        anchor = self._calibrate(pending)
        self._anchor = anchor
        self._deferred_record = {
            "origin_evaluation_index": origin.index,
            "reference_origin_label_reused": True,
            "anchor": self._anchor_record(anchor),
            "unusable_probe_records": pending.probe_records if anchor is None else [],
        }
        self._deferred_origin = None
        self._next_reason = "direction_unavailable_reference" if anchor is None else "reference_required"

    def _freeze(self, atoms: Atoms, index: int) -> _Pending:
        prediction = self._predict(atoms)
        anchor = self._anchor
        forecasts, open_prefix = [], []
        selected = None
        reason = self._next_reason
        if anchor is not None:
            displacement = atoms.positions - anchor.positions
            elapsed = (index - anchor.index) * self.timestep_fs
            for is_open, response in zip(anchor.open_prefix, anchor.responses, strict=True):
                forecast = response.forecast(displacement, elapsed, self.force_budget,
                                             self.numerical_floor, self.time_cap_fs,
                                             self.transverse_cap)
                open_prefix.append(bool(is_open and forecast.admitted))
                forecasts.append({"linear_error_eV_A": float(forecast.linear_error),
                                  "envelope_eV_A": float(forecast.envelope),
                                  "predicted_work_eV": float(forecast.predicted_work),
                                  "transverse_fraction": float(forecast.transverse_fraction),
                                  "in_domain": bool(forecast.in_domain),
                                  "admitted": bool(forecast.admitted),
                                  "prefix_open": open_prefix[-1]})
            candidates = [j for j, is_open in enumerate(open_prefix) if is_open]
            if candidates:
                selected = min(candidates, key=lambda j: forecasts[j]["envelope_eV_A"])
                reason = "forecast_accepted"
            else:
                reason = "forecast_budget_domain_or_prefix_failure"
        accepted = selected is not None
        # Force and energy are frozen BEFORE the Bernoulli draw and reference.
        forces = prediction.forces + anchor.correction if accepted else None
        energy = (prediction.energy - float(np.sum(anchor.correction *
                  (atoms.positions - anchor.positions))) + anchor.label.energy -
                  anchor.prediction.energy) if accepted else None
        draw = float(self._rng.random()) if accepted and self.check_probability > 0 else None
        checked = draw is not None and draw < self.check_probability
        return _Pending(atoms.copy(), index, prediction, anchor, accepted, reason,
                        forecasts, selected, open_prefix, energy, forces, checked, draw,
                        self._evaluation_calls_before.copy())

    def _observed(self, pending: _Pending) -> dict | None:
        anchor, label = pending.anchor, pending.label
        if anchor is None or label is None:
            return None
        residual = pending.prediction.forces + anchor.correction - label.forces
        error = float(np.linalg.norm(residual, axis=1).max())
        work = residual_work(anchor.positions, pending.atoms.positions,
                             anchor.prediction.energy, pending.prediction.energy,
                             anchor.label.energy, label.energy, anchor.correction)
        return {"segment_id": anchor.segment, "residual_eV_A": residual.tolist(),
                "max_force_error_eV_A": error, "endpoint_work_eV": float(work),
                "force_budget_exceeded": error > self.force_budget,
                "observed_coefficient_A2_eV": 2 * work / error**2 if error > 0 else None}

    def _finish(self, pending: _Pending) -> None:
        if (not pending.accepted or pending.checked) and pending.label is None:
            pending.label = self._reference(pending.atoms, "check" if pending.checked else "anchor")
        observed = self._observed(pending)
        violation = bool(observed["force_budget_exceeded"]) if pending.checked else None
        if not pending.accepted:
            pending.energy, pending.forces = pending.label.energy, pending.label.forces.copy()
            if not pending.calibration_done and self.on_label is None:
                pending.new_anchor = self._calibrate(pending)
                pending.calibration_done = True
        bound = copy.deepcopy(self.verification)
        if bound is not None:
            bound.update(pending.accepted, pending.checked, violation)
        anchor = pending.anchor
        new_anchor = pending.new_anchor
        drive = SurrogatePrediction(pending.energy, pending.forces.copy(), None,
                                    pending.prediction.uncertainty.copy())
        metadata = {
            "method": "energetic_force_error", "evaluation_index": pending.index,
            "time_fs": pending.index * self.timestep_fs, "timestep_fs": self.timestep_fs,
            "gate_scope": "discrete_force_evaluation", "coordinates": "unwrapped",
            "force_budget_eV_A": self.force_budget, "numerical_floor_eV_A": self.numerical_floor,
            "time_cap_fs": self.time_cap_fs, "transverse_cap": self.transverse_cap,
            "segment_id": None if anchor is None else anchor.segment,
            "selected_direction": pending.selected_direction, "forecasts": pending.forecasts,
            "accepted": pending.accepted, "checked": pending.checked, "violation": violation,
            "check_probability": self.check_probability, "check_seed": self.check_seed,
            "check_draw": pending.draw, "observed": observed,
            "verification": None if bound is None else bound.as_dict(),
            "reference_calls_total": self.reference_calls.copy(),
            "reference_calls_this_evaluation": {key: count - pending.calls_before[key]
                for key, count in self.reference_calls.items()},
            "callback_scheduled": pending.label is not None and self.on_label is not None,
            "new_anchor": self._anchor_record(new_anchor),
            "calibration_after_previous_label": self._deferred_record,
            "calibration_deferred_until_after_callback": not pending.accepted and self.on_label is not None,
            "unusable_probe_records": pending.probe_records if new_anchor is None else [],
        }
        self.store.append(self.run_id, pending.index - 1, pending.atoms,
                          "ml" if pending.accepted else "dft",
                          surrogate=pending.prediction, engine=pending.label,
                          reason=pending.reason, metadata=metadata, driving=drive)
        # Commit only after a valid label/calibration and a successful append.
        self.verification = bound
        self.n_evaluations += 1
        self.n_accepted += int(pending.accepted)
        self.n_violations += int(violation is True)
        self.step = pending.index
        if pending.accepted:
            anchor.open_prefix = pending.open_prefix
            self._anchor = None if violation else anchor
            self._next_reason = "previous_independent_check_violation" if violation else "reference_required"
        else:
            self._anchor = new_anchor
            self._next_reason = "direction_unavailable_reference" if new_anchor is None else "reference_required"
        self.results = {"energy": drive.energy, "forces": drive.forces.copy()}
        self._pending = None
        self._deferred_record = None
        self._evaluation_calls_before = None
        if pending.label is not None and self.on_label is not None:
            candidate = self._anchor
            self._anchor = None
            self._next_reason = "model_update_requires_recalibration"
            try:
                changed = self.on_label(LabelObservation(pending.index - 1, pending.atoms.copy(),
                                                        copy.deepcopy(pending.prediction),
                                                        copy.deepcopy(pending.label)))
            except Exception:
                self.results = {}
                self._callback_failed = True
                raise
            if pending.accepted and changed is False:
                self._anchor = candidate
                if violation:
                    self._next_reason = "previous_independent_check_violation"
            elif not violation:
                self._deferred_origin = _CalibrationOrigin(pending.atoms.copy(), pending.index,
                                                          copy.deepcopy(pending.label))
            if violation:
                self._next_reason = "previous_independent_check_violation"

    def calculate(self, atoms: Atoms | None = None,
                  properties: tuple[str, ...] = ("energy", "forces"),
                  system_changes: list[str] = all_changes) -> None:
        if self._callback_failed:
            raise RuntimeError("a stored label callback failed; start a new energetic run")
        super().calculate(atoms, properties, system_changes)
        # ASE may still have the preceding geometry's successful values.
        self.results = {}
        self._validate_atoms(self.atoms)
        index = self.n_evaluations if self._scheduled_index is None else self._scheduled_index
        if index != self.n_evaluations:
            raise ValueError("evaluation clock is not consecutive; do not modify a stored MD state")
        if self._pending is None:
            if self._evaluation_calls_before is None:
                self._evaluation_calls_before = self.reference_calls.copy()
            self._prepare_updated_model()
            self._pending = self._freeze(self.atoms, index)
        elif self._pending.index != index or not _same_state(self._pending.atoms, self.atoms, momenta=True):
            raise ValueError("retry requires the same pending geometry, momenta and evaluation time")
        try:
            self._finish(self._pending)
        except Exception:
            self.results = {}
            raise

    def _schedule(self, index: int, positions: np.ndarray | None = None) -> None:
        if index != self.n_evaluations:
            raise ValueError("the MD clock must advance by one force evaluation")
        self._scheduled_index = index
        self._expected_positions = None if positions is None else positions.copy()
        self.results = {}


class _EnergeticVerlet(VelocityVerlet):
    """Advance the evaluation clock even at an unchanged configuration."""

    def step(self, forces=None):
        atoms = self.atoms
        if forces is None:
            forces = atoms.get_forces(md=True)
        next_positions = atoms.positions + self.dt * (
            atoms.get_momenta() + 0.5 * self.dt * forces
        ) / atoms.get_masses()[:, None]
        atoms.calc._schedule(self.nsteps + 1, next_positions)
        return super().step(forces)


class EnergeticRunner:
    """Fixed-cell NVE with energetic force-error prediction and reference checks.

    Existing momenta are preserved. If missing, a seeded thermal distribution
    at ``temperature_K`` initializes them once. Force-call costs count every
    successful anchor/probe/check separately. Calling ``run`` again continues
    this live instance; loading an old Store as a restart is not implemented.
    A failed MD step stops this runner because Verlet may have advanced to
    half-step momenta; no silent integrator retry is attempted.
    """

    def __init__(
        self, atoms: Atoms, surrogate: Surrogate, engine: Engine, store: Store, run_id: str,
        *, force_budget: float, timestep_fs: float = 0.5,
        probe_steps: tuple[float, float] = (0.02, 0.04), numerical_floor: float = 0.0,
        time_cap_fs: float = 1.0, transverse_cap: float = 0.1,
        check_probability: float = 0.05, check_seed: int = 0,
        failure_probability: float = 0.05, tilt: float = math.log(2.0),
        direction: Direction | None = None, on_label: LabelCallback | None = None,
        temperature_K: float = 300.0, velocity_seed: int = 0,
    ) -> None:
        temperature_K = _positive(temperature_K, "temperature_K", zero=True)
        self.calc = EnergeticCalculator(
            surrogate, engine, store, run_id, force_budget=force_budget, timestep_fs=timestep_fs,
            probe_steps=probe_steps, numerical_floor=numerical_floor, time_cap_fs=time_cap_fs,
            transverse_cap=transverse_cap, check_probability=check_probability,
            check_seed=check_seed, failure_probability=failure_probability, tilt=tilt,
            direction=direction, on_label=on_label,
        )
        self.calc._validate_atoms(atoms)
        if "momenta" not in atoms.arrays:
            thermalize_momenta(atoms, temperature_K, rng=np.random.default_rng(velocity_seed))
        self.atoms = atoms
        self.timestep_fs = self.calc.timestep_fs
        atoms.calc = self.calc
        self.calc._schedule(0)
        self.dyn = _EnergeticVerlet(atoms, self.timestep_fs * units.fs)
        self._failed = False

    def run(self, n_steps: int) -> EnergeticRunSummary:
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer")
        if self._failed:
            raise RuntimeError("an energetic MD step failed; start a new run from a deliberate state")
        before = (self.calc.n_evaluations, self.calc.n_accepted, self.calc.n_violations,
                  self.calc.n_calibrations, self.calc.reference_calls.copy())
        start = time.perf_counter()
        try:
            self.dyn.run(int(n_steps))
        except Exception:
            self._failed = True
            raise
        n_evaluations = self.calc.n_evaluations - before[0]
        n_accepted = self.calc.n_accepted - before[1]
        calls = {key: count - before[4][key] for key, count in self.calc.reference_calls.items()}
        return EnergeticRunSummary(
            int(n_steps), n_evaluations, n_accepted, sum(calls.values()), calls["anchor"],
            calls["probe"], calls["check"], self.calc.n_violations - before[2],
            self.calc.n_calibrations - before[3],
            n_accepted / n_evaluations if n_evaluations else 0.0,
            None if self.calc.verification is None else self.calc.verification.as_dict(),
            time.perf_counter() - start,
        )

    @classmethod
    def resume(cls, *args, **kwargs):
        raise NotImplementedError("energetic restart is not implemented; legacy Runner.resume is unchanged")
