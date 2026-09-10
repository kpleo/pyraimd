"""Energetic force-error gating for fixed-cell, unconstrained ASE dynamics.

The decision concerns the *current discrete force evaluation*. Two-scale
reference probes supply empirical response coefficients, not a certificate
for the next velocity-Verlet interval. A frozen base potential and its
constant anchor correction define each segment. Reference checks measure
the accepted corrected force without replacing that force retrospectively.

Coordinates must be unwrapped and atom order, cell and masses must remain
fixed. EnergeticRunner maintains the physical evaluation clock, including
steps with unchanged positions, and hands each evaluation an explicit
physical time. EnergeticCalculator can also be used directly: each new
uncached configuration then advances its clock by ``timestep_fs`` — a
fixed-timestep compatibility assumption, not a general time source.
Every committed evaluation carries an explicit EvaluationContext (run_id,
step_id, evaluation_id, phase, physical_time_fs, model_id) recorded in the
stored metadata; probes share their parent evaluation's identity and never
advance physical time. Model updates announced through ``on_label`` advance
a decision-layer model generation that keys pending proposals, anchors and
cached results.

Resumable runs (WP03) write complete-step checkpoints (positions, full-step
momenta, real time, committed evaluation id, reusable driving force, anchor,
check RNG bit state, model identity and updater state) through
``run_dir``/``checkpoint_interval_steps``, and ``EnergeticRunner.resume``
restores the last valid checkpoint in a new process and replays the events
after its cursor — reusing frozen proposals, check draws and driving forces
without re-sampling, re-training or re-consuming. Plain ``on_label``
callbacks without a state export are not resumable; unsupported combinations
are refused, never silently degraded.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

from pyraimd2 import __version__
from pyraimd2.energetics import (
    DegenerateResponseError,
    DirectionalResponse,
    IndependentCheckBound,
    estimate_responses,
    residual_work,
)
from pyraimd2.engines.base import (
    EnergyKind,
    Engine,
    EngineError,
    EngineResult,
    engine_capabilities,
)
from pyraimd2.loop.constraints import (
    FORCE_METRICS,
    FixAtomsProjection,
    validate_constraints,
)
from pyraimd2.loop.integrators import (
    DIGEST_FORMAT,
    STREAM_SCHEME,
    CommittedStepState,
    IntegratorSpec,
    complete_langevin_momenta,
    derive_stream_seed,
    state_digest,
)
from pyraimd2.runtime import (
    EvaluationContext,
    EvaluationPhase,
    fingerprint_of,
    model_id_for,
)
from pyraimd2.runtime.checkpoint import (
    CheckpointManager,
    ResumeError,
    rng_state_to_json,
)
from pyraimd2.runtime.events import (
    ATTEMPT_LEDGER_PHYSICAL_V1,
    EVALUATION_COMMITTED,
    EVALUATION_PROPOSED,
    EVENT_SCHEMA_VERSION,
    LABEL_CONSUMED,
    MODEL_UPDATE,
    PROBE_COMPLETED,
    RESUMED,
    RUN_END,
    RUN_START,
    RUN_SUMMARY,
    STEP_COMPLETED,
    TASK,
    UPDATE_REJECTED,
    EventLog,
    EventLogError,
    physical_attempt,
)
from pyraimd2.runtime.labels import LabelCache, atoms_input_hash
from pyraimd2.runtime.models import (
    MODEL_ARTIFACT_FORMAT_VERSION,
    ModelRegistry,
    array_placeholder,
    artifact_digest,
    content_array_sink,
    content_array_source,
    dict_array_source,
    dump_state_arrays,
    load_state_arrays,
    resolve_artifact_state,
)
from pyraimd2.runtime.updater import StatefulUpdater
from pyraimd2.store.store import STORE_SCHEMA_VERSION, Store
from pyraimd2.surrogate.base import (
    Surrogate,
    SurrogatePrediction,
    assert_compatible_energy_contract,
    surrogate_capabilities,
)
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
    model_generation: int = 0


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
    context: EvaluationContext | None = None
    model_generation: int = 0
    label_id: str | None = None
    label_task_id: str | None = None
    label_attempt: int = 0
    # NVT: the bath stream state before this step's draws (rng_before) —
    # the realized increments stay with the dynamics and are committed
    # verbatim; a rebuilt pending re-derives them from rng_before through
    # ASE itself, never through a fresh draw (M3A-3).
    bath_step: dict | None = None
    # (segment, n_calibrations) frozen by a recalibration that ran before an
    # uncommitted proposal; applied when the rebuilt evaluation commits (C4).
    restored_counters: tuple[int, int] | None = None


@dataclass
class _CalibrationOrigin:
    atoms: Atoms
    index: int
    label: EngineResult
    label_id: str | None = None


def _response_from_dict(record: dict) -> DirectionalResponse:
    """Rebuild a response from its stored snapshot (derived fields recompute
    identically from the same inputs)."""
    return DirectionalResponse(
        np.array(record["direction"], dtype=float),
        np.array(record["response"], dtype=float),
        float(record["eta"]),
        float(record["transverse_coefficient"]),
        float(record["remainder_coefficient"]),
    )


def _anchor_from_record(record: dict, model_generation: int | None = None) -> _Anchor:
    """Rebuild an anchor from its stored record (store metadata, checkpoint
    state or proposal events all share ``_anchor_record``'s shape)."""
    calibration = record["calibration"]
    responses = tuple(_response_from_dict(r) for r in calibration["responses"])
    n_atoms = len(record["positions_A"])
    prediction = SurrogatePrediction(
        float(record["base_energy_eV"]),
        np.array(record["base_forces_eV_A"], dtype=float), None,
        np.full(n_atoms, np.nan))
    label = EngineResult(
        float(record["reference_energy_eV"]),
        np.array(record["reference_forces_eV_A"], dtype=float), None, 0.0)
    open_prefix = record.get("open_prefix")
    generation = record.get("model_generation", model_generation)
    return _Anchor(
        int(record["segment_id"]), int(record["evaluation_index"]),
        np.array(record["positions_A"], dtype=float), prediction, label,
        np.array(record["correction_eV_A"], dtype=float), responses,
        [bool(v) for v in open_prefix] if open_prefix is not None
        else [True] * len(responses),
        calibration, model_generation=0 if generation is None else int(generation))


def _id_suffix(identity: str | None) -> int:
    """Numeric suffix of a ``...-N`` run identifier (0 when absent/invalid)."""
    if identity is None:
        return 0
    try:
        return int(str(identity).rsplit("-", 1)[1])
    except (ValueError, IndexError):
        return 0


def _is_stateful(callback: object) -> bool:
    """A resumable updater exports and restores its continuation state."""
    return callable(callback) and hasattr(callback, "state_dict") \
        and hasattr(callback, "load_state_dict")


class StopRequested(RuntimeError):
    """Raised inside the integrator loop to stop at the last complete step."""


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

    Every committed evaluation carries an :class:`EvaluationContext` recorded
    in the stored metadata; physical time comes from the integrator's
    schedule, while ``evaluation_id`` counts logical force evaluations.
    Any ``on_label`` return except exactly ``False`` declares a model change
    and advances the decision-layer model generation: cached results for the
    same geometry, pending proposals and anchors from the old generation are
    never reused for a new evaluation (WP01 decision-cache rule; the numeric
    geometry-keyed label cache is WP02 and deliberately separate).

    Direct use as a plain ASE Calculator is a compatibility layer with a
    fixed-timestep assumption: each new uncached configuration advances the
    physical clock by exactly ``timestep_fs``. Supported workflows drive the
    calculator through an integrator that schedules explicit physical times.

    With ``event_log`` (a :class:`pyraimd2.runtime.events.EventLog`), every
    physical execution (reference anchor/refusal/probe/verification,
    inference, training, I/O) is recorded as a cost-ledger task event with
    durable task/label IDs, and proposals/commits/model updates become
    idempotent logical events. With ``label_cache`` (default on, §5.4), a
    verification check may reuse an exactly matching cached reference label
    when the engine declares a fingerprint — the check counts normally with
    zero new physical executions; unidentifiable backends get no cache.
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
        event_log: EventLog | None = None,
        label_cache: bool = True,
        force_metric: str = "active_dofs_max_atom",
        integrator_spec: IntegratorSpec | dict | None = None,
        velocity_seed: int | None = None,
        _resume_state: dict | None = None,
    ) -> None:
        super().__init__()
        if force_metric not in FORCE_METRICS:
            raise ValueError(f"force_metric must be one of {FORCE_METRICS}, "
                             f"got {force_metric!r}")
        self.force_metric = force_metric
        self._projection: FixAtomsProjection | None = None
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
        # A resume restores its state over the existing run deliberately.
        if _resume_state is None and next(store._db.select(run_id=run_id), None) is not None:
            raise ValueError("run_id already exists; direct restart is rejected, "
                             "use EnergeticRunner.resume")
        self.surrogate, self.engine, self.store, self.run_id = surrogate, engine, store, run_id
        # Contract preflight: a declared unit/energy-convention mismatch fails
        # here, before the first (expensive) SCF or inference call. Undeclared
        # capabilities read as unknown and cannot prove a mismatch.
        assert_compatible_energy_contract(
            engine_capabilities(engine), surrogate_capabilities(surrogate)
        )
        self._engine_fingerprint = fingerprint_of(engine)
        self.direction, self.on_label = direction, on_label
        if integrator_spec is None:
            integrator_spec = IntegratorSpec(
                algorithm="velocity_verlet", ensemble="nve",
                timestep_fs=float(timestep_fs))
        elif isinstance(integrator_spec, dict):
            integrator_spec = IntegratorSpec(**dict(integrator_spec))
        if integrator_spec.timestep_fs != float(timestep_fs):
            raise ValueError("the integrator spec's timestep must match "
                             "timestep_fs")
        self._integrator_spec = integrator_spec
        self.check_probability, self.check_seed = float(check_probability), int(check_seed)
        self.failure_probability, self.tilt = float(failure_probability), float(tilt)
        # The check stream's effective seed (M3A-4): adaptive NVT derives it
        # by role so the check, bath and velocity streams never share a
        # generator, even when all seed fields are equal; historical NVE
        # behavior (the raw seed) is unchanged.
        self._check_seed_effective = (
            derive_stream_seed(self.check_seed, "verification")
            if self._integrator_spec.ensemble == "nvt" else self.check_seed)
        self._rng = np.random.default_rng(self._check_seed_effective)
        self.verification = (IndependentCheckBound(check_probability, failure_probability, tilt)
                             if check_probability > 0 else None)
        self.step = -1
        self.n_evaluations = self.n_accepted = self.n_violations = self.n_calibrations = 0
        self.reference_calls = {"anchor": 0, "probe": 0, "check": 0}
        self._anchor: _Anchor | None = None
        self._segment = 0
        self._model_generation = 0
        self._identity: Atoms | None = None
        self._pending: _Pending | None = None
        self._scheduled_index: int | None = None
        self._scheduled_time_fs: float | None = None
        self._expected_positions: np.ndarray | None = None
        self._callback_failed = False
        self._next_reason = "initial_reference"
        self._deferred_origin: _CalibrationOrigin | None = None
        self._deferred_record: dict | None = None
        self._evaluation_calls_before: dict | None = None
        self._results_model_generation: int | None = None
        self._last_committed_positions: np.ndarray | None = None
        # Set once the integrator schedules its first evaluation: afterwards
        # unscheduled property requests are re-reads of committed facts.
        self._integrator_owned = False
        # Bath wiring for adaptive NVT (M3A-2/3): the runner points
        # ``_bath_dyn`` at the Langevin dynamics so commits can pin the
        # thermostat stream and the realized per-step increments;
        # ``_bath_step`` carries the pre-draw RNG state of the in-flight
        # step from the integrator to the proposal record.
        self._bath_dyn: object | None = None
        self._bath_step: dict | None = None
        self._last_committed_route: str | None = None
        self._last_committed_segment: int | None = None
        self._last_committed_model_id: str | None = None
        # WP02 run records: authoritative event log (optional — direct legacy
        # use stays event-free), task/label ID counters, and the numeric
        # label cache (§5.4; disabled unless the reference declares a
        # fingerprint, i.e. its settings are reliably identifiable).
        self._event_log = event_log
        self._task_counter = 0
        self._label_counter = 0
        self._active_evaluation_id: int | None = None
        self._label_cache = LabelCache(self._engine_fingerprint, enabled=label_cache)
        # Resume machinery: verified probes reusable across a crashed
        # calibration, and the model-artifact publisher the runner installs.
        self._probe_reuse: dict[tuple, dict] = {}
        self._model_publisher: Callable[[str, dict | None], None] | None = None
        if _resume_state is None:
            self._emit(RUN_START, run_id=self.run_id,
                       schema_version=STORE_SCHEMA_VERSION,
                       event_schema_version=EVENT_SCHEMA_VERSION,
                       attempt_ledger=ATTEMPT_LEDGER_PHYSICAL_V1,
                       software_version=__version__,
                       reference_id=self._engine_fingerprint,
                       model_id=self.model_id,
                       surrogate_fingerprint=fingerprint_of(self.surrogate),
                       # The effective random-stream identities (M3A-4): NVT
                       # derives all three by role (role-derive-v1); NVE
                       # keeps the historical raw-seed check stream.
                       streams=({
                           "scheme": STREAM_SCHEME,
                           "velocity_seed": (None if velocity_seed is None
                                             else derive_stream_seed(
                                                 velocity_seed, "velocity")),
                           "thermostat_seed": self._integrator_spec.thermostat_seed,
                           "check_seed": self._check_seed_effective,
                       } if self._integrator_spec.ensemble == "nvt" else {
                           "scheme": None,
                           "check_seed": self.check_seed,
                       }),
                       policy={"force_budget_eV_A": self.force_budget,
                               "timestep_fs": self.timestep_fs,
                               "probe_steps_A": self.probe_steps.tolist(),
                               "numerical_floor_eV_A": self.numerical_floor,
                               "time_cap_fs": self.time_cap_fs,
                               "transverse_cap": self.transverse_cap,
                               "check_probability": self.check_probability,
                               "check_seed": self.check_seed,
                               "failure_probability": failure_probability,
                               "tilt": tilt,
                               "integrator": self._integrator_spec.as_dict()})
        else:
            self._apply_checkpoint_state(_resume_state["state"],
                                         _resume_state["arrays"])

    def _emit(self, event_type: str, **payload: object) -> int | None:
        if self._event_log is None:
            return None
        return self._event_log.append(event_type, payload)

    def _emit_once(self, key: str, event_type: str, **payload: object) -> int | None:
        if self._event_log is None:
            return None
        return self._event_log.append_once(key, event_type, payload)

    def _new_task_id(self) -> str:
        self._task_counter += 1
        return f"{self.run_id}-task-{self._task_counter}"

    def _new_label_id(self) -> str:
        self._label_counter += 1
        return f"{self.run_id}-label-{self._label_counter}"

    def _emit_task(self, *, task_id: str, attempt: int, operation: str,
                   purpose: str | None, status: str, started_unix: float,
                   elapsed_s: float, label_id: str | None = None,
                   cache_hit: bool = False, error: str | None = None) -> None:
        # Task events are the cost-ledger leaves; cpu/gpu/queue stay null
        # when unknown rather than invented.
        self._emit(TASK, task_id=task_id, attempt=attempt, operation=operation,
                   purpose=purpose, status=status,
                   evaluation_id=self._active_evaluation_id,
                   started_unix=started_unix, elapsed_s=elapsed_s,
                   cpu_cores=None, gpu=None, queue_s=None,
                   source="energetic", label_id=label_id,
                   cache_hit=cache_hit, error=error)

    @property
    def n_reference(self) -> int:
        """Actual successful reference calls, including off-trajectory probes."""
        return sum(self.reference_calls.values())

    @property
    def model_generation(self) -> int:
        """Decision-layer model generation; advances on each announced update."""
        return self._model_generation

    @property
    def model_id(self) -> str:
        """Model identity of the current generation (decision-cache key)."""
        return model_id_for(self.surrogate, self._model_generation)

    def _context_for(self, index: int) -> EvaluationContext:
        if self._scheduled_time_fs is not None:
            physical_time_fs = self._scheduled_time_fs
        else:
            # Compatibility layer: direct Calculator use assumes one fixed
            # timestep_fs per evaluation (see the class docstring).
            physical_time_fs = index * self.timestep_fs
        return EvaluationContext(
            run_id=self.run_id,
            step_id=index - 1,
            evaluation_id=index,
            phase=EvaluationPhase.INITIAL if index == 0 else EvaluationPhase.MD_STEP,
            physical_time_fs=physical_time_fs,
            model_id=self.model_id,
        )

    def _policy_dict(self) -> dict:
        return {"force_budget": self.force_budget, "timestep_fs": self.timestep_fs,
                "probe_steps": tuple(float(v) for v in self.probe_steps),
                "numerical_floor": self.numerical_floor,
                "time_cap_fs": self.time_cap_fs,
                "transverse_cap": self.transverse_cap,
                "check_probability": self.check_probability,
                "check_seed": self.check_seed,
                "failure_probability": self.failure_probability,
                "tilt": self.tilt,
                "force_metric": self.force_metric,
                "integrator_spec": self._integrator_spec.as_dict()}

    def _checkpoint_payload(self, boundary_atoms: Atoms) -> tuple[dict, dict]:
        """Complete-step state for ``CheckpointManager.write``.

        ``boundary_atoms`` carries the full-step momenta (the runner calls
        this only at a complete-step boundary); the driving force comes from
        the last committed evaluation and is reusable for the next first
        half-kick.
        """
        origin = self._deferred_origin
        state = {
            "run_id": self.run_id,
            "n_evaluations": self.n_evaluations,
            "n_accepted": self.n_accepted,
            "n_violations": self.n_violations,
            "n_calibrations": self.n_calibrations,
            "reference_calls": dict(self.reference_calls),
            "step": self.step,
            "segment": self._segment,
            "model_generation": self._model_generation,
            "model_id": self.model_id,
            "engine_fingerprint": self._engine_fingerprint,
            "next_reason": self._next_reason,
            "task_counter": self._task_counter,
            "label_counter": self._label_counter,
            "verification": (None if self.verification is None
                             else self.verification.as_dict()),
            "check_rng": rng_state_to_json(self._rng.bit_generator.state),
            # The bath stream at this boundary (NVT): the runner wires the
            # Langevin dynamics in; resume restores it verbatim, never
            # re-seeds.  NVE checkpoints carry None (no bath exists).
            "thermostat": (None if self._bath_dyn is None
                           else {"rng": rng_state_to_json(
                               self._bath_dyn.rng.bit_generator.state)}),
            "policy": self._policy_dict(),
            "driving_energy_eV": (None if not self.results
                                  else float(self.results["energy"])),
            "anchor": self._anchor_record(self._anchor),
            "deferred_origin": None if origin is None else {
                "index": origin.index,
                "label_id": origin.label_id,
                "label_energy_eV": float(origin.label.energy),
                "label_wall_time_s": float(origin.label.wall_time_s),
                "positions_A": origin.atoms.positions.tolist(),
                "momenta": origin.atoms.get_momenta().tolist(),
                "label_forces_eV_A": origin.label.forces.tolist(),
            },
            "constraint": (None if self._projection is None
                           else self._projection.as_dict()),
        }
        updater_state = (self.on_label.state_dict()
                         if _is_stateful(self.on_label) else None)
        arrays = {
            "numbers": boundary_atoms.numbers,
            "cell": boundary_atoms.cell.array,
            "pbc": np.asarray(boundary_atoms.pbc),
            "masses": boundary_atoms.get_masses(),
            "initial_charges": boundary_atoms.get_initial_charges(),
            "initial_magmoms": boundary_atoms.get_initial_magnetic_moments(),
            "positions": boundary_atoms.positions,
            "momenta": boundary_atoms.get_momenta(),
            "driving_forces": (np.zeros((len(boundary_atoms), 3)) if not self.results
                               else np.asarray(self.results["forces"], dtype=float)),
        }
        if updater_state is not None:
            # Tensor states cannot go into state.json: their arrays ride in
            # the checkpoint's own arrays.npz, the JSON keeps digest
            # placeholders (tensor-artifact-v1, see runtime.models).
            ck_arrays: dict[str, np.ndarray] = {}

            def sink(array: np.ndarray, _ck=ck_arrays) -> dict:
                key = f"updater_state:arr{len(_ck)}"
                array = np.asarray(array)
                _ck[key] = (array.copy() if array.ndim == 0
                            else np.ascontiguousarray(array))
                return array_placeholder(key, _ck[key])

            updater_state = dump_state_arrays(updater_state, sink)
            arrays.update(ck_arrays)
        state["updater_state"] = updater_state
        return state, arrays

    def _event_updater_state(self, updater_state: dict | None) -> dict | None:
        """JSON-safe updater state for event payloads: arrays go to the
        run's content-addressed store at ``models/state-arrays/`` (one file
        per unique array), the event keeps digest placeholders."""
        if updater_state is None or self._event_log is None:
            return updater_state
        sink = content_array_sink(
            Path(self._event_log.run_dir) / "models" / "state-arrays")
        return dump_state_arrays(updater_state, sink)

    def _identity_from_arrays(self, arrays: dict) -> Atoms:
        identity = Atoms(numbers=np.array(arrays["numbers"]),
                         cell=np.array(arrays["cell"]),
                         pbc=np.array(arrays["pbc"]))
        identity.set_masses(np.array(arrays["masses"]))
        identity.set_initial_charges(np.array(arrays["initial_charges"]))
        identity.set_initial_magnetic_moments(np.array(arrays["initial_magmoms"]))
        return identity

    def _apply_checkpoint_state(self, state: dict, arrays: dict) -> None:
        """Restore a payload written by :meth:`_checkpoint_payload`."""
        self.n_evaluations = int(state["n_evaluations"])
        self.n_accepted = int(state["n_accepted"])
        self.n_violations = int(state["n_violations"])
        self.n_calibrations = int(state["n_calibrations"])
        self.reference_calls = {key: int(value)
                                for key, value in state["reference_calls"].items()}
        self.step = int(state["step"])
        self._segment = int(state["segment"])
        self._model_generation = int(state["model_generation"])
        self._next_reason = state["next_reason"]
        self._task_counter = int(state["task_counter"])
        self._label_counter = int(state["label_counter"])
        recorded = state["verification"]
        if recorded is not None:
            bound = IndependentCheckBound(float(recorded["probability"]),
                                          float(recorded["failure_probability"]),
                                          float(recorded["tilt"]))
            object.__setattr__(bound, "accepted_count", int(recorded["accepted_count"]))
            object.__setattr__(bound, "detected_count", int(recorded["detected_count"]))
            self.verification = bound
        self._rng = np.random.default_rng(self._check_seed_effective)
        self._rng.bit_generator.state = state["check_rng"]
        self._anchor = (_anchor_from_record(state["anchor"])
                        if state["anchor"] is not None else None)
        constraint = state.get("constraint")
        self._projection = (None if constraint is None else
                            FixAtomsProjection(len(arrays["numbers"]),
                                               list(constraint["indices"])))
        self._identity = self._identity_from_arrays(arrays)
        origin = state["deferred_origin"]
        if origin is not None:
            atoms = self._identity.copy()
            atoms.positions = np.array(origin["positions_A"], dtype=float)
            atoms.set_momenta(np.array(origin["momenta"], dtype=float))
            label = EngineResult(float(origin["label_energy_eV"]),
                                 np.array(origin["label_forces_eV_A"], dtype=float),
                                 None, float(origin["label_wall_time_s"]))
            self._deferred_origin = _CalibrationOrigin(atoms, int(origin["index"]),
                                                       label, label_id=origin["label_id"])
        if state["driving_energy_eV"] is not None:
            # Reusable driving force of the boundary evaluation: the resumed
            # integrator's first half-kick reads this from the ASE cache.
            self.results = {"energy": float(state["driving_energy_eV"]),
                            "forces": np.array(arrays["driving_forces"], dtype=float)}
            self._results_model_generation = self._model_generation

    def _rebuild_label_cache(self, store: Store) -> None:
        """Repopulate the numeric label cache from durable store rows.

        Every row with an engine payload and a durable label ID supplies one
        exact cache entry keyed by (geometry, reference identity, energy
        kind).  The reference identity itself is validated separately at
        resume; a cache entry that no longer matches the current reference
        simply never hits, because the key carries the stored energy kind
        and the same engine fingerprint.
        """
        if not self._label_cache.enabled:
            return
        for row in store._db.select(run_id=self.run_id):
            label_id = row.data.get("engine_label_id")
            payload = row.data.get("engine")
            if label_id is None or payload is None:
                continue
            atoms = row.toatoms()
            label = EngineResult(
                float(payload["energy"]),
                np.asarray(payload["forces"], dtype=float),
                None if payload.get("stress") is None
                else np.asarray(payload["stress"], dtype=float),
                float(payload.get("wall_time_s", 0.0)),
                energy_kind=payload.get("energy_kind", EnergyKind.UNKNOWN),
                force_consistent=payload.get("force_consistent"),
            )
            self._label_cache.put(atoms, label, str(label_id))

    def _replay_committed(self, event: dict, row: object) -> None:
        """Apply one committed evaluation from its records, never recompute.

        Counters, the check stream, the independent bound and the anchor
        evolve exactly as the original commit made them; every value comes
        from the authoritative records (event + store row).
        """
        metadata = row.data.get("metadata") or {}
        accepted = event["route"] == "ml"
        violation = event["violation"]
        # A recalibration before this evaluation's decision defines the
        # anchor the decision used; the live path clears the deferred origin
        # once it has run.
        deferred_rec = metadata.get("calibration_after_previous_label")
        if deferred_rec is not None:
            self._deferred_origin = None
            anchor_rec = deferred_rec.get("anchor")
            self._anchor = (_anchor_from_record(anchor_rec)
                            if anchor_rec is not None else None)
            if anchor_rec is not None:
                self.n_calibrations += 1
        if accepted:
            anchor = self._anchor
            if anchor is not None and metadata.get("forecasts"):
                anchor.open_prefix = [bool(f["prefix_open"])
                                      for f in metadata["forecasts"]]
            self._anchor = None if violation else anchor
            self._next_reason = ("previous_independent_check_violation" if violation
                                 else "reference_required")
        else:
            new_anchor_rec = metadata.get("new_anchor")
            if new_anchor_rec is not None:
                self._anchor = _anchor_from_record(new_anchor_rec)
                self.n_calibrations += 1
            else:
                self._anchor = None
            self._next_reason = ("direction_unavailable_reference"
                                 if self._anchor is None else "reference_required")
        for key, count in event["reference_calls_this_evaluation"].items():
            self.reference_calls[key] += int(count)
        self.n_evaluations += 1
        self.n_accepted += int(accepted)
        self.n_violations += int(violation is True)
        self.step = int(event["context"]["evaluation_id"])
        if accepted and self.check_probability > 0:
            # Each accepted evaluation consumed exactly one draw; advancing
            # the stream keeps every later draw identical to the live run.
            self._rng.random()
        if self.verification is not None:
            self.verification.update(bool(accepted), bool(event["checked"]),
                                     violation if event["checked"] else None)
        self._segment = max(self._segment, int(metadata.get("segment_id") or 0))
        if self._anchor is not None:
            self._segment = max(self._segment, self._anchor.segment)

    def _replay_label_event(self, event: dict, store: Store, updater: object,
                            models_dir: Path) -> None:
        """Apply a label-consumption or model-update event to the updater and
        the deferred-calibration bookkeeping — never re-running training."""
        if not _is_stateful(updater):
            raise ResumeError(
                "the run consumed labels through an updater; supply the same "
                "stateful updater to resume")
        if event["type"] == LABEL_CONSUMED:
            updater_state = event.get("updater_state")
            if updater_state is None:
                raise ResumeError(
                    f"label {event.get('label_id')} has no persisted updater "
                    "state; automatic resume stops here as pending")
            # Array placeholders resolve against the run's content-addressed
            # store (tensor-artifact-v1); digest-verified on load.
            updater.load_state_dict(load_state_arrays(
                updater_state,
                content_array_source(Path(models_dir) / "state-arrays")))
            # An unchanged model on a reference route still recalibrates
            # before the next proposal (live semantics); on an accepted
            # route the anchor is kept untouched.  The label is read from
            # the commit-bound row, never from an orphan at the same step.
            row = store.committed_row(self._event_log, self.run_id,
                                      int(event["evaluation_id"]))
            if row.key_value_pairs["route"] == "dft":
                payload = row.data["engine"]
                label = EngineResult(float(payload["energy"]),
                                     np.asarray(payload["forces"], dtype=float),
                                     None, 0.0)
                self._anchor = None
                self._next_reason = "model_update_requires_recalibration"
                self._deferred_origin = _CalibrationOrigin(
                    row.toatoms(), int(event["evaluation_id"]), label,
                    label_id=event["label_id"])
            return
        artifact = _model_artifact(models_dir, event["model_id"])
        if artifact is None or artifact.get("updater_state") is None:
            raise ResumeError(
                f"model artifact for {event['model_id']!r} is missing or "
                "incomplete; automatic resume stops here as pending")
        problems: list[str] = []
        if artifact.get("model_id") != event["model_id"]:
            problems.append("model_id mismatch with the commit")
        if int(artifact.get("format_version", -1)) != MODEL_ARTIFACT_FORMAT_VERSION:
            problems.append("unsupported artifact schema version")
        if int(artifact.get("generation", -1)) != int(event["generation"]):
            problems.append("generation mismatch with the commit")
        expected_parent = model_id_for(self.surrogate, self._model_generation)
        if artifact.get("parent_model_id") != expected_parent:
            problems.append(
                f"parent chain mismatch: artifact names "
                f"{artifact.get('parent_model_id')!r}, the replayed chain "
                f"expects {expected_parent!r}")
        committed_digest = event.get("artifact_digest")
        if committed_digest is not None and \
                artifact_digest(artifact) != committed_digest:
            problems.append(
                "artifact content does not match the digest bound into the "
                "commit; the artifact looks tampered with")
        if problems:
            raise ResumeError(
                f"model artifact for {event['model_id']!r} failed "
                "verification: " + "; ".join(problems))
        updater.load_state_dict(resolve_artifact_state(
            artifact["updater_state"],
            Path(models_dir) / str(event["model_id"]).replace("/", "_")))
        self._model_generation = int(event["generation"])
        self._anchor = None
        if event.get("origin_violation"):
            self._next_reason = "previous_independent_check_violation"
            return
        origin_eval = int(event["origin_evaluation_id"])
        # The re-anchor label comes from the commit-bound row of the origin
        # evaluation — the same verified row the resume boundary uses (C1).
        row = store.committed_row(self._event_log, self.run_id, origin_eval)
        payload = row.data.get("engine")
        label = EngineResult(float(payload["energy"]),
                             np.asarray(payload["forces"], dtype=float), None, 0.0)
        self._next_reason = "model_update_requires_recalibration"
        self._deferred_origin = _CalibrationOrigin(
            row.toatoms(), origin_eval, label,
            label_id=event["origin_label_id"])

    def _rebuild_pending(self, proposal: dict, atoms: Atoms) -> _Pending:
        """Rebuild the frozen pending decision of an uncommitted evaluation."""
        prediction_payload = proposal["prediction"]
        prediction = SurrogatePrediction(
            float(prediction_payload["energy_eV"]),
            np.array(prediction_payload["forces_eV_A"], dtype=float), None,
            np.array(prediction_payload["uncertainty"], dtype=float))
        anchor_rec = proposal.get("anchor_record")
        anchor = (_anchor_from_record(anchor_rec)
                  if anchor_rec is not None else None)
        context = EvaluationContext(**proposal["context"])
        self._deferred_record = proposal.get("deferred_record")
        if self._deferred_record is not None:
            # The recalibration for the replayed update already ran and its
            # outcome is frozen in this proposal (anchor record, counters);
            # the replayed origin must not trigger it a second time (C4).
            self._deferred_origin = None
        if proposal.get("segment") is not None \
                and self._deferred_record is not None:
            # A *deferred* recalibration that completed before this proposal
            # advanced these counters but is committed nowhere else, so the
            # frozen values restore it when the rebuilt evaluation commits —
            # exactly once.  Proposals without a deferred calibration (e.g.
            # the reference route, whose calibration runs later in _finish)
            # must NOT restore: their frozen counters are older than the
            # calibration their own commit is about to complete (R6).
            pending_counters = (int(proposal["segment"]),
                                int(proposal["n_calibrations"]))
        else:
            pending_counters = None
        if proposal.get("check_rng_after") is not None:
            # Continue the check stream exactly after this evaluation's
            # consumed draw; without it the next fresh draw repeats the
            # persisted one (F01).
            self._rng.bit_generator.state = proposal["check_rng_after"]
        bath_step = proposal.get("bath_step")
        return _Pending(
            atoms, int(context.evaluation_id), prediction, anchor,
            bool(proposal["accepted"]), proposal["reason"],
            [dict(f) for f in proposal["forecasts"]],
            proposal["selected_direction"],
            [bool(v) for v in proposal["open_prefix"]],
            proposal["frozen_energy_eV"],
            None if proposal["frozen_forces_eV_A"] is None
            else np.array(proposal["frozen_forces_eV_A"], dtype=float),
            bool(proposal["checked"]), proposal["check_draw"],
            dict(proposal["calls_before"]),
            context=context,
            model_generation=int(proposal["model_generation"]),
            bath_step=None if bath_step is None else dict(bath_step),
            restored_counters=pending_counters)

    def _check_identity(self, atoms: Atoms) -> None:
        """Reject any change that must never happen mid-run.

        Masses and constraints are not part of ASE's cache-invalidation
        state, so this check must run before cached properties are served
        too — not only inside :meth:`calculate`.  Constraints are FixAtoms
        only (everything else is rejected explicitly); the fixed set is
        frozen for the run's lifetime.
        """
        if not len(atoms):
            raise ValueError("energetic dynamics requires nonempty Atoms")
        projection = validate_constraints(atoms)  # rejects non-FixAtoms kinds
        if self._identity is None:
            self._projection = projection
        elif (None if projection is None else projection.indices) != (
                None if self._projection is None else self._projection.indices):
            raise ValueError("constraints must not change mid-run (started "
                             "unconstrained or with a fixed FixAtoms set)")
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

    def _validate_atoms(self, atoms: Atoms) -> None:
        self._check_identity(atoms)
        if self._expected_positions is not None and not np.allclose(
            atoms.positions, self._expected_positions, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("positions do not match the scheduled integrator step")

    def get_property(self, name, atoms: Atoms | None = None, allow_calculation: bool = True):
        # A committed evaluation's cached results are valid only for the model
        # generation that produced them. In the compatibility layer (no
        # integrator schedule), a same-geometry request after a model update
        # is a NEW logical evaluation with a new decision, not a replay of
        # the old one. Integrator-driven runs schedule every evaluation
        # explicitly: an unscheduled request is by definition a re-read of
        # the committed fact and must replay (§5.2) — the integrator itself
        # re-reads forces between steps for bookkeeping.
        if (not self._integrator_owned and self.results
                and self._results_model_generation is not None
                and self._results_model_generation != self._model_generation):
            self.results = {}
        # ASE skips calculate() entirely when nothing it tracks has changed,
        # but it does not track masses or constraints: validate the immutable
        # physical state before serving even a fully cached property. A
        # rejected evaluation must leave no stale results behind.
        if atoms is not None and self._identity is not None:
            try:
                self._check_identity(atoms)
            except ValueError:
                self.results = {}
                raise
        return super().get_property(name, atoms, allow_calculation)

    def _predict(self, atoms: Atoms, purpose: str = "proposal") -> SurrogatePrediction:
        work = atoms.copy()
        task_id = self._new_task_id()
        started_unix = time.time()
        start = time.perf_counter()
        try:
            prediction = self.surrogate.predict(work)
        except Exception as error:
            self._emit_task(task_id=task_id, attempt=1, operation="inference",
                            purpose=purpose, status="failed",
                            started_unix=started_unix,
                            elapsed_s=time.perf_counter() - start,
                            error=repr(error))
            raise
        self._emit_task(task_id=task_id, attempt=1, operation="inference",
                        purpose=purpose, status="success",
                        started_unix=started_unix,
                        elapsed_s=time.perf_counter() - start)
        if not _same_state(atoms, work):
            raise ValueError("surrogate.predict mutated its atomic input")
        energy, forces, stress = _label_arrays(prediction, len(atoms))
        uncertainty = np.array(prediction.uncertainty, dtype=float, copy=True)
        if (uncertainty.shape != (len(atoms),) or np.isinf(uncertainty).any()
                or np.any(uncertainty < 0)):
            raise ValueError("uncertainty must be a nonnegative (N,) array or NaN")
        return SurrogatePrediction(
            energy, forces, stress, uncertainty,
            energy_kind=getattr(prediction, "energy_kind", EnergyKind.UNKNOWN),
            force_consistent=getattr(prediction, "force_consistent", None),
        )

    def _reference(self, atoms: Atoms, purpose: str, *,
                   event_purpose: str | None = None,
                   task_id: str | None = None,
                   attempt: int = 1) -> tuple[EngineResult, str]:
        """One reference execution; returns ``(label, durable label_id)``.

        ``purpose`` keys the legacy success counters (anchor/probe/check);
        ``event_purpose`` is the ledger vocabulary (anchor/refusal/probe/
        verification/diagnostic) and defaults to the counter key mapping.
        """
        # Reference settings identity is fixed for a run; check it before the
        # expensive call so a mid-run settings swap fails cheap, not after an
        # SCF whose label would silently mix conventions with older anchors.
        if fingerprint_of(self.engine) != self._engine_fingerprint:
            raise ValueError(
                "reference settings identity changed mid-run; start a new run "
                "instead of mixing labels from different reference settings"
            )
        if task_id is None:
            task_id = self._new_task_id()
        ledger_purpose = {"check": "verification"}.get(purpose, purpose) \
            if event_purpose is None else event_purpose
        work = atoms.copy()
        started_unix = time.time()
        start = time.perf_counter()
        try:
            # One logical request; the physical launches inside it are
            # attempt events (self-reported by engines accepting
            # ``request_id``, one-per-call otherwise — see
            # events.physical_attempt).
            with physical_attempt(self.engine, self._event_log,
                                  operation="reference",
                                  request_id=task_id,
                                  purpose=ledger_purpose,
                                  source="energetic") as attempt_kwargs:
                result = self.engine.compute(work, **attempt_kwargs)
            if not _same_state(atoms, work):
                raise ValueError("engine.compute mutated its atomic input")
            energy, forces, stress = _label_arrays(result, len(atoms))
            wall = _positive(result.wall_time_s, "reference wall_time_s", zero=True)
        except Exception as error:
            self._emit_task(task_id=task_id, attempt=attempt, operation="reference",
                            purpose=ledger_purpose, status="failed",
                            started_unix=started_unix,
                            elapsed_s=time.perf_counter() - start,
                            error=repr(error))
            if isinstance(error, (EngineError, EventLogError)):
                raise
            raise EngineError(f"invalid {purpose} reference evaluation: {error}") from error
        label_id = self._new_label_id()
        self.reference_calls[purpose] += 1
        label = EngineResult(
            energy, forces, stress, wall,
            energy_kind=getattr(result, "energy_kind", EnergyKind.UNKNOWN),
            force_consistent=getattr(result, "force_consistent", None),
        )
        self._emit_task(task_id=task_id, attempt=attempt, operation="reference",
                        purpose=ledger_purpose, status="success",
                        started_unix=started_unix,
                        elapsed_s=time.perf_counter() - start,
                        label_id=label_id)
        self._label_cache.put(atoms, label, label_id)
        return label, label_id

    def _directions(self, atoms: Atoms) -> np.ndarray | None:
        """Probe direction(s) for calibration.

        NVE keeps the historical default (``atoms.velocities``).  For NVT
        the force evaluation happens mid-step: the Atoms momenta are the
        *previous* boundary momenta then, not ASE's internal instantaneous
        velocity (that lives on the dynamics object), so the defined
        default is the realized displacement of the step that produced
        this configuration — random increment included, transverse
        component and all.  Unavailable (no committed boundary yet) or
        exactly zero displacements defer calibration to the existing safe
        fallback (reference route, reason recorded).
        """
        if self.direction is not None:
            raw = self.direction(atoms.copy())
        elif self._integrator_spec.ensemble == "nvt":
            if self._last_committed_positions is None:
                return None
            raw = atoms.positions - self._last_committed_positions
        else:
            raw = atoms.get_velocities()
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
        directions = directions / lengths[:, None, None]
        if self._projection is not None:
            # Fixed DOFs never enter a probe direction (and a probe then
            # never displaces a fixed atom, on either force path).
            return self._projection.project_directions(directions)
        return directions

    def _calibrate(self, pending: _Pending) -> _Anchor | None:
        directions = self._directions(pending.atoms)
        if directions is None:
            return None
        correction = pending.label.forces - pending.prediction.forces
        shape = (len(directions), 2, len(pending.atoms), 3)
        plus_d, minus_d, plus_r, minus_r = [np.empty(shape) for _ in range(4)]
        records = []
        n_reused = 0
        for d, direction in enumerate(directions):
            for h_index, h in enumerate(self.probe_steps):
                for sign, displacements, residuals in ((1, plus_d, plus_r), (-1, minus_d, minus_r)):
                    # A crashed calibration's verified probes are reused
                    # instead of re-executed (each was persisted on success).
                    reuse_key = (pending.index, d, h_index, sign, self._model_generation)
                    reused = self._probe_reuse.get(reuse_key)
                    if (reused is not None and not np.allclose(
                            reused["displacement_A"], sign * h * direction,
                            rtol=0, atol=1e-7)):
                        reused = None  # stale record from a different origin
                    if reused is not None:
                        record = dict(reused)
                        displacement = np.array(record["displacement_A"], dtype=float)
                        prediction = SurrogatePrediction(
                            float(record["base_energy_eV"]),
                            np.array(record["base_forces_eV_A"], dtype=float),
                            None, np.full(len(pending.atoms), np.nan))
                        label = EngineResult(
                            float(record["reference_energy_eV"]),
                            np.array(record["reference_forces_eV_A"], dtype=float),
                            None, 0.0)
                        n_reused += 1
                    else:
                        probe = pending.atoms.copy()
                        probe.positions += sign * h * direction
                        prediction = self._predict(probe, purpose="probe")
                        label, probe_label_id = self._reference(probe, "probe")
                        displacement = probe.positions - pending.atoms.positions
                        record = {"direction": d, "step_A": float(h), "sign": sign,
                                        "phase": str(EvaluationPhase.PROBE),
                                        "evaluation_id": pending.index,
                                        "label_id": probe_label_id,
                                        "displacement_A": displacement.tolist(),
                                        "base_energy_eV": prediction.energy,
                                        "reference_energy_eV": label.energy,
                                        "base_forces_eV_A": prediction.forces.tolist(),
                                        "reference_forces_eV_A": label.forces.tolist()}
                        # Persist every verified probe immediately — never wait
                        # for the whole probe set before writing.
                        self._emit_once(
                            f"probe:{self.run_id}:{pending.index}:{d}:{h_index}:{sign}",
                            PROBE_COMPLETED, evaluation_id=pending.index,
                            model_generation=self._model_generation,
                            model_id=self.model_id,
                            direction=d, step_A=float(h), sign=sign,
                            record=record)
                    displacements[d, h_index] = displacement
                    residual = prediction.forces + correction - label.forces
                    if (self._projection is not None
                            and self.force_metric == "active_dofs_max_atom"):
                        # The budget controls the free coordinates: fixed-DOF
                        # components leave the error norms (the raw residual
                        # stays in the probe records as the diagnostic).
                        residual = self._projection.project_forces(residual)
                    residuals[d, h_index] = residual
                    records.append(record)
        pending.probe_records = records
        try:
            responses = estimate_responses(directions, self.probe_steps, plus_d, minus_d, plus_r, minus_r)
        except DegenerateResponseError:
            # C_rw is undefined, so keep using reference forces. A zero
            # finite-probe derivative does not establish a global bound.
            return None
        model_id = pending.context.model_id if pending.context is not None else self.model_id
        calibration = {"probe_steps_A": self.probe_steps.tolist(), "probes": records,
                       "responses": [response.as_dict() for response in responses],
                       "force_call_count": 1 + 4 * len(directions),
                       "reused_probes": n_reused,
                       "model_id": model_id,
                       "type": "two_scale_empirical_reference_probes"}
        self._segment += 1
        self.n_calibrations += 1
        return _Anchor(self._segment, pending.index, pending.atoms.positions.copy(),
                       pending.prediction, pending.label, correction, responses,
                       [True] * len(responses), calibration,
                       model_generation=self._model_generation)

    @staticmethod
    def _anchor_record(anchor: _Anchor | None) -> dict | None:
        if anchor is None:
            return None
        return {"segment_id": anchor.segment, "evaluation_index": anchor.index,
                "positions_A": anchor.positions.tolist(),
                "base_energy_eV": anchor.prediction.energy,
                "reference_energy_eV": anchor.label.energy,
                "base_forces_eV_A": anchor.prediction.forces.tolist(),
                "reference_forces_eV_A": anchor.label.forces.tolist(),
                "correction_eV_A": anchor.correction.tolist(),
                "open_prefix": [bool(v) for v in anchor.open_prefix],
                "model_generation": anchor.model_generation,
                "calibration": anchor.calibration}

    def _prepare_updated_model(self) -> None:
        """Calibrate only after a stored label's update, before a new proposal."""
        origin = self._deferred_origin
        if origin is None:
            return
        # Recalibration tasks belong to the origin evaluation's identity.
        self._active_evaluation_id = origin.index
        prediction = self._predict(origin.atoms, purpose="calibration")
        # Recalibration probes belong to the origin evaluation: they reuse its
        # identity and physical time (fixed-step value), never advancing the
        # clock, but run under the NEW model generation.
        context = EvaluationContext(self.run_id, origin.index - 1, origin.index,
                                    EvaluationPhase.PROBE,
                                    origin.index * self.timestep_fs, self.model_id)
        pending = _Pending(origin.atoms, origin.index, prediction, None, False,
                           "model_update_calibration", [], None, [], None, None,
                           False, None, self.reference_calls.copy(), label=origin.label,
                           context=context, model_generation=self._model_generation)
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
        self._active_evaluation_id = index
        prediction = self._predict(atoms)
        anchor = self._anchor
        if anchor is not None and anchor.model_generation != self._model_generation:
            # A calibration from an older model generation must never drive a
            # new evaluation (WP01 decision-cache rule).
            anchor = None
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
        if accepted and self._projection is not None:
            # The driving force that actually propagates: fixed DOFs zeroed.
            forces = self._projection.project_forces(forces)
        energy = (prediction.energy - float(np.sum(anchor.correction *
                  (atoms.positions - anchor.positions))) + anchor.label.energy -
                  anchor.prediction.energy) if accepted else None
        draw = float(self._rng.random()) if accepted and self.check_probability > 0 else None
        checked = draw is not None and draw < self.check_probability
        bath_step = None
        if (self._bath_step is not None
                and self._bath_step["evaluation_id"] == index):
            # The integrator pinned the pre-draw bath state for this step;
            # the proposal record carries it so a rebuilt pending resumes
            # the identical stochastic step (M3A-3).  The boundary the step
            # started from is frozen too: at freeze time
            # ``_last_committed_positions`` still names the previous
            # committed evaluation (it advances at this one's commit).
            bath_step = {
                "rng_before": self._bath_step["rng_before"],
                "boundary_positions_A": None
                if self._last_committed_positions is None
                else self._last_committed_positions.tolist()}
        return _Pending(atoms.copy(), index, prediction, anchor, accepted, reason,
                        forecasts, selected, open_prefix, energy, forces, checked, draw,
                        self._evaluation_calls_before.copy(),
                        context=self._context_for(index),
                        model_generation=self._model_generation,
                        bath_step=bath_step)

    def _constraint_record(self, pending: _Pending) -> dict | None:
        """Raw physical forces, the projected driving force and the actual
        constrained displacement of one evaluation (FixAtoms runs only)."""
        if self._projection is None:
            return None
        if pending.accepted:
            raw = pending.prediction.forces + pending.anchor.correction
        else:
            raw = pending.label.forces
        record = self._projection.as_dict()
        record["force_metric"] = self.force_metric
        record["raw_forces_eV_A"] = np.asarray(raw, dtype=float).tolist()
        if self._last_committed_positions is not None:
            displacement = pending.atoms.positions - self._last_committed_positions
            record["actual_displacement_A"] = displacement.tolist()
            record["max_fixed_displacement_A"] = \
                self._projection.max_fixed_displacement(displacement)
        return record

    def _observed(self, pending: _Pending) -> dict | None:
        anchor, label = pending.anchor, pending.label
        if anchor is None or label is None:
            return None
        residual = pending.prediction.forces + anchor.correction - label.forces
        if self._projection is not None:
            error = self._projection.metric_norm(residual, self.force_metric)
        else:
            error = float(np.linalg.norm(residual, axis=1).max())
        work = residual_work(anchor.positions, pending.atoms.positions,
                             anchor.prediction.energy, pending.prediction.energy,
                             anchor.label.energy, label.energy, anchor.correction)
        return {"segment_id": anchor.segment, "residual_eV_A": residual.tolist(),
                "max_force_error_eV_A": error,
                "force_metric": self.force_metric,
                "endpoint_work_eV": float(work),
                "force_budget_exceeded": error > self.force_budget,
                "observed_coefficient_A2_eV": 2 * work / error**2 if error > 0 else None}

    def _finish(self, pending: _Pending) -> None:
        self._active_evaluation_id = pending.index
        if (not pending.accepted or pending.checked) and pending.label is None:
            counter_purpose = "check" if pending.checked else "anchor"
            event_purpose = ("verification" if pending.checked else
                             "anchor" if pending.reason == "initial_reference"
                             else "refusal")
            if pending.checked:
                # §5.4: the check draw already happened; a fully matching
                # cached label (same reference settings, geometry and energy
                # convention) may complete the verification — the check
                # counts normally, new physical SCF executions: zero.
                cached = self._label_cache.get(
                    pending.atoms, engine_capabilities(self.engine).energy_kind)
                if cached is not None:
                    pending.label, pending.label_id = cached
                    self._emit_task(task_id=self._new_task_id(), attempt=1,
                                    operation="reference", purpose="verification",
                                    status="cache_hit", started_unix=time.time(),
                                    elapsed_s=0.0, label_id=pending.label_id,
                                    cache_hit=True)
            if pending.label is None:
                if pending.label_task_id is None:
                    pending.label_task_id = self._new_task_id()
                pending.label_attempt += 1
                pending.label, pending.label_id = self._reference(
                    pending.atoms, counter_purpose, event_purpose=event_purpose,
                    task_id=pending.label_task_id, attempt=pending.label_attempt)
        observed = self._observed(pending)
        violation = bool(observed["force_budget_exceeded"]) if pending.checked else None
        if not pending.accepted:
            pending.energy, pending.forces = pending.label.energy, pending.label.forces.copy()
            if self._projection is not None:
                # The reference route drives with the same constraint
                # semantics as the surrogate path: fixed DOFs zeroed.
                pending.forces = self._projection.project_forces(pending.forces)
            if not pending.calibration_done and self.on_label is None:
                pending.new_anchor = self._calibrate(pending)
                pending.calibration_done = True
        bound = copy.deepcopy(self.verification)
        if bound is not None:
            bound.update(pending.accepted, pending.checked, violation)
        anchor = pending.anchor
        new_anchor = pending.new_anchor
        energy_source = pending.prediction if pending.accepted else pending.label
        drive = SurrogatePrediction(pending.energy, pending.forces.copy(), None,
                                    pending.prediction.uncertainty.copy(),
                                    energy_kind=getattr(energy_source, "energy_kind",
                                                        EnergyKind.UNKNOWN),
                                    force_consistent=getattr(energy_source, "force_consistent",
                                                             None))
        metadata = {
            "method": "energetic_force_error", "evaluation_index": pending.index,
            "time_fs": pending.index * self.timestep_fs, "timestep_fs": self.timestep_fs,
            "context": None if pending.context is None else pending.context.as_dict(),
            "reference_id": self._engine_fingerprint,
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
            "constraint": self._constraint_record(pending),
        }
        io_task_id = self._new_task_id()
        io_started = time.time()
        io_start = time.perf_counter()
        try:
            row_id = self.store.append(self.run_id, pending.index - 1, pending.atoms,
                                       "ml" if pending.accepted else "dft",
                                       surrogate=pending.prediction, engine=pending.label,
                                       reason=pending.reason, metadata=metadata, driving=drive,
                                       label_id=pending.label_id, dedupe=True)
        except Exception as error:
            self._emit_task(task_id=io_task_id, attempt=1, operation="io",
                            purpose="trajectory_append", status="failed",
                            started_unix=io_started,
                            elapsed_s=time.perf_counter() - io_start,
                            error=repr(error))
            raise
        self._emit_task(task_id=io_task_id, attempt=1, operation="io",
                        purpose="trajectory_append", status="success",
                        started_unix=io_started,
                        elapsed_s=time.perf_counter() - io_start)
        # Commit only after a valid label/calibration and a successful append.
        self.verification = bound
        self.n_evaluations += 1
        self.n_accepted += int(pending.accepted)
        self.n_violations += int(violation is True)
        self.step = pending.index
        self._last_committed_positions = pending.atoms.positions.copy()
        if pending.accepted:
            anchor.open_prefix = pending.open_prefix
            self._anchor = None if violation else anchor
            self._next_reason = "previous_independent_check_violation" if violation else "reference_required"
        else:
            self._anchor = new_anchor
            self._next_reason = "direction_unavailable_reference" if new_anchor is None else "reference_required"
        self.results = {"energy": drive.energy, "forces": drive.forces.copy()}
        self._results_model_generation = self._model_generation
        committed_payload = {
            "context": None if pending.context is None else pending.context.as_dict(),
            "route": "ml" if pending.accepted else "dft",
            "reason": pending.reason,
            "label_id": pending.label_id,
            # The commit binds the exact row it authorizes; an orphan row at
            # the same step from a crash between append and commit is never
            # authoritative (F03).
            "row_id": int(row_id),
            "row_digest": self.store.row_digest(self.store.row_by_id(int(row_id))),
            "driving_energy_eV": float(pending.energy),
            "checked": pending.checked,
            "violation": violation,
            "observed": observed,
            "verification": None if bound is None else bound.as_dict(),
            "reference_calls_this_evaluation": {
                key: count - pending.calls_before[key]
                for key, count in self.reference_calls.items()},
            "segment_id": None if anchor is None else anchor.segment,
            "model_id": self.model_id,
        }
        if pending.index == 0:
            committed_payload["input_hash"] = atoms_input_hash(pending.atoms)
        if (self._integrator_spec.algorithm == "langevin"
                and self._bath_dyn is not None):
            # The commit pins the bath stream at this evaluation (post-draw
            # for a step evaluation, the seeded state for the initial one);
            # a step evaluation additionally records the realized random
            # increments verbatim — the strict completion key for the
            # boundary, so resume never re-draws them (M3A-3).  Units are
            # ASE-internal (ase.md.langevin rnd_pos/rnd_vel), consumed only
            # by the matching completion formula.
            committed_payload["thermostat_rng"] = rng_state_to_json(
                self._bath_dyn.rng.bit_generator.state)
            # The commit names its own integrator settings so the boundary
            # completion (resume and export) is self-describing.
            committed_payload["integrator"] = self._integrator_spec.as_dict()
            if pending.bath_step is not None:
                committed_payload["bath_step"] = {
                    "boundary_positions_A":
                        pending.bath_step["boundary_positions_A"],
                    "rnd_pos": np.asarray(self._bath_dyn.rnd_pos,
                                          dtype=float).tolist(),
                    "rnd_vel": np.asarray(self._bath_dyn.rnd_vel,
                                          dtype=float).tolist()}
        self._last_committed_route = committed_payload["route"]
        self._last_committed_segment = committed_payload["segment_id"]
        # The driving model identity of this evaluation — a label callback
        # below may advance the generation within the same MD step, so the
        # step record must bind the commit's frozen identity, not the live
        # one.
        self._last_committed_model_id = committed_payload["model_id"]
        self._emit_once(f"evaluation:{self.run_id}:{pending.index}",
                        EVALUATION_COMMITTED, **committed_payload)
        if pending.restored_counters is not None:
            # The recalibration frozen into a rebuilt proposal becomes
            # visible exactly once, with its evaluation's commit (C4).
            self._segment, self.n_calibrations = pending.restored_counters
        self._pending = None
        self._deferred_record = None
        self._evaluation_calls_before = None
        if pending.label is not None and self.on_label is not None:
            candidate = self._anchor
            self._anchor = None
            self._next_reason = "model_update_requires_recalibration"
            training_task_id = self._new_task_id()
            training_started = time.time()
            training_start = time.perf_counter()
            try:
                changed = self.on_label(LabelObservation(pending.index - 1, pending.atoms.copy(),
                                                        copy.deepcopy(pending.prediction),
                                                        copy.deepcopy(pending.label),
                                                        label_id=pending.label_id))
            except Exception as error:
                self._emit_task(task_id=training_task_id, attempt=1,
                                operation="training", purpose="model_update",
                                status="failed", started_unix=training_started,
                                elapsed_s=time.perf_counter() - training_start,
                                label_id=pending.label_id, error=repr(error))
                self.results = {}
                self._callback_failed = True
                raise
            # Everything after a successful callback — the success-task log,
            # the state snapshot, the rejection record, the artifact persist
            # and the commit event — is ONE rollback domain (C3): a failure
            # anywhere in it restores the parent model and updater state
            # before the error stops the run, so a stopped run always
            # satisfies the atomic publish contract.
            try:
                self._emit_task(task_id=training_task_id, attempt=1,
                                operation="training", purpose="model_update",
                                status="success", started_unix=training_started,
                                elapsed_s=time.perf_counter() - training_start,
                                label_id=pending.label_id)
                updater_state = (self.on_label.state_dict()
                                 if _is_stateful(self.on_label) else None)
                pop_rejection = getattr(self.on_label, "pop_rejection", None)
                rejection = pop_rejection() if callable(pop_rejection) else None
                if rejection is not None:
                    # A rolled-back update attempt: the parent model carries on,
                    # the attempt's cost and outcome are recorded (ledger is
                    # append-only, so the training time stays billed).
                    self._emit_once(f"update-rejected:{pending.label_id}:eval-{pending.index}",
                                    UPDATE_REJECTED,
                                    evaluation_id=pending.index,
                                    origin_label_id=pending.label_id,
                                    model_id=self.model_id,
                                    reason=rejection.get("reason"),
                                    metrics=rejection.get("metrics"),
                                    label_ids=rejection.get("label_ids"))
                if changed is not False:
                    # Anything except exactly False declares a model change.
                    # The update activates only after the artifact is
                    # persisted and the commit event is written: candidate →
                    # validate → persist → activate (R3).
                    parent_model_id = self.model_id
                    next_generation = self._model_generation + 1
                    new_model_id = model_id_for(self.surrogate, next_generation)
                    update_record = None
                    get_update_record = getattr(self.on_label, "update_record", None)
                    if callable(get_update_record):
                        update_record = get_update_record()
                    artifact = {
                        "generation": next_generation,
                        "parent_model_id": parent_model_id,
                        "label_ids": (update_record or {}).get("label_ids",
                                                              [pending.label_id]),
                        "recipe": (update_record or {}).get("recipe"),
                        "training": (update_record or {}).get("training"),
                        "updater_state": updater_state,
                    }
                    published = None
                    if self._model_publisher is not None:
                        published = self._model_publisher(new_model_id, artifact)
                    if self._event_log is not None:
                        # The artifact's content digest is bound into the
                        # commit — outside the rewritable artifact file — so
                        # a tampered artifact is detected at load time (R2).
                        digest_source = (published if published is not None else
                                         {"model_id": new_model_id,
                                          "format_version": MODEL_ARTIFACT_FORMAT_VERSION,
                                          **artifact})
                        artifact_digest_value = artifact_digest(digest_source)
                        self._emit_once(
                            f"model-update:{pending.label_id}:eval-{pending.index}",
                            MODEL_UPDATE,
                            generation=next_generation,
                            model_id=new_model_id,
                            origin_evaluation_id=pending.index,
                            origin_label_id=pending.label_id,
                            origin_violation=bool(violation),
                            label_ids=artifact["label_ids"],
                            artifact_digest=artifact_digest_value,
                            updater_state=(published["updater_state"]
                                           if published is not None
                                           else updater_state))
                    self._model_generation = next_generation
                else:
                    # Label consumed without a model change: the updater's
                    # continuation state still advanced — persist it so a replay
                    # never re-consumes or misaligns the update queue.
                    self._emit_once(f"consumed:{pending.label_id}", LABEL_CONSUMED,
                                    label_id=pending.label_id,
                                    evaluation_id=pending.index,
                                    model_id=self.model_id,
                                    updater_state=self._event_updater_state(
                                        updater_state))
            except Exception:
                if changed is not False:
                    rollback = getattr(self.on_label, "rollback_accepted", None)
                    if callable(rollback):
                        rollback()
                raise
            if pending.accepted and changed is False:
                self._anchor = candidate
                if violation:
                    self._next_reason = "previous_independent_check_violation"
            elif not violation:
                self._deferred_origin = _CalibrationOrigin(pending.atoms.copy(), pending.index,
                                                          copy.deepcopy(pending.label),
                                                          label_id=pending.label_id)
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
        if (self._scheduled_index is not None
                and self._expected_positions is None
                and self._integrator_spec.algorithm == "langevin"):
            # NVT hook (M3A-2): the schedule fixed the evaluation index and
            # physical time before the step; the configuration itself is
            # adopted here, at evaluation time — ASE finalizes positions
            # (random increments and FixAtoms applied) before requesting
            # forces, so the forecast and the commit see exactly the
            # configuration that propagates.
            self._expected_positions = self.atoms.positions.copy()
        index = self.n_evaluations if self._scheduled_index is None else self._scheduled_index
        if index != self.n_evaluations:
            raise ValueError("evaluation clock is not consecutive; do not modify a stored MD state")
        if self._pending is None:
            if self._evaluation_calls_before is None:
                self._evaluation_calls_before = self.reference_calls.copy()
            self._prepare_updated_model()
            self._pending = self._freeze(self.atoms, index)
            # Persist the frozen proposal and check draw BEFORE any external
            # computation of this evaluation (§5.3); idempotent per
            # evaluation ID, so a retry never duplicates it. The record is
            # complete enough to rebuild the pending decision after a crash:
            # positions, frozen prediction and decision, and the anchor used.
            self._emit_once(
                f"proposal:{self.run_id}:{index}", EVALUATION_PROPOSED,
                context=self._pending.context.as_dict(),
                positions_A=self._pending.atoms.positions.tolist(),
                prediction={"energy_eV": float(self._pending.prediction.energy),
                            "forces_eV_A": self._pending.prediction.forces.tolist(),
                            "uncertainty": self._pending.prediction.uncertainty.tolist()},
                accepted=self._pending.accepted, reason=self._pending.reason,
                forecasts=self._pending.forecasts,
                checked=self._pending.checked, check_draw=self._pending.draw,
                # The bit-generator state after this evaluation's draw:
                # resuming this pending continues the check stream after the
                # consumed draw instead of re-drawing it (F01).
                check_rng_after=(None if self._pending.draw is None else
                                 rng_state_to_json(self._rng.bit_generator.state)),
                frozen_energy_eV=self._pending.energy,
                frozen_forces_eV_A=(None if self._pending.forces is None
                                    else self._pending.forces.tolist()),
                open_prefix=[bool(v) for v in self._pending.open_prefix],
                selected_direction=self._pending.selected_direction,
                # NVT pending record: the pre-draw bath state of this step
                # (M3A-3).  The realized increments stay on the dynamics and
                # enter the commit; rng_before is the strict rebuild key.
                bath_step=(None if self._pending.bath_step is None else
                           dict(self._pending.bath_step)),
                calls_before=dict(self._pending.calls_before),
                model_generation=self._pending.model_generation,
                anchor_record=self._anchor_record(self._pending.anchor),
                deferred_record=self._deferred_record,
                segment_id=(None if self._pending.anchor is None
                            else self._pending.anchor.segment),
                # The recalibration that ran before this proposal (when any)
                # advanced these counters but is committed nowhere else; the
                # frozen values restore it exactly once on rebuild (C4).
                segment=self._segment,
                n_calibrations=self.n_calibrations,
            )
        elif (self._pending.index != index
              or self._pending.model_generation != self._model_generation
              or not _same_state(self._pending.atoms, self.atoms, momenta=True)):
            raise ValueError("retry requires the same pending geometry, momenta, "
                             "model generation and evaluation time")
        try:
            self._finish(self._pending)
        except Exception:
            self.results = {}
            raise

    def _schedule(self, index: int, positions: np.ndarray | None = None, *,
                  physical_time_fs: float | None = None) -> None:
        if index != self.n_evaluations:
            raise ValueError("the MD clock must advance by one force evaluation")
        if physical_time_fs is not None:
            physical_time_fs = float(physical_time_fs)
            if not math.isfinite(physical_time_fs) or physical_time_fs < 0:
                raise ValueError("physical_time_fs must be finite and >= 0")
        self._scheduled_index = index
        self._scheduled_time_fs = physical_time_fs
        self._integrator_owned = True
        self._expected_positions = None if positions is None else positions.copy()
        self.results = {}


def _model_artifact(models_dir: Path, model_id: str) -> dict | None:
    path = Path(models_dir) / model_id.replace("/", "_") / "state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _check_resume_safety(events: list[dict], updater: object) -> None:
    """Refuse resume paths that cannot be honest before touching state."""
    for event in events:
        if (event.get("type") == TASK and event.get("operation") == "training"
                and event.get("status") == "failed"):
            raise ResumeError(
                "the run stopped on a failed model update; fork from the "
                "last valid checkpoint instead of resuming")
    consumed = [e for e in events
                if e.get("type") in (LABEL_CONSUMED, MODEL_UPDATE)]
    if consumed and not _is_stateful(updater):
        raise ResumeError(
            "the run consumed labels through an updater; supply the same "
            "stateful updater to resume")


def _replay_window(calc: EnergeticCalculator, store: Store, events: list[dict], *,
                   cursor: int, updater: object, models_dir: Path) -> dict | None:
    """Replay committed events after a checkpoint cursor in original order.

    Returns the last uncommitted proposal event (the frozen decision to
    resume with the same check draw), or None. Replayed commits apply their
    recorded values only — no re-sampling, no re-training, no re-consuming.
    ``events`` is the full log; the consumed-label set is built from the
    whole history, because a label first consumed before the checkpoint and
    only *reused* in the window must not be flagged as never consumed (F07).
    """
    consumed_anywhere: set[str] = set()
    for event in events:
        event_type = event.get("type")
        if event_type == LABEL_CONSUMED:
            consumed_anywhere.add(str(event["label_id"]))
        elif event_type == MODEL_UPDATE:
            consumed_anywhere.add(str(event["origin_label_id"]))
    tail_proposal = None
    unconsumed: set[str] = set()
    for event in events:
        if int(event.get("seq", 0)) <= cursor:
            continue
        event_type = event.get("type")
        if event_type == TASK:
            calc._task_counter = max(calc._task_counter,
                                     _id_suffix(event.get("task_id")))
            calc._label_counter = max(calc._label_counter,
                                      _id_suffix(event.get("label_id")))
        elif event_type == PROBE_COMPLETED:
            record = event["record"]
            calc._label_counter = max(calc._label_counter,
                                      _id_suffix(record.get("label_id")))
            try:
                h_index = list(calc.probe_steps).index(float(event["step_A"]))
            except ValueError:
                continue  # probe step from another policy: not reusable
            calc._probe_reuse[(int(event["evaluation_id"]),
                               int(event["direction"]), h_index,
                               int(event["sign"]),
                               int(event["model_generation"]))] = record
        elif event_type == EVALUATION_PROPOSED:
            tail_proposal = event
        elif event_type == EVALUATION_COMMITTED:
            tail_proposal = None
            row = store.committed_row(
                calc._event_log, calc.run_id,
                int(event["context"]["evaluation_id"]))
            calc._replay_committed(event, row)
            if updater is not None and event.get("label_id") is not None:
                unconsumed.add(str(event["label_id"]))
        elif event_type in (LABEL_CONSUMED, MODEL_UPDATE):
            calc._replay_label_event(event, store, updater, models_dir)
            unconsumed.discard(str(event.get("label_id")
                                       or event.get("origin_label_id")))
    # A label first consumed anywhere in the history (including before the
    # checkpoint) and only reused in the window is not a consumption loss.
    unconsumed -= consumed_anywhere
    if unconsumed:
        raise ResumeError(
            f"labels {sorted(unconsumed)} were committed but their consumption "
            "state was never persisted; automatic resume stops here as pending")
    return tail_proposal


class _EnergeticVerlet(VelocityVerlet):
    """Advance the evaluation clock even at an unchanged configuration.

    The integrator owns the physical clock: each scheduled evaluation gets
    its explicit physical time from the step count and the fixed timestep,
    keeping real integration time separate from the evaluation counter.
    """

    def step(self, forces=None):
        atoms = self.atoms
        if forces is None:
            forces = atoms.get_forces(md=True)
        next_positions = atoms.positions + self.dt * (
            atoms.get_momenta() + 0.5 * self.dt * forces
        ) / atoms.get_masses()[:, None]
        calc = atoms.calc
        if calc._projection is not None:
            # Schedule the constrained drift: FixAtoms displacements are
            # zeroed exactly as ASE's adjust_positions applies them below.
            next_positions = atoms.positions + calc._projection.project_displacement(
                next_positions - atoms.positions)
        calc._schedule(self.nsteps + 1, next_positions,
                       physical_time_fs=(self.nsteps + 1) * calc.timestep_fs)
        result = super().step(forces)
        # The full-step boundary is now complete: ASE promoted the half-step
        # momenta inside step().  Commit the step with the integrator
        # identity and the complete boundary bound (M1, S0b): the v2 digest
        # covers exactly the state resume later reconstructs and verifies.
        boundary = CommittedStepState(
            step=self.nsteps,
            physical_time_fs=(self.nsteps + 1) * calc.timestep_fs,
            positions=atoms.positions.copy(),
            momenta=atoms.get_momenta().copy(),
            driving_source=("surrogate" if calc._last_committed_route == "ml"
                            else "reference"),
            model_id=str(calc._last_committed_model_id),
            spec=calc._integrator_spec,
            nsteps=self.nsteps + 1)
        calc._emit_once(
            f"step:{calc.run_id}:{self.nsteps}", STEP_COMPLETED,
            step_id=self.nsteps,
            physical_time_fs=(self.nsteps + 1) * calc.timestep_fs,
            integrator=calc._integrator_spec.as_dict(),
            digest_format=DIGEST_FORMAT,
            state_digest=state_digest(atoms.positions,
                                      atoms.get_momenta()),
            boundary_digest=boundary.digest(),
            segment_id=calc._last_committed_segment,
            model_id=calc._last_committed_model_id)
        return result


class _EnergeticLangevin(Langevin):
    """ASE Langevin (fixcm=False) behind the complete-boundary contract.

    ASE finalizes the new configuration — both bath draws, the drift and
    FixAtoms — before requesting forces, so the schedule fixes only the
    deterministic clock (evaluation index, physical time) before the step,
    and the calculator adopts the actual positions at evaluation time:
    forecast, proposal record and commit all see exactly the configuration
    that propagates (M3A-2).  The bath draws happen once per step, inside
    ASE; no decision path (refusal, probe, check, re-anchor, logging or
    retry) ever draws again.
    """

    def step(self, forces=None):
        atoms = self.atoms
        calc = atoms.calc
        if forces is None:
            forces = atoms.get_forces(md=True)
        # Pin the pre-draw bath state of this step: a pending record rebuilt
        # after a crash restores it and ASE re-draws the identical
        # increments — never a fresh draw on resume (M3A-3).
        calc._bath_step = {"evaluation_id": self.nsteps + 1,
                           "rng_before": rng_state_to_json(
                               self.rng.bit_generator.state)}
        calc._schedule(self.nsteps + 1, None,
                       physical_time_fs=(self.nsteps + 1) * calc.timestep_fs)
        result = super().step(forces)
        # Langevin momenta are complete at the boundary (no half-kick
        # applies).  The step commit binds the full stochastic state:
        # positions, complete momenta, bath stream, driving label, anchor
        # segment and model identity (M3A-3).
        if not np.array_equal(atoms.positions, calc._expected_positions):
            raise RuntimeError(
                "the completed Langevin boundary differs from the "
                "configuration the force evaluation saw")
        boundary = CommittedStepState(
            step=self.nsteps,
            physical_time_fs=(self.nsteps + 1) * calc.timestep_fs,
            positions=atoms.positions.copy(),
            momenta=atoms.get_momenta().copy(),
            driving_source=("surrogate" if calc._last_committed_route == "ml"
                            else "reference"),
            model_id=str(calc._last_committed_model_id),
            spec=calc._integrator_spec,
            nsteps=self.nsteps + 1,
            thermostat_rng=dict(self.rng.bit_generator.state))
        calc._emit_once(
            f"step:{calc.run_id}:{self.nsteps}", STEP_COMPLETED,
            step_id=self.nsteps,
            physical_time_fs=(self.nsteps + 1) * calc.timestep_fs,
            integrator=calc._integrator_spec.as_dict(),
            digest_format=DIGEST_FORMAT,
            state_digest=state_digest(atoms.positions,
                                      atoms.get_momenta()),
            boundary_digest=boundary.digest(),
            thermostat_rng=boundary.thermostat_rng,
            segment_id=calc._last_committed_segment,
            model_id=calc._last_committed_model_id)
        return result

    def complete_momenta(self, boundary_positions: np.ndarray,
                         committed_positions: np.ndarray,
                         rnd_pos: np.ndarray, rnd_vel: np.ndarray,
                         driving_forces: np.ndarray,
                         projection: FixAtomsProjection | None) -> np.ndarray:
        """Boundary completion for resume via the shared primitive
        (:func:`pyraimd2.loop.integrators.complete_langevin_momenta` —
        ASE 3.29.0's post-force update applied to the committed step's
        recorded increments, never a recomputation); FixAtoms zeroes the
        fixed momenta exactly as ``set_momenta`` does."""
        momenta = complete_langevin_momenta(
            timestep_fs=self.dt / units.fs,
            friction_per_fs=self.fr * units.fs,
            masses=self.masses[:, 0],
            boundary_positions=boundary_positions,
            committed_positions=committed_positions,
            rnd_pos=rnd_pos, rnd_vel=rnd_vel,
            driving_forces=driving_forces)
        if projection is not None:
            momenta = projection.project_displacement(momenta)
        return momenta


class EnergeticRunner:
    """Fixed-cell NVE/NVT with energetic force-error prediction and reference checks.

    Existing momenta are preserved. If missing, a seeded thermal distribution
    at ``temperature_K`` initializes them once. Force-call costs count every
    successful anchor/probe/check separately. Calling ``run`` again continues
    this live instance. A failed MD step stops this runner because Verlet may
    have advanced to half-step momenta; no silent integrator retry is
    attempted — resume from the last complete-step checkpoint instead.

    With ``run_dir`` (requires ``event_log``) the runner writes complete-step
    checkpoints every ``checkpoint_interval_steps`` and on a stop request
    (:meth:`request_stop`, optionally installed as a SIGINT flag via
    ``handle_sigint``): positions and full-step momenta, real time, committed
    evaluation id, reusable driving force, anchor, check RNG bit state, model
    identity and updater state. :meth:`resume` restores the last valid
    checkpoint in a fresh process and replays the events after its cursor;
    :meth:`fork` starts a new run from a checkpoint with the model chain
    carried over. Resumable model updates require a stateful updater (state
    export/restore); plain callbacks are not resumable and are refused.
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
        force_metric: str = "active_dofs_max_atom",
        integrator_spec: IntegratorSpec | None = None,
        event_log: EventLog | None = None, label_cache: bool = True,
        run_dir: str | Path | None = None, checkpoint_interval_steps: int | None = None,
        handle_sigint: bool = False,
    ) -> None:
        temperature_K = _positive(temperature_K, "temperature_K", zero=True)
        if checkpoint_interval_steps is not None and (
                isinstance(checkpoint_interval_steps, bool)
                or not isinstance(checkpoint_interval_steps, (int, np.integer))
                or checkpoint_interval_steps < 1):
            raise ValueError("checkpoint_interval_steps must be a positive integer")
        if integrator_spec is None:
            integrator_spec = IntegratorSpec(
                algorithm="velocity_verlet", ensemble="nve",
                timestep_fs=float(timestep_fs))
        elif isinstance(integrator_spec, dict):
            # resume/fork restore the spec from the checkpoint's policy
            integrator_spec = IntegratorSpec(**dict(integrator_spec))
        if integrator_spec.timestep_fs != float(timestep_fs):
            raise ValueError("the integrator spec's timestep must match "
                             "timestep_fs")
        self.calc = EnergeticCalculator(
            surrogate, engine, store, run_id, force_budget=force_budget, timestep_fs=timestep_fs,
            probe_steps=probe_steps, numerical_floor=numerical_floor, time_cap_fs=time_cap_fs,
            transverse_cap=transverse_cap, check_probability=check_probability,
            check_seed=check_seed, failure_probability=failure_probability, tilt=tilt,
            direction=direction, on_label=on_label, event_log=event_log,
            label_cache=label_cache, force_metric=force_metric,
            integrator_spec=integrator_spec, velocity_seed=velocity_seed,
        )
        self.calc._validate_atoms(atoms)
        if "momenta" not in atoms.arrays:
            # NVT derives the initialization stream by role so it never
            # shares a generator with the bath (same convention as the
            # plain driver); NVE keeps the historical raw seed.
            init_seed = (derive_stream_seed(velocity_seed, "velocity")
                         if integrator_spec.ensemble == "nvt" else velocity_seed)
            thermalize_momenta(atoms, temperature_K, rng=np.random.default_rng(init_seed))
        self.atoms = atoms
        self.timestep_fs = self.calc.timestep_fs
        atoms.calc = self.calc
        self.calc._schedule(0)
        if integrator_spec.algorithm == "langevin":
            self.dyn = _EnergeticLangevin(
                atoms, self.timestep_fs * units.fs,
                temperature_K=integrator_spec.temperature_K,
                friction=integrator_spec.friction_per_fs / units.fs,
                fixcm=False,
                rng=np.random.default_rng(integrator_spec.thermostat_seed))
            self.calc._bath_dyn = self.dyn
        else:
            self.dyn = _EnergeticVerlet(atoms, self.timestep_fs * units.fs)
        self._failed = False
        self._stop_requested = False
        self.run_dir = None if run_dir is None else Path(run_dir)
        # The caller's store outlives the runner; a store created by
        # resume/fork is owned (and closed) by the runner (R4).
        self._owns_store = False
        self.checkpoint_interval_steps = (None if checkpoint_interval_steps is None
                                          else int(checkpoint_interval_steps))
        self._checkpoints: CheckpointManager | None = None
        self._model_registry: ModelRegistry | None = None
        if self.run_dir is not None:
            if event_log is None:
                raise ValueError("checkpointing requires an event log")
            self._checkpoints = CheckpointManager(self.run_dir)
            self._model_registry = ModelRegistry(self.run_dir)
            self.calc._model_publisher = self._publish_model_artifact
            self.dyn.attach(self._maybe_checkpoint, interval=1)
        if handle_sigint:
            self._install_sigint_handler()

    def _install_sigint_handler(self) -> None:
        """SIGINT only sets the stop flag; the checkpoint is written by the
        normal control flow at the next complete-step boundary."""
        import signal

        signal.signal(signal.SIGINT, lambda signum, frame: self.request_stop())

    def request_stop(self) -> None:
        """Ask the run to checkpoint and stop at the next complete step."""
        self._stop_requested = True

    def close(self) -> None:
        """Release the event-log writer lock (a deliberate end of writing)."""
        if self.calc._event_log is not None:
            self.calc._event_log.close()
        if self._owns_store:
            self.calc.store.close()

    def _publish_model_artifact(self, model_id: str, record: dict) -> dict:
        """Immutable model artifact, persisted before the update event;
        returns the stored payload (the placeholder form the commit digest
        binds)."""
        return self._model_registry.publish(model_id, record)

    def _write_checkpoint(self) -> int | None:
        if self._checkpoints is None:
            return None
        generation = self._checkpoints.next_generation()
        state, arrays = self.calc._checkpoint_payload(self.atoms)
        model_artifact_digest = None
        if self._model_registry is not None:
            artifact = self._model_registry.read(self.calc.model_id)
            if artifact is not None:
                model_artifact_digest = artifact_digest(artifact)
        self._checkpoints.write(generation, state, arrays, {
            "run_id": self.calc.run_id,
            "nsteps": self.dyn.nsteps,
            "physical_time_fs": self.dyn.nsteps * self.timestep_fs,
            "last_event_seq": (self.calc._event_log.last_seq
                               if self.calc._event_log is not None else 0),
            "store_schema_version": STORE_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "software_version": __version__,
            "model_artifact_digest": model_artifact_digest,
        })
        return generation

    def _maybe_checkpoint(self) -> None:
        if self._checkpoints is None:
            return
        if self._stop_requested:
            self._write_checkpoint()
            raise StopRequested
        if (self.checkpoint_interval_steps
                and self.dyn.nsteps % self.checkpoint_interval_steps == 0):
            self._write_checkpoint()

    def run(self, n_steps: int) -> EnergeticRunSummary:
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer")
        if self._failed:
            raise RuntimeError("an energetic MD step failed; start a new run from a deliberate state")
        before = (self.calc.n_evaluations, self.calc.n_accepted, self.calc.n_violations,
                  self.calc.n_calibrations, self.calc.reference_calls.copy())
        start = time.perf_counter()
        stopped = False
        try:
            self.dyn.run(int(n_steps))
        except StopRequested:
            stopped = True
            self._stop_requested = False
            self.calc._emit(RUN_END, run_id=self.calc.run_id, status="stopped",
                            reason="stop requested; checkpoint saved at the last "
                                   "complete step")
        except Exception as error:
            self._failed = True
            self.calc._emit(RUN_END, run_id=self.calc.run_id, status="failed",
                            reason=repr(error))
            raise
        n_evaluations = self.calc.n_evaluations - before[0]
        n_accepted = self.calc.n_accepted - before[1]
        calls = {key: count - before[4][key] for key, count in self.calc.reference_calls.items()}
        summary = EnergeticRunSummary(
            int(n_steps), n_evaluations, n_accepted, sum(calls.values()), calls["anchor"],
            calls["probe"], calls["check"], self.calc.n_violations - before[2],
            self.calc.n_calibrations - before[3],
            n_accepted / n_evaluations if n_evaluations else 0.0,
            None if self.calc.verification is None else self.calc.verification.as_dict(),
            time.perf_counter() - start,
        )
        # The outer wall time is measured directly here — never re-summed
        # from nested task timings downstream.
        self.calc._emit(RUN_SUMMARY, run_id=self.calc.run_id,
                        n_steps=summary.n_steps, n_evaluations=summary.n_evaluations,
                        n_accepted=summary.n_accepted, n_reference=summary.n_reference,
                        wall_time_s=summary.wall_time_s, stopped_early=stopped)
        return summary

    @staticmethod
    def _read_resume_checkpoint(run_dir: Path) -> tuple[dict, dict, dict, int]:
        checkpoint = CheckpointManager(run_dir).read_latest_valid()
        if checkpoint is None:
            raise ResumeError(f"no valid checkpoint under {run_dir}")
        return (checkpoint.state, checkpoint.arrays, checkpoint.manifest,
                checkpoint.generation)

    @classmethod
    def resume(cls, run_dir: str | Path, surrogate: Surrogate, engine: Engine, *,
               updater: StatefulUpdater | None = None,
               direction: Direction | None = None,
               checkpoint_interval_steps: int | None = None,
               handle_sigint: bool = False, event_log_force: bool = False,
               label_cache: bool = True,
               event_log: EventLog | None = None) -> EnergeticRunner:
        """Resume a run from its last valid checkpoint plus event replay.

        Restores the complete-step boundary in a fresh process, replays the
        committed events after the checkpoint cursor (no re-sampling,
        re-training or re-consuming), and resumes an uncommitted frozen
        proposal with its original check draw. ``event_log_force`` reclaims
        the writer lock left by a crashed process — a deliberate assertion
        that no live writer exists.  A caller may instead pass an already
        open ``event_log`` (the workflow resume path does, so backends
        created with that same log keep recording after the restart); the
        runner then takes over its lifecycle.
        """
        run_dir = Path(run_dir)
        state, arrays, manifest, generation = cls._read_resume_checkpoint(run_dir)
        run_id = state["run_id"]
        if fingerprint_of(engine) != state["engine_fingerprint"]:
            raise ResumeError(
                "reference settings identity does not match the checkpoint; "
                "resume requires the same physical settings — use fork to change them")
        if model_id_for(surrogate, state["model_generation"]) != state["model_id"]:
            raise ResumeError(
                "surrogate identity does not match the checkpoint's model chain; "
                "resume requires the same model lineage")
        if state["updater_state"] is not None and not _is_stateful(updater):
            raise ResumeError(
                "the checkpoint references an updater state; supply the same "
                "stateful updater to resume")
        checkpoint_digest = manifest.get("model_artifact_digest")
        if checkpoint_digest is not None:
            artifact = _model_artifact(run_dir / "models", state["model_id"])
            if artifact is None:
                raise ResumeError(
                    f"the checkpoint names a model artifact for "
                    f"{state['model_id']!r} that is missing or unreadable")
            if artifact_digest(artifact) != checkpoint_digest:
                raise ResumeError(
                    f"model artifact for {state['model_id']!r} does not match "
                    "the digest bound into the checkpoint; it looks tampered "
                    "with — resume refuses to load it")
        store = Store(run_dir / "trajectory.db")
        if event_log is None:
            event_log = EventLog(run_dir, force=event_log_force)
        policy = dict(state["policy"])
        calc = EnergeticCalculator(
            surrogate, engine, store, run_id, direction=direction,
            on_label=updater, event_log=event_log, label_cache=label_cache,
            _resume_state={"state": state, "arrays": arrays}, **policy)
        events = list(event_log.iter_events())
        _check_resume_safety(events, updater)
        cursor = int(manifest["last_event_seq"])
        if _is_stateful(updater) and state["updater_state"] is not None:
            # Placeholders resolve against the checkpoint's own arrays.npz
            # (tensor-artifact-v1), digest-verified.
            updater.load_state_dict(load_state_arrays(
                state["updater_state"], dict_array_source(arrays)))
        tail = _replay_window(calc, store, events, cursor=cursor,
                              updater=updater, models_dir=run_dir / "models")
        # Rebuild the numeric label cache from durable records: rows carrying
        # an engine payload and a durable label ID fully determine (geometry,
        # reference identity, energy kind) -> label ID. Without this, a cold
        # cache re-executes the same reference and assigns a NEW label ID,
        # silently re-consuming and re-training (F02).
        calc._rebuild_label_cache(store)
        runner = cls.__new__(cls)
        runner.calc = calc
        runner.run_dir = run_dir
        runner._owns_store = True  # resume created the store; runner closes it
        runner.checkpoint_interval_steps = (None if checkpoint_interval_steps is None
                                            else int(checkpoint_interval_steps))
        runner._stop_requested = False
        runner._failed = False
        runner._checkpoints = CheckpointManager(run_dir)
        runner._model_registry = ModelRegistry(run_dir)
        calc._model_publisher = runner._publish_model_artifact
        # Boundary atoms: full-step momenta of the last committed evaluation.
        atoms = calc._identity_from_arrays(arrays)
        if calc._projection is not None:
            from ase.constraints import FixAtoms

            atoms.set_constraint(FixAtoms(indices=list(calc._projection.indices)))
        atoms.positions = np.array(arrays["positions"], dtype=float)
        atoms.set_momenta(np.array(arrays["momenta"], dtype=float))
        timestep_ase = calc.timestep_fs * units.fs
        last_eval = calc.n_evaluations - 1
        spec = calc._integrator_spec
        driving_energy = state["driving_energy_eV"]
        driving_forces = (np.array(arrays["driving_forces"], dtype=float)
                          if driving_energy is not None else None)
        boundary_row = None
        boundary_commit = None
        if last_eval >= 1:
            boundary_row = store.committed_row(event_log, run_id, last_eval)
            atoms.positions = boundary_row.toatoms().positions
            # Positions, driving forces and the boundary momenta all come
            # from the same verified committed row — an orphan row at the
            # same step is never read (C1).
            driving_energy, driving_forces = store.driving_label_for_row(
                boundary_row)
            boundary_commit = next(
                (e for e in reversed(events)
                 if e.get("type") == EVALUATION_COMMITTED
                 and int((e.get("context") or {})
                         .get("evaluation_id", -1)) == last_eval), None)
            if spec.algorithm == "velocity_verlet":
                # ASE's velocity-Verlet kick adds 0.5*dt*F to the momenta (no
                # mass division — momenta, not velocities); the drift divides.
                full_step = (boundary_row.toatoms().get_momenta()
                             + 0.5 * timestep_ase * driving_forces)
                atoms.set_momenta(full_step)
            # The Langevin boundary momenta are completed below, once the
            # dynamics (and with it c1/c2 and the masses layout) exists.
        runner.atoms = atoms
        runner.timestep_fs = calc.timestep_fs
        if spec.algorithm == "langevin":
            # Restore the bath stream — checkpoint state first, then the
            # newest post-cursor commit's pinned state, then the pending
            # proposal's pre-draw state — so the resume never re-seeds and
            # never re-draws a committed step (M3A-3/M3A-4).
            thermostat = (state.get("thermostat") or {}).get("rng")
            if thermostat is None:
                raise ResumeError(
                    "the checkpoint of this adaptive NVT run carries no "
                    "thermostat stream; the run directory predates the "
                    "complete-state protocol — fork instead of resume")
            for event in events:
                if (int(event.get("seq", 0)) > cursor
                        and event.get("type") == EVALUATION_COMMITTED
                        and event.get("thermostat_rng") is not None):
                    thermostat = event["thermostat_rng"]
            if tail is not None and tail.get("bath_step") is not None:
                thermostat = tail["bath_step"]["rng_before"]
            bath_rng = np.random.default_rng()
            bath_rng.bit_generator.state = thermostat
            runner.dyn = _EnergeticLangevin(
                atoms, timestep_ase, temperature_K=spec.temperature_K,
                friction=spec.friction_per_fs / units.fs, fixcm=False,
                rng=bath_rng)
            calc._bath_dyn = runner.dyn
            if last_eval >= 1:
                bath = (boundary_commit or {}).get("bath_step")
                if bath is None:
                    raise ResumeError(
                        f"the committed evaluation {last_eval} has no "
                        "recorded bath increments; the stochastic boundary "
                        "cannot be reconstructed exactly — the run directory "
                        "predates the complete-state protocol")
                previous = store.committed_row(
                    event_log, run_id, last_eval - 1).toatoms().positions
                atoms.set_momenta(runner.dyn.complete_momenta(
                    previous, atoms.positions,
                    np.array(bath["rnd_pos"], dtype=float),
                    np.array(bath["rnd_vel"], dtype=float),
                    driving_forces, calc._projection))
        else:
            runner.dyn = _EnergeticVerlet(atoms, runner.timestep_fs * units.fs)
        atoms.calc = calc
        calc.atoms = atoms.copy()
        calc._last_committed_positions = atoms.positions.copy()
        # The step record of the first post-resume step binds the identity
        # of the newest commit — replay restored the counters, and these
        # fields come from the same authoritative event.
        newest_commit = next(
            (e for e in reversed(events)
             if e.get("type") == EVALUATION_COMMITTED), None)
        if newest_commit is not None:
            calc._last_committed_route = newest_commit.get("route")
            calc._last_committed_segment = newest_commit.get("segment_id")
            calc._last_committed_model_id = newest_commit.get("model_id")
        runner.dyn.nsteps = max(last_eval, 0)
        if last_eval >= 1 and boundary_commit is not None:
            # The boundary's step record: verify it when present in the
            # versioned format (S0b/M3A-3); heal it when the crash window
            # left a committed evaluation without its step commit (the same
            # window the plain driver heals — the commit plus the recorded
            # bath increments prove the step completed, so resume re-emits
            # the record idempotently instead of leaving a gap the
            # committed-frames view would silently drop).  Older adaptive
            # records predate the format marker; their binding is the
            # commit's row digest, already verified by committed_row.
            step_event = next(
                (e for e in reversed(events)
                 if e.get("type") == STEP_COMPLETED
                 and int(e.get("step_id", -1)) == last_eval - 1), None)
            if step_event is None and spec.algorithm == "langevin":
                boundary = CommittedStepState(
                    step=last_eval - 1,
                    physical_time_fs=last_eval * calc.timestep_fs,
                    positions=atoms.positions.copy(),
                    momenta=atoms.get_momenta().copy(),
                    driving_source=("surrogate"
                                    if boundary_commit.get("route") == "ml"
                                    else "reference"),
                    model_id=str(boundary_commit.get("model_id", "")),
                    spec=spec, nsteps=last_eval,
                    thermostat_rng=boundary_commit.get("thermostat_rng"))
                calc._emit_once(
                    f"step:{run_id}:{last_eval - 1}", STEP_COMPLETED,
                    step_id=last_eval - 1,
                    physical_time_fs=last_eval * calc.timestep_fs,
                    integrator=spec.as_dict(),
                    digest_format=DIGEST_FORMAT,
                    state_digest=state_digest(atoms.positions,
                                              atoms.get_momenta()),
                    boundary_digest=boundary.digest(),
                    thermostat_rng=boundary_commit.get("thermostat_rng"),
                    segment_id=boundary_commit.get("segment_id"),
                    model_id=boundary_commit.get("model_id"))
            elif (step_event is not None
                    and step_event.get("digest_format") == DIGEST_FORMAT):
                boundary = CommittedStepState(
                    step=last_eval - 1,
                    physical_time_fs=last_eval * calc.timestep_fs,
                    positions=atoms.positions.copy(),
                    momenta=atoms.get_momenta().copy(),
                    driving_source=("surrogate"
                                    if boundary_commit.get("route") == "ml"
                                    else "reference"),
                    model_id=str(boundary_commit.get("model_id", "")),
                    spec=spec, nsteps=last_eval,
                    thermostat_rng=boundary_commit.get("thermostat_rng"))
                problems = []
                if state_digest(atoms.positions, atoms.get_momenta()) \
                        != step_event.get("state_digest"):
                    problems.append("state digest")
                if boundary.digest() != step_event.get("boundary_digest"):
                    problems.append("boundary digest")
                if step_event.get("segment_id") != boundary_commit.get(
                        "segment_id"):
                    problems.append("anchor segment")
                if step_event.get("model_id") != boundary_commit.get(
                        "model_id"):
                    problems.append("model identity")
                if problems:
                    raise ResumeError(
                        f"the step-{last_eval - 1} record does not match the "
                        f"reconstructed boundary ({'; '.join(problems)}); "
                        "the run directory is inconsistent")
        if tail is not None:
            eval_id = int(tail["context"]["evaluation_id"])
            if eval_id != calc.n_evaluations:
                raise ResumeError(
                    f"uncommitted proposal {eval_id} does not follow the "
                    f"replayed state {calc.n_evaluations}")
            positions = np.array(tail["positions_A"], dtype=float)
            pending_atoms = atoms.copy()
            pending_atoms.positions = positions.copy()
            if spec.algorithm == "velocity_verlet":
                # The integrator redoes exactly one first half-kick and
                # drift, landing on the persisted positions bit-identically.
                half_step = (atoms.get_momenta()
                             + 0.5 * timestep_ase * driving_forces)
                pending_atoms.set_momenta(half_step)
            # Langevin: the atoms still carry the boundary momenta at the
            # force evaluation (ASE updates momenta only at step end), and
            # the restored bath stream re-draws the identical increments —
            # the step itself lands on the persisted positions.
            calc._pending = calc._rebuild_pending(tail, pending_atoms)
            calc._schedule(eval_id, positions,
                           physical_time_fs=eval_id * calc.timestep_fs)
        else:
            calc._schedule(calc.n_evaluations, None,
                           physical_time_fs=calc.n_evaluations * calc.timestep_fs)
        # The schedule cleared the ASE cache; restore the reusable driving
        # force so the resumed integrator's first half-kick does not
        # recalculate the boundary evaluation.
        if driving_energy is not None:
            calc.results = {"energy": float(driving_energy),
                            "forces": driving_forces.copy()}
            calc._results_model_generation = calc._model_generation
        runner.dyn.attach(runner._maybe_checkpoint, interval=1)
        if handle_sigint:
            runner._install_sigint_handler()
        calc._emit(RESUMED, run_id=run_id, from_event_seq=cursor,
                   checkpoint_generation=generation)
        return runner

    @classmethod
    def fork(cls, run_dir: str | Path, new_run_dir: str | Path, new_run_id: str,
             surrogate: Surrogate, engine: Engine, *,
             updater: StatefulUpdater | None = None,
             direction: Direction | None = None,
             policy_overrides: dict | None = None,
             checkpoint_interval_steps: int | None = None,
             handle_sigint: bool = False, label_cache: bool = True) -> EnergeticRunner:
        """Start a new run from the parent's last valid checkpoint.

        The physical state (positions, full-step momenta) and the model chain
        (surrogate lineage, updater state) carry over; policy parameters may
        change (a new check segment starts, recorded as such). The parent run
        is only read, never written.
        """
        run_dir = Path(run_dir)
        new_run_dir = Path(new_run_dir)
        if new_run_dir.resolve() == run_dir.resolve():
            raise ValueError("fork requires a new run directory distinct from the parent")
        state, arrays, _, generation = cls._read_resume_checkpoint(run_dir)
        if fingerprint_of(engine) != state["engine_fingerprint"]:
            raise ResumeError(
                "fork keeps the same reference backend identity; change "
                "reference settings in a fresh run instead")
        if model_id_for(surrogate, state["model_generation"]) != state["model_id"]:
            raise ResumeError(
                "fork keeps the same surrogate lineage; the checkpoint's model "
                "chain does not match the supplied surrogate")
        if state["updater_state"] is not None:
            if not _is_stateful(updater):
                raise ResumeError(
                    "the checkpoint references an updater state; supply the "
                    "same stateful updater to fork")
            updater.load_state_dict(load_state_arrays(
                state["updater_state"], dict_array_source(arrays)))
        policy = dict(state["policy"])
        policy.update(policy_overrides or {})
        atoms = Atoms(numbers=np.array(arrays["numbers"]),
                      positions=np.array(arrays["positions"], dtype=float),
                      cell=np.array(arrays["cell"]), pbc=np.array(arrays["pbc"]))
        atoms.set_masses(np.array(arrays["masses"]))
        atoms.set_initial_charges(np.array(arrays["initial_charges"]))
        constraint = state.get("constraint")
        if constraint is not None:
            from ase.constraints import FixAtoms

            atoms.set_constraint(FixAtoms(indices=list(constraint["indices"])))
        atoms.set_initial_magnetic_moments(np.array(arrays["initial_magmoms"]))
        atoms.set_momenta(np.array(arrays["momenta"], dtype=float))
        event_log = EventLog(new_run_dir)
        runner = cls(atoms, surrogate, engine, Store(new_run_dir / "trajectory.db"),
                     new_run_id, direction=direction, on_label=updater,
                     event_log=event_log, label_cache=label_cache,
                     run_dir=new_run_dir,
                     checkpoint_interval_steps=checkpoint_interval_steps,
                     handle_sigint=handle_sigint, **policy)
        runner._owns_store = True  # fork created the store; runner closes it
        runner.calc._emit("forked_from", parent_run_id=state["run_id"],
                          parent_run_dir=str(run_dir),
                          checkpoint_generation=generation,
                          parent_model_id=state["model_id"])
        return runner
