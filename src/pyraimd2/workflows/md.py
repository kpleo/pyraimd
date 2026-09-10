"""MD workflow orchestration: one implementation for the Python API and the CLI.

Three modes over the same run-directory layout:

- ``adaptive``: the energetic runner — anchored surrogate forces,
  independent reference checks, complete-step checkpoints, resumable.
- ``reference``: plain velocity-Verlet NVE driven by the reference engine.
- ``surrogate``: plain velocity-Verlet NVE driven by the frozen surrogate.

The two plain modes share one small driver built on ASE's VelocityVerlet and
write the same store-row and event shapes the energetic runner writes, so
``inspect`` and ``export`` work uniformly. Plain runs resume from their
complete-step checkpoints the same way adaptive runs do.

``singlepoint`` and ``relax`` are fixed-model tasks in the same layout: one
backend evaluation, or a FIRE/BFGS optimization driven by a fixed backend
(reference or frozen surrogate) — never the time-gated adaptive calculator.
"""

from __future__ import annotations

import dataclasses
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import thermalize_momenta

from pyraimd2 import __version__
from pyraimd2.config import PyramidConfig, load_resolved_config
from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.loop.constraints import validate_constraints
from pyraimd2.loop.energetic import EnergeticRunSummary
from pyraimd2.loop.integrators import (
    DIGEST_FORMAT,
    STREAM_SCHEME,
    CommittedStepState,
    IntegratorSpec,
    LangevinAdapter,
    VelocityVerletAdapter,
    derive_stream_seed,
    state_digest,
)
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.runtime.context import EvaluationContext, EvaluationPhase
from pyraimd2.runtime.events import (
    ATTEMPT_LEDGER_PHYSICAL_V1,
    EVALUATION_COMMITTED,
    EVENT_SCHEMA_VERSION,
    RESUMED,
    RUN_END,
    RUN_START,
    RUN_SUMMARY,
    STEP_COMPLETED,
    TASK,
    EventLog,
    physical_attempt,
)
from pyraimd2.runtime.identity import fingerprint_of, model_id_for
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.store.store import STORE_SCHEMA_VERSION
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.workflows.setup import (
    RunOutputs,
    WorkflowError,
    build_backends,
    check_run_directory_available,
    create_configured_backend,
    load_structure,
    prepare_run_directory,
    validate_setup,
)


@dataclass
class WorkflowResult:
    """Outcome of one run/resume call."""

    run_dir: Path
    run_id: str
    mode: str
    steps_completed: int  # total complete steps in the run now
    steps_this_call: int
    stopped_early: bool
    summary: EnergeticRunSummary | None


# ---------------------------------------------------------------------------
# shared helpers


class _SigintGuard:
    """Save/restore the previous SIGINT handler around a run whose runner
    installs its own (a flag-only handler; checkpoints are written by normal
    control flow, WP03).  Enter the guard *before* constructing the runner,
    or the captured "previous" handler is the runner's own."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.previous: Any = None

    def __enter__(self) -> None:
        if self.enabled:
            self.previous = signal.getsignal(signal.SIGINT)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.enabled and self.previous is not None:
            signal.signal(signal.SIGINT, self.previous)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _finalize_quietly(outputs: RunOutputs) -> None:
    """Derived outputs must never mask the run's own outcome; they can be
    regenerated from trajectory.db/events.jsonl at any time."""
    try:
        outputs.finalize()
    except Exception as error:  # noqa: BLE001 - previews must not mask the run
        _warn(f"final output refresh failed ({error}); summary.json/csv and "
              "trajectory.extxyz are derived artifacts — regenerate with "
              "`pyramid inspect`/`pyramid export`")


def _attach_outputs(runner: EnergeticRunner, outputs: RunOutputs) -> None:
    """Refresh derived outputs at step boundaries; previews never crash a run."""

    def _trajectory() -> None:
        try:
            outputs.append_trajectory_step(runner.dyn.nsteps)
        except Exception as error:  # noqa: BLE001 - previews never crash a run
            _warn(f"trajectory preview refresh failed ({error}); the "
                  "authoritative trajectory.db is unaffected")

    def _summaries() -> None:
        try:
            outputs.write_summaries()
        except Exception as error:  # noqa: BLE001 - previews never crash a run
            _warn(f"summary refresh failed ({error}); the final write at run "
                  "end regenerates them")

    runner.dyn.attach(_trajectory, interval=1)
    runner.dyn.attach(_summaries, interval=outputs.summary_interval)


def _remove_fresh_event_log(run_dir: Path) -> None:
    """Undo a just-created, still-empty event log after a setup failure, so
    fixing the configuration and re-running is not blocked by our own
    leftover files.  A log with content is a real run record and stays."""
    path = run_dir / "events.jsonl"
    if path.is_file() and path.stat().st_size == 0:
        path.unlink()
    lock = run_dir / "events.jsonl.lock"
    if lock.exists():
        lock.unlink()


def _complete_steps(run_dir: Path, run_id: str) -> int:
    return int(inspect_run(run_dir, run_id=run_id)["n_complete_steps"] or 0)


def _stopped_early(run_dir: Path, run_id: str) -> bool:
    info = inspect_run(run_dir, run_id=run_id)
    return bool(info["failure"] and info["failure"].get("status") == "stopped")


# ---------------------------------------------------------------------------
# adaptive mode


def _policy_kwargs(config: PyramidConfig) -> dict:
    policy = config.policy
    verification = config.verification
    return {
        "force_budget": policy.force_budget_eV_A,
        "timestep_fs": config.dynamics.timestep_fs,
        "probe_steps": tuple(policy.probe_steps_A),
        "numerical_floor": policy.numerical_floor_eV_A,
        "time_cap_fs": policy.time_cap_fs,
        "transverse_cap": policy.transverse_cap,
        "force_metric": policy.force_metric,
        "check_probability": verification.probability,
        "check_seed": verification.seed,
        "failure_probability": verification.failure_probability,
        "tilt": verification.tilt,
    }


def _run_adaptive(config: PyramidConfig, atoms: Atoms, run_dir: Path, *,
                  verbose: bool, handle_sigint: bool) -> WorkflowResult:
    event_log = EventLog(run_dir)
    with _SigintGuard(handle_sigint):
        try:
            # The reference is created with the run's event log when its
            # factory accepts one (QE density I/O then enters the ledger).
            store = None
            engine = create_configured_backend("reference", config.reference,
                                               run_dir=run_dir,
                                               event_log=event_log)
            surrogate = create_configured_backend("surrogate", config.surrogate,
                                                  run_dir=run_dir)
            store = Store(run_dir / "trajectory.db")
            prepare_run_directory(config, engine=engine, surrogate=surrogate)
            runner = EnergeticRunner(
                atoms, surrogate, engine, store, config.run.id,
                temperature_K=config.dynamics.temperature_K,
                velocity_seed=config.dynamics.velocity_seed,
                integrator_spec=_spec_from_dynamics(config),
                event_log=event_log, run_dir=run_dir,
                checkpoint_interval_steps=config.checkpoint.interval_steps,
                handle_sigint=handle_sigint,
                **_policy_kwargs(config))
            outputs = RunOutputs(
                run_dir, config.run.id,
                trajectory_interval_steps=config.output.trajectory_interval_steps,
                summary_interval_steps=config.output.summary_interval_steps)
        except Exception:
            if store is not None:
                store.close()
            event_log.close()
            _remove_fresh_event_log(run_dir)
            raise

        _attach_outputs(runner, outputs)
        total = config.dynamics.steps
        if verbose:
            print(f"pyramid run: {config.run.id} — adaptive "
                  f"{config.dynamics.ensemble.upper()}, "
                  f"{total} steps x {config.dynamics.timestep_fs} fs")
            print(f"run directory: {run_dir}")

        def _progress() -> None:
            if (verbose and runner.dyn.nsteps > 0
                    and runner.dyn.nsteps % outputs.summary_interval == 0):
                print(f"  step {runner.dyn.nsteps}/{total} "
                      f"(t = {runner.dyn.nsteps * config.dynamics.timestep_fs:.2f} fs)",
                      flush=True)

        runner.dyn.attach(_progress, interval=outputs.summary_interval)
        try:
            summary = runner.run(total)
        finally:
            _finalize_quietly(outputs)
            runner.close()
            # The workflow created this store and the runner does not own
            # caller-passed stores — close our own (S0a).
            store.close()
    after = _complete_steps(run_dir, config.run.id)
    stopped = _stopped_early(run_dir, config.run.id)
    if verbose:
        status = "stopped early (checkpoint saved)" if stopped else "completed"
        print(f"run {status}: {after} complete steps")
        _print_run_report(summary, run_dir)
    return WorkflowResult(run_dir=run_dir, run_id=config.run.id,
                          mode="adaptive", steps_completed=after,
                          steps_this_call=after, stopped_early=stopped,
                          summary=summary)


def _print_run_report(summary: EnergeticRunSummary, run_dir: Path) -> None:
    print(f"  evaluations {summary.n_evaluations}, accepted "
          f"{summary.n_accepted} ({summary.accepted_fraction:.0%}), "
          f"violations {summary.n_violations}")
    print(f"  reference calls {summary.n_reference} "
          f"(anchor {summary.n_anchor}, probe {summary.n_probe}, "
          f"check {summary.n_checks}); wall time {summary.wall_time_s:.2f} s")
    print(f"  outputs: {run_dir}/summary.json, summary.csv, trajectory.extxyz")


# ---------------------------------------------------------------------------
# plain reference/surrogate modes


class _BackendCalculator(Calculator):
    """Thin ASE Calculator over an engine/surrogate for plain NVE."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, backend: object, section: str, *,
                 on_evaluation: Any = None,
                 event_log: EventLog | None = None,
                 attempt_fields: Any = None) -> None:
        super().__init__()
        self._backend = backend
        self._section = section
        self._on_evaluation = on_evaluation
        self._event_log = event_log
        # Callable returning the current logical task's attempt context
        # fields (request_id/purpose), or None outside a logical request.
        self._attempt_fields = attempt_fields
        self.n_calculations = 0
        self.last_label: Any = None

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.n_calculations += 1
        fields = (self._attempt_fields() if self._attempt_fields is not None
                  else None)
        method = "compute" if self._section == "reference" else "predict"
        if fields is None or self._event_log is None:
            if self._section == "reference":
                label = self._backend.compute(self.atoms)
            else:
                label = self._backend.predict(self.atoms)
        else:
            with physical_attempt(self._backend, self._event_log,
                                  operation=("reference"
                                             if self._section == "reference"
                                             else "inference"),
                                  source="workflow", method=method,
                                  **fields) as attempt_kwargs:
                if self._section == "reference":
                    label = self._backend.compute(self.atoms,
                                                  **attempt_kwargs)
                else:
                    label = self._backend.predict(self.atoms,
                                                  **attempt_kwargs)
        forces = np.asarray(label.forces, dtype=float)
        if not np.isfinite(label.energy) or not np.isfinite(forces).all():
            raise EngineError(
                f"{self._section} backend returned non-finite energy/forces")
        self.last_label = label
        self.results = {"energy": float(label.energy), "forces": forces}
        if self._on_evaluation is not None:
            self._on_evaluation(label)


def _spec_from_dynamics(config: PyramidConfig) -> IntegratorSpec:
    """The run's integrator identity from its dynamics configuration.

    A new NVT run draws its thermostat stream from the role-derived seed
    (``role-derive-v1``: the user seed plus the fixed thermostat role
    code, so the bath never shares a generator with velocity
    initialization — not even when both fields default to ``run.seed``).
    Resumes never use seeds; they restore the persisted stream.
    """
    dynamics = config.dynamics
    if dynamics.integrator == "langevin":
        raw_seed = (dynamics.thermostat_seed
                    if dynamics.thermostat_seed is not None
                    else config.run.seed)
        return IntegratorSpec(
            algorithm="langevin", ensemble="nvt",
            timestep_fs=dynamics.timestep_fs,
            temperature_K=dynamics.temperature_K,
            friction_per_fs=dynamics.friction_per_fs,
            thermostat_seed=derive_stream_seed(raw_seed, "thermostat"))
    return IntegratorSpec(algorithm="velocity_verlet", ensemble="nve",
                          timestep_fs=dynamics.timestep_fs)


def _make_integrator(spec: IntegratorSpec, atoms: Atoms,
                     resume_state: dict | None):
    """Create the adapter, restoring the thermostat stream on NVT resume."""
    if spec.algorithm == "langevin":
        thermostat = (None if resume_state is None
                      else (resume_state.get("thermostat") or {}).get("rng"))
        return LangevinAdapter(atoms, spec, thermostat_rng_state=thermostat)
    return VelocityVerletAdapter(atoms, spec)


def _boundary_from_event(row: object, step_event: dict,
                         commit: dict | None) -> CommittedStepState:
    """Rebuild the boundary state a persisted step event claims to bind.

    Reconstruction sources the authoritative committed row for the arrays,
    the row's route for the driving source, and the evaluation commit for
    the model identity; the step event itself contributes the step number,
    physical time, integrator block and thermostat stream.  The digest of
    this state is what both the normal step commit and the crash-healing
    commit write, and what resume verifies (S0b).
    """
    atoms = row.toatoms()
    context = (commit or {}).get("context") or {}
    route = row.key_value_pairs.get("route")
    return CommittedStepState(
        step=int(step_event["step_id"]),
        physical_time_fs=float(step_event["physical_time_fs"]),
        positions=np.asarray(atoms.positions, dtype=float),
        momenta=np.asarray(atoms.get_momenta(), dtype=float),
        driving_source=("reference" if route == "dft" else "surrogate"),
        model_id=str(context.get("model_id", "")),
        spec=IntegratorSpec(**dict(step_event["integrator"])),
        nsteps=int(step_event["step_id"]) + 1,
        thermostat_rng=step_event.get("thermostat_rng"))


def _label_from_row(row: object, section: str) -> object | None:
    """Rebuild the boundary label object from a committed store row.

    A resumed run's first step can legitimately hit ASE's calculator cache
    (zero displacement at a stationary boundary), leaving ``last_label``
    unset; the committed boundary row carries the same payload a continuous
    run would have recorded.
    """
    key = "engine" if section == "reference" else "surrogate"
    payload = row.data.get(key)
    if payload is None:
        return None
    stress = payload.get("stress")
    common = {
        "energy": float(payload["energy"]),
        "forces": np.asarray(payload["forces"], dtype=float),
        "stress": None if stress is None else np.asarray(stress, dtype=float),
        "energy_kind": payload.get("energy_kind", "unknown"),
        "force_consistent": payload.get("force_consistent"),
    }
    if section == "reference":
        return EngineResult(wall_time_s=float(payload.get("wall_time_s", 0.0)),
                            **common)
    uncertainty = payload.get("uncertainty")
    return SurrogatePrediction(
        uncertainty=(np.asarray(uncertainty, dtype=float)
                     if uncertainty is not None
                     else np.full(len(common["forces"]), np.nan)),
        **common)


class _PlainDriver:
    """Velocity-Verlet NVE over one fixed backend, with energetic-style
    store rows and event records (no anchors, probes or checks).

    FixAtoms is supported through the same projection semantics as the
    energetic runner (constraints applied once, raw forces kept, projected
    driving force recorded).  Complete-step checkpoints use the WP03
    CheckpointManager unchanged, so plain runs resume exactly like adaptive
    ones.
    """

    def __init__(self, config: PyramidConfig, atoms: Atoms, backend: object,
                 run_dir: Path, *, event_log: EventLog | None = None,
                 resume_state: dict | None = None) -> None:
        self.config = config
        self.atoms = atoms
        self.backend = backend
        self.section = config.task.mode  # "reference" or "surrogate"
        self.run_dir = run_dir
        self.run_id = config.run.id
        self._stop_requested = False
        self._task_counter = 0
        self.projection = validate_constraints(atoms)
        self.store = Store(run_dir / "trajectory.db")
        self.event_log = event_log if event_log is not None else EventLog(run_dir)
        self._attempt_context_fields: dict | None = None
        atoms.calc = _BackendCalculator(
            backend, self.section, event_log=self.event_log,
            attempt_fields=lambda: self._attempt_context_fields)
        if "momenta" not in atoms.arrays:
            # NVT derives the initialization stream by role so it can never
            # share a generator with the bath (R1); NVE keeps the 0.4.x
            # behavior bit-compatible (no bath stream exists there).
            init_seed = config.dynamics.velocity_seed
            if config.dynamics.ensemble == "nvt":
                init_seed = derive_stream_seed(init_seed, "velocity")
            thermalize_momenta(atoms, config.dynamics.temperature_K,
                               rng=np.random.default_rng(init_seed))
        self.spec = _spec_from_dynamics(config)
        self.integrator = _make_integrator(self.spec, atoms, resume_state)
        self.dyn = self.integrator.dyn
        self.model_id = (model_id_for(backend, 0) if self.section == "surrogate"
                         else (fingerprint_of(backend) or type(backend).__qualname__))
        self._checkpoints: CheckpointManager | None = CheckpointManager(run_dir)
        self._resume_state = resume_state
        if resume_state is not None:
            self._task_counter = int(resume_state["task_counter"])
            self.dyn.nsteps = int(resume_state["nsteps"])

    def request_stop(self) -> None:
        self._stop_requested = True

    def close(self) -> None:
        self.event_log.close()
        self.store.close()

    def _emit_run_start(self) -> None:
        engine_fp = (fingerprint_of(self.backend)
                     if self.section == "reference" else None)
        self.event_log.append(RUN_START, {
            "run_id": self.run_id,
            "schema_version": STORE_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "attempt_ledger": ATTEMPT_LEDGER_PHYSICAL_V1,
            "software_version": __version__,
            "reference_id": engine_fp,
            "model_id": self.model_id,
            "surrogate_fingerprint": (fingerprint_of(self.backend)
                                      if self.section == "surrogate" else None),
            "workflow": {"driver": f"plain-{self.spec.ensemble}",
                         "mode": self.section,
                         "integrator": self.spec.as_dict()},
            # The effective stream identities (R1): the scheme name plus the
            # seeds actually used — derived ones differ from the raw fields
            # by the fixed role codes and are what a direct ASE comparison
            # must reproduce.
            "streams": {"scheme": STREAM_SCHEME,
                        "velocity_seed": (derive_stream_seed(
                            self.config.dynamics.velocity_seed, "velocity")
                            if self.spec.ensemble == "nvt"
                            else self.config.dynamics.velocity_seed),
                        "thermostat_seed": self.spec.thermostat_seed},
            "policy": None,
        })

    def _new_task_id(self) -> str:
        self._task_counter += 1
        return f"{self.run_id}-task-{self._task_counter}"

    def _evaluate(self, evaluation_id: int, compute: object) -> None:
        """One logical force evaluation: the physical compute under its
        attempt context, then the ledger task event.

        ``compute`` performs the actual force evaluation (ASE's
        VelocityVerlet evaluates the new-step forces inside ``step()``).
        When ASE serves the forces from the calculator cache (a stationary
        boundary), the logical request stays on the ledger as a cache hit,
        not as a physical execution.
        """
        task_id = self._new_task_id()
        calc = self.atoms.calc
        self._attempt_context_fields = {
            "request_id": task_id, "purpose": "md"}
        calculations_before = calc.n_calculations
        started_unix = time.time()
        zero = time.perf_counter()
        error: Exception | None = None
        try:
            compute()
        except Exception as exc:  # noqa: BLE001 — the single exit records
            error = exc                 # any failure before re-raising
        finally:
            self._attempt_context_fields = None
        calculated = calc.n_calculations > calculations_before
        measured = time.perf_counter() - zero
        label = calc.last_label
        operation = ("reference" if self.section == "reference"
                     else "inference")
        label_id = (f"{self.run_id}-label-{evaluation_id}"
                    if self.section == "reference" else None)
        # A cache hit bills the access time (this lookup), never the old
        # execution's wall time (B3); a real calculation bills its own.
        elapsed = (float(getattr(label, "wall_time_s", 0.0) or measured)
                   if calculated else measured)
        # One logical request, one task record from one exit — success and
        # failure share this task id, and the attempt's request_id points at
        # exactly this record (R5).
        self.event_log.append(TASK, {
            "task_id": task_id, "attempt": 1,
            "operation": operation, "purpose": "md",
            "status": ("failed" if error is not None
                       else "success" if calculated else "cache_hit"),
            "started_unix": started_unix,
            "elapsed_s": elapsed if error is None else measured,
            "cpu_cores": None, "gpu": None, "queue_s": None,
            "source": "workflow", "evaluation_id": evaluation_id,
            "label_id": label_id if error is None else None,
            "cache_hit": error is None and not calculated,
            "error": None if error is None else repr(error)})
        if error is not None:
            raise error

    def _record_evaluation(self, evaluation_id: int) -> None:
        ctx = EvaluationContext(
            run_id=self.run_id, step_id=evaluation_id - 1,
            evaluation_id=evaluation_id,
            phase=(EvaluationPhase.INITIAL if evaluation_id == 0
                   else EvaluationPhase.MD_STEP),
            physical_time_fs=evaluation_id * self.config.dynamics.timestep_fs,
            model_id=self.model_id)
        label = self.atoms.calc.last_label
        if label is None:
            raise WorkflowError(
                f"no {self.section} label for evaluation {evaluation_id}: "
                "the backend produced no record for this state — refusing "
                "to commit an empty payload")
        route = "dft" if self.section == "reference" else "ml"
        label_id = (f"{self.run_id}-label-{evaluation_id}"
                    if self.section == "reference" else None)
        driving = label
        constraint_record = None
        if self.projection is not None:
            # Raw physical forces stay in the backend payload; the driving
            # force that actually propagates has fixed DOFs zeroed.
            driving = dataclasses.replace(
                label, forces=self.projection.project_forces(label.forces))
            constraint_record = self.projection.as_dict()
            constraint_record["raw_forces_eV_A"] = np.asarray(
                label.forces, dtype=float).tolist()
        # snapshot without the calculator: the db row must not resurrect a
        # SinglePointCalculator on read (export writes its own info/arrays)
        row_id = self.store.append(
            self.run_id, ctx.step_id, self.atoms.copy(), route,
            surrogate=label if self.section == "surrogate" else None,
            engine=label if self.section == "reference" else None,
            reason="md",
            metadata={"context": ctx.as_dict(), "accepted": True,
                      "checked": False, "constraint": constraint_record},
            driving=driving, label_id=label_id)
        # The commit binds the row it authorizes (row id + digest): an
        # orphan row at the same step never becomes trajectory data (A3).
        # For NVT it also pins the bath stream at this boundary, so a resume
        # heals the step without re-running anything (R2/R3).
        payload = {"run_id": self.run_id, "context": ctx.as_dict(),
                   "route": route,
                   "checked": False, "verification": None,
                   "row_id": int(row_id),
                   "row_digest": self.store.row_digest(
                       self.store.row_by_id(int(row_id)))}
        if self.spec.algorithm == "langevin":
            payload["thermostat_rng"] = self.integrator.thermostat_state()
        self.event_log.append_once(
            f"evaluation:{self.run_id}:{evaluation_id}", EVALUATION_COMMITTED,
            payload)

    def _fail(self, error: Exception, evaluation_id: int) -> None:
        # The evaluation's own task record is already written by _evaluate's
        # single exit; the run only records its own termination here — no
        # extra fake request (R5).
        self.event_log.append(RUN_END, {"run_id": self.run_id,
                                        "status": "failed",
                                        "reason": repr(error)})

    def _write_checkpoint(self, step: int) -> int | None:
        if self._checkpoints is None:
            return None
        generation = self._checkpoints.next_generation()
        state = {
            "run_id": self.run_id,
            "driver": f"plain-{self.spec.ensemble}", "section": self.section,
            "nsteps": int(step), "task_counter": self._task_counter,
            "model_id": self.model_id,
            "engine_fingerprint": (fingerprint_of(self.backend)
                                   if self.section == "reference" else None),
            "timestep_fs": self.config.dynamics.timestep_fs,
            "integrator": self.spec.as_dict(),
            "thermostat": (None if self.spec.algorithm != "langevin"
                           else {"rng": self.integrator.thermostat_state()}),
            "driving_energy_eV": float(self.atoms.calc.results["energy"]),
            "constraint": (None if self.projection is None
                           else self.projection.as_dict()),
        }
        arrays = {
            "numbers": self.atoms.numbers,
            "cell": self.atoms.cell.array,
            "pbc": np.asarray(self.atoms.pbc),
            "masses": self.atoms.get_masses(),
            "initial_charges": self.atoms.get_initial_charges(),
            "initial_magmoms": self.atoms.get_initial_magnetic_moments(),
            "positions": self.atoms.positions,
            "momenta": self.atoms.get_momenta(),
            "driving_forces": np.asarray(self.atoms.calc.results["forces"],
                                         dtype=float),
        }
        self._checkpoints.write(generation, state, arrays, {
            "run_id": self.run_id,
            "nsteps": int(step),
            "physical_time_fs": step * self.config.dynamics.timestep_fs,
            "last_event_seq": self.event_log.last_seq,
            "store_schema_version": STORE_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "attempt_ledger": ATTEMPT_LEDGER_PHYSICAL_V1,
            "software_version": __version__,
        })
        return generation

    def run(self, n_steps: int, outputs: RunOutputs, *, verbose: bool,
            start_step: int = 0) -> dict:
        interval = outputs.summary_interval
        if start_step == 0 and self._resume_state is None:
            self._emit_run_start()
            self._evaluate(0, lambda: self.atoms.get_forces())
            self._record_evaluation(0)
            outputs.regenerate_trajectory()
        run_start = time.perf_counter()
        completed = start_step
        checkpoint_interval = self.config.checkpoint.interval_steps
        try:
            for step in range(start_step + 1, start_step + n_steps + 1):
                if self._stop_requested:
                    break
                # ASE's VelocityVerlet evaluates the new-step forces inside
                # step(): exactly one evaluation per step.  The committed
                # record afterwards reads the calculator cache.
                self._evaluate(
                    step,
                    lambda: self.dyn.step(self.atoms.calc.results["forces"]))
                self.dyn.nsteps = step
                self._record_evaluation(step)
                # Commit the step only with the complete boundary state
                # persisted and bound (M1): the step event names the
                # integrator and the boundary content digest.
                boundary = self.integrator.committed_state(
                    step - 1, self.atoms,
                    driving_source=("reference" if self.section == "reference"
                                    else "surrogate"),
                    model_id=self.model_id,
                    timestep_fs=self.config.dynamics.timestep_fs)
                step_payload = {"run_id": self.run_id, "step_id": step - 1,
                                "physical_time_fs": step * self.config.dynamics.timestep_fs,
                                "integrator": self.spec.as_dict(),
                                # format marker (S0b): records without it are
                                # verified by their recognizable older
                                # semantics, never silently skipped
                                "digest_format": DIGEST_FORMAT,
                                # array-level digest: verified against the
                                # authoritative row on resume (R3)
                                "state_digest": state_digest(
                                    self.atoms.positions,
                                    self.atoms.get_momenta()),
                                "boundary_digest": boundary.digest()}
                if self.spec.algorithm == "langevin":
                    step_payload["thermostat_rng"] = \
                        self.integrator.thermostat_state()
                self.event_log.append_once(
                    f"step:{self.run_id}:{step - 1}", STEP_COMPLETED,
                    step_payload)
                completed = step
                if step % checkpoint_interval == 0 or self._stop_requested:
                    self._write_checkpoint(step)
                outputs.append_trajectory_step(step)
                if step % interval == 0:
                    outputs.write_summaries()
                    if verbose:
                        print(f"  step {step}/{start_step + n_steps} "
                              f"(t = {step * self.config.dynamics.timestep_fs:.2f} fs)",
                              flush=True)
        except Exception as error:
            self._fail(error, completed + 1)
            raise
        stopped = self._stop_requested and completed < start_step + n_steps
        wall = time.perf_counter() - run_start
        if stopped:
            self.event_log.append(RUN_END, {
                "run_id": self.run_id, "status": "stopped",
                "reason": "stop requested; checkpoint saved at the last "
                          "complete step"})
        self.event_log.append(RUN_SUMMARY, {
            "run_id": self.run_id, "n_steps": completed,
            "n_evaluations": completed + 1, "n_accepted": completed + 1,
            "n_reference": (completed + 1 if self.section == "reference" else 0),
            "wall_time_s": wall, "stopped_early": stopped})
        return {"completed": completed, "stopped": stopped,
                "wall_time_s": wall}


def _run_plain(config: PyramidConfig, atoms: Atoms, run_dir: Path, *,
               verbose: bool, handle_sigint: bool) -> WorkflowResult:
    event_log = EventLog(run_dir)
    driver = None
    try:
        # The reference is created with the run's event log when its factory
        # accepts one (QE density I/O then enters the ledger).
        backend = _plain_backend(config, run_dir, event_log=event_log)
        prepare_run_directory(
            config, engine=backend if config.task.mode == "reference" else None,
            surrogate=backend if config.task.mode == "surrogate" else None)
        driver = _PlainDriver(config, atoms, backend, run_dir,
                              event_log=event_log)
        # Outputs construction joins the ownership scope: any setup failure
        # releases the log AND the driver's store before the error
        # continues (R4/S0a).
        outputs = RunOutputs(run_dir, config.run.id,
                             trajectory_interval_steps=config.output.trajectory_interval_steps,
                             summary_interval_steps=config.output.summary_interval_steps)
    except Exception:
        if driver is not None:
            driver.close()
        event_log.close()
        _remove_fresh_event_log(run_dir)
        raise
    if verbose:
        print(f"pyramid run: {config.run.id} — {config.task.mode}-only "
              f"{config.dynamics.ensemble.upper()} ({config.dynamics.integrator}), "
              f"{config.dynamics.steps} steps x {config.dynamics.timestep_fs} fs")
        print(f"run directory: {run_dir}")
    previous = signal.getsignal(signal.SIGINT) if handle_sigint else None
    if handle_sigint:
        signal.signal(signal.SIGINT,
                      lambda signum, frame: driver.request_stop())
    try:
        outcome = driver.run(config.dynamics.steps, outputs, verbose=verbose)
    finally:
        _finalize_quietly(outputs)
        driver.close()
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    steps = _complete_steps(run_dir, config.run.id)
    if verbose:
        status = ("stopped early at the last complete step"
                  if outcome["stopped"] else "completed")
        print(f"run {status}: {steps} complete steps "
              f"(wall time {outcome['wall_time_s']:.2f} s)")
        print(f"  outputs: {run_dir}/summary.json, summary.csv, trajectory.extxyz")
    return WorkflowResult(run_dir=run_dir, run_id=config.run.id,
                          mode=config.task.mode, steps_completed=steps,
                          steps_this_call=outcome["completed"],
                          stopped_early=outcome["stopped"], summary=None)


# ---------------------------------------------------------------------------
# public entry points


# ---------------------------------------------------------------------------
# singlepoint and relax tasks


def _plain_backend(config: PyramidConfig, run_dir: Path, *,
                   event_log: EventLog | None = None) -> object:
    return (create_configured_backend("reference", config.reference,
                                      run_dir=run_dir, event_log=event_log)
            if config.task.mode == "reference"
            else create_configured_backend("surrogate", config.surrogate,
                                           run_dir=run_dir))


def _run_singlepoint(config: PyramidConfig, atoms: Atoms, run_dir: Path, *,
                     verbose: bool) -> WorkflowResult:
    """One backend evaluation of the structure (reference or fixed surrogate)."""
    section = config.task.mode
    event_log = EventLog(run_dir)
    try:
        backend = _plain_backend(config, run_dir, event_log=event_log)
        prepare_run_directory(
            config, engine=backend if section == "reference" else None,
            surrogate=backend if section == "surrogate" else None)
    except Exception:
        event_log.close()
        _remove_fresh_event_log(run_dir)
        raise
    store = Store(run_dir / "trajectory.db")
    model_id = (model_id_for(backend, 0) if section == "surrogate"
                else (fingerprint_of(backend) or type(backend).__qualname__))
    ctx = EvaluationContext(run_id=config.run.id, step_id=-1, evaluation_id=0,
                            phase=EvaluationPhase.SINGLE_POINT,
                            physical_time_fs=0.0, model_id=model_id)
    event_log.append(RUN_START, {
        "run_id": config.run.id, "schema_version": STORE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "attempt_ledger": ATTEMPT_LEDGER_PHYSICAL_V1,
        "software_version": __version__,
        "reference_id": (fingerprint_of(backend) if section == "reference"
                         else None),
        "model_id": model_id,
        "workflow": {"driver": "singlepoint", "mode": section},
        "policy": None})
    started_unix = time.time()
    start = time.perf_counter()
    try:
        with physical_attempt(backend, event_log,
                              operation=("reference" if section == "reference"
                                         else "inference"),
                              request_id=f"{config.run.id}-task-1",
                              purpose="singlepoint",
                              source="workflow",
                              method=("compute" if section == "reference"
                                      else "predict")) as attempt_kwargs:
            if section == "reference":
                label = backend.compute(atoms, **attempt_kwargs)
            else:
                label = backend.predict(atoms, **attempt_kwargs)
        forces = np.asarray(label.forces, dtype=float)
        if not np.isfinite(label.energy) or not np.isfinite(forces).all():
            raise EngineError(f"{section} backend returned non-finite "
                              "energy/forces")
    except Exception as error:
        event_log.append(TASK, {
            "task_id": f"{config.run.id}-task-1", "attempt": 1,
            "operation": "reference" if section == "reference" else "inference",
            "purpose": "singlepoint", "status": "failed",
            "started_unix": started_unix,
            "elapsed_s": time.perf_counter() - start,
            "cpu_cores": None, "gpu": None, "queue_s": None,
            "source": "workflow", "evaluation_id": 0,
            "label_id": None, "cache_hit": False, "error": repr(error)})
        event_log.append(RUN_END, {"run_id": config.run.id,
                                   "status": "failed", "reason": repr(error)})
        store.close()
        event_log.close()
        raise
    elapsed = time.perf_counter() - start
    label_id = f"{config.run.id}-label-0" if section == "reference" else None
    event_log.append(TASK, {
        "task_id": f"{config.run.id}-task-1", "attempt": 1,
        "operation": "reference" if section == "reference" else "inference",
        "purpose": "singlepoint", "status": "success",
        "started_unix": started_unix, "elapsed_s": elapsed,
        "cpu_cores": None, "gpu": None, "queue_s": None,
        "source": "workflow", "evaluation_id": 0,
        "label_id": label_id, "cache_hit": False})
    projection = validate_constraints(atoms)
    driving = label
    constraint_record = None
    if projection is not None:
        # Raw physical forces stay in the backend payload; the driving
        # force is the constraint-projected one, same as in MD runs.
        driving = dataclasses.replace(
            label, forces=projection.project_forces(label.forces))
        constraint_record = projection.as_dict()
        constraint_record["raw_forces_eV_A"] = np.asarray(
            label.forces, dtype=float).tolist()
    row_id = store.append(config.run.id, -1, atoms.copy(),
             "dft" if section == "reference" else "ml",
             surrogate=label if section == "surrogate" else None,
             engine=label if section == "reference" else None,
             reason="singlepoint",
             metadata={"context": ctx.as_dict(), "accepted": True,
                       "checked": False, "constraint": constraint_record},
             driving=driving, label_id=label_id)
    event_log.append_once(f"evaluation:{config.run.id}:0", EVALUATION_COMMITTED,
                          {"run_id": config.run.id, "context": ctx.as_dict(),
                           "route": "dft" if section == "reference" else "ml",
                           "checked": False, "verification": None,
                           "row_id": int(row_id),
                           "row_digest": store.row_digest(
                               store.row_by_id(int(row_id)))})
    event_log.append(RUN_SUMMARY, {
        "run_id": config.run.id, "n_steps": 0, "n_evaluations": 1,
        "n_accepted": 1,
        "n_reference": 1 if section == "reference" else 0,
        "wall_time_s": elapsed, "stopped_early": False})
    store.close()
    event_log.close()
    driving_forces = np.asarray(driving.forces, dtype=float)
    if verbose:
        print(f"singlepoint ({section}): energy {float(label.energy):.10f} eV, "
              f"max |F| {float(np.linalg.norm(driving_forces, axis=1).max()):.6f} eV/A")
        print(f"run directory: {run_dir}")
    return WorkflowResult(run_dir=run_dir, run_id=config.run.id,
                          mode=section, steps_completed=0, steps_this_call=0,
                          stopped_early=False, summary=None)


def _run_relax(config: PyramidConfig, atoms: Atoms, run_dir: Path, *,
               verbose: bool) -> WorkflowResult:
    """Fixed-model structure optimization with ASE FIRE/BFGS.

    Only a fixed surrogate or reference-only optimization is allowed — the
    energetic calculator (time gating + model switching) is never placed
    under an optimizer, because it does not define a fixed potential surface.
    """
    section = config.task.mode
    event_log = EventLog(run_dir)
    try:
        backend = _plain_backend(config, run_dir, event_log=event_log)
        prepare_run_directory(
            config, engine=backend if section == "reference" else None,
            surrogate=backend if section == "surrogate" else None)
    except Exception:
        event_log.close()
        _remove_fresh_event_log(run_dir)
        raise
    store = Store(run_dir / "trajectory.db")
    model_id = (model_id_for(backend, 0) if section == "surrogate"
                else (fingerprint_of(backend) or type(backend).__qualname__))
    task_counter = 0
    event_log.append(RUN_START, {
        "run_id": config.run.id, "schema_version": STORE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "attempt_ledger": ATTEMPT_LEDGER_PHYSICAL_V1,
        "software_version": __version__,
        "reference_id": (fingerprint_of(backend) if section == "reference"
                         else None),
        "model_id": model_id,
        "workflow": {"driver": "relax", "mode": section,
                     "optimizer": config.relax.optimizer},
        "policy": None})

    def _record_evaluation(label: object, evaluation_id: int) -> None:
        nonlocal task_counter
        task_counter += 1
        label_id = (f"{config.run.id}-label-{evaluation_id}"
                    if section == "reference" else None)
        event_log.append(TASK, {
            "task_id": f"{config.run.id}-task-{task_counter}", "attempt": 1,
            "operation": "reference" if section == "reference" else "inference",
            "purpose": "relax", "status": "success",
            "started_unix": time.time(),
            "elapsed_s": float(getattr(label, "wall_time_s", 0.0) or 0.0),
            "cpu_cores": None, "gpu": None, "queue_s": None,
            "source": "workflow", "evaluation_id": evaluation_id,
            "label_id": label_id, "cache_hit": False})

    evaluations = 0

    def _on_evaluation(label: object) -> None:
        nonlocal evaluations
        evaluations += 1
        _record_evaluation(label, evaluations)

    atoms.calc = _BackendCalculator(
        backend, section, on_evaluation=_on_evaluation, event_log=event_log,
        # calculate() runs before _on_evaluation bumps the counters, so the
        # attempt's parent is the task id this evaluation is about to get.
        attempt_fields=lambda: {
            "request_id": f"{config.run.id}-task-{task_counter + 1}",
            "purpose": "relax"})
    projection = validate_constraints(atoms)
    if config.relax.optimizer == "bfgs":
        from ase.optimize import BFGS

        optimizer = BFGS(atoms)
    else:
        from ase.optimize import FIRE

        optimizer = FIRE(atoms)
    start = time.perf_counter()

    last_frame_key: list = [None]

    def _record_frame() -> None:
        step = int(optimizer.nsteps)
        ctx = EvaluationContext(run_id=config.run.id, step_id=step - 1,
                                evaluation_id=evaluations,
                                phase=EvaluationPhase.OPTIMIZATION_TRIAL,
                                physical_time_fs=0.0, model_id=model_id)
        key = (step, evaluations)
        if last_frame_key[0] == key:
            return  # the terminal frame duplicates the last trial's commit
        last_frame_key[0] = key
        label = atoms.calc.last_label
        driving = label
        constraint_record = None
        if projection is not None:
            # Raw physical forces stay in the backend payload; the driving
            # force is the constraint-projected one the optimizer used.
            driving = dataclasses.replace(
                label, forces=projection.project_forces(label.forces))
            constraint_record = projection.as_dict()
            constraint_record["raw_forces_eV_A"] = np.asarray(
                label.forces, dtype=float).tolist()
        label_id = (f"{config.run.id}-label-{evaluations}"
                    if section == "reference" else None)
        row_id = store.append(config.run.id, step, atoms.copy(),
                              "dft" if section == "reference" else "ml",
                              surrogate=label if section == "surrogate" else None,
                              engine=label if section == "reference" else None,
                              reason="relax",
                              metadata={"context": ctx.as_dict(),
                                        "accepted": True, "checked": False,
                                        "constraint": constraint_record},
                              driving=driving, label_id=label_id,
                              dedupe=True)
        # Every exported frame is a committed record: bind the row like the
        # MD drivers do, so export reads commit→row identity (A2/A3).
        event_log.append_once(
            f"evaluation:{config.run.id}:{evaluations}", EVALUATION_COMMITTED,
            {"run_id": config.run.id, "context": ctx.as_dict(),
             "route": "dft" if section == "reference" else "ml",
             "checked": False, "verification": None,
             "row_id": int(row_id),
             "row_digest": store.row_digest(store.row_by_id(int(row_id)))})

    optimizer.attach(_record_frame, interval=1)
    try:
        converged = bool(optimizer.run(fmax=config.relax.fmax_eV_A,
                                       steps=config.relax.steps))
    except Exception as error:
        # The evaluation that was running still closes its one logical task
        # record — same task id the attempt carries, no renumbering (B3).
        event_log.append(TASK, {
            "task_id": f"{config.run.id}-task-{task_counter + 1}",
            "attempt": 1,
            "operation": "reference" if section == "reference" else "inference",
            "purpose": "relax", "status": "failed",
            "started_unix": time.time(),
            "elapsed_s": time.perf_counter() - start,
            "cpu_cores": None, "gpu": None, "queue_s": None,
            "source": "workflow", "evaluation_id": evaluations + 1,
            "label_id": None, "cache_hit": False, "error": repr(error)})
        event_log.append(RUN_END, {"run_id": config.run.id,
                                   "status": "failed", "reason": repr(error)})
        store.close()
        event_log.close()
        raise
    _record_frame()
    wall = time.perf_counter() - start
    # The convergence metric is the one the optimizer used: the maximum
    # single-atom force norm after constraints are applied (ASE's
    # get_forces zeroes fixed DOFs). The raw all-atom value — which
    # includes reaction forces on fixed atoms — is recorded alongside.
    final_forces = np.asarray(atoms.get_forces(), dtype=float)
    final_fmax = float(np.linalg.norm(final_forces, axis=1).max())
    raw_fmax = float(np.linalg.norm(
        np.asarray(atoms.calc.results["forces"], dtype=float), axis=1).max())
    event_log.append(RUN_SUMMARY, {
        "run_id": config.run.id, "n_steps": int(optimizer.nsteps),
        "n_evaluations": evaluations, "n_accepted": evaluations,
        "n_reference": evaluations if section == "reference" else 0,
        "wall_time_s": wall, "stopped_early": False,
        "converged": converged, "final_fmax_eV_A": final_fmax,
        "raw_all_atom_fmax_eV_A": raw_fmax})
    store.close()
    event_log.close()
    if verbose:
        status = "converged" if converged else "not converged"
        print(f"relax ({section}, {config.relax.optimizer}): {status} in "
              f"{optimizer.nsteps} steps — final max |F| "
              f"{final_fmax:.6f} eV/A (target {config.relax.fmax_eV_A} eV/A)")
        if projection is not None:
            print(f"  raw all-atom max |F| {raw_fmax:.6f} eV/A "
                  "(includes reaction forces on fixed atoms)")
        print(f"run directory: {run_dir}")
    return WorkflowResult(run_dir=run_dir, run_id=config.run.id,
                          mode=section, steps_completed=int(optimizer.nsteps),
                          steps_this_call=int(optimizer.nsteps),
                          stopped_early=not converged, summary=None)


def run_workflow(config: PyramidConfig, *, verbose: bool = True,
                 handle_sigint: bool = True) -> WorkflowResult:
    """Validate, set up the run directory and execute the configured task."""
    if config.task.kind not in ("singlepoint", "relax", "md"):
        raise WorkflowError(
            f"task.kind {config.task.kind!r} is not supported; choose "
            "singlepoint, relax or md")
    validate_setup(config)  # dry pass: identical failures as `validate`
    atoms = load_structure(config)
    check_run_directory_available(config)
    run_dir = config.run.directory
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.task.kind == "singlepoint":
        return _run_singlepoint(config, atoms, run_dir, verbose=verbose)
    if config.task.kind == "relax":
        return _run_relax(config, atoms, run_dir, verbose=verbose)
    if config.task.mode == "adaptive":
        return _run_adaptive(config, atoms, run_dir,
                             verbose=verbose, handle_sigint=handle_sigint)
    return _run_plain(config, atoms, run_dir,
                      verbose=verbose, handle_sigint=handle_sigint)


def resume_workflow(run_dir: str | Path, extra_steps: int, *,
                    force_unlock: bool = False, verbose: bool = True,
                    handle_sigint: bool = True,
                    updater: object | None = None) -> WorkflowResult:
    """Continue a run for ``extra_steps`` additional steps.

    Settings come from the run's ``resolved_config.json`` — the same physics,
    model chain and check stream; ``--steps`` is always *additional* steps
    and the current/target step numbers are printed up front. Plain
    reference/surrogate runs resume from their complete-step checkpoints
    (WP07); adaptive runs resume through the WP03 protocol, with an optional
    stateful updater passed through (WP06).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise WorkflowError(
            f"run directory not found: {run_dir}; pass the directory written "
            "by `pyramid run` (it contains resolved_config.json)")
    if isinstance(extra_steps, bool) or not isinstance(extra_steps, int) \
            or extra_steps < 1:
        raise WorkflowError(
            f"resume --steps must be a positive integer, got {extra_steps!r}")
    config = load_resolved_config(run_dir)
    if config.task.kind != "md":
        raise WorkflowError(
            f"this run used task.kind {config.task.kind!r}; resume is "
            "implemented for md runs (singlepoint has nothing to continue, "
            "relax runs reach their target or stop)")
    if config.task.mode != "adaptive":
        return _resume_plain(config, run_dir, extra_steps,
                             force_unlock=force_unlock, verbose=verbose,
                             handle_sigint=handle_sigint)
    current = _complete_steps(run_dir, config.run.id)
    target = current + extra_steps
    if verbose:
        print(f"resume: run {config.run.id} is at complete step {current}; "
              f"running {extra_steps} additional steps (target {target})")
    event_log = EventLog(run_dir, force=force_unlock)
    runner = None
    try:
        # Backends are created with the run's event log when their factory
        # accepts one, so post-resume physical attempts keep entering the
        # ledger instead of going silent after the restart.
        engine, surrogate = build_backends(config, run_dir=run_dir,
                                           event_log=event_log)
        with _SigintGuard(handle_sigint):
            runner = EnergeticRunner.resume(
                run_dir, surrogate, engine, updater=updater,
                checkpoint_interval_steps=config.checkpoint.interval_steps,
                handle_sigint=handle_sigint, event_log=event_log)
            outputs = RunOutputs(
                run_dir, config.run.id,
                trajectory_interval_steps=config.output.trajectory_interval_steps,
                summary_interval_steps=config.output.summary_interval_steps)
            outputs.regenerate_trajectory()  # heal any crash-window preview holes
            _attach_outputs(runner, outputs)

            def _progress() -> None:
                if (verbose and runner.dyn.nsteps > 0
                        and runner.dyn.nsteps % outputs.summary_interval == 0):
                    print(f"  step {runner.dyn.nsteps}/{target} "
                          f"(t = {runner.dyn.nsteps * config.dynamics.timestep_fs:.2f} fs)",
                          flush=True)

            runner.dyn.attach(_progress, interval=outputs.summary_interval)
            try:
                summary = runner.run(extra_steps)
            finally:
                _finalize_quietly(outputs)
                runner.close()
    finally:
        if runner is None:
            event_log.close()
    after = _complete_steps(run_dir, config.run.id)
    stopped = _stopped_early(run_dir, config.run.id)
    if verbose:
        status = "stopped early (checkpoint saved)" if stopped else "done"
        print(f"resume {status}: run is now at complete step {after}")
        _print_run_report(summary, run_dir)
    return WorkflowResult(run_dir=run_dir, run_id=config.run.id,
                          mode="adaptive", steps_completed=after,
                          steps_this_call=after - current,
                          stopped_early=stopped, summary=summary)


def _resume_plain(config: PyramidConfig, run_dir: Path, extra_steps: int, *,
                  force_unlock: bool, verbose: bool,
                  handle_sigint: bool) -> WorkflowResult:
    """Resume a plain reference/surrogate MD run from its last valid
    complete-step checkpoint (same CheckpointManager schema as adaptive)."""
    manager = CheckpointManager(run_dir)
    checkpoint = manager.read_latest_valid()
    if checkpoint is None:
        raise WorkflowError(
            f"no valid checkpoint under {run_dir}; this plain run has no "
            "complete-step checkpoint to resume from (checkpoints are written "
            f"every {config.checkpoint.interval_steps} steps and on a stop "
            "request)")
    state, arrays, manifest = checkpoint.state, checkpoint.arrays, checkpoint.manifest
    expected_driver = f"plain-{config.dynamics.ensemble}"
    if state.get("driver") != expected_driver \
            or state.get("section") != config.task.mode:
        raise WorkflowError(
            f"checkpoint under {run_dir} belongs to driver "
            f"{state.get('driver')!r}/{state.get('section')!r}, not to a "
            f"plain {config.task.mode!r} {config.dynamics.ensemble} run; "
            "resume requires the same run")
    # The integrator identity is part of the run: a change of algorithm,
    # timestep, temperature or friction means a new run, not a resume.
    spec = _spec_from_dynamics(config)
    checkpoint_integrator = state.get("integrator")
    if checkpoint_integrator is not None and \
            checkpoint_integrator != spec.as_dict():
        raise WorkflowError(
            f"integrator settings changed from "
            f"{checkpoint_integrator!r} to {spec.as_dict()!r}; resume "
            "continues the same integration settings — start a new run")
    if spec.algorithm == "langevin":
        thermostat = state.get("thermostat") or {}
        if thermostat.get("rng") is None:
            raise WorkflowError(
                f"checkpoint under {run_dir} has no persisted thermostat "
                "state; this NVT run cannot be resumed honestly (an old or "
                "torn checkpoint — start a new run)")
    event_log = EventLog(run_dir, force=force_unlock)
    try:
        backend = _plain_backend(config, run_dir, event_log=event_log)
        if config.task.mode == "reference" and \
                fingerprint_of(backend) != state.get("engine_fingerprint"):
            raise WorkflowError(
                "reference backend identity does not match the checkpoint; "
                "resume requires the same physical settings")
        if config.task.mode == "surrogate":
            # Same alignment as the reference path: the model identity and the
            # immutable integration settings are compared before anything is
            # written or advanced; a physical change means a new run (or fork).
            if model_id_for(backend, 0) != state.get("model_id"):
                raise WorkflowError(
                    f"surrogate identity does not match the checkpoint "
                    f"({state.get('model_id')!r} vs "
                    f"{model_id_for(backend, 0)!r}); resume requires the same "
                    "model — start a new run, or fork from the checkpoint")
            if float(state.get("timestep_fs", -1.0)) != \
                    float(config.dynamics.timestep_fs):
                raise WorkflowError(
                    f"timestep_fs changed from {state.get('timestep_fs')} to "
                    f"{config.dynamics.timestep_fs} fs; resume continues the same "
                    "integration settings — start a new run, or fork")
    except Exception:
        event_log.close()
        raise
    # From here on the workflow owns the log: any failure before the
    # driver's own finally — boundary healing/verification, driver or
    # outputs construction — releases it before the error continues (R4).
    store = None
    driver = None
    try:
        current = _complete_steps(run_dir, config.run.id)
        if current < int(state["nsteps"]):
            raise WorkflowError(
                f"the trajectory ({current} complete steps) is behind the "
                f"checkpoint (step {state['nsteps']}); the run directory is "
                "inconsistent")
        target = current + extra_steps
        if verbose:
            print(f"resume: run {config.run.id} ({config.task.mode}-only) is at "
                  f"complete step {current}; running {extra_steps} additional "
                  f"steps (target {target})")
        from ase.constraints import FixAtoms

        store = Store(run_dir / "trajectory.db")
        # Heal the committed-evaluation / uncommitted-step window (R3): the
        # plain driver logs the post-step state with the evaluation commit, so
        # a committed evaluation one past the last complete step IS a finished
        # step — bind it instead of recomputing anything.  The plain driver
        # can leave at most one such evaluation; more means an inconsistent
        # log.
        events = list(event_log.iter_events())
        committed_evaluations = sorted({
            int((e.get("context") or {})["evaluation_id"])
            for e in events if e.get("type") == EVALUATION_COMMITTED})
        n_committed = len(committed_evaluations)
        if n_committed == current + 2:
            heal_id = current + 1
            heal_row = store.committed_row(events, config.run.id, heal_id)
            commit = next(e for e in events
                          if e.get("type") == EVALUATION_COMMITTED
                          and int((e.get("context") or {})
                                  ["evaluation_id"]) == heal_id)
            # The step commit names the full boundary state of the authoritative
            # row; its bath stream comes from the evaluation commit.  Resume
            # re-emits it idempotently instead of recomputing the step.  The
            # payload is field-identical to a normal step commit (S0b),
            # including the boundary digest over the bath stream.
            heal_payload = {"run_id": config.run.id, "step_id": current,
                            "physical_time_fs": (current + 1)
                            * config.dynamics.timestep_fs,
                            "integrator": spec.as_dict(),
                            "digest_format": DIGEST_FORMAT,
                            "state_digest": state_digest(
                                heal_row.toatoms().positions,
                                heal_row.toatoms().get_momenta())}
            if commit.get("thermostat_rng") is not None:
                heal_payload["thermostat_rng"] = commit["thermostat_rng"]
            heal_payload["boundary_digest"] = _boundary_from_event(
                heal_row, heal_payload, commit).digest()
            event_log.append_once(f"step:{config.run.id}:{current}",
                                  STEP_COMPLETED, heal_payload)
            current += 1
        elif n_committed > current + 2:
            raise WorkflowError(
                f"the event log under {run_dir} has {n_committed} committed "
                f"evaluations but only {current} complete steps; the run "
                "directory is inconsistent — refusing to guess a boundary")
        # The boundary is the last committed evaluation's row — never an orphan
        # and never a recomputed state (C1/R3).  NVT additionally restores the
        # bath stream from that commit; a step boundary digest, when present,
        # is verified against the row before use, by the semantics of the
        # format that wrote it (S0b).
        if current >= 1:
            row = store.committed_row(event_log, config.run.id, current)
            atoms = row.toatoms()
            driving_energy, driving_forces = store.driving_label_for_row(row)
            boundary_step = next((e for e in reversed(events)
                                  if e.get("type") == STEP_COMPLETED
                                  and int(e["step_id"]) == current - 1), None)
            if boundary_step is not None and boundary_step.get("state_digest"):
                commit = next(
                    (e for e in reversed(events)
                     if e.get("type") == EVALUATION_COMMITTED
                     and int((e.get("context") or {})
                             ["evaluation_id"]) == current), None)
                boundary = _boundary_from_event(row, boundary_step, commit)
                digest_format = boundary_step.get("digest_format")
                if digest_format == DIGEST_FORMAT:
                    # boundary-v2: the array digest binds the row's
                    # positions/momenta; the boundary digest binds the
                    # complete reconstructed state, bath stream included.
                    mismatches = []
                    actual = state_digest(atoms.positions, atoms.get_momenta())
                    if actual != boundary_step["state_digest"]:
                        mismatches.append(
                            f"state digest ({boundary_step['state_digest']} "
                            f"vs {actual})")
                    actual = boundary.digest()
                    if actual != boundary_step.get("boundary_digest"):
                        mismatches.append(
                            f"boundary digest "
                            f"({boundary_step.get('boundary_digest')} vs "
                            f"{actual})")
                elif digest_format is None:
                    # Unmarked historical records, each verified by the
                    # semantics it was written with — verified, never
                    # rewritten:
                    # - the previous development batch's single JSON digest
                    #   (state_digest alone);
                    # - 182cc8d's dual record: array state_digest plus a JSON
                    #   boundary_digest that does NOT cover the bath stream
                    #   (the thermostat_rng field still restores the bath per
                    #   the existing policy, but this format never proved the
                    #   complete RNG identity);
                    # - old healed records that may carry only the array
                    #   digest.
                    array_digest = state_digest(atoms.positions,
                                                atoms.get_momenta())
                    recorded = boundary_step["state_digest"]
                    if boundary_step.get("boundary_digest") is not None:
                        mismatches = []
                        if recorded != array_digest:
                            mismatches.append(
                                f"array state digest ({recorded} vs "
                                f"{array_digest})")
                        actual = boundary.legacy_digest()
                        if boundary_step["boundary_digest"] != actual:
                            mismatches.append(
                                f"boundary digest "
                                f"({boundary_step['boundary_digest']} vs "
                                f"{actual})")
                    elif recorded == boundary.legacy_digest():
                        mismatches = []  # single-JSON semantics verified
                    elif recorded == array_digest:
                        mismatches = []  # array-only heal record verified
                    else:
                        legacy = boundary.legacy_digest()
                        mismatches = [
                            (f"state digest matches no known unmarked format "
                             f"({recorded} vs legacy {legacy} / array "
                             f"{array_digest})"),
                        ]
                else:
                    raise WorkflowError(
                        f"the step-{current - 1} record claims an unknown "
                        f"digest format {digest_format!r}; the run directory "
                        "is inconsistent")
                if mismatches:
                    raise WorkflowError(
                        f"the step-{current - 1} boundary does not match the "
                        f"committed row ({'; '.join(mismatches)}); the run "
                        "directory is inconsistent")
            # A boundary with no step summary at all is a 0.4.x record: the
            # committed-row binding above (row id + row digest) is the only
            # verification that format carries.
        else:
            atoms = Atoms(numbers=np.array(arrays["numbers"]),
                          positions=np.array(arrays["positions"], dtype=float),
                          cell=np.array(arrays["cell"]), pbc=np.array(arrays["pbc"]))
            atoms.set_masses(np.array(arrays["masses"]))
            atoms.set_initial_charges(np.array(arrays["initial_charges"]))
            atoms.set_initial_magnetic_moments(np.array(arrays["initial_magmoms"]))
            atoms.set_momenta(np.array(arrays["momenta"], dtype=float))
            driving_energy = float(state["driving_energy_eV"])
            driving_forces = np.array(arrays["driving_forces"], dtype=float)
            row = store._row_at_step(config.run.id, -1)  # the initial evaluation
        boundary_label = _label_from_row(row, config.task.mode)
        store.close()  # the boundary read is done; the driver owns its store
        constraint = state.get("constraint")
        if constraint is not None:
            atoms.set_constraint(FixAtoms(indices=list(constraint["indices"])))
        task_counter = int(state["task_counter"])
        for event in events:
            if event.get("type") == TASK:
                try:
                    task_counter = max(task_counter,
                                       int(str(event.get("task_id", "")).rsplit("-", 1)[1]))
                except (ValueError, IndexError):
                    continue
        resume_state = state
        if spec.algorithm == "langevin":
            boundary_commit = next((e for e in reversed(events)
                                    if e.get("type") == EVALUATION_COMMITTED
                                    and int((e.get("context") or {})
                                            ["evaluation_id"]) == current), None)
            thermostat_rng = (boundary_commit or {}).get("thermostat_rng")
            if thermostat_rng is None:
                raise WorkflowError(
                    f"the boundary evaluation of {run_dir} has no persisted "
                    "thermostat state; this NVT log predates bath persistence "
                    "and cannot be resumed without guessing — start a new run")
            resume_state = dict(state)
            resume_state["thermostat"] = {"rng": thermostat_rng}
        driver = _PlainDriver(config, atoms, backend, run_dir,
                              event_log=event_log, resume_state=resume_state)
        driver._task_counter = task_counter
        driver.dyn.nsteps = current
        driver.atoms.calc.atoms = atoms.copy()
        # The reusable driving force of the boundary evaluation feeds the first
        # half-kick without recalculating it.
        driver.atoms.calc.results = {
            "energy": float(driving_energy),
            "forces": np.asarray(driving_forces, dtype=float).copy()}
        # If the first resumed step does not move the atoms (a stationary
        # boundary), ASE legitimately skips calculate() and last_label would
        # stay unset; the committed boundary row is the same label.
        driver.atoms.calc.last_label = boundary_label
        event_log.append(RESUMED, {
            "run_id": config.run.id, "driver": f"plain-{config.dynamics.ensemble}",
            "from_event_seq": int(manifest["last_event_seq"]),
            "checkpoint_generation": checkpoint.generation})
        outputs = RunOutputs(
            run_dir, config.run.id,
            trajectory_interval_steps=config.output.trajectory_interval_steps,
            summary_interval_steps=config.output.summary_interval_steps)
        outputs.regenerate_trajectory()
        previous = signal.getsignal(signal.SIGINT) if handle_sigint else None
        if handle_sigint:
            signal.signal(signal.SIGINT,
                          lambda signum, frame: driver.request_stop())
        try:
            outcome = driver.run(extra_steps, outputs, verbose=verbose,
                                 start_step=current)
        finally:
            _finalize_quietly(outputs)
            driver.close()
            if handle_sigint:
                signal.signal(signal.SIGINT, previous)
        after = _complete_steps(run_dir, config.run.id)
        if verbose:
            status = ("stopped early at the last complete step"
                      if outcome["stopped"] else "done")
            print(f"resume {status}: run is now at complete step {after}")
        return WorkflowResult(run_dir=run_dir, run_id=config.run.id,
                              mode=config.task.mode, steps_completed=after,
                              steps_this_call=after - current,
                              stopped_early=outcome["stopped"], summary=None)
    except Exception:
        # idempotent: the driver's own finally may already have closed them
        if driver is not None:
            driver.close()
        if store is not None:
            store.close()
        event_log.close()
        raise
