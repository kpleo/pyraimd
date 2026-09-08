"""MD workflow orchestration: one implementation for the Python API and the CLI.

Three modes over the same run-directory layout:

- ``adaptive``: the energetic runner (WP03) — anchored surrogate forces,
  independent reference checks, complete-step checkpoints, resumable.
- ``reference``: plain velocity-Verlet NVE driven by the reference engine.
- ``surrogate``: plain velocity-Verlet NVE driven by the frozen surrogate.

The two plain modes share one small driver built on ASE's VelocityVerlet and
write the same store-row and event shapes the energetic runner writes, so
``inspect`` and ``export`` work uniformly.  Plain-mode resume is **not**
claimed in 0.4.0 (it needs the complete-step checkpoint protocol, which is
energetic-specific): ``resume`` on such a directory fails with an explicit
message instead of guessing from the trajectory database.

``singlepoint`` / ``relax`` are refused here with a pointer to WP07 — they
are not silently mapped onto the MD path.
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

from pyraimd2 import __version__
from pyraimd2.config import PyramidConfig, load_resolved_config
from pyraimd2.engines.base import EngineError
from pyraimd2.loop import EnergeticRunner
from pyraimd2.loop.energetic import EnergeticRunSummary
from pyraimd2.runtime.context import EvaluationContext, EvaluationPhase
from pyraimd2.runtime.events import (
    EVALUATION_COMMITTED,
    EVENT_SCHEMA_VERSION,
    RUN_END,
    RUN_START,
    RUN_SUMMARY,
    STEP_COMPLETED,
    TASK,
    EventLog,
)
from pyraimd2.runtime.identity import fingerprint_of, model_id_for
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.store.store import STORE_SCHEMA_VERSION
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
                event_log=event_log, run_dir=run_dir,
                checkpoint_interval_steps=config.checkpoint.interval_steps,
                handle_sigint=handle_sigint,
                **_policy_kwargs(config))
        except Exception:
            event_log.close()
            _remove_fresh_event_log(run_dir)
            raise

        outputs = RunOutputs(
            run_dir, config.run.id,
            trajectory_interval_steps=config.output.trajectory_interval_steps,
            summary_interval_steps=config.output.summary_interval_steps)
        _attach_outputs(runner, outputs)
        total = config.dynamics.steps
        if verbose:
            print(f"pyramid run: {config.run.id} — adaptive NVE, "
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

    def __init__(self, backend: object, section: str) -> None:
        super().__init__()
        self._backend = backend
        self._section = section
        self.last_label: Any = None

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self._section == "reference":
            label = self._backend.compute(self.atoms)
        else:
            label = self._backend.predict(self.atoms)
        forces = np.asarray(label.forces, dtype=float)
        if not np.isfinite(label.energy) or not np.isfinite(forces).all():
            raise EngineError(
                f"{self._section} backend returned non-finite energy/forces")
        self.last_label = label
        self.results = {"energy": float(label.energy), "forces": forces}


class _PlainDriver:
    """Velocity-Verlet NVE over one fixed backend, with energetic-style
    store rows and event records (no anchors, probes or checks)."""

    def __init__(self, config: PyramidConfig, atoms: Atoms, backend: object,
                 run_dir: Path) -> None:
        self.config = config
        self.atoms = atoms
        self.backend = backend
        self.section = config.task.mode  # "reference" or "surrogate"
        self.run_dir = run_dir
        self.run_id = config.run.id
        self._stop_requested = False
        self._task_counter = 0
        self.store = Store(run_dir / "trajectory.db")
        self.event_log = EventLog(run_dir)
        atoms.calc = _BackendCalculator(backend, self.section)
        if "momenta" not in atoms.arrays:
            thermalize_momenta(atoms, config.dynamics.temperature_K,
                               rng=np.random.default_rng(config.dynamics.velocity_seed))
        self.dyn = VelocityVerlet(atoms, config.dynamics.timestep_fs * units.fs)
        self.model_id = (model_id_for(backend, 0) if self.section == "surrogate"
                         else (fingerprint_of(backend) or type(backend).__qualname__))

    def request_stop(self) -> None:
        self._stop_requested = True

    def close(self) -> None:
        self.event_log.close()

    def _emit_run_start(self) -> None:
        engine_fp = (fingerprint_of(self.backend)
                     if self.section == "reference" else None)
        self.event_log.append(RUN_START, {
            "run_id": self.run_id,
            "schema_version": STORE_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "software_version": __version__,
            "reference_id": engine_fp,
            "model_id": self.model_id,
            "surrogate_fingerprint": (fingerprint_of(self.backend)
                                      if self.section == "surrogate" else None),
            "workflow": {"driver": "plain-nve", "mode": self.section},
            "policy": None,
        })

    def _new_task_id(self) -> str:
        self._task_counter += 1
        return f"{self.run_id}-task-{self._task_counter}"

    def _evaluate(self, evaluation_id: int) -> tuple[float, float]:
        """One force evaluation at the current positions + task event."""
        started_unix = time.time()
        zero = time.perf_counter()
        self.atoms.get_forces()
        measured = time.perf_counter() - zero
        label = self.atoms.calc.last_label
        operation = ("reference" if self.section == "reference"
                     else "inference")
        label_id = (f"{self.run_id}-label-{evaluation_id}"
                    if self.section == "reference" else None)
        self.event_log.append(TASK, {
            "task_id": self._new_task_id(), "attempt": 1,
            "operation": operation, "purpose": "md", "status": "success",
            "started_unix": started_unix,
            "elapsed_s": float(getattr(label, "wall_time_s", 0.0) or measured),
            "cpu_cores": None, "gpu": None, "queue_s": None,
            "source": "workflow", "evaluation_id": evaluation_id,
            "label_id": label_id, "cache_hit": False})
        return started_unix, measured

    def _record_evaluation(self, evaluation_id: int) -> None:
        ctx = EvaluationContext(
            run_id=self.run_id, step_id=evaluation_id - 1,
            evaluation_id=evaluation_id,
            phase=(EvaluationPhase.INITIAL if evaluation_id == 0
                   else EvaluationPhase.MD_STEP),
            physical_time_fs=evaluation_id * self.config.dynamics.timestep_fs,
            model_id=self.model_id)
        label = self.atoms.calc.last_label
        route = "dft" if self.section == "reference" else "ml"
        label_id = (f"{self.run_id}-label-{evaluation_id}"
                    if self.section == "reference" else None)
        # snapshot without the calculator: the db row must not resurrect a
        # SinglePointCalculator on read (export writes its own info/arrays)
        self.store.append(
            self.run_id, ctx.step_id, self.atoms.copy(), route,
            surrogate=label if self.section == "surrogate" else None,
            engine=label if self.section == "reference" else None,
            reason="md",
            metadata={"context": ctx.as_dict(), "accepted": True,
                      "checked": False},
            driving=label, label_id=label_id)
        self.event_log.append_once(
            f"evaluation:{self.run_id}:{evaluation_id}", EVALUATION_COMMITTED,
            {"run_id": self.run_id, "context": ctx.as_dict(), "route": route,
             "checked": False, "verification": None})

    def _fail(self, error: Exception, evaluation_id: int) -> None:
        self.event_log.append(TASK, {
            "task_id": self._new_task_id(), "attempt": 1,
            "operation": ("reference" if self.section == "reference"
                          else "inference"),
            "purpose": "md", "status": "failed",
            "started_unix": time.time(), "elapsed_s": 0.0,
            "cpu_cores": None, "gpu": None, "queue_s": None,
            "source": "workflow", "evaluation_id": evaluation_id,
            "label_id": None, "cache_hit": False, "error": repr(error)})
        self.event_log.append(RUN_END, {"run_id": self.run_id,
                                        "status": "failed",
                                        "reason": repr(error)})

    def run(self, n_steps: int, outputs: RunOutputs, *, verbose: bool) -> dict:
        interval = outputs.summary_interval
        self._emit_run_start()
        run_start = time.perf_counter()
        completed = 0
        try:
            self._evaluate(0)
            self._record_evaluation(0)
            outputs.regenerate_trajectory()
            for step in range(1, n_steps + 1):
                if self._stop_requested:
                    break
                # ASE's VelocityVerlet evaluates the new-step forces inside
                # step() and returns them: exactly one evaluation per step.
                forces = self.dyn.step(self.atoms.calc.results["forces"])
                self.dyn.nsteps = step
                del forces  # the committed record reads the calculator cache
                self._evaluate(step)
                self._record_evaluation(step)
                self.event_log.append_once(
                    f"step:{self.run_id}:{step - 1}", STEP_COMPLETED,
                    {"run_id": self.run_id, "step_id": step - 1,
                     "physical_time_fs": step * self.config.dynamics.timestep_fs})
                completed = step
                outputs.append_trajectory_step(step)
                if step % interval == 0:
                    outputs.write_summaries()
                    if verbose:
                        print(f"  step {step}/{n_steps} "
                              f"(t = {step * self.config.dynamics.timestep_fs:.2f} fs)",
                              flush=True)
        except Exception as error:
            self._fail(error, completed + 1)
            raise
        stopped = self._stop_requested and completed < n_steps
        wall = time.perf_counter() - run_start
        if stopped:
            self.event_log.append(RUN_END, {
                "run_id": self.run_id, "status": "stopped",
                "reason": "stop requested; stopped at the last complete step"})
        self.event_log.append(RUN_SUMMARY, {
            "run_id": self.run_id, "n_steps": completed,
            "n_evaluations": completed + 1, "n_accepted": completed + 1,
            "n_reference": (completed + 1 if self.section == "reference" else 0),
            "wall_time_s": wall, "stopped_early": stopped})
        return {"completed": completed, "stopped": stopped,
                "wall_time_s": wall}


def _run_plain(config: PyramidConfig, atoms: Atoms, run_dir: Path, *,
               verbose: bool, handle_sigint: bool) -> WorkflowResult:
    backend = (create_configured_backend("reference", config.reference,
                                         run_dir=run_dir)
               if config.task.mode == "reference"
               else create_configured_backend("surrogate", config.surrogate,
                                              run_dir=run_dir))
    prepare_run_directory(
        config, engine=backend if config.task.mode == "reference" else None,
        surrogate=backend if config.task.mode == "surrogate" else None)
    try:
        driver = _PlainDriver(config, atoms, backend, run_dir)
    except Exception:
        _remove_fresh_event_log(run_dir)
        raise
    outputs = RunOutputs(run_dir, config.run.id,
                         trajectory_interval_steps=config.output.trajectory_interval_steps,
                         summary_interval_steps=config.output.summary_interval_steps)
    if verbose:
        print(f"pyramid run: {config.run.id} — {config.task.mode}-only NVE, "
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


def run_workflow(config: PyramidConfig, *, verbose: bool = True,
                 handle_sigint: bool = True) -> WorkflowResult:
    """Validate, set up the run directory and execute the configured task."""
    if config.task.kind != "md":
        raise WorkflowError(
            f"task.kind {config.task.kind!r} is not implemented in this "
            "version (planned for WP07); nothing is run in its place — "
            "task.kind = 'md' is the supported workflow today")
    validate_setup(config)  # dry pass: identical failures as `validate`
    atoms = load_structure(config)
    check_run_directory_available(config)
    run_dir = config.run.directory
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.task.mode == "adaptive":
        return _run_adaptive(config, atoms, run_dir,
                             verbose=verbose, handle_sigint=handle_sigint)
    return _run_plain(config, atoms, run_dir,
                      verbose=verbose, handle_sigint=handle_sigint)


def resume_workflow(run_dir: str | Path, extra_steps: int, *,
                    force_unlock: bool = False, verbose: bool = True,
                    handle_sigint: bool = True) -> WorkflowResult:
    """Continue an adaptive run for ``extra_steps`` additional steps.

    Settings come from the run's ``resolved_config.json`` — the same physics,
    model chain and check stream; ``--steps`` is always *additional* steps
    and the current/target step numbers are printed up front.
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
    if config.task.mode != "adaptive":
        raise WorkflowError(
            f"this run used task.mode {config.task.mode!r}; resume is "
            "implemented for adaptive runs (complete-step checkpoints) — "
            "plain reference/surrogate resume arrives with WP07; use "
            "`pyramid export` to extract what this run produced")
    current = _complete_steps(run_dir, config.run.id)
    target = current + extra_steps
    if verbose:
        print(f"resume: run {config.run.id} is at complete step {current}; "
              f"running {extra_steps} additional steps (target {target})")
    engine, surrogate = build_backends(config, run_dir=run_dir)
    with _SigintGuard(handle_sigint):
        runner = EnergeticRunner.resume(
            run_dir, surrogate, engine,
            checkpoint_interval_steps=config.checkpoint.interval_steps,
            handle_sigint=handle_sigint, event_log_force=force_unlock)
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
