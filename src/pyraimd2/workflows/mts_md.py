"""Fixed-model symmetric-MTS (respa) NVE driver — task.mode 'mts'.

One committed record per COMPLETE outer boundary: the kernel
(:func:`pyraimd2.loop.mts.run_mts`, the reviewed fixed algorithm)
advances one outer step per segment, the driver commits the boundary
(store row + commit event + checkpoint) before starting the next.  The
outer-endpoint reference/fast labels carry into the next segment
(``initial_labels``), so a continuous run and a segmented/resumed run
accumulate exactly the same backend calls — no duplicated initial
evaluation, ever.

Scope (enforced before the first evaluation): fixed model, fixed
cell/composition, no constraints, NVE only; the structure must carry
momenta (this mode never thermalizes).  There is no single "driving
force" for an outer step — store rows carry the reference and surrogate
boundary labels separately and keep ``driving`` absent.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ase import Atoms

from pyraimd2 import __version__
from pyraimd2.config import PyramidConfig
from pyraimd2.engines.base import EngineResult
from pyraimd2.loop.mts import MtsLabels, run_mts
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.runtime.context import EvaluationContext, EvaluationPhase
from pyraimd2.runtime.events import (
    ATTEMPT_LEDGER_PHYSICAL_V1,
    EVALUATION_COMMITTED,
    EVENT_SCHEMA_VERSION,
    RUN_END,
    RUN_START,
    RUN_SUMMARY,
    STEP_COMPLETED,
    EventLog,
)
from pyraimd2.runtime.identity import fingerprint_of, model_id_for
from pyraimd2.store import STORE_SCHEMA_VERSION, Store
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.workflows.setup import WorkflowError

MTS_DRIVER_ID = "mts-nve-respa"
MTS_STATE_VERSION = 1


def _labels_of(boundary) -> MtsLabels:
    return MtsLabels(boundary.U_ref_eV, boundary.F_ref_eV_A,
                     boundary.U_fast_eV, boundary.F_fast_eV_A)


def _engine_result(label: MtsLabels, *, wall_time_s: float = 0.0,
                   energy_kind: str = "energy") -> EngineResult:
    return EngineResult(energy=label.U_ref_eV, forces=label.F_ref_eV_A,
                        stress=None, wall_time_s=wall_time_s,
                        energy_kind=energy_kind, force_consistent=True)


def _surrogate_result(label: MtsLabels, *, n_atoms: int,
                      energy_kind: str = "energy") -> SurrogatePrediction:
    return SurrogatePrediction(energy=label.U_fast_eV,
                               forces=label.F_fast_eV_A, stress=None,
                               uncertainty=np.full(n_atoms, np.nan),
                               energy_kind=energy_kind,
                               force_consistent=True)


class MtsDriver:
    """Drive the reviewed kernel one outer step at a time with the run's
    store/event/checkpoint plumbing; resumable from complete boundaries.
    """

    def __init__(self, config: PyramidConfig, atoms: Atoms,
                 reference: object, surrogate: object, run_dir: Path, *,
                 event_log: EventLog | None = None,
                 resume_state: dict | None = None,
                 budget_t0: float | None = None) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_id = config.run.id
        self.reference = reference
        self.surrogate = surrogate
        self.event_log = (event_log if event_log is not None
                          else EventLog(run_dir))
        self.store = Store(self.run_dir / "trajectory.db")
        self._checkpoints = CheckpointManager(self.run_dir)
        self._stop_requested = False
        self._budget_t0 = (budget_t0 if budget_t0 is not None
                           else time.perf_counter())
        self.h_fs = float(config.dynamics.timestep_fs)
        self.m = int(config.dynamics.outer_ratio)
        self.model_id = model_id_for(surrogate, 0)
        self.engine_fingerprint = fingerprint_of(reference)
        self._kinds = {"reference": "energy", "surrogate": "energy"}
        self._evaluation_counter = 0

        if resume_state is None and "momenta" not in atoms.arrays:
            raise WorkflowError(
                "task.mode 'mts' needs a structure WITH momenta (this "
                "fixed-model NVE mode never thermalizes; give velocities "
                "in the structure file)")
        if resume_state is None:
            self.atoms = atoms.copy()
            self.atoms.calc = None
            self.outer_done = 0
            self.inner_done = 0
            self._task_counter = 0
            self._carried: MtsLabels | None = None
        else:
            arrays = resume_state["arrays"]
            state = resume_state["state"]
            self.atoms = Atoms(numbers=arrays["numbers"],
                               positions=arrays["positions"],
                               cell=arrays["cell"], pbc=arrays["pbc"],
                               masses=arrays["masses"])
            self.atoms.set_momenta(arrays["momenta"])
            self.atoms.calc = None
            self.outer_done = int(state["outer_done"])
            self.inner_done = int(state["inner_done"])
            self._task_counter = int(state["task_counter"])
            self._evaluation_counter = int(state["evaluation_counter"])
            labels = state["labels"]
            self._carried = MtsLabels(
                float(labels["U_ref_eV"]), arrays["F_ref"],
                float(labels["U_fast_eV"]), arrays["F_fast"])
            self._kinds = state["energy_kinds"]

    # -- identity surface -------------------------------------------------
    def integrator_record(self) -> dict:
        return {"algorithm": "mts-respa-symmetric-v1",
                "ensemble": "nve", "timestep_fs": self.h_fs,
                "outer_ratio": self.m, "com_convention": "free",
                "algorithm_version": f"pyraimd2-{__version__}"}

    def _emit_run_start(self) -> None:
        self.event_log.append(RUN_START, {
            "run_id": self.run_id,
            "schema_version": STORE_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "attempt_ledger": ATTEMPT_LEDGER_PHYSICAL_V1,
            "software_version": __version__,
            "reference_id": self.engine_fingerprint,
            "model_id": self.model_id,
            "workflow": {"driver": MTS_DRIVER_ID, "mode": "mts",
                         "integrator": self.integrator_record()},
            "streams": None, "policy": None})

    def request_stop(self) -> None:
        self._stop_requested = True

    def close(self) -> None:
        self.event_log.close()
        self.store.close()

    # -- per-boundary commits ----------------------------------------------
    def _record_boundary(self, boundary, *, initial: bool = False) -> None:
        """Commit one outer boundary: store row (both labels, driving
        absent) + commit event + step event with inner/outer identity.

        The initial boundary stores at step -1 (the shared initial-row
        convention); outer boundary k stores at its inner step k*m.
        """
        self._evaluation_counter += 1
        evaluation_id = self._evaluation_counter - 1
        inner_step = 0 if initial else self.inner_done
        outer = self.outer_done
        ctx = EvaluationContext(
            run_id=self.run_id, step_id=(-1 if initial else inner_step),
            evaluation_id=evaluation_id,
            phase=(EvaluationPhase.INITIAL if initial
                   else EvaluationPhase.MD_STEP),
            physical_time_fs=inner_step * self.h_fs,
            model_id=self.model_id)
        labels = _labels_of(boundary)
        label_id = f"{self.run_id}-label-{evaluation_id}"
        frame = self.atoms.copy()
        frame.calc = None
        row_id = self.store.append(
            self.run_id, ctx.step_id, frame, "mts",
            surrogate=_surrogate_result(labels, n_atoms=len(frame)),
            engine=_engine_result(labels),
            reason="mts_outer_boundary",
            metadata={"context": ctx.as_dict(), "accepted": True,
                      "checked": False, "constraint": None,
                      "method": "mts_respa",
                      "mts": {"outer_index": 0 if initial else outer,
                              "inner_step": inner_step,
                              "time_fs": inner_step * self.h_fs,
                              "inner_timestep_fs": self.h_fs,
                              "outer_ratio": self.m}},
            driving=None,  # MTS has no single driving force
            label_id=label_id)
        self.event_log.append_once(
            f"evaluation:{self.run_id}:{evaluation_id}",
            EVALUATION_COMMITTED,
            {"run_id": self.run_id, "context": ctx.as_dict(),
             "route": "mts", "checked": False, "verification": None,
             "row_id": int(row_id),
             "row_digest": self.store.row_digest(
                 self.store.row_by_id(int(row_id)))})
        if initial:
            return  # the initial boundary commits as the initial
            # evaluation only (same convention as the plain driver)
        self.event_log.append_once(
            f"step:{self.run_id}:{inner_step}", STEP_COMPLETED, {
                "run_id": self.run_id, "step_id": inner_step,
                "outer_step": outer,
                "physical_time_fs": inner_step * self.h_fs,
                "integrator": self.integrator_record(),
                "digest_format": "boundary-v2",
                "state_digest": None,
                "boundary_digest": None,
                "mts": {"outer_index": outer, "inner_step": inner_step}})

    def _write_checkpoint(self) -> int:
        generation = self._checkpoints.next_generation()
        state = {
            "run_id": self.run_id, "driver": MTS_DRIVER_ID,
            "mts_state_version": MTS_STATE_VERSION,
            "inner_timestep_fs": self.h_fs, "outer_ratio": self.m,
            "outer_done": self.outer_done, "inner_done": self.inner_done,
            "physical_time_fs": self.inner_done * self.h_fs,
            "task_counter": self._task_counter,
            "evaluation_counter": self._evaluation_counter,
            "model_id": self.model_id,
            "engine_fingerprint": self.engine_fingerprint,
            "integrator": self.integrator_record(),
            "energy_kinds": self._kinds,
            "labels": {"U_ref_eV": self._carried.U_ref_eV,
                       "U_fast_eV": self._carried.U_fast_eV},
        }
        arrays = {
            "numbers": self.atoms.numbers,
            "cell": self.atoms.cell.array,
            "pbc": np.asarray(self.atoms.pbc),
            "masses": self.atoms.get_masses(),
            "positions": self.atoms.positions.copy(),
            "momenta": self.atoms.get_momenta(),
            "F_ref": np.asarray(self._carried.F_ref_eV_A, dtype=float),
            "F_fast": np.asarray(self._carried.F_fast_eV_A, dtype=float),
        }
        manifest_extra = {
            "run_id": self.run_id, "nsteps": self.inner_done,
            "outer_steps": self.outer_done,
            "physical_time_fs": self.inner_done * self.h_fs,
            "last_event_seq": self.event_log.last_seq,
            "store_schema_version": STORE_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "attempt_ledger": ATTEMPT_LEDGER_PHYSICAL_V1,
            "software_version": __version__,
        }
        self._checkpoints.write(generation, state, arrays, manifest_extra)
        return generation

    # -- main loop -----------------------------------------------------------
    def run(self, n_outer: int, outputs, *, verbose: bool) -> dict:
        interval = outputs.summary_interval
        checkpoint_interval = max(self.m,
                                  self.config.checkpoint.interval_steps)
        budget_s = (None if self.config.dynamics.max_wall_hours is None
                    else self.config.dynamics.max_wall_hours * 3600.0)
        fresh = self.outer_done == 0 and self.inner_done == 0 \
            and self._carried is None
        if fresh:
            self._emit_run_start()
        run_start = time.perf_counter()
        last_segment_wall: float | None = None
        stopped = False
        try:
            for _ in range(n_outer):
                if self._stop_requested:
                    stopped = True
                    break
                if budget_s is not None and last_segment_wall is not None:
                    # soft decision at the complete outer boundary only:
                    # remaining budget vs the reserve for one more outer
                    # step (a running step is never interrupted)
                    remaining = budget_s - (time.perf_counter()
                                            - self._budget_t0)
                    if remaining < 2.0 * last_segment_wall + 5.0:
                        self._write_checkpoint()
                        stopped = True
                        break
                segment_t0 = time.perf_counter()
                view = self.atoms.copy()
                view.calc = None
                result = run_mts(
                    view, self.reference, self.surrogate,
                    inner_timestep_fs=self.h_fs, outer_ratio=self.m,
                    n_outer_steps=1, initial_labels=self._carried,
                    run_id=self.run_id,
                    task_counter_start=self._task_counter,
                    event_log=self.event_log, emit_run_start=False)
                self._task_counter = result.task_counter_end
                if fresh and self.outer_done == 0:
                    # the fresh run's first segment evaluated the initial
                    # state: commit it as the initial evaluation (step -1)
                    self._record_boundary(result.boundaries[0],
                                          initial=True)
                endpoint = result.boundaries[-1]
                self.outer_done += 1
                self.inner_done += self.m
                self.atoms.positions = np.array(result.final_positions_A)
                self.atoms.set_momenta(result.final_momenta_ase)
                self._carried = result.final_labels
                self._record_boundary(endpoint)
                if self.inner_done % checkpoint_interval == 0:
                    self._write_checkpoint()
                if self.inner_done % max(self.m, interval) == 0:
                    outputs.write_summaries()
                last_segment_wall = time.perf_counter() - segment_t0
                if verbose:
                    print(f"  outer {self.outer_done} "
                          f"(t = {self.inner_done * self.h_fs:.2f} fs)",
                          flush=True)
        except Exception as error:
            self.event_log.append(RUN_END, {
                "run_id": self.run_id, "status": "failed",
                "reason": repr(error)})
            raise
        wall = time.perf_counter() - run_start
        if stopped and not self._stop_requested:
            self.event_log.append(RUN_END, {
                "run_id": self.run_id, "status": "stopped",
                "reason": "walltime budget exhausted; checkpoint saved at "
                          "the last complete outer boundary"})
        if stopped and self._stop_requested:
            self.event_log.append(RUN_END, {
                "run_id": self.run_id, "status": "stopped",
                "reason": "stop requested; checkpoint saved at the last "
                          "complete outer boundary"})
        self.event_log.append(RUN_SUMMARY, {
            "run_id": self.run_id, "n_steps": self.inner_done,
            "n_evaluations": self._evaluation_counter,
            "n_accepted": self._evaluation_counter,
            "n_reference": self.outer_done + (0 if self._carried else 1),
            "wall_time_s": wall, "stopped_early": stopped})
        return {"completed": self.inner_done, "stopped": stopped,
                "wall_time_s": wall}


# ---------------------------------------------------------------------------
# workflow entry points (deferred md imports: md.py owns the dispatchers)

def _run_mts(config: PyramidConfig, atoms: Atoms, run_dir: Path, *,
             verbose: bool, handle_sigint: bool):
    """Fresh fixed-model MTS run (task.mode 'mts'): build backends, set up
    the run directory, drive outer steps with per-boundary commits."""
    import signal

    from pyraimd2.engines.ase_resources import file_resource_baseline_sha256
    from pyraimd2.workflows.md import (
        WorkflowResult,
        _complete_steps,
        _finalize_quietly,
        _remove_fresh_event_log,
    )
    from pyraimd2.workflows.setup import (
        RunOutputs,
        build_backends,
        prepare_run_directory,
    )

    budget_t0 = time.perf_counter()
    event_log = EventLog(run_dir)
    driver = None
    try:
        engine, surrogate = build_backends(config, run_dir=run_dir,
                                           event_log=event_log)
        prepare_run_directory(config, engine=engine, surrogate=surrogate)
        driver = MtsDriver(
            config, atoms, engine, surrogate, run_dir, event_log=event_log,
            budget_t0=budget_t0)
        driver._file_resource_baseline_sha256 = \
            file_resource_baseline_sha256(run_dir)
        outputs = RunOutputs(
            run_dir, config.run.id,
            trajectory_interval_steps=config.output.trajectory_interval_steps,
            summary_interval_steps=config.output.summary_interval_steps)
    except Exception:
        if driver is not None:
            driver.close()
        event_log.close()
        _remove_fresh_event_log(run_dir)
        raise
    if verbose:
        print(f"pyramid run: {config.run.id} — mts NVE (respa), "
              f"{config.dynamics.steps} inner steps x "
              f"{config.dynamics.timestep_fs} fs, outer ratio "
              f"{config.dynamics.outer_ratio}")
        print(f"run directory: {run_dir}")
    previous = signal.getsignal(signal.SIGINT) if handle_sigint else None
    if handle_sigint:
        signal.signal(signal.SIGINT,
                      lambda signum, frame: driver.request_stop())
    try:
        outcome = driver.run(config.dynamics.steps
                             // config.dynamics.outer_ratio,
                             outputs, verbose=verbose)
    finally:
        _finalize_quietly(outputs)
        driver.close()
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    steps = _complete_steps(run_dir, config.run.id)
    if verbose:
        status = ("stopped early at the last complete outer boundary"
                  if outcome["stopped"] else "completed")
        print(f"run {status}: {steps} complete inner steps "
              f"({driver.outer_done} outer boundaries, wall time "
              f"{outcome['wall_time_s']:.2f} s)")
        print(f"  outputs: {run_dir}/summary.json, summary.csv, "
              "trajectory.extxyz")
    return WorkflowResult(run_dir=run_dir, run_id=config.run.id,
                          mode=config.task.mode, steps_completed=steps,
                          steps_this_call=outcome["completed"],
                          stopped_early=outcome["stopped"], summary=None)


def _resume_mts(config: PyramidConfig, run_dir: Path, extra_steps: int, *,
                force_unlock: bool, verbose: bool, handle_sigint: bool):
    """Resume an MTS run from its last valid complete-outer-boundary
    checkpoint; identity (structure, integrator, reference settings,
    model content) is verified before any evaluation."""
    import signal

    from pyraimd2.workflows.md import WorkflowResult, _complete_steps, _finalize_quietly
    from pyraimd2.workflows.setup import RunOutputs, build_backends

    manager = CheckpointManager(run_dir)
    checkpoint = manager.read_latest_valid()
    if checkpoint is None:
        raise WorkflowError(
            f"no valid checkpoint under {run_dir}; this MTS run has no "
            "complete outer boundary to resume from")
    state, arrays = checkpoint.state, checkpoint.arrays
    if state.get("driver") != MTS_DRIVER_ID:
        raise WorkflowError(
            f"checkpoint under {run_dir} belongs to driver "
            f"{state.get('driver')!r}, not {MTS_DRIVER_ID!r}; resume "
            "requires the same run")
    if int(extra_steps) % int(state["outer_ratio"]) != 0:
        raise WorkflowError(
            f"resume --steps must be a multiple of outer_ratio "
            f"({state['outer_ratio']}) — only complete outer steps exist, "
            f"got {extra_steps}")
    # integration settings are run identity
    if float(state["inner_timestep_fs"]) != \
            float(config.dynamics.timestep_fs) or \
            int(state["outer_ratio"]) != int(config.dynamics.outer_ratio):
        raise WorkflowError(
            "timestep_fs/outer_ratio changed; resume continues the same "
            "integration settings — start a new run")
    event_log = EventLog(run_dir, force=force_unlock)
    budget_t0 = time.perf_counter()
    try:
        reference, surrogate = build_backends(config, run_dir=run_dir,
                                              event_log=event_log)
        # identity BEFORE any evaluation: reference settings + model
        # content (the structure state itself comes from the checkpoint)
        ref_fp = fingerprint_of(reference)
        if ref_fp != state.get("engine_fingerprint"):
            raise WorkflowError(
                "reference backend identity does not match the checkpoint; "
                "resume requires the same physical settings")
        if model_id_for(surrogate, 0) != state.get("model_id"):
            raise WorkflowError(
                "surrogate identity does not match the checkpoint "
                f"({state.get('model_id')!r} vs "
                f"{model_id_for(surrogate, 0)!r}); resume requires the "
                "same model — start a new run, or fork")
    except Exception:
        event_log.close()
        raise
    driver = None
    try:
        # STEP_COMPLETED events count complete OUTER boundaries for MTS
        current = _complete_steps(run_dir, config.run.id)
        if current < int(state["outer_done"]):
            raise WorkflowError(
                f"the trajectory ({current} complete outer boundaries) is "
                f"behind the checkpoint (outer {state['outer_done']}); the "
                "run directory is inconsistent")
        if verbose:
            print(f"resume: run {config.run.id} (mts) is at inner step "
                  f"{current} (outer {state['outer_done']}); running "
                  f"{extra_steps} additional inner steps")
        driver = MtsDriver(config, None, reference, surrogate, run_dir,
                           event_log=event_log,
                           resume_state={"state": state, "arrays": arrays},
                           budget_t0=budget_t0)
        outputs = RunOutputs(
            run_dir, config.run.id,
            trajectory_interval_steps=config.output.trajectory_interval_steps,
            summary_interval_steps=config.output.summary_interval_steps)
    except Exception:
        if driver is not None:
            driver.close()
        event_log.close()
        raise
    previous = signal.getsignal(signal.SIGINT) if handle_sigint else None
    if handle_sigint:
        signal.signal(signal.SIGINT,
                      lambda signum, frame: driver.request_stop())
    try:
        outcome = driver.run(extra_steps // int(state["outer_ratio"]),
                             outputs, verbose=verbose)
    finally:
        _finalize_quietly(outputs)
        driver.close()
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    steps = _complete_steps(run_dir, config.run.id)
    if verbose:
        print(f"resume completed: {steps} complete inner steps")
    return WorkflowResult(run_dir=run_dir, run_id=config.run.id,
                          mode=config.task.mode, steps_completed=steps,
                          steps_this_call=outcome["completed"],
                          stopped_early=outcome["stopped"], summary=None)
