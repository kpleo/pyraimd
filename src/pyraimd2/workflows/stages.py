"""Serial material recipes: an authoritative completed-state reader and a
small stage controller (M4).

``load_completed_state`` returns the last genuinely complete boundary of a
run plus its provenance — the one entry point the ``relax → NVT → NVE``
chain builds on, reusing the committed-row and complete-step machinery
(never a raw last-row copy, never a live backend call).

``run_serial_recipe`` drives the three stages through the existing
``run_workflow``/``resume_workflow`` entry points with per-stage run
identities, a short ``workflow.json`` manifest, stage dedup, and MD
continuation through the ordinary resume protocol.  It is a controller,
not a second event/checkpoint system.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from ase import Atoms

from pyraimd2.config import PyramidConfig, load_config, load_resolved_config
from pyraimd2.loop.integrators import state_digest
from pyraimd2.runtime.events import RUN_END, RUN_SUMMARY, STEP_COMPLETED
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.workflows.md import (
    _check_boundary_record,
    _spec_from_dynamics,
    resume_workflow,
    run_workflow,
)
from pyraimd2.workflows.setup import WorkflowError

RECIPE_SCHEMA = "serial-recipe-v1"


@dataclass(frozen=True)
class CompletedState:
    """The authoritative completed boundary of a run and where it came from."""

    atoms: Atoms  # physical arrays only — no calculator, no derived results
    provenance: dict


def _clean_atoms(frame: Atoms, constraint_indices) -> Atoms:
    """Physical identity only: numbers/order, unwrapped positions, cell/pbc,
    masses, momenta when present, initial charges/magmoms and the FixAtoms
    set — no calculator, no cached forces/energy, no run info fields."""
    atoms = Atoms(numbers=frame.numbers.copy(),
                  positions=np.array(frame.positions, dtype=float),
                  cell=frame.cell.array.copy(), pbc=np.asarray(frame.pbc))
    atoms.set_masses(frame.get_masses())
    atoms.set_initial_charges(frame.get_initial_charges())
    atoms.set_initial_magnetic_moments(frame.get_initial_magnetic_moments())
    if "momenta" in frame.arrays:
        atoms.set_momenta(np.array(frame.get_momenta(), dtype=float))
    if constraint_indices:
        from ase.constraints import FixAtoms

        atoms.set_constraint(FixAtoms(indices=list(constraint_indices)))
    return atoms


def _md_completed_state(run_dir: Path, config: PyramidConfig,
                        events: list[dict], *,
                        require_finished: bool) -> CompletedState:
    """The last true STEP_COMPLETED boundary of an MD run (plain or
    adaptive), verified against its committed row by the record's own
    format semantics."""
    steps = sorted(int(e["step_id"]) for e in events
                   if e.get("type") == STEP_COMPLETED)
    target = int(config.dynamics.steps)
    finished = len(steps) >= target
    if require_finished:
        failed = next((e for e in reversed(events) if e.get("type") == RUN_END
                       and e.get("status") == "failed"), None)
        if failed is not None:
            raise WorkflowError(
                f"{run_dir} ended failed ({failed.get('reason', '')[:120]}); "
                "fix or fork the run instead of chaining from it")
        if not finished:
            raise WorkflowError(
                f"{run_dir} completed {len(steps)} of {target} configured "
                "steps; a stage chain only continues from a stage that "
                "finished as planned (pass require_finished=False to use "
                "the last complete boundary deliberately)")
    if not steps:
        raise WorkflowError(
            f"{run_dir} has no complete MD step boundary; a tail evaluation "
            "without its step is never a next-stage initial state")
    step_id = steps[-1]
    step_event = next(e for e in reversed(events)
                      if e.get("type") == STEP_COMPLETED
                      and int(e["step_id"]) == step_id)
    with Store(run_dir / "trajectory.db") as store:
        commit = next(
            (e for e in reversed(events) if e.get("type")
             == "evaluation_committed"
             and int((e.get("context") or {})["evaluation_id"])
             == step_id + 1), None)
        row = store.committed_row(events, config.run.id, step_id + 1)
        frame = store.complete_step_frame(
            row, Store.row_timestep_fs(row) or 0.0, commit=commit)
        if step_event.get("state_digest"):
            _check_boundary_record(row, step_event, commit,
                                   _spec_from_dynamics(config), frame=frame)
    metadata = row.data.get("metadata") or {}
    constraint = metadata.get("constraint") or {}
    atoms = _clean_atoms(frame, constraint.get("indices") or [])
    start = next((e for e in events if e.get("type") == "run_start"), None)
    provenance = {
        "source_run_id": config.run.id,
        "source_run_dir": str(run_dir),
        "task_kind": "md",
        "mode": config.task.mode,
        "ensemble": config.dynamics.ensemble,
        "integrator": config.dynamics.integrator,
        "boundary_step_id": step_id,
        "evaluation_id": step_id + 1,
        "physical_time_fs": float(step_event["physical_time_fs"]),
        "finished": finished,
        "configured_steps": target,
        "record_digest": step_event.get("boundary_digest")
        or step_event.get("state_digest"),
        "momenta_source": frame.info.get("momenta_source"),
        "reference_id": (start or {}).get("reference_id"),
        "model_id": (start or {}).get("model_id"),
    }
    return CompletedState(atoms, provenance)


def _relax_completed_state(run_dir: Path, config: PyramidConfig,
                           events: list[dict], *,
                           require_converged: bool) -> CompletedState:
    """The committed terminal state of a fixed-model geometry optimization.

    Relax has no physical MD time and no complete-step boundary; the
    terminal committed frame is the record.  Optimizer iterations are not
    MD evaluations and are reported as such in the provenance.
    """
    summary = next((e for e in reversed(events)
                    if e.get("type") == RUN_SUMMARY), None)
    converged = bool((summary or {}).get("converged"))
    if require_converged and not converged:
        raise WorkflowError(
            f"{run_dir} did not converge "
            f"(final fmax {(summary or {}).get('final_fmax_eV_A')}); the "
            "chain stops here instead of thermalizing an unconverged "
            "structure (pass require_converged=False to override "
            "deliberately)")
    commits = [e for e in events if e.get("type") == "evaluation_committed"]
    if not commits:
        raise WorkflowError(f"{run_dir} has no committed optimization state")
    last = commits[-1]
    with Store(run_dir / "trajectory.db") as store:
        row = store.committed_row(
            events, config.run.id,
            int((last.get("context") or {})["evaluation_id"]))
        frame = store.complete_step_frame(row, 0.0, commit=last)
    metadata = row.data.get("metadata") or {}
    constraint = metadata.get("constraint") or {}
    atoms = _clean_atoms(frame, constraint.get("indices") or [])
    start = next((e for e in events if e.get("type") == "run_start"), None)
    provenance = {
        "source_run_id": config.run.id,
        "source_run_dir": str(run_dir),
        "task_kind": "relax",
        "mode": config.task.mode,
        "optimizer": config.relax.optimizer,
        "optimizer_steps": int((summary or {}).get("n_steps", 0)),
        "converged": converged,
        "final_fmax_eV_A": (summary or {}).get("final_fmax_eV_A"),
        "evaluation_id": int((last.get("context") or {})["evaluation_id"]),
        "record_digest": last.get("row_digest"),
        "momenta_source": None,
        "reference_id": (start or {}).get("reference_id"),
        "model_id": (start or {}).get("model_id"),
    }
    return CompletedState(atoms, provenance)


def load_completed_state(run_dir: str | Path, *,
                         require_finished: bool = True) -> CompletedState:
    """The authoritative completed state of a run directory and its
    provenance — the one entry point for stage handoff.

    MD selects the last true STEP_COMPLETED boundary (verified against the
    committed row by the record's own digest format); a tail force
    evaluation without its step is never selected, and a failed run
    refuses.  Relax selects the committed terminal state and requires
    convergence by default.  No live backend is touched: the next stage
    re-evaluates its initial driving force under its own billing.
    """
    run_dir = Path(run_dir)
    config = load_resolved_config(run_dir)
    events = [json.loads(line)
              for line in (run_dir / "events.jsonl").read_text().splitlines()]
    if config.task.kind == "relax":
        return _relax_completed_state(run_dir, config, events,
                                      require_converged=require_finished)
    if config.task.kind == "md":
        return _md_completed_state(run_dir, config, events,
                                   require_finished=require_finished)
    raise WorkflowError(
        f"load_completed_state supports relax and md runs, got "
        f"task.kind {config.task.kind!r}")


# --- the serial recipe controller ----------------------------------------------


@dataclass(frozen=True)
class RecipeStage:
    """One stage of a serial recipe: a normal config file whose run
    directory is ``<root>/<name>`` and whose structure file the controller
    materializes from the previous stage's completed state."""

    name: str
    config_path: Path
    momenta: str | None = None  # None (non-MD) | "initialize" | "preserve"


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _state_file_digest(atoms: Atoms) -> str:
    arrays = [np.asarray(atoms.numbers), np.asarray(atoms.positions, float),
              np.asarray(atoms.get_masses()),
              np.asarray(atoms.get_momenta(), dtype=float)
              if "momenta" in atoms.arrays else np.zeros(0)]
    return state_digest(*arrays)


def _materialize_initial(stage_dir: Path, atoms: Atoms) -> Path:
    """Write the stage's initial structure as an ASE .traj — full double
    precision for positions, masses, momenta and the FixAtoms set (verified
    by the recipe's own round-trip checks); never a lossy text coordinate
    file."""
    path = stage_dir / "initial.traj"
    if not path.exists():
        from ase.io import write as ase_write

        stage_dir.mkdir(parents=True, exist_ok=True)
        ase_write(path, atoms, format="traj")
    return path


def run_serial_recipe(root: str | Path, stages: list[RecipeStage], *,
                      verbose: bool = True) -> dict:
    """Run the serial recipe (e.g. relax → NVT → NVE), idempotently.

    Each stage owns an independent run id, config and cost ledger under
    ``root/<name>``.  A finished stage is never recomputed; a partially
    completed MD stage continues through the ordinary resume protocol to
    its configured total (never stacked with extra steps); a stage whose
    config or source state changed after it started refuses before any new
    computation.  Source runs are read-only.  Returns the manifest dict
    (also written to ``root/workflow.json``).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "workflow.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != RECIPE_SCHEMA:
            raise WorkflowError(
                f"{manifest_path} has schema {manifest.get('schema')!r}, "
                f"expected {RECIPE_SCHEMA!r}")
    else:
        manifest = {"schema": RECIPE_SCHEMA, "created_unix": time.time(),
                    "stages": []}

    previous: CompletedState | None = None
    for stage in stages:
        stage_dir = root / stage.name
        config = load_config(stage.config_path)
        if config.run.directory.resolve() != stage_dir.resolve():
            raise WorkflowError(
                f"stage {stage.name!r}: the config's run.directory must be "
                f"{stage_dir} (got {config.run.directory}); each stage owns "
                "its directory under the recipe root")
        record = next((s for s in manifest["stages"]
                       if s["name"] == stage.name), None)
        if record is None:
            record = {"name": stage.name, "kind": config.task.kind,
                      "run_id": config.run.id, "run_dir": stage.name,
                      "status": "pending", "config_sha256": None,
                      "source": None, "momenta": stage.momenta,
                      "wall_time_s": 0.0, "result": None}
            manifest["stages"].append(record)
        config_text = Path(stage.config_path).read_bytes()
        config_sha = sha256(config_text).hexdigest()
        if record["config_sha256"] is None:
            record["config_sha256"] = config_sha
        elif record["config_sha256"] != config_sha:
            raise WorkflowError(
                f"stage {stage.name!r} already started with a different "
                "config; refusing to continue under changed settings — "
                "use a new recipe root")
        if record["status"] == "done":
            previous = load_completed_state(stage_dir)
            continue
        if record["status"] == "failed":
            raise WorkflowError(
                f"stage {stage.name!r} previously failed; fix the stage and "
                "restart from a clean recipe root — the recipe does not "
                "silently rerun it")

        if config.task.kind == "md":
            if previous is None:
                raise WorkflowError(
                    f"stage {stage.name!r}: an MD stage needs a previous "
                    "stage's completed state")
            _prepare_md_stage(record, stage, config, stage_dir, previous)
        record["status"] = "running"  # before the first compute: a stage
        # left "running" is resumed, never freshly restarted
        started = time.perf_counter()
        try:
            if config.task.kind == "md" and _md_partial(stage_dir, config):
                completed = _md_complete_steps(stage_dir)
                target = int(config.dynamics.steps)
                # The exact step conversion: resume adds `extra` NEW steps
                # and first binds a committed tail evaluation (at most one)
                # as its step record — never stacking the heal as an extra
                # step.
                healable = _md_healable(stage_dir, completed)
                extra = target - completed - healable
                if verbose:
                    print(f"recipe {stage.name}: resume {completed} -> "
                          f"{target} steps"
                          + (" (binding one committed tail step)"
                             if healable else ""))
                # A crashed stage leaves its writer lock behind; the serial
                # controller reclaims it deliberately when resuming.
                resume_workflow(stage_dir, extra, verbose=verbose,
                                handle_sigint=False, force_unlock=True)
            else:
                if verbose:
                    print(f"recipe {stage.name}: running "
                          f"{config.task.kind} ({config.task.mode})")
                run_workflow(config, verbose=verbose, handle_sigint=False)
            state = load_completed_state(stage_dir)
            # an unconverged relax or an unfinished MD stage raises here —
            # the chain stops instead of continuing from a bad state
        except Exception:
            # MD stages recover through the ordinary resume protocol on the
            # next invocation (status stays "running"); relax stages have
            # no resume — a failure marks them failed and stops the chain.
            record["status"] = "running" if config.task.kind == "md" \
                else "failed"
            record["wall_time_s"] += time.perf_counter() - started
            _write_json_atomic(manifest_path, manifest)
            raise
        record["wall_time_s"] += time.perf_counter() - started
        record["status"] = "done"
        record["result"] = _stage_result(stage_dir, config, state)
        _write_json_atomic(manifest_path, manifest)
        previous = state
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _prepare_md_stage(record: dict, stage: RecipeStage,
                      config: PyramidConfig, stage_dir: Path,
                      previous: CompletedState) -> None:
    """Materialize the stage's initial state once, with the declared
    momentum policy, and record the source provenance."""
    if record["status"] != "pending":
        return  # a started stage already has its initial state; resume it
    momenta_policy = stage.momenta
    has_momenta = "momenta" in previous.atoms.arrays
    if momenta_policy == "preserve" and not has_momenta:
        raise WorkflowError(
            f"stage {stage.name!r}: momenta = 'preserve' but the source "
            "state carries none — refusing a silent zero-temperature start")
    if momenta_policy == "initialize" and has_momenta:
        raise WorkflowError(
            f"stage {stage.name!r}: momenta = 'initialize' but the source "
            "already has momenta — refusing to overwrite them; use "
            "'preserve'")
    # "initialize" means: hand the structure over WITHOUT momenta and let
    # the stage's own driver thermalize once at its configured temperature
    # and velocity seed — recorded here, and in the run's RUN_START streams.
    atoms = previous.atoms
    if momenta_policy == "initialize" and "momenta" in atoms.arrays:
        atoms = atoms.copy()
        del atoms.arrays["momenta"]
    initial = _materialize_initial(stage_dir, atoms)
    configured = Path(config.structure.file)
    resolved_structure = configured if configured.is_absolute() else \
        (Path(stage.config_path).parent / configured).resolve()
    if resolved_structure != initial.resolve():
        raise WorkflowError(
            f"stage {stage.name!r}: the config's structure.file must be "
            f"{initial.name} in the stage directory — the controller owns "
            "the materialized initial state")
    record["source"] = dict(previous.provenance)
    record["source"]["state_digest"] = _state_file_digest(previous.atoms)
    if momenta_policy == "initialize":
        record["source"]["momenta_initialization"] = {
            "policy": "initialize",
            "temperature_K": config.dynamics.temperature_K,
            "velocity_seed": config.dynamics.velocity_seed,
        }
    elif momenta_policy == "preserve":
        record["source"]["momenta_initialization"] = {"policy": "preserve"}


def _md_complete_steps(stage_dir: Path) -> int:
    events_path = stage_dir / "events.jsonl"
    if not events_path.exists():
        return 0
    return sum(1 for line in events_path.read_text().splitlines()
               if '"step_completed"' in line)


def _md_healable(stage_dir: Path, completed: int) -> int:
    """1 when the stage has exactly one committed tail evaluation without
    its step record (the plain driver's healable window); else 0."""
    events_path = stage_dir / "events.jsonl"
    if not events_path.exists():
        return 0
    committed = sum(1 for line in events_path.read_text().splitlines()
                    if '"evaluation_committed"' in line)
    return 1 if committed == completed + 2 else 0


def _md_partial(stage_dir: Path, config: PyramidConfig) -> bool:
    """A started-but-incomplete MD stage (an existing event log with fewer
    complete steps than configured) resumes; anything else runs fresh."""
    if config.task.kind != "md":
        return False
    completed = _md_complete_steps(stage_dir)
    return 0 < completed < int(config.dynamics.steps)


def _stage_result(stage_dir: Path, config: PyramidConfig,
                  state: CompletedState) -> dict:
    """Per-stage status for the manifest: completion, steps and physical
    time, identity, and the purpose-split costs from the task ledger."""
    status = inspect_run(stage_dir, run_id=config.run.id)
    cost = status["cost"]
    result = {
        "status": "done",
        "run_id": config.run.id,
        "kind": config.task.kind,
        "mode": config.task.mode,
        "reference_executions": cost["reference"]["actual_executions"],
        "reference_failed": cost["reference"]["failed_attempts"],
        "inference_executions": cost["counts"]["inference"],
        "training_callbacks": cost["counts"]["training"],
        "io_tasks": cost["counts"]["io"],
        "provenance": state.provenance,
    }
    by_purpose = {}
    for bucket in cost.get("by", []):
        if bucket["operation"] == "reference":
            by_purpose[bucket["purpose"]] = bucket["count"]
    result["reference_by_purpose"] = by_purpose
    if config.task.kind == "md":
        result["complete_steps"] = int(
            state.provenance["boundary_step_id"]) + 1
        result["physical_time_fs"] = state.provenance["physical_time_fs"]
        result["ensemble"] = config.dynamics.ensemble
    else:
        result["optimizer_steps"] = state.provenance["optimizer_steps"]
        result["converged"] = state.provenance["converged"]
    return result
