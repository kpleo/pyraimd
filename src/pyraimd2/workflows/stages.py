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
from pyraimd2.loop.integrators import IntegratorSpec, state_digest
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


@dataclass(frozen=True)
class _RunIdentity:
    """The minimal run identity the MD reader needs, from a workflow
    resolved config or from a runner-level RUN_START record."""

    run_id: str
    mode: str
    ensemble: str
    integrator: str
    spec: IntegratorSpec
    target_steps: int | None  # None: the configured total is unprovable


def _identity_from_start(start: dict) -> _RunIdentity:
    """Reconstruct the run identity from a runner-level RUN_START (no
    resolved_config.json was written): the integrator block is authoritative
    either way; the configured total stays unprovable (None)."""
    workflow = start.get("workflow") or {}
    driver = workflow.get("driver")
    if driver is not None:
        spec = IntegratorSpec(**dict(workflow["integrator"]))
        mode = str(workflow.get("mode"))
    else:
        policy = start.get("policy") or {}
        spec = IntegratorSpec(**dict(policy["integrator"]))
        mode = "adaptive"
    return _RunIdentity(run_id=str(start["run_id"]), mode=mode,
                        ensemble=spec.ensemble, integrator=spec.algorithm,
                        spec=spec, target_steps=None)


def _md_completed_state(run_dir: Path, ident: _RunIdentity,
                        events: list[dict], *,
                        require_finished: bool) -> CompletedState:
    """The last true STEP_COMPLETED boundary of an MD run (plain or
    adaptive), verified against its committed row by the record's own
    format semantics."""
    steps = sorted(int(e["step_id"]) for e in events
                   if e.get("type") == STEP_COMPLETED)
    target = ident.target_steps
    finished = target is not None and len(steps) >= target
    if require_finished:
        if target is None:
            raise WorkflowError(
                f"{run_dir} does not record the configured total steps (a "
                "runner-level run without resolved_config.json); pass "
                "require_finished=False to use the last complete boundary "
                "deliberately")
        # The LATEST terminal outcome decides (R3): a failed RUN_END after
        # the last successful RUN_SUMMARY is an unresolved failure; an
        # earlier failure followed by a successful resume is recovered
        # history, not a permanent block.
        last_failed = -1
        last_summary = -1
        for event in events:
            if event.get("type") == RUN_END and event.get("status") == "failed":
                last_failed = max(last_failed, int(event.get("seq", 0)))
            elif event.get("type") == RUN_SUMMARY:
                last_summary = max(last_summary, int(event.get("seq", 0)))
        if last_failed > last_summary:
            reason = next(e for e in reversed(events)
                          if e.get("type") == RUN_END
                          and e.get("status") == "failed")
            raise WorkflowError(
                f"{run_dir} ended failed ({reason.get('reason', '')[:120]}); "
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
        row = store.committed_row(events, ident.run_id, step_id + 1)
        frame = store.complete_step_frame(
            row, Store.row_timestep_fs(row) or 0.0, commit=commit)
        _check_boundary_record(row, step_event, commit, ident.spec,
                               frame=frame)
    metadata = row.data.get("metadata") or {}
    constraint = metadata.get("constraint") or {}
    atoms = _clean_atoms(frame, constraint.get("indices") or [])
    start = next((e for e in events if e.get("type") == "run_start"), None)
    provenance = {
        "source_run_id": ident.run_id,
        "source_run_dir": str(run_dir),
        "task_kind": "md",
        "mode": ident.mode,
        "ensemble": ident.ensemble,
        "integrator": ident.integrator,
        "boundary_step_id": step_id,
        "evaluation_id": step_id + 1,
        "physical_time_fs": float(step_event["physical_time_fs"]),
        "finished": finished,
        "configured_steps": target,
        "record_digest": step_event.get("boundary_digest")
        or step_event.get("state_digest"),
        "momenta_source": frame.info.get("momenta_source"),
        "reference_id": (start or {}).get("reference_id"),
        # the verified boundary commit's driving model identity (R3) — the
        # RUN_START value is the initial model and stays named separately
        "model_id": ((commit.get("context") or {}).get("model_id")
                     if commit is not None
                     else (start or {}).get("model_id")),
        "initial_model_id": (start or {}).get("model_id"),
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
    events_path = run_dir / "events.jsonl"
    if not (run_dir / "resolved_config.json").is_file():
        # A runner-level run directory (no resolved_config): the identity
        # comes from RUN_START; the configured total stays unprovable.
        if not events_path.is_file():
            raise WorkflowError(
                f"{run_dir} is not a run directory (no resolved_config.json "
                "and no events.jsonl)")
        events = [json.loads(line)
                  for line in events_path.read_text().splitlines()]
        start = next((e for e in events if e.get("type") == "run_start"),
                     None)
        if start is None:
            raise WorkflowError(
                f"{run_dir} has no RUN_START record; cannot determine the "
                "run identity")
        return _md_completed_state(run_dir, _identity_from_start(start),
                                   events,
                                   require_finished=require_finished)
    config = load_resolved_config(run_dir)
    events = [json.loads(line)
              for line in events_path.read_text().splitlines()]
    if config.task.kind == "relax":
        return _relax_completed_state(run_dir, config, events,
                                      require_converged=require_finished)
    if config.task.kind == "md":
        ident = _RunIdentity(
            run_id=config.run.id, mode=config.task.mode,
            ensemble=config.dynamics.ensemble,
            integrator=config.dynamics.integrator,
            spec=_spec_from_dynamics(config),
            target_steps=int(config.dynamics.steps))
        return _md_completed_state(run_dir, ident, events,
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
    """Full physical handoff identity: element order, unwrapped positions,
    cell/PBC, masses, momenta when present, initial charges/magmoms and the
    FixAtoms set — the digest the stage's source record and its
    materialized ``initial.traj`` must both carry (R1)."""
    arrays = [np.asarray(atoms.numbers), np.asarray(atoms.positions, float),
              np.asarray(atoms.cell.array, float),
              np.asarray(atoms.pbc), np.asarray(atoms.get_masses()),
              np.asarray(atoms.get_momenta(), dtype=float)
              if "momenta" in atoms.arrays else np.zeros(0),
              np.asarray(atoms.get_initial_charges(), dtype=float),
              np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)]
    fixed = set()
    for constraint in atoms.constraints or []:
        # the supported constraint universe is FixAtoms only
        fixed.update(int(i) for i in np.atleast_1d(constraint.index))
    arrays.append(np.asarray(sorted(fixed), dtype=int))
    return state_digest(*arrays)


def _materialize_initial(stage_dir: Path, atoms: Atoms) -> Path:
    """Write the stage's initial structure as an ASE .traj — full double
    precision for positions, masses, momenta and the FixAtoms set (verified
    by the recipe's own round-trip checks); never a lossy text coordinate
    file.  Written atomically, and an existing file is returned only after
    its content verifies against the same atoms (R1)."""
    path = stage_dir / "initial.traj"
    if path.exists():
        return path
    from ase.io import write as ase_write

    stage_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    ase_write(tmp, atoms, format="traj")
    os.replace(tmp, path)
    return path


def _read_initial_digest(path: Path) -> str:
    from ase.io import read as ase_read

    return _state_file_digest(ase_read(path, format="traj"))


def run_serial_recipe(root: str | Path, stages: list[RecipeStage], *,
                      verbose: bool = True,
                      force_unlock: bool = False) -> dict:
    """Run the serial recipe (e.g. relax → NVT → NVE), idempotently.

    Each stage owns an independent run id, config and cost ledger under
    ``root/<name>``.  A stage's stable identity (config settings, backend
    declarations, source boundary, momentum policy, input-structure
    identity) is persisted atomically before its first backend computation
    (R1); on re-invocation the controller reconciles against the run's own
    resolved config and parsed events — finished stages are adopted without
    recomputation (R2), resumable ones continue through the ordinary resume
    protocol with the exact step conversion, and changed
    settings/policies/sources refuse before any new computation with a
    new-output-directory instruction.  Source runs are read-only.
    ``force_unlock`` is the caller's deliberate assertion that an
    interrupted stage's writer lock is stale — never taken unconditionally.
    Returns the manifest dict (also written to ``root/workflow.json``).
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
        # 1. identity first — persisted atomically before any backend
        #    computation (R1); every refusal above happens without touching
        #    the existing run directory
        record = _reconcile_stage_identity(
            manifest, stage, config, stage_dir, previous)
        record["status"] = "running" if record["status"] == "pending" \
            else record["status"]
        _write_json_atomic(manifest_path, manifest)

        if record["status"] == "done":
            previous = load_completed_state(stage_dir)
            continue
        if record["status"] == "failed":
            raise WorkflowError(
                f"stage {stage.name!r} previously failed; fix the stage and "
                "restart from a clean recipe root — the recipe does not "
                "silently rerun it")

        # 2. authoritative run state from parsed events (R2), then dispatch
        started = time.perf_counter()
        try:
            state = _advance_stage(record, stage, config, stage_dir,
                                   previous, verbose=verbose,
                                   force_unlock=force_unlock)
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
        _finalize_stage_record(record, stage_dir, config, state)
        _write_json_atomic(manifest_path, manifest)
        previous = state
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _reconcile_stage_identity(manifest: list | dict, stage: RecipeStage,
                              config: PyramidConfig, stage_dir: Path,
                              previous: CompletedState | None) -> dict:
    """Bind or verify the stage's stable identity before any computation.

    A manifest record must match the declared run id, config file digest,
    and momenta policy exactly.  A stage with run evidence but no manifest
    record (an interrupted first start, R1) is adopted only when the
    current config's resolved settings equal the run's own
    resolved_config.json — never blessing new settings over old results.
    """
    stages = manifest["stages"]
    config_sha = sha256(Path(stage.config_path).read_bytes()).hexdigest()
    record = next((s for s in stages if s["name"] == stage.name), None)
    if record is not None:
        if record["run_id"] != config.run.id:
            raise WorkflowError(
                f"stage {stage.name!r}: the manifest binds run id "
                f"{record['run_id']!r}, the config now declares "
                f"{config.run.id!r} — use a new recipe root")
        if record["config_sha256"] != config_sha:
            raise WorkflowError(
                f"stage {stage.name!r} already started with a different "
                "config; refusing to continue under changed settings — "
                "use a new recipe root")
        if record.get("momenta") != stage.momenta:
            raise WorkflowError(
                f"stage {stage.name!r}: the momenta policy changed from "
                f"{record['momenta']!r} to {stage.momenta!r} after the "
                "stage started — use a new recipe root")
        return record
    has_run_evidence = (stage_dir / "resolved_config.json").is_file() or \
        (stage_dir / "events.jsonl").is_file()
    if has_run_evidence:
        stored_path = stage_dir / "resolved_config.json"
        if not stored_path.is_file():
            raise WorkflowError(
                f"stage {stage.name!r} has run evidence but no "
                "resolved_config.json; the original settings cannot be "
                "proven — use a new output directory (the existing run is "
                "preserved)")
        stored = json.loads(stored_path.read_text())
        if config.resolved_dict() != stored:
            raise WorkflowError(
                f"stage {stage.name!r} has an existing run under different "
                "resolved settings (its interrupted start never persisted a "
                "stage record); the recipe refuses to bind new settings to "
                "existing results — use a new output directory")
        # The semantic match is proven against the run's own record: adopt
        # the stage from run evidence (M4-era outputs are compatible this
        # way only when the facts agree).
        record = {"name": stage.name, "kind": config.task.kind,
                  "run_id": stored["run"]["id"], "run_dir": stage.name,
                  "status": "running", "config_sha256": config_sha,
                  "source": None, "momenta": stage.momenta,
                  "wall_time_s": 0.0, "result": None,
                  "reconciled_from_run": True}
        stages.append(record)
        return record
    record = {"name": stage.name, "kind": config.task.kind,
              "run_id": config.run.id, "run_dir": stage.name,
              "status": "pending", "config_sha256": config_sha,
              "source": None, "momenta": stage.momenta,
              "wall_time_s": 0.0, "result": None}
    stages.append(record)
    return record


def _advance_stage(record: dict, stage: RecipeStage, config: PyramidConfig,
                   stage_dir: Path, previous: CompletedState | None, *,
                   verbose: bool, force_unlock: bool) -> CompletedState:
    """Dispatch one non-done stage by its parsed run state (R2)."""
    if config.task.kind == "md":
        if previous is None:
            raise WorkflowError(
                f"stage {stage.name!r}: an MD stage needs a previous "
                "stage's completed state")
        _prepare_md_stage(record, stage, config, stage_dir, previous)
        facts = _run_facts(stage_dir)
        target = int(config.dynamics.steps)
        if facts["complete_steps"] >= target:
            # Finished before the bookkeeping survived (R2): adopt the
            # completed run — no resume, no extra step.
            if verbose:
                print(f"recipe {stage.name}: already complete "
                      f"({facts['complete_steps']} steps) — adopting")
            return load_completed_state(stage_dir)
        has_checkpoint = (stage_dir / "checkpoints"
                          / "latest.json").is_file()
        if has_checkpoint and (facts["complete_steps"] > 0
                               or facts["healable"]):
            # The exact step conversion: resume adds `extra` NEW steps and
            # first binds a committed tail evaluation (at most one) as its
            # step record — never stacking the heal as an extra step.
            extra = target - facts["complete_steps"] - facts["healable"]
            if verbose:
                print(f"recipe {stage.name}: resume "
                      f"{facts['complete_steps']} -> {target} steps"
                      + (" (binding one committed tail step)"
                         if facts["healable"] else ""))
            resume_workflow(stage_dir, extra, verbose=verbose,
                            handle_sigint=False, force_unlock=force_unlock)
            return load_completed_state(stage_dir)
        if facts["started"]:
            # Run records exist but nothing resumable: no complete boundary
            # and no committed tail to bind — never silently restart over
            # it (R2).
            raise WorkflowError(
                f"stage {stage.name!r} has run records but no resumable "
                "initial state (no complete boundary and no committed "
                "tail step to bind); the recipe refuses to restart over "
                "it — resolve the stage manually or use a new recipe root "
                "(the existing run is preserved)")
        if verbose:
            print(f"recipe {stage.name}: running md ({config.task.mode})")
        run_workflow(config, verbose=verbose, handle_sigint=False)
        return load_completed_state(stage_dir)
    # relax: no optimizer resume anywhere; a converged run is adopted, a
    # started unconverged one is a hard stop (R2)
    facts = _run_facts(stage_dir)
    if facts["started"]:
        if facts["converged"]:
            if verbose:
                print(f"recipe {stage.name}: already converged — adopting")
            return load_completed_state(stage_dir)
        raise WorkflowError(
            f"stage {stage.name!r} has a started but unconverged relax run; "
            "optimizer resume is not supported — fix the stage and use a "
            "clean recipe root (the existing run is preserved)")
    if verbose:
        print(f"recipe {stage.name}: running relax ({config.task.mode})")
    run_workflow(config, verbose=verbose, handle_sigint=False)
    return load_completed_state(stage_dir)


def _run_facts(stage_dir: Path) -> dict:
    """Authoritative run state from the parsed event log — never from
    substring counts or the manifest (R2)."""
    facts = {"started": False, "complete_steps": 0, "committed": 0,
             "healable": 0, "converged": False}
    events_path = stage_dir / "events.jsonl"
    if not events_path.is_file():
        return facts
    facts["started"] = True
    complete: set[int] = set()
    committed: set[int] = set()
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event_type = event.get("type")
        if event_type == "step_completed":
            complete.add(int(event["step_id"]))
        elif event_type == "evaluation_committed":
            committed.add(int((event.get("context") or {})
                              ["evaluation_id"]))
        elif event_type == "run_summary" and "converged" in event:
            facts["converged"] = bool(event["converged"])
    facts["complete_steps"] = len(complete)
    facts["committed"] = len(committed)
    facts["healable"] = int(len(committed) == len(complete) + 2)
    return facts


def _prepare_md_stage(record: dict, stage: RecipeStage,
                      config: PyramidConfig, stage_dir: Path,
                      previous: CompletedState) -> None:
    """Materialize or verify the stage's initial state, and bind the source
    identity once (R1).  A started stage's existing ``initial.traj`` must
    still match the parent boundary it was bound to."""
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
    expected = _state_file_digest(atoms)
    initial = _materialize_initial(stage_dir, atoms)
    actual = _read_initial_digest(initial)
    if actual != expected:
        raise WorkflowError(
            f"stage {stage.name!r}: initial.traj no longer matches the "
            "parent stage's completed boundary (the file or the source "
            "state changed) — use a new recipe root")
    configured = Path(config.structure.file)
    resolved_structure = configured if configured.is_absolute() else \
        (Path(stage.config_path).parent / configured).resolve()
    if resolved_structure != initial.resolve():
        raise WorkflowError(
            f"stage {stage.name!r}: the config's structure.file must be "
            f"{initial.name} in the stage directory — the controller owns "
            "the materialized initial state")
    if record["source"] is None:
        record["source"] = dict(previous.provenance)
        record["source"]["state_digest"] = expected
        if momenta_policy == "initialize":
            record["source"]["momenta_initialization"] = {
                "policy": "initialize",
                "temperature_K": config.dynamics.temperature_K,
                "velocity_seed": config.dynamics.velocity_seed,
            }
        elif momenta_policy == "preserve":
            record["source"]["momenta_initialization"] = {
                "policy": "preserve"}
    elif record["source"].get("state_digest") != expected:
        raise WorkflowError(
            f"stage {stage.name!r}: the bound source state changed after "
            "the stage started — use a new recipe root")


def _finalize_stage_record(record: dict, stage_dir: Path,
                           config: PyramidConfig,
                           state: CompletedState) -> None:
    """Mark a stage done with its result and honest wall time (R1/R2):
    measured controller time accumulates across invocations; when durable
    RUN_SUMMARY timings cover more, they win; an invocation whose time is
    unrecoverable is marked, never fabricated."""
    record["status"] = "done"
    record["result"] = _stage_result(stage_dir, config, state)
    durable, complete = _durable_wall_time(stage_dir)
    record["wall_time_s"] = max(record["wall_time_s"], durable)
    record["wall_time_complete"] = complete


def _durable_wall_time(stage_dir: Path) -> tuple[float, bool]:
    """(sum of persisted RUN_SUMMARY wall times, fully measured?) — a
    crashed invocation leaves no summary for its tail, and the missing
    portion is marked rather than guessed."""
    events_path = stage_dir / "events.jsonl"
    if not events_path.is_file():
        return 0.0, True
    total = 0.0
    has_summary = False
    crashed = False
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "run_summary":
            has_summary = True
            total += float(event.get("wall_time_s") or 0.0)
        elif event.get("type") == "run_end" \
                and event.get("status") == "failed":
            crashed = True
    return total, has_summary and not crashed


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
