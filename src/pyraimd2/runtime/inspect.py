"""Library-level run inspection: status, cost summary and CSV export.

``inspect_run`` reads the run directory's authoritative event log
(``events.jsonl``) and its trajectory database, and returns one structured
dict — the same source for the human-readable ``format_inspection`` and for
any future ``--json`` CLI (WP04).  ``summary_csv`` flattens one line per
committed evaluation for direct plotting of energy, error and reference
calls against time, without reading the full event log.  Rows from older
libraries (no ``schema_version``/``metadata``) export with blank fields.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
from ase import units

from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import (
    EVALUATION_COMMITTED,
    MODEL_UPDATE,
    RUN_END,
    RUN_START,
    RUN_SUMMARY,
    STEP_COMPLETED,
    EventLogError,
)
from pyraimd2.store.store import Store


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [(number, line.strip())
             for number, line in
             enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
             if line]
    events = []
    last_line = lines[-1][0] if lines else None
    for line_number, line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            if line_number == last_line:
                # Torn tail from a crash: the final event never committed.
                # Reading skips it (the writer side recovers it on open);
                # a corrupt committed middle event still fails loud.
                break
            raise EventLogError(
                f"corrupt event at {path}:{line_number}: {error}"
            ) from error
    return events


def _temperature_K(atoms, n_fixed: int = 0) -> float | None:
    """Instantaneous temperature from momenta over the free degrees of
    freedom. With FixAtoms, the fixed coordinates carry no momentum and no
    kinetic share — the divisor is the actual unconstrained DOF, not 3N."""
    momenta = atoms.get_momenta()
    masses = atoms.get_masses()
    if not len(atoms) or not np.isfinite(momenta).all():
        return None
    dof = 3 * (len(atoms) - int(n_fixed))
    if dof <= 0:
        return None
    kinetic = float((momenta**2 / (2.0 * masses[:, None])).sum())
    return 2.0 * kinetic / (dof * units.kB)


def _find_db(run_dir: Path) -> Path | None:
    dbs = sorted(run_dir.glob("*.db"))
    return dbs[0] if len(dbs) == 1 else None


def _last_checkpoint(run_dir: Path) -> dict | None:
    pointer = run_dir / "checkpoints" / "latest.json"
    if not pointer.exists():
        return None
    try:
        generation = int(json.loads(pointer.read_text())["generation"])
        manifest = json.loads(
            (run_dir / "checkpoints" / str(generation) / "manifest.json").read_text())
    except (OSError, ValueError, KeyError):
        return {"valid": False}
    return {"generation": generation, "nsteps": manifest.get("nsteps"),
            "physical_time_fs": manifest.get("physical_time_fs"),
            "last_event_seq": manifest.get("last_event_seq"), "valid": True}


def inspect_run(run_dir: str | Path, run_id: str | None = None) -> dict:
    """Structured run status from the run directory (events + trajectory db).

    Cost numbers come from the authoritative task events: actual physical
    reference executions, logical reference requests, cache hits, failed
    attempts and independent checks are reported separately — never merged
    into one "reference calls" figure.  ``last_checkpoint`` is None until
    WP03 exists.
    """
    run_dir = Path(run_dir)
    events = _read_events(run_dir / "events.jsonl")
    start = next((e for e in events if e.get("type") == RUN_START), None)
    if run_id is None:
        if start is not None:
            run_id = start["run_id"]
        else:
            db_path = _find_db(run_dir)
            if db_path is None:
                raise EventLogError(
                    f"no events.jsonl and no unique *.db in {run_dir}; pass run_id"
                )
            run_ids = {str(r.key_value_pairs.get("run_id"))
                       for r in Store(db_path)._db.select()}
            if len(run_ids) != 1:
                raise EventLogError(
                    f"cannot infer run_id in {run_dir} (candidates: {sorted(run_ids)})"
                )
            run_id = run_ids.pop()

    cost = summarize_tasks(events)
    committed = [e for e in events if e.get("type") == EVALUATION_COMMITTED]
    run_summaries = [e for e in events if e.get("type") == RUN_SUMMARY]
    run_end = next((e for e in reversed(events) if e.get("type") == RUN_END), None)
    updates = [e for e in events if e.get("type") == MODEL_UPDATE]
    complete_steps = {int(e["step_id"]) for e in events
                      if e.get("type") == STEP_COMPLETED}
    driver = ((start or {}).get("workflow") or {}).get("driver")
    # Completion semantics are task-specific (A2/A4): MD drivers complete a
    # step only at its boundary; relax/singlepoint commits are complete
    # records themselves.
    md_kind = driver == "plain-nve" or (driver is None
                                        and (start or {}).get("policy"))

    trajectory = {"n_rows": 0, "n_committed": 0, "last_energy_eV": None,
                  "last_temperature_K": None, "n_accepted": 0,
                  "last_step": None}
    last_evaluation = None
    db_path = _find_db(run_dir)
    if db_path is not None:
        store = Store(db_path)
        rows = sorted(store._db.select(run_id=run_id),
                      key=lambda r: int(r.key_value_pairs["step"]))
        trajectory["n_rows"] = len(rows)
        pairs = list(store.iter_committed(events, run_id))
        trajectory["n_committed"] = len(pairs)
        trajectory["n_accepted"] = sum(
            row.key_value_pairs["route"] == "ml" for _event, row in pairs)
        if pairs:
            # The current trajectory state is the last COMPLETE boundary
            # (or the initial evaluation); the latest committed evaluation
            # of any phase is reported separately with its phase marked (A4).
            last_event, last_row = pairs[-1]
            last_context_row = last_event.get("context") or {}
            last_step_id = int(last_row.key_value_pairs["step"])
            last_complete = (not md_kind) or last_step_id == -1 \
                or last_step_id in complete_steps
            driving = last_row.data.get("driving")
            last_evaluation = {
                "evaluation_id": int(last_context_row.get("evaluation_id",
                                                         last_step_id + 1)),
                "step_id": last_step_id,
                "phase": last_context_row.get("phase"),
                "physical_time_fs": last_context_row.get("physical_time_fs"),
                "energy_eV": (None if driving is None
                              else float(driving["energy"])),
                "complete": bool(last_complete),
            }
            boundary = next(
                ((event, row) for event, row in reversed(pairs)
                 if (not md_kind)
                 or int(row.key_value_pairs["step"]) == -1
                 or int(row.key_value_pairs["step"]) in complete_steps),
                None)
            if boundary is not None:
                _event, row = boundary
                step = int(row.key_value_pairs["step"])
                trajectory["last_step"] = step
                driving = row.data.get("driving")
                if driving is not None:
                    trajectory["last_energy_eV"] = float(driving["energy"])
                metadata = row.data.get("metadata") or {}
                constraint = metadata.get("constraint") or {}
                frame = store.complete_step_frame(
                    row, Store.row_timestep_fs(row) or 0.0)
                trajectory["last_temperature_K"] = _temperature_K(
                    frame, n_fixed=int(constraint.get("n_fixed", 0)))

    last_context = (committed[-1].get("context") if committed else None) or {}
    if (start or {}).get("event_schema_version") is not None:
        # Current writers emit one step_completed per finished integration
        # step; a committed evaluation without its boundary (a crash before
        # the last half-kick) is not a complete step.
        n_complete_steps = len(complete_steps)
    elif last_context.get("step_id") is None:
        n_complete_steps = None
    else:
        # Older logs without step boundaries: approximate from the last
        # committed evaluation (may count a crashed mid-step evaluation).
        n_complete_steps = last_context["step_id"] + 1
    if md_kind:
        # Trajectory time comes from the last complete-step boundary (or the
        # initial evaluation at t=0) — never from an unfinished tail
        # evaluation (A4).
        completed_times = [float(e["physical_time_fs"]) for e in events
                           if e.get("type") == STEP_COMPLETED]
        if completed_times:
            physical_time_fs = completed_times[-1]
        elif committed:
            physical_time_fs = 0.0
        else:
            physical_time_fs = None
    else:
        physical_time_fs = last_context.get("physical_time_fs")
    checks = {"independent_checks": 0, "accepted_count": 0, "detected_count": 0,
              "bound": None, "probability": None}
    for event in committed:
        verification = event.get("verification")
        if verification:
            checks["accepted_count"] = verification["accepted_count"]
            checks["detected_count"] = verification["detected_count"]
            checks["bound"] = verification["bound"]
            checks["probability"] = verification["probability"]
        checks["independent_checks"] += int(bool(event.get("checked")))
    failure = None
    if run_end is not None and run_end.get("status") != "success":
        failure = {"status": run_end.get("status"), "reason": run_end.get("reason")}
    else:
        failed = [e for e in events
                  if e.get("type") == "task" and e.get("status") == "failed"]
        if failed and not committed:
            failure = {"status": "failed",
                       "reason": failed[-1].get("error", "reference task failed")}
    return {
        "run_id": run_id,
        "schema_version": (start or {}).get("schema_version"),
        "event_schema_version": (start or {}).get("event_schema_version"),
        "reference_id": (start or {}).get("reference_id"),
        "n_evaluations": len(committed),
        "n_complete_steps": n_complete_steps,
        "physical_time_fs": physical_time_fs,
        "last_evaluation": last_evaluation,
        "model_id": last_context.get("model_id", (start or {}).get("model_id")),
        "n_model_updates": len(updates),
        "trajectory": trajectory,
        "checks": checks,
        "cost": cost,
        "wall_time_s": (run_summaries[-1]["wall_time_s"] if run_summaries
                        else None),
        "last_checkpoint": _last_checkpoint(run_dir),
        "failure": failure,
        "events": {"count": len(events),
                   "last_seq": max((int(e.get("seq", 0)) for e in events), default=0)},
    }


def format_inspection(info: dict) -> str:
    """Human-readable rendering of :func:`inspect_run`'s dict (same source)."""
    reference = info["cost"]["reference"]
    checks = info["checks"]
    lines = [
        f"run {info['run_id']}",
        f"  evaluations committed : {info['n_evaluations']}",
        f"  complete steps        : {info['n_complete_steps']}",
        f"  physical time         : {info['physical_time_fs']} fs",
        (f"  model                 : {info['model_id']} "
         f"({info['n_model_updates']} updates)"),
        f"  last energy           : {info['trajectory']['last_energy_eV']} eV",
        f"  last temperature      : {info['trajectory']['last_temperature_K']} K",
        f"  last evaluation       : {info['last_evaluation']}",
        (f"  reference (actual)    : {reference['actual_executions']} executions "
         f"({reference['successful_executions']} ok, "
         f"{reference['failed_attempts']} failed)"),
        (f"  reference (logical)   : {reference['logical_requests']} requests, "
         f"{reference['cache_hits']} cache hits"),
        (f"  independent checks    : {checks['independent_checks']} "
         f"(accepted {checks['accepted_count']}, "
         f"detected {checks['detected_count']}, bound {checks['bound']})"),
        (f"  inference/training/io : {info['cost']['counts']['inference']}/"
         f"{info['cost']['counts']['training']}/{info['cost']['counts']['io']}"),
        f"  leaf task elapsed     : {info['cost']['total_elapsed_s']:.3f} s",
        f"  run wall time         : {info['wall_time_s']}",
        f"  last checkpoint       : {info['last_checkpoint']}",
        f"  failure               : {info['failure']}",
    ]
    return "\n".join(lines)


_CSV_COLUMNS = [
    "evaluation_id", "step_id", "physical_time_fs", "route", "reason",
    "energy_eV", "max_force_error_eV_A", "endpoint_work_eV",
    "accepted", "checked", "violation",
    "reference_anchor", "reference_probe", "reference_check",
    "segment_id", "model_id",
]


def summary_csv(store: Store, run_id: str) -> str:
    """One CSV line per committed evaluation: energy/error/reference vs time.

    Read from the trajectory database (event log not required).  Older rows
    without energetic metadata export with blank fields.
    """
    rows = sorted(store._db.select(run_id=run_id),
                  key=lambda r: int(r.key_value_pairs["step"]))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    for row in rows:
        data = row.data
        metadata = data.get("metadata") or {}
        context = metadata.get("context") or {}
        observed = metadata.get("observed") or {}
        calls = metadata.get("reference_calls_this_evaluation") or {}
        driving = data.get("driving")
        energy = (driving["energy"] if driving is not None else
                  (data.get("engine") or data.get("surrogate") or {}).get("energy"))
        writer.writerow({
            "evaluation_id": context.get("evaluation_id",
                                         metadata.get("evaluation_index", "")),
            "step_id": context.get("step_id", row.key_value_pairs["step"]),
            "physical_time_fs": context.get("physical_time_fs",
                                            metadata.get("time_fs", "")),
            "route": row.key_value_pairs["route"],
            "reason": data.get("reason", ""),
            "energy_eV": "" if energy is None else float(energy),
            "max_force_error_eV_A": observed.get("max_force_error_eV_A", ""),
            "endpoint_work_eV": observed.get("endpoint_work_eV", ""),
            "accepted": metadata.get("accepted", ""),
            "checked": metadata.get("checked", ""),
            "violation": metadata.get("violation", ""),
            "reference_anchor": calls.get("anchor", ""),
            "reference_probe": calls.get("probe", ""),
            "reference_check": calls.get("check", ""),
            "segment_id": metadata.get("segment_id", ""),
            "model_id": context.get("model_id", ""),
        })
    return buffer.getvalue()
