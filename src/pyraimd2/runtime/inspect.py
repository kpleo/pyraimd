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
    EventLogError,
)
from pyraimd2.store.store import Store


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise EventLogError(
                    f"corrupt event at {path}:{line_number}: {error}"
                ) from error
    return events


def _temperature_K(atoms) -> float | None:
    momenta = atoms.get_momenta()
    masses = atoms.get_masses()
    if not len(atoms) or not np.isfinite(momenta).all():
        return None
    kinetic = float((momenta**2 / (2.0 * masses[:, None])).sum())
    return 2.0 * kinetic / (3.0 * len(atoms) * units.kB)


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

    trajectory = {"n_rows": 0, "last_energy_eV": None, "last_temperature_K": None,
                  "n_accepted": 0, "last_step": None}
    db_path = _find_db(run_dir)
    if db_path is not None:
        rows = sorted(Store(db_path)._db.select(run_id=run_id),
                      key=lambda r: int(r.key_value_pairs["step"]))
        trajectory["n_rows"] = len(rows)
        if rows:
            last = rows[-1]
            trajectory["last_step"] = int(last.key_value_pairs["step"])
            driving = last.data.get("driving")
            if driving is not None:
                trajectory["last_energy_eV"] = float(driving["energy"])
            trajectory["last_temperature_K"] = _temperature_K(last.toatoms())
            trajectory["n_accepted"] = sum(
                r.key_value_pairs["route"] == "ml" for r in rows
            )

    last_context = (committed[-1].get("context") if committed else None) or {}
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
        "n_complete_steps": (None if last_context.get("step_id") is None
                             else last_context["step_id"] + 1),
        "physical_time_fs": last_context.get("physical_time_fs"),
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
