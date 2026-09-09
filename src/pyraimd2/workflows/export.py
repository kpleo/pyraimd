"""Trajectory export: store rows -> extxyz frames with an explicit force source.

One frame builder serves both the run-directory ``trajectory.extxyz``
preview (driving forces, thinned by ``output.trajectory_interval_steps``)
and ``pyramid export``.  The force source is always stated on the frame:

- ``driving``: the force that actually propagated the MD (stored ``driving``
  payload, with the route payload as fallback for older rows).
- ``reference``: the reference engine label.  Rows without one (accepted,
  unchecked evaluations) are **missing data, not zero force**: the frame
  carries ``forces_available=F`` and an all-NaN forces array — never zeros.
- ``base``: the uncorrected surrogate prediction.

Frames also record run_id, step/evaluation ids, physical time and route, so
an exported file stays interpretable without the event log.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from pyraimd2.runtime.inspect import _read_events, inspect_run
from pyraimd2.store import Store


class ExportError(RuntimeError):
    """An export cannot be produced honestly from the run directory."""


FORCE_SOURCES = ("driving", "reference", "base")


def _select_forces(data: dict, route: str, force_source: str):
    """(energy, forces, available, label_id) for one store row."""
    if force_source == "driving":
        payload = data.get("driving")
        if payload is None:
            payload = data.get("engine") if route == "dft" else data.get("surrogate")
        if payload is None:
            raise ExportError(
                f"store row has no {route!r} payload to drive from; the "
                "trajectory database looks corrupt")
        return float(payload["energy"]), np.asarray(payload["forces"], float), True, None
    key = "engine" if force_source == "reference" else "surrogate"
    payload = data.get(key)
    if payload is None:
        return None, None, False, None
    label_id = data.get("engine_label_id") if force_source == "reference" else None
    return (float(payload["energy"]), np.asarray(payload["forces"], float),
            True, label_id)


def _row_timestep_fs(row) -> float | None:
    return Store.row_timestep_fs(row)


def completed_step_ids(run_dir: str | Path) -> set[int]:
    """Step indices with a committed complete-step boundary (STEP_COMPLETED).

    An evaluation committed without its boundary (a crash before the second
    half-kick) is a computation record, not a completed step. With no event
    log, the empty set is returned and callers fall back to store rows.
    """
    import json

    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return set()
    completed = set()
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # torn tail from a crash: never a committed boundary
            raise
        if event.get("type") == "step_completed":
            completed.add(int(event["step_id"]))
    return completed


def frame_from_row(row, run_id: str, *, force_source: str,
                   wrap: bool = False, timestep_fs: float | None = None,
                   store: Store | None = None) -> Atoms:
    """One export frame from one store row (see module docstring for the
    missing-data marking rules).

    Coordinates in the store are continuous unwrapped positions; pass
    ``wrap=True`` to export them wrapped back into the cell instead (the
    store itself is never rewritten). Momenta are the complete-step values:
    mid-step (half-step) records are reconstructed with the row's driving
    force and marked in ``info['momenta_source']`` — the original record is
    never overwritten.
    """
    data = row.data
    route = str(row.key_value_pairs["route"])
    step = int(row.key_value_pairs["step"])
    energy, forces, available, label_id = _select_forces(data, route,
                                                         force_source)
    if store is not None:
        atoms = store.complete_step_frame(row, timestep_fs
                                          if timestep_fs is not None
                                          else (_row_timestep_fs(row) or 0.0))
    else:
        atoms = row.toatoms()
        atoms.calc = None
    if wrap:
        atoms.wrap()
        atoms.info["coordinates"] = "wrapped"
    else:
        atoms.info["coordinates"] = "unwrapped"
    atoms.arrays["forces"] = (forces if available
                              else np.full((len(atoms), 3), np.nan))
    metadata = data.get("metadata") or {}
    context = metadata.get("context") or {}
    atoms.info.update({
        "run_id": run_id,
        "step_id": step,
        "row_id": int(row.id),
        "evaluation_id": step + 1,
        "physical_time_fs": float(context.get("physical_time_fs", np.nan)),
        "route": route,
        "force_source": force_source,
        "forces_available": bool(available),
    })
    if available:
        atoms.info["energy"] = float(energy)
    if label_id:
        atoms.info["reference_label_id"] = str(label_id)
    return atoms


def frames_from_store(store: Store, run_id: str, *, force_source: str,
                      interval_steps: int = 1,
                      only_step: int | None = None,
                      wrap: bool = False,
                      complete_steps: set[int] | None = None,
                      committed: list | None = None) -> list[Atoms]:
    """Export frames in step order, thinned by evaluation id.

    ``interval_steps = k`` keeps every k-th committed evaluation (the
    initial evaluation, id 0, always passes); ``only_step`` selects the one
    row stored at that step index.  Missing forces for the requested source
    are marked, never zero-filled.  ``wrap=True`` wraps the continuous
    unwrapped store coordinates back into the cell for output.

    With ``committed`` given (``Store.iter_committed`` pairs), frames come
    from the authoritative commit→row binding only; orphan rows (written
    but never committed) never become trajectory frames (A3).  With
    ``committed=None`` the function is in raw/legacy mode: **all** rows are
    candidates, step filtering cannot pick the authoritative row among
    several at one step, and orphans are NOT excluded.  Asking for
    complete-step filtering in raw mode is rejected — select the committed
    view (``Store.iter_committed`` or :func:`frames_for_run`) instead.
    Runs without an event log intentionally use the raw fallback: no log
    does not mean provably no complete frames.
    """
    if force_source not in FORCE_SOURCES:
        raise ExportError(
            f"force_source must be one of {list(FORCE_SOURCES)}, got "
            f"{force_source!r}")
    if complete_steps is not None and committed is None:
        raise ExportError(
            "complete_steps selects a finished step, not the authoritative "
            "row at that step: pass committed=store.iter_committed(events, "
            "run_id) (or frames_for_run), or call without complete_steps "
            "for the explicit raw/legacy row selection")
    if committed is not None:
        pairs = committed
    else:
        pairs = [(None, row) for row in
                 sorted(store._db.select(run_id=run_id),
                        key=lambda r: int(r.key_value_pairs["step"]))]
    frames: list[Atoms] = []
    for event, row in pairs:
        step = int(row.key_value_pairs["step"])
        evaluation_id = (int((event.get("context") or {})["evaluation_id"])
                         if event is not None else step + 1)
        if only_step is not None and step != only_step:
            continue
        if only_step is None and evaluation_id % interval_steps != 0:
            continue
        if complete_steps is not None and step >= 0 and step not in complete_steps:
            continue
        frame = frame_from_row(row, run_id, force_source=force_source,
                               wrap=wrap, store=store)
        frame.info["evaluation_id"] = evaluation_id
        frame.info["integration_phase"] = (
            "initial_evaluation" if step < 0 else "complete_step")
        frames.append(frame)
    return frames


def write_extxyz(path: Path, frames: list[Atoms]) -> None:
    """Write frames atomically (temp file, then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    ase_write(tmp, frames, format="extxyz")
    os.replace(tmp, path)


def append_extxyz(path: Path, frame: Atoms) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(path, frame, format="extxyz", append=True)


def infer_run_id(run_dir: str | Path) -> str:
    """The run's identity, via the same inference rules as inspect."""
    return str(inspect_run(run_dir)["run_id"])


def frames_for_run(store: Store, run_dir: str | Path, run_id: str, *,
                   force_source: str,
                   interval_steps: int = 1) -> list[Atoms]:
    """Committed frames of one run directory through the shared
    commit→row view — the one row selection used by the CLI export, the
    automatic trajectory and the summaries (R7).

    Completion semantics are task-specific (A2): MD drivers (plain NVE,
    adaptive energetic) complete a step only at its STEP_COMPLETED
    boundary; relax and singlepoint commit complete records per evaluation
    and have no step boundaries.  A run without an event log falls back to
    all rows (no log does not mean provably no complete frames).
    """
    events = _read_events(Path(run_dir) / "events.jsonl")
    start = next((e for e in events if e.get("type") == "run_start"), None)
    driver = ((start or {}).get("workflow") or {}).get("driver")
    md_kind = driver == "plain-nve" or (driver is None
                                        and (start or {}).get("policy"))
    complete_steps = completed_step_ids(run_dir) if md_kind else None
    committed = (list(store.iter_committed(events, run_id))
                 if events else None)
    return frames_from_store(store, run_id, force_source=force_source,
                             interval_steps=interval_steps,
                             complete_steps=complete_steps,
                             committed=committed)


def export_run(run_dir: str | Path, *, force_source: str = "driving",
               output: str | Path | None = None, force: bool = False) -> dict:
    """Export the committed trajectory of a run directory to extxyz.

    Returns a small report (frames written, missing-label frames, path).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ExportError(
            f"run directory not found: {run_dir}; pass the directory written "
            "by `pyramid run` (it contains trajectory.db and events.jsonl)")
    db_path = run_dir / "trajectory.db"
    if not db_path.is_file():
        raise ExportError(
            f"no trajectory.db in {run_dir}; nothing to export — run the "
            "configuration first with `pyramid run`")
    run_id = infer_run_id(run_dir)
    store = Store(db_path)
    frames = frames_for_run(store, run_dir, run_id,
                            force_source=force_source)
    if not frames:
        raise ExportError(
            f"run {run_id!r} has no committed evaluations to export")
    if output is None:
        output = run_dir / f"export-{force_source}.extxyz"
    output = Path(output)
    if output.exists() and not force:
        raise ExportError(
            f"output file exists: {output}; pass --force to overwrite or "
            "choose a different --output")
    write_extxyz(output, frames)
    missing = sum(1 for frame in frames if not frame.info["forces_available"])
    return {"run_id": run_id, "frames": len(frames),
            "missing_forces_frames": missing, "force_source": force_source,
            "output": output}
