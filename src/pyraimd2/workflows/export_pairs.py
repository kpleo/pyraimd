"""Export paired force labels saved in a completed MTS run for calibrate-scale.

A finished (or boundary-stopped) fixed-model ``mts-nve-respa`` run already
stores, at every committed outer boundary, the reference engine label and
the fast backend's prediction for the SAME configuration.  This module
turns an explicit selection of those saved records into the
``pairs.npz`` the existing ``pyramid calibrate-scale`` consumes — no
backend is constructed, no model is loaded, nothing is evaluated or
recomputed, and the source run directory is opened strictly read-only
(the existing WAL/sidecar refusal rules apply).

Selection is by ``evaluation_id`` from the committed evaluation context
(never row ids, steps or times): the initial configuration is
evaluation_id 1 stored at step -1.  Every selected commit must bind its
row explicitly (row_id + row_digest — no legacy step fallback), a
non-initial frame additionally needs its STEP_COMPLETED boundary, and
any missing frame aborts the whole export naming that id.  Both force
arrays come from the SAME row's ``data.engine.forces`` and
``data.surrogate.forces`` — never step/time-matched across trajectories,
never the driving force, never interpolated, zero-filled or recomputed.
Runs whose fast side is a correction wrapper (``scaled`` /
``quadratic-corrected``, nested included) are refused: those stored
labels are the CORRECTED model's output, not the raw base model's, and
un-scaling or stripping terms is never attempted.

Provenance, honestly: the export guarantees provenance-by-record — the
committed bindings are digest-checked, the run's identities are
cross-checked between events, manifest and resolved config, and the
source files' SHA256 are recorded and re-verified after reading.  The
stored row digest covers only parts of a row's identity/driving fields;
it is NOT a cryptographic certification of the forces or configurations,
and this export cannot detect arbitrary historical data tampering or
prove the labels physically accurate.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from pyraimd2.runtime.inspect import _read_events
from pyraimd2.store import Store

PAIRS_SCHEMA_VERSION = 1
FORCE_UNIT = "eV/angstrom"
MTS_DRIVER = "mts-nve-respa"
SOURCE_FILES = ("trajectory.db", "events.jsonl", "manifest.json",
                "resolved_config.json")
CORRECTION_BACKENDS = ("scaled", "quadratic-corrected")


class ExportPairsError(RuntimeError):
    """The saved labels cannot be exported honestly (input/scope error)."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_evaluation_ids(evaluation_ids) -> list[int]:
    try:
        ids = [int(value) for value in evaluation_ids]
    except (TypeError, ValueError) as error:
        raise ExportPairsError(
            f"--evaluation-ids must be integers, got {error}") from error
    if not ids:
        raise ExportPairsError(
            "--evaluation-ids requires at least one evaluation id")
    invalid = [value for value in ids if value < 1]
    if invalid:
        raise ExportPairsError(
            f"evaluation ids must be >= 1, got {invalid}")
    if len(set(ids)) != len(ids):
        raise ExportPairsError(
            f"duplicate evaluation ids: {sorted(ids)}; each frame is "
            "exported once, in commit order")
    return ids


def _check_output(run_dir: Path, output: Path, force: bool) -> Path:
    if output.exists() or output.is_symlink():
        real_output = os.path.realpath(output)
        sources = {os.path.realpath(run_dir / name) for name in SOURCE_FILES}
        if real_output in sources:
            raise ExportPairsError(
                f"the output path resolves to a source file of the run: "
                f"{output}; refusing to touch it")
        if output.is_symlink():
            raise ExportPairsError(
                f"the output path is a symlink: {output}; refusing to "
                "write through it")
        if not force:
            raise ExportPairsError(
                f"output file exists: {output}; pass --force to overwrite "
                "or choose a different --output")
    run_real = os.path.realpath(run_dir)
    parent = output.parent if output.parent != Path("") else Path(".")
    real_output = os.path.realpath(parent) + os.sep + output.name
    if os.path.commonpath((run_real, real_output)) == run_real:
        raise ExportPairsError(
            f"the output must be OUTSIDE the source run directory "
            f"({run_real}): {output}")
    return output


def _check_backend_declarations(manifest: dict, resolved: dict) -> tuple[str, str]:
    """(reference backend name, fast backend name) — correction wrappers
    refused anywhere in the declared chain (nested included)."""
    surrogate_section = resolved.get("surrogate") or {}
    chain = []
    section = surrogate_section
    while section:
        name = section.get("backend")
        if name is None:
            break
        chain.append(str(name))
        base = (section.get("options") or {}).get("base")
        section = {"backend": base.get("name"),
                   "options": base.get("kwargs") or {}} \
            if isinstance(base, dict) else None
    wrappers = [name for name in chain if name in CORRECTION_BACKENDS]
    manifest_backend = ((manifest.get("surrogate") or {}).get("backend"))
    if manifest_backend in CORRECTION_BACKENDS and \
            manifest_backend not in wrappers:
        wrappers.append(str(manifest_backend))
    if wrappers:
        raise ExportPairsError(
            f"the run's fast backend is the correction wrapper "
            f"{wrappers[0]!r}: its stored labels are the CORRECTED "
            "model's output, not the raw base model's forces — this "
            "export never un-scales or strips correction terms, so "
            "scaled/quadratic-corrected runs are unsupported (export "
            "from an uncorrected fixed-model run instead)")
    if manifest_backend is None or str(manifest_backend) != (chain[0] if chain else None):
        raise ExportPairsError(
            "the manifest and the resolved config disagree on the fast "
            "backend identity; the run's records are not trusted "
            "verbatim")
    reference_backend = (manifest.get("reference") or {}).get("backend")
    if not reference_backend or not chain:
        raise ExportPairsError(
            "the run lacks reference or fast backend declarations in "
            "manifest.json/resolved_config.json; the saved records are "
            "insufficient to identify")
    return str(reference_backend), str(chain[0])


def _read_source(run_dir: Path, ids: list[int]) -> dict:
    missing = [name for name in SOURCE_FILES
               if not (run_dir / name).is_file()]
    if missing:
        raise ExportPairsError(
            f"{run_dir} is missing required record file(s) {missing}; "
            "export needs a completed mts-nve-respa run directory with "
            + ", ".join(SOURCE_FILES))
    hashes_before = {name: _sha256_file(run_dir / name)
                     for name in SOURCE_FILES}
    try:
        events = _read_events(run_dir / "events.jsonl")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        resolved = json.loads(
            (run_dir / "resolved_config.json").read_text())
    except (OSError, ValueError, KeyError) as error:
        raise ExportPairsError(
            f"cannot read the run's record files in {run_dir}: "
            f"{type(error).__name__}: {error}") from error
    starts = [event for event in events
              if event.get("type") == "run_start"]
    if not starts:
        raise ExportPairsError(
            f"{run_dir}: the event log has no run_start record; the "
            "run's driver and identities are unknown")
    drivers = {(start.get("workflow") or {}).get("driver")
               for start in starts}
    if drivers != {MTS_DRIVER}:
        raise ExportPairsError(
            f"this export covers only fixed-model {MTS_DRIVER!r} runs; "
            f"the run start records driver(s) {sorted(d for d in drivers if d)}")
    run_ids = {str(start.get("run_id")) for start in starts}
    reference_ids = {str(start.get("reference_id"))
                     for start in starts}
    model_ids = {str(start.get("model_id")) for start in starts}
    if len(run_ids) != 1 or len(reference_ids) != 1 or len(model_ids) != 1:
        raise ExportPairsError(
            "the event log mixes run identities across run_start records "
            "(a same-identity duplicate on resume is legal, an identity "
            "change is not); refusing to pair across identities")
    run_id = run_ids.pop()
    reference_id = reference_ids.pop()
    model_id = model_ids.pop()
    if not reference_id or not model_id:
        raise ExportPairsError(
            "the run_start record carries blank reference/model "
            "identities; the saved records are insufficient to identify")
    if str(manifest.get("run_id")) != run_id:
        raise ExportPairsError(
            "manifest.json's run_id disagrees with the event log; the "
            "run directory's records are contradictory")
    manifest_reference = (manifest.get("reference") or {}).get("fingerprint")
    manifest_model = (manifest.get("surrogate") or {}).get("fingerprint")
    if manifest_reference != reference_id:
        raise ExportPairsError(
            "the manifest's reference fingerprint disagrees with the "
            "event log's reference_id; refusing to trust the records "
            "verbatim")
    if not manifest_model or f"{manifest_model}#g0" != model_id:
        raise ExportPairsError(
            "the manifest's fast-model fingerprint disagrees with the "
            "event log's model_id (#g0 expected for this driver); "
            "refusing to trust the records verbatim")
    reference_backend, fast_backend = _check_backend_declarations(
        manifest, resolved)

    completed_steps = {int(event["step_id"]) for event in events
                       if event.get("type") == "step_completed"}
    with Store(run_dir / "trajectory.db", read_only=True) as store:
        try:
            committed = list(store.iter_committed(events, run_id))
        except (RuntimeError, KeyError, OSError) as error:
            raise ExportPairsError(
                f"the committed rows cannot be resolved against their "
                f"recorded bindings: {error}") from error

    selected: list[tuple[dict, object]] = []
    by_id: dict[int, tuple[dict, object]] = {}
    for commit, row in committed:
        context = commit.get("context") or {}
        evaluation_id = context.get("evaluation_id")
        if evaluation_id in by_id:
            raise ExportPairsError(
                f"evaluation_id {evaluation_id} is committed twice in "
                "this run; contradictory records are never paired")
        by_id[evaluation_id] = (commit, row)
    for evaluation_id in ids:
        pair = by_id.get(evaluation_id)
        if pair is None:
            raise ExportPairsError(
                f"evaluation_id {evaluation_id} has no committed record "
                "in this run; one missing frame aborts the whole export "
                "— nothing was written")
        commit, row = pair
        if commit.get("row_id") is None or commit.get("row_digest") is None:
            raise ExportPairsError(
                f"evaluation_id {evaluation_id}'s commit lacks an "
                "explicit row binding (row_id/row_digest); legacy "
                "step-based fallback is never used for pairing")
        selected.append(pair)

    frames = []
    for commit, row in selected:
        context = commit["context"]
        step_id = int(context["step_id"])
        is_initial = step_id == -1 or context.get("phase") == "initial"
        if not is_initial and step_id not in completed_steps:
            raise ExportPairsError(
                f"evaluation_id {context['evaluation_id']} (step "
                f"{step_id}) has no STEP_COMPLETED boundary; an "
                "evaluation-only record is not a pairing frame")
        kvp = row.key_value_pairs
        if str(kvp.get("run_id")) != run_id \
                or int(kvp["step"]) != step_id \
                or str(kvp.get("route")) != "mts":
            raise ExportPairsError(
                f"evaluation_id {context['evaluation_id']}: the bound "
                "row's identity does not match its commit context "
                "(run/step/route); refusing to pair contradictory "
                "records")
        if str(context.get("model_id")) != model_id:
            raise ExportPairsError(
                f"evaluation_id {context['evaluation_id']} carries a "
                "different model_id than the run start; identity changes "
                "are never paired")
        frames.append((commit, row))
    return {"events": events, "manifest": manifest, "resolved": resolved,
            "run_id": run_id, "reference_id": reference_id,
            "model_id": model_id,
            "reference_backend": reference_backend,
            "fast_backend": fast_backend,
            "frames": frames, "hashes_before": hashes_before}


def _frame_arrays(frames: list) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    reference_forces = []
    fast_forces = []
    positions = []
    numbers = None
    cell = None
    pbc = None
    records = []
    for commit, row in frames:
        context = commit["context"]
        engine = row.data.get("engine") or {}
        surrogate = row.data.get("surrogate") or {}
        label = context["evaluation_id"]
        if engine.get("forces") is None:
            raise ExportPairsError(
                f"evaluation_id {label}: the reference (engine) label "
                "is missing; one missing label aborts the whole export")
        if surrogate.get("forces") is None:
            raise ExportPairsError(
                f"evaluation_id {label}: the fast (surrogate) label "
                "is missing; one missing label aborts the whole export")
        ref = np.asarray(engine["forces"], dtype=np.float64)
        fast = np.asarray(surrogate["forces"], dtype=np.float64)
        atoms = row.toatoms()
        if ref.shape != (len(atoms), 3) or fast.shape != (len(atoms), 3):
            raise ExportPairsError(
                f"evaluation_id {label}: force arrays must be strictly "
                f"(N, 3) with N the frame's atom count, got "
                f"{ref.shape} / {fast.shape}")
        if not np.isfinite(ref).all() or not np.isfinite(fast).all():
            raise ExportPairsError(
                f"evaluation_id {label}: non-finite force values in the "
                "stored labels; refusing to export")
        if numbers is None:
            numbers = np.asarray(atoms.numbers)
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
            pbc = np.asarray(atoms.pbc, dtype=bool)
        else:
            if not np.array_equal(atoms.numbers, numbers):
                raise ExportPairsError(
                    f"evaluation_id {label}: the element order changes "
                    "across selected frames; pairing needs one fixed "
                    "atom order")
            if not np.array_equal(np.asarray(atoms.cell.array,
                                             dtype=np.float64), cell) \
                    or not np.array_equal(np.asarray(atoms.pbc,
                                                     dtype=bool), pbc):
                raise ExportPairsError(
                    f"evaluation_id {label}: the cell or pbc changes "
                    "across selected frames; pairing needs a fixed cell")
        reference_forces.append(ref)
        fast_forces.append(fast)
        positions.append(np.asarray(atoms.positions, dtype=np.float64))
        records.append({
            "evaluation_id": int(context["evaluation_id"]),
            "step": int(context["step_id"]),
            "row_id": int(commit["row_id"]),
            "physical_time_fs": float(context.get("physical_time_fs",
                                                  np.nan)),
        })
    return (np.asarray(reference_forces), np.asarray(fast_forces),
            positions), (numbers, cell, pbc), records


def export_pairs(run_dir: str | Path, *, evaluation_ids, output,
                 force: bool = False) -> dict:
    """Export saved reference/fast force labels of one MTS run to pairs.npz.

    ``evaluation_ids`` name stored context evaluation ids (the initial
    configuration is 1); the export is written in commit order.  Returns
    the report dict; raises :class:`ExportPairsError` on any input,
    identity or scope problem (nothing is written on failure).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ExportPairsError(f"run directory not found: {run_dir}")
    ids = _check_evaluation_ids(evaluation_ids)
    output = _check_output(run_dir, Path(output), force)
    source = _read_source(run_dir, ids)
    (reference_forces, fast_forces, positions), (numbers, cell, pbc), \
        records = _frame_arrays(source["frames"])
    hashes_after = {name: _sha256_file(run_dir / name)
                    for name in SOURCE_FILES}
    if hashes_after != source["hashes_before"]:
        changed = [name for name in SOURCE_FILES
                   if hashes_after[name] != source["hashes_before"][name]]
        raise ExportPairsError(
            f"source file(s) {changed} changed while being read; "
            "refusing to export from an unstable record")
    run_identity = ("pyramid-run-sha256:"
                    + hashlib.sha256(source["run_id"].encode("utf-8"))
                    .hexdigest())
    reference_out = ("pyramid-reference-sha256:"
                     + hashlib.sha256(source["reference_id"].encode("utf-8"))
                     .hexdigest())
    fast_out = ("pyramid-fast-sha256:"
                + hashlib.sha256(source["model_id"].encode("utf-8"))
                .hexdigest())
    frame_ids = [f"{run_identity}-evaluation-{record['evaluation_id']}"
                 for record in records]
    provenance = {
        "schema_version": PAIRS_SCHEMA_VERSION,
        "driver": MTS_DRIVER,
        "source_files": {name: source["hashes_before"][name]
                         for name in SOURCE_FILES},
        "run_identity": run_identity,
        "reference_backend": source["reference_backend"],
        "fast_backend": source["fast_backend"],
        "force_sources": {
            "reference_forces_eV_A": "store row data.engine.forces",
            "fast_forces_eV_A": "store row data.surrogate.forces"},
        "identity_encoding": {
            "reference_id": "pyramid-reference-sha256 = SHA256(UTF-8 of "
                            "the stored run_start.reference_id)",
            "fast_model_id": "pyramid-fast-sha256 = SHA256(UTF-8 of the "
                             "stored run_start.model_id)",
            "run_identity": "pyramid-run-sha256 = SHA256(UTF-8 of the "
                            "stored run_id); frame_ids append "
                            "'-evaluation-<id>'",
        },
        "frames": [dict(record, frame_id=frame_id)
                   for record, frame_id in zip(records, frame_ids)],
        "note": ("provenance-by-record plus saved source content hashes; "
                 "the stored row digest covers only parts of a row's "
                 "identity/driving fields and is not a cryptographic "
                 "certification of forces or configurations — this "
                 "export cannot detect arbitrary historical data "
                 "tampering or prove labels physically accurate"),
    }
    arrays = {
        "reference_forces_eV_A": reference_forces,
        "fast_forces_eV_A": fast_forces,
        "frame_ids": np.asarray(frame_ids),
        "reference_id": np.asarray(reference_out),
        "fast_model_id": np.asarray(fast_out),
        "force_unit": np.asarray(FORCE_UNIT),
        "numbers": numbers,
        "positions_A": np.asarray(positions),
        "cell_A": cell,
        "pbc": pbc,
        "provenance_json": np.asarray(json.dumps(
            provenance, sort_keys=True, allow_nan=False)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=output.parent,
                                    prefix=output.name + ".",
                                    suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {
        "schema_version": PAIRS_SCHEMA_VERSION,
        "status": "ok",
        "driver": MTS_DRIVER,
        "run_identity": run_identity,
        "frames": len(records),
        "n_atoms": len(numbers),
        "evaluation_ids": ids,
        "frame_ids": frame_ids,
        "reference_id": reference_out,
        "fast_model_id": fast_out,
        "force_unit": FORCE_UNIT,
        "output": str(output),
        "output_sha256": _sha256_file(output),
        "next_command": f"pyramid calibrate-scale --pairs {output} "
                        "--output scale.json",
    }
