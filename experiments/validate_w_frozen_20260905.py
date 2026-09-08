"""P0: one local, immutable committee evaluation of the archived W24 geometries.

Run with uv run --no-sync python experiments/validate_w_frozen_20260905.py
--freeze, then --infer. No fitting, MD, reference engine, or network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/execution_20260905/tungsten"
SOURCE = ROOT / "analysis/materials_revision_20260905/tungsten"
REPORT = ROOT / "docs/execution_20260905/tungsten_results.md"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for variable, subdir in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "mplconfig"),
                         ("XDG_CACHE_HOME", "cache")):
    path = OUT / "runtime" / subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)


def deny_network(event, args):
    if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo"):
        raise RuntimeError("Network access disabled for frozen local W evaluation")


sys.addaudithook(deny_network)

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read, write  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_write(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_hash(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(json.dumps([key, str(a.dtype), list(a.shape)], separators=(",", ":")).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def geometry_hash(atoms):
    return array_hash({"numbers": np.asarray(atoms.numbers, dtype="<i8"),
                       "positions": np.asarray(atoms.positions, dtype="<f8"),
                       "cell": np.asarray(atoms.cell, dtype="<f8"),
                       "pbc": np.asarray(atoms.pbc, dtype="u1")})


def frame_context(index):
    if index < 36:
        return ("bulk300K", "bulk3000K", "bulk6000K")[index // 12], float(index % 12 * 10)
    return "spike", float((index - 36) * 25)


def warning_records(records):
    return [{"category": w.category.__name__, "message": str(w.message),
             "file": w.filename, "line": w.lineno} for w in records]


def load_committee(config):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from pyraimd2.surrogate import CommitteeSurrogate

    class CapturedCommittee(CommitteeSurrogate):
        def _forward(self, member, batch, training):
            assert training is False, "Training is forbidden"
            result, batch_dict = super()._forward(member, batch, training)
            self.captured.append({"forces": result["forces"].detach().cpu().numpy().copy(),
                                  "energy": float(result["energy"].detach().cpu().reshape(-1)[0])})
            return result, batch_dict

        def finetune(self, labels):
            raise RuntimeError("Fitting is forbidden in this frozen evaluation")

    committee = CapturedCommittee(**config)
    committee._ensure_loaded()  # Load and apply the archived seeded initialization; no forward.
    committee.captured = []
    assert 74 in committee._calc.z_table.zs
    assert committee._energy_shifts == [0.0, 0.0]
    for member in committee._models:
        for p in member.parameters():
            p.requires_grad_(False)  # Forces still differentiate positions, never parameters.
        assert all(torch.isfinite(t).all() for t in member.state_dict().values())
    return committee, {"torch_intraop_threads": torch.get_num_threads(),
                       "torch_interop_threads": torch.get_num_interop_threads(),
                       "environment_threads": {k: os.environ[k] for k in
                                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                                "NUMEXPR_NUM_THREADS")}}


def state_identity(committee):
    hashes = [array_hash({k: v.detach().cpu().numpy() for k, v in m.state_dict().items()})
              for m in committee._models]
    return {"member_tensor_sha256": hashes, "energy_shifts_ev": committee._energy_shifts,
            "r_max_A": float(committee._calc.r_max), "head": committee._calc.head,
            "supported_atomic_numbers": list(map(int, committee._calc.z_table.zs)),
            "energy_units_to_ev": float(committee._calc.energy_units_to_eV),
            "length_units_to_A": float(committee._calc.length_units_to_A),
            "parameter_counts": [sum(p.numel() for p in m.parameters()) for m in committee._models],
            "parameter_gradients_enabled": False,
            "member_training_flags": [m.training for m in committee._models]}


def freeze():
    plan_path = OUT / "frozen_plan.json"
    if plan_path.exists():
        raise RuntimeError("Frozen plan already exists; refusing to replace it")
    evidence = SOURCE / "evidence/remote"
    frame_path = evidence / "calculations/w_bootstrap_frames_432/all.extxyz"
    frames = read(frame_path, ":")
    assert len(frames) == 49
    files = sorted((evidence / "sync").glob("w_bootstrap_labels_432_t*.jsonl"))
    labels, label_origins, raw_lines = [], [], []
    for path in files:
        for line, raw in enumerate(path.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            forces = np.asarray(row["forces_ev_a"])
            assert forces.shape == (432, 3) and np.isfinite(forces).all()
            assert np.isfinite(row["energy_ev"])
            labels.append(row)
            label_origins.append({"source_path": str(path), "line_1based": line,
                                  "frame_index": row["frame_index"],
                                  "raw_record_sha256": hashlib.sha256(raw.encode()).hexdigest()})
            raw_lines.append(raw)
    assert len(labels) == 27 and len({r["frame_index"] for r in labels}) == 27
    order = sorted(range(27), key=lambda j: labels[j]["frame_index"])
    labels = [labels[j] for j in order]
    label_origins = [label_origins[j] for j in order]
    raw_lines = [raw_lines[j] for j in order]
    (OUT / "labels_27_original.jsonl").write_text("\n".join(raw_lines) + "\n")
    groups = {}
    for row in labels:
        idx = row["frame_index"]
        atoms = frames[idx]
        assert len(atoms) == 432 and np.all(atoms.numbers == 74) and np.all(atoms.pbc)
        groups.setdefault(geometry_hash(atoms), []).append(idx)
    geometries = []
    for j, (key, indices) in enumerate(groups.items()):
        idx = min(indices)
        segment, t = frame_context(idx)
        geometries.append({"geometry_id": f"W24_{j:02d}", "coordinate_sha256": key,
                           "representative_frame_index": idx, "label_frame_indices": indices,
                           "segment": "shared_initial" if len(indices) > 1 else segment,
                           "time_fs": t, "canonical_label_policy": "smallest frame index"})
    assert len(geometries) == 24
    assert [g["label_frame_indices"] for g in geometries if len(g["label_frame_indices"]) > 1] == [[0, 12, 24, 36]]
    core_ids = json.loads((SOURCE / "snapshot_manifest.json").read_text())[0]["initial_core_atom_indices"]
    core = np.isin(np.arange(432), core_ids)
    d = frames[0].positions - np.diag(frames[0].cell) / 2
    assert np.array_equal(core, np.linalg.norm(d, axis=1) < 5.5) and core.sum() == 40
    unique_atoms = []
    for g in geometries:
        old = frames[g["representative_frame_index"]]
        atoms = Atoms(numbers=old.numbers, positions=old.positions, cell=old.cell, pbc=old.pbc)
        atoms.info.update(geometry_id=g["geometry_id"], coordinate_sha256=g["coordinate_sha256"],
                          representative_frame_index=g["representative_frame_index"],
                          label_frame_indices=" ".join(map(str, g["label_frame_indices"])))
        atoms.set_array("initial_core", core.astype(int))
        unique_atoms.append(atoms)
    write(OUT / "geometries_24.extxyz", unique_atoms)
    roundtrip = read(OUT / "geometries_24.extxyz", ":")
    assert [geometry_hash(a) for a in roundtrip] == [g["coordinate_sha256"] for g in geometries]
    np.savez_compressed(OUT / "geometry_reference_archive.npz",
                        positions=np.stack([a.positions for a in unique_atoms]),
                        cells=np.stack([a.cell.array for a in unique_atoms]),
                        numbers=np.stack([a.numbers for a in unique_atoms]),
                        pbc=np.stack([a.pbc for a in unique_atoms]), initial_core=core,
                        frame_indices_27=[r["frame_index"] for r in labels],
                        reference_forces_27=np.asarray([r["forces_ev_a"] for r in labels]),
                        reference_energies_27=[r["energy_ev"] for r in labels])
    mapping = []
    for j, (row, origin) in enumerate(zip(labels, label_origins)):
        g = next(g for g in geometries if row["frame_index"] in g["label_frame_indices"])
        segment, t = frame_context(row["frame_index"])
        mapping.append({"label_row_0based": j, **origin, "geometry_id": g["geometry_id"],
                        "segment": segment, "time_fs": t,
                        "is_canonical_geometry_label": row["frame_index"] == g["representative_frame_index"],
                        "use": "evaluation_only; no fitting, calibration, model selection, or energy shift"})
    csv_write(OUT / "label_geometry_map_27.csv", mapping)
    json_write(OUT / "geometry_manifest_24.json", geometries)
    status = json.loads((SOURCE / "status.json").read_text())
    missing = []
    for idx in range(49):
        if idx in {r["frame_index"] for r in labels}:
            continue
        state = ("failed_reference_attempt" if idx in status["failed_indices_latest_array"] else
                 "running_at_archive_snapshot" if idx in status["active_frame_indices"] else
                 "not_started_at_archive_snapshot")
        segment, t = frame_context(idx)
        missing.append({"frame_index": idx, "segment": segment, "time_fs": t,
                        "status_at_archive": state, "archive_time_utc": status["captured_at"],
                        "current_remote_status": "not_queried", "new_inference": "not_requested_without_label"})
    csv_write(OUT / "missing_labels_22.csv", missing)
    model_paths = [Path("/Users/pengkang/.cache/mace/macemp0b3mediummodel"),
                   Path("/Users/pengkang/.cache/mace/macempa0mediummodel")]
    assert all(p.is_file() for p in model_paths), "Only existing local models permitted"
    config = {"model": list(map(str, model_paths)), "device": "cpu", "default_dtype": "float64",
              "n_members": 2, "seed": 20260834, "perturbation": 0.01,
              "trainable_filters": ["readout", "products"]}
    provenance_paths = [frame_path, *files, SOURCE / "status.json", SOURCE / "snapshot_manifest.json",
                        ROOT / "src/pyraimd2/surrogate/committee.py",
                        evidence / "software/pyraimd2/src/pyraimd2/surrogate/committee.py",
                        evidence / "software/pyraimd2/experiments/committee_md.py",
                        evidence / "software/pyraimd2/hpc/neimeng/submit/w_bootstrap_frames_432.sbatch",
                        ROOT / "hpc/neimeng/submit/w_spike_prod_432.sbatch",
                        ROOT / "docs/materials_revision_20260905/tungsten_findings.md",
                        ROOT / "docs/submission_plan_20260905/NCS_PRX_upgrade_plan_zh.md",
                        ROOT / "pyproject.toml", ROOT / "uv.lock", Path(__file__)]
    manifest = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in provenance_paths]
    json_write(OUT / "source_manifest.json", manifest)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        committee, threads = load_committee(config)
        identity = state_identity(committee)
    json_write(OUT / "freeze_warnings.json", warning_records(caught))
    model_files = [{"name": name, "path": str(p), "sha256": sha(p), "bytes": p.stat().st_size,
                    "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
                   for name, p in zip(["MACE-MP-0b3-medium", "MACE-MPA-0-medium"], model_paths)]
    plan = {"frozen_at_utc": now(), "status": "frozen_before_any_forward_prediction",
            "evaluation": "current frozen-model predictions, NOT historical online errors",
            "committee_constructor": config, "model_files": model_files, "effective_model_identity": identity,
            "seed_policy": "Single fixed seed 20260831+3=20260834 from archived W432 spike sampler, applied to all 24 geometries. Original bulk segment seeds differ; no historical reconstruction claimed.",
            "initialization": "Archived API: member 0 clean MP-0b3, member 1 MPA-0 with 1% seeded tensor-std readout/products perturbation. No optimizer or training. Parameters then frozen.",
            "model_identity_limits": "Names/cache provenance match original sampling models; remote historical binary hashes unavailable. Current local files and effective tensor hashes are authoritative for this run.",
            "labels_usage": {"local_training": 0, "calibration": 0, "model_selection": 0,
                             "energy_offset_fitting": 0, "evaluation_records": 27, "evaluation_geometries": 24,
                             "foundation_pretraining_overlap": "unknown; not audited; no claim of globally unseen structures"},
            "trained_W_checkpoint": "not_found_in_existing_W_inventory; not used",
            "geometry_manifest": geometries, "initial_core_atom_indices_0based": core_ids,
            "geometry_hash_recipe": "SHA256 sorted field header [name,numpy_dtype,shape] compact JSON plus contiguous little-endian bytes; numbers int64, positions/cell float64, pbc uint8; no momenta",
            "tensor_hash_recipe": "SHA256 sorted state_dict tensor names, NumPy dtype and shape compact JSON headers, then contiguous bytes; includes parameters and buffers",
            "force_budget_ev_A": 0.2, "budget_source": "Existing W432 production script EPS_ACC default; fixed before inference, not a universal physical tolerance",
            "primary_metric": "e=max_i ||Fmean_i-F_QE_i||_2; exceed iff e>0.2 eV/A",
            "spread_definition": "sigma_i=sqrt(mean_k ||F_ik-Fmean_i||^2), ddof=0; scalar s=max_i sigma_i",
            "ratio_diagnostic": "r=e/(s+0.001); atom ratios use e_i/(sigma_i+0.001), descriptive only, no fitted calibration or probability guarantee",
            "grouping": "40 fixed initial-core atom IDs, 392 matrix; on bulk segments the core is a geometrically matched region, not a hot region",
            "deduplication": "0/12/24/36 share one geometry and one prediction; all four QE records retained. Primary 24-geometry aggregation uses the minimum-frame-index label, with duplicate-label differences reported separately.",
            "primary_energy_metric": "None. Raw totals retained only; foundation/QE references differ and no energy shift is fit.",
            "physical_limits": ["QE PBE+D3 reference versus original MACE committee without added D3; discrepancy belongs to this force comparison and cannot be isolated here",
                                "Successful partial labels are selected and temporally correlated; no iid confidence intervals or generalization/risk certificate",
                                "Only 24 archived 432-atom geometries; no 686 force validation, cutoff validation of hot W, melting or defect-yield claim"],
            "run_policy": {"forward_predictions": 24, "members_per_prediction": 2,
                           "max_attempts_per_geometry": 1, "retries": 0, "warmup_predictions": 0,
                           "on_geometry_failure": "record traceback and continue remaining geometries once",
                           "on_invalid_model": "report failure; no alternate model search, fitting, downloads or new DFT",
                           "network": "blocked by audit hook", "device": "cpu", **threads},
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "ase", "torch", "mace-torch", "e3nn")},
            "platform": platform.platform(), "python": sys.version,
            "missing_reference_count": len(missing), "source_manifest": manifest,
            "frozen_artifacts": [{"path": str(OUT / f), "sha256": sha(OUT / f)} for f in
                                 ("labels_27_original.jsonl", "geometries_24.extxyz", "geometry_reference_archive.npz",
                                  "label_geometry_map_27.csv", "geometry_manifest_24.json", "missing_labels_22.csv")]}
    plan["model_identity_sha256"] = hashlib.sha256(json.dumps(
        {"config": config, "files": model_files, "effective": identity}, sort_keys=True).encode()).hexdigest()
    json_write(plan_path, plan)
    (OUT / "frozen_plan.sha256").write_text(sha(plan_path) + "  frozen_plan.json\n")
    print(json.dumps({"frozen_plan": str(plan_path), "sha256": sha(plan_path),
                      "geometries": 24, "labels": 27, "identity": identity["member_tensor_sha256"]}), flush=True)


def stats(error_vectors, spreads, references, core, budget):
    rows = {}
    for name, mask in (("all", np.ones(432, bool)), ("core", core), ("matrix", ~core)):
        errors = np.linalg.norm(error_vectors[..., mask, :], axis=-1)
        spread = spreads[..., mask]
        refnorm = np.linalg.norm(references[..., mask, :], axis=-1)
        rows[name] = {"atom_evaluations": int(errors.size),
                      "force_error_vector_rms_ev_A": float(np.sqrt(np.mean(errors**2))),
                      "force_error_component_rmse_ev_A": float(np.sqrt(np.mean(errors**2)/3)),
                      "force_error_vector_mae_ev_A": float(errors.mean()),
                      "force_error_max_ev_A": float(errors.max()),
                      "force_error_p95_ev_A": float(np.quantile(errors, .95)),
                      "atoms_exceed_budget": int(np.count_nonzero(errors > budget)),
                      "atom_exceed_fraction": float(np.mean(errors > budget)),
                      "spread_rms_ev_A": float(np.sqrt(np.mean(spread**2))),
                      "spread_max_ev_A": float(spread.max()),
                      "reference_force_vector_rms_ev_A": float(np.sqrt(np.mean(refnorm**2))),
                      "atoms_error_greater_than_spread": int(np.count_nonzero(errors > spread))}
    return rows


def analyze(plan, predictions, attempts, started, elapsed, verification):
    labels = [json.loads(line) for line in (OUT / "labels_27_original.jsonl").read_text().splitlines()]
    label_by_index = {r["frame_index"]: r for r in labels}
    geometries = plan["geometry_manifest"]
    atoms24 = read(OUT / "geometries_24.extxyz", ":")
    core = np.isin(np.arange(432), plan["initial_core_atom_indices_0based"])
    eps = plan["force_budget_ev_A"]
    label_rows, per_atom_rows, frame_rows, canonical_errors, canonical_spreads, canonical_refs = [], [], [], [], [], []
    residuals27 = np.full((27, 432, 3), np.nan)
    label_prediction_rows = np.full(27, -1, int)
    segment_data = {}
    for gi, (g, atoms) in enumerate(zip(geometries, atoms24)):
        if gi not in predictions:
            continue
        p = predictions[gi]
        ref = np.asarray(label_by_index[g["representative_frame_index"]]["forces_ev_a"])
        residual = p["mean_forces"] - ref
        errors = np.linalg.norm(residual, axis=1)
        group_stats = stats(residual, p["spread"], ref, core, eps)
        worst = int(np.argmax(errors))
        row = {"geometry_id": g["geometry_id"], "representative_frame_index": g["representative_frame_index"],
               "label_frame_indices": ";".join(map(str, g["label_frame_indices"])),
               "segment": g["segment"], "time_fs": g["time_fs"],
               "e_max_ev_A": float(errors.max()), "s_max_ev_A": float(p["spread"].max()),
               "r_frame_delta_0p001": float(errors.max()/(p["spread"].max()+.001)),
               "exceeds_0p2_ev_A": bool(errors.max() > eps), "worst_atom_index_0based": worst,
               "worst_atom_region": "core" if core[worst] else "matrix",
               "core_error_vector_rms_ev_A": group_stats["core"]["force_error_vector_rms_ev_A"],
               "matrix_error_vector_rms_ev_A": group_stats["matrix"]["force_error_vector_rms_ev_A"],
               "all_error_vector_rms_ev_A": group_stats["all"]["force_error_vector_rms_ev_A"],
               "raw_mean_energy_ev": p["mean_energy"], "reference_energy_ev": label_by_index[g["representative_frame_index"]]["energy_ev"],
               "inference_wall_s": p["wall_s"]}
        frame_rows.append(row)
        canonical_errors.append(residual)
        canonical_spreads.append(p["spread"])
        canonical_refs.append(ref)
        segment_data.setdefault(g["segment"], []).append((residual, p["spread"], ref, row))
        displacement = atoms.positions - atoms24[0].positions
        cell_diagonal = np.diag(atoms.cell)
        displacement -= np.rint(displacement/cell_diagonal)*cell_diagonal
        displacement -= displacement.mean(axis=0)
        for ai in range(432):
            arow = {"geometry_id": g["geometry_id"], "representative_frame_index": g["representative_frame_index"],
                    "segment": g["segment"], "time_fs": g["time_fs"], "atom_index_0based": ai,
                    "atom_id_1based": ai+1, "element": "W", "region": "core" if core[ai] else "matrix",
                    "x_A": float(atoms.positions[ai,0]), "y_A": float(atoms.positions[ai,1]), "z_A": float(atoms.positions[ai,2]),
                    "displacement_from_initial_COM_removed_A": float(np.linalg.norm(displacement[ai]))}
            for prefix, value in (("reference", ref), ("prediction", p["mean_forces"]), ("residual", residual),
                                  ("member0", p["member_forces"][0]), ("member1", p["member_forces"][1])):
                for axis, component in zip("xyz", value[ai]):
                    arow[f"{prefix}_f{axis}_ev_A"] = float(component)
            arow.update(error_norm_ev_A=float(errors[ai]), spread_ev_A=float(p["spread"][ai]),
                        error_over_spread_delta_0p001=float(errors[ai]/(p["spread"][ai]+.001)),
                        exceeds_0p2_ev_A=bool(errors[ai] > eps))
            per_atom_rows.append(arow)
        for lj, label in enumerate(labels):
            if label["frame_index"] not in g["label_frame_indices"]:
                continue
            lr = p["mean_forces"] - np.asarray(label["forces_ev_a"])
            le = np.linalg.norm(lr, axis=1)
            residuals27[lj] = lr
            label_prediction_rows[lj] = gi
            segment, t = frame_context(label["frame_index"])
            label_rows.append({"frame_index": label["frame_index"], "geometry_id": g["geometry_id"],
                               "segment": segment, "time_fs": t, "e_max_ev_A": float(le.max()),
                               "error_vector_rms_ev_A": float(np.sqrt(np.mean(le**2))),
                               "exceeds_0p2_ev_A": bool(le.max() > eps),
                               "is_canonical_geometry_label": label["frame_index"] == g["representative_frame_index"]})
    csv_write(OUT / "per_atom_residuals_spread_24.csv", per_atom_rows)
    csv_write(OUT / "per_geometry_metrics_24.csv", frame_rows)
    csv_write(OUT / "per_label_metrics_27.csv", sorted(label_rows, key=lambda r: r["frame_index"]))
    np.savez_compressed(OUT / "all_27_label_residuals.npz", residuals_ev_A=residuals27,
                        label_frame_indices=[r["frame_index"] for r in labels], geometry_row=label_prediction_rows,
                        label_prediction_succeeded=label_prediction_rows>=0, initial_core=core)
    duplicates = []
    for idx in (0, 12, 24, 36):
        diff = np.asarray(label_by_index[idx]["forces_ev_a"]) - np.asarray(label_by_index[0]["forces_ev_a"])
        duplicates.append({"frame_index": idx, "force_difference_from_frame0_max_ev_A": float(np.linalg.norm(diff,axis=1).max()),
                           "force_difference_from_frame0_vector_rms_ev_A": float(np.sqrt(np.mean(np.sum(diff**2,axis=1)))),
                           "energy_difference_from_frame0_ev": label_by_index[idx]["energy_ev"]-label_by_index[0]["energy_ev"]})
    json_write(OUT / "duplicate_initial_labels.json", duplicates)
    summary = {"evaluation": plan["evaluation"], "started_at_utc": started, "finished_at_utc": now(),
               "frozen_plan_sha256": sha(OUT / "frozen_plan.json"), "model_identity_sha256": plan["model_identity_sha256"],
               "attempted_geometries": len(attempts), "successful_geometries": len(predictions),
               "failed_geometries": [a for a in attempts if a["status"] != "success"],
               "reference_records": 27, "unique_reference_geometries": 24, "missing_reference_frames": 22,
               "atoms_per_geometry": 432, "core_atoms": 40, "matrix_atoms": 392,
               "force_budget_ev_A": eps, "new_DFT": 0, "training_calls": 0, "downloaded_models": 0,
               "model_or_threshold_selection_using_labels": False,
               "inference_wall_s": sum(p["wall_s"] for p in predictions.values()),
               "elapsed_before_analysis_s": elapsed, "verification": verification,
               "canonical_label_policy": "minimum frame index; duplicates retained separately", "duplicate_label_comparison": duplicates,
               "groups": {}, "by_segment_excluding_repeated_initial": {}, "key_frames": {},
               "physical_limits": plan["physical_limits"]}
    if predictions:
        summary["groups"] = stats(np.stack(canonical_errors), np.stack(canonical_spreads), np.stack(canonical_refs), core, eps)
        summary["geometries_exceed_budget"] = sum(r["exceeds_0p2_ev_A"] for r in frame_rows)
        summary["label_records_exceed_budget"] = sum(r["exceeds_0p2_ev_A"] for r in label_rows)
        summary["worst_geometry"] = max(frame_rows, key=lambda r: r["e_max_ev_A"])
        summary["frame_error_max_median_ev_A"] = float(np.median([r["e_max_ev_A"] for r in frame_rows]))
        summary["frame_error_to_spread_ratio_range"] = [min(r["r_frame_delta_0p001"] for r in frame_rows), max(r["r_frame_delta_0p001"] for r in frame_rows)]
        for segment, entries in segment_data.items():
            summary["by_segment_excluding_repeated_initial"][segment] = {
                "geometry_count": len(entries), "exceed_count": sum(t[3]["exceeds_0p2_ev_A"] for t in entries),
                "groups": stats(np.stack([t[0] for t in entries]), np.stack([t[1] for t in entries]),
                                np.stack([t[2] for t in entries]), core, eps)}
        for idx in (0, 38, 40, 45, 47):
            gi = next((i for i,g in enumerate(geometries) if idx in g["label_frame_indices"]), None)
            if gi is not None and gi in predictions:
                p = predictions[gi]
                ref = np.asarray(label_by_index[idx]["forces_ev_a"])
                summary["key_frames"][str(idx)] = stats(p["mean_forces"]-ref, p["spread"], ref, core, eps)
    json_write(OUT / "summary.json", summary)
    return summary


def infer():
    if (OUT / "inference_started.json").exists():
        raise RuntimeError("An inference attempt already exists; refusing a second run")
    plan = json.loads((OUT / "frozen_plan.json").read_text())
    assert sha(OUT / "frozen_plan.json") == (OUT / "frozen_plan.sha256").read_text().split()[0]
    for entry in plan["model_files"] + plan["source_manifest"] + plan["frozen_artifacts"]:
        assert sha(entry["path"]) == entry["sha256"], f"Frozen input changed: {entry['path']}"
    started, t0 = now(), time.perf_counter()
    json_write(OUT / "inference_started.json", {"at_utc": started, "plan_sha256": sha(OUT / "frozen_plan.json")})
    predictions, attempts = {}, []
    verification = {}
    caught_all = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            committee, threads = load_committee(plan["committee_constructor"])
            identity_before = state_identity(committee)
        caught_all.extend(warning_records(caught))
        assert identity_before == plan["effective_model_identity"], "Effective frozen model mismatch"
        verification.update(model_identity_matches_plan=True, runtime_threads=threads)
        frames = read(OUT / "geometries_24.extxyz", ":")
        (OUT / "predictions").mkdir(exist_ok=True)
        for gi, (atoms, geometry) in enumerate(zip(frames, plan["geometry_manifest"])):
            assert geometry_hash(atoms) == geometry["coordinate_sha256"]
            # Fresh Atoms carries no labels, momenta or annotations into the model.
            clean = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
            attempt = {"geometry_id": geometry["geometry_id"], "frame_index": geometry["representative_frame_index"],
                       "started_at_utc": now(), "attempt_number": 1}
            ti = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    committee.captured = []
                    pred = committee.predict(clean)
                caught_all.extend(warning_records(caught))
                elapsed = time.perf_counter()-ti
                assert len(committee.captured) == 2
                member_forces = np.stack([x["forces"] for x in committee.captured]) * identity_before["energy_units_to_ev"] / identity_before["length_units_to_A"]
                member_energies = np.asarray([x["energy"] for x in committee.captured]) * identity_before["energy_units_to_ev"]
                spread_check = np.sqrt(np.mean(np.sum((member_forces-member_forces.mean(axis=0))**2,axis=2),axis=0))
                assert pred.forces.shape == (432,3) and pred.uncertainty.shape == (432,)
                assert np.isfinite(pred.forces).all() and np.isfinite(pred.uncertainty).all() and np.isfinite(pred.energy)
                assert np.isfinite(member_forces).all() and np.isfinite(member_energies).all()
                assert np.allclose(pred.forces, member_forces.mean(axis=0), atol=1e-12, rtol=1e-12)
                assert np.allclose(pred.uncertainty, spread_check, atol=1e-12, rtol=1e-12)
                assert np.allclose(pred.energy, member_energies.mean(), atol=1e-10, rtol=1e-12)
                payload = {"mean_forces": pred.forces.copy(), "spread": pred.uncertainty.copy(),
                           "mean_energy": pred.energy, "member_forces": member_forces,
                           "member_energies": member_energies, "wall_s": elapsed}
                np.savez_compressed(OUT / "predictions" / f"{geometry['geometry_id']}.npz", **payload)
                predictions[gi] = payload
                attempt.update(status="success", wall_s=elapsed)
                print(f"[{gi+1}/24] frame {attempt['frame_index']} prediction saved; {elapsed:.2f}s", flush=True)
            except Exception as error:
                attempt.update(status="failed", wall_s=time.perf_counter()-ti,
                               error=repr(error), traceback=traceback.format_exc())
                print(f"[{gi+1}/24] FAILED frame {attempt['frame_index']}: {error}", flush=True)
            attempts.append(attempt)
            json_write(OUT / "inference_attempts.json", attempts)
        verification["effective_tensors_unchanged_after_all_predictions"] = state_identity(committee) == identity_before
        verification["model_files_unchanged_after_all_predictions"] = all(sha(p["path"]) == p["sha256"] for p in plan["model_files"])
        assert verification["effective_tensors_unchanged_after_all_predictions"]
        assert verification["model_files_unchanged_after_all_predictions"]
        verification["means_and_population_spread_independently_checked"] = True
        verification["one_prediction_per_geometry_no_warmups_no_retries"] = True
    except Exception as error:
        verification["fatal_error"] = repr(error)
        verification["fatal_traceback"] = traceback.format_exc()
        json_write(OUT / "fatal_error.json", verification)
    json_write(OUT / "inference_warnings.json", caught_all)
    summary = analyze(plan, predictions, attempts, started, time.perf_counter()-t0, verification)
    json_write(OUT / "inference_finished.json", {"at_utc": now(), "successes": len(predictions), "attempts": len(attempts)})
    print(json.dumps({"successful_geometries": len(predictions),
                      "exceed_budget": summary.get("geometries_exceed_budget"),
                      "worst_geometry": summary.get("worst_geometry"),
                      "inference_wall_s": summary["inference_wall_s"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        infer()


if __name__ == "__main__":
    main()
