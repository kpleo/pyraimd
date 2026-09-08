"""One fixed-recipe, zero-DFT adaptation screen on saved H2O labels.

Run only inside a Slurm allocation. All evaluation is retrospective development
validation: neither a blind forward test nor evidence of acceleration.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import cloud_hot_forward_20260905 as hot
import numpy as np

base = hot.base
ROOT = hot.ROOT
OUT = ROOT / "analysis/execution_20260905/h2o_hot_adapt"
BLOCKED_DFT_CALLS = 0
RECIPE = {
    "n_members": 4, "seed": 20250819, "epochs": 50,
    "perturbation": .01, "lr": .001, "force_weight": 10.,
    "trainable_filters": ["readout"],
}


def forbid_dft(*_args, **_kwargs):
    global BLOCKED_DFT_CALLS
    BLOCKED_DFT_CALLS += 1
    raise RuntimeError("Zero-DFT screening: PyscfEngine.compute is forbidden")


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def archive_record(index, archived):
    atoms, label = archived[index]
    return {
        "source": str(base.DB.relative_to(ROOT)), "archive_run": "collect-h2o-nve-300K",
        "archive_index": index, "positions": atoms.positions.tolist(),
        "reference_forces": np.asarray(label.forces).tolist(),
        "reference_energy_ev": float(label.energy), "template_archive_index": index,
    }


def path_records(path, start, count):
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if len(rows) != count or [row["step"] for row in rows] != list(range(count)):
        raise ValueError(f"Expected exactly {count} ordered states: {path}")
    return [{
        "source": str(path.relative_to(ROOT)), "step": row["step"],
        "time_fs": row["time_fs"], "positions": row["positions"],
        "reference_forces": row["reference_forces"],
        "reference_energy_ev": row.get("reference_energy_ev"),
        "template_archive_index": start,
    } for row in rows]


def atoms_for(record, archived):
    atoms = archived[record["template_archive_index"]][0].copy()
    atoms.positions = np.asarray(record["positions"], dtype=float)
    atoms.calc = None
    forces = np.asarray(record["reference_forces"], dtype=float)
    if forces.shape != atoms.positions.shape or not np.isfinite(forces).all():
        raise ValueError("Invalid saved reference forces")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Invalid saved geometry")
    return atoms


def predict(model, datasets, archived):
    output = {}
    for name, records in datasets.items():
        rows = []
        for record in records:
            p = model.predict(atoms_for(record, archived))
            residual = p.forces - np.asarray(record["reference_forces"])
            rows.append({
                "source": record["source"],
                "index": record.get("step", record.get("archive_index")),
                "forces_ev_A": p.forces.tolist(),
                "spread_per_atom_ev_A": p.uncertainty.tolist(),
                "spread_max_ev_A": float(p.uncertainty.max()),
                "energy_ev": float(p.energy),
                "error_max_atom_ev_A": base.force_norm(residual),
                "error_mean_atom_ev_A": float(np.linalg.norm(residual, axis=1).mean()),
            })
        output[name] = rows
    return output


def summarize(predictions):
    result = {}
    for name, rows in predictions.items():
        error = np.array([row["error_max_atom_ev_A"] for row in rows])
        result[name] = {
            "n_states": len(rows), "max_force_error_ev_A": float(error.max()),
            "mean_state_max_force_error_ev_A": float(error.mean()),
            "mean_atom_force_error_ev_A": float(np.mean([
                row["error_mean_atom_ev_A"] for row in rows])),
            "fraction_states_at_or_below_0_1_ev_A": float(np.mean(error <= base.EPS)),
        }
    return result


def q1_diagnostic(predictions):
    calibration = [(r["spread_max_ev_A"], r["error_max_atom_ev_A"])
                   for r in predictions["cold_calibration"]]
    q = base.qhat(calibration)
    paths = [{"seed": seed, "pairs": [
        (r["spread_max_ev_A"], r["error_max_atom_ev_A"])
        for r in predictions[f"old_hot_{seed}"]
    ]} for seed in [2026090501, 2026090502]]
    replay = hot.replay(paths, calibration, "calibrated", 1.)
    return {
        "qhat_from_32_cold_calibration_states": q,
        "fixed_initial_q_admissible_states": {
            str(p["seed"]): sum(q * (s + .001) <= base.EPS for s, _ in p["pairs"])
            for p in paths},
        "sequential_q1_replay": replay,
        "interpretation": (
            "Saved fixed paths only; replay uses existing revealed-label updates and audit RNG. "
            "q multiplier is fixed at 1, with no candidate selection. Usable means >=4/20 "
            "accepted states on each old path; this does not establish forward usability, "
            "a risk guarantee, or reference savings."
        ),
    }


def run():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Compute-node-only runner: a Slurm allocation is required")
    base.PyscfEngine.compute = forbid_dft
    hot.configure()
    base.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint = hot.OLD / "common_checkpoint.pt"
    ref_train = hot.OUT / "reference_2026090521_steps.jsonl"
    ref_validation = hot.OUT / "reference_2026090522_steps.jsonl"
    old_paths = [hot.OLD / f"velocity_{seed}_2_steps.json"
                 for seed in [2026090501, 2026090502]]
    inputs = [base.DB, ref_train, ref_validation, *old_paths,
              hot.OLD / "locked_protocol.json", hot.OUT / "locked_protocol.json"]
    sources = [Path(__file__), Path(__file__).with_suffix(".sbatch"),
               Path(hot.__file__), Path(base.__file__),
               ROOT / "src/pyraimd2/surrogate/committee.py"]
    try:
        if base.sha(checkpoint) != hot.CHECKPOINT_SHA or base.sha(base.MODEL) != hot.MODEL_SHA:
            raise ValueError("Original checkpoint/base-model identity mismatch")
        archived = base.archived()
        training = [archive_record(i, archived) for i in range(24)]
        training += path_records(ref_train, 0, 40)
        datasets = {
            "hot_reference_validation": path_records(ref_validation, 151, 40),
            "cold_calibration": [archive_record(i, archived) for i in range(64, 96)],
            "old_hot_2026090501": path_records(old_paths[0], 0, 20),
            "old_hot_2026090502": path_records(old_paths[1], 151, 20),
        }
        labels = []
        for record in training:
            atoms = atoms_for(record, archived)
            energy = record["reference_energy_ev"]
            if energy is None or not np.isfinite(energy):
                raise ValueError("Training requires a finite saved reference energy")
            labels.append((atoms, base.EngineResult(
                energy=float(energy), forces=np.asarray(record["reference_forces"], dtype=float),
                stress=None, wall_time_s=0.,
            )))
        training_keys = [base.key(atoms) for atoms, _ in labels]
        overlap = {name: sum(base.key(atoms_for(r, archived)) in training_keys for r in rows)
                   for name, rows in datasets.items()}
        if overlap["hot_reference_validation"] or overlap["cold_calibration"]:
            raise ValueError("Training geometry overlaps excluded validation/calibration set")
        write("saved_inputs.json", {
            "training": training, "validation": datasets,
            "templates": {str(i): {"numbers": archived[i][0].numbers.tolist(),
                                   "cell": archived[i][0].cell.tolist(),
                                   "pbc": archived[i][0].pbc.tolist()}
                          for i in sorted({r["template_archive_index"]
                                           for rows in [training, *datasets.values()] for r in rows})},
        })
        for path in sources:
            with (OUT / f"source_{path.name}").open("xb") as dest:
                dest.write(path.read_bytes())
        protocol = {
            "locked_unix": time.time(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "purpose": "Retrospective zero-DFT model-domain adaptation screen",
            "training": "Exactly archive frames 0..23 then reference seed 2026090521 steps 0..39",
            "n_training_labels": 64, "n_unique_training_geometries": len(set(training_keys)),
            "duplicate_policy": "Keep all 64 requested entries, including repeated initial geometry",
            "validation": {name: len(rows) for name, rows in datasets.items()},
            "validation_training_geometry_overlap_counts": overlap,
            "recipe": RECIPE, "fit_calls": 1, "torch_threads": 1,
            "fit_semantics": "Existing finetune resets each readout from foundation plus seeded noise",
            "new_dft_attempt_budget": 0, "epsilon_ev_A": base.EPS,
            "qhat_rule": "Existing qhat; 95% finite-sample rank, spread offset 0.001, 32 cold states",
            "q_multiplier": 1., "hyperparameter_selection": "none",
            "source_sha256": {str(p.relative_to(ROOT)): base.sha(p) for p in sources},
            "input_sha256": {str(p.relative_to(ROOT)): base.sha(p) for p in inputs},
            "saved_inputs_sha256": base.sha(OUT / "saved_inputs.json"),
            "original_checkpoint_sha256": hot.CHECKPOINT_SHA,
            "base_model_sha256": hot.MODEL_SHA,
            "checkpoint_relocation": "Existing hot.restored in-memory model_specs relocation only",
            "packages": {p: importlib.metadata.version(p)
                         for p in ["torch", "mace-torch", "e3nn", "numpy", "ase"]},
            "limitations": (
                "All available test data are now retrospective development validation. "
                "Old P2 paths include training anchor overlap and were generated with the old model. "
                "No new blind trajectories, controller changes, hyperparameter grid, or acceleration claim."
            ),
        }
        write("locked_protocol.json", protocol)
        model = hot.restored()
        state = model.model.state_dict()
        for name, expected in RECIPE.items():
            actual = list(state[name]) if name == "trainable_filters" else state[name]
            if actual != expected:
                raise ValueError(f"Unexpected training recipe: {name}")
        before = predict(model, datasets, archived)
        write("validation_pre.json", before)
        write("q1_pre.json", q1_diagnostic(before))
        with (OUT / "training.log").open("x") as stream, contextlib.redirect_stdout(stream):
            report = model.model.finetune(labels)
        write("training_report.json", asdict(report))
        with (OUT / "adapted_checkpoint.pt").open("xb") as stream:
            import torch
            torch.save(model.model.state_dict(), stream)
        after = predict(model, datasets, archived)
        write("validation_post.json", after)
        write("q1_post.json", q1_diagnostic(after))
        if base.sha(checkpoint) != hot.CHECKPOINT_SHA:
            raise RuntimeError("Original checkpoint changed during screening")
        summary = {
            "status": "complete", "pre": summarize(before), "post": summarize(after),
            "training": asdict(report), "new_dft_calls": 0,
            "blocked_dft_calls": BLOCKED_DFT_CALLS,
            "prediction_calls": model.calls, "prediction_seconds": model.seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "original_checkpoint_unchanged": True,
            "adapted_checkpoint_sha256": base.sha(OUT / "adapted_checkpoint.pt"),
            "protocol_sha256": base.sha(OUT / "locked_protocol.json"),
            "interpretation": protocol["limitations"],
        }
        write("summary.json", summary)
        print(json.dumps(summary, allow_nan=False), flush=True)
    except Exception as exc:
        write("failure.json", {"error": repr(exc), "blocked_dft_calls": BLOCKED_DFT_CALLS,
                               "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
