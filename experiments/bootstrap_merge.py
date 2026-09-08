"""Flagship-A bootstrap, stage 3b: merge member-parallel checkpoints.

The member-parallel stage-3 array trains each backbone as a K=1 committee
on its own node — exact, because CommitteeSurrogate.finetune trains members
independently (sequential per-member loops with per-member energy shifts).
This script merges the K=1 checkpoints into the production K=2 mixed
checkpoint and runs the temperature-stratified held-out gate on the
ensemble mean (the production quantity).

Usage:
    python experiments/bootstrap_merge.py \
        --member-checkpoint committee_bootstrap_m0.pt committee_bootstrap_m1.pt \
        --model 0b3.model mpa0.model \
        --labels sync/bootstrap_labels_*.jsonl --frames all.extxyz \
        --checkpoint committee_bootstrap.pt --report merge_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from ase.io import read

from pyraimd2.analysis import parse_boundaries, segment_held_out
from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate import CommitteeSurrogate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member-checkpoint", required=True, type=Path, nargs="+",
                   help="K=1 checkpoints in the same backbone order as --model")
    p.add_argument("--model", required=True, type=Path, nargs="+")
    p.add_argument("--labels", required=True, nargs="+")
    p.add_argument("--frames", required=True, type=Path)
    p.add_argument("--held-out-per-segment", type=int, default=4)
    p.add_argument("--segment-boundary", type=str, default="38",
                   help="comma-separated for multi-segment campaigns")
    p.add_argument("--seed", type=int, default=20250822)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.member_checkpoint) != len(args.model):
        raise ValueError("--member-checkpoint and --model must have equal length")

    records: dict[int, dict] = {}
    for pat in args.labels:
        for path in sorted(glob.glob(pat)):
            for line in Path(path).read_text().splitlines():
                rec = json.loads(line)
                records[int(rec["frame_index"])] = rec
    need = sorted(records)
    all_frames = read(args.frames, index=":")

    b, k = parse_boundaries(args.segment_boundary), args.held_out_per_segment
    idxs = np.array(need)
    held_idx = segment_held_out(idxs, b, k)
    test = [
        (
            all_frames[i],
            EngineResult(
                energy=float(records[i]["energy_ev"]),
                forces=np.asarray(records[i]["forces_ev_a"], dtype=float),
                stress=None,
                wall_time_s=float("nan"),
            ),
        )
        for i in need
        if i in held_idx
    ]
    print(f"held-out gate: {len(test)} frames {sorted(held_idx)}", flush=True)

    import torch

    # Probe-proven (7616841): in-process thread pinning is the only reliable
    # lever on the 474-atom box; 4 threads for the predict-only gate.
    torch.set_num_threads(4)

    member_states = [torch.load(p, map_location="cpu") for p in args.member_checkpoint]
    models = [str(m) for m in args.model]
    committee = CommitteeSurrogate(
        model=models,
        n_members=len(models),
        seed=args.seed,
        epochs=30,
        trainable_filters=("readout", "products"),
    )
    merged = dict(member_states[0])  # recipe echo template
    merged["model_specs"] = list(models)
    merged["n_members"] = len(models)
    merged["member_state_dicts"] = [s["member_state_dicts"][0] for s in member_states]
    merged["energy_shifts"] = [s["energy_shifts"][0] for s in member_states]
    committee.load_state_dict(merged)  # raises on backbone/K mismatch
    torch.save(committee.state_dict(), args.checkpoint)
    print(f"merged K={len(models)} checkpoint -> {args.checkpoint}", flush=True)

    maes, maxs = [], []
    for atoms, result in test:
        pred = committee.predict(atoms)
        d = np.linalg.norm(pred.forces - np.asarray(result.forces), axis=1)
        maes.append(float(d.mean()))
        maxs.append(float(d.max()))
    report = {
        "n_test": len(test),
        "held_out_indices": sorted(held_idx),
        "ensemble_fine_tuned": {
            "fmae": float(np.mean(maes)),
            "fmax": float(np.mean(maxs)),
            "per_frame_fmax": [round(x, 4) for x in maxs],
        },
        "member_checkpoints": [str(p) for p in args.member_checkpoint],
        "models": models,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
