"""Flagship-A bootstrap, stage 3: offline committee pre-training + held-out
validation on the SCF-labeled frames.

Reads the labeling array's jsonl records (frame_index -> energy/forces) and
the source extxyz, reconstructs (atoms, EngineResult) pairs, trains the
production committee (K=2 mixed [0b3, MPA-0], readout+products, 30 epochs),
and reports held-out force accuracy on a temperature-stratified split
(both the 450 K and 300 K segments contribute train and test frames).
The trained committee is checkpointed for the warm-started production run.

Usage:
    python experiments/bootstrap_train.py \
        --labels sync/bootstrap_labels_*.jsonl --frames all.extxyz \
        --model 0b3.model mpa0.model --held-out-per-segment 4 \
        --checkpoint committee_bootstrap.pt --report report.json
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

EV_PER_RY = 13.605693122994  # not used; labels are already in eV / eV-per-A


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, nargs="+", help="labels jsonl files/globs")
    p.add_argument("--frames", required=True, type=Path, help="source all.extxyz")
    p.add_argument("--model", required=True, type=Path, nargs="+")
    p.add_argument("--n-members", type=int, default=None,
                   help="committee size with a single --model (default 2); the "
                        "member-parallel stage-3 array trains one backbone per "
                        "node with --n-members 1 (exact: finetune trains members "
                        "independently), merged afterwards by bootstrap_merge.py")
    p.add_argument("--held-out-per-segment", type=int, default=4,
                   help="held-out frames per condition segment (last k labeled "
                        "frames of each segment)")
    p.add_argument("--segment-boundary", type=str, default="38",
                   help="frame index/indices where segments 1..N-1 start; "
                        "comma-separated for multi-segment campaigns "
                        "(e.g. \"12,24,36\")")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--torch-threads", type=int, default=16,
                   help="in-process torch threads for the fine-tune phase; the "
                        "predict phase is pinned to 4. Probe 7616841 proved "
                        "in-process set_num_threads is the ONLY reliable lever: "
                        "env-set threads (16 or 28) both died on the same "
                        "28.96 GB einsum allocation on the 474-atom box, while "
                        "in-process 4 (predict) / 16 (finetune) ran clean "
                        "(VmHWM 6.9 / 41.5 GB)")
    p.add_argument("--seed", type=int, default=20250822)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    return p.parse_args()


def load_label_records(patterns: list[str]) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for line in Path(path).read_text().splitlines():
                rec = json.loads(line)
                records[int(rec["frame_index"])] = rec
    return records


def main() -> int:
    args = parse_args()
    records = load_label_records(args.labels)
    print(f"{len(records)} labeled frames: indices {sorted(records)[:3]}...{sorted(records)[-3:]}")

    need = sorted(records)
    all_frames = read(args.frames, index=":")
    frames = [all_frames[i] for i in need]

    labels = []
    for atoms, idx in zip(frames, need):
        rec = records[idx]
        labels.append(
            (
                atoms,
                EngineResult(
                    energy=float(rec["energy_ev"]),
                    forces=np.asarray(rec["forces_ev_a"], dtype=float),
                    stress=None,
                    wall_time_s=float(rec.get("wall_time_s", float("nan"))),
                ),
            )
        )

    # Condition-segment-stratified split: last K frames of each segment
    # held out (multi-segment campaigns pass comma-separated boundaries).
    bounds = parse_boundaries(args.segment_boundary)
    k = args.held_out_per_segment
    idxs = np.array(need)
    held_idx = segment_held_out(idxs, bounds, k)
    train = [p for p, i in zip(labels, need) if i not in held_idx]
    test = [p for p, i in zip(labels, need) if i in held_idx]
    seg_counts = [
        int(((idxs >= lo) & (idxs < hi)).sum())
        for lo, hi in zip([-np.inf, *bounds], [*bounds, np.inf])
    ]
    print(f"train {len(train)} / held-out {len(test)} "
          f"(labeled per segment: {seg_counts})")

    models = [str(m) for m in args.model]
    committee = CommitteeSurrogate(
        model=models if len(models) > 1 else models[0],
        n_members=len(models) if len(models) > 1 else (args.n_members or 2),
        seed=args.seed,
        epochs=args.epochs,
        trainable_filters=("readout", "products"),
    )

    import os

    import torch

    def vmhwm_gb() -> float:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1e6
        return float("nan")

    print(f"torch threads at import: intraop={torch.get_num_threads()} "
          f"interop={torch.get_num_interop_threads()} "
          f"env OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    # Probe-proven sequence (7616841): compile/predict at 4 in-process
    # threads, then widen for the fine-tune phase.
    torch.set_num_threads(4)

    def heldout() -> tuple[float, float]:
        maes, maxs = [], []
        for atoms, result in test:
            pred = committee.predict(atoms)
            d = np.linalg.norm(pred.forces - np.asarray(result.forces), axis=1)
            maes.append(float(d.mean()))
            maxs.append(float(d.max()))
        return float(np.mean(maes)), float(np.mean(maxs))

    mae0, max0 = heldout()
    print(f"zero-shot done: fMAE {mae0:.4f} fMax {max0:.4f}, "
          f"VmHWM {vmhwm_gb():.1f} GB", flush=True)
    torch.set_num_threads(args.torch_threads)
    print(f"fine-tune phase: torch threads -> {torch.get_num_threads()}",
          flush=True)
    report_ft = committee.finetune(train)
    print(f"fine-tune done: loss {report_ft.initial_loss:.4f} -> "
          f"{report_ft.final_loss:.4f}, VmHWM {vmhwm_gb():.1f} GB", flush=True)
    mae1, max1 = heldout()

    torch.save(committee.state_dict(), args.checkpoint)
    report = {
        "n_labels": len(labels),
        "n_train": len(train),
        "n_test": len(test),
        "held_out_indices": sorted(held_idx),
        "zero_shot": {"fmae": mae0, "fmax": max0},
        "fine_tuned": {"fmae": mae1, "fmax": max1},
        "train_final_loss": report_ft.final_loss,
        "member_losses": list(report_ft.member_losses),
        "wall_time_s": report_ft.wall_time_s,
        "epochs": args.epochs,
        "models": models,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
