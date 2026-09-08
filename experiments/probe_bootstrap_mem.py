"""Bootstrap stage-3 sizing probe: memory high-water mark + per-forward and
per-epoch timing for the 474-atom interface box at several torch thread
counts.

Motivation: the first stage-3 submission (7616839) OOMed in the zero-shot
held-out predict — the e3nn tensor-product einsum workspace is replicated
per torch thread (28 threads x ~1.03 GB ~= 29 GB on a 120 G node), while
committee_md frame generation ran the same forwards fine (no explicit
thread env). This probe measures, in ONE job, for T in the given thread
counts:

- wall time per committee.predict on a 474-atom frame (both members)
- process VmHWM after predicts
- wall time and VmHWM for one fine-tune epoch on a small label subset

The numbers size the real stage-3 job (threads vs 14 h wall tradeoff).

Usage:
    python experiments/probe_bootstrap_mem.py --labels sync/labels.jsonl \
        --frames all.extxyz --model 0b3.model mpa0.model \
        --thread-counts 4 8 16 --n-train-probe 4
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read

from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate import CommitteeSurrogate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, nargs="+")
    p.add_argument("--frames", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path, nargs="+")
    p.add_argument("--thread-counts", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--n-train-probe", type=int, default=4,
                   help="labels for the 1-epoch fine-tune timing probe")
    p.add_argument("--n-predict-probe", type=int, default=2,
                   help="held-out frames timed per thread count")
    p.add_argument("--seed", type=int, default=20250822)
    return p.parse_args()


def vmhwm_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1e6
    return float("nan")


def main() -> int:
    args = parse_args()
    records: dict[int, dict] = {}
    for pat in args.labels:
        for path in sorted(glob.glob(pat)):
            for line in Path(path).read_text().splitlines():
                rec = json.loads(line)
                records[int(rec["frame_index"])] = rec
    need = sorted(records)
    all_frames = read(args.frames, index=":")

    # Same temperature-stratified held-out as bootstrap_train: last 4 of
    # each segment (boundary 38).
    idxs = np.array(need)
    held = set(idxs[idxs < 38][-4:].tolist()) | set(idxs[idxs >= 38][-4:].tolist())
    test = [all_frames[i] for i in need if i in held]
    train = [
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
        if i not in held
    ][: args.n_train_probe]
    print(f"probe: {len(test)} held-out frames, {len(train)} train labels", flush=True)

    models = [str(m) for m in args.model]
    committee = CommitteeSurrogate(
        model=models if len(models) > 1 else models[0],
        n_members=len(models) if len(models) > 1 else 2,
        seed=args.seed,
        epochs=1,  # probe only times a single epoch
        trainable_filters=("readout", "products"),
    )

    import torch

    results: dict[int, dict] = {}
    for t in args.thread_counts:
        torch.set_num_threads(t)
        per_forward: list[float] = []
        for atoms in test[: args.n_predict_probe]:
            t0 = time.perf_counter()
            committee.predict(atoms)
            per_forward.append(time.perf_counter() - t0)
        results[t] = {
            "predict_s_per_frame": [round(x, 2) for x in per_forward],
            "vmhwm_gb_after_predicts": round(vmhwm_gb(), 2),
        }
        print(f"T={t}: predict {per_forward} s/frame, VmHWM "
              f"{results[t]['vmhwm_gb_after_predicts']} GB", flush=True)

    # Fine-tune timing at the largest requested thread count (the candidate
    # for the real job): one epoch on the small probe subset.
    t_best = max(args.thread_counts)
    torch.set_num_threads(t_best)
    t0 = time.perf_counter()
    report = committee.finetune(train)
    wall = time.perf_counter() - t0
    results[t_best]["finetune_1epoch_s"] = round(wall, 1)
    results[t_best]["finetune_s_per_label_epoch"] = round(wall / max(len(train), 1), 2)
    results[t_best]["vmhwm_gb_after_finetune"] = round(vmhwm_gb(), 2)
    print(f"T={t_best}: 1 epoch on {len(train)} labels = {wall:.0f} s "
          f"({wall / max(len(train), 1):.1f} s/label/epoch), VmHWM "
          f"{results[t_best]['vmhwm_gb_after_finetune']} GB, "
          f"loss {report.initial_loss:.3f} -> {report.final_loss:.3f}", flush=True)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
