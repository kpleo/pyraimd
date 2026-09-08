"""Offline capacity probe: readout-only vs deeper fine-tuning (post-7615761).

Smoke 7615761 showed the conformal calibration is fixed (q̂ ≈ 2, B/e ≈ 1)
but readout-only fine-tuning (2192 params on 0b3-medium — 0.03% of the
backbone) plateaus above ε_acc.  This probe measures held-out force accuracy
of three trainable sets on the 32 stored smoke labels, train/test split in
trajectory order (first N train / last 8 held out):

  - ``readout``:  ("readout",)                                  lr 1e-3
  - ``products``: ("readout", "products")                       lr 1e-3
  - ``full``:     ("readout", "products", "interactions", "node_embedding")
                                                                lr 1e-4
Single member per variant (accuracy question, not spread).  One Slurm array
task per variant; each writes its own JSON line to --out.

Usage:
    python experiments/capacity_probe.py --db loop.db --model 0b3.model \
        --variant readout --train-size 24 --out result.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate

VARIANTS = {
    "readout": (("readout",), 1e-3),
    "products": (("readout", "products"), 1e-3),
    "full": (("readout", "products", "interactions", "node_embedding"), 1e-4),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, type=Path, help="loop.db with stored labels")
    p.add_argument("--run-id", default="smoke-7615761")
    p.add_argument("--model", required=True, type=Path, help="single backbone path")
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--train-size", type=int, default=24)
    p.add_argument("--test-last", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=20250819)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    filters, lr = VARIANTS[args.variant]

    labels = list(Store(args.db).iter_labels(args.run_id))
    # Labels arrive in step order; step -1 is the initial evaluation.
    n_total = len(labels)
    n_train, n_test = args.train_size, args.test_last
    if n_train + n_test > n_total:
        raise ValueError(f"train({n_train})+test({n_test}) > {n_total} stored labels")
    train = labels[:n_train]
    test = labels[n_total - n_test :]

    committee = CommitteeSurrogate(
        model=str(args.model),
        n_members=1,
        seed=args.seed,
        epochs=args.epochs,
        lr=lr,
        trainable_filters=filters,
    )

    # Held-out accuracy of the zero-shot backbone first (the reference line).
    def heldout_errors() -> tuple[float, float]:
        maes, maxs = [], []
        for atoms, result in test:
            pred = committee.predict(atoms)
            d = np.linalg.norm(pred.forces - np.asarray(result.forces), axis=1)
            maes.append(float(d.mean()))
            maxs.append(float(d.max()))
        return float(np.mean(maes)), float(np.mean(maxs))

    mae0, max0 = heldout_errors()
    t0 = time.perf_counter()
    report = committee.finetune(train)
    wall = time.perf_counter() - t0
    mae1, max1 = heldout_errors()

    record = {
        "variant": args.variant,
        "filters": filters,
        "lr": lr,
        "train_size": n_train,
        "test_size": n_test,
        "epochs": args.epochs,
        "zero_shot": {"fmae": mae0, "fmax": max0},
        "fine_tuned": {"fmae": mae1, "fmax": max1},
        "train_final_loss": report.final_loss,
        "wall_time_s": wall,
    }
    with args.out.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
