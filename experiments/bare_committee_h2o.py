"""Bare-committee control on the water validation (manuscript bare-run todo).

The third switching baseline, after the scheduled ablation and the tuned
threshold: no supervision at all. A fresh committee (same foundation
backbone, same seed, never labeled, never fine-tuned) predicts every frame
of the 302-frame water trajectory; the miscoverage at the operating budget
is the fraction of frames whose zero-shot committee error exceeds eps.

This is the honest "do nothing" number: it isolates what the supervisor's
label economy buys, measured against the same stored DFT labels
(PySCF RKS PBE/def2-SVP) used by every other baseline. Caveat, stated in
the manuscript: the geometries are the certified replay's trajectory; a live
bare run would integrate on its own (poorer) forces and diverge from these
geometries, so this evaluates the bare committee on the certified
trajectory's frames.

Usage:
    uv run python experiments/bare_committee_h2o.py \
        --store analysis/h2o_streak/coverage_h2o.db \
        --out analysis/h2o_streak/bare_committee_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate

COLLECT_RUN_ID = "collect-h2o-nve-300K"
SEED = 20250819
EPS_ACC = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frames = list(Store(args.store).iter_labels(COLLECT_RUN_ID))
    print(f"{len(frames)} labeled frames", flush=True)

    committee = CommitteeSurrogate(n_members=4, seed=SEED)  # fresh, never trained
    rows = []
    for i, (atoms, label) in enumerate(frames):
        pred = committee.predict(atoms)
        e = float(np.linalg.norm(label.forces - pred.forces, axis=1).max())
        s = float(np.max(pred.uncertainty))
        rows.append({"step": i, "e": e, "s": s})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(frames)}  median e so far: "
                  f"{np.median([r['e'] for r in rows]):.3f}", flush=True)

    es = np.array([r["e"] for r in rows])
    eps_grid = np.round(np.arange(0.05, 0.51, 0.05), 2)
    curve = [{"eps": float(eps), "alpha_hat": float(np.mean(es > eps))}
             for eps in eps_grid]
    report = {
        "n_frames": len(frames),
        "committee": "fresh MACE-MP-0 committee, 4 members, seed "
                     f"{SEED}, zero labels, zero fine-tunes",
        "labels": "PySCF RKS PBE/def2-SVP (conv_tol 1e-9)",
        "eps_acc": EPS_ACC,
        "alpha_hat_at_eps": float(np.mean(es > EPS_ACC)),
        "e_stats": {"min": float(es.min()), "median": float(np.median(es)),
                    "max": float(es.max())},
        "miscoverage_curve": curve,
        "caveat": "geometries are the certified replay's trajectory; a live "
                  "bare run would diverge from them after its first steps",
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"bare alpha_hat @ eps={EPS_ACC}: {report['alpha_hat_at_eps']:.4f} "
          f"(e min/med/max {es.min():.3f}/{np.median(es):.3f}/{es.max():.3f})")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
