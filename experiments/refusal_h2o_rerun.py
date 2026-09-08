"""Refusal-regime rerun at eps=0.10 with the CURRENT code (referee-driven).

The manuscript's refusal-regime paragraph (Sec. h2o) and the Table I refusal
row descend from run-1 (2026-08-19, superseded committee code + preliminary
spread estimator). The current run-2 artifact contradicts the premise for the
current committee (285/302 frames meet eps=0.10 after its learning
trajectory). This rerun replays the refusal experiment with the current code
(RMS spread estimator, member diversification, current fine-tune path) so the
refusal claims either reproduce with verified numbers or get honestly
rewritten.

Same protocol as the conformal replay: fresh committee (same seed), same
fine-tune trigger (every 8 labels), identical replay code path, eps_acc=0.10.
Every distrusted frame reveals its stored PySCF label.

Usage:
    uv run python experiments/refusal_h2o_rerun.py \
        --store analysis/h2o_streak/coverage_h2o.db \
        --out analysis/h2o_streak/refusal_eps010_report.json \
        --log analysis/h2o_streak/decision_log_refusal_eps010.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pyraimd2.loop import OnlineUpdater
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate
from pyraimd2.switch import ConformalSwitch, replay

COLLECT_RUN_ID = "collect-h2o-nve-300K"
SEED = 20250819
EPS_ACC = 0.10
ALPHA = 0.05
WINDOW = 64
W_MIN = 16
DELTA = 1e-3
N_LABEL = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--eps", type=float, default=EPS_ACC,
                   help="accuracy budget (default 0.10)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eps = args.eps
    frames = list(Store(args.store).iter_labels(COLLECT_RUN_ID))
    print(f"{len(frames)} labeled frames, eps={eps}", flush=True)

    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ConformalSwitch(committee, alpha=ALPHA, eps_acc=eps,
                             window=WINDOW, w_min=W_MIN, delta=DELTA)
    updater = OnlineUpdater(committee, observe=switch.observe, n_label=N_LABEL)
    records, summary = replay(frames, committee, switch, updater=updater,
                              eps_acc=eps)

    es = [r.error for r in records]
    bs = [r.bound for r in records if np.isfinite(r.bound)]
    report = {
        "eps_acc": eps, "alpha": ALPHA, "window": WINDOW,
        "w_min": W_MIN, "delta": DELTA, "n_label": N_LABEL, "seed": SEED,
        "code": "current (RMS spread estimator, member diversification)",
        "n_frames": summary.n_frames,
        "n_dft": summary.n_dft,
        "dft_fraction": summary.dft_fraction,
        "n_accepted": summary.n_accepted,
        "alpha_hat": summary.alpha_hat,
        "n_finetunes": summary.n_finetunes,
        "error_median": float(np.median(es)),
        "error_min": float(np.min(es)),
        "error_max": float(np.max(es)),
        "bound_median_finite": float(np.median(bs)),
        "late_frame_error_median": float(np.median(es[50:])),
        "late_frame_bound_median": float(np.median(bs[50:])),
        "wall_time_s": summary.wall_time_s,
    }
    args.out.write_text(json.dumps(report, indent=2))
    args.log.write_text(json.dumps(
        {"records": [asdict(r) for r in records]}, indent=1))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
