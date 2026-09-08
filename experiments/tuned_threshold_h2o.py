"""Tuned-threshold baseline on the water validation (red-team review E6):
the fair incumbent, replacing the scheduled-fallback strawman.

The scheduled ablation in the manuscript (call the engine every 4th frame)
is nobody's production practice. The real incumbent is a DP-GEN-style raw
spread threshold, tuned so its label budget matches the calibrated
supervisor's (DFT fraction 0.252 on the 302-frame water trajectory). This
script sweeps the threshold tau over the spread distribution, replays each
point self-consistently (fresh committee, same seed, same fine-tune
trigger, identical replay code path), and reports the tuned point's
miscoverage alpha_hat against the conformal supervisor's zero.

Usage:
    uv run python experiments/tuned_threshold_h2o.py \
        --store analysis/h2o_streak/coverage_h2o.db \
        --out analysis/h2o_streak/tuned_threshold_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyraimd2.loop import OnlineUpdater
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate
from pyraimd2.switch import ThresholdSwitch, replay

COLLECT_RUN_ID = "collect-h2o-nve-300K"
SEED = 20250819
EPS_ACC = 0.25  # the conformal run's budget (violation criterion)
N_LABEL = 8
TARGET_DFT_FRACTION = 0.252  # conformal supervisor's label budget


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--taus", type=float, nargs="+",
                   default=None, help="explicit sweep values; default: spread "
                                       "percentile grid around the target budget")
    return p.parse_args()


def run_at_tau(frames, tau: float):
    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ThresholdSwitch(committee, tau)
    updater = OnlineUpdater(committee, observe=lambda s, e: None, n_label=N_LABEL)
    return replay(frames, committee, switch, updater=updater, eps_acc=EPS_ACC)


def main() -> int:
    args = parse_args()
    frames = list(Store(args.store).iter_labels(COLLECT_RUN_ID))
    print(f"{len(frames)} labeled frames", flush=True)

    if args.taus is None:
        # Place the sweep on the spread distribution: the conformal run's
        # spreads span ~1e-3..3e-3; cover the range that sweeps the engine
        # fraction across the target budget.
        taus = [0.0012, 0.0015, 0.0018, 0.0022, 0.0027, 0.0033, 0.0040]
    else:
        taus = args.taus

    sweep: list[dict] = []
    for tau in taus:
        records, summary = run_at_tau(frames, tau)
        row = {
            "tau": tau,
            "dft_fraction": summary.dft_fraction,
            "n_dft": summary.n_dft,
            "n_accepted": summary.n_accepted,
            "alpha_hat": summary.alpha_hat,
            "n_finetunes": summary.n_finetunes,
            "wall_time_s": summary.wall_time_s,
        }
        sweep.append(row)
        print(f"tau={tau:.4f}: dft_fraction={row['dft_fraction']:.3f} "
              f"alpha_hat={row['alpha_hat']:.4f} "
              f"({summary.wall_time_s:.0f} s)", flush=True)

    # The tuned point: engine fraction closest to the conformal budget.
    tuned = min(sweep, key=lambda r: abs(r["dft_fraction"] - TARGET_DFT_FRACTION))
    report = {
        "target_dft_fraction": TARGET_DFT_FRACTION,
        "eps_acc": EPS_ACC,
        "alpha": 0.05,
        "tuned_point": tuned,
        "sweep": sweep,
        "conformal_reference": {"dft_fraction": 0.252, "alpha_hat": 0.0},
        "scheduled_reference": {"dft_fraction": 0.252, "alpha_hat": 0.0929},
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\ntuned tau={tuned['tau']:.4f}: dft_fraction "
          f"{tuned['dft_fraction']:.3f} (target {TARGET_DFT_FRACTION}), "
          f"alpha_hat {tuned['alpha_hat']:.4f} vs conformal 0.000 / "
          f"scheduled 0.093")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
