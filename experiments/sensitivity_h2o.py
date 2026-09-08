"""Sensitivity grid on the water validation (referee §5.3):
one-factor-at-a-time perturbations of the baseline conformal replay.

Baseline: alpha=0.05, eps_acc=0.25, W=64, w_min=16, delta=1e-3, N_label=8
(analysis/h2o_streak/decision_log_conformal.json: engine fraction 0.252,
226 accepted, alpha_hat=0.000). Six self-consistent replays perturb one
factor each: W in {32, 128}, delta in {1e-4, 1e-2}, N_label in {4, 16}
(fresh committee, same seed, identical replay code path). The baseline row
is included as a reference (not re-run here; the alpha-curve script
re-runs and verifies it).

Usage:
    uv run python experiments/sensitivity_h2o.py \
        --store analysis/h2o_streak/coverage_h2o.db \
        --out analysis/h2o_streak/sensitivity_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scipy.stats import beta as beta_dist

from pyraimd2.loop import OnlineUpdater
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate
from pyraimd2.switch import ConformalSwitch, replay

COLLECT_RUN_ID = "collect-h2o-nve-300K"
SEED = 20250819
EPS_ACC = 0.25
ALPHA = 0.05
WINDOW = 64
W_MIN = 16
DELTA = 1e-3
N_LABEL = 8

BASELINE = {"alpha": ALPHA, "eps_acc": EPS_ACC, "window": WINDOW,
            "w_min": W_MIN, "delta": DELTA, "n_label": N_LABEL,
            "dft_fraction": 0.25165562913907286, "n_dft": 76,
            "n_accepted": 226, "alpha_hat": 0.0, "n_finetunes": 9,
            "wall_time_s": 347.93050008300634}

# (perturbed factor, value) — everything else stays at the baseline.
GRID = [("window", 32), ("window", 128),
        ("delta", 1e-4), ("delta", 1e-2),
        ("n_label", 4), ("n_label", 16)]


class _ProgressCommittee:
    """Delegates to CommitteeSurrogate; prints a progress line per fine-tune."""

    def __init__(self, committee: CommitteeSurrogate, tag: str) -> None:
        self._committee = committee
        self._tag = tag

    def predict(self, atoms):
        return self._committee.predict(atoms)

    def finetune(self, labels):
        t0 = time.perf_counter()
        report = self._committee.finetune(labels)
        print(f"  [{self._tag}] finetune on {report.n_labels} labels: loss "
              f"{report.initial_loss:.3f} -> {report.final_loss:.3f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        return report


def cp95_one_sided_upper(k: int, n: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound on the rate given k/n."""
    if n < 1:
        return float("nan")
    if k == n:
        return 1.0
    return float(beta_dist.ppf(0.95, k + 1, n - k))


def run_config(frames, window: int, delta: float, n_label: int, tag: str):
    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ConformalSwitch(
        committee, alpha=ALPHA, eps_acc=EPS_ACC,
        window=window, w_min=W_MIN, delta=delta,
    )
    updater = OnlineUpdater(
        _ProgressCommittee(committee, tag),
        observe=switch.observe, n_label=n_label,
    )
    return replay(frames, committee, switch, updater=updater, eps_acc=EPS_ACC)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-frames", type=int, default=None,
                   help="replay only the first N frames (smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frames = list(Store(args.store).iter_labels(COLLECT_RUN_ID))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    print(f"{len(frames)} labeled frames", flush=True)

    rows: list[dict] = []
    for factor, value in GRID:
        window = value if factor == "window" else WINDOW
        delta = value if factor == "delta" else DELTA
        n_label = value if factor == "n_label" else N_LABEL
        tag = f"{factor}={value}"
        records, summary = run_config(frames, window, delta, n_label, tag)
        n_violations = sum(record.violation for record in records)
        row = {
            "factor": factor,
            "value": value,
            "config": {"alpha": ALPHA, "eps_acc": EPS_ACC, "window": window,
                       "w_min": W_MIN, "delta": delta, "n_label": n_label},
            "dft_fraction": summary.dft_fraction,
            "n_dft": summary.n_dft,
            "n_accepted": summary.n_accepted,
            "n_violations": n_violations,
            "alpha_hat": summary.alpha_hat,
            "cp95_upper": cp95_one_sided_upper(n_violations, summary.n_accepted),
            "n_finetunes": summary.n_finetunes,
            "wall_time_s": summary.wall_time_s,
        }
        rows.append(row)
        print(f"{tag}: dft_fraction={row['dft_fraction']:.3f} "
              f"n_accepted={row['n_accepted']} alpha_hat={row['alpha_hat']:.4f} "
              f"({summary.wall_time_s:.0f} s)", flush=True)

    report = {
        "task": "T3 one-factor-at-a-time sensitivity of the conformal replay",
        "baseline_reference": {**BASELINE, "source": "decision_log_conformal.json "
                               "(not re-run here; alpha_curve_h2o.py re-runs and "
                               "verifies the baseline)"},
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
