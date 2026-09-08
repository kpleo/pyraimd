"""Alpha calibration curve on the water validation (referee §5.2):
target alpha vs measured miscoverage alpha_hat.

Self-consistent conformal replays at alpha in {0.01, 0.02, 0.05, 0.10, 0.20}
with everything else at the baseline (eps_acc=0.25, W=64, w_min=16,
delta=1e-3, N_label=8, fresh committee, same seed, identical replay code
path). For each point: engine fraction, n_accepted, alpha_hat, and the
one-sided 95% Clopper-Pearson upper bound on the true violation rate
(scipy.stats.beta; for k=0 this is the exact 1 - 0.05**(1/n)).

The alpha=0.05 point re-runs the baseline and must reproduce
analysis/h2o_streak/decision_log_conformal.json (226 accepted,
alpha_hat=0.000) — a determinism check, flagged in the report.

NOTE: on water the conformal bound is expected to over-cover (alpha_hat = 0
across the sweep, far below target). Report what you get; do not force a
diagonal.

Usage:
    uv run python experiments/alpha_curve_h2o.py \
        --store analysis/h2o_streak/coverage_h2o.db \
        --out analysis/h2o_streak/alpha_curve_report.json
"""

from __future__ import annotations

import argparse
import json
import math
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
WINDOW = 64
W_MIN = 16
DELTA = 1e-3
N_LABEL = 8
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.20]

BASELINE = {"alpha": 0.05, "dft_fraction": 0.25165562913907286, "n_dft": 76,
            "n_accepted": 226, "alpha_hat": 0.0, "n_finetunes": 9}


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
    """One-sided 95% Clopper-Pearson upper bound on the rate given k/n.

    beta.ppf(0.95, k+1, n-k); at k=0 this is the exact 1 - 0.05**(1/n).
    """
    if n < 1:
        return float("nan")
    if k == n:
        return 1.0
    return float(beta_dist.ppf(0.95, k + 1, n - k))


def run_at_alpha(frames, alpha: float):
    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ConformalSwitch(
        committee, alpha=alpha, eps_acc=EPS_ACC,
        window=WINDOW, w_min=W_MIN, delta=DELTA,
    )
    updater = OnlineUpdater(
        _ProgressCommittee(committee, f"alpha={alpha}"),
        observe=switch.observe, n_label=N_LABEL,
    )
    return replay(frames, committee, switch, updater=updater, eps_acc=EPS_ACC)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
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
    for alpha in args.alphas:
        records, summary = run_at_alpha(frames, alpha)
        n_violations = sum(record.violation for record in records)
        row = {
            "alpha": alpha,
            "dft_fraction": summary.dft_fraction,
            "n_dft": summary.n_dft,
            "n_accepted": summary.n_accepted,
            "n_violations": n_violations,
            "alpha_hat": summary.alpha_hat,
            "cp95_upper": cp95_one_sided_upper(n_violations, summary.n_accepted),
            "n_finetunes": summary.n_finetunes,
            "wall_time_s": summary.wall_time_s,
        }
        if alpha == BASELINE["alpha"] and args.max_frames is None:
            row["baseline_reproduced"] = bool(
                summary.n_accepted == BASELINE["n_accepted"]
                and summary.n_dft == BASELINE["n_dft"]
                and summary.alpha_hat == BASELINE["alpha_hat"]
                and summary.n_finetunes == BASELINE["n_finetunes"]
                and math.isclose(summary.dft_fraction, BASELINE["dft_fraction"],
                                 rel_tol=0, abs_tol=1e-12)
            )
            if not row["baseline_reproduced"]:
                print(f"  WARNING: alpha={alpha} does not reproduce the "
                      f"baseline decision log!", flush=True)
        rows.append(row)
        print(f"alpha={alpha:.2f}: dft_fraction={row['dft_fraction']:.3f} "
              f"n_accepted={row['n_accepted']} alpha_hat={row['alpha_hat']:.4f} "
              f"cp95_upper={row['cp95_upper']:.4f} "
              f"({summary.wall_time_s:.0f} s)", flush=True)

    report = {
        "task": "T2 alpha calibration curve (target alpha vs alpha_hat)",
        "params": {"eps_acc": EPS_ACC, "window": WINDOW, "w_min": W_MIN,
                   "delta": DELTA, "n_label": N_LABEL, "seed": SEED},
        "cp_method": ("one-sided 95% Clopper-Pearson upper via "
                      "scipy.stats.beta.ppf(0.95, k+1, n-k); k=0 exact "
                      "1-0.05**(1/n)"),
        "rows": rows,
        "baseline_conformal_reference": BASELINE,
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
