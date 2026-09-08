"""Canonical current-code water baseline suite (post code-drift finding).

The archived analysis/h2o_streak/decision_log_conformal.json (76 dft / 226
accepted) was produced by the ORIGINAL M2 committee (e61c93a, 2026-08-19:
identical members at init, first-frame spreads exactly 0.0, qhat pinned at
the cold-start max for a full window). It predates the seeded load-time
readout perturbation (1bd1fc4, 2026-08-21) and the population-RMS spread
estimator (febb372, 2026-08-21). This script re-runs the complete baseline
suite with the current committee so every manuscript number is current-code
and internally consistent:

1. Baseline conformal replay (eps=0.25, alpha=0.05, W=64, w_min=16,
   delta=1e-3, N_label=8, seed 20250819) -> full per-frame decision log in
   the archived schema (decision_log_conformal_current.json). Deterministic:
   three prior reruns of this config gave bit-identical summaries
   (75 dft / 227 accepted, alpha_hat=0).
2. Scheduled ablation at the baseline's measured DFT fraction (period =
   round(1/fraction) = 4), mirroring coverage_h2o.py's run_scheduled.
3. Oracle recomputation from the current-code baseline log (fraction of
   frames with realized error > eps), with the counterfactual-detection-
   bound caveat.
4. Tightness statistics on accepted frames (median e, B, B/e, s, e/s,
   qhat median/range, r=e/(s+delta) median), plus the legacy late-cut
   (step>=50, all frames) tightness from coverage_h2o.py for continuity.
5. Streak inventory: maximal acceptance streaks (start, length), per-streak
   linear drift of r vs streak position k, pooled r(k) fit — the water
   input to the leak phase diagram (leak_phase_diagram.py --h2o-log accepts
   the new decision log directly).

Usage:
    uv run python experiments/baseline_current_h2o.py \
        --store analysis/h2o_streak/coverage_h2o.db \
        --outdir analysis/h2o_streak
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from pyraimd2.loop import OnlineUpdater
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate
from pyraimd2.switch import ConformalSwitch, ScheduledSwitch, replay

COLLECT_RUN_ID = "collect-h2o-nve-300K"
SEED = 20250819
EPS_ACC = 0.25
ALPHA = 0.05
WINDOW = 64
W_MIN = 16
DELTA = 1e-3
N_LABEL = 8

CODE_PROVENANCE = (
    "current committee: seeded load-time readout perturbation (1bd1fc4, "
    "2026-08-21) + population-RMS spread estimator (febb372, 2026-08-21); "
    "supersedes archived decision_log_conformal.json (original M2 committee "
    "e61c93a: identical members at init, first-frame spreads exactly 0.0). "
    "This config reproduced bit-identically across 3 independent replays on "
    "2026-08-29 (dft_fraction=0.24834437086092714, 75/227, alpha_hat=0)."
)


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
    """beta.ppf(0.95, k+1, n-k); at k=0 the exact 1 - 0.05**(1/n)."""
    if n < 1:
        return float("nan")
    if k == n:
        return 1.0
    return float(beta_dist.ppf(0.95, k + 1, n - k))


def cp95_two_sided(k: int, n: int) -> list[float]:
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def save_log(path: Path, records, summary) -> None:
    """Same schema as the archived decision_log_conformal.json."""
    payload = {
        "summary": asdict(summary),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=True))


def run_conformal(frames):
    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ConformalSwitch(
        committee, alpha=ALPHA, eps_acc=EPS_ACC,
        window=WINDOW, w_min=W_MIN, delta=DELTA,
    )
    updater = OnlineUpdater(
        _ProgressCommittee(committee, "conformal-current"),
        observe=switch.observe, n_label=N_LABEL,
    )
    return replay(frames, committee, switch, updater=updater, eps_acc=EPS_ACC)


def run_scheduled(frames, dft_fraction: float):
    period = max(1, round(1.0 / dft_fraction))
    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ScheduledSwitch(period)
    updater = OnlineUpdater(
        _ProgressCommittee(committee, "scheduled-current"),
        observe=lambda s, e: None, n_label=N_LABEL,
    )
    records, summary = replay(frames, committee, switch, updater=updater,
                              eps_acc=EPS_ACC)
    return records, summary, period


def streak_analysis(records) -> dict:
    """Maximal acceptance streaks + within-streak drift of r = e/(s+delta)."""
    streaks: list[list[dict]] = []
    for rec in records:
        if rec.route == "ml":
            if not streaks or streaks[-1] is None:
                streaks.append([])
            streaks[-1].append(rec)
        elif streaks and streaks[-1] is not None:
            streaks.append(None)
    streaks = [s for s in streaks if s is not None]

    out_streaks: list[dict] = []
    all_k: list[int] = []
    all_r: list[float] = []
    for sid, streak in enumerate(streaks):
        rs = [rec.error / (rec.spread + DELTA) for rec in streak]
        ks = list(range(len(streak)))
        all_k.extend(ks)
        all_r.extend(rs)
        if len(streak) >= 3:
            slope, intercept = np.polyfit(ks, rs, 1)
            slope, intercept = float(slope), float(intercept)
        else:
            slope = intercept = None
        out_streaks.append({
            "streak_id": sid,
            "start_step": streak[0].step,
            "end_step": streak[-1].step,
            "length": len(streak),
            "r_first": float(rs[0]),
            "r_last": float(rs[-1]),
            "r_median": float(np.median(rs)),
            "r_vs_k_fit": ({"slope_per_step": slope, "intercept": intercept}
                           if slope is not None else None),
            "e_max": float(max(rec.error for rec in streak)),
            "n_violations": int(sum(rec.violation for rec in streak)),
        })
    if len(all_k) > 2:
        pslope, pintercept = np.polyfit(np.array(all_k), np.array(all_r), 1)
        pooled = {"slope_per_step": float(pslope), "intercept": float(pintercept)}
    else:
        pooled = {"slope_per_step": float("nan"), "intercept": float("nan")}
    return {
        "delta": DELTA,
        "n_streaks": len(out_streaks),
        "streak_lengths": [s["length"] for s in out_streaks],
        "streaks": out_streaks,
        "pooled_r_vs_k_fit": pooled,
        "archived_old_code_reference": {
            "streak_lengths": [18, 12, 196],
            "pooled_slope_per_step": -0.0019185202066644518,
            "source": "analysis/h2o_streak/streak_report.json (old committee)",
        },
    }


def tightness_stats(records) -> dict:
    """Bound/spread calibration statistics on accepted frames (delta=1e-3)."""
    accepted = [rec for rec in records if rec.route == "ml"]
    if not accepted:
        return {
            "population": f"accepted (route=ml) frames, n=0 of {len(records)}",
            "n_accepted": 0,
        }
    e = np.array([rec.error for rec in accepted])
    b = np.array([rec.bound for rec in accepted])
    s = np.array([rec.spread for rec in accepted])
    q = np.array([rec.qhat for rec in accepted])
    r = e / (s + DELTA)
    finite_q = q[np.isfinite(q)]
    # Legacy continuity metric: coverage_h2o.py's late-cut tightness
    # (median B/e over ALL frames with step >= 50, not just accepted).
    late = np.arange(len(records)) >= 50
    errors_all = np.array([rec.error for rec in records])
    bounds_all = np.array([rec.bound for rec in records])
    legacy = float(np.median(bounds_all[late] / np.maximum(errors_all[late], 1e-12)))
    return {
        "population": (f"accepted (route=ml) frames of the current-code "
                       f"baseline replay, n={len(accepted)} of "
                       f"{len(records)}; eps_acc={EPS_ACC}, delta={DELTA}"),
        "n_accepted": len(accepted),
        "e_median": float(np.median(e)),
        "e_min": float(e.min()),
        "e_max": float(e.max()),
        "bound_median": float(np.median(b)),
        "bound_min": float(b.min()),
        "bound_max": float(b.max()),
        "tightness_B_over_e_median": float(np.median(b / np.maximum(e, 1e-12))),
        "spread_median": float(np.median(s)),
        "spread_min": float(s.min()),
        "spread_max": float(s.max()),
        "e_over_s_median": float(np.median(e / s)),
        "r_e_over_s_plus_delta_median": float(np.median(r)),
        "r_min": float(r.min()),
        "r_max": float(r.max()),
        "qhat_median": float(np.median(finite_q)),
        "qhat_min": float(finite_q.min()),
        "qhat_max": float(finite_q.max()),
        "qhat_population": (f"decision-time qhat on accepted frames; finite "
                            f"on {len(finite_q)}/{len(accepted)} (accepted "
                            f"frames always have |W| >= w_min = {W_MIN})"),
        "legacy_late_cut": {
            "definition": ("coverage_h2o.py continuity metric: median B/e "
                           "over ALL frames (any route) with step >= 50"),
            "tightness_median_B_over_e": legacy,
            "archived_old_code_value": None,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--outdir", type=Path,
                   default=Path("analysis/h2o_streak"))
    p.add_argument("--max-frames", type=int, default=None,
                   help="replay only the first N frames (smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frames = list(Store(args.store).iter_labels(COLLECT_RUN_ID))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    print(f"{len(frames)} labeled frames", flush=True)

    print("[1/2] baseline conformal replay (current committee) ...", flush=True)
    records_c, summary_c = run_conformal(frames)
    print(f"      dft_fraction={summary_c.dft_fraction:.4f} "
          f"n_accepted={summary_c.n_accepted} "
          f"alpha_hat={summary_c.alpha_hat:.4f} "
          f"({summary_c.wall_time_s:.0f} s)", flush=True)

    print("[2/2] scheduled ablation at the measured DFT fraction ...",
          flush=True)
    records_s, summary_s, period = run_scheduled(frames, summary_c.dft_fraction)
    n_viol_s = sum(rec.violation for rec in records_s)
    print(f"      period={period} dft_fraction={summary_s.dft_fraction:.4f} "
          f"n_accepted={summary_s.n_accepted} "
          f"alpha_hat={summary_s.alpha_hat:.4f} "
          f"({summary_s.wall_time_s:.0f} s)", flush=True)

    log_c = args.outdir / "decision_log_conformal_current.json"
    log_s = args.outdir / "decision_log_scheduled_current.json"
    save_log(log_c, records_c, summary_c)
    save_log(log_s, records_s, summary_s)

    # Oracle on the realized fine-tune history of the baseline replay.
    n_over = sum(1 for rec in records_c if rec.error > EPS_ACC)
    n_oracle_accept = len(records_c) - n_over
    oracle = {
        "eps_acc": EPS_ACC,
        "n_frames": len(records_c),
        "n_frames_over_eps": n_over,
        "must_label_fraction": n_over / len(records_c),
        "oracle_accepted": n_oracle_accept,
        "oracle_accept_fraction": n_oracle_accept / len(records_c),
        "oracle_violations_on_accepted": 0,
        "cp95_upper": cp95_one_sided_upper(0, n_oracle_accept),
        "caveat": (
            "Detection-bound oracle conditioned on the REALIZED fine-tune "
            "history of this replay: e_t is the error of the committee as "
            "fine-tuned on exactly the labels the conformal switch revealed. "
            "A different routing history produces a different committee and "
            "different e_t, so this oracle is not a counterfactual minimum "
            "label budget over all histories — it bounds what a perfect "
            "per-frame error detector would have had to label along THIS "
            "trajectory of the surrogate."),
        "archived_old_code_reference": {"n_frames_over_eps": 8,
                                        "oracle_accepted": 294},
    }

    report = {
        "task": "canonical current-code water baseline suite",
        "code_provenance": CODE_PROVENANCE,
        "params": {"alpha": ALPHA, "eps_acc": EPS_ACC, "window": WINDOW,
                   "w_min": W_MIN, "delta": DELTA, "n_label": N_LABEL,
                   "seed": SEED, "collect_run_id": COLLECT_RUN_ID},
        "baseline": {
            "summary": asdict(summary_c),
            "n_violations": int(sum(rec.violation for rec in records_c)),
            "cp95_upper": cp95_one_sided_upper(
                int(sum(rec.violation for rec in records_c)),
                summary_c.n_accepted),
            "decision_log": str(log_c),
            "archived_old_code_reference": {
                "dft_fraction": 0.25165562913907286, "n_dft": 76,
                "n_accepted": 226, "alpha_hat": 0.0, "n_finetunes": 9},
        },
        "scheduled": {
            "period": period,
            "summary": asdict(summary_s),
            "n_violations": int(n_viol_s),
            "alpha_hat_cp95_two_sided": cp95_two_sided(int(n_viol_s),
                                                       summary_s.n_accepted),
            "decision_log": str(log_s),
            "archived_old_code_reference": {
                "period": 4, "n_accepted": 226, "n_violations": 21,
                "alpha_hat": 0.0929,
                "source": ("docs/design-m2.md:108 / manuscript/main.tex:447; "
                           "original decision log was session-temporary")},
        },
        "oracle": oracle,
        "tightness": tightness_stats(records_c),
        "streaks": streak_analysis(records_c),
    }
    out = args.outdir / "baseline_current_report.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"wrote {log_c}\nwrote {log_s}\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
