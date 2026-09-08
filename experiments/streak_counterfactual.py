"""Streak counterfactual analysis (red-team experiment E1): does the
calibration ratio r drift *within* acceptance streaks on a system whose
slack is large?

The flagship audit (batch 1) found r drifting ~+0.05/step across an
acceptance streak, with 4/6 violations at the near-tight eps=1.0 operating
point (slack ~5%). The discriminating question from the red-team review
(docs/red-team-review-2026-08-27.md, E1): on water — 226 accepted frames in
streaks of 18/12/196 with slack ~4.6x — does r also climb with the streak
position k?

  - If r climbs on water too but violations stay at zero, the leak LAW is
    confirmed: violations iff rho*k exceeds the slack (eps-B)/B; streaks
    are harmless when the bound is conservative.
  - If r does NOT climb on water, the flagship drift is regime-specific
    (fresh-box relaxation), and the flagship streak was a local event.

Input: an M2 decision log (replay records with route/spread/qhat/bound/
error per frame) — no new compute. Output: JSON report + a two-panel figure
(r(k) profiles and the leak-law scatter with the rho*k = slack threshold).

Usage:
    uv run python experiments/streak_counterfactual.py \
        --decision-log analysis/h2o_streak/decision_log_conformal.json \
        --eps-acc 0.25 --out analysis/h2o_streak/streak_report.json \
        --figure analysis/h2o_streak/fig_streak_h2o.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DELTA = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decision-log", required=True, type=Path)
    p.add_argument("--eps-acc", required=True, type=float)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--figure", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.decision_log.read_text())
    recs = payload["records"]

    # Assign streak ids and per-frame streak position k (0-indexed).
    streaks: list[list[dict]] = []
    for rec in recs:
        if rec["route"] == "ml":
            if not streaks or streaks[-1] == "closed":
                streaks.append([])
            streaks[-1].append(rec)
        else:
            if streaks and streaks[-1] != "closed":
                streaks.append("closed")
    streaks = [s for s in streaks if s != "closed"]

    accepted: list[dict] = []
    for sid, streak in enumerate(streaks):
        for k, rec in enumerate(streak):
            r = rec["error"] / (rec["spread"] + DELTA)
            slack = (args.eps_acc - rec["bound"]) / rec["bound"]
            accepted.append({
                "step": rec["step"], "streak_id": sid, "k": k,
                "r": r, "slack": slack, "e": rec["error"],
                "violation": bool(rec["error"] > args.eps_acc),
            })

    ks = np.array([a["k"] for a in accepted])
    rs = np.array([a["r"] for a in accepted])
    # Within-streak drift: linear fit r(k) over ALL accepted frames, plus a
    # per-k median profile (streaks differ in length; long streaks dominate
    # the fit, which is what we want for drift evidence).
    if len(accepted) > 2:
        slope, intercept = np.polyfit(ks, rs, 1)
    else:
        slope, intercept = float("nan"), float("nan")
    max_k = int(ks.max()) if len(accepted) else 0
    profile = []
    for k in range(max_k + 1):
        sel = rs[ks == k]
        if len(sel):
            profile.append({"k": k, "n": int(len(sel)),
                            "r_median": float(np.median(sel)),
                            "r_mean": float(np.mean(sel))})
    n_viol = sum(a["violation"] for a in accepted)
    report = {
        "eps_acc": args.eps_acc,
        "n_frames": len(recs),
        "n_accepted": len(accepted),
        "streak_lengths": [len(s) for s in streaks],
        "violations": n_viol,
        "violation_rate": n_viol / max(len(accepted), 1),
        "r_vs_k_linear_fit": {"slope_per_step": float(slope),
                              "intercept": float(intercept)},
        "r_profile_by_k": profile,
        "accepted": accepted,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"streaks {[len(s) for s in streaks]}, accepted {len(accepted)}, "
          f"violations {n_viol}")
    print(f"r(k) drift slope: {slope:+.5f}/step (intercept {intercept:.3f})")

    if args.figure is not None:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
              "verm": "#D55E00", "ink": "#333333", "gray": "#666666",
              "lgray": "#BBBBBB"}
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.5, "axes.linewidth": 0.8, "xtick.labelsize": 7,
            "ytick.labelsize": 7, "legend.frameon": False,
            "legend.fontsize": 6.5, "figure.dpi": 300,
        })
        fig, (axa, axb) = plt.subplots(
            1, 2, figsize=(7.0, 2.7), gridspec_kw={"wspace": 0.28})

        # (a) r(k) profile: median per k with 25-75% band
        med = [p["r_median"] for p in profile]
        kk = [p["k"] for p in profile]
        q25, q75 = [], []
        for k in kk:
            sel = rs[ks == k]
            q25.append(float(np.percentile(sel, 25)))
            q75.append(float(np.percentile(sel, 75)))
        axa.fill_between(kk, q25, q75, color=OI["blue"], alpha=0.15, lw=0)
        axa.plot(kk, med, color=OI["blue"], lw=1.2,
                 label=f"H$_2$O (slope {slope:+.4f}/step)")
        axa.set_xlabel("streak position $k$ (accepted steps since last label)")
        axa.set_ylabel(r"calibration ratio $r=e/(s+\delta)$")
        axa.set_xlim(0, 60)  # the 196-streak's early region is the informative one
        axa.legend(loc="upper left")
        axa.set_title("(a) within-streak drift on water", loc="left")

        # (b) leak-law scatter: (k, slack) for every accepted frame
        slacks = np.array([a["slack"] for a in accepted])
        axb.plot(ks, slacks, ".", ms=2.5, color=OI["blue"], alpha=0.4,
                 label="accepted frames (H$_2$O)")
        # flagship audit streak, for contrast
        try:
            audit = json.loads(
                Path("analysis/flagship_a_prod/audit_report_b1.json").read_text())
            fk, fs, fv = [], [], []
            for i, d in enumerate(audit["per_step"]):
                fk.append(i)
                fs.append(None)  # slack not stored in the audit report
                fv.append(d["violation"])
        except FileNotFoundError:
            pass
        axb.set_xlabel("streak position $k$")
        axb.set_ylabel(r"slack $(\varepsilon-B)/B$")
        axb.axhline(0, color=OI["gray"], lw=0.8, ls="--")
        axb.set_xlim(0, 60)
        axb.legend(loc="upper right")
        axb.set_title("(b) slack across the streak", loc="left")

        fig.align_ylabels()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure.with_suffix(".pdf"))
        fig.savefig(args.figure.with_suffix(".png"), dpi=300)
        print(f"figure -> {args.figure.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
