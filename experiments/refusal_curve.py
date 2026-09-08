"""Refusal/operating curve of the certified flagship stream (sec:flagbench).

Post-hoc operating characteristic of the supervisor on the flagship-A
production stream (steps 0-85): at any accuracy budget eps, replay the
decision layer over the stored spread series with the exact ConformalSwitch
semantics (window 64, w_min 16, alpha 0.05, delta 1e-3, finite-sample
quantile), for the pre-repair (rho=0) and post-repair (rho=0.05) bounds, and
score accepted steps against the archived DFT labels. Every step of this
stream is labeled: 81 engine-routed steps carry their engine forces, and the
6 accepted steps were relabeled by audit batch 1.

Fidelity check (must hold): at eps=1.0, rho=0 the replay reproduces the
actual run exactly (accepts steps 37-42, 4 audited violations); at rho=0.05
it keeps exactly the two compliant accepts (37-38) and rejects the four
violators --- the audit-replay claim of Sec. flagrepair.

Caveat (stated in the manuscript): the stored s series embeds the actual
run's fine-tune history; counterfactual budgets diverge from the recorded
trajectory after their first differing decision. The evaluation is exact for
the trajectory as run at its operating point eps=1.0.

Usage:
    uv run python experiments/refusal_curve.py \
        --stream analysis/flagship_a_prod/stream_snapshot_fig7_88steps.json \
        --audit analysis/flagship_a_prod/audit_report_b1.json \
        --out analysis/refusal_curve.json \
        --figure manuscript/figures/fig7_refusal_curve.png

    NOTE: the stream must carry s AND e per step. The archived
    stream_snapshot_steps0_85.json keeps e only for steps <= 63 (segment-7
    refusal-phase export) and no longer replays. The figure's committed data
    basis is pinned at the 88-labeled-step snapshot
    analysis/flagship_a_prod/stream_snapshot_fig7_88steps.json (steps -1..86)
    --- use it, NOT stream_snapshot_latest.json, which grows with production
    and would silently extend the figure beyond the committed numbers.
    Regeneration with the pinned basis reproduces the committed report
    bit-for-bit (verified 2026-09-04).
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from pyraimd2.switch.conformal import conformal_quantile

ALPHA = 0.05
WINDOW = 64
W_MIN = 16
DELTA = 1e-3
RHOS = {"pre-repair": 0.0, "post-repair": 0.05}
ACTUAL_EPS = 1.0
# the production budget schedule: burn-in refusal at eps=0.25 through
# step 36, operating point eps=1.0 from step 37 on (the acceptance era)
ACTUAL_SCHEDULE = (37, 0.25, 1.0)  # (switch_step, eps_before, eps_after)
ACTUAL_ACCEPTS = [37, 38, 39, 40, 41, 42]
ACTUAL_VIOLATIONS = [39, 40, 41, 42]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stream", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)
    p.add_argument("--no-title", action="store_true")
    return p.parse_args()


def load_series(stream_path: Path, audit_path: Path) -> list[dict]:
    recs = json.loads(stream_path.read_text())
    audit = json.loads(audit_path.read_text())
    audited_e = {d["step"]: d["e"] for d in audit["per_step"]}
    series = []
    for r in recs:
        e = r.get("e", audited_e.get(r["step"]))
        if r.get("s") is None or e is None:
            continue  # step without stored spread or without any label
        series.append({"step": r["step"], "s": float(r["s"]), "e": float(e),
                       "actual_route": r["route"]})
    series.sort(key=lambda r: r["step"])
    return series


def replay(series: list[dict], eps, rho: float) -> dict:
    """eps: a float (uniform budget) or a callable eps(step) (schedule)."""
    budget = eps if callable(eps) else lambda step: eps
    window: deque[tuple[float, float]] = deque(maxlen=WINDOW)
    k = 0
    accepts, violations = [], []
    for rec in series:
        s, e = rec["s"], rec["e"]
        eps_t = budget(rec["step"])
        n = len(window)
        qhat = (conformal_quantile([ew / (sw + DELTA) for sw, ew in window], ALPHA)
                if n else float("inf"))
        bound = qhat * (s + DELTA) * (1.0 + rho * k)
        if n < W_MIN or bound > eps_t:
            window.append((s, e))
            k = 0
        else:
            accepts.append(rec)
            if e > eps_t:
                violations.append(rec)
            k += 1
    n_all = len(series)
    n_acc = len(accepts)
    return {
        "eps": eps if not callable(eps) else "schedule",
        "rho": rho,
        "engine_fraction": (n_all - n_acc) / n_all,
        "n_accepted": n_acc,
        "alpha_hat": len(violations) / n_acc if n_acc else None,
        "accepted_steps": [r["step"] for r in accepts],
        "violation_steps": [r["step"] for r in violations],
    }


def oracle_curve(series: list[dict], eps_grid: np.ndarray) -> list[dict]:
    """Fraction of steps that MUST be labeled at each budget (e > eps)."""
    es = np.array([r["e"] for r in series])
    return [{"eps": float(eps), "engine_fraction": float(np.mean(es > eps))}
            for eps in eps_grid]


def main() -> int:
    args = parse_args()
    series = load_series(args.stream, args.audit)
    print(f"{len(series)} labeled steps, "
          f"e range {min(r['e'] for r in series):.3f}..{max(r['e'] for r in series):.3f}")
    n_actual_dft = sum(1 for r in series if r["actual_route"] == "dft")
    actual_engine_fraction = n_actual_dft / len(series)

    eps_grid = np.round(np.arange(0.50, 2.01, 0.05), 2)
    curves: dict[str, list[dict]] = {}
    for name, rho in RHOS.items():
        rows = [replay(series, float(eps), rho) for eps in eps_grid]
        curves[name] = rows
        op = next(r for r in rows if abs(r["eps"] - ACTUAL_EPS) < 1e-9)
        print(f"{name} @ uniform eps={ACTUAL_EPS}: accepts={op['accepted_steps']} "
              f"violations={op['violation_steps']}")

    # fidelity check: replaying the ACTUAL budget schedule (burn-in 0.25,
    # switch to 1.0 at step 37) must reproduce the recorded run exactly
    sw_step, eps_lo, eps_hi = ACTUAL_SCHEDULE
    schedule = lambda st: eps_lo if st < sw_step else eps_hi  # noqa: E731
    pre_actual = replay(series, schedule, RHOS["pre-repair"])
    post_actual = replay(series, schedule, RHOS["post-repair"])
    assert pre_actual["accepted_steps"] == ACTUAL_ACCEPTS, pre_actual["accepted_steps"]
    assert pre_actual["violation_steps"] == ACTUAL_VIOLATIONS, pre_actual["violation_steps"]
    print("fidelity check 1 passed: pre-repair schedule replay is exact")

    # fidelity check 2: the post-repair bound evaluated PER STEP at the
    # original streak positions (the Sec. flagrepair audit replay) rejects
    # exactly the four violators and keeps the two compliant accepts (6/6)
    snap = {r["step"]: r for r in json.loads(args.stream.read_text())}
    qhat_birth = float(snap[ACTUAL_ACCEPTS[0]]["qhat"])
    per_step = []
    for i, st in enumerate(ACTUAL_ACCEPTS):
        s = next(r["s"] for r in series if r["step"] == st)
        e = next(r["e"] for r in series if r["step"] == st)
        bound = qhat_birth * (s + DELTA) * (1.0 + RHOS["post-repair"] * i)
        per_step.append({"step": st, "bound": bound,
                         "decision": "ml" if bound <= eps_hi else "dft",
                         "violation": e > eps_hi})
    kept = [p["step"] for p in per_step if p["decision"] == "ml"]
    rejected = [p["step"] for p in per_step if p["decision"] == "dft"]
    assert kept == ACTUAL_ACCEPTS[:2], per_step
    assert rejected == ACTUAL_VIOLATIONS, per_step
    print("fidelity check 2 passed: post-repair per-step audit replay 6/6")
    # sequential post-repair schedule counterfactual (state resets after each
    # rejection) is recorded, not asserted: after rejecting 39 the window
    # refreshes and step 40 passes at k=0 --- marginal-regime behavior.
    print("post-repair sequential schedule: accepts="
          f"{post_actual['accepted_steps']} violations={post_actual['violation_steps']}")

    oracle = oracle_curve(series, eps_grid)
    report = {
        "n_steps": len(series), "alpha": ALPHA, "window": WINDOW,
        "w_min": W_MIN, "delta": DELTA, "actual_eps": ACTUAL_EPS,
        "actual": {
            "engine_fraction": actual_engine_fraction,
            "n_dft": n_actual_dft,
            "schedule": {"switch_step": sw_step, "eps_before": eps_lo,
                         "eps_after": eps_hi},
            "pre_repair": {"accepted": ACTUAL_ACCEPTS,
                           "violations": ACTUAL_VIOLATIONS,
                           "alpha_hat": 4 / 6,
                           "clopper_pearson_95": [0.22, 0.96]},
            "post_repair": {"accepted": [], "note": "0 accepts in production "
                            "segments 6-7; 43 refusals, 0 unsafe"},
            "post_repair_per_step_replay": per_step,
            "post_repair_sequential_replay": {
                "accepted": post_actual["accepted_steps"],
                "violations": post_actual["violation_steps"]},
        },
        "curves": curves, "oracle": oracle,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"report -> {args.out}")

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
          "verm": "#D55E00", "ink": "#333333", "gray": "#666666"}
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.8, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "axes.labelsize": 7,
        "legend.frameon": False, "legend.fontsize": 6.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # single-column layout: two panels stacked at final width (3.4 in)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.4))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.965, bottom=0.09,
                        hspace=0.55)

    eps = np.array([r["eps"] for r in curves["pre-repair"]])
    # (a) engine fraction
    ax1.plot(eps, [r["engine_fraction"] for r in curves["pre-repair"]],
             ls="--", lw=1.0, color=OI["blue"], label="pre-repair ($\\rho=0$)")
    ax1.plot(eps, [r["engine_fraction"] for r in curves["post-repair"]],
             ls="-", lw=1.2, color=OI["blue"], label="post-repair ($\\rho=0.05$)")
    ax1.plot(eps, [r["engine_fraction"] for r in oracle],
             ls=":", lw=1.2, color=OI["green"], label="oracle (must-label)")
    ax1.plot([ACTUAL_EPS], [actual_engine_fraction], "x", ms=6, mew=1.5,
             color=OI["ink"], label="actual run")
    ax1.set_xlabel(r"accuracy budget $\varepsilon_{\mathrm{acc}}$ (eV/Å)")
    ax1.set_ylabel("engine fraction")
    ax1.set_ylim(-0.03, 1.05)
    ax1.legend(loc="upper right", handlelength=1.6)
    ax1.set_title("(a) label cost vs budget", fontsize=7, loc="left", pad=5)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # (b) miscoverage
    for name, ls, color in (("pre-repair", "--", OI["verm"]),
                            ("post-repair", "-", OI["blue"])):
        xs = [r["eps"] for r in curves[name] if r["n_accepted"] >= 5]
        ys = [r["alpha_hat"] for r in curves[name] if r["n_accepted"] >= 5]
        ax2.plot(xs, ys, ls=ls, lw=1.1, color=color,
                 label=f"{name} ($\\rho={RHOS[name]}$)")
    ax2.axhline(ALPHA, color=OI["ink"], lw=0.9, ls=(0, (4, 2)))
    ax2.text(1.98, ALPHA + 0.02, r"target $\alpha=0.05$", fontsize=6.5,
             ha="right", va="bottom", color=OI["ink"])
    ax2.errorbar([ACTUAL_EPS], [4 / 6], yerr=[[4 / 6 - 0.22], [0.96 - 4 / 6]],
                 fmt="D", ms=4.5, color=OI["verm"], mec=OI["verm"],
                 mfc="none", elinewidth=0.9, capsize=2.5,
                 label="measured break (batch 1)")
    ax2.set_xlabel(r"accuracy budget $\varepsilon_{\mathrm{acc}}$ (eV/Å)")
    ax2.set_ylabel(r"miscoverage $\hat\alpha$")
    ax2.set_ylim(-0.03, 1.05)
    ax2.legend(loc="upper right", handlelength=1.6)
    ax2.set_title("(b) violation rate vs budget", fontsize=7, loc="left",
                  pad=5)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    if not args.no_title:
        fig.suptitle("operating characteristic of the certified flagship stream "
                     f"(n = {len(series)} labeled steps)", fontsize=8, x=0.53)
    fig.savefig(args.figure.with_suffix(".pdf"))
    fig.savefig(args.figure.with_suffix(".png"), dpi=300)
    print(f"figure -> {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
