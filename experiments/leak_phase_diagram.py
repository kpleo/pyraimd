"""Leak phase diagram (red-team experiment E4): every audited accepted step
from all three systems on one (k, slack) plane, with the drift-rate
threshold lines.

The leak law (red-team review E1-E3, docs/red-team-review-2026-08-27.md):
a step at streak position k violates the budget iff the within-streak drift
of the calibration ratio r carries it across the frozen-qhat-scaled margin:
    rho_r * k  >=~  qhat * slack,   slack = (eps - B) / B  at acceptance.
Equivalently in the (k, slack) plane, each system has a threshold line
slack = (rho_r / qhat) k with its own measured drift rate rho_r; accepted
steps sitting below-left of their system's line are predicted safe, and the
audited violations should sit exactly at/above it.

Three systems, all data already in hand:
  - H2O replay (current-code water decision log, RMS spread estimator:
    227 accepted, slack from stored bounds, weak within-streak drift —
    zero violations observed);
  - smoke box v4/v4.5 (11 accepted, eps=0.25, qhat parsed from the stored
    decision reason strings; violations from audit_report_smoke.json);
  - flagship A segment 4 (6 accepted, eps=1.0, qhat from loop.db reasons;
    violations from audit_report_b1.json).

Output: leak_phase_diagram.json + figures (pdf/png).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DELTA = 1e-3
QHAT_RE = re.compile(r"qhat=([0-9.inf]+)")

# Measured per-system threshold slopes rho_r/qhat (recomputed from the raw
# artifacts; method in experiments/leak_loo.py):
#   flagship474: OLS of r=e/(s+1e-3) vs k over the violating streak 37-42
#     (k=0..5, audit_report_b1.json): +0.0683/step at qhat 3.076 -> 0.0222;
#   H2O: strongest early drifting streak of the CURRENT-code log
#     (decision_log_conformal_current.json, streak 73-83, len 11): OLS of
#     r=error/(spread+1e-3) vs k = +0.161/step at frozen qhat 38.33 -> 0.0042;
#   smoke106: same recipe as flagship on streak C (steps 60-64, k=0..4,
#     audit_report_smoke.json): +0.412/step at qhat 1.760 -> 0.2344.
SLOPES = {"H2O": 0.00420, "smoke106": 0.23437, "flagship474": 0.02219}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h2o-log", type=Path,
                   default=Path("analysis/h2o_streak/decision_log_conformal_current.json"))
    p.add_argument("--smoke-db", type=Path,
                   default=Path("analysis/smoke_7615821/loop.db"))
    p.add_argument("--smoke-audit", type=Path,
                   default=Path("analysis/smoke_7615821/audit_report_smoke.json"))
    p.add_argument("--smoke-run-id", default="smoke-7615821")
    p.add_argument("--flagship-db", type=Path,
                   default=Path("analysis/flagship_a_prod/loop.db"))
    p.add_argument("--flagship-audit", type=Path,
                   default=Path("analysis/flagship_a_prod/audit_report_b1.json"))
    p.add_argument("--out", type=Path, default=Path("analysis/leak_phase_diagram.json"))
    p.add_argument("--figure", type=Path,
                   default=Path("analysis/fig_leak_phase_diagram.png"))
    p.add_argument("--no-title", action="store_true",
                   help="omit the in-axes title (manuscript version)")
    return p.parse_args()


def streak_positions(steps_sorted: list[int]) -> dict[int, int]:
    """k = 0-indexed position within each consecutive-step run."""
    out: dict[int, int] = {}
    k = 0
    prev = None
    for st in steps_sorted:
        k = k + 1 if prev is not None and st == prev + 1 else 0
        out[st] = k
        prev = st
    return out


def load_h2o(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    eps = 0.25
    ml = [r for r in payload["records"] if r["route"] == "ml"]
    ks = streak_positions([r["step"] for r in ml])
    return [{
        "system": "H2O", "step": r["step"], "k": ks[r["step"]],
        "slack": (eps - r["bound"]) / r["bound"],
        "violation": bool(r["error"] > eps),
        "r_over_qhat": r["error"] / r["bound"],  # r/(qhat) since B=qhat(s+d)
    } for r in ml]


def load_prod(db_path: Path, audit_path: Path, run_id: str, eps: float,
              system: str) -> list[dict]:
    import ase.db

    audit = json.loads(audit_path.read_text())
    truth = {d["step"]: d for d in audit["per_step"]}
    db = ase.db.connect(db_path)
    rows = sorted(db.select(run_id=run_id), key=lambda r: r.key_value_pairs["step"])
    ml = [r for r in rows if r.key_value_pairs["route"] == "ml"]
    ks = streak_positions([int(r.key_value_pairs["step"]) for r in ml])
    out = []
    for r in ml:
        st = int(r.key_value_pairs["step"])
        if st not in truth:
            continue  # not (yet) audited
        reason = r.data.get("reason", "")
        m = QHAT_RE.search(reason)
        qhat = float(m.group(1)) if m else float("nan")
        s = float(np.max(np.asarray(r.data["surrogate"]["uncertainty"], dtype=float)))
        bound = qhat * (s + DELTA)
        out.append({
            "system": system, "step": st, "k": ks[st],
            "slack": (eps - bound) / bound,
            "violation": bool(truth[st]["violation"]),
            "r_over_qhat": truth[st]["e"] / bound,
        })
    return out


def main() -> int:
    args = parse_args()
    points = (
        load_h2o(args.h2o_log)
        + load_prod(args.smoke_db, args.smoke_audit, args.smoke_run_id, 0.25, "smoke106")
        + load_prod(args.flagship_db, args.flagship_audit, "flagship-a-prod", 1.0,
                    "flagship474")
    )
    report = {
        "n_points": len(points),
        "per_system": {
            sysname: {
                "n": sum(1 for p in points if p["system"] == sysname),
                "violations": sum(1 for p in points
                                  if p["system"] == sysname and p["violation"]),
            }
            for sysname in ("H2O", "smoke106", "flagship474")
        },
        "slopes_rho_over_qhat": SLOPES,
        "points": points,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["per_system"], indent=2))

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
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300,
    })
    # final-width single column (3.4 in): true-size typography
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.subplots_adjust(left=0.13, right=0.97,
                        top=0.96 if args.no_title else 0.90, bottom=0.16)
    style = {"H2O": (OI["blue"], "o", "H$_2$O"),
             "smoke106": (OI["orange"], "s", "smoke-106"),
             "flagship474": (OI["green"], "D", "flagship-474")}
    for sysname, (color, marker, label) in style.items():
        sel = [p for p in points if p["system"] == sysname]
        ok = [p for p in sel if not p["violation"]]
        bad = [p for p in sel if p["violation"]]
        ax.plot([p["k"] for p in ok], [p["slack"] for p in ok], marker,
                ms=3.2, mfc="none", mec=color, mew=0.9, ls="none",
                label=f"{label} (n={len(sel)})")
        if bad:
            ax.plot([p["k"] for p in bad], [p["slack"] for p in bad], marker,
                    ms=5.5, color=OI["verm"], ls="none", zorder=5,
                    label=f"{label} violated")
    # threshold lines slack = (rho_r/qhat) k, one per system with its own
    # measured drift rate (SLOPES at the top of the file; recomputed from
    # the raw artifacts — see the SLOPES comment for provenance). Water's
    # line spans the full k range (streaks reach k=132); flagship/smoke
    # lines span 0..60 as before (smoke's exits the top at k~13).
    kx = np.linspace(0, 140, 200)
    kx_short = np.linspace(0, 60, 100)
    ax.plot(kx, SLOPES["H2O"] * kx, ls=":", lw=0.9, color=style["H2O"][0])
    ax.plot(kx_short, SLOPES["smoke106"] * kx_short, ls=":", lw=0.9,
            color=style["smoke106"][0])
    ax.plot(kx_short, SLOPES["flagship474"] * kx_short, ls=":", lw=0.9,
            color=style["flagship474"][0])
    ax.text(15, 2.0, f"{SLOPES['smoke106']:.2f}\n(smoke tail)", fontsize=6.5,
            color=OI["ink"], ha="left", va="top", linespacing=1.45)
    ax.text(62, 1.33, f"$\\rho/\\hat q$={SLOPES['flagship474']:.3f} (flagship)",
            fontsize=6.5, color=OI["ink"], ha="left", va="center",
            linespacing=1.45)
    ax.text(100, 0.16, f"{SLOPES['H2O']:.3f} (water-early)", fontsize=6.5,
            color=OI["ink"], ha="center", va="top", linespacing=1.45)
    # thin leader from the label (sitting in the empty pocket below the
    # line) up to the H2O threshold line at (100, 0.42)
    ax.plot([100, 100], [0.175, 0.41], ls="-", lw=0.6, color=OI["gray"],
            zorder=2)
    # provenance tag for the dotted threshold lines (referee: on-figure n
    # and provenance) — top-right corner is empty (max slack ~0.91)
    ax.text(138, 2.6, "lines: per-system measured $\\rho/\\hat q$",
            fontsize=6.5, color="black", ha="right", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("streak position $k$ (accepted steps since last label)")
    ax.set_ylabel(r"slack $(\varepsilon - B)/B$")
    ax.set_xlim(-2, 140)
    ax.set_ylim(1e-2, 3)
    # small font + tight handles: H2O points at k=87-89 sit just left of the
    # corner, so the legend must stay narrow to clear them
    ax.legend(loc="lower right", fontsize=6.0, handlelength=0.9,
              handletextpad=0.5, labelspacing=0.35, borderaxespad=0.1)
    if not args.no_title:
        ax.set_title("leak phase diagram: drift $\\times$ thin margin", loc="left")
    fig.savefig(args.figure.with_suffix(".pdf"))
    fig.savefig(args.figure.with_suffix(".png"), dpi=300)
    print(f"figure -> {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
