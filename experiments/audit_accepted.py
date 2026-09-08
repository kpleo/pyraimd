"""Accepted-step audit (the decisive experiment): extract ML-routed frames
from a run's store and score them against post-hoc DFT labels.

The conformal guarantee is marginal: over accepted steps, the fraction
with true max-atom force error > eps_acc should be ~= alpha. Accepted
steps pay no DFT at run time — that is the point of the switch — but the
store durably banks the surrogate prediction (forces + uncertainty) at
every step, so the ONLY missing half is the DFT truth. This script:

1. ``extract``: dumps every route=="ml" frame of a run to extxyz +
   manifest.jsonl (step, stored max-atom spread s, stored ML forces).
2. ``score``: joins the manifest with the labeling array's DFT jsonl and
   reports per-step true error e, the violation table at eps_acc, and the
   binomial 95% CI vs alpha — the empirical coverage certificate.

Usage:
    python experiments/audit_accepted.py extract --db loop.db \
        --run-id flagship-a-prod --out accepted_frames.extxyz \
        --manifest manifest.jsonl
    python experiments/audit_accepted.py score --manifest manifest.jsonl \
        --labels sync/audit_labels_*.jsonl --eps-acc 1.0 --alpha 0.05 \
        --report audit_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="dump accepted (ml-route) frames + manifest")
    e.add_argument("--db", required=True, type=Path)
    e.add_argument("--run-id", required=True)
    e.add_argument("--out", required=True, type=Path, help="accepted frames extxyz")
    e.add_argument("--manifest", required=True, type=Path)

    s = sub.add_parser("score", help="score DFT-labeled accepted steps vs eps/alpha")
    s.add_argument("--manifest", required=True, type=Path)
    s.add_argument("--labels", required=True, nargs="+", help="DFT labels jsonl")
    s.add_argument("--eps-acc", required=True, type=float)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--report", required=True, type=Path)
    return p.parse_args()


def cmd_extract(args: argparse.Namespace) -> int:
    import ase.db
    from ase.io import write

    db = ase.db.connect(args.db)
    rows = [r for r in db.select(run_id=args.run_id)
            if r.key_value_pairs["route"] == "ml"]
    rows.sort(key=lambda r: r.key_value_pairs["step"])
    if not rows:
        raise SystemExit(f"no accepted (ml) steps in {args.db} for {args.run_id!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as mf:
        for i, row in enumerate(rows):
            atoms = row.toatoms()
            sur = row.data["surrogate"]
            write(args.out, atoms, append=i > 0)
            mf.write(json.dumps({
                "audit_index": i,
                "step": int(row.key_value_pairs["step"]),
                "s_max_spread": float(np.max(np.asarray(sur["uncertainty"]))),
                "ml_forces_ev_a": np.asarray(sur["forces"], dtype=float).tolist(),
                "reason": row.data.get("reason", ""),
            }) + "\n")
    print(f"extracted {len(rows)} accepted steps "
          f"({rows[0].key_value_pairs['step']}..{rows[-1].key_value_pairs['step']}) "
          f"-> {args.out} + {args.manifest}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    labels: dict[int, dict] = {}
    import glob

    for pat in args.labels:
        for path in sorted(glob.glob(pat)):
            for line in Path(path).read_text().splitlines():
                rec = json.loads(line)
                labels[int(rec["frame_index"])] = rec

    per_step: list[dict] = []
    n_viol = 0
    for rec in manifest:
        i = rec["audit_index"]
        if i not in labels:
            continue
        ml = np.asarray(rec["ml_forces_ev_a"], dtype=float)
        dft = np.asarray(labels[i]["forces_ev_a"], dtype=float)
        e = float(np.linalg.norm(ml - dft, axis=1).max())
        viol = e > args.eps_acc
        n_viol += int(viol)
        per_step.append({
            "step": rec["step"], "s": rec["s_max_spread"], "e": e,
            "violation": bool(viol),
        })

    n = len(per_step)
    if n == 0:
        raise SystemExit("no manifest/label overlap — label audit_index mismatch?")
    rate = n_viol / n
    # Clopper-Pearson 95% interval for the violation probability.
    from scipy.stats import beta as beta_dist  # scipy ships with the env

    lo = float(beta_dist.ppf(0.025, n_viol, n - n_viol + 1)) if n_viol else 0.0
    hi = float(beta_dist.ppf(0.975, n_viol + 1, n - n_viol))
    # Three-way verdict on the violation probability vs alpha:
    # confirmed (CI entirely <= alpha), consistent (CI contains alpha),
    # violated (CI entirely above alpha). Small n starts at "consistent"
    # by construction — the interval tightens as accepted steps accrue.
    if hi <= args.alpha:
        verdict = "confirmed"
    elif lo > args.alpha:
        verdict = "violated"
    else:
        verdict = "consistent"
    report = {
        "n_accepted_audited": n,
        "eps_acc": args.eps_acc,
        "alpha": args.alpha,
        "violations": n_viol,
        "violation_rate": rate,
        "clopper_pearson_95": [lo, hi],
        "coverage_verdict": verdict,
        "per_step": per_step,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return cmd_extract(args) if args.cmd == "extract" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
