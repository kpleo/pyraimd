"""Leak-channel separation (red-team experiment E2): how much of the
streak-leak signature is time drift and how much is selection-into-calibration?

Two distinct channels were identified by the red-team review
(docs/red-team-review-2026-08-27.md):

  A. Streak freeze (time drift): qhat refreshes only at labels; the true
     ratio r climbs along an acceptance streak (measured on the flagship
     audit: +0.05/step).
  B. Selection into calibration: the window holds only REJECTED steps'
     (s, e) — the right tail of s — while acceptance uses the left tail;
     if r is not independent of s (e.g. the diversity-collapse regime:
     small s with high r), then the accept rule B(s) <= eps selects
     adversarially for high r, and no streak inflation fixes that.

This script measures channel B on the labeled streams of two systems:
  - flagship A stream (loop.db snapshot, DFT steps only — the actual
    calibration population);
  - the water decision log (every frame labeled — no selection gap at all,
    serving as the control where both populations are observable).

Reports per system: Spearman/Pearson corr(s, r) on labeled steps, median r
in the low-s vs high-s halves, and (flagship) the r trajectory within
rejection stretches as the drift reference. Output: JSON + stdout.

Usage:
    uv run python experiments/leak_channels.py \
        --flagship-db analysis/flagship_a_prod/loop.db \
        --h2o-log analysis/h2o_streak/decision_log_conformal.json \
        --out analysis/leak_channels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sstats

DELTA = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flagship-db", required=True, type=Path)
    p.add_argument("--h2o-log", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def flagship_stream(db_path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    import ase.db

    db = ase.db.connect(db_path)
    rows = sorted(
        db.select(run_id="flagship-a-prod"), key=lambda r: r.key_value_pairs["step"]
    )
    svals, evals, steps = [], [], []
    for r in rows:
        if r.key_value_pairs["route"] != "dft":
            continue
        sur, eng = r.data["surrogate"], r.data["engine"]
        s = float(np.max(np.asarray(sur["uncertainty"], dtype=float)))
        err = np.asarray(sur["forces"], dtype=float) - np.asarray(eng["forces"], dtype=float)
        svals.append(s)
        evals.append(float(np.linalg.norm(err, axis=1).max()))
        steps.append(int(r.key_value_pairs["step"]))
    return np.asarray(svals), np.asarray(evals), steps


def h2o_stream(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(log_path.read_text())
    svals = np.array([r["spread"] for r in payload["records"]])
    evals = np.array([r["error"] for r in payload["records"]])
    return svals, evals


def channel_b_report(s: np.ndarray, e: np.ndarray) -> dict:
    r = e / (s + DELTA)
    mid = float(np.median(s))
    low, high = r[s <= mid], r[s > mid]
    return {
        "n": int(len(s)),
        "pearson_s_r": float(sstats.pearsonr(s, r).statistic),
        "pearson_p": float(sstats.pearsonr(s, r).pvalue),
        "spearman_s_r": float(sstats.spearmanr(s, r).statistic),
        "spearman_p": float(sstats.spearmanr(s, r).pvalue),
        "r_median_low_s": float(np.median(low)),
        "r_median_high_s": float(np.median(high)),
        "r_median_ratio_low_over_high": float(np.median(low) / np.median(high)),
        "s_median_split": mid,
    }


def main() -> int:
    args = parse_args()
    report: dict = {}

    s_f, e_f, steps_f = flagship_stream(args.flagship_db)
    report["flagship_labeled"] = channel_b_report(s_f, e_f)

    # Drift reference: r trajectory within the longest rejection stretch
    # (steps 43-50 in segment 4): monotone rise = time-drift component.
    stretches: list[list[float]] = []
    cur: list[float] = []
    prev = None
    for st, s, e in zip(steps_f, s_f, e_f):
        if prev is not None and st == prev + 1:
            cur.append(e / (s + DELTA))
        else:
            if len(cur) >= 6:
                stretches.append(cur)
            cur = [e / (s + DELTA)]
        prev = st
    if len(cur) >= 6:
        stretches.append(cur)
    report["flagship_rejection_stretch_drifts"] = [
        {"len": len(st), "slope_per_step": float(np.polyfit(np.arange(len(st)), st, 1)[0])}
        for st in stretches
    ]

    s_w, e_w = h2o_stream(args.h2o_log)
    report["h2o_all_frames"] = channel_b_report(s_w, e_w)

    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
