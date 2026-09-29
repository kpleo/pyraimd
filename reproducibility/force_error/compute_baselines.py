"""Prospective free-predictor baselines for the 1160 formal
successive-anchor segments (tau = 1 fs). Uses only formal_strips.csv.
Every predictor for segment j+1 uses information available at anchor a_{j+1}:
  probe            : 0.5 v_{j+1}^2 kappa_{j+1} tau^2 (two extra reference calls)
  persistence      : W_j (previous measured segment work)
  persistence_v2   : W_j v_{j+1}^2 / v_j^2
  constant_rate    : mean(W_1..W_j)  (constant-drift assumption)
The first segment of each path has no history and is excluded for all
predictors; S is the total absolute measured work of segments 2..M."""
import csv, sys
from pathlib import Path
import numpy as np
src = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = list(csv.DictReader(src.open()))
res = []
for ref in ("MACE", "PBE"):
    for conf in ("1", "2"):
        s = sorted([r for r in rows if r["reference"] == ref and r["configuration"] == conf], key=lambda r: int(r["segment"]))
        W = np.array([float(r["W_meV"]) for r in s]); P = np.array([float(r["W_pred_meV"]) for r in s])
        v2 = np.array([float(r["speed_A_fs"]) ** 2 for r in s])
        S = np.abs(W[1:]).sum()
        preds = dict(probe=P[1:], persistence=W[:-1], persistence_v2=W[:-1] * v2[1:] / v2[:-1],
                     constant_rate=np.cumsum(W)[:-1] / np.arange(1, len(W)))
        for name, p in preds.items():
            e = p - W[1:]
            res.append(dict(reference=ref, configuration=conf, predictor=name, n_segments=len(e),
                            Eseg_percent=100 * np.abs(e).sum() / S,
                            Ecum_percent=100 * np.abs(np.cumsum(e)).max() / S,
                            end_bias_percent=100 * e.sum() / S,
                            max_segment_error_percent=100 * np.max(np.abs(e) / np.abs(W[1:])),
                            extra_reference_calls_per_segment=2 if name == "probe" else 0))
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
for r in res:
    print(f"{r['reference']}{r['configuration']} {r['predictor']:15s} Eseg={r['Eseg_percent']:.3f}% Ecum={r['Ecum_percent']:.3f}% bias={r['end_bias_percent']:+.3f}% max={r['max_segment_error_percent']:.2f}%")
