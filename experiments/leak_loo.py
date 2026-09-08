"""Gate-0 leave-one-system-out (LOSO) validation of the leak-law threshold.

Pre-registered gate (docs/research-log.md, "玩具系统运输判据预注册", 0 号门,
2026-08-29, before any toy-system result): fit the threshold-line slope on
TWO of the three systems {H2O box, smoke box, flagship} and predict the
third system's per-step violation classification; success = >=95% of the
audited accepted steps classified to the correct side (243 at
preregistration; 244 with the current-code water log: 227+11+6 — the
existing phase diagram has zero counterexamples; this gate is its
leave-one-out robustness form). Each point is classified exactly once — in
the run where its own system is held out — so the pooled score is over all
points.

Classification rule (task specification, the leak law of E1-E3):
an accepted step at streak position k is predicted to violate iff
    slack < slope * k,   slack = (eps - B)/B at acceptance,
equivalent to rho*k >~ qhat*slack for a threshold line slack=(rho/qhat)*k.

Slopes rho/qhat per system — two sets, both reported (they coincide since
2026-08-29: the figure now carries the recomputed values):
  * documented: the values carried by experiments/leak_phase_diagram.py
    (SLOPES dict): flagship 0.02219 (absolute r drift +0.068/step at
    qhat~3.08), water early drift 0.00420 (+0.161/step at qhat~38.3,
    CURRENT-code log), smoke streak-C tail 0.23437 (+0.41/step at qhat~1.76).
  * recomputed from the raw artifacts (method per system, all verified
    against the stored r_over_qhat series):
    - flagship474: r_k = e/(s+1e-3) for the audited streak steps 37-42
      (k=0..5) from analysis/flagship_a_prod/audit_report_b1.json; rho =
      OLS slope of r vs k; qhat = median_k B_k/(s_k+1e-3) with
      B_k = eps/(1+slack_k) from the phase-diagram artifact (eps=1.0).
    - smoke106: same recipe on streak C (steps 60-64, k=0..4, eps=0.25)
      from analysis/smoke_7615821/audit_report_smoke.json.
    - H2O: current-code log analysis/h2o_streak/decision_log_conformal_current.json
      (5 streaks, lens [3,11,45,35,133]). With the RMS spread estimator
      there is no monotone early linear regime inside the first six
      positions of the longer streaks (their head-6 OLS slopes are
      negative — r dips before rising), so the per-streak drift is the
      FULL-streak OLS of r=error/(spread+1e-3) vs k divided by the
      streak's frozen qhat; the water slope is the maximum ratio among
      the early streaks (all but the dominant late 133-step one):
      streak 73-83 (len 11), +0.161/step at qhat 38.33 -> 0.00420.

Independence nuance (reported, not adjusted for): the flagship and smoke
slopes are fit on the violating streaks themselves (semi-independent — when
that system is in the training pair, its slope carries information about
the outcomes of its own streak, though not of the held-out system); the
water slope is outcome-independent (fit on early streaks with zero
violations). The H2O-held-out run is therefore the cleanest test design:
both training slopes are derived on OTHER systems, so no information about
the held-out outcomes enters the boundary at all.

Output: analysis/leak_loo.json + printed summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DELTA = 1e-3
GATE = 0.95
SYSTEMS = ("H2O", "smoke106", "flagship474")
# documented slopes (SLOPES dict of experiments/leak_phase_diagram.py —
# since the 2026-08-29 re-derivation these ARE the recomputed values)
SLOPES_DOC = {"H2O": 0.00420, "smoke106": 0.23437, "flagship474": 0.02219}
SLOPES_DOC_RANGE: dict[str, list[float]] = {}
EPS = {"H2O": 0.25, "smoke106": 0.25, "flagship474": 1.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagram", type=Path,
                   default=Path("analysis/leak_phase_diagram.json"))
    p.add_argument("--flagship-audit", type=Path,
                   default=Path("analysis/flagship_a_prod/audit_report_b1.json"))
    p.add_argument("--smoke-audit", type=Path,
                   default=Path("analysis/smoke_7615821/audit_report_smoke.json"))
    p.add_argument("--h2o-log", type=Path,
                   default=Path("analysis/h2o_streak/decision_log_conformal_current.json"))
    p.add_argument("--out", type=Path, default=Path("analysis/leak_loo.json"))
    return p.parse_args()


def ols_slope(k: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.asarray(k, dtype=float),
                            np.asarray(y, dtype=float), 1)[0])


def recompute_flagship(points: list[dict], audit: dict) -> dict:
    """rho = OLS drift of r=e/(s+delta) over the violating streak 37-42;
    qhat = median B/(s+delta) with B = eps/(1+slack) from the artifact."""
    per_step = {d["step"]: d for d in audit["per_step"]}
    sel = sorted(points, key=lambda p: p["k"])
    k, r, q = [], [], []
    for p in sel:
        d = per_step[p["step"]]
        B = EPS["flagship474"] / (1.0 + p["slack"])
        k.append(p["k"])
        r.append(d["e"] / (d["s"] + DELTA))
        q.append(B / (d["s"] + DELTA))
    rho = ols_slope(k, r)
    qhat = float(np.median(q))
    xcheck = ols_slope(k, [p["r_over_qhat"] for p in sel])
    return {"value": rho / qhat, "rho_per_step": rho, "qhat": qhat,
            "cross_check_ols_r_over_qhat": xcheck,
            "method": "OLS of r=e/(s+1e-3) vs k over steps 37-42 (k=0..5) "
                      "from audit_report_b1.json, divided by median "
                      "qhat=B/(s+1e-3) with B=eps/(1+slack) from "
                      "leak_phase_diagram.json"}


def recompute_smoke(points: list[dict], audit: dict) -> dict:
    """Same recipe as flagship, on streak C (steps 60-64)."""
    per_step = {d["step"]: d for d in audit["per_step"]}
    sel = sorted((p for p in points if p["step"] >= 60), key=lambda p: p["k"])
    k, r, q = [], [], []
    for p in sel:
        d = per_step[p["step"]]
        B = EPS["smoke106"] / (1.0 + p["slack"])
        k.append(p["k"])
        r.append(d["e"] / (d["s"] + DELTA))
        q.append(B / (d["s"] + DELTA))
    rho = ols_slope(k, r)
    qhat = float(np.median(q))
    xcheck = ols_slope(k, [p["r_over_qhat"] for p in sel])
    return {"value": rho / qhat, "rho_per_step": rho, "qhat": qhat,
            "cross_check_ols_r_over_qhat": xcheck,
            "method": "OLS of r=e/(s+1e-3) vs k over streak-C steps 60-64 "
                      "(k=0..4) from audit_report_smoke.json, divided by "
                      "median qhat=B/(s+1e-3) with B=eps/(1+slack)"}


def recompute_water(log: dict, points: list[dict]) -> dict:
    """Strongest early drifting streak of the CURRENT-code log (5 streaks,
    lens [3,11,45,35,133], longest 133 = steps 169-301). With the RMS
    spread estimator the old head-6 recipe is not applicable: the first
    six positions of the longer streaks have NEGATIVE OLS slopes (r dips
    before rising — no early linear regime), so the per-streak drift rate
    is the FULL-streak OLS of r=error/(spread+delta) vs k divided by the
    streak's frozen qhat. 'Early' = every streak except the dominant late
    one; the water slope is the maximum ratio among the early streaks
    (streak 73-83, len 11: +0.161/step at qhat 38.33 -> 0.00420)."""
    ml = [r for r in log["records"] if r["route"] == "ml"]
    streaks: list[list[dict]] = []
    prev = None
    for r in ml:
        if prev is not None and r["step"] == prev + 1:
            streaks[-1].append(r)
        else:
            streaks.append([r])
        prev = r["step"]
    late = max(streaks, key=len)
    early = [s for s in streaks if s is not late]
    detail = []
    for s in early:
        k = np.arange(len(s))
        r = np.array([x["error"] / (x["spread"] + DELTA) for x in s])
        rho = ols_slope(k, r) if len(s) > 1 else 0.0
        qhat = float(s[0]["qhat"])
        head = s[:6]
        rho6 = (ols_slope(np.arange(len(head)),
                          [x["error"] / (x["spread"] + DELTA) for x in head])
                if len(head) > 1 else float("nan"))
        detail.append({"steps": [s[0]["step"], s[-1]["step"]], "len": len(s),
                       "rho_per_step": rho, "qhat": qhat,
                       "ratio": rho / qhat,
                       "head6_rho_per_step_diagnostic": rho6})
    chosen = max(detail, key=lambda d: d["ratio"])
    # cross-check on the artifact's stored r_over_qhat, same streak
    lo, hi = chosen["steps"]
    sel = sorted((p for p in points if lo <= p["step"] <= hi),
                 key=lambda p: p["k"])
    xc = ols_slope([p["k"] for p in sel], [p["r_over_qhat"] for p in sel])
    return {"value": chosen["ratio"], "chosen_streak": chosen,
            "per_streak": detail,
            "cross_check_ols_r_over_qhat": xc,
            "method": "max over the early streaks (all but the dominant "
                      "133-step late one) of OLS(r=error/(spread+1e-3) vs k, "
                      "full streak) / frozen streak qhat, from "
                      "decision_log_conformal_current.json; chosen streak "
                      "73-83 (len 11)"}


def classify(sel: list[dict], slope: float) -> dict:
    """Predicted violation iff slack < slope*k; score vs audited flags."""
    tp = fp = tn = fn = 0
    wrong = []
    for p in sel:
        pred = p["slack"] < slope * p["k"]
        if pred and p["violation"]:
            tp += 1
        elif pred and not p["violation"]:
            fp += 1
            wrong.append({"step": p["step"], "k": p["k"],
                          "slack": p["slack"], "kind": "FP"})
        elif not pred and p["violation"]:
            fn += 1
            wrong.append({"step": p["step"], "k": p["k"],
                          "slack": p["slack"], "kind": "FN"})
        else:
            tn += 1
    n = len(sel)
    n_viol = tp + fn
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_violations": n_viol,
            "accuracy": (tp + tn) / n,
            "violation_recall": (tp / n_viol) if n_viol else None,
            "misclassified": wrong}


def separability(sel: list[dict]) -> dict:
    """Slope interval (lo, hi] that would classify this system perfectly:
    need slope > slack/k for every violation and slope <= slack/k for every
    compliant point with k>0. Feasible iff lo < hi."""
    lo, hi = 0.0, float("inf")
    for p in sel:
        if p["k"] == 0:
            # k=0: predicted violation iff slack<0 — infeasible for any
            # violation with slack>=0; compliant k=0 points never bind
            if p["violation"] and p["slack"] >= 0:
                return {"feasible": False, "slope_lo": None, "slope_hi": None,
                        "note": "violation at k=0 with slack>=0"}
            continue
        ratio = p["slack"] / p["k"]
        if p["violation"]:
            lo = max(lo, ratio)
        else:
            hi = min(hi, ratio)
    return {"feasible": lo < hi,
            "slope_lo": lo, "slope_hi": (hi if hi < float("inf") else None)}


def main() -> int:
    args = parse_args()
    diagram = json.loads(args.diagram.read_text())
    points = diagram["points"]
    by_sys = {s: [p for p in points if p["system"] == s] for s in SYSTEMS}
    n_total = len(points)

    f_audit = json.loads(args.flagship_audit.read_text())
    s_audit = json.loads(args.smoke_audit.read_text())
    h2o_log = json.loads(args.h2o_log.read_text())

    slopes_re = {
        "flagship474": recompute_flagship(by_sys["flagship474"], f_audit),
        "smoke106": recompute_smoke(by_sys["smoke106"], s_audit),
        "H2O": recompute_water(h2o_log, by_sys["H2O"]),
    }
    re_vals = {s: slopes_re[s]["value"] for s in SYSTEMS}

    print("measured slopes rho/qhat")
    for s in SYSTEMS:
        print(f"  {s:11s} recomputed {re_vals[s]:.5f}   "
              f"documented {SLOPES_DOC[s]:.3f}"
              + (f" (range {SLOPES_DOC_RANGE[s]})" if s in SLOPES_DOC_RANGE
                 else ""))

    variants = {
        "recomputed_max": (re_vals, max),
        "recomputed_mean": (re_vals, lambda v: float(np.mean(v))),
        "documented_max": (SLOPES_DOC, max),
        "documented_mean": (SLOPES_DOC, lambda v: float(np.mean(v))),
    }

    runs: dict[str, dict] = {}
    pooled: dict[str, dict] = {}
    for held in SYSTEMS:
        train = [s for s in SYSTEMS if s != held]
        runs[held] = {"held_out": held, "train_systems": train,
                      "n": len(by_sys[held]),
                      "n_violations": sum(p["violation"] for p in by_sys[held]),
                      "variants": {}}
        for vname, (slope_set, combine) in variants.items():
            slope = combine([slope_set[s] for s in train])
            res = classify(by_sys[held], slope)
            res.update({"slope": slope, "combine": vname.split("_")[1],
                        "train_slopes": {s: slope_set[s] for s in train},
                        "gate_pass": res["accuracy"] >= GATE})
            runs[held]["variants"][vname] = res
            pooled.setdefault(vname, {"n_correct": 0, "tp": 0, "fn": 0})
            pooled[vname]["n_correct"] += res["tp"] + res["tn"]
            pooled[vname]["tp"] += res["tp"]
            pooled[vname]["fn"] += res["fn"]

    print("\nleave-one-system-out runs (rule: predicted violation iff "
          "slack < slope*k)")
    for held in SYSTEMS:
        r = runs[held]
        print(f"\n  held out {held} (n={r['n']}, violations="
              f"{r['n_violations']}; trained on {r['train_systems']})")
        for vname in variants:
            v = r["variants"][vname]
            rec = ("n/a (no violations)" if v["violation_recall"] is None
                   else f"{v['violation_recall']:.3f}")
            print(f"    {vname:16s} slope={v['slope']:.5f}  "
                  f"acc {v['accuracy']:.4f} ({v['tp'] + v['tn']}/{v['n']})  "
                  f"recall {rec}  "
                  f"tp{v['tp']} fp{v['fp']} fn{v['fn']} tn{v['tn']}  "
                  f"gate {'PASS' if v['gate_pass'] else 'FAIL'}")

    print(f"\npooled gate ({n_total} steps, each classified once; "
          f"criterion >= {GATE:.2f})")
    for vname in variants:
        p = pooled[vname]
        acc = p["n_correct"] / n_total
        p.update({"n_total": n_total, "accuracy": acc,
                  "violation_recall": (p["tp"] / (p["tp"] + p["fn"])
                                       if p["tp"] + p["fn"] else None),
                  "gate_pass": acc >= GATE})
        print(f"  {vname:16s} {p['n_correct']}/{n_total} = {acc:.4f}  "
              f"gate {'PASS' if p['gate_pass'] else 'FAIL'}")

    # diagnostics ------------------------------------------------------------
    self_consistency = {}
    for s in SYSTEMS:
        res = classify(by_sys[s], re_vals[s])
        self_consistency[s] = {"own_slope": re_vals[s],
                               "accuracy": res["accuracy"],
                               "violation_recall": res["violation_recall"],
                               "fp": res["fp"], "fn": res["fn"]}
    sep = {s: separability(by_sys[s]) for s in SYSTEMS}

    w = by_sys["H2O"]
    w_ks = np.array([p["k"] for p in w])
    w_sl = np.array([p["slack"] for p in w])
    smoke_k3 = [{"step": p["step"], "slack": p["slack"],
                 "violation": p["violation"]}
                for p in by_sys["smoke106"] if p["k"] == 3]
    structural = {
        "water_slack_saturation": {
            "k_max": int(w_ks.max()), "slack_max": float(w_sl.max()),
            "min_slack_over_k": float((w_sl[w_ks >= 1]
                                       / w_ks[w_ks >= 1]).min()),
            "note": "the 133-step late streak keeps slack bounded (spread "
                    "floored by delta -> B floored -> slack <= 0.911) while "
                    "slope*k grows unbounded; only slopes <= 4.6e-4 classify "
                    "all 227 water points correctly, so any transported "
                    "drift-regime slope is pure false alarm there"},
        "smoke_k3_inversion": {
            "points": smoke_k3,
            "note": "at k=3 the compliant point (slack 0.0499, regime-B "
                    "streak 40-43) sits BELOW the violation (slack 0.1395, "
                    "regime-C streak 60-64); no single line through the "
                    "origin can separate them (need slope>0.0465 to catch "
                    "the violation but <=0.0166 to keep the compliant "
                    "point)"},
        "flagship_fp_origin": "in the flagship-held-out run the single "
            "error is a false positive at k=1 (slack 0.0576): the "
            "conservative max boundary inherits smoke's event-driven tail "
            "slope (~0.23), ~10x steeper than flagship's own 0.022",
    }

    nuance = ("flagship474 and smoke106 slopes are fit on the violating "
              "streaks themselves (semi-independent: when such a system is "
              "in the training pair its slope encodes its own streak's "
              "outcomes, though never the held-out system's); the H2O "
              "slope is outcome-independent (early streaks, zero "
              "violations). The H2O-held-out run is the cleanest test "
              "design: both training slopes come from OTHER systems, so "
              "zero information about the held-out outcomes enters the "
              "boundary; and with 0 violations it is a pure false-alarm "
              "control.")
    strongest = ("Strongest positive evidence: the flagship474-held-out "
                 "run — the only run where the held-out system actually "
                 "contains violations AND the transferred boundary catches "
                 "all of them (recall 4/4 under every variant; accuracy "
                 "5/6; one training slope, water's, is outcome-independent). "
                 "The H2O-held-out run is the cleanest test by design "
                 "(zero circularity, pure false-alarm control) and it fails "
                 "decisively (accuracy 0.03-0.07 across variants): any "
                 "slope that transports from the two drift regimes "
                 "over-predicts violations across the 133-step late streak, "
                 "where slack stays bounded at ~0.91 while slope*k grows "
                 "unbounded. The smoke106-held-out run shows the reverse "
                 "limitation: shallow slow-drift slopes (0.004-0.022) "
                 "cannot anticipate smoke's event-driven tail (recall "
                 "1/2), and its k=3 compliant/violation pair is "
                 "structurally unseparable. Net reading: the leak-law "
                 "threshold is regime-local — a single global slope does "
                 "not transport across systems; within a matched "
                 "(slow-drift) regime the boundary does transport "
                 "(flagship recall 4/4).")
    best_smoke = max(v["accuracy"] for v in runs["smoke106"]["variants"].values())
    verdict = ("GATE-0 FAIL under the literal pre-registered rule: pooled "
               f"over the {n_total} steps, the best variant classifies "
               f"{max(p['n_correct'] for p in pooled.values())}/{n_total} "
               "(<< 95%) correctly; every per-system run is also below "
               f"95% (flagship 5/6, smoke best {best_smoke:.2f} "
               "(10/11), H2O ~6-16/227).")

    report = {
        "task": "gate-0 leave-one-system-out leak-law validation",
        "preregistration": "docs/research-log.md 2026-08-29 '玩具系统运输判据"
                           "预注册' 0号门: >=95% of the 243 steps classified "
                           "to the correct side (243 at preregistration; "
                           "244 with the current-code water log: 227+11+6)",
        "rule": "predicted violation iff slack < slope*k "
                "(equivalently rho*k >~ qhat*slack); one threshold line "
                "per run = combination of the two training systems' "
                "measured slopes (max = conservative boundary; mean "
                "variant also reported)",
        "n_total": n_total,
        "gate_threshold": GATE,
        "slopes": {
            s: {"recomputed": slopes_re[s],
                "documented": {"value": SLOPES_DOC[s],
                               "range": SLOPES_DOC_RANGE.get(s),
                               "source": "SLOPES dict of "
                                         "experiments/leak_phase_diagram.py "
                                         "(= recomputed values since the "
                                         "2026-08-29 re-derivation)"}}
            for s in SYSTEMS
        },
        "runs": runs,
        "pooled": pooled,
        "diagnostics": {
            "self_consistency_own_recomputed_slope": self_consistency,
            "per_system_separability_interval": sep,
            "structural_findings": structural,
        },
        "independence_nuance": nuance,
        "strongest_evidence": strongest,
        "verdict": verdict,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n{verdict}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
