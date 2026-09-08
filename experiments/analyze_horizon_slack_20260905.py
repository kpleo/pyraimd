"""Decompose saved one-step envelope slack; retrospective, no new force calls."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/execution_20260905/h2o_feasibility"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def norm(force):
    return float(np.linalg.norm(force, axis=1).max())


rows = []
for directory in sorted(DATA.glob("kinetic_*")):
    for anchors_file in sorted(directory.glob("horizon_*_anchors.jsonl")):
        seed = int(anchors_file.name.split("_")[1])
        states = {r["step"]: r for r in read_rows(directory / f"horizon_{seed}_steps.jsonl")}
        for anchor in read_rows(anchors_file):
            k = anchor["step"]
            if not anchor["forecast"] or k + 1 not in states:
                continue
            a, b = states[k], states[k + 1]
            r0 = np.array(a["surrogate_forces"]) - a["reference_forces"]
            r1 = np.array(b["surrogate_forces"]) - b["reference_forces"]
            e0, e1, increment = norm(r0), norm(r1), norm(r1 - r0)
            _, predicted_length, upper = anchor["forecast"][0]
            length = float(np.linalg.norm(np.array(b["positions"]) - a["positions"]))
            growth_slack = upper - e0 - increment
            triangle_slack = e0 + increment - e1
            assert abs((growth_slack + triangle_slack) - (upper - e1)) < 1e-12
            assert abs(e0 - anchor["e0"]) < 1e-12
            assert triangle_slack >= -1e-12
            assert abs(length - predicted_length) < 1e-12
            rows.append({
                "level": directory.name, "seed": seed, "anchor_step": k,
                "e0_ev_A": e0, "next_error_ev_A": e1,
                "residual_increment_ev_A": increment,
                "first_forecast_bound_ev_A": upper,
                "growth_magnitude_slack_ev_A": growth_slack,
                "norm_triangle_slack_ev_A": triangle_slack,
                "forecast_length_error_A": abs(length - predicted_length),
                "zero_horizon": anchor["horizon_fs"] == 0,
                "next_within_budget": e1 <= 0.1,
                "exact_increment_scalar_bound_within_budget": e0 + increment <= 0.1,
            })
summary = {
    "interpretation": "Retrospective decomposition on the actually generated paths. "
    "Future reference residuals enter diagnosis only; these are neither a new "
    "admission rule nor a prospective certificate or a replay of changed dynamics.",
    "identity": "B1-e1 = (B1-e0-||r1-r0||) + (e0+||r1-r0||-e1)",
    "first_step_forecast_matches_actual": True,
    "new_reference_evaluations": 0,
    "new_surrogate_evaluations": 0,
    "levels": [],
}
for level in sorted({r["level"] for r in rows}):
    selected = [r for r in rows if r["level"] == level]
    refused = [r for r in selected if r["zero_horizon"]]
    safe = [r for r in refused if r["next_within_budget"]]
    summary["levels"].append({
        "level": level, "valid_anchors_with_next_state": len(selected),
        "zero_horizon_anchors": len(refused),
        "zero_horizon_next_observed_within_budget": len(safe),
        "among_these_exact_increment_scalar_bound_within_budget": sum(
            r["exact_increment_scalar_bound_within_budget"] for r in safe),
        "among_these_scalar_triangle_bound_still_over_budget": sum(
            not r["exact_increment_scalar_bound_within_budget"] for r in safe),
        "recorded_growth_magnitude_below_actual_increment": sum(
            r["growth_magnitude_slack_ev_A"] < -1e-12 for r in selected),
        "next_error_above_first_forecast_scalar_bound": sum(
            r["next_error_ev_A"] > r["first_forecast_bound_ev_A"] + 1e-12
            for r in selected),
        "mean_growth_slack_on_zero_horizon_safe_next_ev_A":
            float(np.mean([r["growth_magnitude_slack_ev_A"] for r in safe])) if safe else None,
        "mean_triangle_slack_on_zero_horizon_safe_next_ev_A":
            float(np.mean([r["norm_triangle_slack_ev_A"] for r in safe])) if safe else None,
    })
with (DATA / "horizon_slack.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
(DATA / "horizon_slack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
