"""One fixed-protocol, forward oscillator experiment; standard library only.

Run from the repository root:
    uv run --no-sync python -m experiments.toy.prospective_horizon

The force approximation is a reference-corrected harmonic force, not a learned
model. Read analysis/revision_20260905/prospective_horizon/protocol.md first.
No production modules, numerical training, or archived trajectory replay are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "analysis/revision_20260905/prospective_horizon"
POLICIES = ("class_horizon", "empirical_horizon", "fixed_streak")
STEP_FIELDS = (
    "split", "policy", "seed", "stiffness", "amplitude", "step", "time_start",
    "time_end", "route", "eligible_before_random", "random_uniform", "x", "v",
    "bias_before", "bias_used", "error_before_correction", "interval_max_error",
    "start_violation", "interval_violation", "anchor_step", "anchor_initial_error",
    "anchor_margin", "anchor_speed_bound", "drift_coefficient", "drift_rate",
    "trust_time", "age_at_interval_end", "prospective_envelope_at_end",
    "envelope_counterexample", "unreferenced_streak", "position_error_at_end",
    "velocity_error_at_end", "relative_energy_change_at_end",
)
REFERENCE_FIELDS = (
    "split", "policy", "seed", "step", "time", "reason", "x", "v",
    "reference_force", "bias_before", "bias_after", "initial_error", "margin",
    "speed_bound", "drift_coefficient", "drift_rate", "trust_time",
    "full_correction_used",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_force(x: float, stiffness: float) -> float:
    return -stiffness * x


def advance(x: float, v: float, bias: float, dt: float) -> tuple[float, float]:
    """Exact flow of x'' = -x + bias over the fixed physical interval."""
    c, s = math.cos(dt), math.sin(dt)
    q = x - bias
    return bias + q * c + v * s, -q * s + v * c


def interval_max_error(
    x: float, v: float, bias: float, stiffness: float, dt: float, x_end: float,
) -> tuple[float, int]:
    """Evaluation only: exact maximum absolute force error on a whole interval.

    The signed error is affine in the sinusoidal position. Its absolute maximum
    is at an endpoint or a stationary position; no temporal sampling is needed.
    Return the number of exact force evaluations used for measurement.
    """
    positions = [x, x_end]
    phase = math.atan2(v, x - bias)
    first = math.ceil(-phase / math.pi)
    last = math.floor((dt - phase) / math.pi)
    for index in range(first, last + 1):
        tau = phase + index * math.pi
        if 0.0 < tau < dt:
            positions.append(advance(x, v, bias, tau)[0])
    errors = [abs(reference_force(q, stiffness) - (-q + bias)) for q in positions]
    return max(errors), len(positions)


@dataclass(frozen=True)
class Anchor:
    step: int
    initial_error: float
    margin: float
    speed_bound: float
    coefficient: float
    drift_rate: float
    trust_time: float


def correct_and_anchor(
    x: float, v: float, bias: float, observed_force: float, step: int,
    coefficient: float, cfg: dict,
) -> tuple[float, Anchor, bool]:
    """Uses only the current state, an acquired label, and a frozen coefficient."""
    bias += cfg["bias_correction_fraction"] * (observed_force - (-x + bias))
    initial_error = abs(observed_force - (-x + bias))
    full = initial_error >= cfg["force_tolerance"]
    if full:
        bias = observed_force + x
        initial_error = abs(observed_force - (-x + bias))
    margin = max(0.0, cfg["force_tolerance"] - initial_error)
    speed = math.hypot(x - bias, v)
    drift_rate = coefficient * speed
    # The declared initial conditions have positive speed bounds.
    if drift_rate <= 0.0:
        raise ValueError("This fixed protocol requires a positive drift rate")
    return bias, Anchor(step, initial_error, margin, speed, coefficient,
                        drift_rate, margin / drift_rate), full


def trajectory_specs(cfg: dict) -> list[dict]:
    specs = []
    for split in ("calibration", "heldout", "outside_class"):
        group = cfg[split]
        for seed, amplitude in zip(group["seeds"], group["amplitudes"], strict=True):
            phase = random.Random(seed).random() * 2.0 * math.pi
            stiffness = group["stiffness"]
            specs.append({
                "split": split, "seed": seed, "amplitude": amplitude,
                "stiffness": stiffness, "phase": phase,
                "x0": amplitude * math.cos(phase),
                "v0": -amplitude * math.sqrt(stiffness) * math.sin(phase),
                "random_check_seed": seed + cfg["random_seed_offset"],
            })
    return specs


def simulate(
    spec: dict, policy: str, coefficient: float, fixed_k: int | None, cfg: dict,
    step_writer: csv.DictWriter, ref_writer: csv.DictWriter,
    failure_writer: csv.DictWriter,
) -> dict:
    dt, epsilon = cfg["dt"], cfg["force_tolerance"]
    tolerance = cfg["floating_point_tolerance"]
    rng = random.Random(spec["random_check_seed"])
    x, v, bias = spec["x0"], spec["v0"], 0.0
    stiffness = spec["stiffness"]
    energy0 = 0.5 * (v * v + stiffness * x * x)
    anchor = None
    streak = maximum_streak = 0
    count_names = (
        "n_reference", "n_initial_reference", "n_forced_reference", "n_random_reference",
        "n_unreferenced", "n_start_violations", "n_interval_violations",
        "n_envelope_counterexamples", "n_eligible", "n_eligible_start_violations",
        "n_random_start_violations", "n_full_corrections", "n_measurement_force_calls",
    )
    counts = dict.fromkeys(count_names, 0)
    squared_position_error = squared_velocity_error = 0.0
    maximum_energy_change = maximum_force_error = maximum_envelope_excess = 0.0
    initial_errors, margins, horizons = [], [], []
    first_counterexample = None

    for step in range(cfg["n_steps"]):
        # Critical ordering: eligibility and the random draw precede reference
        # acquisition and ALL measurement-only force evaluations at this state.
        if anchor is None:
            eligible = False
        elif policy == "fixed_streak":
            eligible = streak < fixed_k
        else:
            end_age = (step - anchor.step + 1) * dt
            eligible = end_age <= anchor.trust_time
        random_uniform = rng.random()  # one draw per physical step for pairing
        random_check = eligible and random_uniform < cfg["random_reference_probability"]
        route = ("initial_reference" if anchor is None else
                 "forced_reference" if not eligible else
                 "random_reference" if random_check else "unreferenced")
        bias_before = bias
        if route != "unreferenced":
            observed_force = reference_force(x, stiffness)
            counts["n_reference"] += 1
            counts[f"n_{route}"] += 1
            bias, anchor, full = correct_and_anchor(
                x, v, bias, observed_force, step, coefficient, cfg,
            )
            counts["n_full_corrections"] += int(full)
            initial_errors.append(anchor.initial_error)
            margins.append(anchor.margin)
            horizons.append(anchor.trust_time)
            ref_writer.writerow({
                "split": spec["split"], "policy": policy, "seed": spec["seed"],
                "step": step, "time": step * dt, "reason": route, "x": x, "v": v,
                "reference_force": observed_force, "bias_before": bias_before,
                "bias_after": bias, "initial_error": anchor.initial_error,
                "margin": anchor.margin, "speed_bound": anchor.speed_bound,
                "drift_coefficient": coefficient, "drift_rate": anchor.drift_rate,
                "trust_time": anchor.trust_time, "full_correction_used": int(full),
            })
            streak = 0
        else:
            counts["n_unreferenced"] += 1
            streak += 1
            maximum_streak = max(maximum_streak, streak)

        x_end, v_end = advance(x, v, bias, dt)
        # Measurement only below this line. These values never update a policy.
        error_before = abs(reference_force(x, stiffness) - (-x + bias_before))
        maximum_error, measurement_calls = interval_max_error(
            x, v, bias, stiffness, dt, x_end,
        )
        counts["n_measurement_force_calls"] += measurement_calls + 1
        start_violation = error_before > epsilon + tolerance
        interval_violation = maximum_error > epsilon + tolerance
        end_age = (step - anchor.step + 1) * dt
        envelope = anchor.initial_error + anchor.drift_rate * end_age
        envelope_counterexample = maximum_error > envelope + tolerance
        if eligible:
            counts["n_eligible"] += 1
            counts["n_eligible_start_violations"] += int(start_violation)
        if random_check:
            counts["n_random_start_violations"] += int(start_violation)
        if route == "unreferenced":
            counts["n_start_violations"] += int(start_violation)
            counts["n_interval_violations"] += int(interval_violation)
            counts["n_envelope_counterexamples"] += int(envelope_counterexample)
            maximum_force_error = max(maximum_force_error, maximum_error)
            maximum_envelope_excess = max(maximum_envelope_excess, maximum_error-envelope)

        time_end = (step + 1) * dt
        omega = math.sqrt(stiffness)
        c, s = math.cos(omega * time_end), math.sin(omega * time_end)
        true_x = spec["x0"] * c + spec["v0"] / omega * s
        true_v = -spec["x0"] * omega * s + spec["v0"] * c
        position_error, velocity_error = x_end - true_x, v_end - true_v
        squared_position_error += position_error ** 2
        squared_velocity_error += velocity_error ** 2
        energy_change = (0.5 * (v_end*v_end + stiffness*x_end*x_end) - energy0) / energy0
        maximum_energy_change = max(maximum_energy_change, abs(energy_change))
        row = {
            "split": spec["split"], "policy": policy, "seed": spec["seed"],
            "stiffness": stiffness, "amplitude": spec["amplitude"], "step": step,
            "time_start": step*dt, "time_end": time_end, "route": route,
            "eligible_before_random": int(eligible), "random_uniform": random_uniform,
            "x": x, "v": v, "bias_before": bias_before, "bias_used": bias,
            "error_before_correction": error_before, "interval_max_error": maximum_error,
            "start_violation": int(start_violation), "interval_violation": int(interval_violation),
            "anchor_step": anchor.step, "anchor_initial_error": anchor.initial_error,
            "anchor_margin": anchor.margin, "anchor_speed_bound": anchor.speed_bound,
            "drift_coefficient": coefficient, "drift_rate": anchor.drift_rate,
            "trust_time": anchor.trust_time, "age_at_interval_end": end_age,
            "prospective_envelope_at_end": envelope,
            "envelope_counterexample": int(envelope_counterexample),
            "unreferenced_streak": streak, "position_error_at_end": position_error,
            "velocity_error_at_end": velocity_error,
            "relative_energy_change_at_end": energy_change,
        }
        step_writer.writerow(row)
        if route == "unreferenced" and (interval_violation or envelope_counterexample):
            failure_writer.writerow(row)
            if first_counterexample is None:
                first_counterexample = row
        x, v = x_end, v_end

    n = cfg["n_steps"]
    if counts["n_reference"] + counts["n_unreferenced"] != n:
        raise AssertionError("All intervals must be classified exactly once")
    if counts["n_eligible"] != counts["n_random_reference"] + counts["n_unreferenced"]:
        raise AssertionError("Eligible candidates must have the stated random-check denominator")
    return {
        **spec, "policy": policy, "drift_coefficient": coefficient, "fixed_k": fixed_k,
        "n_steps": n, "physical_duration": n*dt, **counts,
        "reference_fraction": counts["n_reference"]/n,
        "unreferenced_physical_duration": counts["n_unreferenced"]*dt,
        "interval_violation_rate": counts["n_interval_violations"]/counts["n_unreferenced"],
        "start_violation_rate": counts["n_start_violations"]/counts["n_unreferenced"],
        "maximum_unreferenced_streak": maximum_streak,
        "maximum_unreferenced_streak_time": maximum_streak*dt,
        "maximum_unreferenced_force_error": maximum_force_error,
        "maximum_positive_envelope_excess": maximum_envelope_excess,
        "initial_error_min": min(initial_errors), "initial_error_mean": statistics.mean(initial_errors),
        "initial_error_max": max(initial_errors), "margin_min": min(margins),
        "margin_mean": statistics.mean(margins), "margin_max": max(margins),
        "trust_time_min": min(horizons), "trust_time_median": statistics.median(horizons),
        "trust_time_max": max(horizons),
        "squared_position_error_sum": squared_position_error,
        "squared_velocity_error_sum": squared_velocity_error,
        "position_rmse": math.sqrt(squared_position_error/n),
        "velocity_rmse": math.sqrt(squared_velocity_error/n),
        "final_relative_energy_change": energy_change,
        "maximum_absolute_relative_energy_change": maximum_energy_change,
        "first_counterexample": first_counterexample,
    }


def aggregate(rows: list[dict], cfg: dict) -> dict:
    keys = [k for k in rows[0] if k.startswith("n_")]
    totals = {k: sum(row[k] for row in rows) for k in keys}
    n, n_free, n_eligible = totals["n_steps"], totals["n_unreferenced"], totals["n_eligible"]
    return {
        "n_trajectories": len(rows), **totals, "physical_duration": n*cfg["dt"],
        "unreferenced_physical_duration": n_free*cfg["dt"],
        "reference_fraction": totals["n_reference"]/n,
        "unreferenced_fraction": n_free/n,
        "interval_violation_rate": totals["n_interval_violations"]/n_free,
        "start_violation_rate": totals["n_start_violations"]/n_free,
        "random_checked_start_violation_rate": (
            totals["n_random_start_violations"]/totals["n_random_reference"]
            if totals["n_random_reference"] else None),
        "eligible_start_violation_rate_full_measurement": totals["n_eligible_start_violations"]/n_eligible,
        "eligible_start_violation_rate_inverse_probability_estimate": (
            totals["n_random_start_violations"]/cfg["random_reference_probability"]/n_eligible),
        "trajectories_with_interval_violation": sum(row["n_interval_violations"] > 0 for row in rows),
        "trajectories_with_envelope_counterexample": sum(row["n_envelope_counterexamples"] > 0 for row in rows),
        "maximum_unreferenced_streak_time": max(row["maximum_unreferenced_streak_time"] for row in rows),
        "maximum_unreferenced_force_error": max(row["maximum_unreferenced_force_error"] for row in rows),
        "maximum_positive_envelope_excess": max(row["maximum_positive_envelope_excess"] for row in rows),
        "position_rmse": math.sqrt(sum(row["squared_position_error_sum"] for row in rows)/n),
        "velocity_rmse": math.sqrt(sum(row["squared_velocity_error_sum"] for row in rows)/n),
        "mean_final_relative_energy_change": statistics.mean(row["final_relative_energy_change"] for row in rows),
        "maximum_absolute_relative_energy_change": max(row["maximum_absolute_relative_energy_change"] for row in rows),
        "first_counterexample": next((row["first_counterexample"] for row in rows if row["first_counterexample"]), None),
    }


def invariants(cfg: dict) -> dict:
    """One inexpensive analytic consistency check, no pilot trajectory or tuning."""
    if cfg["mass"] != 1.0 or cfg["approximate_stiffness"] != 1.0:
        raise ValueError("Closed-form propagator assumes mass and approximate stiffness 1")
    seeds = [seed for split in ("calibration", "heldout", "outside_class") for seed in cfg[split]["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise AssertionError("Trajectory seeds must be disjoint")
    x, v, bias = 0.7, -0.4, 0.1
    x1, v1 = advance(x, v, bias, cfg["dt"])
    defect = abs((x-bias)**2 + v*v - ((x1-bias)**2 + v1*v1))
    # This interval straddles a turning point, exercising the interior maximum.
    radius, phase, dt = 0.8, 0.003, cfg["dt"]
    q, speed = radius*math.cos(phase), radius*math.sin(phase)
    peak, _ = interval_max_error(q, speed, 0.0, 1.2, dt, advance(q, speed, 0.0, dt)[0])
    peak_defect = abs(peak - 0.2*radius)
    if max(defect, peak_defect) > cfg["floating_point_tolerance"]:
        raise AssertionError("Analytic propagation or continuous maximum is inconsistent")
    return {"shifted_energy_invariant_defect": defect, "interior_maximum_defect": peak_defect,
            "trajectory_seeds_disjoint": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Configuration remains the fixed, checked-in protocol even for a fresh output path.
    protocol_path = DEFAULT_OUTPUT / "protocol.json"
    protocol_text_path = DEFAULT_OUTPUT / "protocol.md"
    cfg = json.loads(protocol_path.read_text())
    products = ("run_started.json", "calibration.json", "summary.json", "steps.csv",
                "reference_events.csv", "counterexamples.csv", "trajectories.csv", "trajectories.json")
    if any((out/name).exists() for name in products):
        raise FileExistsError("Preserve prior results: choose a fresh --output directory")
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    metadata = {
        "started_utc": started, "protocol_sha256": sha256(protocol_path),
        "protocol_text_sha256": sha256(protocol_text_path), "source_sha256": sha256(Path(__file__)),
        "python_version": sys.version, "platform": platform.platform(),
        "command": "uv run --no-sync python -m experiments.toy.prospective_horizon",
        "output_directory": str(out), "execution": "forward, not replay",
    }
    write_json(out/"run_started.json", metadata)
    checks = invariants(cfg)
    specs = trajectory_specs(cfg)
    results = []
    with (out/"steps.csv").open("w", newline="") as step_file, \
         (out/"reference_events.csv").open("w", newline="") as ref_file, \
         (out/"counterexamples.csv").open("w", newline="") as failure_file:
        step_writer = csv.DictWriter(step_file, fieldnames=STEP_FIELDS)
        ref_writer = csv.DictWriter(ref_file, fieldnames=REFERENCE_FIELDS)
        failure_writer = csv.DictWriter(failure_file, fieldnames=STEP_FIELDS)
        for writer in (step_writer, ref_writer, failure_writer):
            writer.writeheader()
        d_class = cfg["class_stiffness_mismatch_bound"]
        for spec in specs:
            if spec["split"] == "calibration":
                results.append(simulate(spec, "class_horizon", d_class, None, cfg,
                                        step_writer, ref_writer, failure_writer))
        calibration_totals = aggregate(results, cfg)
        positions = cfg["empirical_calibration_positions"]
        force_values = [reference_force(x, cfg["empirical_calibration_stiffness"]) for x in positions]
        residuals = [force+x for force, x in zip(force_values, positions, strict=True)]
        d_empirical = abs((residuals[1]-residuals[0])/(positions[1]-positions[0]))
        p = cfg["random_reference_probability"]
        low, high = cfg["fixed_streak_candidates"]
        rates = {k: p/(1.0-(1.0-p)**(k+1)) for k in range(low, high+1)}
        fixed_k = min(rates, key=lambda k: (abs(rates[k]-calibration_totals["reference_fraction"]), k))
        development_calls = calibration_totals["n_reference"] + len(positions)
        calibration = {
            "frozen_before_heldout_utc": datetime.now(timezone.utc).isoformat(),
            "fixed_k": fixed_k, "fixed_reference_spacing_without_random": (fixed_k+1)*cfg["dt"],
            "fixed_maximum_unreferenced_streak_time": fixed_k*cfg["dt"],
            "fixed_expected_reference_fraction": rates[fixed_k],
            "empirical_drift_coefficient": d_empirical,
            "empirical_coefficient_reference_calls": len(positions),
            "empirical_calibration_positions": positions, "empirical_calibration_forces": force_values,
            "class_drift_coefficient": d_class, "calibration_results": calibration_totals,
            "shared_development_reference_calls": development_calls,
            "trajectory_manifest": specs,
        }
        write_json(out/"calibration.json", calibration)
        print(json.dumps({"calibration_frozen": True, "fixed_k": fixed_k,
                          "d_empirical": d_empirical, "development_reference_calls": development_calls}), flush=True)
        for spec in specs:
            if spec["split"] == "calibration":
                continue
            for policy in POLICIES:
                coefficient = d_empirical if policy == "empirical_horizon" else d_class
                results.append(simulate(spec, policy, coefficient, fixed_k if policy == "fixed_streak" else None,
                                        cfg, step_writer, ref_writer, failure_writer))

    grouped = {
        split: {policy: aggregate([r for r in results if r["split"] == split and r["policy"] == policy], cfg)
                for policy in POLICIES}
        for split in ("heldout", "outside_class")
    }
    horizon = grouped["heldout"]["class_horizon"]
    fixed = grouped["heldout"]["fixed_streak"]
    cost_ratio = horizon["n_reference"]/fixed["n_reference"]
    reduction = (1.0-horizon["interval_violation_rate"]/fixed["interval_violation_rate"]
                 if fixed["n_interval_violations"] else None)
    gates = {
        "reference_cost_within_20_percent": abs(cost_ratio-1.0) <= cfg["comparable_label_cost_relative_tolerance"],
        "at_least_half_intervals_unreferenced": horizon["unreferenced_fraction"] >= cfg["minimum_unreferenced_fraction"],
        "no_class_envelope_counterexample": horizon["n_envelope_counterexamples"] == 0,
        "baseline_has_violations": fixed["n_interval_violations"] > 0,
        "at_least_half_violation_rate_reduction": reduction is not None and reduction >= cfg["minimum_relative_violation_reduction"],
    }
    paired = []
    for seed in cfg["heldout"]["seeds"]:
        selected = {r["policy"]: r for r in results if r["split"] == "heldout" and r["seed"] == seed}
        h, f = selected["class_horizon"], selected["fixed_streak"]
        paired.append({"seed": seed, "amplitude": h["amplitude"],
                       "reference_difference_horizon_minus_fixed": h["n_reference"]-f["n_reference"],
                       "interval_violation_rate_difference_horizon_minus_fixed": h["interval_violation_rate"]-f["interval_violation_rate"]})
    summary = {
        **metadata, "config": cfg, "invariants": checks, "calibration": calibration,
        "results": grouped, "primary_comparison": {
            "reference_cost_ratio_horizon_over_fixed": cost_ratio,
            "relative_interval_violation_reduction": reduction,
            "criteria": gates, "positive_cost_matched_result": all(gates.values()),
            "independent_trajectory_pairs": paired,
        },
        "cost_accounting": {
            "shared_development_reference_calls": development_calls,
            "heldout_reference_calls_plus_shared_development": {
                policy: grouped["heldout"][policy]["n_reference"]+development_calls for policy in POLICIES},
            "total_actual_controller_and_calibration_reference_calls_all_runs": sum(r["n_reference"] for r in results)+len(positions),
            "total_measurement_only_force_calls_all_runs": sum(r["n_measurement_force_calls"] for r in results),
            "measurement_only_exact_reference_trajectory_points": sum(r["n_steps"] for r in results),
            "note": "Measurement work is excluded only from the simulated label budget, never from wall time.",
        },
        "interpretation_limits": [
            "Class envelope is an analytical construction using a declared stiffness bound, not empirical discovery.",
            "The secant coefficient is an empirical derivative estimate, not a learned force model or confidence bound.",
            "All policy trajectories are generated forward; there is no decision replay.",
            "Random-check rates concern eligible candidates at interval starts; primary violations concern whole unreferenced intervals.",
            "Correlated intervals and streaks are not independent samples; there are 16 held-out trajectory pairs.",
            "A local force envelope does not guarantee trajectory fidelity or energy conservation.",
            "Reference-call reduction for a cheap exact force is not a measured wall-clock speedup.",
            "No novelty claim is made for martingale or active anytime-valid risk principles.",
        ],
        "wall_time_seconds_before_final_serialization": time.perf_counter()-start,
    }
    write_json(out/"trajectories.json", results)
    scalar_fields = [key for key in results[0] if key != "first_counterexample"]
    with (out/"trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    write_json(out/"summary.json", summary)
    manifest = {name: {"sha256": sha256(out/name), "bytes": (out/name).stat().st_size}
                for name in products}
    write_json(out/"manifest.json", {"files": manifest, "wall_time_seconds": time.perf_counter()-start})
    print(json.dumps({"primary_comparison": summary["primary_comparison"],
                      "wall_time_seconds": time.perf_counter()-start}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
