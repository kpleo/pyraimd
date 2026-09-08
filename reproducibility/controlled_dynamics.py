"""Controlled harmonic dynamics using only the Python standard library.

Usage: python controlled_dynamics.py --output results/controlled.json

Unit mass, reference force -k*x, approximate force -x+b; b is constant
between reference calls. Each call corrects 80% of the force residual,
with full correction if the remaining error reaches the force tolerance.
The horizon is (tolerance - initial_error)/(coefficient * speed_bound),
where speed_bound = hypot(x-b, v). Eligibility covers the next whole
interval; one paired random draw is consumed at every physical step.

Calibration uses eight class-horizon trajectories and two force probes.
The fixed streak minimizes the absolute difference between the calibration
call fraction and p/(1-(1-p)**(K+1)), breaking ties toward smaller K.
Sixteen heldout and eight outside-class initializations use all three policies.

Signed residual work between reference anchors a and z is
W = ((k-1)*x_a+b)*(x_z-x_a) + (k-1)*(x_z-x_a)**2/2.
The final segment ends at the final state. Work includes all intervals.
Group means weight trajectories equally after division by their own H0.
Energy extrema are measured at step endpoints; force maxima include all
interior turning points. Violation counts concern unreferenced intervals.
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


CFG = {
    "dt": 0.01, "n_steps": 2000, "mass": 1.0, "approximate_stiffness": 1.0,
    "force_tolerance": 0.05, "bias_correction_fraction": 0.8,
    "class_stiffness_mismatch_bound": 0.2,
    "random_reference_probability": 0.02, "random_seed_offset": 10000000,
    "floating_point_tolerance": 1e-12,
    "empirical_calibration_stiffness": 1.1,
    "empirical_calibration_positions": [-1.0, 1.0],
    "fixed_streak_candidates": [1, 200],
    "calibration": {
        "stiffness": 1.1, "seeds": list(range(2026090500, 2026090508)),
        "amplitudes": [0.6, 0.8, 1.2, 1.6] * 2,
    },
    "heldout": {
        "stiffness": 1.18, "seeds": list(range(2026090600, 2026090616)),
        "amplitudes": [0.65, 0.9, 1.25, 1.55] * 4,
    },
    "outside_class": {
        "stiffness": 1.4, "seeds": list(range(2026090700, 2026090708)),
        "amplitudes": [0.65, 0.9, 1.25, 1.55] * 2,
    },
}
POLICIES = ("class_horizon", "empirical_horizon", "fixed_streak")


def reference_force(x, stiffness):
    return -stiffness * x


def advance(x, v, bias, dt):
    """Exact flow of x'' = -x+bias."""
    c, s = math.cos(dt), math.sin(dt)
    q = x - bias
    return bias + q*c + v*s, -q*s + v*c


def interval_max_error(x, v, bias, stiffness, dt, x_end):
    positions = [x, x_end]
    phase = math.atan2(v, x-bias)
    first, last = math.ceil(-phase/math.pi), math.floor((dt-phase)/math.pi)
    for index in range(first, last+1):
        tau = phase + index*math.pi
        if 0.0 < tau < dt:
            positions.append(advance(x, v, bias, tau)[0])
    errors = [abs(reference_force(q, stiffness)-(-q+bias)) for q in positions]
    return max(errors), len(positions)


def segment_work(x_anchor, x_end, bias, stiffness):
    dx, mismatch = x_end-x_anchor, stiffness-1.0
    total = 0.5*mismatch*(x_end*x_end-x_anchor*x_anchor) + bias*dx
    residual = (mismatch*x_anchor+bias)*dx
    curvature = 0.5*mismatch*dx*dx
    return total, residual, curvature


def trajectory_specs(split):
    group = CFG[split]
    for seed, amplitude in zip(group["seeds"], group["amplitudes"]):
        phase = random.Random(seed).random()*2.0*math.pi
        stiffness = group["stiffness"]
        yield dict(split=split, seed=seed, amplitude=amplitude, stiffness=stiffness,
                   x0=amplitude*math.cos(phase),
                   v0=-amplitude*math.sqrt(stiffness)*math.sin(phase))


def simulate(spec, policy, coefficient, fixed_k=None):
    dt, n = CFG["dt"], CFG["n_steps"]
    epsilon, tolerance = CFG["force_tolerance"], CFG["floating_point_tolerance"]
    rng = random.Random(spec["seed"]+CFG["random_seed_offset"])
    x, v, bias, stiffness = spec["x0"], spec["v0"], 0.0, spec["stiffness"]
    h0 = 0.5*(v*v+stiffness*x*x)
    anchor_step = None
    streak = maximum_streak = 0
    counts = dict.fromkeys((
        "n_reference", "n_initial_reference", "n_forced_reference", "n_random_reference",
        "n_unreferenced", "n_start_violations", "n_interval_violations",
        "n_envelope_counterexamples", "n_eligible", "n_eligible_start_violations",
        "n_random_start_violations", "n_full_corrections", "n_measurement_force_calls",
    ), 0)
    maximum_energy = maximum_force = maximum_excess = 0.0
    position_squared = velocity_squared = 0.0
    segments, initial_errors, horizons = [], [], []
    first_counterexample = None
    for step in range(n):
        if anchor_step is None:
            eligible = False
        elif policy == "fixed_streak":
            eligible = streak < fixed_k
        else:
            eligible = (step-anchor_step+1)*dt <= trust_time
        draw = rng.random()
        random_check = eligible and draw < CFG["random_reference_probability"]
        route = ("initial_reference" if anchor_step is None else
                 "forced_reference" if not eligible else
                 "random_reference" if random_check else "unreferenced")
        bias_before = bias
        if route != "unreferenced":
            if anchor_step is not None:
                segments.append(segment_work(anchor_x, x, bias, stiffness))
            observed = reference_force(x, stiffness)
            counts["n_reference"] += 1
            counts["n_"+route] += 1
            bias += CFG["bias_correction_fraction"]*(observed-(-x+bias))
            initial_error = abs(observed-(-x+bias))
            if initial_error >= epsilon:
                bias = observed+x
                initial_error = abs(observed-(-x+bias))
                counts["n_full_corrections"] += 1
            drift_rate = coefficient*math.hypot(x-bias, v)
            trust_time = max(0.0, epsilon-initial_error)/drift_rate
            anchor_step, anchor_x, streak = step, x, 0
            initial_errors.append(initial_error)
            horizons.append(trust_time)
        else:
            counts["n_unreferenced"] += 1
            streak += 1
            maximum_streak = max(maximum_streak, streak)

        x_end, v_end = advance(x, v, bias, dt)
        # These measurements do not enter eligibility, correction, or calibration.
        error_before = abs(reference_force(x, stiffness)-(-x+bias_before))
        peak, calls = interval_max_error(x, v, bias, stiffness, dt, x_end)
        counts["n_measurement_force_calls"] += calls+1
        start_violation, violation = error_before > epsilon+tolerance, peak > epsilon+tolerance
        envelope = initial_error+drift_rate*((step-anchor_step+1)*dt)
        counterexample = peak > envelope+tolerance
        if eligible:
            counts["n_eligible"] += 1
            counts["n_eligible_start_violations"] += int(start_violation)
        if random_check:
            counts["n_random_start_violations"] += int(start_violation)
        if route == "unreferenced":
            counts["n_start_violations"] += int(start_violation)
            counts["n_interval_violations"] += int(violation)
            counts["n_envelope_counterexamples"] += int(counterexample)
            maximum_force, maximum_excess = max(maximum_force, peak), max(maximum_excess, peak-envelope)
            if first_counterexample is None and (violation or counterexample):
                first_counterexample = dict(
                    seed=spec["seed"], step=step, x=x, v=v, bias=bias,
                    anchor_step=anchor_step, interval_max_error=peak, envelope=envelope,
                    interval_violation=violation, envelope_counterexample=counterexample)
        omega, time_end = math.sqrt(stiffness), (step+1)*dt
        c, s = math.cos(omega*time_end), math.sin(omega*time_end)
        true_x = spec["x0"]*c+spec["v0"]/omega*s
        true_v = -spec["x0"]*omega*s+spec["v0"]*c
        position_squared += (x_end-true_x)**2
        velocity_squared += (v_end-true_v)**2
        energy_change = (0.5*(v_end*v_end+stiffness*x_end*x_end)-h0)/h0
        maximum_energy = max(maximum_energy, abs(energy_change))
        x, v = x_end, v_end

    segments.append(segment_work(anchor_x, x, bias, stiffness))
    work = dict(zip(("work", "anchor_residual_work", "curvature_work"),
                    (math.fsum(values) for values in zip(*segments))))
    assert counts["n_reference"]+counts["n_unreferenced"] == n
    assert counts["n_eligible"] == counts["n_random_reference"]+counts["n_unreferenced"]
    return {
        **spec, "policy": policy, "drift_coefficient": coefficient, "fixed_k": fixed_k,
        "n_steps": n, **counts, "n_work_segments": len(segments), "H0": h0,
        "maximum_unreferenced_streak_time": maximum_streak*dt,
        "maximum_unreferenced_force_error": maximum_force,
        "maximum_positive_envelope_excess": maximum_excess,
        "initial_error_min": min(initial_errors), "initial_error_max": max(initial_errors),
        "trust_time_min": min(horizons), "trust_time_median": statistics.median(horizons),
        "trust_time_max": max(horizons),
        "squared_position_error_sum": position_squared, "squared_velocity_error_sum": velocity_squared,
        "final_relative_energy_change": energy_change,
        "maximum_absolute_relative_energy_change": maximum_energy,
        **work, **{key+"_over_H0": value/h0 for key, value in work.items()},
        "absolute_work_energy_defect_over_H0": abs(work["work"]/h0-energy_change),
        "maximum_segment_work_decomposition_defect": max(abs(w-a-c) for w, a, c in segments),
        "first_counterexample": first_counterexample,
    }


def aggregate(rows):
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key.startswith("n_")}
    n, free = totals["n_steps"], totals["n_unreferenced"]
    return {
        "n_trajectories": len(rows), **totals, "physical_duration": n*CFG["dt"],
        "reference_fraction": totals["n_reference"]/n, "unreferenced_fraction": free/n,
        "unreferenced_physical_duration": free*CFG["dt"],
        "interval_violation_rate": totals["n_interval_violations"]/free,
        "start_violation_rate": totals["n_start_violations"]/free,
        "trajectories_with_interval_violation": sum(r["n_interval_violations"] > 0 for r in rows),
        "trajectories_with_envelope_counterexample": sum(r["n_envelope_counterexamples"] > 0 for r in rows),
        "position_rmse": math.sqrt(sum(r["squared_position_error_sum"] for r in rows)/n),
        "velocity_rmse": math.sqrt(sum(r["squared_velocity_error_sum"] for r in rows)/n),
        "mean_final_relative_energy_change": statistics.mean(r["final_relative_energy_change"] for r in rows),
        **{key: max(r[key] for r in rows) for key in rows[0] if key.startswith("maximum_")},
        **{"mean_"+key: statistics.mean(r[key] for r in rows)
           for key in ("work_over_H0", "anchor_residual_work_over_H0", "curvature_work_over_H0")},
        "maximum_absolute_work_energy_defect_over_H0": max(r["absolute_work_energy_defect_over_H0"] for r in rows),
        "first_counterexample": next((r["first_counterexample"] for r in rows if r["first_counterexample"]), None),
    }


def experiment():
    d_class = CFG["class_stiffness_mismatch_bound"]
    rows = [simulate(spec, "class_horizon", d_class) for spec in trajectory_specs("calibration")]
    calibration = aggregate(rows)
    positions = CFG["empirical_calibration_positions"]
    forces = [reference_force(x, CFG["empirical_calibration_stiffness"]) for x in positions]
    residuals = [force+x for force, x in zip(forces, positions)]
    d_empirical = abs((residuals[1]-residuals[0])/(positions[1]-positions[0]))
    p = CFG["random_reference_probability"]
    low, high = CFG["fixed_streak_candidates"]
    rates = {k: p/(1.0-(1.0-p)**(k+1)) for k in range(low, high+1)}
    fixed_k = min(rates, key=lambda k: (abs(rates[k]-calibration["reference_fraction"]), k))
    groups = {"calibration": {"class_horizon": calibration}}
    for split in ("heldout", "outside_class"):
        groups[split] = {}
        for policy in POLICIES:
            coefficient = d_empirical if policy == "empirical_horizon" else d_class
            group = [simulate(spec, policy, coefficient, fixed_k if policy == "fixed_streak" else None)
                     for spec in trajectory_specs(split)]
            rows.extend(group)
            groups[split][policy] = aggregate(group)
    calibration_calls = calibration["n_reference"]+len(positions)
    return {
        "parameters": CFG,
        "calibration": {
            "fixed_k": fixed_k, "fixed_expected_reference_fraction": rates[fixed_k],
            "fixed_reference_spacing_without_random": (fixed_k+1)*CFG["dt"],
            "class_drift_coefficient": d_class, "empirical_drift_coefficient": d_empirical,
            "empirical_calibration_forces": forces, "empirical_coefficient_reference_calls": len(positions),
            "shared_calibration_reference_calls": calibration_calls,
        },
        "results": groups, "trajectories": rows,
        "cost_accounting": {
            "total_controller_and_calibration_reference_calls": sum(r["n_reference"] for r in rows)+len(positions),
            "total_measurement_force_calls": sum(r["n_measurement_force_calls"] for r in rows),
            "total_exact_reference_trajectory_points": sum(r["n_steps"] for r in rows),
            "heldout_reference_calls_plus_shared_calibration": {
                policy: groups["heldout"][policy]["n_reference"]+calibration_calls for policy in POLICIES},
        },
    }


def main():
    parser = argparse.ArgumentParser(prog="controlled_dynamics.py", description=__doc__)
    parser.add_argument("--output", default="results/controlled.json", metavar="FILE")
    args = parser.parse_args()
    if not args.output.strip() or "\0" in args.output:
        parser.error("output must be a valid file name")
    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            parser.error("output file already exists")
        result = experiment()
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError:
        parser.error("cannot write output file")


if __name__ == "__main__":
    main()
