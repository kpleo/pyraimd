"""Check bundled records using NumPy and the standard library; print JSON.

No trajectory generation, random sampling, model loading, fitting, or DFT.
The checker does not create or modify files.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECKS = 0


def require(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(label)


def close(actual, expected, label, atol=1e-12):
    require(np.allclose(actual, expected, atol=atol, rtol=0), label)


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def archive(path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def norm(force):
    return np.linalg.norm(force, axis=-1).max(axis=-1)


def verlet(x, p, force, masses, dt):
    dx = x[:, 1:] - x[:, :-1] - dt * (p[:, :-1] + .5 * dt * force[:, :-1]) / masses[None, None, :, None]
    dp = p[:, 1:] - p[:, :-1] - .5 * dt * (force[:, 1:] + force[:, :-1])
    close(dx, 0, "velocity-Verlet positions")
    close(dp, 0, "velocity-Verlet full-step momenta")
    return {"maximum_position_defect_A": float(abs(dx).max()),
            "maximum_momentum_defect_sqrt_u_eV": float(abs(dp).max())}


def inventory(data):
    schema = read_json(data / "schema.json")
    n_npz = n_csv = 0
    for path in sorted(data.rglob("*.npz")):
        name = path.relative_to(data).as_posix()
        values = archive(path)
        require(name in schema, "missing NPZ schema: " + name)
        require(set(values) == set(schema[name]["arrays"]), "array names: " + name)
        for key, a in values.items():
            require(a.dtype.kind in "biuf" and np.isfinite(a).all(), name + ": finite numeric " + key)
            s = schema[name]["arrays"][key]
            require(list(a.shape) == s["shape"] and str(a.dtype) == s["dtype"], name + ": shape/dtype " + key)
        n_npz += 1
    for path in sorted(data.rglob("*.csv")):
        name = path.relative_to(data).as_posix()
        rr = rows(path)
        require(name in schema, "missing CSV schema: " + name)
        expected = schema[name]
        require(list(rr[0]) == expected["columns"], "CSV columns: " + name)
        count = expected.get("rows", expected.get("row_count"))
        if count is not None:
            require(len(rr) == count, "CSV row count: " + name)
        n_csv += 1
    return {"npz_archives": n_npz, "csv_tables": n_csv}


def velocity(data):
    folder = data / "water_velocity"
    a = archive(folder / "trajectories.npz")
    settings = read_json(folder / "settings.json")
    x, p = a["positions_A"], a["momenta_sqrt_u_eV"]
    residual = a["surrogate_forces_eV_A"] - a["reference_forces_eV_A"]
    error = norm(residual)
    close(error, a["recorded_force_error_eV_A"], "velocity maximum force errors")
    conversion = settings["conversions"]["ase_fs_in_internal_time"]
    close(a["velocities_A_fs"], p / a["masses_u"][None, None, :, None] * conversion, "velocity conversion")
    defects = verlet(x, p, a["surrogate_forces_eV_A"], a["masses_u"], settings["dt_fs"] * conversion)
    close(a["time_fs"], np.arange(20) * .5, "velocity time grid")
    length = np.concatenate((np.zeros((6, 1)), np.cumsum(
        np.linalg.norm(np.diff(x, axis=1).reshape(6, 19, -1), axis=-1), axis=1)), axis=1)
    secants = norm(np.diff(residual, axis=1)) / np.diff(length, axis=1)
    crossing, horizons = [], []
    for i, case in enumerate(rows(folder / "motion_cases.csv")):
        seed, factor = int(case["seed"]), float(case["factor"])
        require(seed == int(a["seed"][i]) and factor == a["velocity_factor"][i], "velocity path identity")
        hits = np.flatnonzero(error[i] > settings["force_budget_eV_A"])
        first = float(a["time_fs"][hits[0]]) if hits.size else None
        saved = float(case["first_observed_crossing_fs"]) if case["first_observed_crossing_fs"] else None
        require(first == saved and (not hits.size) == (case["right_censored"] == "True"), "crossing/censoring")
        crossing.append(first)
        initial = settings["recorded_initial_predictions"][i]
        forecasts = [r for r in rows(folder / "motion_forecasts.csv")
                     if int(r["seed"]) == seed and float(r["factor"]) == factor]
        safe = 0.
        for r in forecasts:
            j = int(round(float(r["time_fs"]) / settings["dt_fs"]))
            close(length[i, j], float(r["saved_forecast_length_A"]), "saved versus actual forecast length")
            bound = initial["e0_eV_A"] + initial["kappa_eV_A2"] * length[i, j]
            close(bound, float(r["saved_forecast_bound_ev_A"]), "forecast bound arithmetic")
            if bound <= settings["force_budget_eV_A"]:
                safe = float(r["time_fs"])
        close(safe, initial["horizon_fs"], "forecast horizon")
        horizons.append(safe)
        close(secants[i].max(), float(case["max_observed_vector_secant_ev_A2"]), "path residual secants")
        require(np.all(error[i] <= initial["e0_eV_A"] + initial["kappa_eV_A2"] * length[i] + 1e-12), "observed empirical envelope")
        ref = 1 if i < 3 else 4
        close(x[i, 0], x[ref, 0], "shared velocity-intervention structure")
        close(p[i, 0], factor * p[ref, 0], "paired velocity factor")
    probes = archive(folder / "probes.npz")
    q = probes["recorded_residual_derivative_eV_A2"]
    magnitudes = norm(q)
    close(np.linalg.norm(probes["unit_directions"], axis=(1, 2)), 1, "probe unit directions")
    for i, r in enumerate(rows(folder / "motion_directional_probes.csv")):
        close(magnitudes.reshape(-1)[i], float(r["derivative_norm_ev_A2"]), "retained derivative norms")
    for r in rows(folder / "probe_mapping.csv"):
        i, j, k = [int(r[key]) for key in ("anchor_index", "scale_index", "sign_index")]
        intended = probes["origin_positions_A"][i] + probes["sign"][k] * probes["probe_steps_A"][j] * probes["unit_directions"][i]
        close(intended, probes["reconstructed_positions_A"][i, j, k], "reconstructed probe geometry")
        text = str(a["atomic_numbers"].tolist()) + "|" + ";".join(
            f"{xx:.10f},{yy:.10f},{zz:.10f}" for xx, yy, zz in intended / settings["conversions"]["bohr_A"])
        require(hashlib.sha256(text.encode()).hexdigest() == r["geometry_sha256"], "retained probe geometry hash")
    return {"paths": 6, "states": int(error.size), "signed_reference_probes": 8,
            "horizons_fs": horizons, "first_observed_crossing_fs": crossing,
            "derivative_norms_eV_A2": magnitudes.tolist(),
            "two_spacing_relative_vector_difference_percent": (100 * norm(q[:, 0] - q[:, 1]) / magnitudes[:, 0]).tolist(),
            "path_secant_range_eV_A2": [float(secants.max(axis=1).min()), float(secants.max())], **defects}


def molecular(data, name):
    folder = data / name
    a = archive(folder / "trajectories.npz")
    settings = read_json(folder / "settings.json")
    paths, metrics = rows(folder / "paths.csv"), rows(folder / "metrics.csv")
    x, p = a["positions_A"], a["momenta_sqrt_u_eV"]
    force, fref = a["driving_forces_eV_A"], a["reference_forces_eV_A"]
    eps, valid = settings["epsilon_ev_A"], a["surrogate_available"]
    error = norm(a["surrogate_forces_eV_A"] - fref)
    close(error[valid], a["recorded_force_error_eV_A"][valid], name + ": force-error reconstruction")
    require(np.array_equal(valid, a["proposal_available"]), name + ": proposal availability")
    close(a["surrogate_forces_eV_A"][valid], a["proposal_forces_eV_A"][valid], name + ": frozen proposal retention")
    require(np.array_equal(a["accepted"], a["proposal_accepted"]), name + ": admission retention")
    require(not np.any(a["accepted"] & ~valid), name + ": acceptance without proposal")
    require(not np.any(a["checked"] & ~a["accepted"]), name + ": checks only after acceptance")
    require(np.array_equal(a["reference_requested"], ~a["accepted"] | a["checked"]), name + ": request accounting")
    require(np.array_equal(a["accepted_violation"], a["accepted"] & (error > eps)), name + ": violation flags")
    close(force, np.where(a["accepted"][:, :, None, None], a["surrogate_forces_eV_A"], fref), name + ": applied force")
    conversion = settings["conversions"]["ase_fs_in_internal_time"]
    defects = verlet(x, p, force, a["masses_u"], settings["dt_fs"] * conversion)
    kinetic = .5 * np.sum(p * p / a["masses_u"][None, None, :, None], axis=(2, 3))
    close(kinetic, a["kinetic_energy_eV"], name + ": kinetic energy")
    href = kinetic + a["reference_energy_eV"]
    delta = href - href[:, :1]
    power = np.sum((p / a["masses_u"][None, None, :, None]) * (force - fref), axis=(2, 3))
    work = np.concatenate((np.zeros((len(paths), 1)), np.cumsum(
        .5 * settings["dt_fs"] * conversion * (power[:, 1:] + power[:, :-1]), axis=1)), axis=1)
    oxygen, = np.flatnonzero(a["atomic_numbers"] == 8)
    hydrogen = np.flatnonzero(a["atomic_numbers"] == 1)
    displacements = x[:, :, hydrogen] - x[:, :, [oxygen]]
    bonds = np.linalg.norm(displacements, axis=-1)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(displacements[:, :, 0] * displacements[:, :, 1], axis=-1) /
                                         (bonds[:, :, 0] * bonds[:, :, 1]), -1, 1)))
    counts, screens = [], defaultdict(list)
    for i, (path, saved) in enumerate(zip(paths, metrics)):
        require(int(path["path_index"]) == int(saved["path_index"]) == i, name + ": path mapping")
        n, v = int(a["accepted"][i].sum()), int(a["accepted_violation"][i].sum())
        d = int((a["checked"][i] & a["accepted_violation"][i]).sum())
        requests, checks = int(a["reference_requested"][i].sum()), int(a["checked"][i].sum())
        target = next(j for j, other in enumerate(paths) if other["policy"] == "reference" and
                      other["seed"] == path["seed"] and other["kinetic_temperature_K"] == path["kinetic_temperature_K"])
        close(x[i, 0], x[target, 0], name + ": paired initial positions")
        close(p[i, 0], p[target, 0], name + ": paired initial momenta")
        values = {"states": x.shape[1], "accepted": n, "violations": v, "audit_detections": d,
                  "online_requests": requests, "audit_requests": checks,
                  "new_reference_attempts_completed": int(a["new_reference"][i].sum()),
                  "max_Href_drift_meV": float(abs(delta[i]).max() * 1000),
                  "end_Href_drift_meV": float(delta[i, -1] * 1000),
                  "max_work_balance_residual_meV": float(abs(delta[i] - work[i]).max() * 1000),
                  "paired_geometry_states": x.shape[1],
                  "OH_bond_RMSE_A": float(np.sqrt(np.mean((bonds[i] - bonds[target]) ** 2))),
                  "HOH_angle_RMSE_degrees": float(np.sqrt(np.mean((angles[i] - angles[target]) ** 2))),
                  "max_OH_bond_difference_A": float(np.max(abs(bonds[i] - bonds[target]))),
                  "max_HOH_angle_difference_degrees": float(np.max(abs(angles[i] - angles[target])))}
        for key, value in values.items():
            close(value, float(saved[key]), name + ": retained metric " + key, atol=1e-9 if "meV" in key else 1e-12)
        if n:
            close(v / n, float(saved["risk"]), name + ": accepted risk")
            bound = min(1., (settings["confidence_lambda"] * d + math.log(1 / settings["confidence_error_probability"])) /
                        (n * -math.log(1 - settings["audit_p"] + settings["audit_p"] * math.exp(-settings["confidence_lambda"]))))
            close(bound, float(saved["sequential_upper"]), name + ": sequential bound")
        else:
            require(saved["risk"] == saved["sequential_upper"] == "", name + ": undefined zero-acceptance values")
        counts.append({"path_index": i, "policy": path["policy"], "seed": int(path["seed"]),
                       "kinetic_temperature_K": int(path["kinetic_temperature_K"]),
                       "accepted": n, "violations": v, "reference_requests": requests})
        screens[(path["policy"], path["kinetic_temperature_K"])].append(
            n >= .2 * x.shape[1] and (v / n <= .05 if n else False) and requests <= .9 * x.shape[1])
    by_anchor = defaultdict(list)
    for r in rows(folder / "forecasts.csv"):
        by_anchor[int(r["anchor_index"])].append(r)
    for r in rows(folder / "anchors.csv"):
        safe = 0.
        for f in by_anchor[int(r["anchor_index"])]:
            bound = float(r["e0_eV_A"]) + float(r["kappa_eV_A2"]) * float(f["path_length_A"])
            close(bound, float(f["bound_eV_A"]), name + ": anchor forecast arithmetic")
            if bound <= eps:
                safe = max(safe, float(f["elapsed_fs"]))
        close(safe, float(r["horizon_fs"]), name + ": anchor horizon inversion")
    reference_drifts = [float(abs(delta[i]).max() * 1000) for i, r in enumerate(paths) if r["policy"] == "reference"]
    return {"paths": len(paths), "states": int(x.shape[0] * x.shape[1]),
            "new_reference_evaluations": int(a["new_reference"].sum()),
            "zero_acceptance_paths": int(np.sum(~a["accepted"].any(axis=1))),
            "reference_drift_range_meV": [min(reference_drifts), max(reference_drifts)],
            "path_counts": counts,
            "screening_by_policy_and_temperature": {policy + "_" + level: all(v) for (policy, level), v in screens.items()},
            **defects}


def binomial_upper_tail(k, n, p):
    logs = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) +
            j * math.log(p) + (n - j) * math.log1p(-p) for j in range(k, n + 1)]
    maximum = max(logs)
    return math.exp(maximum) * math.fsum(math.exp(x - maximum) for x in logs)


def controlled(data):
    folder = data / "controlled"
    trajectories = rows(folder / "oscillator_trajectories.csv")
    work = rows(folder / "oscillator_trajectory_work.csv")
    intervals, segments = archive(folder / "oscillator_intervals.npz"), archive(folder / "oscillator_segments.npz")
    settings = read_json(folder / "settings.json")
    free_code = next(int(k) for k, v in settings["route_codes"].items() if v == "unreferenced")
    aggregate = defaultdict(lambda: {"references": 0, "unreferenced_intervals": 0, "violations": 0})
    by_key = {(r["split"], r["policy"], r["seed"]): r for r in work}
    for i, tr in enumerate(trajectories):
        pick = intervals["trajectory_index"] == i
        free = pick & (intervals["route_code"] == free_code)
        violations = free & (intervals["interval_max_error"] > .05)
        require(int(pick.sum()) == int(tr["n_steps"]), "oscillator interval count")
        require(int(free.sum()) == int(tr["n_unreferenced"]) and int(violations.sum()) == int(tr["n_interval_violations"]), "oscillator interval outcomes")
        group = aggregate[tr["split"] + "/" + tr["policy"]]
        group["references"] += int((pick & ~free).sum())
        group["unreferenced_intervals"] += int(free.sum())
        group["violations"] += int(violations.sum())
        seg = segments["trajectory_index"] == i
        dx = segments["x_end"][seg] - segments["x_start"][seg]
        anchor = segments["residual_at_segment_start"][seg] * dx
        curvature = .5 * (float(tr["stiffness"]) - 1) * dx * dx
        close(anchor, segments["anchor_residual_work"][seg], "oscillator anchor work")
        close(curvature, segments["curvature_growth_work"][seg], "oscillator curvature work")
        close(anchor + curvature, segments["work"][seg], "oscillator segment identity")
        wr = by_key[(tr["split"], tr["policy"], tr["seed"])]
        close((anchor + curvature).sum(), float(wr["W"]), "oscillator accumulated work")
        close(float(wr["W"]), float(wr["delta_H"]), "oscillator work-energy identity")
    bars = rows(folder / "oscillator_work_components.csv")
    for r in bars:
        i = next(i for i, tr in enumerate(trajectories) if tr["split"] == "heldout" and tr["policy"] == "class_horizon" and tr["seed"] == r["seed"])
        sel = segments["trajectory_index"] == i
        h0 = float(by_key[("heldout", "class_horizon", r["seed"])]["H0"])
        close(segments["anchor_residual_work"][sel].sum() / h0, float(r["anchor_work_over_H0"]), "Fig3 anchor contribution")
        close(segments["curvature_growth_work"][sel].sum() / h0, float(r["curvature_work_over_H0"]), "Fig3 curvature contribution")
    rep = archive(folder / "verification_replicates.npz")
    summary = read_json(folder / "verification_summary.json")
    n, events, zero = len(rep["replicate"]), int(rep["ever_crossed"].sum()), int(rep["ever_zero_event"].sum())
    require(events == summary["primary_ever_log_M_crossing"]["events"], "finite-lambda event count")
    require(zero == summary["zero_detection"]["events"], "zero-detection event count")
    require(np.array_equal(rep["ever_crossed"], rep["ever_capped_bound_failed"]), "equivalent saved primary events")
    traces = rows(folder / "verification_traces.csv")
    for i in range(4):
        rr = [r for r in traces if int(r["replicate"]) == i]
        aa, yy, zz = [np.array([int(r[k]) for r in rr]) for k in ["A", "Y", "Z"]]
        nn, vv, dd = np.cumsum(aa), np.cumsum(aa * yy), np.cumsum(aa * yy * zz)
        for k, values in zip(["N", "V", "D"], [nn, vv, dd]):
            close(values, [int(r[k]) for r in rr], "trace cumulative " + k)
            require(int(values[-1]) == int(rep[k][i]), "trace terminal outcome " + k)
        close(-summary["lambda"] * dd - np.log(.95) * vv, [float(r["log_M"]) for r in rr], "trace martingale arithmetic")
    B = .9 ** 29
    with localcontext() as ctx:
        ctx.prec = 40
        gap = (Decimal(9) / 10) ** 29 * Decimal(sum(math.comb(911, j) * 98 ** j * 27 ** (911-j) for j in range(29))) / Decimal(125 ** 911)
        recorded = Decimal(summary["deterministic_probability"]["analytic_finite_horizon_bracket"]["finite_horizon_gap_upper_decimal_approx"])
        require(abs(gap / recorded - 1) < Decimal("1e-35"), "analytical finite-horizon gap arithmetic")
    return {"oscillator_trajectories": len(trajectories), "oscillator_intervals": int(intervals["step"].size),
            "oscillator_segments": int(segments["work"].size), "oscillator_counts": dict(aggregate),
            "heldout_mean_anchor_work_over_H0": float(np.mean([float(r["anchor_work_over_H0"]) for r in bars])),
            "heldout_mean_curvature_work_over_H0": float(np.mean([float(r["curvature_work_over_H0"]) for r in bars])),
            "verification_repetitions": n, "primary_events": events, "zero_detection_events": zero,
            "zero_detection_comparator": B, "posthoc_binomial_upper_tail": binomial_upper_tail(zero, n, B),
            "analytical_gap_upper": str(gap), "zero_detection_example_U0": math.log(20) / -math.log(.95)}


def tungsten(data):
    folder = data / "tungsten"
    g, p, r = [archive(folder / name) for name in ("geometries.npz", "base_predictions.npz", "reference_labels.npz")]
    lookup = {int(v): i for i, v in enumerate(r["record_index"])}
    ref = r["reference_forces_eV_A"][[lookup[int(v)] for v in g["representative_record_index"]]]
    residual = p["mean_forces_eV_A"] - ref
    close(p["mean_forces_eV_A"], p["member_forces_eV_A"].mean(axis=1), "W committee mean")
    close(p["spread_eV_A"], np.linalg.norm(p["member_forces_eV_A"][:, 0] - p["member_forces_eV_A"][:, 1], axis=-1) / 2, "W population vector spread")
    d3 = archive(folder / "d3_pair.npz")
    close(d3["reference_PBE_without_D3_forces_eV_A"], d3["reference_PBE_D3_forces_eV_A"] - d3["d3_forces_eV_A"], "D3 subtraction sign")
    identity = read_json(folder / "model_identity.json")
    for member in identity["members"]:
        for k in ("model_file_sha256", "effective_evaluation_tensor_sha256", "stock_tensor_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", member[k]) is not None, "W hash syntax")
    require(identity["members"][0]["effective_evaluation_tensor_sha256"] == identity["members"][0]["stock_tensor_sha256"], "W unchanged first member identity")
    require(identity["members"][1]["effective_evaluation_tensor_sha256"] != identity["members"][1]["stock_tensor_sha256"], "W perturbed second member identity")
    return {"geometries": len(g["geometry_index"]), "reference_records": len(r["record_index"]),
            "pooled_vector_RMS_eV_A": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=-1)))),
            "maximum_residual_eV_A": float(norm(residual).max()), "model_identity_records": len(identity["members"])}


def compare_expected(actual, expected, prefix=""):
    for key, target in expected.items():
        value = actual[key]
        if isinstance(target, dict):
            compare_expected(value, target, prefix + key + ".")
        elif isinstance(target, list) or target is None or isinstance(target, (int, str, bool)):
            require(value == target, "expected " + prefix + key)
        else:
            close(value, target, "expected " + prefix + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    args = parser.parse_args()
    try:
        result = {"version": (HERE / "VERSION").read_text().strip(), "status": "passed",
                  "inventory": inventory(args.data), "water_velocity": velocity(args.data),
                  "molecular_comparison": molecular(args.data, "molecular_comparison"),
                  "hot_forward": molecular(args.data, "hot_forward"), "controlled": controlled(args.data),
                  "tungsten": tungsten(args.data)}
        compare_expected(result, read_json(HERE / "expected_results.json")["supplementary_records"])
        result["arithmetic_and_inventory_checks"] = CHECKS
        result["model_evaluations"] = result["new_trajectories"] = result["new_reference_evaluations"] = 0
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, f"Record check failed: {error}\n")


if __name__ == "__main__":
    main()
