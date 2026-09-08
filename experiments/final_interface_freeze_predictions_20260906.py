"""Freeze one empirical growth rule and physical predictions before future DFT.

Reads development labels and already generated surrogate-only paths. Writes no
reference protocol and never calls an engine. All output files are exclusive.
"""

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read as read_atoms
from ase.units import fs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"
EPSILON = 0.25
AUXILIARY_BUDGETS = [0.05, 0.10, 1.0]
H_CAP_FS, TRANSVERSE_FRACTION_CAP = 1.0, 0.10


def norm(a):
    return float(np.linalg.norm(a, axis=1).max())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = ["prediction_spec.json", "prospective_path_predictions.json", "future_predictions.json"]
    if any((DATA / name).exists() for name in names):
        raise FileExistsError("Never overwrite prospective predictions")
    for folder in ("future_reference_results", "check_reference_results"):
        if (DATA / folder).exists() and any((DATA / folder).iterdir()):
            raise RuntimeError("Future/check reference activity already exists")
    inputs = {}

    def remember(path):
        inputs[str(path.relative_to(DATA))] = sha(path)
        return path

    def read(name):
        return json.loads(remember(DATA / name).read_text())

    development = read("development_analysis.json")
    directions = read("directions.json")
    dev_cases = read("development_cases.json")
    dynamics = read("dynamics_protocol.json")
    cases = read("future_cases.json")
    read("future_cases_provenance.json")
    read("dynamics_numerics.json")
    read("goal_based_closure.json")
    for name, expected in development["input_sha256"].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"Changed development input: {name}")
        remember(path)
    frames = read_atoms(remember(DATA / "development_inputs.extxyz"), index=":")
    refs = {c["case"]: read(f"reference_results/case_{c['case']:02d}/result.json") for c in dev_cases}
    bases = {c["case"]: read(f"development_surrogate/case_{c['case']:02d}.json") for c in dev_cases}
    estimates, paths = [], []
    for path_spec in dynamics["paths"]:
        i = path_spec["path_index"]
        row, = [x for x in development["directions"] if x["seed"] == path_spec["seed"]]
        direction, = [x for x in directions if x["seed"] == path_spec["seed"]]
        u = np.asarray(direction["unit_direction"])
        small, large = row["scales"]
        q = np.asarray(large["residual_derivative_ev_A2"])
        qmax = norm(q)
        C = large["force_directional_curvature_ev_A2"]
        # These gates establish a resolved development signal, not derivative bounds.
        if (qmax <= 0 or C == 0
                or abs(small["max_atom_derivative_ev_A2"] - qmax) > 0.10 * qmax
                or abs(small["force_directional_curvature_ev_A2"] - C) > 0.10 * abs(C)
                or abs(large["energy_directional_curvature_ev_A2"] - C) > 0.10 * abs(C)):
            raise ValueError(f"Development consistency gate failed on path {i}")
        eta = 2 * row["derivative_scale_difference_ev_A2"]
        K = 2 * max(s["max_atom_derivative_ev_A2"] for r in development["directions"]
                    if r["anchor_step"] == path_spec["anchor_step"] for s in r["scales"])
        origin, = [c["case"] for c in dev_cases if c["kind"] == "tight_anchor"
                   and c["anchor_step"] == path_spec["anchor_step"]]
        correction = np.asarray(refs[origin]["forces_ev_a"]) - bases[origin]["forces_ev_a"]
        M, defects = 0.0, []
        for case in dev_cases:
            if case["seed"] != path_spec["seed"]:
                continue
            j = case["case"]
            h = frames[j].positions - frames[origin].positions
            R = np.asarray(bases[j]["forces_ev_a"]) + correction - refs[j]["forces_ev_a"]
            defect = norm(R - float(np.sum(u * h)) * q)
            M = max(M, 4 * defect / float(np.sum(h * h)))
            defects.append({"case": j, "linear_vector_defect_ev_a": defect})
        atoms = read_atoms(remember(DATA / path_spec["initial_state_file"]))
        speed = float(np.linalg.norm(atoms.get_velocities() * fs))
        kappas = [speed**2 * s["force_directional_curvature_ev_A2"] for s in row["scales"]]
        S = (max(0.0, min(abs(k) for k in kappas) - abs(kappas[0] - kappas[1]))
             if kappas[0] * kappas[1] > 0 else 0.0)
        estimate = {**path_spec, "q_hat_ev_a2": q.tolist(), "unit_direction": u.tolist(),
                    "q_max_ev_a2": qmax, "directional_curvature_ev_a2": C,
                    "response_participation_max_norm": large["response_participation_max_norm"],
                    "signed_inverse_curvature_a2_ev": large["signed_inverse_curvature_A2_per_ev"],
                    "boundary_2W_over_e2_a2_ev": C / qmax**2,
                    "initial_speed_a_fs": speed, "initial_error_rate_ev_a_fs": speed * qmax,
                    "time_curvature_ev_fs2": speed**2 * C,
                    "two_scale_time_curvatures_ev_fs2": kappas, "check_selection_score": S,
                    "eta_parallel_ev_a2": eta, "K_perp_empirical_ev_a2": K,
                    "M_empirical_ev_a3": M, "development_vector_defects": defects}
        estimates.append(estimate)
        complete = read(f"dynamics/path_{i}/complete.json")
        path_file = remember(DATA / f"dynamics/path_{i}/primary_states.jsonl")
        primary, = [x for x in complete["integrations"] if x["name"] == "primary"]
        if primary["states_sha256"] != sha(path_file):
            raise ValueError("Changed surrogate path")
        states = [json.loads(line) for line in path_file.read_text().splitlines()]
        x0 = np.asarray(states[0]["positions_angstrom"])
        budgets = [EPSILON, *AUXILIARY_BUDGETS]
        alive, horizons = {e: True for e in budgets}, {e: 0.0 for e in budgets}
        predictions = []
        for state in states[1:]:
            h = np.asarray(state["positions_angstrom"]) - x0
            alpha, distance = float(np.sum(u * h)), float(np.linalg.norm(h))
            transverse = float(np.linalg.norm(h - alpha * u))
            fraction = transverse / distance if distance else 0.0
            center = norm(alpha * q)
            common_radius = eta * abs(alpha) + K * transverse + 0.5 * M * distance**2
            in_domain = state["time_fs"] <= H_CAP_FS and fraction <= TRANSVERSE_FRACTION_CAP
            by_budget = []
            for e in budgets:
                bound = center + common_radius + 0.001 * EPSILON
                alive[e] = alive[e] and in_domain and bound <= e
                if alive[e]:
                    horizons[e] = state["time_fs"]
                by_budget.append({"budget_ev_a": e, "score_ev_a": bound,
                                  "prefix_admitted": alive[e]})
            predictions.append({"path_index": i, "step": state["step"], "time_fs": state["time_fs"],
                                "geometry_sha256": state["geometry_sha256"],
                                "directional_displacement_a": alpha, "displacement_norm_a": distance,
                                "transverse_displacement_a": transverse, "transverse_fraction": fraction,
                                "inside_forecast_domain": in_domain, "linear_error_prediction_ev_a": center,
                                "time_linear_error_prediction_ev_a": speed * qmax * state["time_fs"],
                                "directional_work_prediction_ev": 0.5 * C * alpha**2,
                                "time_quadratic_work_prediction_ev": 0.5 * speed**2 * C * state["time_fs"]**2,
                                "radius_linear_uncertainty_ev_a": eta * abs(alpha),
                                "radius_transverse_ev_a": K * transverse,
                                "radius_quadratic_ev_a": 0.5 * M * distance**2,
                                "by_budget": by_budget, "primary_admitted": alive[EPSILON],
                                "primary_score_ev_a": by_budget[0]["score_ev_a"]})
        paths.append({"path_index": i, "horizons_fs": {str(e): h for e, h in horizons.items()},
                      "predictions": predictions})
        nu = max(0.001 * EPSILON, row["tight_anchor_force_change_ev_A"])
        estimate["growth_sensitivity_proxy_ev_a"] = nu
        estimate["growth_diagnostic_times_fs"] = [r["time_fs"] for r in predictions
            if r["time_fs"] in [0.125, 0.25, 0.375, 0.5]
            and r["linear_error_prediction_ev_a"] >= 3 * nu]
    winners = [max((x for x in estimates if x["anchor_step"] == a),
                   key=lambda x: (x["check_selection_score"], -x["path_index"]))
               for a in sorted({x["anchor_step"] for x in estimates})]
    winners.sort(key=lambda x: (-x["check_selection_score"], x["path_index"]))
    selection = {"primary_check_path": winners[0]["path_index"], "other_anchor_check_path": winners[1]["path_index"],
                 "end_time_fs": 1.0, "scores": {str(x["path_index"]): x["check_selection_score"] for x in estimates}}
    future = []
    for case in cases:
        row, = [r for p in paths if p["path_index"] == case["path_index"]
                for r in p["predictions"] if r["step"] == case["step"]]
        if row["geometry_sha256"] != case["geometry_sha256"]:
            raise ValueError("Prediction and future-case geometry differ")
        future.append({"case": case["case"], **row})
    spec = {"status": "frozen", "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "scope": "Prospective empirical force envelopes and local-work predictions; no future labels used",
            "primary_force_budget_ev_a": EPSILON, "descriptive_auxiliary_budgets_ev_a": AUXILIARY_BUDGETS,
            "budget_rationale": "Quarter of the historical interface tolerance; fixed before future reference evaluation",
            "q_hat_rule": "0.04 A central residual difference", "eta_rule": "2 * norm_inf2(q_.02 - q_.04)",
            "K_rule": "2 * largest norm_inf2(q) over both directions and scales at the same anchor",
            "M_rule": "4 * max_dev norm_inf2(R - (u.dot(h))*q_hat)/norm2(h)^2; no subtraction of other radius terms",
            "numerical_floor_rule": "0.001 * primary_force_budget = 0.00025 eV/A, fixed for all descriptive budgets",
            "envelope": "abs(alpha)*norm_inf2(q_hat) + 0.00025 + eta*abs(alpha) + K*norm2(z) + M*norm2(h)^2/2",
            "interpretation": "Finite probes estimate these coefficients; neither transverse nor Taylor remainder bounds are certified",
            "domain": {"maximum_time_fs": H_CAP_FS, "maximum_transverse_fraction": TRANSVERSE_FRACTION_CAP},
            "horizon_semantics": "First failed primary-grid point ends the admitted prefix, including times without evaluation labels; no continuous-time certificate",
            "check_selection": selection,
            "growth_judgment": {"candidate_times_fs": [0.125, 0.25, 0.375, 0.5],
                                "signal_multiple": 3, "maximum_squared_error_ratio": 0.5,
                                "amplitude_ratio_range": [0.5, 2.0], "primary_paths": [0, 2],
                                "rule": "On the prespecified per-path diagnostic times, require finite measured e >= 3*nu; Sdir=sum(norm_inf2(R-P)^2)/sum(norm_inf2(R)^2) <= 0.5 and every e/L in [0.5,2]. Missing or unresolved signals do not pass. Report all four directions, vector-radius coverage, e/B and radius/B separately from force-budget admission."},
            "work_judgment": {"relative_tolerance": 0.05, "absolute_allowance_ev": 0.0005,
                              "resolved_signal_ratio": 10, "output_precision_floor_ev": 0.00001,
                              "reference_force_sensitivity_gate_ev_a": 0.01 * EPSILON,
                              "rule": "Require positive endpoint W, finest Q, fixed-c tightened W and half-step W, as predicted before evaluation. Require Q-W, paired-SCF W change and half-step W change each within 0.05*abs(W)+0.0005 eV. Require abs(W) at least ten times each of Q-W, Q_.125-Q_.25, Q_.25-Q_.5, deltaW_fixed, deltaW_curv, W_half-W and 1e-5 eV. This is a declared numerical resolution test, not a total error bound. Report all differences and all four paths; Danch separately, with abs(Danch)<abs(W)/3 required to call work the dominant reference-energy contribution. A resolved work identity does not by itself validate the leading local amplitude or force-boundary expansion."},
            "directions": estimates, "input_sha256": inputs, "source_sha256": sha(Path(__file__))}
    outputs = dict(zip(names, [spec, paths, future]))
    for name, value in outputs.items():
        with (DATA / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selection": selection, "horizons": [{"path_index": p["path_index"],
                      "horizons_fs": p["horizons_fs"]} for p in paths], "future_prediction_count": len(future)}, indent=2))


if __name__ == "__main__":
    main()
