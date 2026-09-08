"""Evaluate the frozen material predictions using completed reference artifacts.

No simulation, fitting, protocol mutation or missing-label imputation. Each run
writes an exclusive snapshot; all incomplete judgments remain explicitly pending.
"""

import argparse
import datetime as dt
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read as read_atoms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"
FROZEN = {
    "future_protocol.json": "01ab35a2d5efe3d2607231cfe8a8930d159322d5e35b3ad3fdfdc92f61ba1e76",
    "check_protocol.json": "2491df9b1a55f13c95205b908a978a2e0b6a94b6c6b15c15e14799a1458c2ab2",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def norm(a):
    return float(np.linalg.norm(a, axis=1).max())


def ratio(a, b):
    return float(a / b) if b != 0 else None


def endpoint(ref_energy, frame, origin_ref_energy, origin, correction):
    displacement = np.asarray(frame["positions_angstrom"]) - origin["positions_angstrom"]
    work = ((ref_energy - origin_ref_energy)
            - (frame["base_energy_ev"] - origin["base_energy_ev"])
            + float(np.sum(correction * displacement)))
    delta_h = ((ref_energy - origin_ref_energy)
               + frame["kinetic_energy_ev"] - origin["kinetic_energy_ev"])
    drift = frame["anchored_hamiltonian_drift_ev"]
    require(abs(delta_h - work - drift) < 1e-8, "Reference energy decomposition failed")
    return {"endpoint_work_ev": work, "anchored_hamiltonian_drift_ev": drift,
            "reference_hamiltonian_change_ev": delta_h,
            "energy_identity_defect_ev": delta_h - work - drift}


def quadrature(points, spacing, end):
    times = [float(x * spacing) for x in range(round(end / spacing) + 1)]
    missing = [t for t in times if t not in points]
    if missing:
        return {"status": "pending", "missing_times_fs": missing}
    segments = [float(np.sum(0.5 * (points[a]["residual"] + points[b]["residual"])
                            * (points[b]["positions"] - points[a]["positions"])))
                for a, b in itertools.pairwise(times)]
    total = sum(segments)
    return {"status": "complete", "work_ev": total, "segment_work_ev": segments,
            "absolute_segment_work_sum_ev": sum(abs(w) for w in segments),
            "endpoint_difference_ev": total - points[end]["endpoint_work_ev"]}


class Evaluation:
    def __init__(self, data):
        self.data, self.hashes = data, {}

    def raw(self, name, expected=None):
        raw = (self.data / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None:
            require(digest == expected, f"Changed artifact: {name}")
        self.hashes[name] = digest
        return raw

    def read(self, name, expected=None):
        return json.loads(self.raw(name, expected))

    def reference(self, directory, case, protocol=None):
        folder = f"{directory}/case_{case['case']:02d}"
        name = f"{folder}/result.json"
        if not (self.data / name).is_file():
            failure = f"{folder}/failure.json"
            if (self.data / failure).is_file():
                result = self.read(failure)
                return {"status": "failed", "case": case["case"], "error": result["error"]}
            return {"status": "pending", "case": case["case"]}
        result = self.read(name)
        require(result["status"] == "complete" and result["specification"] == case,
                f"Mismatched result: {name}")
        if protocol:
            require(result["protocol_sha256"] == FROZEN[protocol], "Wrong reference protocol")
        self.raw(f"{folder}/qe_runs/step/pw.in", result["input_sha256"])
        output = self.raw(f"{folder}/qe_runs/step/pw.out", result["output_sha256"]).decode()
        require("JOB DONE." in output and "convergence has been achieved" in output
                and "convergence NOT achieved" not in output, "Incomplete reference SCF")
        # Independent direct re-read of printed energies and all atomic forces.
        energies = re.findall(r"^!\s+total energy\s+=\s+([-+0-9.EeDd]+)\s+Ry", output, re.MULTILINE)
        force_tokens = re.findall(
            r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)", output)
        require(len(force_tokens) == 474 and len(energies) == 1, "Expected one 474-atom SCF")
        energy = float(energies[0].replace("D", "E")) * units.Hartree / 2
        forces = np.array([[float(x.replace("D", "E")) for x in row] for row in force_tokens])
        forces *= units.Hartree / 2 / units.Bohr
        require(np.isfinite(forces).all() and np.isfinite(energy), "Nonfinite reference")
        require(abs(energy - result["energy_ev"]) < 1e-9
                and np.max(abs(forces - result["forces_ev_a"])) < 1e-10, "Raw label mismatch")
        return result

    def evaluate(self):
        for name, digest in FROZEN.items():
            protocol = self.read(name, digest)
            for source, expected in protocol["input_sha256"].items():
                self.raw(source, expected)
        spec = self.read("prediction_spec.json")
        future_cases = self.read("future_cases.json")
        check_cases = self.read("check_cases.json")["cases"]
        predictions = self.read("future_predictions.json")
        path_predictions = self.read("prospective_path_predictions.json")
        development_cases = self.read("development_cases.json")
        frames, origins, corrections, origin_refs, symbols = {}, {}, {}, {}, {}
        directions = {p["path_index"]: p for p in spec["directions"]}
        points = {p: {} for p in range(4)}
        for p, direction in directions.items():
            for integration, filename in (("primary", "primary_states.jsonl"),
                                          ("half_step", "half_step_states.jsonl")):
                rows = [json.loads(line) for line in self.raw(f"dynamics/path_{p}/{filename}").splitlines()]
                frames[p, integration] = {row["time_fs"]: row for row in rows}
            origins[p] = frames[p, "primary"][0.0]
            anchor = self.read(f"dynamics/path_{p}/anchor.json")
            corrections[p] = np.array(anchor["correction_ev_A"])
            index = 0 if direction["anchor_step"] == 36 else 9
            origin_refs[p] = self.reference("reference_results", development_cases[index])
            require(origin_refs[p]["status"] == "complete", "Missing original reference")
            require(origin_refs[p]["energy_ev"] == anchor["reference_energy_ev"], "Changed anchor energy")
            symbols[p] = read_atoms(self.data / direction["initial_state_file"]).get_chemical_symbols()
            f0 = np.asarray(origin_refs[p]["forces_ev_a"])
            require(norm(np.asarray(origins[p]["base_forces_ev_a"]) + corrections[p] - f0) < 1e-10,
                    "Origin no longer force matched")
            points[p][0.0] = {"positions": np.array(origins[p]["positions_angstrom"]),
                             "residual": np.zeros_like(f0), "endpoint_work_ev": 0.0}
        rows, references, checks, checked_points = [], {}, [], {}
        for kind, cases, output in (("future", future_cases, rows), ("check", check_cases, checks)):
            for case in cases:
                result = self.reference(f"{kind}_reference_results", case, f"{kind}_protocol.json")
                row = {k: case[k] for k in ("case", "path_index", "time_fs")}
                row["status"] = result["status"]
                if result["status"] != "complete":
                    if "error" in result:
                        row["error"] = result["error"]
                    output.append(row)
                    continue
                p, t = case["path_index"], case["time_fs"]
                integration = case.get("integration", "primary")
                frame = frames[p, integration][t]
                require(frame["geometry_sha256"] == case["geometry_sha256"]
                        and np.array_equal(frame["positions_angstrom"], case["positions_angstrom"]),
                        "Reference and model geometries differ")
                residual = np.array(frame["base_forces_ev_a"]) + corrections[p] - result["forces_ev_a"]
                origin_energy = origin_refs[p]["energy_ev"]
                if kind == "check" and case["kind"] == "scf_origin":
                    origin_energy = result["energy_ev"]
                elif kind == "check" and case["kind"] == "scf_endpoint":
                    origin_energy = checked_points[0][1]["energy_ev"] if 0 in checked_points else None
                row["reference_origin_energy_ev"] = origin_energy
                if origin_energy is None:
                    row.update(endpoint_work_ev=None, reference_hamiltonian_change_ev=None,
                               energy_identity_defect_ev=None,
                               anchored_hamiltonian_drift_ev=frame["anchored_hamiltonian_drift_ev"],
                               work_status="pending_tightened_origin")
                else:
                    row.update(endpoint(result["energy_ev"], frame, origin_energy, origins[p], corrections[p]),
                               work_status="complete")
                error = norm(residual)
                row.update(max_atom_error_ev_a=error, reference_energy_ev=result["energy_ev"],
                           reference_seconds=result["reference_seconds"],
                           estimated_final_scf_accuracy_Ry=result["estimated_final_scf_accuracy_Ry"],
                           integration=integration)
                point = {**row, "residual": residual, "positions": np.array(frame["positions_angstrom"])}
                if kind == "future":
                    references[case["case"]] = result
                    points[p][t] = point
                    prediction = predictions[case["case"]]
                    vector = prediction["directional_displacement_a"] * np.asarray(directions[p]["q_hat_ev_a2"])
                    defect = norm(residual - vector)
                    linear, score = prediction["linear_error_prediction_ev_a"], prediction["primary_score_ev_a"]
                    radius = score - linear
                    top = np.argsort(np.linalg.norm(residual, axis=1))[::-1][:10]
                    row.update(prediction=prediction, linear_vector_defect_ev_a=defect,
                               vector_radius_covered=defect <= radius, error_over_linear=ratio(error, linear),
                               error_over_score=ratio(error, score), radius_over_score=ratio(radius, score),
                               vector_defect_over_radius=ratio(defect, radius),
                               primary_force_budget_pass=error <= spec["primary_force_budget_ev_a"],
                               growth_signal_resolved=error >= spec["growth_judgment"]["signal_multiple"]
                               * directions[p]["growth_sensitivity_proxy_ev_a"],
                               observed_error_rate_ev_a_fs=ratio(error, t),
                               observed_2W_over_e2_a2_ev=ratio(2 * row["endpoint_work_ev"], error**2),
                               work_over_directional_prediction=ratio(row["endpoint_work_ev"],
                                                                     prediction["directional_work_prediction_ev"]),
                               top_residual_atoms=[{"index_zero_based": int(i), "element": symbols[p][i],
                                                    "residual_ev_a": float(np.linalg.norm(residual[i]))} for i in top])
                else:
                    row["kind"] = case["kind"]
                    checked_points[case["case"]] = (point, result)
                    if case["kind"] == "quadrature":
                        require(t not in points[p], "Quadrature point duplicates primary label")
                        points[p][t] = point
                output.append(row)

        path_summaries = []
        for p, direction in directions.items():
            subset = [r for r in rows if r["path_index"] == p]
            completed = {r["time_fs"]: r for r in subset if r["status"] == "complete"}
            diagnostic = direction["growth_diagnostic_times_fs"]
            missing = [t for t in diagnostic if t not in completed]
            growth = {"status": "pending" if missing else "complete", "missing_times_fs": missing,
                      "passed": None, "primary_test_path": p in spec["growth_judgment"]["primary_paths"]}
            if not missing:
                tested = [completed[t] for t in diagnostic]
                squared_ratio = ratio(sum(r["linear_vector_defect_ev_a"]**2 for r in tested),
                                      sum(r["max_atom_error_ev_a"]**2 for r in tested))
                lo, hi = spec["growth_judgment"]["amplitude_ratio_range"]
                growth.update(squared_vector_error_ratio=squared_ratio,
                              all_signal_resolved=all(r["growth_signal_resolved"] for r in tested),
                              all_amplitudes_in_range=all(lo <= r["error_over_linear"] <= hi for r in tested))
                growth["passed"] = bool(growth["all_signal_resolved"] and growth["all_amplitudes_in_range"]
                                        and squared_ratio is not None
                                        and squared_ratio <= spec["growth_judgment"]["maximum_squared_error_ratio"])
            admitted_times = [q["time_fs"] for q in predictions if q["path_index"] == p and q["primary_admitted"]]
            admitted = [completed[t] for t in admitted_times if t in completed]
            admission = {"expected_count": len(admitted_times), "completed_count": len(admitted),
                         "missing_times_fs": [t for t in admitted_times if t not in completed],
                         "violations": [r["time_fs"] for r in admitted if not r["primary_force_budget_pass"]],
                         "all_complete": len(admitted) == len(admitted_times)}
            admission["at_least_three_passed"] = (admission["all_complete"] and len(admitted) >= 3
                                                   and not admission["violations"])
            path_summaries.append({"path_index": p, "anchor_step": direction["anchor_step"],
                                   "frozen_horizons_fs": path_predictions[p]["horizons_fs"],
                                   "admission": admission, "growth": growth,
                                   "quadrature_to_half_fs": {str(h): quadrature(points[p], h, 0.5)
                                                              for h in (0.5, 0.25, 0.125)},
                                   "quadrature_to_one_fs": {str(h): quadrature(points[p], h, 1.0)
                                                             for h in (0.5, 0.25, 0.125)}})

        p = spec["check_selection"]["primary_check_path"]
        r = spec["check_selection"]["other_anchor_check_path"]
        work = {"primary_path": p, "status": "pending", "passed": None}
        timestep_checks = []
        for c, path in ((4, p), (5, r)):
            if c in checked_points and 1.0 in points[path]:
                half = checked_points[c][0]
                primary = points[path][1.0]
                timestep_checks.append({"path_index": path,
                                        **{f"half_minus_primary_{k}": half[k] - primary[k] for k in
                                           ("endpoint_work_ev", "max_atom_error_ev_a", "reference_hamiltonian_change_ev")},
                                        "half_work_ev": half["endpoint_work_ev"]})
        paired = None
        if 0 in checked_points and 1 in checked_points and 1.0 in points[p]:
            standard = references[p * 10 + 5]
            tight0, tight1 = checked_points[0][1], checked_points[1][1]
            df0 = np.array(tight0["forces_ev_a"]) - origin_refs[p]["forces_ev_a"]
            df1 = np.array(tight1["forces_ev_a"]) - standard["forces_ev_a"]
            delta_fixed = ((tight1["energy_ev"] - standard["energy_ev"])
                           - (tight0["energy_ev"] - origin_refs[p]["energy_ev"]))
            linear_change = float(np.sum(df0 * (points[p][1.0]["positions"] - points[p][0.0]["positions"])))
            paired = {"delta_work_fixed_c_ev": delta_fixed, "delta_anchor_linear_projection_ev": linear_change,
                      "delta_work_retangent_ev": delta_fixed + linear_change,
                      "origin_force_change_ev_a": norm(df0), "endpoint_force_change_ev_a": norm(df1),
                      "tightened_work_fixed_c_ev": points[p][1.0]["endpoint_work_ev"] + delta_fixed,
                      "force_sensitivity_pass": max(norm(df0), norm(df1))
                      <= spec["work_judgment"]["reference_force_sensitivity_gate_ev_a"]}
            require(abs(checked_points[1][0]["endpoint_work_ev"] - paired["tightened_work_fixed_c_ev"]) < 1e-9,
                    "Tightened endpoint work must use its matched tightened origin")
        q = path_summaries[p]["quadrature_to_one_fs"]
        timecheck = next((item for item in timestep_checks if item["path_index"] == p), None)
        if paired is not None and timecheck and all(item["status"] == "complete" for item in q.values()):
            w = points[p][1.0]["endpoint_work_ev"]
            rule = spec["work_judgment"]
            differences = {"fine_quadrature_minus_endpoint_ev": q["0.125"]["work_ev"] - w,
                           "quadrature_fine_minus_medium_ev": q["0.125"]["work_ev"] - q["0.25"]["work_ev"],
                           "quadrature_medium_minus_coarse_ev": q["0.25"]["work_ev"] - q["0.5"]["work_ev"],
                           "delta_work_fixed_c_ev": paired["delta_work_fixed_c_ev"],
                           "delta_work_retangent_ev": paired["delta_work_retangent_ev"],
                           "half_minus_primary_work_ev": timecheck["half_minus_primary_endpoint_work_ev"],
                           "output_precision_floor_ev": rule["output_precision_floor_ev"]}
            allowed = rule["relative_tolerance"] * abs(w) + rule["absolute_allowance_ev"]
            positive = all(x > 0 for x in (w, q["0.125"]["work_ev"], paired["tightened_work_fixed_c_ev"],
                                           timecheck["half_work_ev"]))
            agreement = all(abs(differences[key]) <= allowed for key in (
                "fine_quadrature_minus_endpoint_ev", "delta_work_fixed_c_ev", "half_minus_primary_work_ev"))
            resolved = all(abs(w) >= rule["resolved_signal_ratio"] * abs(x) for x in differences.values())
            work.update(status="complete", endpoint_work_ev=w, all_prespecified_signs_positive=positive,
                        absolute_agreement_allowance_ev=allowed, agreement_pass=agreement,
                        resolved_signal_pass=resolved, differences_ev=differences,
                        resolution_ratios={k: ratio(abs(w), abs(x)) for k, x in differences.items()},
                        passed=bool(positive and agreement and resolved and paired["force_sensitivity_pass"]),
                        work_dominates_integrator_drift=abs(points[p][1.0]["anchored_hamiltonian_drift_ev"]) < abs(w) / 3,
                        endpoint_inside_primary_admission=False)
        all_complete = all(row["status"] == "complete" for row in rows + checks)
        closure = {
            "all_46_labels_complete": all_complete,
            "all_admitted_labels_complete": all(s["admission"]["all_complete"] for s in path_summaries),
            "observed_admitted_violation_count": sum(len(s["admission"]["violations"]) for s in path_summaries),
            "each_anchor_has_three_point_path": all(any(s["anchor_step"] == anchor
                                                         and s["admission"]["at_least_three_passed"]
                                                         for s in path_summaries) for anchor in (36, 161)),
            "both_primary_growth_tests_pass": all(path_summaries[i]["growth"]["passed"] is True for i in (0, 2)),
            "primary_work_test_pass": work["passed"],
            "interpretation": "Evidence checklist only; manuscript closure also requires physical interpretation and complete disclosure. Missing labels cannot count as passed."
        }
        return {"created_utc": dt.datetime.now(dt.UTC).isoformat(), "future": rows, "checks": checks,
                "paths": path_summaries, "paired_scf": paired, "timestep_checks": timestep_checks,
                "primary_work": work, "closure_evidence": closure,
                "complete_counts": {"future": sum(x["status"] == "complete" for x in rows),
                                    "checks": sum(x["status"] == "complete" for x in checks)},
                "input_sha256": self.hashes, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reference_energy": "Fixed-smearing variational QE E-TS, same surface as printed forces",
                "numerical_interpretation": "Paired sensitivities and quadrature differences are operational resolution checks, not statistical independence or total error bounds"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = Evaluation(DATA).evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "complete_counts": result["complete_counts"],
                      "closure_evidence": result["closure_evidence"], "primary_work": result["primary_work"]}, indent=2))


if __name__ == "__main__":
    main()
