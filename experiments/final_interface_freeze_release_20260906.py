"""Bind six check geometries, scientific choices and both executors before release."""

import datetime as dt
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((DATA / name).read_text())


def save(name, value):
    with (DATA / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    if any((DATA / f).exists() for f in ("check_cases.json", "check_protocol.json", "future_protocol.json")):
        raise FileExistsError("Never overwrite frozen release artifacts")
    spec, paths, dynamics = read("prediction_spec.json"), read("prospective_path_predictions.json"), read("dynamics_protocol.json")
    if spec["status"] != "frozen":
        raise ValueError("Freeze predictions first")
    inputs = dict(spec["input_sha256"])
    for name in ("prediction_spec.json", "prospective_path_predictions.json", "future_predictions.json",
                 "future_cases.json", "future_cases_provenance.json", "runtime_contract.json",
                 "goal_based_closure.json"):
        inputs[name] = sha(DATA / name)
    if any(sha(DATA / name) != digest for name, digest in inputs.items()):
        raise ValueError("Changed scientific input")
    p, r = spec["check_selection"]["primary_check_path"], spec["check_selection"]["other_anchor_check_path"]
    if (p, r) != (2, 0):
        raise ValueError("Unexpected development-only check selection")
    schedule = [("scf_origin", p, "primary", 0.0, 1e-10),
                ("scf_endpoint", p, "primary", 1.0, 1e-10),
                ("quadrature", p, "primary", 0.625, 1e-8),
                ("quadrature", p, "primary", 0.875, 1e-8),
                ("timestep", p, "half_step", 1.0, 1e-8),
                ("timestep", r, "half_step", 1.0, 1e-8)]
    checks = []
    for i, (kind, path_index, integration, t, threshold) in enumerate(schedule):
        path = dynamics["paths"][path_index]
        source, fixed_c = f"dynamics/path_{path_index}/{integration}_states.jsonl", f"dynamics/path_{path_index}/anchor.json"
        for name in (source, fixed_c, f"dynamics/path_{path_index}/complete.json"):
            inputs[name] = sha(DATA / name)
        states = [json.loads(line) for line in (DATA / source).read_text().splitlines()]
        row, = [x for x in states if x["time_fs"] == t]
        checks.append({"case": i, "kind": kind, **{k: path[k] for k in
                       ("path_index", "anchor_step", "seed", "initial_state_file")},
                       "integration": integration, "integration_step_fs": dynamics["integration_steps_fs"][integration],
                       "step": row["step"], "time_fs": t, "conv_thr": threshold,
                       "reference_label": "tightened" if threshold == 1e-10 else "standard",
                       "positions_angstrom": row["positions_angstrom"], "geometry_sha256": row["geometry_sha256"],
                       "source_file": source, "source_sha256": inputs[source],
                       "momenta_sha256": hashlib.sha256(np.asarray(row["momenta"], dtype="<f8").tobytes()).hexdigest(),
                       "fixed_c_file": fixed_c, "fixed_c_sha256": inputs[fixed_c]})
    check_module = runpy.run_path(str(ROOT / "experiments/final_interface_check_reference_20260906.py"))
    future_module = runpy.run_path(str(ROOT / "experiments/final_interface_future_reference_20260905.py"))
    runtime_names = future_module["QE_EXECUTABLES"]
    contract = read("runtime_contract.json")
    runtime = {name: digest for name, digest in contract["runtime_files_sha256"].items()
               if name in runtime_names or name.endswith(".UPF")}
    if len(runtime) != 10:
        raise ValueError("Expected original QE runtime and eight pseudopotentials")
    selection = {"rule": "development_kappa_stability_v1", "p": p, "r": r,
                 "input_sha256": {name: inputs[name] for name in ("development_analysis.json", "directions.json")},
                 "scores": [{"path_index": x["path_index"],
                             "kappa_0p02": x["two_scale_time_curvatures_ev_fs2"][0],
                             "kappa_0p04": x["two_scale_time_curvatures_ev_fs2"][1],
                             "score": x["check_selection_score"], "nonfinite_disclosure": ""}
                            for x in spec["directions"]]}
    scientific = {"force_budget": spec["primary_force_budget_ev_a"],
                  "admission_windows": {str(x["path_index"]): x["horizons_fs"] for x in paths},
                  "development_numerical_margins": {str(x["path_index"]):
                      {k: x[k] for k in ("eta_parallel_ev_a2", "K_perp_empirical_ev_a2", "M_empirical_ev_a3",
                                        "growth_sensitivity_proxy_ev_a")} for x in spec["directions"]},
                  "interpretation_rules": {"growth": spec["growth_judgment"], "work": spec["work_judgment"],
                                           "horizon": spec["horizon_semantics"]}}
    save("check_cases.json", {"status": "frozen", "cases": checks})
    inputs["check_cases.json"] = sha(DATA / "check_cases.json")
    common = {"status": "frozen", "created_at": dt.datetime.now(dt.UTC).isoformat(),
              "compute_cutoff": None, "stopping_policy_file": "goal_based_closure.json",
              "per_attempt_timeout_s": 10800, "campaign_attempt_ceiling": 64,
              "predictions_file": "future_predictions.json", "runtime_files_sha256": runtime,
              "preparation_source_sha256": sha(Path(__file__))}
    check_protocol = {**common, "protocol_name": "six_check_reference_20260906_v1", "mode": "mainfuture",
                      "cases_file": "check_cases.json", "selection": selection, "scientific_freeze": scientific,
                      "input_sha256": dict(inputs),
                      "source_files_sha256": {name: sha(ROOT / name) for name in sorted(check_module["SOURCES"])}}
    save("check_protocol.json", check_protocol)
    inputs["check_protocol.json"] = sha(DATA / "check_protocol.json")
    future_protocol = {**common, "cases_file": "future_cases.json", "input_sha256": inputs,
                       "spec": {"kind": "primary_future", "case_count": 40, "path_indices": [0, 1, 2, 3],
                                "integration_step_fs": dynamics["integration_steps_fs"]["primary"],
                                "evaluation_times_fs": dynamics["future_primary_evaluation_times_fs"]},
                       "source_files_sha256": {name: sha(ROOT / name) for name in sorted(future_module["SOURCES"])}}
    save("future_protocol.json", future_protocol)
    print(json.dumps({name: sha(DATA / name) for name in ("check_cases.json", "check_protocol.json", "future_protocol.json")}, indent=2))


if __name__ == "__main__":
    main()
