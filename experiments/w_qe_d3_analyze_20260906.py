"""Validate the fixed two-geometry QE-D3 batch, then decompose archived forces.

uv run --no-sync python experiments/w_qe_d3_analyze_20260906.py --attempt 7623747
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/final_campaign_20260905/w_fairness_d3_20260906"
STOCK = ROOT / "analysis/final_campaign_20260905/w_fairness_stock_20260906"
OLD = ROOT / "analysis/execution_20260905/tungsten"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, x):
    (OUT / name).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True)
    a = ap.parse_args()
    run = OUT / ("attempt_" + a.attempt)
    raw = run / "w_d3_raw.dat"
    plan = json.loads((OUT / "plan.json").read_text())
    exit_code = (run / "exit_code.txt").read_text().strip()
    if exit_code == "0":
        assert sha(raw) == (run / "result.sha256").read_text().split()[0]
    else:
        # An explicit, preserved post-output failure adjudication is required.
        # Never accept arbitrary failed jobs merely because a file exists.
        failure = json.loads((OUT / "repair_03.json").read_text())
        assert a.attempt == failure["numerical_job"] == "7623747" and exit_code == "5"
        assert sha(raw) == failure["raw_sha256_observed_remotely"]
        assert failure["runtime_and_release_hashes_rechecked_after_job"]
        assert "MPI_Comm_free" in (run / "compute.stderr").read_text()
    assert sha(OUT / "pair_qe_d3.in") == plan["input_file_sha256"]
    stockplan = json.loads((STOCK / "plan.json").read_text())
    assert all(sha(f["path"]) == f["sha256"] for f in stockplan["input_files"])
    lines = raw.read_text().splitlines()
    assert lines[-1] == "COMPLETE"
    abohr, eha = map(float, lines[0].split()[-2:])
    parameters = np.array(list(map(float, lines[1].split()[-5:])))
    cutoffs = np.array(list(map(float, lines[2].split()[-2:])))
    f, energy, differences = [], [], []
    i = 3
    for k in range(2):
        tokens = lines[i].split()
        assert tokens[:3] == ["geometry", str(k), "energy_Ha"]
        energy.append(float(tokens[3]))
        vals = np.array([list(map(float, line.split())) for line in lines[i+1:i+433]])
        assert vals.shape == (432, 4) and np.array_equal(vals[:, 0], np.arange(1, 433))
        f.append(vals[:, 1:] * eha / abohr)  # Driver has already applied force=-gradient.
        i += 433
    for line in lines[i:-1]:
        assert line.startswith("gradient_check ")
        h, analytic, central, diff = map(float, line.split()[-4:])
        assert abs((central-analytic)-diff) < 1e-14
        tolerance = 1e-6 + 1e-4*abs(analytic)
        differences.append({"h_bohr": h, "analytic_Ha_bohr": analytic,
                            "central_Ha_bohr": central, "difference_Ha_bohr": diff,
                            "tolerance_Ha_bohr": tolerance, "pass": abs(diff) <= tolerance})
    f = np.array(f)
    energy = np.array(energy)*eha
    assert np.isfinite(f).all() and np.isfinite(energy).all()
    assert [x["h_bohr"] for x in differences] == [.001, .0005]
    ediff = float(2*energy[0]/eha-plan["qe_scalar_validation"]["DFT_D3_energy_Ry"])
    gates = {"jobid": a.attempt, "raw_sha256": sha(raw),
             "executable_sha256": (run / "executable.sha256").read_text().split()[0],
             "units": {"bohr_A": abohr, "hartree_eV": eha},
             "parameters_observed": parameters.tolist(),
             "parameters_match": bool(np.allclose(parameters, [1, 1.217, .722, 1, 14], atol=1e-7, rtol=0)),
             "squared_cutoffs_match": bool(np.allclose(cutoffs, [9000, 1600], atol=1e-10, rtol=0)),
             "origin_energy_difference_Ry": ediff,
             "origin_energy_tolerance_Ry": 1e-7,
             "origin_energy_pass": abs(ediff) <= 1e-7,
             "directional_checks": differences,
             "gradient_checks_pass": all(x["pass"] for x in differences),
             "D3_source_unchanged": sha(OUT / "release_03/w_qe_d3_component_20260906.f90") ==
                    json.loads((OUT / "preparation_checks.json").read_text())["source_sha256"],
             "scheduler_exit_code": exit_code,
             "post_output_finalization_failure": exit_code == "5"}
    gates["pass"] = all(gates[k] for k in ("parameters_match", "squared_cutoffs_match", "origin_energy_pass",
                                           "gradient_checks_pass", "D3_source_unchanged"))
    save("validation.json", gates)
    if not gates["pass"]:
        raise SystemExit("Fixed validation failed; no validated residual decomposition emitted")
    oldplan = json.loads((OLD / "frozen_plan.json").read_text())
    core = np.isin(np.arange(432), oldplan["initial_core_atom_indices_0based"])
    masks = {"all": np.ones(432, bool), "core": core, "matrix": ~core}

    def stats(v, mask):
        norms = np.linalg.norm(v[mask], axis=1)
        return {"max_ev_A": float(norms.max()), "vector_rms_ev_A": float(np.sqrt(np.mean(norms**2))),
                "vector_mae_ev_A": float(norms.mean()), "atoms_above_0p2": int(np.count_nonzero(norms > .2))}

    rows, results, arrays = [], [], {"d3_forces_ev_A": f, "d3_energy_ev": energy, "core_mask": core}
    names = ["stock_MP0b3", "stock_MPA0", "stock_mean", "old_perturbed_MPA0", "old_mean"]
    for k, gid in enumerate(["W24_00", "W24_20"]):
        stock = np.load(STOCK / (gid + "_stock.npz"))
        old = np.load(OLD / "predictions" / (gid + ".npz"))
        ref = stock["reference_forces"]
        ref_pbe = ref-f[k]
        model = np.stack([*stock["stock_forces"], stock["stock_mean_forces"],
                          old["member_forces"][1], old["mean_forces"]])
        arrays[gid + "_reference_PBE_D3"] = ref
        arrays[gid + "_reference_PBE"] = ref_pbe
        arrays[gid + "_model_forces"] = model
        r = {"geometry_id": gid, "D3_energy_eV": float(energy[k]),
             "D3_force": {region: stats(f[k], mask) for region, mask in masks.items()},
             "net_D3_force_ev_A": f[k].sum(axis=0).tolist(), "models": {}}
        for mi, name in enumerate(names):
            before, after = model[mi]-ref, model[mi]-ref_pbe
            assert np.allclose(after-before, f[k], atol=1e-13, rtol=1e-12)
            group = {"PBE_D3": {}, "PBE_without_D3": {}}
            for ref_name, residual in (("PBE_D3", before), ("PBE_without_D3", after)):
                for region, mask in masks.items():
                    st = stats(residual, mask)
                    group[ref_name][region] = st
                    rows.append({"geometry_id": gid, "model": name, "reference": ref_name, "region": region, **st})
            old_max = group["PBE_D3"]["all"]["max_ev_A"]
            new_max = group["PBE_without_D3"]["all"]["max_ev_A"]
            lower_bound = max(0., old_max-r["D3_force"]["all"]["max_ev_A"])
            assert new_max + 1e-12 >= lower_bound
            group["triangle_lower_bound_without_D3_max_ev_A"] = lower_bound
            group["max_error_change_ev_A"] = new_max-old_max
            r["models"][name] = group
        results.append(r)
    with (OUT / "residual_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(OUT / "force_decomposition.npz", **arrays, model_names=names)
    stocksummary = json.loads((STOCK / "summary.json").read_text())
    deltas = stocksummary["pair_energy_increment_ev"]
    d3delta = float(energy[1]-energy[0])
    pbedelta = deltas["reference"]-d3delta
    result = {"status": "validated_complete", "jobid": a.attempt, "gates": gates,
              "reference_decomposition": "F_PBE = F_PBE_D3 - F_D3; residual_after = residual_before + F_D3",
              "reference_after_removal": "Same fixed electronic reference with additive geometry-only D3 removed; other reference differences remain",
              "geometry_results": results, "D3_pair_energy_increment_ev": d3delta,
              "PBE_pair_energy_increment_ev": pbedelta,
              "model_pair_increment_errors_without_D3_ev": {name: deltas[name]-pbedelta for name in names},
              "new_DFT": 0, "new_model_inference": 0,
              "numerical_scope": "Two fixed geometries; total D3 includes threebody; directional gradient check is not an all-component error bound",
              "input_hashes": {str(p): sha(p) for p in [raw, OUT / "plan.json", STOCK / "summary.json",
                 STOCK / "W24_00_stock.npz", STOCK / "W24_20_stock.npz", ROOT / "experiments/w_qe_d3_analyze_20260906.py"]}}
    save("summary.json", result)
    print(json.dumps({"gates": gates, "D3_50fs": results[1]["D3_force"],
                      "without_D3_50fs": {n: r["PBE_without_D3"] for n, r in results[1]["models"].items()},
                      "D3_energy_increment_ev": d3delta}, indent=2))


if __name__ == "__main__":
    main()
