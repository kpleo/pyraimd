"""NumPy-only reconstruction from the compact silicon archive; read-only."""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import json
import re

import numpy as np


def equal(a, b, atol=2e-12):
    np.testing.assert_allclose(a, b, rtol=2e-10, atol=atol)


def load(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def maxnorm(x):
    return np.linalg.norm(x, axis=-1).max()


def coefficients(u, response):
    q = np.sum(u * response, axis=(-2, -1))
    r2 = np.sum(response**2, axis=-1)
    s2, g2 = r2.sum(axis=-1), r2.max(axis=-1)
    return dict(Q_eV_A2=q, g2_eV2_A4=g2, S2_eV2_A4=s2, C_A2_eV=q/g2,
                chi_A2_eV=q/s2, n_infty=s2/g2, n2=s2**2/(r2**2).sum(axis=-1))


def response_at(z, h):
    plus, = np.flatnonzero(np.isclose(z["h_A"], h, atol=1e-14, rtol=0))
    minus, = np.flatnonzero(np.isclose(z["h_A"], -h, atol=1e-14, rtol=0))
    residual = z["F_DFT_eV_A"] - z["F_base_eV_A"]
    response = (residual[minus] - residual[plus]) / (2*h)
    return response, coefficients(z["u"], response)


def work(center, endpoint):
    c = center["F_DFT_eV_A"] - center["F_base_eV_A"]
    return ((endpoint["E_DFT_eV"] - center["E_DFT_eV"])
            - (endpoint["E_base_eV"] - center["E_base_eV"])
            + np.sum(c*(endpoint["coords_A"] - center["coords_A"])))


def point(z, index):
    return {key: z[key][index] for key in ("coords_A", "E_DFT_eV", "E_base_eV", "F_DFT_eV_A", "F_base_eV_A")}


def probe_geometry(z):
    equal(z["numbers"], np.full(64, 14))
    equal(z["pbc"], [1, 1, 1])
    equal(np.sum(z["u"]**2), 1)
    equal(z["u"].sum(axis=0), [0, 0, 0])
    equal(z["coords_A"] - z["coords_A"][0], z["h_A"][:, None, None]*z["u"], atol=2e-14)


def numeric_tree(value):
    if isinstance(value, dict):
        for k, v in value.items():
            assert re.fullmatch(r"[A-Za-z0-9_]+", k), "Nonphysical JSON key"
            numeric_tree(v)
    elif isinstance(value, list):
        for v in value:
            numeric_tree(v)
    else:
        assert isinstance(value, (int, float, bool)) and np.isfinite(value)


def integrity(root):
    manifest = root / "MANIFEST.sha256"
    names = []
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert re.fullmatch(r"[A-Za-z0-9_./]+", name)
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
        names.append(name)
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != manifest}
    assert len(names) == len(set(names)) and set(names) == present
    assert len(present) == 31, "Unexpected archive inventory"
    assert not any(re.search(r"metadata|prediction\.json|\d{7,}", n, re.I) for n in names)
    for p in root.rglob("*.json"):
        numeric_tree(load(p))
    for p in root.rglob("*.npz"):
        for key, x in arrays(p).items():
            assert re.fullmatch(r"[A-Za-z0-9_]+", key)
            assert x.dtype.kind in "biuf" and np.isfinite(x).all(), (p.name, key)


def check_sizes(root):
    stats = load(root/"size_series/statistics.json")
    expected = {f"Si{n}_{g}" for n in (64, 216, 512, 1000) for g in ("ideal", "displaced")}
    assert set(stats) == expected
    values = {}
    for n in (64, 216, 512, 1000):
        for geom in ("ideal", "displaced"):
            key = f"Si{n}_{geom}"
            z = arrays(root/f"size_series/{key}/directions.npz")
            assert z["u"].shape == z["H_delta_u"].shape == (32, n, 3)
            equal(z["numbers"], np.full(n, 14))
            equal(z["index"], np.arange(32))
            equal(np.sum(z["u"]**2, axis=(1, 2)), np.ones(32))
            equal(z["u"].sum(axis=1), np.zeros((32, 3)))
            v = coefficients(z["u"], z["H_delta_u"])
            v["C_rms_A2_eV"] = n*v["chi_A2_eV"]
            for field, x in v.items():
                equal(x, z[field])
            equal(v["C_A2_eV"], v["chi_A2_eV"]*v["n_infty"])
            s = stats[key]
            for field in ("C_A2_eV", "chi_A2_eV", "n_infty", "n2", "C_rms_A2_eV"):
                report = s["statistics"]["curvature_factor_A2_eV" if field == "chi_A2_eV" else field]
                x = v[field]
                equal(x.mean(), report["mean"])
                equal(x.std(ddof=1), report["sd"])
                equal(x.std(ddof=1)/abs(x.mean()), report["direction_CV"])
                equal(np.quantile(x, [.05, .5, .95]), report["direction_quantiles_05_50_95"])
            csv = s["summary_csv"]
            for field, csvkey in (("C_A2_eV", "mean_C"), ("chi_A2_eV", "mean_chi"),
                                  ("n_infty", "mean_n_infty"), ("n2", "mean_n2")):
                equal(v[field].mean(), csv[csvkey])
            for field, csvkey in (("C_A2_eV", "C_CV"), ("curvature_factor_A2_eV", "chi_CV")):
                equal(s["statistics"][field]["direction_CV"], csv[csvkey])
            equal(s["statistics"]["C_A2_eV"]["conditional_mean_bootstrap95"],
                  [csv["C_mean_CI_low"], csv["C_mean_CI_high"]])
            covariance = np.mean((v["chi_A2_eV"]-v["chi_A2_eV"].mean())*(v["n_infty"]-v["n_infty"].mean()))
            equal(covariance, s["covariance_chi_ninf"])
            equal(v["C_A2_eV"].mean(), s["mean_product_chi_ninf"]+covariance)
            matrix = np.einsum("iak,jak->ij", z["u"], z["H_delta_u"])
            equal(np.linalg.norm(matrix-matrix.T)/np.linalg.norm(matrix), s["hessian_symmetry_relative"])
            assert len(z["half_h_direction_indices"]) == 4
            equal(z["fd_two_scale_relative_norm"].max(), s["fd_two_scale_max"])
            values[n, geom] = v["C_A2_eV"].mean()
            print(f"{key}: C={v['C_A2_eV'].mean():.8f}, chi={v['chi_A2_eV'].mean():.8f}, "
                  f"n_inf={v['n_infty'].mean():.6f}, n2={v['n2'].mean():.6f}")
    for (n, geom), c in values.items():
        s = stats[f"Si{n}_{geom}"]
        ratios = s["budget_ratios"]
        equal(np.sqrt(abs(values[64, geom]/c)), ratios["force_budget_ratio_for_equal_mean_total_work"])
        equal(np.sqrt(abs(n*values[64, geom]/(64*c))), ratios["force_budget_ratio_for_equal_mean_per_atom_work"])
        if "frozen_transfer" in s:
            t = s["frozen_transfer"]
            equal(c, t["measured_mean_A2_eV"])
            equal(abs(c-t["frozen_prediction_A2_eV"])/abs(c), t["absolute_relative_mean_discrepancy"])


def check_thermal(root):
    for anchor in (1, 2):
        directory = root/f"thermal/Si64_anchor{anchor}"
        z = arrays(directory/"probes.npz")
        probe_geometry(z)
        equal([z["ecutwfc_Ry"], z["ecutrho_Ry"]], [50, 400])
        equal(z["k_grid"], [2, 2, 2])
        assert z["functional_PBE"] == 1 and z["dispersion_D3"] == 0
        report = load(directory/"probe_statistics.json")
        responses, coeff = {}, {}
        for name, h, indices in (("large", .04, (1, 2)), ("small", .02, (3, 4))):
            response, c = response_at(z, h)
            responses[name], coeff[name] = response, c
            for key, value in c.items():
                equal(value, report["scales"][name][key])
            for index, tag in zip(indices, ("plus_h", "minus_h") if name == "large" else ("plus_half", "minus_half")):
                w = work(point(z, 0), point(z, index))
                equal(w, report["scales"][name]["anchored_work_eV"][tag])
                equal(abs(w-.5*h*h*c["Q_eV_A2"])/abs(w), report["scales"][name]["relative_energy_force_error"][tag])
        c = coeff["small"]
        equal(z["H_delta_u_frozen"], responses["small"])
        equal(z["h_frozen_A"], .02)
        for key in c:
            if f"frozen_{key}" in z:
                equal(c[key], z[f"frozen_{key}"])
        equal(z["c_eV_A"], z["F_DFT_eV_A"][0]-z["F_base_eV_A"][0])
        equal(z["velocity_A_per_fs"], z["u"]*z["velocity_norm_A_per_fs"])
        hvp_change = np.linalg.norm(responses["small"]-responses["large"])/np.linalg.norm(responses["large"])
        c_change = abs(c["C_A2_eV"]-coeff["large"]["C_A2_eV"])/abs(coeff["large"]["C_A2_eV"])
        equal(hvp_change, report["HVP_scale_relative_difference"])
        equal(c_change, report["C_scale_relative_difference"])
        print(f"Si64_anchor{anchor}: h=0.02 A C={c['C_A2_eV']:.10f}; "
              f"h/2 vs h HVP change={100*hvp_change:.6f}%, C change={100*c_change:.6f}%")
        traces = [arrays(directory/f"md_dt_{tag}_fs.npz") for tag in ("0p05", "0p025")]
        for tr in traces:
            equal(tr["coords_A"][0], z["coords_A"][0])
            equal(tr["velocity_A_per_fs"][0], z["velocity_A_per_fs"])
            equal(tr["F_base_eV_A"][0], z["F_base_eV_A"][0], atol=1e-7)
            equal(tr["E_base_eV"][0], z["E_base_eV"][0], atol=1e-7)
            equal(tr["cell_A"], z["cell_A"])
            kinetic = .5*np.sum(tr["masses_amu"][None, :, None]*(tr["velocity_A_per_fs"]/z["ase_time_unit_per_fs"])**2, axis=(1, 2))
            potential = tr["E_base_eV"]-np.einsum("ij,tij->t", z["c_eV_A"], tr["coords_A"]-z["coords_A"][0])
            defect = kinetic+potential-kinetic[0]-potential[0]
            equal(kinetic, tr["kinetic_energy_eV"])
            equal(potential, tr["U_star_eV"])
            equal(defect, tr["energy_defect_eV"])
            equal(np.sum(tr["masses_amu"][None, :, None]*tr["velocity_A_per_fs"], axis=1)/tr["masses_amu"].sum(), np.zeros((len(kinetic), 3)))
            assert np.all(np.diff(tr["time_fs"]) > 0)
            assert np.diff(tr["time_fs"]).max() <= tr["dt_fs"]+1e-12
        for j, budget in enumerate(("0p02", "0p04")):
            end = arrays(directory/f"budget_{budget}_eV_A.npz")
            for key in ("cell_A", "numbers", "ecutwfc_Ry", "ecutrho_Ry", "k_grid", "dispersion_D3"):
                equal(end[key], z[key])
            w = work(point(z, 0), end)
            actual = maxnorm(end["F_DFT_eV_A"]-end["F_base_eV_A"]-z["c_eV_A"])
            eps = end["epsilon_target_eV_A"]
            equal(eps, (.02, .04)[j])
            p0, pe = .5*c["C_A2_eV"]*eps**2, .5*c["C_A2_eV"]*actual**2
            nominal, converted = (p0-w)/abs(w), (pe-w)/abs(w)
            equal(end["tau_fs"], eps/(np.sqrt(c["g2_eV2_A4"])*np.linalg.norm(z["velocity_A_per_fs"])))
            equal(end["W_pred_frozen_eV"], p0)
            for key, value in dict(W_direct_eV=w, epsilon_actual_eV_A=actual,
                                   epsilon_relative_deviation=(actual-eps)/eps,
                                   W_pred_nominal_eV=p0, W_pred_from_actual_eV=pe,
                                   nominal_work_relative_error=nominal, converted_work_relative_error=converted,
                                   nominal_work_absolute_error_eV=p0-w, converted_work_absolute_error_eV=pe-w).items():
                equal(value, end[f"reported_{key}"])
            ip, ih = (int(t["endpoint_indices"][j]) for t in traces)
            for t, index in zip(traces, (ip, ih)):
                equal(t["time_fs"][index], end["tau_fs"])
            for field in ("coords_A", "velocity_A_per_fs", "E_base_eV", "F_base_eV_A"):
                equal(traces[0][field][ip], end[field])
                equal(traces[1][field][ih], end[f"halfstep_{field}"])
            dx = traces[0]["coords_A"][ip]-traces[1]["coords_A"][ih]
            displacement = end["coords_A"]-z["coords_A"][0]
            norm = np.linalg.norm(displacement)
            along = np.sum(displacement*z["u"])
            controls = dict(step_position_difference_norm_A=np.linalg.norm(dx),
                            step_position_difference_max_atom_A=maxnorm(dx),
                            step_position_relative_to_displacement=np.linalg.norm(dx)/norm,
                            transverse_displacement_A=np.sqrt(max(0, norm**2-along**2)),
                            displacement_turn_deg=np.degrees(np.arccos(np.clip(along/norm, -1, 1))),
                            max_primary_energy_defect_eV=np.abs(traces[0]["energy_defect_eV"][:ip+1]).max(),
                            max_half_energy_defect_eV=np.abs(traces[1]["energy_defect_eV"][:ih+1]).max())
            controls["primary_energy_defect_over_predicted_work"] = controls["max_primary_energy_defect_eV"]/abs(p0)
            for key, value in controls.items():
                equal(value, end[f"reported_{key}"])
            print(f"  budget={eps:.2f}: tau={end['tau_fs']:.8f} fs, epsilon_actual={actual:.10f} eV/A, "
                  f"W={1000*w:.7f} meV, P0={1000*p0:.7f}, Pe={1000*pe:.7f}; "
                  f"nominal={100*nominal:+.6f}%, converted={100*converted:+.6f}% "
                  f"(absolute: {100*abs(nominal):.6f}%, {100*abs(converted):.6f}%)")
        print(f"  MD dt=0.05/0.025 fs: max |energy defect|="
              f"{np.abs(traces[0]['energy_defect_eV']).max():.6e}/"
              f"{np.abs(traces[1]['energy_defect_eV']).max():.6e} eV; both controls verified")


def check_development(root):
    directory = root/"development/Si64_displaced"
    saved = load(directory/"comparisons.json")
    baseline = arrays(directory/"S0_h0p04_A.npz")
    probe_geometry(baseline)
    r0, c0 = response_at(baseline, .04)
    w0 = np.array([work(point(baseline, 0), point(baseline, i)) for i in (1, 2)])
    for key, value in saved["baseline"].items():
        equal(value, c0[key])
    equal(w0, [saved["baseline_work_eV"][k] for k in ("plus", "minus")])
    for name, file, h in (("cutoff", "PBE_70_560_Ry_h0p04_A", .04),
                          ("kpoints", "PBE_k3x3x3_h0p04_A", .04), ("half_step", "S0_h0p02_A", .02)):
        z = arrays(directory/f"{file}.npz")
        probe_geometry(z)
        equal(z["u"], baseline["u"])
        equal(z["coords_A"][0], baseline["coords_A"][0])
        r, c = response_at(z, h)
        w = np.array([work(point(z, 0), point(z, i)) for i in (1, 2)])
        dr = np.linalg.norm(r-r0)/np.linalg.norm(r0)
        dc = abs(c["C_A2_eV"]-c0["C_A2_eV"])/abs(c0["C_A2_eV"])
        dw = abs(w*(.04/h)**2-w0)/abs(w0)
        report = saved["comparisons"][name]
        for key in saved["baseline"]:
            equal(c[key], report[key])
        equal(dr, report["HVP_relative_difference"])
        equal(dc, report["C_relative_difference"])
        equal(w, [report["anchored_work_eV"][k] for k in ("plus", "minus")])
        wk = "scaled_work_relative_difference" if name == "half_step" else "work_relative_difference"
        equal(dw, [report[wk][k] for k in ("plus", "minus")])
        print(f"Development {name}: HVP={100*dr:.5f}%, C={100*dc:.5f}%, max work={100*dw.max():.5f}%")


def main():
    here = Path(__file__).resolve().parent
    default = here if (here/"MANIFEST.sha256").exists() else here.parent/"output/source_data_prl/silicon"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default)
    root = parser.parse_args().data
    integrity(root)
    check_sizes(root)
    check_thermal(root)
    check_development(root)
    check_theory(root)
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print("PASS: 8 geometries / 256 directions; 10 thermal probe labels; 4 independent PBE endpoints; "
          "4 short MD control traces; 11 unique development labels; manifest and numeric-only data.")
    print(f"Archive: {total:,} bytes ({total/1024**2:.3f} MiB); NumPy-only, no inference/DFT.")


def check_theory(root):
    directory = root/"theory"
    cases = load(directory/"finite_n_integrals.json")["cases"]
    tables = []
    for name in ("finite_n_summary.csv", "finite_n_predictions.csv"):
        with (directory/name).open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["case"] in cases
            for key, value in row.items():
                if key in ("case", "model", "metric"):
                    assert re.fullmatch(r"[A-Za-z0-9_]+", value)
                elif value != "":
                    assert np.isfinite(float(value))
        tables.append(rows)
    rows, predictions = tables
    assert len(rows) == 715 and len(predictions) == 23 and len(cases) == 55
    assert len({r["model"] for r in rows}) == 8
    assert sum(r["model"] == "periodic_L3d" for r in predictions) == 7
    by_metric = {(r["case"], r["metric"]): r for r in rows}
    assert len(by_metric) == 715
    for row in rows:
        for key in ("N", "active_N", "samples", "pairing_group"):
            equal(float(row[key]), cases[row["case"]][key])
    for row in predictions:
        cmax = by_metric[row["case"], "Cmax"]
        equal(float(row["mean_MC"]), float(cmax["mean"]))
        equal(float(row["mean_MC_se"]), float(cmax["mean_se"]))
        equal(float(row["CV_MC"]), float(cmax["cv"]))
        equal(float(row["iid_finite_surrogate_mean"])/float(row["mean_MC"])-1,
              float(row["iid_surrogate_rel_error"]))
    print("Theory: 8 model families / 55 cases; all 7 periodic-3D points; 715 metric rows / 23 prediction rows; saved finite-N integrals retained (no MC/quadrature rerun).")


if __name__ == "__main__":
    main()
