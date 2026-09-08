"""Compute directional residual growth and curvature work from development only."""

import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read as read_atoms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"


def norm(a):
    return float(np.linalg.norm(a, axis=1).max())


def main():
    paths = []

    def read(path):
        paths.append(path)
        return json.loads(path.read_text())

    cases = read(DATA / "development_cases.json")
    directions = read(DATA / "directions.json")
    geometry_path = DATA / "development_inputs.extxyz"
    frames = read_atoms(geometry_path, index=":")
    paths.append(geometry_path)
    ref, model = {}, {}
    for case in cases:
        i = case["case"]
        ref[i] = read(DATA / f"reference_results/case_{i:02d}/result.json")
        model[i] = read(DATA / f"development_surrogate/case_{i:02d}.json")
        assert ref[i]["status"] == "complete"
        assert ref[i]["specification"] == model[i]["specification"] == case
        for result in (ref[i], model[i]):
            assert np.asarray(result["forces_ev_a"]).shape == (474, 3)
            assert np.isfinite(result["forces_ev_a"]).all()
            assert np.isfinite(result["energy_ev"])
    rows = []
    for direction in directions:
        origin = next(c["case"] for c in cases if c["kind"] == "tight_anchor"
                      and c["anchor_step"] == direction["anchor_step"])
        assert ref[origin]["anchor_force_consistency_passed"]
        probes = {c["displacement_A"]: c["case"] for c in cases
                  if c["seed"] == direction["seed"]}
        u = np.asarray(direction["unit_direction"])
        fref0 = np.asarray(ref[origin]["forces_ev_a"])
        fb0 = np.asarray(model[origin]["forces_ev_a"])
        correction = fref0 - fb0
        delta0 = ref[origin]["energy_ev"] - model[origin]["energy_ev"]
        scales = []
        for h in (0.02, 0.04):
            plus, minus = probes[h], probes[-h]
            rp = np.asarray(model[plus]["forces_ev_a"]) + correction - ref[plus]["forces_ev_a"]
            rm = np.asarray(model[minus]["forces_ev_a"]) + correction - ref[minus]["forces_ev_a"]
            dr = (rp - rm) / (2 * h)
            ep = ref[plus]["energy_ev"] - model[plus]["energy_ev"]
            em = ref[minus]["energy_ev"] - model[minus]["energy_ev"]
            # Use the serialized coordinates actually sent to QE. Their decimal
            # rounding need not preserve perfect plus/minus symmetry.
            dp = frames[plus].positions - frames[origin].positions
            dm = frames[minus].positions - frames[origin].positions
            effective_direction = (dp - dm) / (2 * h)
            force_curvature = float(np.sum(u * dr))
            wp = ep - delta0 + float(np.sum(correction * dp))
            wm = em - delta0 + float(np.sum(correction * dm))
            energy_curvature = (wp + wm) / h**2
            qnorm2 = float(np.sum(dr * dr))
            qmax = norm(dr)
            # These local coefficients follow from the exact-anchor Taylor
            # expansion. Ratios are not evidence of resolved signal when q is
            # small relative to reference noise; that is a separate gate.
            participation = qnorm2 / qmax**2 if qmax > 0 else None
            inverse_curvature = force_curvature / qnorm2 if qnorm2 > 0 else None
            boundary_coefficient = force_curvature / qmax**2 if qmax > 0 else None
            checks = {}
            for name, results, f0 in (("reference", ref, fref0), ("base", model, fb0)):
                slope = (results[plus]["energy_ev"] - results[minus]["energy_ev"]) / (2 * h)
                force_slope = -float(np.sum(f0 * effective_direction))
                checks[name] = {"central_energy_slope_ev_A": slope,
                                "negative_force_projection_ev_A": force_slope,
                                "nominal_negative_force_projection_ev_A": -float(np.sum(f0 * u)),
                                "energy_force_discrepancy_ev_A": slope - force_slope}
            scales.append({"h_A": h, "residual_derivative_ev_A2": dr.tolist(),
                           "max_atom_derivative_ev_A2": qmax,
                           "squared_full_derivative_norm": qnorm2,
                           "response_participation_max_norm": participation,
                           "signed_inverse_curvature_A2_per_ev": inverse_curvature,
                           "leading_2W_over_force_error_squared_A2_per_ev": boundary_coefficient,
                           "force_directional_curvature_ev_A2": force_curvature,
                           "energy_directional_curvature_ev_A2": energy_curvature,
                           "curvature_discrepancy_ev_A2": energy_curvature - force_curvature,
                           "endpoint_curvature_work_plus_ev": wp,
                           "endpoint_curvature_work_minus_ev": wm,
                           "serialized_direction_difference": float(np.linalg.norm(effective_direction - u)),
                           "residual_plus_ev_A": norm(rp), "residual_minus_ev_A": norm(rm),
                           "symmetric_residual_defect_ev_A": norm(rp + rm),
                           "energy_force_checks": checks})
        derivative_difference = norm(np.asarray(scales[0]["residual_derivative_ev_A2"])
                                     - scales[1]["residual_derivative_ev_A2"])
        rows.append({"anchor_step": direction["anchor_step"], "seed": direction["seed"],
                     "scales": scales, "derivative_scale_difference_ev_A2": derivative_difference,
                     "tight_anchor_force_change_ev_A": ref[origin]["max_force_change_from_archive_ev_A"]})
    result = {"purpose": "Development measurements; no future trajectory labels or fitted outcomes",
              "scope": "Local two-scale measurements, not certified future or transverse derivative bounds",
              "growth_prediction": "For local straight displacement s*u, e(s) has leading term |s|*||JR*u||_inf2",
              "work_prediction": "For local straight displacement s*u, curvature work has leading term s^2*(u.JR.u)/2",
              "local_boundary_prediction": "For exact anchor matching, nonzero slope and small force budget, 2W/e^2 tends to (u.JR.u)/||JR.u||_inf2^2 = response_participation * signed_inverse_curvature; not a finite-budget certificate",
              "directions": rows, "input_sha256": {
                  str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = DATA / "development_analysis.json"
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"directions": [{k: v for k, v in row.items() if k != "scales"}
                                    for row in rows], "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
