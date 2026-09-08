"""Estimate directional residual response and predict force error and work."""

import argparse
import json
from pathlib import Path

import numpy as np

from residual_work import checked_motion, geometry_digest, load_arrays, norm, require, write_json


def estimate(args):
    names = ("origin_positions_A", "origin_base_forces_eV_A", "origin_reference_forces_eV_A",
             "unit_directions", "probe_steps_A", "plus_displacements_A", "minus_displacements_A",
             "plus_base_forces_eV_A", "minus_base_forces_eV_A",
             "plus_reference_forces_eV_A", "minus_reference_forces_eV_A")
    a = load_arrays(args.input, names)
    origin, u, steps = a["origin_positions_A"], a["unit_directions"], a["probe_steps_A"]
    require(origin.ndim == 2 and origin.shape[0] > 0 and origin.shape[1] == 3, "Invalid origin shape")
    require(u.ndim == 3 and u.shape[0] > 0 and u.shape[1:] == origin.shape, "Invalid direction shape")
    require(np.allclose(np.linalg.norm(u.reshape(len(u), -1), axis=1), 1, rtol=0, atol=1e-8),
            "Directions must have unit full-configuration Euclidean norm")
    require(steps.shape == (2,) and np.all(steps > 0) and steps[0] < steps[1], "Two increasing positive probe steps are required")
    for name in names[1:3]:
        require(a[name].shape == origin.shape, f"Invalid {name} shape")
    expected = (len(u), 2, *origin.shape)
    for name in names[5:]:
        require(a[name].shape == expected, f"Invalid {name} shape")
    ideal = steps[None, :, None, None] * u[:, None]
    require(np.allclose(a["plus_displacements_A"], ideal, rtol=0, atol=1e-7)
            and np.allclose(a["minus_displacements_A"], -ideal, rtol=0, atol=1e-7),
            "Probe displacements must match the signed steps and directions within 1e-7 angstrom")
    correction = a["origin_reference_forces_eV_A"] - a["origin_base_forces_eV_A"]
    plus = a["plus_base_forces_eV_A"] + correction - a["plus_reference_forces_eV_A"]
    minus = a["minus_base_forces_eV_A"] + correction - a["minus_reference_forces_eV_A"]
    q = (plus - minus) / (2 * steps[None, :, None, None])
    qmax = norm(q)
    shared_transverse = float(2 * qmax.max())
    directions = []
    for i in range(len(u)):
        response = q[i, -1]
        largest = float(qmax[i, -1])
        require(largest > 0, "Directional residual response is zero")
        curvature = float(np.sum(u[i] * response))
        squared = float(np.sum(response**2))
        remainder = 0.0
        for displacement, residual in ((a["plus_displacements_A"][i], plus[i]),
                                       (a["minus_displacements_A"][i], minus[i])):
            for d, r in zip(displacement, residual):
                distance2 = float(np.sum(d**2))
                require(distance2 > 0, "A probe displacement is zero")
                alpha = float(np.sum(u[i] * d))
                remainder = max(remainder, float(4 * norm(r - alpha * response) / distance2))
        directions.append({"unit_direction": u[i].tolist(), "response_eV_A2": response.tolist(),
                           "directional_curvature_eV_A2": curvature,
                           "directional_residual_work_coefficient_A2_eV": curvature / largest**2,
                           "response_participation": squared / largest**2,
                           "signed_inverse_curvature_A2_eV": curvature / squared,
                           "response_maximum_eV_A2": largest,
                           "parallel_sensitivity_eV_A2": float(2 * norm(q[i, 0] - q[i, 1])),
                           "transverse_coefficient_eV_A2": shared_transverse,
                           "remainder_coefficient_eV_A3": remainder})
    return {"kind": "directional_response", "origin_positions_A": origin.tolist(),
            "probe_steps_A": steps.tolist(), "directions": directions}


def predict(args):
    require(np.isfinite([args.budget, args.floor, args.time_cap, args.transverse_cap]).all()
            and args.budget > 0 and args.floor >= 0 and args.time_cap > 0
            and 0 <= args.transverse_cap <= 1, "Invalid forecast settings")
    a = load_arrays(args.input, ("time_fs", "positions_A"))
    require(set(a) == {"time_fs", "positions_A"}, "Motion input must contain only times and positions")
    times, positions = checked_motion(a)
    response = json.loads(Path(args.response).read_text(encoding="utf-8"))
    require(response.get("kind") == "directional_response", "Invalid response file")
    require(0 <= args.direction < len(response["directions"]), "Invalid direction index")
    d = response["directions"][args.direction]
    u, q = np.asarray(d["unit_direction"]), np.asarray(d["response_eV_A2"])
    origin = np.asarray(response["origin_positions_A"])
    require(positions.shape[1:] == origin.shape and np.allclose(positions[0], origin, rtol=0, atol=1e-8),
            "Motion and response origins differ")
    active, horizon, points = True, 0.0, []
    for t, position in zip(times, positions):
        displacement = position - positions[0]
        alpha = float(np.sum(u * displacement))
        distance = float(np.linalg.norm(displacement))
        transverse = float(np.linalg.norm(displacement - alpha * u))
        fraction = transverse / distance if distance else 0.0
        linear = float(norm(alpha * q))
        radius = (args.floor + d["parallel_sensitivity_eV_A2"] * abs(alpha)
                  + d["transverse_coefficient_eV_A2"] * transverse
                  + 0.5 * d["remainder_coefficient_eV_A3"] * distance**2)
        domain = bool(t <= args.time_cap and fraction <= args.transverse_cap)
        active = bool(active and domain and linear + radius <= args.budget)
        if active:
            horizon = float(t)
        points.append({"time_fs": float(t), "geometry_sha256": geometry_digest(position),
                       "linear_error_eV_A": linear, "envelope_eV_A": linear + radius,
                       "predicted_work_eV": 0.5 * alpha**2 * d["directional_curvature_eV_A2"],
                       "transverse_fraction": fraction, "in_domain": domain, "admitted": active})
    return {"kind": "directional_predictions", "force_budget_eV_A": args.budget,
            "numerical_floor_eV_A": args.floor, "time_cap_fs": args.time_cap,
            "transverse_fraction_cap": args.transverse_cap, "force_accuracy_horizon_fs": horizon,
            "directional_residual_work_coefficient_A2_eV": d["directional_residual_work_coefficient_A2_eV"],
            "points": points}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("estimate", help="Estimate response using two central-displacement scales")
    probe.add_argument("input")
    probe.add_argument("--output", required=True)
    motion = commands.add_parser("predict", help="Freeze predictions using a trajectory without future reference labels")
    motion.add_argument("input")
    motion.add_argument("--response", required=True)
    motion.add_argument("--direction", type=int, default=0)
    motion.add_argument("--budget", type=float, default=0.25)
    motion.add_argument("--floor", type=float, default=0.00025)
    motion.add_argument("--time-cap", type=float, default=1.0)
    motion.add_argument("--transverse-cap", type=float, default=0.1)
    motion.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        write_json(args.output, estimate(args) if args.command == "estimate" else predict(args))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Input or output error: {error}\n")


if __name__ == "__main__":
    main()
