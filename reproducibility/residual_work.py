"""Compute signed residual work and accepted-error bounds from supplied inputs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_arrays(filename, required):
    with np.load(filename, allow_pickle=False) as data:
        require(set(required) <= set(data.files), "Required input arrays are missing")
        arrays = {name: np.array(data[name]) for name in data.files}
    for name in required:
        require(np.issubdtype(arrays[name].dtype, np.number), f"{name} must be numeric")
        require(np.isfinite(arrays[name]).all(), f"{name} contains nonfinite values")
    return arrays


def write_json(filename, value):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def norm(values):
    return np.linalg.norm(values, axis=-1).max(axis=-1)


def geometry_digest(positions):
    return hashlib.sha256(np.asarray(positions, dtype="<f8").tobytes()).hexdigest()


def checked_motion(arrays):
    times, positions = arrays["time_fs"], arrays["positions_A"]
    require(times.ndim == 1 and len(times) >= 2, "At least two times are required")
    require(times[0] == 0 and np.all(np.diff(times) > 0), "Times must start at zero and increase")
    require(positions.ndim == 3 and positions.shape[0] == len(times)
            and positions.shape[1] > 0 and positions.shape[2] == 3, "Invalid position shape")
    return times, positions


def analyze(args):
    required = ("time_fs", "positions_A", "base_forces_eV_A", "reference_forces_eV_A",
                "base_energy_eV", "reference_energy_eV")
    arrays = load_arrays(args.input, required)
    times, positions = checked_motion(arrays)
    for name in ("base_forces_eV_A", "reference_forces_eV_A"):
        require(arrays[name].shape == positions.shape, f"Invalid {name} shape")
    for name in ("base_energy_eV", "reference_energy_eV"):
        require(arrays[name].shape == times.shape, f"Invalid {name} shape")
    if args.end is not None:
        require(args.end > 0, "End time must be positive")
        matches = np.flatnonzero(np.isclose(times, args.end, rtol=0, atol=1e-10))
        require(len(matches) == 1, "Requested end time has no unique reference label")
        selected = np.arange(matches[0] + 1)
    else:
        selected = np.arange(len(times))
    if args.spacing is not None:
        require(args.spacing > 0, "Spacing must be positive")
        intervals = times[selected[-1]] / args.spacing
        require(abs(intervals - round(intervals)) < 1e-8, "End time is not a grid multiple")
        selected = []
        for t in np.arange(round(intervals) + 1) * args.spacing:
            match = np.flatnonzero(np.isclose(times, t, rtol=0, atol=1e-10))
            require(len(match) == 1, "A requested quadrature time has no unique reference label")
            selected.append(int(match[0]))
        selected = np.array(selected)
    require(len(selected) >= 2, "At least one propagation interval is required")
    times, positions = times[selected], positions[selected]
    base, reference = arrays["base_forces_eV_A"][selected], arrays["reference_forces_eV_A"][selected]
    correction = reference[0] - base[0]
    residual = base + correction - reference
    displacement = positions - positions[0]
    correction_work = np.einsum("ij,tij->t", correction, displacement)
    delta_base = arrays["base_energy_eV"][selected] - arrays["base_energy_eV"][0]
    delta_reference = arrays["reference_energy_eV"][selected] - arrays["reference_energy_eV"][0]
    work = delta_reference - delta_base + correction_work
    products = 0.5 * (residual[:-1] + residual[1:]) * np.diff(positions, axis=0)
    atomic_cumulative = np.vstack((np.zeros((1, positions.shape[1])),
                                   np.cumsum(products.sum(axis=2), axis=0)))
    quadrature = atomic_cumulative.sum(axis=1)
    error = norm(residual)
    rows = []
    predictions = None
    if args.prediction:
        predictions = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
        require(predictions.get("kind") == "directional_predictions", "Invalid prediction file")
    kinetic = arrays.get("kinetic_energy_eV")
    if kinetic is not None:
        require(kinetic.shape == arrays["time_fs"].shape and np.isfinite(kinetic).all(),
                "Invalid kinetic-energy array")
        kinetic = kinetic[selected] - kinetic[0]
    for i, t in enumerate(times):
        row = {"time_fs": float(t), "maximum_force_error_eV_A": float(error[i]),
               "force_error_vector_rms_eV_A": float(np.sqrt(np.mean(np.sum(residual[i]**2, axis=1)))),
               "endpoint_work_eV": float(work[i]), "force_integral_eV": float(quadrature[i]),
               "integral_minus_endpoint_eV": float(quadrature[i] - work[i]),
               "observed_2W_over_e2_A2_eV": float(2 * work[i] / error[i]**2) if error[i] else None}
        if kinetic is not None:
            row["reference_hamiltonian_change_eV"] = float(delta_reference[i] + kinetic[i])
            row["anchored_hamiltonian_drift_eV"] = float(delta_base[i] + kinetic[i] - correction_work[i])
        if predictions is not None:
            matches = [p for p in predictions["points"] if abs(p["time_fs"] - t) < 1e-10]
            require(len(matches) == 1, "A measurement time has no unique prediction")
            p = matches[0]
            require(p["geometry_sha256"] == geometry_digest(positions[i]), "Prediction and measurement geometries differ")
            row.update(predicted_work_eV=p["predicted_work_eV"],
                       predicted_linear_error_eV_A=p["linear_error_eV_A"],
                       empirical_envelope_eV_A=p["envelope_eV_A"], admitted=p["admitted"])
            w = p["predicted_work_eV"]
            row["relative_work_prediction_error"] = float((work[i] - w) / abs(w)) if w else None
            row["force_budget_satisfied"] = bool(error[i] <= predictions["force_budget_eV_A"])
        rows.append(row)
    last = atomic_cumulative[-1]
    maximum_atom = int(np.argmax(np.linalg.norm(residual[-1], axis=1)))
    result = {"points": rows, "atomic_force_integral_eV": last.tolist(),
              "positive_atomic_work_sum_eV": float(last[last > 0].sum()),
              "negative_atomic_work_sum_eV": float(last[last < 0].sum()),
              "maximum_residual_atom_index": maximum_atom,
              "maximum_residual_atom_work_eV": float(last[maximum_atom]),
              "maximum_residual_atom_absolute_work_rank": int(1 + np.sum(np.abs(last) > abs(last[maximum_atom])))}
    if "group_ids" in arrays:
        groups = arrays["group_ids"]
        require(groups.shape == (positions.shape[1],) and np.issubdtype(groups.dtype, np.integer),
                "group_ids must contain one integer per atom")
        result["group_force_integral_eV"] = {str(g): float(last[groups == g].sum()) for g in np.unique(groups)}
    return result


def verification(args):
    require(all(math.isfinite(v) for v in (args.probability, args.failure_probability, args.tilt))
            and 0 < args.probability < 1 and 0 < args.failure_probability < 1 and args.tilt > 0,
            "Probabilities must be in (0,1) and tilt positive")
    denominator = -math.log1p(args.probability * math.expm1(-args.tilt))
    accepted = detected = 0
    rows = []
    with Path(args.input).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(set(("accepted", "checked", "violation")) <= set(reader.fieldnames or []), "Missing event columns")
        for step, row in enumerate(reader):
            a, z = int(row["accepted"]), int(row["checked"])
            require(a in (0, 1) and z in (0, 1) and z <= a, "Invalid acceptance or check flag")
            require(row["violation"] in ("", "0", "1"), "Invalid violation flag")
            require(not z or row["violation"] != "", "A checked force requires its observed outcome")
            accepted += a
            detected += z * int(row["violation"] or 0)
            bound = min(1.0, (args.tilt * detected + math.log(1 / args.failure_probability))
                        / (accepted * denominator)) if accepted else None
            rows.append({"step": step, "accepted_count": accepted, "detected_violation_count": detected,
                         "accepted_violation_fraction_upper_bound": bound})
    return {"check_probability": args.probability, "failure_probability": args.failure_probability,
            "tilt": args.tilt, "points": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    work = commands.add_parser("analyze", help="Recalculate residuals, endpoint work and force integrals")
    work.add_argument("input")
    work.add_argument("--prediction")
    work.add_argument("--end", type=float, help="Final time in fs")
    work.add_argument("--spacing", type=float, help="Require an exact reference grid, in fs")
    work.add_argument("--output", required=True)
    check = commands.add_parser("verify", help="Compute sequential bounds from an independent check record")
    check.add_argument("input")
    check.add_argument("--probability", type=float, required=True)
    check.add_argument("--failure-probability", type=float, default=0.05)
    check.add_argument("--tilt", type=float, default=math.log(2))
    check.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        write_json(args.output, analyze(args) if args.command == "analyze" else verification(args))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Input or output error: {error}\n")


if __name__ == "__main__":
    main()
