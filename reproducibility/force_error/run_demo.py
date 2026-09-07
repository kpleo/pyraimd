"""Recompute and validate compact demonstration results; no network access."""

import argparse
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path


HERE = Path(__file__).resolve().parent


def check(actual, expected, tolerance, prefix=""):
    for key, target in expected.items():
        name = f"{prefix}{key}"
        value = actual[key]
        if isinstance(target, dict):
            check(value, target, tolerance, name + ".")
            continue
        matches = (value == target if isinstance(target, int) else
                   math.isclose(value, target, rel_tol=tolerance["relative"],
                                abs_tol=tolerance["absolute"]))
        if not matches:
            raise ValueError(f"{name}: expected {target}, got {value}")


def run_script(folder, script, output_name, *arguments, stdlib=False):
    output = folder / output_name
    command = [sys.executable, "-B"] + (["-S"] if stdlib else [])
    command += [str(HERE / script), *map(str, arguments), "--output", str(output)]
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"{script} failed: {result.stderr.strip()}")
    return json.loads(output.read_text(encoding="utf-8"))


def controlled_demo(folder, expected):
    result = run_script(folder, "controlled_dynamics.py", "controlled.json", stdlib=True)
    observed = {"trajectory_count": len(result["trajectories"]),
                "fixed_streak_k": result["calibration"]["fixed_k"]}
    for policy in ("class_horizon", "fixed_streak"):
        observed[policy] = {key: result["results"]["heldout"][policy][key]
                            for key in expected["controlled"][policy]}
    check(observed, expected["controlled"], expected["tolerances"])
    return observed


def material_demo(folder, data, expected):
    response = run_script(folder, "directional_response.py", "response.json",
                          "estimate", data / "probes_1.npz")
    prediction = run_script(folder, "directional_response.py", "predictions.json",
                            "predict", data / "motion_2.npz",
                            "--response", folder / "response.json", "--direction", 0,
                            "--budget", 0.25, "--floor", 0.00025,
                            "--time-cap", 1, "--transverse-cap", 0.1)
    work = run_script(folder, "residual_work.py", "work.json",
                      "analyze", data / "labels_2.npz",
                      "--prediction", folder / "predictions.json",
                      "--end", 1, "--spacing", 0.125)
    observed = {
        "directional_residual_work_coefficient_A2_eV":
            response["directions"][0]["directional_residual_work_coefficient_A2_eV"],
        "force_accuracy_horizon_fs": prediction["force_accuracy_horizon_fs"],
        **{key: work["points"][-1][key] for key in
           ("time_fs", "maximum_force_error_eV_A", "endpoint_work_eV", "force_integral_eV")},
    }
    check(observed, expected["interface"], expected["tolerances"])
    carbon = {"status": "unavailable"}
    index = expected["carbon"]["atom_index"]
    if index < len(work["atomic_force_integral_eV"]):
        carbon = {"atom_index": index,
                  "force_integral_eV": work["atomic_force_integral_eV"][index]}
        check(carbon, expected["carbon"], expected["tolerances"])
    return observed, carbon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, metavar="PATH",
                        help="extracted force_error_data folder or its interface subfolder")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        expected = json.loads((HERE / "expected_results.json").read_text(encoding="utf-8"))
        runtime = {"python": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "system": platform.system(),
                   "system_version": platform.mac_ver()[0] or platform.release(),
                   "architecture": platform.machine(), "numpy": None}
        data = args.data.resolve() if args.data is not None else None
        if data is not None:
            if (data / "interface").is_dir():
                data = data / "interface"
            for name in ("probes_1.npz", "motion_2.npz", "labels_2.npz"):
                if not (data / name).is_file():
                    raise ValueError(f"Missing {name}; supply the extracted data folder")
            runtime["numpy"] = version("numpy")
        summary = {"status": "passed", "version": (HERE / "VERSION").read_text().strip(),
                   "runtime": runtime}
        elapsed = {}
        with tempfile.TemporaryDirectory(prefix="force-error-demo-") as temporary:
            folder = Path(temporary)
            stage = time.perf_counter()
            summary["controlled"] = controlled_demo(folder, expected)
            elapsed["controlled"] = round(time.perf_counter() - stage, 3)
            if data is not None:
                stage = time.perf_counter()
                summary["interface"], summary["carbon"] = material_demo(folder, data, expected)
                elapsed["interface"] = round(time.perf_counter() - stage, 3)
            else:
                summary["interface"] = {"status": "not_requested"}
        elapsed["total"] = round(time.perf_counter() - started, 3)
        summary["elapsed_seconds"] = elapsed
        print(json.dumps(summary, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"Demo failed: {error}\n")


if __name__ == "__main__":
    main()
