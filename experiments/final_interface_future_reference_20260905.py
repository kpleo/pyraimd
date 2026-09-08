"""One bounded, non-retriable future QE label; never creates a protocol.

Required future_protocol.json schema (all scientific choices belong to its author):
  status: "frozen" (mandatory; no truthy aliases)
  cases_file: "future_cases.json", a JSON list ordered by case ID, exactly 0..39
  predictions_file: "future_predictions.json", a nonempty JSON list of 40 objects
    ordered by the same case IDs 0..39. Each object must contain "case" plus the
    author's prospective prediction/admission fields; scientific field semantics
    belong to the author. This executor checks case identity only, not those
    scientific fields. Both the manifest AND predictions must be hash protected.
  compute_cutoff: null (paper closure is evidence based, without a calendar cutoff)
  stopping_policy_file: "goal_based_closure.json", included in input_sha256
  per_attempt_timeout_s: 10800 (resource protection for one QE call)
  campaign_attempt_ceiling: 64 (shared with development, including failed attempts)
  spec:
    kind: "primary_future"
    case_count: 40
    path_indices: [0, 1, 2, 3]
    integration_step_fs: dynamics_protocol.integration_steps_fs.primary
    evaluation_times_fs: dynamics_protocol.future_primary_evaluation_times_fs
      (the same ordered ten times; all four paths must cover each time once)
  input_sha256: {canonical DATA-relative file name: lowercase SHA256}; must include
    future_cases.json, future_predictions.json, future_cases_provenance.json, dynamics_protocol.json,
    runtime_contract.json, goal_based_closure.json, all four initial_state_file values and every input in
    future_cases_provenance.input_sha256, with matching digests. Extra frozen
    inputs are allowed and are checked too. The protocol cannot hash itself.
  source_files_sha256: {canonical ROOT-relative file name: lowercase SHA256}; keys
    must be exactly experiments/final_interface_future_reference_20260905.py,
    experiments/final_interface_future_reference_20260905.sbatch, and
    src/pyraimd2/engines/qe_engine.py. Do not copy the development source map.
  runtime_files_sha256: {absolute file name: lowercase SHA256}; exactly the QE
    mpirun, pw.x and eight UPF paths selected from runtime_contract by this script,
    with identical hashes. No model or checkpoint hashes are required or loaded.

Each case requires case (integer), kind="primary_future", path_index (0..3),
anchor_step, seed, step (positive primary integration step), time_fs,
initial_state_file (DATA-relative, matching its dynamics path), geometry_sha256,
and positions_angstrom (474 x 3 finite JSON numbers, in original atom order).
geometry_sha256 = SHA256(np.asarray(positions_angstrom, dtype="<f8").tobytes("C")),
matching the dynamics writer. Extra case/protocol fields are preserved/allowed.
The initial state supplies elements, cell and PBC; its positions are replaced
directly from JSON without wrapping or an intermediate extxyz serialization.
The unchanged QeEngine writes coordinates AND cell with ten decimal places.
Records distinguish the expected coordinate rounding from the actual pw.in
rounding, and retain raw QE SCF, completion, warning and floating-point lines.
Warnings alone do not fail a label; engine failure, incomplete/nonconverged SCF
or nonfinite labels do. Six additional checks are intentionally unsupported.

Preflight rejection creates no output and consumes no slot. After exclusive
started.json creation, a shared budget_slots/000..063.json is claimed with "x";
neither marker is ever deleted, even on failure. Existing output is never reused.
"""

import datetime as dt
import hashlib
import json
import math
import os
import re
import resource
import socket
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"
PROJECT = Path("/data/home/df103967/df103967/cloud_projects/pyraimd2")
SOURCES = {
    "experiments/final_interface_future_reference_20260905.py",
    "experiments/final_interface_future_reference_20260905.sbatch",
    "src/pyraimd2/engines/qe_engine.py",
}
QE_EXECUTABLES = {str(PROJECT / "envs/qe/bin" / name) for name in ("mpirun", "pw.x")}
NCASES, NATOMS, ATTEMPT_CEILING = 40, 474, 64


def require(condition, message):
    # These release gates must also work with python -O.
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relative_file(base, name):
    require(isinstance(name, str) and bool(name), "Expected a relative file name")
    rel = Path(name)
    require(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == name,
            f"Noncanonical relative path: {name}")
    path = base / rel
    require(path.resolve().is_relative_to(base.resolve()), f"Escaping file path: {name}")
    return path


def verify_hashes(mapping, base=None):
    require(isinstance(mapping, dict) and bool(mapping), "Missing SHA256 map")
    for name, expected in mapping.items():
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                f"Invalid SHA256: {name}")
        path = relative_file(base, name) if base is not None else Path(name)
        require(base is not None or path.is_absolute(), f"Runtime path is not absolute: {name}")
        require(sha(path) == expected, f"Changed file: {name}")


def frozen_json(name, hashes):
    """Hash the very bytes being parsed, not a separate earlier read."""
    raw = relative_file(DATA, name).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == hashes[name], f"Changed JSON: {name}")
    return json.loads(raw)


def validate_stopping_policy(protocol, policy):
    require("compute_cutoff" in protocol and protocol["compute_cutoff"] is None,
            "Calendar cutoff was withdrawn; compute_cutoff must be null")
    require(policy["mode"] == "evidence_based" and policy["compute_cutoff"] is None,
            "Expected the user-authorized evidence-based closure policy")
    require(policy["material_case_count"] == {"minimum": 1, "maximum": 2},
            "Paper closure requires one or two material cases")
    require(type(protocol["per_attempt_timeout_s"]) is int
            and protocol["per_attempt_timeout_s"] == policy["per_attempt_timeout_s"] == 10800,
            "Retain the three-hour timeout for each QE attempt")


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_cases(protocol, dynamics, cases, np):
    """Validate the entire frozen 4 x 10 manifest, not just this array element."""
    spec = protocol["spec"]
    require(spec["kind"] == "primary_future" and type(spec["case_count"]) is int
            and spec["case_count"] == NCASES, "Only 40 primary_future labels are supported")
    require(spec["path_indices"] == [0, 1, 2, 3]
            and all(type(i) is int for i in spec["path_indices"]), "Expected paths 0..3")
    times = dynamics["future_primary_evaluation_times_fs"]
    step_fs = dynamics["integration_steps_fs"]["primary"]
    require(finite_number(step_fs) and step_fs > 0, "Invalid primary timestep")
    require(isinstance(times, list) and len(times) == 10
            and all(finite_number(t) and 0 < t <= dynamics["duration_fs"] for t in times)
            and times == sorted(set(times)), "Expected ten distinct increasing future times")
    require(spec["evaluation_times_fs"] == times and spec["integration_step_fs"] == step_fs,
            "Future spec differs from the dynamics time schedule")
    require(dynamics["future_primary_DFT_label_ceiling"] == NCASES,
            "Dynamics label ceiling differs from 40")
    require(len(dynamics["paths"]) == 4, "Expected four dynamics paths")
    paths = {p["path_index"]: p for p in dynamics["paths"]}
    require(set(paths) == {0, 1, 2, 3}, "Invalid dynamics path indices")
    require(isinstance(cases, list) and len(cases) == NCASES, "Expected case 0..39 only")
    seen = set()
    for index, case in enumerate(cases):
        require(type(case["case"]) is int and case["case"] == index
                and case["kind"] == "primary_future", f"Invalid case ID/kind: {index}")
        path_index = case["path_index"]
        require(type(path_index) is int and path_index in paths, f"Invalid path: {index}")
        path = paths[path_index]
        for key in ("initial_state_file", "anchor_step", "seed"):
            require(case[key] == path[key], f"Case {index} differs from dynamics path: {key}")
        require(case["initial_state_file"] in protocol["input_sha256"], "Unhashed initial state")
        relative_file(DATA, case["initial_state_file"])
        t, step = case["time_fs"], case["step"]
        require(finite_number(t) and t in times and type(step) is int and step > 0
                and math.isclose(step * step_fs, t, rel_tol=0, abs_tol=1e-12),
                f"Case {index} is not at a fixed primary evaluation step/time")
        pair = (path_index, t)
        require(pair not in seen, f"Duplicate path/time: {pair}")
        seen.add(pair)
        coordinates = case["positions_angstrom"]
        require(isinstance(coordinates, list) and len(coordinates) == NATOMS
                and all(isinstance(row, list) and len(row) == 3
                        and all(finite_number(x) for x in row) for row in coordinates),
                f"Case {index} needs 474 x 3 finite JSON coordinates")
        positions = np.asarray(coordinates, dtype="<f8")
        require(hashlib.sha256(positions.tobytes(order="C")).hexdigest() == case["geometry_sha256"],
                f"Changed geometry: case {index}")
    require(seen == {(p, t) for p in range(4) for t in times}, "Incomplete future schedule")


def preflight():
    protocol_path = DATA / "future_protocol.json"
    if not protocol_path.is_file():
        raise ValueError("Missing future_protocol.json; no future DFT is authorized")
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    require(protocol.get("status") == "frozen", "future_protocol.json is not frozen")
    require(type(protocol["campaign_attempt_ceiling"]) is int
            and protocol["campaign_attempt_ceiling"] == ATTEMPT_CEILING, "Attempt ceiling must be 64")
    require(protocol["cases_file"] == "future_cases.json", "Expected future_cases.json")
    require(protocol["predictions_file"] == "future_predictions.json", "Expected future_predictions.json")
    require(protocol["stopping_policy_file"] == "goal_based_closure.json",
            "Expected goal_based_closure.json")
    sources, inputs = protocol["source_files_sha256"], protocol["input_sha256"]
    require(set(sources) == SOURCES, "Future source map must contain exactly the two new files and QE engine")
    verify_hashes(sources, ROOT)
    required_inputs = {"future_cases.json", "future_predictions.json", "future_cases_provenance.json",
                       "dynamics_protocol.json", "runtime_contract.json", "goal_based_closure.json"}
    require(required_inputs <= set(inputs) and "future_protocol.json" not in inputs,
            "Missing frozen inputs or self-referential protocol hash")
    verify_hashes(inputs, DATA)
    policy = frozen_json(protocol["stopping_policy_file"], inputs)
    validate_stopping_policy(protocol, policy)
    contract = frozen_json("runtime_contract.json", inputs)
    dynamics = frozen_json("dynamics_protocol.json", inputs)
    cases = frozen_json(protocol["cases_file"], inputs)
    predictions = frozen_json(protocol["predictions_file"], inputs)
    require(isinstance(predictions, list) and len(predictions) == NCASES
            and all(isinstance(row, dict) and type(row.get("case")) is int
                    and row["case"] == index for index, row in enumerate(predictions)),
            "Predictions must contain exactly one corresponding record for each case 0..39")
    provenance = frozen_json("future_cases_provenance.json", inputs)
    require(provenance["case_count"] == NCASES
            and provenance["manifest_sha256"] == inputs[protocol["cases_file"]],
            "Future manifest disagrees with provenance")
    for name, expected in provenance["input_sha256"].items():
        require(inputs.get(name) == expected, f"Unfrozen/changed provenance input: {name}")
    # The original dynamics protocol remains an immutable provenance record.
    # Its historical calendar cutoff is superseded by the frozen closure policy.

    # Import the engine only after its source digest has passed. No model loading.
    import numpy as np
    from ase.io import read

    from pyraimd2.engines.qe_engine import DEFAULT_PSEUDOS, QeConfig, QeEngine

    runtime_names = QE_EXECUTABLES | {str(PROJECT / "inputs" / name)
                                    for name in DEFAULT_PSEUDOS.values()}
    require(len(runtime_names) == 10, "Expected two QE executables and eight UPFs")
    runtime = protocol["runtime_files_sha256"]
    require(set(runtime) == runtime_names, "Runtime map must contain only the QE executables and UPFs")
    for name in runtime_names:
        require(runtime[name] == contract["runtime_files_sha256"][name], f"Changed runtime contract: {name}")
    verify_hashes(runtime)
    validate_cases(protocol, dynamics, cases, np)
    require(os.environ.get("SLURM_JOB_ID"), "Use an allocated compute node")
    case_id, cpus = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_CPUS_PER_TASK"])
    require(0 <= case_id < NCASES and cpus == 96, "Expected case 0..39 and 96 CPUs")
    require(os.environ.get("OMP_NUM_THREADS") == "1", "OMP_NUM_THREADS must be 1")
    case = cases[case_id]
    original = read(relative_file(DATA, case["initial_state_file"]))
    require(len(original) == NATOMS and not original.constraints, "Invalid constrained/size initial state")
    require(np.isfinite(original.cell.array).all() and abs(np.linalg.det(original.cell.array)) >= 1e-8,
            "Invalid initial cell")
    atoms = original.copy()
    atoms.calc = None
    atoms.set_positions(np.asarray(case["positions_angstrom"], dtype="<f8"), apply_constraint=False)
    require(hashlib.sha256(atoms.positions.astype("<f8").tobytes()).hexdigest() == case["geometry_sha256"],
            "JSON positions changed during atom construction")
    rounded = np.array([[float(f"{x:.10f}") for x in row] for row in atoms.positions])
    geometry = {"geometry_sha256": case["geometry_sha256"],
                "initial_state_sha256": inputs[case["initial_state_file"]],
                "chemical_symbols": atoms.get_chemical_symbols(),
                "cell_angstrom": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist(),
                "qe_coordinate_decimal_places": 10,
                "qe_positions_max_rounding_difference_angstrom": float(np.max(np.abs(rounded - atoms.positions))),
                "qe_rounded_geometry_sha256": hashlib.sha256(rounded.astype("<f8").tobytes()).hexdigest()}
    return protocol, hashlib.sha256(raw).hexdigest(), case, cpus, atoms, geometry, QeConfig, QeEngine


def qe_diagnostics(output):
    """Keep all matching raw lines (with line numbers), including on failure."""
    matches = list(re.finditer(r"estimated scf accuracy\s*<\s*(\S+)\s*Ry", output, re.IGNORECASE))
    values = []
    for match in matches:
        try:
            value = float(match[1].replace("D", "E").replace("d", "e"))
        except ValueError:
            value = math.nan
        values.append(value if math.isfinite(value) else None)
    patterns = {
        "scf_lines": r"estimated scf accuracy|convergence.*achieved",
        "completion_lines": r"JOB DONE\.|convergence.*achieved|Maximum CPU time|stopping",
        "warning_error_lines": r"warning|error|convergence\s+NOT\s+achieved|not converged",
        "floating_point_lines": r"floating[- ]point|IEEE_|SIGFPE|forrtl|underflow|overflow|divide.by.zero|invalid.operation|\b(?:nan|inf(?:inity)?)\b",
        "timing_lines": r"\bCPU\b.*\bWALL\b|total cpu time|Parallel version|MPI processes|Threads/MPI process",
    }
    diagnostics = {"job_done": "JOB DONE." in output,
                   "scf_converged": "convergence has been achieved" in output,
                   "scf_not_converged": bool(re.search(r"convergence\s+NOT\s+achieved", output, re.IGNORECASE)),
                   "scf_accuracy_tokens_Ry": [m[1] for m in matches],
                   "scf_accuracy_values_Ry": values,
                   "estimated_final_scf_accuracy_Ry": values[-1] if values else None,
                   "warning_policy": "Report all diagnostics; warnings alone do not fail a label"}
    for key, pattern in patterns.items():
        diagnostics[key] = [{"line": i, "text": line} for i, line in enumerate(output.splitlines(), 1)
                            if re.search(pattern, line, re.IGNORECASE)]
    return diagnostics


def artifacts(folder, atoms):
    import numpy as np

    input_path, output_path = folder / "qe_runs/step/pw.in", folder / "qe_runs/step/pw.out"
    output = output_path.read_text(errors="replace") if output_path.is_file() else ""
    record = {"input_sha256": sha(input_path) if input_path.is_file() else None,
              "output_sha256": sha(output_path) if output_path.is_file() else None,
              "input_file": str(input_path.relative_to(DATA)),
              "output_file": str(output_path.relative_to(DATA)),
              "qe_diagnostics": qe_diagnostics(output),
              "qe_input_positions_max_abs_difference_angstrom": None}
    record["estimated_final_scf_accuracy_Ry"] = record["qe_diagnostics"]["estimated_final_scf_accuracy_Ry"]
    if input_path.is_file():
        try:
            lines = input_path.read_text().splitlines()
            offset = lines.index("ATOMIC_POSITIONS angstrom") + 1
            rows = [line.split() for line in lines[offset:offset + NATOMS]]
            require([row[0] for row in rows] == atoms.get_chemical_symbols(), "QE atom order differs")
            positions = np.array([[float(x) for x in row[1:]] for row in rows])
            require(positions.shape == (NATOMS, 3) and np.isfinite(positions).all(), "Invalid QE positions")
            record["qe_input_positions_max_abs_difference_angstrom"] = float(np.max(np.abs(positions - atoms.positions)))
            record["qe_input_geometry_sha256"] = hashlib.sha256(positions.astype("<f8").tobytes()).hexdigest()
        except (ValueError, IndexError) as exc:
            record["qe_input_geometry_error"] = repr(exc)
    return record


def cpu_usage():
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"process_cpu_seconds": own.ru_utime + own.ru_stime,
            "reaped_children_cpu_seconds": children.ru_utime + children.ru_stime}


def main():
    protocol, protocol_sha, case, cpus, atoms, geometry, QeConfig, QeEngine = preflight()
    folder = DATA / "future_reference_results" / f"case_{case['case']:02d}"
    require(not folder.exists() or not any(folder.iterdir()), "Case output already exists; never retry")
    folder.mkdir(parents=True, exist_ok=True)
    start = {"case": case["case"], "specification": case, "status": "started",
             "job_id": os.environ["SLURM_JOB_ID"], "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
             "host": socket.gethostname(), "cpus": cpus, "omp_num_threads": 1,
             "started_unix": time.time(), "protocol_sha256": protocol_sha,
             "frozen_input_sha256": protocol["input_sha256"],
             "source_files_sha256": protocol["source_files_sha256"],
             "runtime_files_sha256": protocol["runtime_files_sha256"],
             "compute_cutoff": None,
             "stopping_policy_file": protocol["stopping_policy_file"],
             "per_attempt_timeout_s": protocol["per_attempt_timeout_s"], **geometry}
    start["started_utc"] = dt.datetime.fromtimestamp(start["started_unix"], dt.UTC).isoformat()
    with (folder / "started.json").open("x") as stream:
        stream.write(json.dumps(start, indent=2, allow_nan=False) + "\n")
    started_clock, initial_cpu = time.perf_counter(), cpu_usage()
    record, slot, reference_clock = dict(start), None, None
    failure = None
    try:
        slots = DATA / "budget_slots"
        slots.mkdir(exist_ok=True)
        for candidate in range(ATTEMPT_CEILING):
            try:
                with (slots / f"{candidate:03d}.json").open("x") as stream:
                    slot = candidate  # A failed marker write still consumes this slot.
                    stream.write(json.dumps(dict(start, attempt_slot=slot), allow_nan=False) + "\n")
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("Campaign reference-attempt budget exhausted")
        record["attempt_slot"] = slot
        # Recheck after slot acquisition and initial-state reading, just before QE.
        require(sha(DATA / "future_protocol.json") == protocol_sha, "Changed future protocol")
        verify_hashes(protocol["input_sha256"], DATA)
        verify_hashes(protocol["source_files_sha256"], ROOT)
        verify_hashes(protocol["runtime_files_sha256"])
        cfg = QeConfig(
            pseudo_dir=str(PROJECT / "inputs"),
            pw_cmd=(str(PROJECT / "envs/qe/bin/mpirun"), "-np", str(cpus),
                    "--oversubscribe", str(PROJECT / "envs/qe/bin/pw.x")),
            ecutwfc=60, ecutrho=600, nbnd=942, kpts=None, metallic=True,
            smearing="fd", degauss=0.02, conv_thr=1e-8, mixing_beta=0.20,
            mixing_ndim=12, diago_david_ndim=8, diago_full_acc=True,
            electron_maxstep=400, startpot_file=False,
            timeout_s=protocol["per_attempt_timeout_s"],
        )
        record["qe_config"] = asdict(cfg)
        record["reference_started_unix"] = time.time()
        reference_clock = time.perf_counter()
        result = QeEngine(cfg, run_root=folder / "qe_runs").compute(atoms)
        record["reference_seconds"] = result.wall_time_s
        import numpy as np

        audit = artifacts(folder, atoms)
        record.update(audit)
        diag = audit["qe_diagnostics"]
        require(diag["job_done"] and diag["scf_converged"] and not diag["scf_not_converged"],
                "Missing complete, converged QE termination")
        require(result.forces.shape == (NATOMS, 3) and np.isfinite(result.forces).all()
                and np.isfinite(result.energy), "Nonfinite energy/force or invalid force shape")
        require(result.stress is None or (np.asarray(result.stress).shape == (6,)
                and np.isfinite(result.stress).all()), "Nonfinite/invalid stress")
        require(audit.get("qe_input_geometry_sha256") == geometry["qe_rounded_geometry_sha256"],
                "QE input differs from the documented ten-decimal JSON geometry")
        record.update(status="complete", energy_ev=float(result.energy),
                      forces_ev_a=result.forces.tolist(),
                      stress_ev_a3=result.stress.tolist() if result.stress is not None else None)
    except BaseException as exc:
        failure = exc
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        # The shared engine retains pw.out (including stderr) on timeout/failure.
        # Preserve its diagnostics even when parsing or SCF failed.
        if "qe_diagnostics" not in record:
            try:
                record.update(artifacts(folder, atoms))
            except Exception as exc:  # noqa: BLE001 - Retain the original failure if auditing fails.
                record["artifact_audit_error"] = repr(exc)
        finished = time.time()
        elapsed = time.perf_counter() - started_clock
        record.update(attempt_slot=slot, finished_unix=finished,
                      finished_utc=dt.datetime.fromtimestamp(finished, dt.UTC).isoformat(),
                      wall_seconds=elapsed, allocated_cpu_seconds=cpus * elapsed,
                      cpu_accounting_note="Allocated CPU seconds = CPUs x executor wall time; rusage covers this process and reaped children, not Slurm billing. Raw QE CPU/WALL lines are retained.")
        record["failed_unix" if failure else "completed_unix"] = finished
        record.update({key: value - initial_cpu[key] for key, value in cpu_usage().items()})
        if reference_clock is not None:
            record["reference_elapsed_seconds"] = time.perf_counter() - reference_clock
        save(folder / ("failure.json" if failure else "result.json"), record)
        # Full per-case coordinates/forces remain in JSON; never suppress diagnostics.
        print(json.dumps({key: value for key, value in record.items()
                          if key not in {"forces_ev_a", "specification", "chemical_symbols"}},
                         allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
