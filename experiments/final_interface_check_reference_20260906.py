"""Six immutable reference checks; one SCF per array task, no protocol writer.

Inputs live in analysis/final_campaign_20260905/interface (DATA). Required schema:
check_protocol.json:
  protocol_name: "six_check_reference_20260906_v1", status: "frozen"
  mode: "mainfuture" (the only execution mode; no development/retry bypass)
  cases_file: "check_cases.json"; predictions_file: "future_predictions.json"
  compute_cutoff: null; stopping_policy_file: "goal_based_closure.json"
  per_attempt_timeout_s: 10800; campaign_attempt_ceiling: 64
  selection: {rule: "development_kappa_stability_v1", p: 2, r: 0,
    input_sha256: {DATA-relative development input: SHA256},
    scores: [{path_index: 0..3, kappa_0p02: number|null,
      kappa_0p04: number|null, score: number, nonfinite_disclosure: string}]}
    scores are ordered 0..3; kappa = ||v||_2^2 u.qhat_h, with the original
    displacement rounding. Same-sign finite kappas give
    S=max(0,min(abs(k1),abs(k2))-abs(k1-k2)); otherwise S=0. Null represents
    a nonfinite estimate and requires a nonempty disclosure. Rank within each
    anchor, then between winners, by descending S and ascending path_index.
    Ranking is checked; deriving qhat/kappa from development is the author's job.
  scientific_freeze: {force_budget: ..., admission_windows: ...,
    development_numerical_margins: ..., interpretation_rules: ...}
    Each value must be nonempty; scientific semantics belong to the author.
    Results may validate/lower claims, never refit predictions, c or margins.
  input_sha256: {canonical DATA-relative name: lowercase SHA256}; includes
    check_cases.json, future_predictions.json, dynamics_protocol.json,
    runtime_contract.json, goal_based_closure.json, selection inputs, and for
    each selected path: initial_state_file, anchor_reference_file,
    anchor_base_file, dynamics/path_N/{anchor.json,complete.json}, and each
    used {primary,half_step}_states.jsonl. Extra inputs are also checked.
    Excludes check_protocol.json and future_protocol.json (avoids hash cycles).
  source_files_sha256: {canonical ROOT-relative name: SHA256}; EXACTLY SOURCES
    below, covering this .py/.sbatch, reused future .py, QE and its local import
    chain. No model is loaded. runtime_files_sha256 is exactly the two QE
    executables plus eight UPFs, matching runtime_contract.json.
check_cases.json: {status: "frozen", cases: [six objects ordered case 0..5]}.
  Each case: case, kind, path_index, anchor_step, seed, initial_state_file,
    integration, integration_step_fs, step, time_fs, conv_thr, reference_label,
    positions_angstrom (474x3, unwrapped, original numbering), geometry_sha256,
    source_file (canonical dynamics/path_N/<integration>_states.jsonl),
    source_sha256, momenta_sha256, fixed_c_file (dynamics/path_N/anchor.json),
    fixed_c_sha256 (whole anchor.json, including the unchanged correction).
  Geometry/momenta SHA256 = SHA256(asarray(array,dtype='<f8').tobytes('C')).
  Exact rows (kind, path, integration, time_fs, conv_thr, reference_label):
    0 scf_origin,   p, primary,   0,     1e-10, tightened
    1 scf_endpoint, p, primary,   1,     1e-10, tightened
    2 quadrature,   p, primary,   0.625, 1e-8,  standard
    3 quadrature,   p, primary,   0.875, 1e-8,  standard
    4 timestep,     p, half_step, 1,     1e-8,  standard
    5 timestep,     r, half_step, 1,     1e-8,  standard
  primary dt=0.125 fs; half_step dt=0.0625 fs. Frames are checked against
  hashed, completed original trajectories, including momenta and fixed c.

Freeze order: predictions + check_cases -> check_protocol -> future_protocol.
The frozen future_protocol.json input_sha256 MUST bind check_protocol.json,
check_cases.json and the same future_predictions.json. It can do so using the
existing future executor's extra-input support; do not change that executor.
Supply FUTURE_PROTOCOL_SHA256 externally (e.g. exported at reviewed submission).
mainfuture validates that digest and status, so no self-referential digest is
needed in check_protocol. This proves the six checks were bound before that
future release; it cannot certify the absence of historical/out-of-band jobs.

Preflight writes nothing. Exclusive case directories + shared budget slots
000..063 prevent reuse; a failure/timeout consumes its slot permanently. QE
uses fixed PBE+D3/60:600 Ry/FD free energy, 96 CPUs on 9242, timeout 10800 s.
Raw QE diagnostics and ten-decimal geometry rounding use the future helpers.
This executor produces labels, not Q/W estimates or a scientific success gate.
"""

import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"
SOURCES = {
    "experiments/final_interface_check_reference_20260906.py",
    "experiments/final_interface_check_reference_20260906.sbatch",
    "experiments/final_interface_future_reference_20260905.py",
    "src/pyraimd2/__init__.py",
    "src/pyraimd2/engines/__init__.py",
    "src/pyraimd2/engines/base.py",
    "src/pyraimd2/engines/pyscf_engine.py",
    "src/pyraimd2/engines/qe_engine.py",
}
NCASES = 6


def load_protocol():
    """Verify the reused module's bytes BEFORE executing any of its code."""
    raw = (DATA / "check_protocol.json").read_bytes()
    protocol = json.loads(raw)
    if (protocol.get("status") != "frozen"
            or protocol.get("protocol_name") != "six_check_reference_20260906_v1"
            or protocol.get("mode") != "mainfuture"):
        raise ValueError("Expected frozen six_check_reference_20260906_v1, mode=mainfuture")
    sources = protocol["source_files_sha256"]
    if set(sources) != SOURCES:
        raise ValueError("Source map must cover exactly SOURCES, including all reused local code")
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed source: {name}")
    name = "experiments/final_interface_future_reference_20260905.py"
    spec = importlib.util.spec_from_file_location("_check_future_helpers", ROOT / name)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return protocol, hashlib.sha256(raw).hexdigest(), helper


def selection_paths(protocol, dynamics, h):
    selection = protocol["selection"]
    h.require(selection["rule"] == "development_kappa_stability_v1", "Wrong selection rule")
    evidence = selection["input_sha256"]
    h.require(isinstance(evidence, dict) and evidence, "Missing development selection evidence")
    for name, digest in evidence.items():
        h.require(protocol["input_sha256"].get(name) == digest, "Unfrozen selection evidence")
        h.require(name.startswith(("development_", "reference_results/"))
                  or name == "directions.json", "Selection must use development inputs only")
    rows, computed_scores = selection["scores"], []
    h.require(len(rows) == 4, "Need four development scores, including rejected directions")
    for index, row in enumerate(rows):
        h.require(type(row["path_index"]) is int and row["path_index"] == index, "Score order")
        a, b = row["kappa_0p02"], row["kappa_0p04"]
        h.require(all(v is None or h.finite_number(v) for v in (a, b)), "Use null for nonfinite kappa")
        score = 0.0
        if a is None or b is None:
            h.require(isinstance(row.get("nonfinite_disclosure"), str)
                      and row["nonfinite_disclosure"].strip(), "Disclose nonfinite development estimates")
        elif (a > 0 and b > 0) or (a < 0 and b < 0):
            score = max(0.0, min(abs(a), abs(b)) - abs(a - b))
        h.require(h.finite_number(row["score"])
                  and math.isclose(row["score"], score, rel_tol=1e-12, abs_tol=0), "Changed selection score")
        computed_scores.append(score)
    paths = dynamics["paths"]
    h.require(len(paths) == 4 and all(type(p["path_index"]) is int
              and p["path_index"] == i for i, p in enumerate(paths)), "Expected paths 0..3")
    groups = [[i for i, p in enumerate(paths) if p["anchor_step"] == a] for a in (36, 161)]
    h.require(all(len(g) == 2 for g in groups), "Expected two paths per original anchor")
    def rank(i):
        return -computed_scores[i], i

    p, r = sorted((min(g, key=rank) for g in groups), key=rank)
    h.require(type(selection["p"]) is int and type(selection["r"]) is int
              and (selection["p"], selection["r"]) == (p, r), "Changed p/r or tie break")
    return p, r


def matrix(value, np, h):
    h.require(isinstance(value, list) and len(value) == h.NATOMS
              and all(isinstance(row, list) and len(row) == 3
                      and all(h.finite_number(x) for x in row) for row in value), "Need finite 474x3 array")
    return np.asarray(value, dtype="<f8")


def array_sha(value):
    return hashlib.sha256(value.astype("<f8").tobytes(order="C")).hexdigest()


def validate_cases(protocol, dynamics, manifest, h, np, read):
    """Read-only full six-case/source validation, independently testable offline."""
    h.require(manifest["status"] == "frozen", "check_cases.json is not frozen")
    cases = manifest["cases"]
    h.require(isinstance(cases, list) and len(cases) == NCASES, "Exactly six checks required")
    p, r = selection_paths(protocol, dynamics, h)
    h.require((p, r) == (2, 0), "Final development selection is frozen at p=2, r=0")
    expected = [("scf_origin", p, "primary", 0, 1e-10),
                ("scf_endpoint", p, "primary", 1, 1e-10),
                ("quadrature", p, "primary", 0.625, 1e-8),
                ("quadrature", p, "primary", 0.875, 1e-8),
                ("timestep", p, "half_step", 1, 1e-8),
                ("timestep", r, "half_step", 1, 1e-8)]
    h.require(dynamics["integration_steps_fs"] == {"primary": 0.125, "half_step": 0.0625}, "Changed dt")
    inputs, prepared, cache = protocol["input_sha256"], [], {}
    for index, (case, target) in enumerate(zip(cases, expected, strict=True)):
        h.require(type(case["case"]) is int and case["case"] == index, "Invalid check ID")
        h.require(type(case["path_index"]) is int
                  and tuple(case[k] for k in ("kind", "path_index", "integration", "time_fs", "conv_thr"))
                  == target, f"Changed six-check definition: {index}")
        _kind, path_index, integration, t, threshold = target
        step_fs = dynamics["integration_steps_fs"][integration]
        h.require(case["reference_label"] == ("tightened" if threshold == 1e-10 else "standard")
                  and h.finite_number(case["time_fs"]) and h.finite_number(case["conv_thr"])
                  and h.finite_number(case["integration_step_fs"])
                  and case["integration_step_fs"] == step_fs
                  and type(case["step"]) is int and case["step"] * step_fs == t, "Changed timestep/reference label")
        path = dynamics["paths"][path_index]
        for key in ("initial_state_file", "anchor_step", "seed"):
            h.require(case[key] == path[key], f"Changed path identity: {key}")
        base = f"dynamics/path_{path_index}"
        source, anchor_file = f"{base}/{integration}_states.jsonl", f"{base}/anchor.json"
        h.require(case["source_file"] == source and case["source_sha256"] == inputs[source]
                  and case["fixed_c_file"] == anchor_file and case["fixed_c_sha256"] == inputs[anchor_file],
                  "Changed trajectory/c provenance")
        if path_index not in cache:
            complete = h.frozen_json(f"{base}/complete.json", inputs)
            h.require(complete["status"] == "complete" and complete["specification"] == path
                      and complete["protocol_sha256"] == inputs["dynamics_protocol.json"], "Incomplete/changed trajectory")
            for key in ("initial_state_file", "anchor_reference_file", "anchor_base_file"):
                name = path[key]
                h.require(inputs[name] == dynamics["input_sha256"][name], "Changed original dynamics input")
            original = read(h.relative_file(DATA, path["initial_state_file"]))
            h.require(len(original) == h.NATOMS and not original.constraints
                      and np.isfinite(original.cell.array).all()
                      and abs(np.linalg.det(original.cell.array)) >= 1e-8, "Invalid initial atoms/cell")
            anchor = h.frozen_json(anchor_file, inputs)
            c = matrix(anchor["correction_ev_A"], np, h)
            ref = h.frozen_json(path["anchor_reference_file"], inputs)
            model = h.frozen_json(path["anchor_base_file"], inputs)
            h.require(np.array_equal(c, matrix(ref["forces_ev_a"], np, h) - matrix(model["forces_ev_a"], np, h)),
                      "Fixed c changed from the original reference minus base force")
            cache[path_index] = original, complete, c
        original, complete, c = cache[path_index]
        summaries = [s for s in complete["integrations"] if s["name"] == integration]
        h.require(len(summaries) == 1 and summaries[0]["states_sha256"] == inputs[source]
                  and summaries[0]["step_fs"] == step_fs, "Changed completed integration")
        raw = h.relative_file(DATA, source).read_bytes()
        h.require(hashlib.sha256(raw).hexdigest() == inputs[source], "Changed raw trajectory")
        rows = [json.loads(line) for line in raw.splitlines()]
        h.require(len(rows) == summaries[0]["nsteps"] + 1
                  and all(type(row["step"]) is int and row["step"] == j
                          and row["time_fs"] == j * step_fs for j, row in enumerate(rows)), "Invalid trajectory grid")
        h.require(np.array_equal(matrix(rows[0]["positions_angstrom"], np, h), original.positions)
                  and np.array_equal(matrix(rows[0]["momenta"], np, h), original.get_momenta()), "Changed trajectory origin/momenta")
        row = rows[case["step"]]
        positions = matrix(case["positions_angstrom"], np, h)
        h.require(array_sha(positions) == case["geometry_sha256"] == row["geometry_sha256"]
                  and np.array_equal(positions, matrix(row["positions_angstrom"], np, h)), "Geometry is not the original frame")
        momenta = matrix(row["momenta"], np, h)
        h.require(array_sha(momenta) == case["momenta_sha256"], "Changed original frame momenta")
        h.require(np.allclose(matrix(row["anchored_forces_ev_a"], np, h),
                              matrix(row["base_forces_ev_a"], np, h) + c, rtol=0, atol=1e-10), "Trajectory c changed")
        atoms = original.copy()
        atoms.calc = None
        atoms.set_positions(positions, apply_constraint=False)
        atoms.set_momenta(momenta, apply_constraint=False)
        rounded = np.array([[float(f"{x:.10f}") for x in v] for v in positions])
        geometry = {"geometry_sha256": case["geometry_sha256"],
                    "initial_state_sha256": inputs[path["initial_state_file"]],
                    "chemical_symbols": atoms.get_chemical_symbols(), "cell_angstrom": atoms.cell.array.tolist(),
                    "pbc": atoms.pbc.tolist(), "qe_coordinate_decimal_places": 10,
                    "qe_positions_max_rounding_difference_angstrom": float(np.max(np.abs(rounded - positions))),
                    "qe_rounded_geometry_sha256": array_sha(rounded), "fixed_c_sha256": inputs[anchor_file]}
        prepared.append((case, atoms, geometry))
    return prepared


def validate_release(protocol, protocol_sha, h):
    """Detached future digest permits reciprocal binding without a hash cycle."""
    expected = os.environ.get("FUTURE_PROTOCOL_SHA256")
    h.verify_hashes({"future_protocol.json": expected}, DATA)
    future = h.frozen_json("future_protocol.json", {"future_protocol.json": expected})
    h.require(future["status"] == "frozen", "Future protocol must already be frozen")
    inputs = future["input_sha256"]
    h.require("future_protocol.json" not in inputs, "Self-referential future protocol")
    for name, digest in {"check_protocol.json": protocol_sha,
                         "check_cases.json": protocol["input_sha256"]["check_cases.json"],
                         "future_predictions.json": protocol["input_sha256"]["future_predictions.json"]}.items():
        h.require(inputs.get(name) == digest, f"Future release did not freeze this check input: {name}")
    h.require(future["predictions_file"] == protocol["predictions_file"]
              and future["runtime_files_sha256"] == protocol["runtime_files_sha256"], "Future/check reference mismatch")
    h.verify_hashes(inputs, DATA)
    h.require(set(future["source_files_sha256"]) == h.SOURCES, "Wrong future source contract")
    h.verify_hashes(future["source_files_sha256"], ROOT)
    return expected


def preflight():
    protocol, protocol_sha, h = load_protocol()
    h.require(protocol["cases_file"] == "check_cases.json"
              and protocol["predictions_file"] == "future_predictions.json"
              and protocol["stopping_policy_file"] == "goal_based_closure.json", "Wrong protocol input names")
    h.require(type(protocol["campaign_attempt_ceiling"]) is int
              and protocol["campaign_attempt_ceiling"] == h.ATTEMPT_CEILING, "Budget must remain 64")
    inputs = protocol["input_sha256"]
    h.require({"check_cases.json", "future_predictions.json", "dynamics_protocol.json",
               "runtime_contract.json", "goal_based_closure.json"} <= set(inputs)
              and not {"check_protocol.json", "future_protocol.json"} & set(inputs), "Missing/cyclic frozen inputs")
    h.verify_hashes(inputs, DATA)
    policy = h.frozen_json("goal_based_closure.json", inputs)
    h.validate_stopping_policy(protocol, policy)
    h.require(policy["manuscript_deadline"] is None
              and policy["current_interface_batch_attempt_ceiling"] == 64, "Changed goal policy")
    for key in ("force_budget", "admission_windows", "development_numerical_margins", "interpretation_rules"):
        value = protocol["scientific_freeze"][key]
        h.require(value is not None and value != "" and value != [] and value != {}, f"Missing prospective choice: {key}")
    predictions = h.frozen_json("future_predictions.json", inputs)
    h.require(isinstance(predictions, list) and len(predictions) == 40
              and all(isinstance(row, dict) and type(row.get("case")) is int
                      and row["case"] == i for i, row in enumerate(predictions)), "Need all 40 frozen predictions")
    future_sha = validate_release(protocol, protocol_sha, h)
    # Force imports from the verified checkout, before loading the verified engine.
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from ase.io import read

    from pyraimd2.engines.qe_engine import DEFAULT_PSEUDOS, QeConfig, QeEngine

    runtime = protocol["runtime_files_sha256"]
    names = h.QE_EXECUTABLES | {str(h.PROJECT / "inputs" / name) for name in DEFAULT_PSEUDOS.values()}
    contract = h.frozen_json("runtime_contract.json", inputs)
    h.require(len(names) == 10 and set(runtime) == names, "Expected QE executables and eight UPFs")
    h.require(all(runtime[n] == contract["runtime_files_sha256"][n] for n in names), "Changed runtime contract")
    h.verify_hashes(runtime)
    prepared = validate_cases(protocol, h.frozen_json("dynamics_protocol.json", inputs),
                              h.frozen_json("check_cases.json", inputs), h, np, read)
    h.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "9242"
              and os.environ.get("OMP_NUM_THREADS") == "1", "Need allocated 9242 node and OMP=1")
    case_id, cpus = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_CPUS_PER_TASK"])
    h.require(0 <= case_id < NCASES and cpus == 96, "Expected case 0..5, 96 CPUs")
    return protocol, protocol_sha, future_sha, h, prepared[case_id], cpus, QeConfig, QeEngine


def main():
    protocol, protocol_sha, future_sha, h, (case, atoms, geometry), cpus, QeConfig, QeEngine = preflight()
    folder = DATA / "check_reference_results" / f"case_{case['case']:02d}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()  # Exclusive, even an abandoned empty directory cannot be retried.
    start = {"case": case["case"], "specification": case, "status": "started",
             "job_id": os.environ["SLURM_JOB_ID"], "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
             "host": socket.gethostname(), "cpus": cpus, "omp_num_threads": 1,
             "started_unix": time.time(), "protocol_sha256": protocol_sha,
             "future_protocol_sha256": future_sha, "mode": protocol["mode"],
             "frozen_input_sha256": protocol["input_sha256"], "source_files_sha256": protocol["source_files_sha256"],
             "runtime_files_sha256": protocol["runtime_files_sha256"], "compute_cutoff": None,
             "stopping_policy_file": protocol["stopping_policy_file"], "per_attempt_timeout_s": 10800, **geometry}
    start["started_utc"] = dt.datetime.fromtimestamp(start["started_unix"], dt.UTC).isoformat()
    with (folder / "started.json").open("x") as stream:
        stream.write(json.dumps(start, indent=2, allow_nan=False) + "\n")
    started_clock, initial_cpu = time.perf_counter(), h.cpu_usage()
    record, slot, reference_clock, failure = dict(start), None, None, None
    try:
        slots = DATA / "budget_slots"
        slots.mkdir(exist_ok=True)
        for candidate in range(h.ATTEMPT_CEILING):
            try:
                with (slots / f"{candidate:03d}.json").open("x") as stream:
                    slot = candidate  # Even a failed marker write permanently consumes a slot.
                    stream.write(json.dumps(dict(start, attempt_slot=slot), allow_nan=False) + "\n")
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("Campaign reference-attempt budget exhausted")
        h.require(h.sha(DATA / "check_protocol.json") == protocol_sha, "Changed check protocol")
        h.verify_hashes(protocol["input_sha256"], DATA)
        h.verify_hashes(protocol["source_files_sha256"], ROOT)
        h.verify_hashes(protocol["runtime_files_sha256"])
        h.require(validate_release(protocol, protocol_sha, h) == future_sha, "Changed future release")
        cfg = QeConfig(pseudo_dir=str(h.PROJECT / "inputs"),
                       pw_cmd=(str(h.PROJECT / "envs/qe/bin/mpirun"), "-np", str(cpus),
                               "--oversubscribe", str(h.PROJECT / "envs/qe/bin/pw.x")),
                       ecutwfc=60, ecutrho=600, nbnd=942, kpts=None, metallic=True,
                       smearing="fd", degauss=0.02, conv_thr=case["conv_thr"], mixing_beta=0.20,
                       mixing_ndim=12, diago_david_ndim=8, diago_full_acc=True,
                       electron_maxstep=400, startpot_file=False, timeout_s=10800)
        record["qe_config"] = asdict(cfg)
        record["reference_started_unix"] = time.time()
        reference_clock = time.perf_counter()
        result = QeEngine(cfg, run_root=folder / "qe_runs").compute(atoms)
        record["reference_seconds"] = result.wall_time_s
        import numpy as np

        record.update(h.artifacts(folder, atoms))
        diag = record["qe_diagnostics"]
        h.require(diag["job_done"] and diag["scf_converged"] and not diag["scf_not_converged"], "Incomplete/nonconverged QE")
        h.require(result.forces.shape == (h.NATOMS, 3) and np.isfinite(result.forces).all()
                  and np.isfinite(result.energy), "Invalid energy/forces")
        h.require(result.stress is None or (np.asarray(result.stress).shape == (6,)
                  and np.isfinite(result.stress).all()), "Invalid stress")
        h.require(record.get("qe_input_geometry_sha256") == geometry["qe_rounded_geometry_sha256"], "Changed QE geometry/order")
        record.update(status="complete", energy_ev=float(result.energy), forces_ev_a=result.forces.tolist(),
                      stress_ev_a3=result.stress.tolist() if result.stress is not None else None)
    except BaseException as exc:
        failure = exc
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        if "qe_diagnostics" not in record:
            try:
                record.update(h.artifacts(folder, atoms))
            except Exception as exc:  # noqa: BLE001
                record["artifact_audit_error"] = repr(exc)
        finished, elapsed = time.time(), time.perf_counter() - started_clock
        record.update(attempt_slot=slot, finished_unix=finished,
                      finished_utc=dt.datetime.fromtimestamp(finished, dt.UTC).isoformat(),
                      wall_seconds=elapsed, allocated_cpu_seconds=cpus * elapsed,
                      cpu_accounting_note="CPUs x executor wall time, not Slurm billing; raw QE CPU/WALL retained.")
        record["failed_unix" if failure else "completed_unix"] = finished
        record.update({k: v - initial_cpu[k] for k, v in h.cpu_usage().items()})
        if reference_clock is not None:
            record["reference_elapsed_seconds"] = time.perf_counter() - reference_clock
        with (folder / ("failure.json" if failure else "result.json")).open("x") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in record.items()
                          if k not in {"forces_ev_a", "specification", "chemical_symbols"}}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
