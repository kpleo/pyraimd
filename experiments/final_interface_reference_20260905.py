"""One isolated, bounded QE reference for the final materials campaign."""

import datetime as dt
import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path

import numpy as np
from ase.io import read

from pyraimd2.engines.qe_engine import QeConfig, QeEngine

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"
CASE = int(os.environ["SLURM_ARRAY_TASK_ID"])
CPUS = int(os.environ["SLURM_CPUS_PER_TASK"])
PROJECT = Path("/data/home/df103967/df103967/cloud_projects/pyraimd2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


protocol = json.loads((DATA / "development_protocol.json").read_text())
contract = json.loads((DATA / "runtime_contract.json").read_text())
cases = json.loads((DATA / "development_cases.json").read_text())
assert 0 <= CASE < len(cases) and cases[CASE]["case"] == CASE
for name, expected in protocol["input_sha256"].items():
    assert sha(DATA / name) == expected, f"Changed input: {name}"
for name, expected in contract["runtime_files_sha256"].items():
    assert sha(Path(name)) == expected, f"Changed runtime: {name}"
for name, expected in contract["source_files_sha256"].items():
    assert sha(ROOT / name) == expected, f"Changed source: {name}"

cutoff = dt.datetime.fromisoformat(protocol["compute_cutoff"]).timestamp()
remaining = cutoff - time.time() - 60
if remaining < 60:
    raise SystemExit("Campaign compute cutoff reached")
folder = DATA / "reference_results" / f"case_{CASE:02d}"
folder.mkdir(parents=True, exist_ok=True)
start = {"case": CASE, "specification": cases[CASE], "status": "started",
         "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
         "cpus": CPUS, "started_unix": time.time(),
         "protocol_sha256": sha(DATA / "development_protocol.json")}
with (folder / "started.json").open("x") as stream:
    stream.write(json.dumps(start, indent=2) + "\n")
slots = DATA / "budget_slots"
slots.mkdir(exist_ok=True)
for slot in range(protocol["campaign_attempt_ceiling"]):
    try:
        with (slots / f"{slot:03d}.json").open("x") as stream:
            stream.write(json.dumps(start) + "\n")
        break
    except FileExistsError:
        continue
else:
    raise SystemExit("Campaign reference-attempt budget exhausted")

cfg = QeConfig(
    pseudo_dir=str(PROJECT / "inputs"),
    pw_cmd=(str(PROJECT / "envs/qe/bin/mpirun"), "-np", str(CPUS),
            "--oversubscribe", str(PROJECT / "envs/qe/bin/pw.x")),
    ecutwfc=60, ecutrho=600, nbnd=942, kpts=None, metallic=True,
    smearing="fd", degauss=0.02, conv_thr=1e-8, mixing_beta=0.20,
    mixing_ndim=12, diago_david_ndim=8, diago_full_acc=True,
    electron_maxstep=400, startpot_file=False,
    timeout_s=min(10800, remaining),
)
try:
    atoms = read(DATA / "development_inputs.extxyz", index=CASE)
    assert len(atoms) == 474 and not atoms.constraints
    result = QeEngine(cfg, run_root=folder / "qe_runs").compute(atoms)
    output_path = folder / "qe_runs/step/pw.out"
    output = output_path.read_text(errors="replace")
    if "JOB DONE." not in output or "convergence has been achieved" not in output:
        raise RuntimeError("Missing complete, converged QE termination")
    if not np.isfinite(result.forces).all() or not np.isfinite(result.energy):
        raise RuntimeError("Nonfinite energy or force")
    accuracy = re.findall(r"estimated scf accuracy\s*<\s*([0-9.EeDd+-]+)\s*Ry", output)
    last_accuracy = float(accuracy[-1].replace("D", "E")) if accuracy else None
    record = dict(start, status="complete", completed_unix=time.time(),
                  energy_ev=result.energy, forces_ev_a=result.forces.tolist(),
                  reference_seconds=result.wall_time_s,
                  estimated_final_scf_accuracy_Ry=last_accuracy,
                  output_sha256=sha(output_path),
                  input_sha256=sha(folder / "qe_runs/step/pw.in"),
                  attempt_slot=slot)
    if cases[CASE]["kind"] == "tight_anchor":
        archived = json.loads((DATA / "archived_anchor_labels.json").read_text())
        old = next(a for a in archived if a["anchor_step"] == cases[CASE]["anchor_step"])
        delta = result.forces - old["reference"]["forces"]
        change = float(np.linalg.norm(delta, axis=1).max())
        record.update(max_force_change_from_archive_ev_A=change,
                      rms_force_change_from_archive_ev_A=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
                      anchor_force_consistency_gate_ev_A=protocol["anchor_force_consistency_gate_ev_A"],
                      anchor_force_consistency_passed=change <= protocol["anchor_force_consistency_gate_ev_A"])
    save(folder / "result.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "forces_ev_a"}), flush=True)
except Exception as exc:
    save(folder / "failure.json", dict(start, status="failed", failed_unix=time.time(),
                                     error=repr(exc), attempt_slot=slot))
    raise
