"""Four fixed anchored-potential paths, each at two timesteps; no new DFT."""

import datetime as dt
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import read

from pyraimd2.surrogate import CommitteeSurrogate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"
PROJECT = Path("/data/home/df103967/df103967/cloud_projects/pyraimd2")
CHECKPOINT = PROJECT / "experiments/final_campaign_20260905/assets/committee.pt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def norm(a):
    return float(np.linalg.norm(a, axis=1).max())


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Use an allocated compute node"
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    protocol = json.loads((DATA / "dynamics_protocol.json").read_text())
    contract = json.loads((DATA / "dynamics_contract.json").read_text())
    for name, expected in contract["source_files_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in contract["runtime_files_sha256"].items():
        assert sha(Path(name)) == expected, name
    for name, expected in protocol["input_sha256"].items():
        assert sha(DATA / name) == expected, name
    spec = protocol["paths"][index]
    out = DATA / "dynamics" / f"path_{index}"
    out.mkdir(parents=True, exist_ok=True)
    start = {"job_id": os.environ["SLURM_JOB_ID"], "array_index": index,
             "host": socket.gethostname(), "started_unix": time.time(),
             "protocol_sha256": sha(DATA / "dynamics_protocol.json"),
             "contract_sha256": sha(DATA / "dynamics_contract.json"),
             "specification": spec, "reference_calls": 0, "status": "started"}
    with (out / "started.json").open("x") as stream:
        stream.write(json.dumps(start, indent=2) + "\n")
    cutoff = dt.datetime.fromisoformat(protocol["compute_cutoff"]).timestamp()
    inference_count = 0
    try:
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        recipe = {k: state[k] for k in ("n_members", "seed", "perturbation", "epochs", "lr",
                                        "force_weight", "trainable_filters")}
        model = CommitteeSurrogate(model=state["model_specs"], device="cpu",
                                   default_dtype="float64", **recipe)
        model.load_state_dict(state)
        original = read(DATA / spec["initial_state_file"])
        assert len(original) == 474 and not original.constraints
        ref = json.loads((DATA / spec["anchor_reference_file"]).read_text())
        cached = json.loads((DATA / spec["anchor_base_file"]).read_text())
        assert ref["anchor_force_consistency_passed"]
        initial = model.predict(original)
        inference_count += 1
        restore_force_error = norm(initial.forces - np.asarray(cached["forces_ev_a"]))
        restore_energy_error = abs(initial.energy - cached["energy_ev"])
        assert restore_force_error <= 1e-8 and restore_energy_error <= 1e-7
        # The same cached base force and tight reference define the correction
        # in both timestep runs. Neither is updated along either path.
        base0 = np.asarray(cached["forces_ev_a"])
        correction = np.asarray(ref["forces_ev_a"]) - base0
        position0 = original.positions.copy()
        reference0 = np.asarray(ref["forces_ev_a"])
        save(out / "anchor.json", {"correction_ev_A": correction.tolist(),
                                   "reference_energy_ev": ref["energy_ev"],
                                   "restore_force_error_ev_A": restore_force_error,
                                   "restore_energy_error_ev": restore_energy_error,
                                   "initial_kinetic_energy_ev": original.get_kinetic_energy(),
                                   "initial_net_momentum": original.get_momenta().sum(axis=0).tolist()})
        summaries = []
        for name, step_fs in protocol["integration_steps_fs"].items():
            atoms = original.copy()
            mass = atoms.get_masses()[:, None]
            p = atoms.get_momenta()
            f = reference0.copy()
            base_energy = cached["energy_ev"]
            base_force = base0.copy()
            spread = np.asarray(cached["spread_ev_a"])
            step_ase = step_fs * units.fs
            nsteps = round(protocol["duration_fs"] / step_fs)
            h0 = base_energy + float(np.sum(p * p / mass) / 2)
            rows, drift, max_motion = [], [], []
            pathfile = out / f"{name}_states.jsonl"
            with pathfile.open("x") as stream:
                for step in range(nsteps + 1):
                    if time.time() >= cutoff - 60:
                        raise TimeoutError("Campaign compute cutoff reached")
                    if step:
                        p += 0.5 * step_ase * f
                        atoms.positions += step_ase * p / mass
                        pred = model.predict(atoms)
                        inference_count += 1
                        base_force = pred.forces.copy()
                        base_energy = pred.energy
                        spread = pred.uncertainty.copy()
                        f = base_force + correction
                        assert np.isfinite(f).all() and np.isfinite(base_energy)
                        p += 0.5 * step_ase * f
                        atoms.set_momenta(p)
                    displacement = atoms.positions - position0
                    anchored_energy = base_energy - float(np.sum(correction * displacement))
                    kinetic = float(np.sum(p * p / mass) / 2)
                    change = anchored_energy + kinetic - h0
                    rec = {"step": step, "time_fs": step * step_fs,
                           "base_energy_ev": base_energy, "anchored_energy_ev": anchored_energy,
                           "kinetic_energy_ev": kinetic, "anchored_hamiltonian_drift_ev": change,
                           "positions_angstrom": atoms.positions.tolist(), "momenta": p.tolist(),
                           "base_forces_ev_a": base_force.tolist(), "anchored_forces_ev_a": f.tolist(),
                           "spread_ev_a": spread.tolist(),
                           "geometry_sha256": hashlib.sha256(atoms.positions.astype("<f8").tobytes()).hexdigest(),
                           "role": "diagnostic path, prospective admission window not yet assigned"}
                    stream.write(json.dumps(rec, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append((atoms.positions.copy(), p.copy()))
                    drift.append(change)
                    max_motion.append(norm(displacement))
                    if step % 8 == 0 or step == nsteps:
                        print(json.dumps({"path": index, "integration": name, "step": step,
                                          "time_fs": rec["time_fs"], "Hanch_drift_ev": change}), flush=True)
            summaries.append({"name": name, "step_fs": step_fs, "nsteps": nsteps,
                              "max_abs_anchored_H_drift_ev": max(abs(x) for x in drift),
                              "final_anchored_H_drift_ev": drift[-1],
                              "max_atom_displacement_A": max(max_motion), "states_sha256": sha(pathfile)})
            if name == "primary":
                primary = rows
            else:
                ratio = round(protocol["integration_steps_fs"]["primary"] / step_fs)
                assert len(rows[::ratio]) == len(primary)
                summaries[-1]["max_position_difference_at_common_times_A"] = max(
                    norm(a[0] - b[0]) for a, b in zip(primary, rows[::ratio], strict=True))
                summaries[-1]["final_position_difference_A"] = norm(primary[-1][0] - rows[-1][0])
        save(out / "complete.json", dict(start, status="complete", completed_unix=time.time(),
                                         inference_count=inference_count, integrations=summaries))
    except Exception as exc:
        save(out / "failure.json", dict(start, status="failed", failed_unix=time.time(),
                                        inference_count=inference_count, error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
