"""Frozen committee predictions on the 18 predeclared development structures."""

import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from pyraimd2.surrogate import CommitteeSurrogate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"
OUT = DATA / "development_surrogate"
PROJECT = Path("/data/home/df103967/df103967/cloud_projects/pyraimd2")
CHECKPOINT = PROJECT / "experiments/final_campaign_20260905/assets/committee.pt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Run inference on an allocated compute node"
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    contract = json.loads((DATA / "inference_contract.json").read_text())
    protocol = json.loads((DATA / "development_protocol.json").read_text())
    for name, expected in contract["source_files_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in contract["runtime_files_sha256"].items():
        assert sha(Path(name)) == expected, name
    for name, expected in protocol["input_sha256"].items():
        assert sha(DATA / name) == expected, name
    OUT.mkdir(exist_ok=True)
    metadata = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
                "started_unix": time.time(), "checkpoint_sha256": sha(CHECKPOINT),
                "contract_sha256": sha(DATA / "inference_contract.json"),
                "purpose": "No training, propagation, future labels or DFT; fixed development inputs"}
    with (OUT / "started.json").open("x") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    recipe = {k: state[k] for k in ("n_members", "seed", "perturbation", "epochs", "lr",
                                    "force_weight", "trainable_filters")}
    model = CommitteeSurrogate(model=state["model_specs"], device="cpu",
                               default_dtype="float64", **recipe)
    model.load_state_dict(state)
    assert len(state["member_state_dicts"]) == model.n_members == 2
    metadata.update(recipe=recipe, energy_shifts=state["energy_shifts"],
                    model_specs=state["model_specs"], threads=torch.get_num_threads())
    cases = json.loads((DATA / "development_cases.json").read_text())
    frames = read(DATA / "development_inputs.extxyz", index=":")
    assert len(cases) == len(frames) == 18
    for case, atoms in zip(cases, frames, strict=True):
        t0 = time.time()
        pred = model.predict(atoms)
        assert pred.forces.shape == (474, 3) and np.isfinite(pred.forces).all()
        assert np.isfinite(pred.energy) and np.isfinite(pred.uncertainty).all()
        result = {"specification": case, "energy_ev": pred.energy,
                  "forces_ev_a": pred.forces.tolist(),
                  "spread_ev_a": pred.uncertainty.tolist(),
                  "inference_seconds": time.time() - t0}
        save(OUT / f"case_{case['case']:02d}.json", result)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("forces_ev_a", "spread_ev_a")}), flush=True)
    metadata.update(completed_unix=time.time(), case_count=18)
    save(OUT / "complete.json", metadata)


if __name__ == "__main__":
    main()
