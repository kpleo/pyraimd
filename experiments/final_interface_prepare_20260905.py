"""Freeze two existing material anchors and four new directions; no force calls."""

import hashlib
import json
from pathlib import Path

import ase.db
import numpy as np
from ase import units
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/final_campaign_20260905/interface"
DB = ROOT / "analysis/execution_20260905/interface/sources/20260905T152542/production/loop.db"
ANCHORS = ((36, (2026090561, 2026090562)), (161, (2026090563, 2026090564)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "development_protocol.json").exists():
    raise RuntimeError("Refuse to replace frozen development inputs")
database = ase.db.connect(str(DB))
stored = list(database.select(run_id="flagship-a-prod"))
manifest, inputs, directions, labels = [], [], [], []
for anchor_step, seeds in ANCHORS:
    row = next(r for r in stored if r.key_value_pairs["step"] == anchor_step)
    assert row.key_value_pairs["route"] == "dft"
    a = row.toatoms()
    assert len(a) == 474 and not a.constraints
    engine = row.data["engine"]
    reference = {"energy": float(engine["energy"]),
                 "forces": np.asarray(engine["forces"]).tolist(),
                 "wall_time_s": float(engine["wall_time_s"])}
    labels.append({"anchor_step": anchor_step, "reference": reference,
                   "positions": a.positions.tolist(), "cell": a.cell.tolist(),
                   "symbols": a.get_chemical_symbols()})
    base = a.copy()
    base.set_momenta(np.zeros((len(base), 3)))
    inputs.append(base)
    manifest.append({"case": len(manifest), "anchor_step": anchor_step,
                     "kind": "tight_anchor", "seed": None, "displacement_A": 0.0})
    for seed in seeds:
        trial = a.copy()
        mass = trial.get_masses()[:, None]
        p = np.random.default_rng(seed).normal(size=(len(trial), 3)) * np.sqrt(mass)
        p -= mass * p.sum(axis=0) / mass.sum()
        trial.set_momenta(p)
        p *= np.sqrt((3 * len(trial) - 3) * units.kB * 600 / (2 * trial.get_kinetic_energy()))
        trial.set_momenta(p)
        velocity = trial.get_velocities()
        direction = velocity / np.linalg.norm(velocity)
        assert np.linalg.norm(p.sum(axis=0)) < 1e-10
        assert abs(np.linalg.norm(direction) - 1) < 1e-12
        path = OUT / f"anchor_{anchor_step}_seed_{seed}.extxyz"
        write(path, trial)
        directions.append({"anchor_step": anchor_step, "seed": seed,
                           "initial_state_file": path.name,
                           "initial_state_sha256": sha(path),
                           "kinetic_energy_eV": trial.get_kinetic_energy(),
                           "velocity_norm_A_per_ASE_time": float(np.linalg.norm(velocity)),
                           "unit_direction": direction.tolist()})
        for displacement in (-0.04, -0.02, 0.02, 0.04):
            probe = base.copy()
            probe.positions += displacement * direction
            inputs.append(probe)
            manifest.append({"case": len(manifest), "anchor_step": anchor_step,
                             "kind": "directional_probe", "seed": seed,
                             "displacement_A": displacement})
write(OUT / "development_inputs.extxyz", inputs)
for name, value in [("development_cases.json", manifest),
                    ("directions.json", directions), ("archived_anchor_labels.json", labels)]:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")
protocol = {
    "purpose": "Material-scale force-growth and residual-work mechanism; development precedes future labels",
    "case_count": len(manifest), "initial_release_cases": [0, 9],
    "anchor_steps": [36, 161], "direction_seeds": [2026090561, 2026090562, 2026090563, 2026090564],
    "initial_kinetic_normalization_K": 600, "degrees_of_freedom": 3 * 474 - 3,
    "probe_displacements_A": [-0.04, -0.02, 0.02, 0.04],
    "reference": {"method": "QE PBE+D3 PAW", "ecutwfc_Ry": 60,
                  "ecutrho_Ry": 600, "kpoints": "Gamma", "nbnd": 942,
                  "smearing": "fd", "degauss_Ry": 0.02, "conv_thr_Ry": 1e-8,
                  "mixing_beta": 0.20, "mixing_ndim": 12, "diago_david_ndim": 8,
                  "diago_full_acc": True, "electron_maxstep": 400,
                  "startpot_chain": False, "per_attempt_timeout_s": 10800},
    "campaign_attempt_ceiling": 64,
    "compute_cutoff": "2026-09-08T20:00:00+08:00",
    "first_gate": "Both tight-anchor SCFs converge with finite complete force blocks; compare against archived forces and quantify the numerical target before probes",
    "anchor_force_consistency_gate_ev_A": 0.025,
    "future_evaluation": "Not yet generated or labelled. Prediction settings and time grid will be frozen after development and before evaluation.",
    "scope": "Selected known geometries, new velocity directions; not equilibrium replicates or a production-speedup benchmark",
    "database_sha256": sha(DB), "preparation_source_sha256": sha(Path(__file__)),
    "input_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
}
(OUT / "development_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
print(json.dumps({"cases": len(manifest), "release": [0, 9], "directions": len(directions)}))
