"""Verify saved integration records and prepare geometry-only forecast inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum(a):
    return float(np.linalg.norm(a, axis=1).max())


def main():
    protocol = json.loads((DATA / "dynamics_protocol.json").read_text())
    directions = json.loads((DATA / "directions.json").read_text())
    provenance, results = {}, []
    for spec in protocol["paths"]:
        i = spec["path_index"]
        folder = DATA / "dynamics" / f"path_{i}"
        complete = json.loads((folder / "complete.json").read_text())
        provenance[str(folder.relative_to(ROOT) / "complete.json")] = sha(folder / "complete.json")
        assert complete["status"] == "complete" and complete["reference_calls"] == 0
        assert complete["protocol_sha256"] == sha(DATA / "dynamics_protocol.json")
        assert complete["specification"] == spec and complete["inference_count"] == 97
        anchor = json.loads((folder / "anchor.json").read_text())
        c = np.asarray(anchor["correction_ev_A"])
        atoms = read(DATA / spec["initial_state_file"])
        mass = atoms.get_masses()[:, None]
        x0 = atoms.positions
        p0 = atoms.get_momenta()
        v0_fs = (p0 / mass) * units.fs
        u = np.asarray(next(d for d in directions if d["seed"] == spec["seed"])["unit_direction"])
        integrations = {}
        discrepancies = {"position_update_A": 0.0, "momentum_update": 0.0,
                         "constant_force_correction_ev_A": 0.0, "recorded_energy_ev": 0.0}
        for summary in complete["integrations"]:
            name = summary["name"]
            path = folder / f"{name}_states.jsonl"
            assert sha(path) == summary["states_sha256"]
            provenance[str(path.relative_to(ROOT))] = sha(path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(records) == summary["nsteps"] + 1
            h_fs = protocol["integration_steps_fs"][name]
            h = h_fs * units.fs
            points = []
            arc = 0.0
            for step, row in enumerate(records):
                assert row["step"] == step and row["time_fs"] == step * h_fs
                x, p = np.asarray(row["positions_angstrom"]), np.asarray(row["momenta"])
                fb, f = np.asarray(row["base_forces_ev_a"]), np.asarray(row["anchored_forces_ev_a"])
                assert x.shape == p.shape == fb.shape == f.shape == (474, 3)
                assert all(np.isfinite(v).all() for v in (x, p, fb, f))
                assert hashlib.sha256(x.astype("<f8").tobytes()).hexdigest() == row["geometry_sha256"]
                if step == 0:
                    assert np.array_equal(x, x0) and np.array_equal(p, p0)
                    h0 = row["base_energy_ev"] + float(np.sum(p0 * p0 / mass) / 2)
                else:
                    prev_x = np.asarray(records[step - 1]["positions_angstrom"])
                    prev_p = np.asarray(records[step - 1]["momenta"])
                    prev_f = np.asarray(records[step - 1]["anchored_forces_ev_a"])
                    expected_x = prev_x + h * (prev_p + 0.5 * h * prev_f) / mass
                    expected_p = prev_p + 0.5 * h * (prev_f + f)
                    discrepancies["position_update_A"] = max(discrepancies["position_update_A"], maximum(x - expected_x))
                    discrepancies["momentum_update"] = max(discrepancies["momentum_update"], maximum(p - expected_p))
                    arc += float(np.linalg.norm(x - prev_x))
                discrepancies["constant_force_correction_ev_A"] = max(
                    discrepancies["constant_force_correction_ev_A"], maximum(f - fb - c))
                kinetic = float(np.sum(p * p / mass) / 2)
                anchored = row["base_energy_ev"] - float(np.sum(c * (x - x0)))
                drift = anchored + kinetic - h0
                discrepancies["recorded_energy_ev"] = max(
                    discrepancies["recorded_energy_ev"], abs(kinetic - row["kinetic_energy_ev"]),
                    abs(anchored - row["anchored_energy_ev"]), abs(drift - row["anchored_hamiltonian_drift_ev"]))
                displacement = x - x0
                length = float(np.linalg.norm(displacement))
                parallel = float(np.sum(u * displacement))
                transverse = float(np.linalg.norm(displacement - parallel * u))
                points.append({"time_fs": row["time_fs"], "arc_length_A": arc,
                               "displacement_norm_A": length, "parallel_displacement_A": parallel,
                               "transverse_displacement_A": transverse,
                               "transverse_fraction": transverse / length if length else None,
                               "initial_ballistic_deviation_A": float(np.linalg.norm(displacement - row["time_fs"] * v0_fs)),
                               "anchored_H_drift_ev": drift, "max_atom_displacement_A": maximum(displacement)})
            integrations[name] = {"points": points, "records": records, "summary": summary}
        assert discrepancies["position_update_A"] < 1e-10
        assert discrepancies["momentum_update"] < 1e-9
        assert discrepancies["constant_force_correction_ev_A"] < 1e-10
        assert discrepancies["recorded_energy_ev"] < 1e-8
        primary, half = integrations["primary"], integrations["half_step"]
        primary_drift = max(abs(x["anchored_H_drift_ev"]) for x in primary["points"])
        half_matched = max(abs(x["anchored_H_drift_ev"]) for x in half["points"][::2])
        paired_positions = [maximum(np.asarray(a["positions_angstrom"]) - b["positions_angstrom"])
                            for a, b in zip(primary["records"], half["records"][::2], strict=True)]
        results.append({"specification": spec, "self_consistency_discrepancies": discrepancies,
                        "initial_speed_norm_A_per_fs": float(np.linalg.norm(v0_fs)),
                        "initial_direction_serialization_difference": float(np.linalg.norm(v0_fs / np.linalg.norm(v0_fs) - u)),
                        "max_primary_H_drift_ev": primary_drift,
                        "max_half_H_drift_on_primary_times_ev": half_matched,
                        "half_to_primary_drift_ratio": half_matched / primary_drift if primary_drift else None,
                        "max_position_difference_at_common_times_A": max(paired_positions),
                        "endpoint_position_difference_A": paired_positions[-1],
                        "primary_geometry": primary["points"],
                        "evaluation_geometry": [p for p in primary["points"] if p["time_fs"] in protocol["future_primary_evaluation_times_fs"]]})
    result = {"scope": "Stored trajectory consistency and timestep/geometry diagnostics, no future DFT or force recomputation",
              "new_reference_calls": 0, "source_sha256": sha(Path(__file__)),
              "protocol_sha256": sha(DATA / "dynamics_protocol.json"),
              "input_sha256": provenance, "paths": results}
    output = DATA / "dynamics_numerics.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k not in ("primary_geometry", "evaluation_geometry")}
                      for row in results], indent=2))


if __name__ == "__main__":
    main()
