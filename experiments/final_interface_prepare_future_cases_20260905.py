"""Select the 40 predeclared primary geometries; does not release reference jobs."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((DATA / "dynamics_protocol.json").read_text())
    diagnostics = json.loads((DATA / "dynamics_numerics.json").read_text())
    assert diagnostics["protocol_sha256"] == sha(DATA / "dynamics_protocol.json")
    provenance = {"dynamics_protocol.json": sha(DATA / "dynamics_protocol.json"),
                  "dynamics_numerics.json": sha(DATA / "dynamics_numerics.json")}
    cases = []
    for spec in protocol["paths"]:
        folder = DATA / "dynamics" / f"path_{spec['path_index']}"
        complete = json.loads((folder / "complete.json").read_text())
        assert complete["status"] == "complete" and complete["specification"] == spec
        summary = next(s for s in complete["integrations"] if s["name"] == "primary")
        states = folder / "primary_states.jsonl"
        assert sha(states) == summary["states_sha256"]
        for path in (states, folder / "complete.json", DATA / spec["initial_state_file"]):
            provenance[str(path.relative_to(DATA))] = sha(path)
        rows = [json.loads(line) for line in states.read_text().splitlines()]
        for time_fs in protocol["future_primary_evaluation_times_fs"]:
            row, = [r for r in rows if r["time_fs"] == time_fs]
            x = np.asarray(row["positions_angstrom"], dtype="<f8")
            assert x.shape == (474, 3) and np.isfinite(x).all()
            assert hashlib.sha256(x.tobytes()).hexdigest() == row["geometry_sha256"]
            cases.append({"case": len(cases), "kind": "primary_future",
                          "path_index": spec["path_index"], "anchor_step": spec["anchor_step"],
                          "seed": spec["seed"], "step": row["step"], "time_fs": time_fs,
                          "initial_state_file": spec["initial_state_file"],
                          "geometry_sha256": row["geometry_sha256"],
                          "positions_angstrom": row["positions_angstrom"]})
    assert len(cases) == 40
    manifest = DATA / "future_cases.json"
    with manifest.open("x") as stream:
        stream.write(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    record = {"scope": "Geometry selection only, no DFT release or outcome-dependent selection",
              "reference_calls": 0, "case_count": len(cases), "manifest_sha256": sha(manifest),
              "input_sha256": provenance, "source_sha256": sha(Path(__file__)),
              "release_condition": "Separate future_protocol.json must freeze scientific predictions and numerical settings before any future SCF"}
    with (DATA / "future_cases_provenance.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case_count": len(cases), "manifest_sha256": sha(manifest),
                      "new_reference_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
