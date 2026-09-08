"""Release the 16 fixed probes once, only after both tight anchors pass."""

import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/final_campaign_20260905/interface"


def main():
    assert os.environ.get("SLURM_JOB_ID")
    path = DATA / "probe_release.json"
    record = {"gate_job_id": os.environ["SLURM_JOB_ID"],
              "anchor_array_job_id": "7623587", "started_unix": time.time(),
              "status": "checking", "probe_cases": list(range(1, 9)) + list(range(10, 18))}
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    try:
        protocol = json.loads((DATA / "development_protocol.json").read_text())
        cutoff = dt.datetime.fromisoformat(protocol["compute_cutoff"]).timestamp()
        assert cutoff - time.time() > 24 * 3600, "Insufficient remaining campaign time for probe stage"
        anchors = []
        for i in (0, 9):
            r = json.loads((DATA / f"reference_results/case_{i:02d}/result.json").read_text())
            assert r["status"] == "complete" and r["anchor_force_consistency_passed"]
            assert r["max_force_change_from_archive_ev_A"] <= protocol["anchor_force_consistency_gate_ev_A"]
            anchors.append({k: r[k] for k in ("case", "reference_seconds", "max_force_change_from_archive_ev_A")})
        record.update(status="submission_requested", anchors=anchors)
        path.write_text(json.dumps(record, indent=2) + "\n")
        cmd = ["sbatch", "--parsable", "--array=1-8,10-17%2",
               "experiments/final_interface_reference_20260905.sbatch"]
        run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
        record.update(command=cmd, returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
        if run.returncode != 0:
            raise RuntimeError("Probe submission returned nonzero; inspect before any retry")
        match = re.fullmatch(r"(\d+)(?:;[^\s]+)?\s*", run.stdout)
        if not match:
            raise RuntimeError("Uncertain submission response; inspect scheduler before any retry")
        record.update(status="released", probe_array_job_id=match.group(1))
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError, subprocess.TimeoutExpired) as exc:
        record.update(status="held" if record["status"] == "checking" else "submission_needs_inspection",
                      reason=repr(exc))
    record["finished_unix"] = time.time()
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
