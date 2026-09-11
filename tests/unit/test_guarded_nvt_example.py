"""The public guarded-NVT example runs fresh and resumes in small CI runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from test_nvt import events

from pyraimd2.store import Store


def test_guarded_nvt_example_new_and_resume(tmp_path):
    example = Path(__file__).parents[2] / "examples" / "guarded_nvt.py"
    out = tmp_path / "run"
    fresh = subprocess.run(
        [sys.executable, str(example), "--output", str(out), "--steps", "6"],
        capture_output=True, text=True, check=False)
    assert fresh.returncode == 0, fresh.stderr[-500:]
    resumed = subprocess.run(
        [sys.executable, str(example), "--output", str(out), "--resume",
         "--extra-steps", "4"],
        capture_output=True, text=True, check=False)
    assert resumed.returncode == 0, resumed.stderr[-500:]
    result = json.loads(resumed.stdout[resumed.stdout.index("{"):])
    assert result["complete_steps"] == 10
    assert result["updates_published"] >= 1
    assert result["reference_calls"]  # purpose-split accounting printed
    assert result["segment_residual_work_eV"]

    # the split example run equals a continuous one bit-for-bit
    control = tmp_path / "control"
    fresh_control = subprocess.run(
        [sys.executable, str(example), "--output", str(control),
         "--steps", "10"],
        capture_output=True, text=True, check=False)
    assert fresh_control.returncode == 0, fresh_control.stderr[-500:]

    def rows(run_dir):
        return sorted(Store(run_dir / "trajectory.db")._db.select(
            run_id="guarded-nvt"),
                      key=lambda row: int(row.key_value_pairs["step"]))

    for row_a, row_b in zip(rows(out), rows(control)):
        np.testing.assert_array_equal(row_a.toatoms().positions,
                                      row_b.toatoms().positions)
        np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                      row_b.toatoms().get_momenta())
    # and the model chains match
    def chain(run_dir):
        return [(e["generation"], e["model_id"], e["origin_label_id"])
                for e in events(run_dir) if e["type"] == "model_update"]

    assert chain(out) == chain(control)
