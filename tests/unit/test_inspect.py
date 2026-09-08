"""WP02 inspect/summary: structured status, human rendering, CSV, old-format compat."""

from __future__ import annotations

import csv
import io

import ase.db
import numpy as np
import pytest
from ase import Atoms

from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.runtime import format_inspection, inspect_run, summary_csv
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction


class Harmonic:
    def __init__(self, k=0.8):
        self.k = k

    def predict(self, atoms):
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None,
                                   np.full(len(atoms), np.nan))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2):
        self.k = k
        self.attempts = 0
        self.fail_on = set()

    def compute(self, atoms):
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise EngineError("deliberate reference failure")
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2)), -self.k * x, None, 0.0)


def _runner(tmp_path, log, engine=None, **kwargs):
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(tmp_path / "run.db")
    options = {"force_budget": 0.1, "timestep_fs": 0.1, "check_probability": 1,
               "time_cap_fs": 1.0}
    options.update(kwargs)
    return (EnergeticRunner(atoms, Harmonic(), engine or Reference(), store, "run",
                            event_log=log, **options),
            store)


def test_inspect_run_status_and_cost_breakdown(tmp_path):
    with EventLog(tmp_path) as log:
        runner, _store = _runner(tmp_path, log)
        runner.run(2)  # eval 0 initial, evals 1-2 accepted+checked
        info = inspect_run(tmp_path)
    assert info["run_id"] == "run"
    assert info["schema_version"] == 2
    assert info["n_evaluations"] == 3
    assert info["n_complete_steps"] == 2
    assert info["physical_time_fs"] == pytest.approx(0.2)
    assert info["model_id"].endswith("#g0")
    reference = info["cost"]["reference"]
    assert reference["actual_executions"] == reference["successful_executions"]
    assert reference["logical_requests"] == reference["actual_executions"]
    assert reference["cache_hits"] == 0 and reference["failed_attempts"] == 0
    assert reference["actual_executions"] == 7  # 1 anchor + 4 probes + 2 checks
    checks = info["checks"]
    assert checks["independent_checks"] == 2
    assert checks["accepted_count"] == 2 and checks["bound"] == 0.0
    trajectory = info["trajectory"]
    assert trajectory["n_rows"] == 3 and trajectory["n_accepted"] == 2
    assert trajectory["last_energy_eV"] is not None
    assert trajectory["last_temperature_K"] == pytest.approx(0.0, abs=300)
    assert info["wall_time_s"] is not None and info["wall_time_s"] >= 0
    assert info["last_checkpoint"] is None  # WP03
    assert info["failure"] is None
    assert info["events"]["last_seq"] == info["events"]["count"]
    text = format_inspection(info)
    assert "run run" in text and "physical time" in text
    assert "reference (actual)" in text and "cache hits" in text


def test_inspect_reports_failure_reason(tmp_path):
    engine = Reference()
    engine.fail_on = {6}  # eval 1's check fails
    with EventLog(tmp_path) as log:
        runner, _store = _runner(tmp_path, log, engine=engine)
        with pytest.raises(EngineError, match="deliberate"):
            runner.run(1)
        info = inspect_run(tmp_path)
    assert info["failure"] is not None
    assert info["failure"]["status"] == "failed"
    assert "deliberate" in info["failure"]["reason"]
    assert info["cost"]["reference"]["failed_attempts"] == 1


def test_summary_csv_rows_and_old_format_compat(tmp_path):
    with EventLog(tmp_path) as log:
        runner, store = _runner(tmp_path, log)
        runner.run(2)
    csv_text = summary_csv(store, "run")
    parsed = list(csv.DictReader(io.StringIO(csv_text)))
    assert [row["evaluation_id"] for row in parsed] == ["0", "1", "2"]
    assert [row["route"] for row in parsed] == ["dft", "ml", "ml"]
    np.testing.assert_allclose([float(r["physical_time_fs"]) for r in parsed],
                               [0.0, 0.1, 0.2])
    assert parsed[0]["reference_anchor"] == "1" and parsed[0]["reference_probe"] == "4"
    assert parsed[1]["reference_check"] == "1"
    assert float(parsed[1]["max_force_error_eV_A"]) >= 0
    assert parsed[1]["model_id"].endswith("#g0")

    # Old-format rows (no schema_version/metadata): readable and exportable.
    old_db = tmp_path / "old" / "legacy.db"
    old_db.parent.mkdir()
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    raw = ase.db.connect(str(old_db))
    raw.write(atoms, run_id="legacy", step=-1, route="ml",
              data={"reason": "old",
                      "surrogate": {"energy": -1.0, "forces": [[0, 0, 0], [0, 0, 0]],
                                    "stress": None, "uncertainty": [np.nan, np.nan]},
                      "engine": None})
    raw.write(atoms, run_id="legacy", step=0, route="dft",
              data={"reason": "old",
                      "surrogate": None,
                      "engine": {"energy": -2.0, "forces": [[0, 0, 0], [0, 0, 0]],
                                 "stress": None, "wall_time_s": 0.5}})
    legacy = Store(old_db)
    assert "schema_version" not in next(iter(legacy._db.select())).data
    energy, forces = legacy.driving_label("legacy", 0)
    assert energy == -2.0 and forces.shape == (2, 3)
    labels = list(legacy.iter_labels("legacy"))
    assert len(labels) == 1 and labels[0][1].energy == -2.0
    old_csv = list(csv.DictReader(io.StringIO(summary_csv(legacy, "legacy"))))
    assert len(old_csv) == 2 and old_csv[1]["energy_eV"] == "-2.0"
    assert old_csv[1]["max_force_error_eV_A"] == ""  # no metadata: blanks
    old_info = inspect_run(old_db.parent)  # no events.jsonl: db-only status
    assert old_info["run_id"] == "legacy"
    assert old_info["n_evaluations"] == 0  # no committed events recorded
    assert old_info["trajectory"]["n_rows"] == 2
    assert old_info["cost"]["reference"]["actual_executions"] == 0
