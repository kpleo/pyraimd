"""M4 acceptance: the serial relax → NVT → NVE recipe and the authoritative
completed-state reader.

A: physical state continuity across stages (positions, complete momenta,
cell/pbc, masses, FixAtoms, charges, momenta initialized exactly once).
B: stage stop/resume/provenance — a mid-stage crash resumes in a new
process to the same target; a stage boundary restart runs only the next
stage; a tail-only source is refused; an unconverged relax stops the
chain; a changed started stage config refuses; a call-that-fails sentinel
proves state loading and handoff preparation compute nothing.

Analytic harmonic backends only — zero real DFT budget.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from test_nvt import events

from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store
from pyraimd2.workflows.setup import WorkflowError
from pyraimd2.workflows.stages import (
    RecipeStage,
    load_completed_state,
    run_serial_recipe,
)

_RELAX = """\
schema_version = 1
[run]
id = "h2-relax"
directory = "."
seed = 42
[task]
kind = "relax"
mode = "reference"
[structure]
file = "structure.extxyz"
[dynamics]
ensemble = "nve"
timestep_fs = 0.5
steps = 1
[reference]
backend = "harmonic-reference"
k = 1.0
r0 = 0.9
[relax]
steps = {relax_steps}
{constraints}
[output]
summary_interval_steps = 50
"""

_MD = """\
schema_version = 1
[run]
id = "h2-{name}"
directory = "."
seed = 42
[task]
kind = "md"
mode = "surrogate"
[structure]
file = "initial.traj"
[surrogate]
backend = "harmonic-surrogate"
k = 1.0
r0 = 0.9
bias = 0.05
[dynamics]
{dyn}
[checkpoint]
interval_steps = 4
[output]
trajectory_interval_steps = 1
summary_interval_steps = 5
"""

_NVT_DYN = """\
ensemble = "nvt"
integrator = "langevin"
timestep_fs = 0.5
steps = {nvt_steps}
temperature_K = 300.0
friction_per_fs = 0.01
thermostat_seed = 123
velocity_seed = 7
"""

_NVE_DYN = """\
ensemble = "nve"
timestep_fs = 0.5
steps = 5
temperature_K = 300.0
velocity_seed = 7
"""

_STRUCTURE = "2\nH2\nH 0.82 0.9 0.9\nH 0.99 0.9 0.9\n"


def write_recipe(root: Path, *, relax_steps=50, nvt_steps=6,
                 constraints="", masses=None):
    """Write a three-stage harmonic recipe; returns the stage list."""
    root = Path(root)
    relax_dir = root / "relax"
    relax_dir.mkdir(parents=True, exist_ok=True)
    (relax_dir / "run.toml").write_text(
        _RELAX.format(relax_steps=relax_steps, constraints=constraints))
    (relax_dir / "structure.extxyz").write_text(_STRUCTURE)
    stages = [RecipeStage("relax", relax_dir / "run.toml")]
    for name, dyn, momenta in (("nvt", _NVT_DYN, "initialize"),
                               ("nve", _NVE_DYN, "preserve")):
        stage_dir = root / name
        stage_dir.mkdir(exist_ok=True)
        text = _MD.replace("{name}", name).replace(
            "{dyn}", dyn.format(nvt_steps=nvt_steps))
        (stage_dir / "run.toml").write_text(text)
        stages.append(RecipeStage(name, stage_dir / "run.toml",
                                  momenta=momenta))
    return stages


def _rows(run_dir, run_id):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def _initial_row(run_dir, run_id):
    return _rows(run_dir, run_id)[0].toatoms()


# --- A. physical state continuity -----------------------------------------------


def test_a_three_stage_state_continuity(tmp_path):
    root = tmp_path / "recipe"
    stages = write_recipe(root)
    manifest = run_serial_recipe(root, stages, verbose=False)
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 3

    relax_state = load_completed_state(root / "relax")
    nvt_state = load_completed_state(root / "nvt")
    nve_state = load_completed_state(root / "nve")

    # relax handed over the converged geometry; NVT started there
    assert relax_state.provenance["converged"] is True
    np.testing.assert_array_equal(
        _initial_row(root / "nvt", "h2-nvt").positions,
        relax_state.atoms.positions)
    # NVT initialized momenta exactly once (the manifest says how); NVE
    # continued with the NVT final complete momenta — never rethermalized
    init_record = manifest["stages"][1]["source"]["momenta_initialization"]
    assert init_record["policy"] == "initialize"
    assert init_record["temperature_K"] == 300.0
    assert init_record["velocity_seed"] == 7
    nvt_initial = _initial_row(root / "nvt", "h2-nvt")
    assert np.abs(nvt_initial.get_momenta()).sum() > 0.0
    nve_initial = _initial_row(root / "nve", "h2-nve")
    np.testing.assert_array_equal(nve_initial.positions,
                                  nvt_state.atoms.positions)
    np.testing.assert_array_equal(nve_initial.get_momenta(),
                                  nvt_state.atoms.get_momenta())
    # kinetic energy is continuous across the handoff
    kinetic_nvt_end = float((0.5 * nvt_state.atoms.get_momenta()**2
                             / nvt_state.atoms.get_masses()[:, None]).sum())
    kinetic_nve_start = float((0.5 * nve_initial.get_momenta()**2
                               / nve_initial.get_masses()[:, None]).sum())
    assert kinetic_nve_start == kinetic_nvt_end
    # provenance: NVE names the NVT run as its source with its end time;
    # each stage's own physical time starts from zero
    assert nve_state.provenance["source_run_id"] == "h2-nve"
    source = manifest["stages"][2]["source"]
    assert source["source_run_id"] == "h2-nvt"
    assert source["physical_time_fs"] == pytest.approx(3.0)  # 6 x 0.5 fs
    assert nve_state.provenance["physical_time_fs"] == pytest.approx(2.5)
    assert source["state_digest"]
    # the NVE run has no bath settings and consumed no velocity stream:
    # its initial momenta are the NVT final ones (checked above) and the
    # config validation forbids thermostat fields under NVE
    nve_start = next(e for e in events(root / "nve")
                     if e["type"] == "run_start")
    assert nve_start["streams"]["thermostat_seed"] is None


def test_a_masses_fixatoms_and_initialize_once(tmp_path):
    root = tmp_path / "recipe"
    stages = write_recipe(
        root, constraints="[constraints]\nfix_atoms_indices = [0]\n")
    manifest = run_serial_recipe(root, stages, verbose=False)
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 3
    nvt_state = load_completed_state(root / "nvt")
    nve_initial = _initial_row(root / "nve", "h2-nve")
    # FixAtoms survived both handoffs; the fixed atom never moved and never
    # carried momenta; the free atom's momenta are continuous
    np.testing.assert_array_equal(nve_initial.get_momenta()[0],
                                  np.zeros(3))
    np.testing.assert_array_equal(nve_initial.get_momenta()[1],
                                  nvt_state.atoms.get_momenta()[1])
    np.testing.assert_array_equal(nve_initial.positions[0],
                                  nvt_state.atoms.positions[0])
    # the trajectory's own constraint record agrees on both stages
    for stage, run_id in (("nvt", "h2-nvt"), ("nve", "h2-nve")):
        row = _rows(root / stage, run_id)[0]
        record = (row.data.get("metadata") or {}).get("constraint") or {}
        assert record.get("indices") == [0]
    # one initialization, zero fixed-atom momenta
    nvt_initial = _initial_row(root / "nvt", "h2-nvt")
    assert np.abs(nvt_initial.get_momenta()[1]).sum() > 0.0
    assert np.all(nvt_initial.get_momenta()[0] == 0.0)


def test_a_traj_roundtrip_preserves_the_state(tmp_path):
    from ase.io import read as ase_read

    root = tmp_path / "recipe"
    run_serial_recipe(root, write_recipe(root), verbose=False)
    nvt_state = load_completed_state(root / "nvt")
    back = ase_read(root / "nve" / "initial.traj", format="traj")
    np.testing.assert_array_equal(back.positions, nvt_state.atoms.positions)
    np.testing.assert_array_equal(back.get_momenta(),
                                  nvt_state.atoms.get_momenta())
    np.testing.assert_array_equal(back.get_masses(),
                                  nvt_state.atoms.get_masses())


# --- B. stop, resume and provenance ---------------------------------------------

_B_CHILD = '''
import os
import sys
from pyraimd2.runtime.events import EventLog
from test_recipe import run_serial_recipe, write_recipe

kill_key = os.environ["KILL_KEY"]
original = EventLog.append_once


def patched(self, key, event_type, payload):
    if key == kill_key:
        os._exit(73)
    return original(self, key, event_type, payload)


EventLog.append_once = patched
run_serial_recipe(os.environ["ROOT"],
                  write_recipe(os.environ["ROOT"]), verbose=False)
'''


def test_b_mid_stage_crash_resumes_in_a_new_process(tmp_path):
    control_root = tmp_path / "control"
    run_serial_recipe(control_root, write_recipe(control_root),
                      verbose=False)
    crash_root = tmp_path / "crash"
    write_recipe(crash_root)
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(_B_CHILD))
    # kill the NVT stage at its last step — not a checkpoint
    # multiple (interval 4); the checkpoint at step 4 exists
    env = dict(os.environ, ROOT=str(crash_root), KILL_KEY="step:h2-nvt:5",
               PYTHONPATH=os.pathsep.join(
                   [str(Path(__file__).parents[2] / "src"),
                    str(Path(__file__).parent)]))
    result = subprocess.run([sys.executable, str(child)],
                            env=env, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 73, result.stderr[-500:]
    # the new-process re-invocation resumes NVT 4 -> 6 and runs NVE
    manifest = run_serial_recipe(crash_root, write_recipe(crash_root),
                                 verbose=False)
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 3
    for stage, run_id in (("nvt", "h2-nvt"), ("nve", "h2-nve")):
        rows_a = _rows(crash_root / stage, run_id)
        rows_b = _rows(control_root / stage, run_id)
        assert len(rows_a) == len(rows_b)
        for row_a, row_b in zip(rows_a, rows_b):
            np.testing.assert_array_equal(row_a.toatoms().positions,
                                          row_b.toatoms().positions)
            np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                          row_b.toatoms().get_momenta())
    # the manifest records the resumed stage's full wall time across both
    # invocations, not just the last call
    nvt = manifest["stages"][1]
    assert nvt["wall_time_s"] > 0
    assert nvt["result"]["complete_steps"] == 6


def test_b_restart_at_a_stage_boundary_runs_only_the_next_stage(tmp_path):
    root = tmp_path / "recipe"
    stages = write_recipe(root)
    run_serial_recipe(root, stages[:2], verbose=False)  # relax + nvt only
    relax_events = (root / "relax" / "events.jsonl").read_bytes()
    nvt_events = (root / "nvt" / "events.jsonl").read_bytes()

    import pyraimd2.workflows.stages as stages_module

    calls = []
    real_run = stages_module.run_workflow
    real_resume = stages_module.resume_workflow

    def counting_run(*args, **kwargs):
        calls.append("run")
        return real_run(*args, **kwargs)

    def counting_resume(*args, **kwargs):
        calls.append("resume")
        return real_resume(*args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(stages_module, "run_workflow", counting_run)
    monkey.setattr(stages_module, "resume_workflow", counting_resume)
    try:
        run_serial_recipe(root, stages, verbose=False)  # adds NVE
        run_serial_recipe(root, stages, verbose=False)  # nothing left
    finally:
        monkey.undo()
    assert calls == ["run"]  # exactly one new computation: the NVE stage
    assert (root / "relax" / "events.jsonl").read_bytes() == relax_events
    assert (root / "nvt" / "events.jsonl").read_bytes() == nvt_events


def test_b_tail_only_source_refused_and_partial_boundary_selected(tmp_path):
    # A crashed MD run with a committed tail evaluation but no complete
    # step beyond it: the reader selects the last complete boundary when
    # explicitly allowed, and refuses by default.
    root = tmp_path / "recipe"
    stages = write_recipe(root)
    run_serial_recipe(root, stages[:1], verbose=False)  # relax only

    monkey = pytest.MonkeyPatch()
    real_append_once = EventLog.append_once

    def crash_step(self, key, event_type, payload):
        if key == "step:h2-nvt:3":
            raise RuntimeError("injected crash")
        return real_append_once(self, key, event_type, payload)

    monkey.setattr(EventLog, "append_once", crash_step)
    try:
        with pytest.raises(RuntimeError, match="injected crash"):
            run_serial_recipe(root, stages, verbose=False)
    finally:
        monkey.undo()
    # 3 complete steps, a committed tail evaluation, a failed run end
    with pytest.raises(WorkflowError, match="ended failed"):
        load_completed_state(root / "nvt")
    state = load_completed_state(root / "nvt", require_finished=False)
    assert state.provenance["boundary_step_id"] == 2
    assert state.provenance["finished"] is False
    # and it is exactly the third complete boundary: the tail evaluation
    # never became the state
    from pyraimd2.loop.integrators import state_digest

    step_event = next(e for e in reversed(events(root / "nvt"))
                      if e["type"] == "step_completed")
    assert int(step_event["step_id"]) == 2
    assert state_digest(state.atoms.positions,
                        state.atoms.get_momenta()) == step_event[
                            "state_digest"]


def test_b_unconverged_relax_stops_the_chain(tmp_path):
    root = tmp_path / "recipe"
    stages = write_recipe(root, relax_steps=1)  # cannot converge in 1 step
    with pytest.raises(WorkflowError, match="did not converge"):
        run_serial_recipe(root, stages, verbose=False)
    manifest = json.loads((root / "workflow.json").read_text())
    assert manifest["stages"][0]["status"] == "failed"
    assert not (root / "nvt" / "events.jsonl").exists()


def test_b_changed_started_stage_config_refused(tmp_path):
    root = tmp_path / "recipe"
    stages = write_recipe(root)
    run_serial_recipe(root, stages[:1], verbose=False)  # relax done
    # the NVT stage starts under its 6-step config
    run_serial_recipe(root, stages[:2], verbose=False)
    # changing a started stage's config refuses before any new computation
    nvt_config = root / "nvt" / "run.toml"
    nvt_config.write_text(nvt_config.read_text().replace("steps = 6",
                                                         "steps = 8"))
    stages = [stages[0], RecipeStage("nvt", nvt_config, momenta="initialize"),
              stages[2]]
    with pytest.raises(WorkflowError, match="different config"):
        run_serial_recipe(root, stages, verbose=False)
    # nothing ran: the NVE stage was never created
    assert not (root / "nve" / "events.jsonl").exists()


def test_b_sentinel_proves_preparation_computes_nothing(tmp_path):
    root = tmp_path / "recipe"
    stages = write_recipe(root)
    run_serial_recipe(root, stages[:1], verbose=False)  # relax done

    import pyraimd2.workflows.md as md_module

    class SentinelBackend:
        name = "harmonic-surrogate"
        fingerprint = "sentinel"

        def __init__(self, **kwargs):
            pass

        @property
        def capabilities(self):
            from pyraimd2.surrogate.base import SurrogateCapabilities

            return SurrogateCapabilities()

        def predict(self, atoms):
            raise AssertionError("the handoff touched a live backend")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: SentinelBackend())
    try:
        with pytest.raises(Exception, match="touched a live backend"):
            run_serial_recipe(root, stages, verbose=False)
    finally:
        monkey.undo()
    # preparation itself completed and computed nothing: the materialized
    # initial state exists and names the relax run; the failure is the
    # sentinel's first real compute inside the NVT run
    assert (root / "nvt" / "initial.traj").is_file()
    manifest = json.loads((root / "workflow.json").read_text())
    assert manifest["stages"][0]["status"] == "done"
    assert manifest["stages"][1]["status"] == "running"
    assert manifest["stages"][1]["source"]["source_run_id"] == "h2-relax"
    nvt_events = events(root / "nvt")
    failed = [e for e in nvt_events if e["type"] == "task"
              and e.get("status") == "failed"]
    assert failed and "touched a live backend" in (failed[0]["error"] or "")
