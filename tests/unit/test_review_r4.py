"""Review R4 regression: force-metric config chain, complete-step trajectory
semantics, plain-resume label restoration and FixAtoms reporting (hermetic).

The counterexamples come from INDEPENDENT_REVIEW_20260909.md §R4: the TOML
force metric never reached the checkpoint, exports carried half-step
momenta, a crashed mid-step evaluation was counted and exported as a
complete step, a stationary plain resume wrote empty labels, and a FixAtoms
relax reported the reaction force on fixed atoms as the convergence metric.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms, units
from ase.io import write as ase_write
from test_review_r1 import resume_world, world

from pyraimd2.config import load_config
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.runtime.events import RUN_SUMMARY, STEP_COMPLETED
from pyraimd2.runtime.inspect import _temperature_K, inspect_run
from pyraimd2.store import Store
from pyraimd2.workflows import export_run, resume_workflow, run_workflow
from pyraimd2.workflows.export import completed_step_ids, frames_from_store
from pyraimd2.workflows.md import _policy_kwargs
from pyraimd2.workflows.templates import HARMONIC_CONFIG

TASK_CONFIG = """\
schema_version = 1

[run]
id = "r4-demo"
directory = "runs/r4-demo"
seed = 42

[task]
kind = "{kind}"
mode = "{mode}"

[structure]
file = "structure.extxyz"

[dynamics]
timestep_fs = 0.5
steps = {steps}

[{backend}]
backend = "harmonic-{backend}"
k = 1.0
r0 = 0.9
{bias}

[relax]
optimizer = "bfgs"
fmax_eV_A = 0.05
steps = 100

[checkpoint]
interval_steps = {checkpoint_interval}
"""


def _write_config(tmp_path, *, kind="md", mode="reference", steps=8,
                  checkpoint_interval=4, positions=((0.9, 0.9, 0.9),),
                  momenta=None, extra=""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    backend = "reference" if mode == "reference" else "surrogate"
    bias = "bias = 0.05" if backend == "surrogate" else ""
    text = TASK_CONFIG.format(kind=kind, mode=mode, backend=backend,
                              bias=bias, steps=steps,
                              checkpoint_interval=checkpoint_interval)
    if mode != "adaptive":
        text += extra
    atoms = Atoms("H" * len(positions), positions=positions)
    if momenta is not None:
        atoms.set_momenta(np.asarray(momenta, dtype=float))
    ase_write(tmp_path / "structure.extxyz", atoms, format="extxyz")
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def _rows(run_dir, run_id="r4-demo"):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def _events(run_dir):
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_force_metric_toml_reaches_runner_checkpoint_and_resume(tmp_path):
    text = HARMONIC_CONFIG.replace("steps = 20", "steps = 4").replace(
        "interval_steps = 5", "interval_steps = 2").replace(
        'directory = "runs/harmonic-demo"', 'directory = "runs/metric"')
    text = text.replace(
        "[policy]\nname = \"energetic\"",
        "[policy]\nname = \"energetic\"\nforce_metric = \"all_atoms_max_atom\"")
    tmp_path.mkdir(parents=True, exist_ok=True)
    from pyraimd2.workflows.templates import HARMONIC_STRUCTURE

    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    path = tmp_path / "run.toml"
    path.write_text(text)
    config = load_config(path)
    assert _policy_kwargs(config)["force_metric"] == "all_atoms_max_atom"

    result = run_workflow(config, verbose=False, handle_sigint=False)
    state = CheckpointManager(result.run_dir).read_latest_valid().state
    assert state["policy"]["force_metric"] == "all_atoms_max_atom"

    resumed = resume_workflow(result.run_dir, 2, verbose=False,
                              handle_sigint=False)
    state = CheckpointManager(resumed.run_dir).read_latest_valid().state
    assert state["policy"]["force_metric"] == "all_atoms_max_atom"


def test_export_frames_carry_complete_step_momenta(tmp_path):
    runner, _, _, _ = world(tmp_path / "mom")
    runner.run(0)
    runner.run(2)
    runner.close()
    run_dir = tmp_path / "mom"
    store = Store(run_dir / "trajectory.db")
    frames = frames_from_store(store, "run", force_source="driving",
                               complete_steps=completed_step_ids(run_dir))
    by_step = {frame.info["step_id"]: frame for frame in frames}
    assert 0 in by_step and 1 in by_step
    for step in (0, 1):
        row = store._row_at_step("run", step)
        recorded = row.toatoms().get_momenta()
        driving = np.asarray(row.data["driving"]["forces"], dtype=float)
        expected = recorded + 0.5 * 0.1 * units.fs * driving
        frame = by_step[step]
        np.testing.assert_allclose(frame.get_momenta(), expected,
                                   rtol=0, atol=1e-12)
        assert frame.info["momenta_source"] == "complete_step_reconstructed"
        # The stored record is never rewritten to hide the phase difference.
        np.testing.assert_allclose(row.toatoms().get_momenta(), recorded,
                                   rtol=0, atol=1e-15)


def test_crashed_mid_step_is_not_counted_or_exported(tmp_path):
    runner, _, _, _ = world(tmp_path / "half")
    runner.run(0)
    calc = runner.calc
    original_emit_once = calc._emit_once

    def fail_boundary(key, event_type, **payload):
        if event_type == STEP_COMPLETED:
            raise RuntimeError("injected crash before the last half-kick")
        return original_emit_once(key, event_type, **payload)

    calc._emit_once = fail_boundary
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(1)  # evaluation committed; the step boundary never was
    del runner

    run_dir = tmp_path / "half"
    completed = [e for e in _events(run_dir) if e["type"] == STEP_COMPLETED]
    assert completed == []
    info = inspect_run(run_dir)
    assert info["n_complete_steps"] == 0
    store = Store(run_dir / "trajectory.db")
    frames = frames_from_store(store, "run", force_source="driving",
                               complete_steps=completed_step_ids(run_dir))
    assert len(frames) == 1  # only the initial evaluation
    assert frames[0].info["step_id"] == -1
    assert frames[0].info["integration_phase"] == "initial_evaluation"


def test_plain_resume_at_stationary_boundary_writes_no_empty_labels(tmp_path):
    for mode in ("reference", "surrogate"):
        stopped = load_config(_write_config(
            tmp_path / f"stopped-{mode}", mode=mode, steps=6,
            momenta=[[0.0, 0.0, 0.0]]))
        run_workflow(stopped, verbose=False, handle_sigint=False)
        result = resume_workflow(stopped.run.directory, 2, verbose=False,
                                 handle_sigint=False)
        assert result.steps_completed == 8
        key = "engine" if mode == "reference" else "surrogate"
        for row in _rows(stopped.run.directory):
            assert row.data.get(key) is not None
            assert row.data.get("driving") is not None
        export_run(stopped.run.directory)  # must not report a corrupt record

        control = load_config(_write_config(
            tmp_path / f"control-{mode}", mode=mode, steps=8,
            momenta=[[0.0, 0.0, 0.0]]))
        run_workflow(control, verbose=False, handle_sigint=False)
        resumed_rows = _rows(stopped.run.directory)
        control_rows = _rows(control.run.directory)
        assert len(resumed_rows) == len(control_rows)
        for a, b in zip(resumed_rows, control_rows):
            np.testing.assert_allclose(
                np.asarray(a.data["driving"]["forces"], dtype=float),
                np.asarray(b.data["driving"]["forces"], dtype=float),
                rtol=0, atol=1e-12)


def test_fixatoms_relax_reports_projected_and_raw_fmax(tmp_path):
    # Atom 0 is fixed far from the minimum (reaction force 1.0 eV/A); the
    # free atom starts already converged (0.02 < fmax = 0.05).
    path = _write_config(tmp_path / "relax", kind="relax", mode="reference",
                         positions=((1.9, 0.9, 0.9), (0.92, 0.9, 0.9)),
                         extra="\n[constraints]\nfix_atoms_indices = [0]\n")
    result = run_workflow(load_config(path), verbose=False)
    summary = [e for e in _events(result.run_dir) if e["type"] == RUN_SUMMARY]
    assert summary[-1]["converged"] is True
    assert summary[-1]["final_fmax_eV_A"] == pytest.approx(0.02)
    assert summary[-1]["raw_all_atom_fmax_eV_A"] == pytest.approx(1.0)

    final = _rows(result.run_dir)[-1]
    driving = np.asarray(final.data["driving"]["forces"], dtype=float)
    raw = np.asarray(final.data["engine"]["forces"], dtype=float)
    np.testing.assert_allclose(driving[0], [0.0, 0.0, 0.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(raw[0], [-1.0, 0.0, 0.0], rtol=0, atol=1e-12)
    constraint = final.data["metadata"]["constraint"]
    assert constraint["n_fixed"] == 1
    np.testing.assert_allclose(constraint["raw_forces_eV_A"], raw,
                               rtol=0, atol=1e-12)


def test_fixatoms_singlepoint_projects_driving_and_keeps_raw(tmp_path):
    path = _write_config(tmp_path / "sp", kind="singlepoint", mode="reference",
                         positions=((1.9, 0.9, 0.9), (0.92, 0.9, 0.9)),
                         extra="\n[constraints]\nfix_atoms_indices = [0]\n")
    result = run_workflow(load_config(path), verbose=False)
    row = _rows(result.run_dir)[-1]
    driving = np.asarray(row.data["driving"]["forces"], dtype=float)
    raw = np.asarray(row.data["engine"]["forces"], dtype=float)
    np.testing.assert_allclose(driving[0], [0.0, 0.0, 0.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(driving[1], raw[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(raw[0], [-1.0, 0.0, 0.0], rtol=0, atol=1e-12)


def test_temperature_divides_by_constrained_dofs(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.set_momenta([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    free = _temperature_K(atoms, n_fixed=0)
    half_fixed = _temperature_K(atoms, n_fixed=1)
    assert half_fixed == pytest.approx(2.0 * free)
    assert _temperature_K(atoms, n_fixed=2) is None


def test_plain_resume_restores_last_label_for_export(tmp_path):
    # Surrogate mode hit the same empty-label path through a cache hit at a
    # stationary boundary; resume_world exercises the energetic twin.
    stopped, _, _, _ = world(tmp_path / "label-restore", stationary=True)
    stopped.run(0)
    stopped.close()
    resumed, _, _ = resume_world(tmp_path / "label-restore")
    resumed.run(2)
    resumed.close()
    for row in _rows(tmp_path / "label-restore", run_id="run"):
        assert row.data.get("driving") is not None
