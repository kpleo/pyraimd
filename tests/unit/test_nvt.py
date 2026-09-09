"""M2 acceptance: plain reference/surrogate Langevin NVT — ASE alignment,
continuous vs resumed, RNG separation, small cases and ensemble
statistics (offline, analytic backends, no real DFT)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms, units
from ase.io import write as ase_write

from pyraimd2.config import load_config
from pyraimd2.store import Store
from pyraimd2.workflows import WorkflowError, resume_workflow, run_workflow
from pyraimd2.workflows.md import _BackendCalculator

NVT_CONFIG = """\
schema_version = 1

[run]
id = "nvt-demo"
directory = "runs/nvt-demo"
seed = 42

[task]
kind = "md"
mode = "{mode}"

[structure]
file = "structure.extxyz"

[dynamics]
ensemble = "nvt"
integrator = "langevin"
timestep_fs = {dt}
steps = {steps}
temperature_K = {temperature}
friction_per_fs = {friction}
thermostat_seed = {thermostat_seed}
velocity_seed = 7

[{backend}]
backend = "harmonic-{backend}"
k = {k}
r0 = 0.9
{bias}

[checkpoint]
interval_steps = {checkpoint_interval}

[output]
trajectory_interval_steps = {trajectory_interval}
summary_interval_steps = {summary_interval}
"""


def write_nvt(tmp_path, *, mode="reference", dt=0.5, steps=8,
              temperature=300.0, friction=0.01, thermostat_seed=123, k=1.0,
              checkpoint_interval=4, positions=((0.85, 0.9, 0.9),
                                                (0.95, 0.9, 0.9)),
              momenta=None, masses=None, trajectory_interval=1,
              summary_interval=5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    backend = "reference" if mode == "reference" else "surrogate"
    bias = "bias = 0.05" if backend == "surrogate" else ""
    text = NVT_CONFIG.format(mode=mode, backend=backend, bias=bias, dt=dt,
                             steps=steps, temperature=temperature,
                             friction=friction,
                             thermostat_seed=thermostat_seed, k=k,
                             checkpoint_interval=checkpoint_interval,
                             trajectory_interval=trajectory_interval,
                             summary_interval=summary_interval)
    atoms = Atoms("H" * len(positions), positions=positions)
    if masses is not None:
        atoms.set_masses(masses)
    if momenta is not None:
        atoms.set_momenta(np.asarray(momenta, dtype=float))
    ase_write(tmp_path / "structure.extxyz", atoms, format="extxyz")
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def rows(run_dir, run_id="nvt-demo"):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def events(run_dir):
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_nvt_matches_direct_ase_langevin_bit_for_bit(tmp_path):
    from pyraimd2.backends import backend_factory

    steps, seed = 8, 123
    config = load_config(write_nvt(tmp_path / "ours", steps=steps,
                                   momenta=[[0.05, 0.02, 0.0],
                                            [-0.03, 0.01, 0.0]]))
    run_workflow(config, verbose=False, handle_sigint=False)

    from ase.md.langevin import Langevin

    engine = backend_factory("harmonic-reference")(k=1.0, r0=0.9)
    atoms = Atoms("H2", positions=[[0.85, 0.9, 0.9], [0.95, 0.9, 0.9]])
    atoms.set_momenta([[0.05, 0.02, 0.0], [-0.03, 0.01, 0.0]])
    atoms.calc = _BackendCalculator(engine, "reference")
    atoms.get_forces()  # prime the calculator cache like the driver does
    dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300.0,
                   friction=0.01 / units.fs, fixcm=False,
                   rng=np.random.default_rng(seed))
    for _ in range(steps):
        dyn.step(atoms.calc.results["forces"])

    final = rows(config.run.directory)[-1].toatoms()
    np.testing.assert_allclose(final.positions, atoms.positions,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(final.get_momenta(), atoms.get_momenta(),
                               rtol=0, atol=1e-12)
    assert (config.run.directory / "trajectory.extxyz").is_file()


def test_nvt_continuous_matches_7_plus_13_resumed(tmp_path):
    continuous = load_config(write_nvt(tmp_path / "cont", steps=20))
    run_workflow(continuous, verbose=False, handle_sigint=False)
    stopped = load_config(write_nvt(tmp_path / "stop", steps=7))
    run_workflow(stopped, verbose=False, handle_sigint=False)
    result = resume_workflow(stopped.run.directory, 13, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 20

    cont_rows = rows(continuous.run.directory)
    stop_rows = rows(stopped.run.directory)
    assert len(cont_rows) == len(stop_rows) == 21
    for a, b in zip(cont_rows, stop_rows):
        np.testing.assert_allclose(a.toatoms().positions,
                                   b.toatoms().positions, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.toatoms().get_momenta(),
                                   b.toatoms().get_momenta(), rtol=0,
                                   atol=1e-12)
    cont_tasks = [e for e in events(continuous.run.directory)
                  if e["type"] == "task"]
    stop_tasks = [e for e in events(stopped.run.directory)
                  if e["type"] == "task"]
    # Same evaluation sequence and the same real cost entries.
    assert [(t["evaluation_id"], t["status"]) for t in cont_tasks] == \
           [(t["evaluation_id"], t["status"]) for t in stop_tasks]


def test_nvt_resume_after_unclean_stop_keeps_the_bath_stream(tmp_path):
    # Unclean stop after 7 steps (resume with force_unlock): the trajectory
    # must continue the same bath draws — no re-seed, no re-thermalize.
    run_dir = tmp_path / "crash"
    config = load_config(write_nvt(run_dir, steps=7))
    from pyraimd2.workflows.md import _run_plain
    from pyraimd2.workflows.setup import load_structure

    atoms = load_structure(config)
    _run_plain(config, atoms, config.run.directory,
               verbose=False, handle_sigint=False)
    result = resume_workflow(config.run.directory, 13, verbose=False,
                             handle_sigint=False, force_unlock=True)
    assert result.steps_completed == 20

    continuous = load_config(write_nvt(tmp_path / "crash-control", steps=20))
    run_workflow(continuous, verbose=False, handle_sigint=False)
    for a, b in zip(rows(config.run.directory),
                    rows(continuous.run.directory)):
        np.testing.assert_allclose(a.toatoms().positions,
                                   b.toatoms().positions, rtol=0, atol=1e-12)


def test_nvt_keeps_given_momenta_and_thermalizes_only_when_missing(tmp_path):
    given = [[0.11, 0.0, 0.0], [-0.11, 0.0, 0.0]]
    config = load_config(write_nvt(tmp_path / "momenta", steps=2,
                                   momenta=given))
    run_workflow(config, verbose=False, handle_sigint=False)
    initial = rows(config.run.directory)[0].toatoms()
    np.testing.assert_allclose(initial.get_momenta(), given,
                               rtol=0, atol=1e-15)

    config = load_config(write_nvt(tmp_path / "thermal", steps=2))
    run_workflow(config, verbose=False, handle_sigint=False)
    initial = rows(config.run.directory)[0].toatoms()
    assert np.abs(initial.get_momenta()).sum() > 0.0


def test_nvt_zero_temperature_bath_only_damps(tmp_path):
    # At the equilibrium geometry the force vanishes: only friction acts on
    # the given momenta (no random kicks at T=0).
    config = load_config(write_nvt(
        tmp_path / "cold", steps=5, temperature=0.0,
        positions=((0.9, 0.9, 0.9), (0.9, 0.9, 0.9)),
        momenta=[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]))
    run_workflow(config, verbose=False, handle_sigint=False)
    speed = [float(np.abs(r.toatoms().get_momenta()).sum())
             for r in rows(config.run.directory)]
    assert speed[-1] < speed[0]  # no random kicks at T=0: only damping


def test_nvt_fixatoms_layer_stays_frozen_and_free_dof_temperature(tmp_path):
    from pyraimd2.runtime.inspect import inspect_run

    path = write_nvt(tmp_path / "fix", steps=6)
    text = path.read_text() + "\n[constraints]\nfix_atoms_indices = [0]\n"
    path.write_text(text)
    config = load_config(path)
    run_workflow(config, verbose=False, handle_sigint=False)
    run_rows = rows(config.run.directory)
    first = run_rows[0].toatoms().positions[0].copy()
    for row in run_rows:
        atoms = row.toatoms()
        np.testing.assert_array_equal(atoms.positions[0], first)
        np.testing.assert_allclose(atoms.get_momenta()[0], [0, 0, 0],
                                   rtol=0, atol=1e-15)
    assert np.abs(run_rows[-1].toatoms().get_momenta()[1]).sum() > 0.0
    info = inspect_run(config.run.directory)
    assert info["trajectory"]["last_temperature_K"] is not None


def test_nvt_template_runs_out_of_the_box(tmp_path):
    from pyraimd2.workflows.templates import write_template

    path = write_template("harmonic-nvt", tmp_path)
    config = load_config(path)
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 20
    assert (config.run.directory / "trajectory.extxyz").is_file()


def test_nvt_resume_rejects_changed_thermostat_settings(tmp_path):
    config = load_config(write_nvt(tmp_path / "ident", steps=4))
    run_workflow(config, verbose=False, handle_sigint=False)
    # Resume reads the run's resolved config; changing it there means a
    # different integrator identity than the checkpoint recorded.
    resolved = json.loads(
        (config.run.directory / "resolved_config.json").read_text())
    resolved["dynamics"]["friction_per_fs"] = 0.02
    (config.run.directory / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2))
    with pytest.raises(WorkflowError, match="integrator|integration"):
        resume_workflow(config.run.directory, 2, verbose=False,
                        handle_sigint=False, force_unlock=True)
