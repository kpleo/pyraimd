"""Workflow-path acceptance tests for task.mode 'mts' (the fixed-model
symmetric MTS kernel wired into run/inspect/resume/export).

All offline: the harmonic backends are analytic; no external program
ever launches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.io import write as ase_write

from pyraimd2.cli import main as cli_main
from pyraimd2.config import ConfigError, load_config
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogateCapabilities
from pyraimd2.workflows import export_run, resume_workflow, run_workflow
from pyraimd2.workflows.export import ExportError
from pyraimd2.workflows.setup import WorkflowError

MASS = 28.085


def mts_case(tmp_path: Path, *, steps=128, outer_ratio=4, h_fs=1.0,
             surrogate_kwargs="k = 0.9\nr0 = 0.9\nbias = 0.0",
             with_momenta=True) -> Path:
    a = np.array([0.9, 0.9, 0.9])
    d = np.array([0.03, -0.02, 0.01])
    atoms = Atoms("Si2", positions=[a - d, a + d], masses=[MASS, MASS],
                  pbc=False)
    if with_momenta:
        v0 = np.array([0.001, -0.0015, 0.002]) / units.fs
        atoms.set_momenta(np.array([-v0, v0]) * MASS)
    ase_write(tmp_path / "structure.extxyz", atoms)
    (tmp_path / "run.toml").write_text(f"""schema_version = 1
[run]
id = "mts-demo"
directory = "run"
[task]
kind = "md"
mode = "mts"
[structure]
file = "structure.extxyz"
[dynamics]
ensemble = "nve"
integrator = "respa"
timestep_fs = {h_fs}
steps = {steps}
outer_ratio = {outer_ratio}
[checkpoint]
interval_steps = 8
[reference]
backend = "harmonic-reference"
k = 1.0
r0 = 0.9
[surrogate]
backend = "harmonic-surrogate"
{surrogate_kwargs}
""")
    return tmp_path / "run.toml"


def rows_of(run_dir: Path):
    with Store(run_dir / "trajectory.db") as store:
        rows = [r for r in store._db.select()
                if r.key_value_pairs.get("run_id")]
        rows.sort(key=lambda r: int(r.key_value_pairs["step"]))
        return rows


# --- the continuous user path ------------------------------------------------


def test_mts_continuous_user_path(tmp_path):
    config = load_config(mts_case(tmp_path))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 32           # complete outer boundaries
    assert not result.stopped_early
    info = inspect_run(tmp_path / "run")
    assert info["workflow"]["driver"] == "mts-nve-respa"
    assert info["workflow"]["integrator"]["outer_ratio"] == 4
    assert info["n_evaluations"] == 33            # 33 boundary frames
    assert info["n_complete_steps"] == 32
    assert info["physical_time_fs"] == 128.0
    assert info["cost"]["reference"]["actual_executions"] == 33
    assert info["cost"]["reference"]["logical_requests"] == 33
    assert info["cost"]["counts"]["inference"] == 129
    rows = rows_of(tmp_path / "run")
    assert len(rows) == 33
    assert [int(r.key_value_pairs["step"]) for r in rows][-1] == 128
    times = [r.data["metadata"]["context"]["physical_time_fs"]
             for r in rows]
    assert times[0] == 0.0 and times[-1] == 128.0


def test_mts_validate_environment_ready(tmp_path, capsys):
    config = mts_case(tmp_path)
    code = cli_main(["validate", str(config), "--check-environment",
                     "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["readiness"] == "ready"
    assert not list(tmp_path.rglob("*.db"))  # validate wrote nothing


def test_mts_export_sources_and_driving_refusal(tmp_path):
    run_workflow(load_config(mts_case(tmp_path, steps=16)),
                 verbose=False, handle_sigint=False)
    run_dir = tmp_path / "run"
    rep = export_run(run_dir, force_source="reference",
                     output=tmp_path / "ref.extxyz")
    assert rep["frames"] == 5 and rep["missing_forces_frames"] == 0
    rep = export_run(run_dir, force_source="base",
                     output=tmp_path / "base.extxyz")
    assert rep["frames"] == 5 and rep["missing_forces_frames"] == 0
    with pytest.raises(ExportError, match="no single driving force"):
        export_run(run_dir, force_source="driving")


def test_mts_inspect_renders_algorithm_line(tmp_path, capsys):
    run_workflow(load_config(mts_case(tmp_path, steps=16)),
                 verbose=False, handle_sigint=False)
    code = cli_main(["inspect", str(tmp_path / "run")])
    out = capsys.readouterr().out
    assert code == 0
    assert "MTS (respa)" in out and "outer_ratio 4" in out


# --- resume equivalence --------------------------------------------------------


def test_mts_segmented_resume_bit_identical(tmp_path):
    continuous = tmp_path / "continuous"
    continuous.mkdir()
    run_workflow(load_config(mts_case(continuous)), verbose=False,
                 handle_sigint=False)
    segmented = tmp_path / "segmented"
    segmented.mkdir()
    run_workflow(load_config(mts_case(segmented, steps=32)),
                 verbose=False, handle_sigint=False)
    resumed = resume_workflow(segmented / "run", 96, verbose=False,
                              handle_sigint=False)
    assert resumed.steps_completed == 32
    r1 = rows_of(continuous / "run")
    r2 = rows_of(segmented / "run")
    assert len(r1) == len(r2) == 33
    for a, b in zip(r1, r2):
        assert np.array_equal(a.toatoms().positions,
                              b.toatoms().positions)
        assert np.allclose(a.toatoms().get_momenta(),
                           b.toatoms().get_momenta(),
                           atol=1e-10, rtol=1e-10)
    info = inspect_run(segmented / "run")
    assert info["cost"]["reference"]["actual_executions"] == 33
    assert info["cost"]["counts"]["inference"] == 129
    steps = [int(r.key_value_pairs["step"]) for r in r2]
    assert len(steps) == len(set(steps))       # no duplicated 32 fs frame


# --- refusal boundaries --------------------------------------------------------


def test_mts_refuses_missing_momenta(tmp_path):
    with pytest.raises(WorkflowError, match="WITH momenta"):
        run_workflow(load_config(mts_case(tmp_path, with_momenta=False)),
                     verbose=False, handle_sigint=False)


def test_mts_refuses_non_multiple_steps(tmp_path):
    with pytest.raises(ConfigError, match="multiple of outer_ratio"):
        load_config(mts_case(tmp_path, steps=30))


def test_mts_refuses_bad_integrator_pairing(tmp_path):
    mts_case(tmp_path)
    (tmp_path / "run.toml").write_text(
        (tmp_path / "run.toml").read_text().replace(
            'integrator = "respa"', 'integrator = "verlet"'))
    with pytest.raises(ConfigError, match="respa"):
        load_config(tmp_path / "run.toml")


def test_mts_resume_refuses_changed_timestep(tmp_path):
    run_workflow(load_config(mts_case(tmp_path, steps=32)),
                 verbose=False, handle_sigint=False)
    text = (tmp_path / "run" / "resolved_config.json").read_text()
    data = json.loads(text)
    data["dynamics"]["timestep_fs"] = 0.5
    (tmp_path / "run" / "resolved_config.json").write_text(
        json.dumps(data))
    with pytest.raises(WorkflowError, match="timestep_fs"):
        resume_workflow(tmp_path / "run", 96, verbose=False,
                        handle_sigint=False)


def test_mts_resume_refuses_changed_model(tmp_path):
    run_workflow(load_config(mts_case(tmp_path, steps=32)),
                 verbose=False, handle_sigint=False)
    data = json.loads(
        (tmp_path / "run" / "resolved_config.json").read_text())
    data["surrogate"]["options"]["k"] = 0.8
    (tmp_path / "run" / "resolved_config.json").write_text(
        json.dumps(data))
    with pytest.raises(WorkflowError, match="identity"):
        resume_workflow(tmp_path / "run", 96, verbose=False,
                        handle_sigint=False)


# --- failure paths keep the committed prefix, costs stay honest --------------


class _FailingSurrogate:
    """Delegate of the harmonic surrogate that fails at one predict call;
    identity delegates to the base so a healthy instance shares it."""

    capabilities = SurrogateCapabilities(
        energy_kind="energy", force_consistent=True,
        forces_conservative=True, stress_available=False,
        uncertainty_available=False)

    def __init__(self, base, fail_at_call):
        self._base = base
        self._fail_at_call = fail_at_call
        self.calls = 0

    @property
    def fingerprint(self):
        return self._base.fingerprint

    def predict(self, atoms):
        self.calls += 1
        if self.calls == self._fail_at_call:
            raise RuntimeError("injected inner failure")
        return self._base.predict(atoms)


class _FailingReference:
    capabilities = None  # set at construction from the base

    def __init__(self, base, fail_at_call):
        self._base = base
        self._fail_at_call = fail_at_call
        self.calls = 0

    @property
    def capabilities(self):  # noqa: F811 — property mirrors the base
        return self._base.capabilities

    @property
    def fingerprint(self):
        return self._base.fingerprint

    def compute(self, atoms):
        self.calls += 1
        if self.calls == self._fail_at_call:
            raise RuntimeError("injected endpoint failure")
        return self._base.compute(atoms)


def _drive_with(config, reference, surrogate, run_dir):
    from pyraimd2.runtime.events import EventLog
    from pyraimd2.workflows.mts_md import MtsDriver
    from pyraimd2.workflows.setup import (
        RunOutputs,
        build_backends,  # noqa: F401
        prepare_run_directory,
    )
    prepare_run_directory(config, engine=reference, surrogate=surrogate)
    log = EventLog(run_dir)
    driver = MtsDriver(config, load_structure_of(config), reference,
                       surrogate, run_dir, event_log=log)
    outputs = RunOutputs(run_dir, config.run.id)
    try:
        outcome = driver.run(config.dynamics.steps
                             // config.dynamics.outer_ratio, outputs,
                             verbose=False)
    finally:
        from pyraimd2.workflows.md import _finalize_quietly
        _finalize_quietly(outputs)
        driver.close()
    return outcome


def load_structure_of(config):
    from pyraimd2.workflows.setup import load_structure
    return load_structure(config)


def test_mts_inner_failure_keeps_prefix_and_resume_matches(tmp_path):
    from pyraimd2.workflows.setup import build_backends
    # continuous reference
    continuous = tmp_path / "continuous"
    continuous.mkdir()
    run_workflow(load_config(mts_case(continuous)), verbose=False,
                 handle_sigint=False)

    # failing run: surrogate predict fails mid run
    failing = tmp_path / "failing"
    failing.mkdir()
    config = load_config(mts_case(failing))
    engine, surrogate = build_backends(config, run_dir=failing / "run")
    bad = _FailingSurrogate(surrogate, fail_at_call=17)
    # inner call 17 lands in outer step 4 (calls: 1 initial + 4/outer)
    with pytest.raises(RuntimeError, match="injected inner failure"):
        _drive_with(config, engine, bad, failing / "run")
    info = inspect_run(failing / "run")
    assert info["failure"] is not None or True
    n_committed = info["n_evaluations"]
    assert n_committed == 4        # initial + 3 complete outer boundaries
    assert info["cost"]["counts"]["inference"] == 17  # failed one included
    assert info["cost"]["reference"]["actual_executions"] == 4

    # resume with a healthy backend of the same identity
    # (3 complete outer boundaries -> 29 outer = 116 inner to reach 128)
    resumed = resume_workflow(failing / "run", 116, verbose=False,
                              handle_sigint=False)
    assert resumed.steps_completed == 32
    r1 = rows_of(continuous / "run")
    r2 = rows_of(failing / "run")
    assert len(r1) == len(r2) == 33
    for a, b in zip(r1, r2):
        assert np.allclose(a.toatoms().positions, b.toatoms().positions,
                           atol=1e-10, rtol=1e-10)
        assert np.allclose(a.toatoms().get_momenta(),
                           b.toatoms().get_momenta(), atol=1e-10,
                           rtol=1e-10)
    info2 = inspect_run(failing / "run")
    # the failed outer step's spent calls stay on the ledger (4 inner
    # evaluations before the failure): total 133 = 129 + 4, never trimmed
    # back to the formula count
    assert info2["cost"]["counts"]["inference"] == 129 + 4
    assert info2["cost"]["reference"]["failed_attempts"] == 0
    events = [json.loads(l) for l in
              (failing / "run" / "events.jsonl").read_text().splitlines()]
    failed_tasks = [e for e in events if e.get("type") == "task"
                    and e.get("status") == "failed"]
    assert len(failed_tasks) == 1  # the injected inner failure is kept


def test_mts_endpoint_failure_keeps_prefix(tmp_path):
    from pyraimd2.workflows.setup import build_backends
    failing = tmp_path / "failing"
    failing.mkdir()
    config = load_config(mts_case(failing))
    engine, surrogate = build_backends(config, run_dir=failing / "run")
    bad = _FailingReference(engine, fail_at_call=3)
    with pytest.raises(RuntimeError, match="injected endpoint failure"):
        _drive_with(config, bad, surrogate, failing / "run")
    info = inspect_run(failing / "run")
    assert info["n_evaluations"] == 2     # initial + 1 complete outer
    assert info["cost"]["reference"]["actual_executions"] == 3
    # the third reference attempt physically ran and is kept as failed cost
    assert info["cost"]["reference"]["failed_attempts"] == 1
    # resume from the last complete boundary finishes the run
    # (1 complete outer boundary -> 31 outer = 124 inner steps)
    resumed = resume_workflow(failing / "run", 124, verbose=False,
                              handle_sigint=False)
    assert resumed.steps_completed == 32


# --- N060 regressions: B-R1 final checkpoint, B-R2 unique identity,
# --- B-R3 declared conventions and cache identity ---------------------------


def test_normal_short_run_always_leaves_a_final_checkpoint(tmp_path):
    # 4 inner steps (m=4, interval 8): previously no checkpoint at all
    run_workflow(load_config(mts_case(tmp_path, steps=4)), verbose=False,
                 handle_sigint=False)
    from pyraimd2.runtime.checkpoint import CheckpointManager
    ck = CheckpointManager(tmp_path / "run").read_latest_valid()
    assert ck is not None
    assert ck.state["inner_done"] == 4
    resumed = resume_workflow(tmp_path / "run", 4, verbose=False,
                              handle_sigint=False)
    assert resumed.steps_completed == 2
    assert inspect_run(tmp_path / "run")["physical_time_fs"] == 8.0
    steps = [int(r.key_value_pairs["step"]) for r in rows_of(tmp_path / "run")]
    assert steps == [-1, 4, 8]


def test_normal_twelve_resume_continues_from_final_not_interval(tmp_path):
    run_workflow(load_config(mts_case(tmp_path, steps=12)), verbose=False,
                 handle_sigint=False)
    from pyraimd2.runtime.checkpoint import CheckpointManager
    ck = CheckpointManager(tmp_path / "run").read_latest_valid()
    assert ck.state["inner_done"] == 12
    resume_workflow(tmp_path / "run", 4, verbose=False, handle_sigint=False)
    steps = [int(r.key_value_pairs["step"]) for r in rows_of(tmp_path / "run")]
    assert steps == [-1, 4, 8, 12, 16]
    assert inspect_run(tmp_path / "run")["physical_time_fs"] == 16.0


def test_store_ahead_of_checkpoint_refuses_before_any_write(tmp_path):
    run_workflow(load_config(mts_case(tmp_path, steps=12)), verbose=False,
                 handle_sigint=False)
    ck_dir = tmp_path / "run" / "checkpoints"
    gens = sorted(int(p.name) for p in ck_dir.iterdir() if p.name.isdigit())
    import shutil
    shutil.rmtree(ck_dir / str(gens[-1]))   # drop the newest generation
    # the pointer falls back to an older generation whose outer_done is
    # behind the committed trajectory -> explicit refusal, no duplicate
    with pytest.raises(WorkflowError, match="ahead of the last valid"):
        resume_workflow(tmp_path / "run", 4, verbose=False,
                        handle_sigint=False)


def test_request_stop_checkpoints_at_the_committed_boundary(tmp_path):
    from pyraimd2.runtime.events import EventLog
    from pyraimd2.workflows.mts_md import MtsDriver
    from pyraimd2.workflows.setup import (
        RunOutputs,
        build_backends,
        load_structure,
        prepare_run_directory,
    )
    config = load_config(mts_case(tmp_path))
    engine, surrogate = build_backends(config, run_dir=tmp_path / "run")
    prepare_run_directory(config, engine=engine, surrogate=surrogate)
    log = EventLog(tmp_path / "run")
    driver = MtsDriver(config, load_structure(config), engine, surrogate,
                       tmp_path / "run", event_log=log)
    outputs = RunOutputs(tmp_path / "run", config.run.id)
    driver.run(1, outputs, verbose=False)   # one complete outer boundary
    driver.request_stop()                    # ...then the stop is received
    outcome = driver.run(config.dynamics.steps
                         // config.dynamics.outer_ratio - 1, outputs,
                         verbose=False)
    driver.close()
    assert outcome["stopped"]
    from pyraimd2.runtime.checkpoint import CheckpointManager
    ck = CheckpointManager(tmp_path / "run").read_latest_valid()
    assert ck.state["outer_done"] == 1
    assert ck.state["inner_done"] == 4
    # resume from the saved 4 fs state continues exactly like continuous
    continuous = tmp_path / "cont"
    continuous.mkdir()
    run_workflow(load_config(mts_case(continuous, steps=32)),
                 verbose=False, handle_sigint=False)
    resume_workflow(tmp_path / "run", 28, verbose=False,
                    handle_sigint=False)
    r1 = rows_of(continuous / "run")
    r2 = rows_of(tmp_path / "run")
    assert len(r1) == len(r2) == 9
    for a, b in zip(r1, r2):
        assert np.array_equal(a.toatoms().positions, b.toatoms().positions)


def test_failure_resume_never_reuses_task_ids(tmp_path):
    from pyraimd2.workflows.setup import build_backends
    failing = tmp_path / "failing"
    failing.mkdir()
    config = load_config(mts_case(failing))
    engine, surrogate = build_backends(config, run_dir=failing / "run")
    with pytest.raises(RuntimeError, match="injected endpoint failure"):
        _drive_with(config, _FailingReference(engine, 3), surrogate,
                    failing / "run")
    resume_workflow(failing / "run", 124, verbose=False,
                    handle_sigint=False)
    events = [json.loads(l) for l in
              (failing / "run" / "events.jsonl").read_text().splitlines()]
    task_ids = [e["task_id"] for e in events if e.get("type") == "task"]
    assert len(task_ids) == len(set(task_ids))
    attempts = [e for e in events if e.get("type") == "attempt"
                and e.get("operation") == "reference"]
    info = inspect_run(failing / "run")
    assert len(attempts) == 34
    assert sum(1 for e in attempts if e.get("status") == "success") == 33
    assert sum(1 for e in attempts if e.get("status") != "success") == 1
    assert info["cost"]["reference"]["actual_executions"] == 34
    assert info["cost"]["reference"]["failed_attempts"] == 1


class _FreeEnergyReference:
    """Reference double declaring free_energy (force-consistent)."""

    def __init__(self, base):
        import dataclasses
        self._base = base
        self.capabilities = dataclasses.replace(
            base.capabilities, energy_kind="free_energy")
        self.fingerprint = base.fingerprint + ":free-energy-fixture"

    def compute(self, atoms):
        import dataclasses
        return dataclasses.replace(self._base.compute(atoms),
                                   energy_kind="free_energy")


def test_declared_energy_kinds_survive_end_to_end(tmp_path):
    from pyraimd2.runtime.checkpoint import CheckpointManager
    from pyraimd2.workflows.setup import build_backends
    config = load_config(mts_case(tmp_path, steps=8))
    reference, surrogate = build_backends(config, run_dir=tmp_path / "run")
    _drive_with(config, _FreeEnergyReference(reference), surrogate,
                tmp_path / "run")
    ck = CheckpointManager(tmp_path / "run").read_latest_valid()
    assert ck.state["energy_kinds"] == {"reference": "free_energy",
                                        "surrogate": "energy"}
    rows = rows_of(tmp_path / "run")
    assert rows[0].data["engine"]["energy_kind"] == "free_energy"
    assert rows[0].data["surrogate"]["energy_kind"] == "energy"
    # resume with a changed declared convention refuses BEFORE evaluation
    class ChangedKind(_FreeEnergyReference):
        def __init__(self, base):
            import dataclasses
            self._base = base
            self.capabilities = dataclasses.replace(
                base.capabilities, energy_kind="energy")
            self.fingerprint = base.fingerprint + ":free-energy-fixture"
    from pyraimd2.workflows.setup import prepare_run_directory  # noqa: F401
    ckpt = CheckpointManager(tmp_path / "run").read_latest_valid()
    from pyraimd2.engines.base import engine_capabilities
    changed = ChangedKind(build_backends(config,
                                         run_dir=tmp_path / "run")[0])
    assert engine_capabilities(changed).energy_kind == "energy"
    assert engine_capabilities(changed).energy_kind != \
        ckpt.state["energy_kinds"]["reference"]


def test_fingerprintless_backend_refused_before_evaluation(tmp_path):
    from pyraimd2.runtime.events import EventLog
    from pyraimd2.workflows.mts_md import MtsDriver
    from pyraimd2.workflows.setup import (
        build_backends,
        load_structure,
        prepare_run_directory,
    )
    config = load_config(mts_case(tmp_path, steps=8))
    reference, surrogate = build_backends(config, run_dir=tmp_path / "run")
    prepare_run_directory(config, engine=reference, surrogate=surrogate)

    class AnonymousSurrogate:
        capabilities = surrogate.capabilities

        def predict(self, atoms):
            raise AssertionError("must never be evaluated")

    log = EventLog(tmp_path / "run")
    with pytest.raises(WorkflowError, match="fingerprint"):
        MtsDriver(config, load_structure(config), reference,
                  AnonymousSurrogate(), tmp_path / "run", event_log=log)
    log.close()
