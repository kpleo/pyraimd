"""S0a regression: every Store closes on its owner's controlled return —
workflow-created stores close in the workflow, driver-owned stores close
with the driver, and a caller-passed store outlives the runner (0.5-M3A
handoff §S0a; the three paths the store-lifecycle probes reproduced)."""

from __future__ import annotations

import pytest
from test_review_r4 import _write_config

from pyraimd2.config import load_config
from pyraimd2.workflows import md as md_module
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows import setup as wf_setup


def _tracking_stores(monkey):
    stores = []
    for module in (md_module, wf_setup):
        real = module.Store

        def track(*args, _real=real, **kwargs):
            store = _real(*args, **kwargs)
            stores.append(store)
            return store

        monkey.setattr(module, "Store", track)
    return stores


class _BrokenOutputs:
    def __init__(self, *args, **kwargs):
        raise OSError("injected outputs failure")


def test_plain_setup_failure_closes_driver_store(tmp_path):
    config = load_config(_write_config(tmp_path / "s0a", mode="reference",
                                       steps=2))
    monkey = pytest.MonkeyPatch()
    stores = _tracking_stores(monkey)
    monkey.setattr(md_module, "RunOutputs", _BrokenOutputs)
    monkey.setattr(wf_setup, "RunOutputs", _BrokenOutputs)
    try:
        with pytest.raises(OSError, match="injected outputs"):
            run_workflow(config, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()
    assert stores
    for store in stores:
        assert store._db is None  # closed before the error returned


def test_resume_setup_failure_closes_driver_store(tmp_path):
    config = load_config(_write_config(tmp_path / "s0a-r", mode="reference",
                                       steps=4))
    run_workflow(config, verbose=False, handle_sigint=False)
    monkey = pytest.MonkeyPatch()
    stores = _tracking_stores(monkey)
    monkey.setattr(md_module, "RunOutputs", _BrokenOutputs)
    monkey.setattr(wf_setup, "RunOutputs", _BrokenOutputs)
    try:
        with pytest.raises(OSError, match="injected outputs"):
            resume_workflow(config.run.directory, 2, verbose=False,
                            handle_sigint=False, force_unlock=True)
    finally:
        monkey.undo()
    assert stores
    for store in stores:
        assert store._db is None


def test_adaptive_workflow_closes_its_own_store(tmp_path):
    from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE

    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    text = HARMONIC_CONFIG.replace("steps = 20", "steps = 3")
    (tmp_path / "run.toml").write_text(text)
    config = load_config(tmp_path / "run.toml")
    monkey = pytest.MonkeyPatch()
    stores = _tracking_stores(monkey)
    try:
        run_workflow(config, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()
    assert stores
    for store in stores:
        assert store._db is None


def test_caller_passed_store_outlives_the_runner(tmp_path):
    from ase import Atoms
    from test_review_r1 import Model, Reference, _direction

    from pyraimd2.loop import EnergeticRunner
    from pyraimd2.runtime.events import EventLog
    from pyraimd2.store import Store

    run_dir = tmp_path / "lib"
    run_dir.mkdir()
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    runner = EnergeticRunner(
        atoms, Model(), Reference(), store, "run", run_dir=run_dir,
        event_log=EventLog(run_dir), checkpoint_interval_steps=100,
        direction=_direction, force_budget=0.08, timestep_fs=0.1,
        time_cap_fs=2.0, check_probability=0.0, check_seed=2)
    runner.run(1)
    runner.close()
    assert store._db is not None  # the caller's store stays usable
    assert len(list(store._db.select(run_id="run"))) == 2
    store.close()
