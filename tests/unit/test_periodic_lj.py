"""Periodic LJ CI recipe: full workflow on a periodic system, no external
software (L0). The example plugin is injected through the real entry-point
discovery path from its source tree."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyraimd2.backends import registry
from pyraimd2.config import load_config
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.workflows import export_run, resume_workflow, run_workflow
from pyraimd2.workflows.export import frames_from_store

RECIPE_DIR = Path(__file__).resolve().parents[2] / "examples" / "periodic_lj"
PLUGIN_SRC = (RECIPE_DIR.parent / "backends" / "pyraimd2_lj" / "src"
              / "pyraimd2_lj" / "__init__.py")


@pytest.fixture(autouse=True)
def _lj_plugin(monkeypatch):
    """Expose examples/backends/pyraimd2_lj through entry-point discovery."""
    registry._reset_entry_point_cache()
    spec = importlib.util.spec_from_file_location("pyraimd2_lj", PLUGIN_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "pyraimd2_lj", module)
    eps = [
        importlib.metadata.EntryPoint(name="lj_reference",
                                      value="pyraimd2_lj:reference_factory",
                                      group="pyraimd2.backends"),
        importlib.metadata.EntryPoint(name="lj_surrogate",
                                      value="pyraimd2_lj:surrogate_factory",
                                      group="pyraimd2.backends"),
    ]
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: eps if group == "pyraimd2.backends" else [])
    yield module
    registry._reset_entry_point_cache()


@pytest.fixture
def recipe(tmp_path):
    shutil.copytree(RECIPE_DIR, tmp_path / "periodic_lj",
                    ignore=shutil.ignore_patterns("runs", "__pycache__"))
    directory = tmp_path / "periodic_lj"
    subprocess.run([sys.executable, "make_structure.py"], cwd=directory,
                   check=True, capture_output=True, text=True)
    assert (directory / "structure.extxyz").exists()
    return directory


def _load(recipe, name):
    return load_config(recipe / name)


def test_singlepoint_and_relax(recipe, _lj_plugin):
    single = run_workflow(_load(recipe, "run-singlepoint.toml"), verbose=False)
    rows = list(Store(single.run_dir / "trajectory.db")._db.select())
    assert len(rows) == 1
    assert np.isfinite(rows[0].data["driving"]["energy"])
    relax = run_workflow(_load(recipe, "run-relax.toml"), verbose=False)
    final_forces = np.asarray(
        list(Store(relax.run_dir / "trajectory.db")._db.select())[-1]
        .data["driving"]["forces"], dtype=float)
    assert np.linalg.norm(final_forces, axis=1).max() < 0.05


def test_plain_modes_periodic(recipe, _lj_plugin):
    for name in ("run-md-reference.toml", "run-md-surrogate.toml"):
        result = run_workflow(_load(recipe, name), verbose=False)
        assert result.steps_completed == 8
        row = list(Store(result.run_dir / "trajectory.db")._db.select())[-1]
        assert row.toatoms().pbc.all()
        assert row.toatoms().cell.volume > 0


def _rows(run_dir, run_id):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def test_adaptive_periodic_acceptance_and_resume(recipe, _lj_plugin):
    # interrupted: 5 steps, then resume +3; continuous: 8 steps elsewhere
    config = _load(recipe, "run-md-adaptive.toml")
    stopped = run_workflow(config, verbose=False, handle_sigint=False)
    assert stopped.steps_completed == 8
    run_dir = stopped.run_dir
    info = inspect_run(run_dir)
    reference = info["cost"]["reference"]
    # the 5%-soft LJ is inside the budget most of the time: real acceptances,
    # with anchors, probes and checks all billed
    assert info["trajectory"]["n_accepted"] >= 2
    assert reference["actual_executions"] >= 1 + 4 + 1  # anchor, probes, check
    # resume continuity: run a second fresh 5-step run, resume +3, compare
    alt = recipe / "alt.md"
    text = (recipe / "run-md-adaptive.toml").read_text()
    text = text.replace('id = "lj-adaptive"', 'id = "lj-adaptive-b"')
    text = text.replace('directory = "runs/lj-adaptive"',
                        'directory = "runs/lj-adaptive-b"')
    text = text.replace("steps = 8", "steps = 5")
    alt.write_text(text)
    part = run_workflow(_load(recipe, "alt.md"), verbose=False,
                        handle_sigint=False)
    assert part.steps_completed == 5
    resumed = resume_workflow(part.run_dir, 3, verbose=False,
                              handle_sigint=False)
    assert resumed.steps_completed == 8
    rows_a = _rows(run_dir, "lj-adaptive")
    rows_b = _rows(part.run_dir, "lj-adaptive-b")
    assert len(rows_a) == len(rows_b) == 9
    for a, b in zip(rows_a, rows_b):
        assert a.key_value_pairs["route"] == b.key_value_pairs["route"]
        np.testing.assert_allclose(a.toatoms().positions,
                                   b.toatoms().positions, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.toatoms().get_momenta(),
                                   b.toatoms().get_momenta(), rtol=0, atol=1e-12)


def test_export_wrapped_and_unwrapped(recipe, _lj_plugin):
    result = run_workflow(_load(recipe, "run-md-surrogate.toml"),
                          verbose=False)
    store = Store(result.run_dir / "trajectory.db")
    run_id = result.run_id
    unwrapped = frames_from_store(store, run_id, force_source="driving")
    wrapped = frames_from_store(store, run_id, force_source="driving",
                                wrap=True)
    assert len(unwrapped) == len(wrapped) == 9
    cell_lengths = np.diag(unwrapped[-1].cell)
    for u, w in zip(unwrapped, wrapped):
        assert w.info["coordinates"] == "wrapped"
        assert u.info["coordinates"] == "unwrapped"
        assert (w.positions >= -1e-12).all()
        assert (w.positions < cell_lengths + 1e-12).all()
    report = export_run(result.run_dir, force_source="driving", force=True)
    assert report["frames"] == 9


def test_serial_recipe_on_periodic_lj(recipe, _lj_plugin):
    """The relax -> NVT -> NVE recipe on the periodic LJ example, driven by
    the shipped run_recipe.py: adaptive NVT in the middle, real handoff of
    the complete state, and idempotent re-invocation."""
    import subprocess
    import sys

    from pyraimd2.workflows.stages import load_completed_state

    out = recipe / "recipe-out"
    script = recipe / "run_recipe.py"
    first = subprocess.run([sys.executable, str(script), "--output",
                            str(out)], capture_output=True, text=True,
                           check=False)
    assert first.returncode == 0, first.stderr[-500:]
    manifest = json.loads((out / "workflow.json").read_text())
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 3
    by_name = {s["name"]: s for s in manifest["stages"]}
    # adaptive NVT in the middle: anchors, probes and checks all billed
    nvt_cost = by_name["nvt"]["result"]["reference_by_purpose"]
    assert nvt_cost.get("anchor") and nvt_cost.get("probe")
    assert by_name["nvt"]["result"]["reference_executions"] >= 1
    # provenance chain and the complete-state handoff
    assert by_name["nvt"]["source"]["source_run_id"] == "lj-relax"
    assert by_name["nve"]["source"]["source_run_id"] == "lj-nvt"
    nvt_state = load_completed_state(out / "nvt")
    nve_initial = _rows(out / "nve", "lj-nve")[0].toatoms()
    np.testing.assert_allclose(nve_initial.positions, nvt_state.atoms.positions,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(nve_initial.get_momenta(),
                               nvt_state.atoms.get_momenta(),
                               rtol=0, atol=1e-12)
    assert nve_initial.pbc.all()
    # idempotent: a second invocation computes nothing new
    second = subprocess.run([sys.executable, str(script), "--output",
                             str(out)], capture_output=True, text=True,
                            check=False)
    assert second.returncode == 0, second.stderr[-500:]
    manifest2 = json.loads((out / "workflow.json").read_text())
    assert [s["status"] for s in manifest2["stages"]] == ["done"] * 3
    nvt_events_1 = (out / "nvt" / "events.jsonl").read_bytes()
    second = subprocess.run([sys.executable, str(script), "--output",
                             str(out)], capture_output=True, text=True,
                            check=False)
    assert (out / "nvt" / "events.jsonl").read_bytes() == nvt_events_1
