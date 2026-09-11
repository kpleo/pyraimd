"""Si recipe orchestration acceptance with stub QE/MACE backends (M5-2).

Zero real QE/MACE: the shipped `examples/si_bulk_qe_mace/recipe` TOMLs are
driven by `run_serial_recipe` with analytic stand-ins injected through the
real entry-point discovery path.  What is proven is the stage orchestration
in exactly the shipped configuration (paths, stages, step counts, intervals)
— transitions, dedup, resume dispatch and the complete-state handoff; the
real-material acceptance belongs to the cluster side.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pyraimd2.backends import registry
from pyraimd2.backends.harmonic import HarmonicReference, HarmonicSurrogate
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store
from pyraimd2.workflows.setup import WorkflowError
from pyraimd2.workflows.stages import RecipeStage, run_serial_recipe

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "si_bulk_qe_mace"


class _FakeQe(HarmonicReference):
    """Analytic stand-in for the QE reference (hermetic; fingerprinted)."""

    @property
    def name(self):
        return "fake-qe"


class _FakeMace(HarmonicSurrogate):
    """Analytic stand-in for the frozen MACE surrogate."""


def _fake_qe_factory(**kwargs):
    return _FakeQe()


def _fake_mace_factory(**kwargs):
    return _FakeMace()


_fake_qe_factory.backend_kind = "engine"
_fake_mace_factory.backend_kind = "surrogate"


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch):
    """Stand-ins for the QE/MACE builtins through the real factory path
    (analytic, hermetic, contract-compatible); no real QE/MACE anywhere."""
    monkeypatch.setitem(
        registry._BUILTINS, "qe",
        (registry._BUILTINS["qe"][0], "fake_qe_module", "create_qe"))
    monkeypatch.setitem(
        registry._BUILTINS, "mace",
        (registry._BUILTINS["mace"][0], "fake_mace_module", "create_mace"))
    import types

    qe_module = types.ModuleType("fake_qe_module")
    qe_module.create_qe = _fake_qe_factory
    mace_module = types.ModuleType("fake_mace_module")
    mace_module.create_mace = _fake_mace_factory
    sys.modules["fake_qe_module"] = qe_module
    sys.modules["fake_mace_module"] = mace_module
    yield


def _write_si_recipe(tmp_path):
    """The shipped TOMLs in an output root, with dummy dependency files."""
    root = tmp_path / "si-recipe"
    root.mkdir(parents=True)
    for name in ("relax.toml", "nvt.toml", "nve.toml"):
        (root / name).write_bytes((EXAMPLE / "recipe" / name).read_bytes())
    import subprocess
    import sys as _sys

    subprocess.run([_sys.executable, str(EXAMPLE / "make_structure.py")],
                   cwd=root, check=True, capture_output=True, text=True)
    pseudos = root / "qe_pseudos"
    pseudos.mkdir()
    (pseudos / "Si.pbe-n-kjpaw_psl.1.0.0.UPF").write_text("dummy upf\n")
    (root / "mace-mpa-0-medium.model").write_bytes(b"dummy model")
    return root


def _stages(root):
    return [RecipeStage("relax", root / "relax.toml"),
            RecipeStage("nvt", root / "nvt.toml", momenta="initialize"),
            RecipeStage("nve", root / "nve.toml", momenta="preserve")]


def _rows(run_dir, run_id):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def test_si_recipe_orchestration_with_stub_backends(tmp_path):
    root = _write_si_recipe(tmp_path)
    import pyraimd2.workflows.stages as stages_module

    calls = []
    real_run = stages_module.run_workflow

    def counting(config, **kwargs):
        calls.append(config.run.id)
        return real_run(config, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(stages_module, "run_workflow", counting)
    try:
        manifest = run_serial_recipe(root, _stages(root), verbose=False)
    finally:
        monkey.undo()
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 3
    assert calls == ["si-relax", "si-nvt", "si-nve"]
    by_name = {s["name"]: s for s in manifest["stages"]}
    # the adaptive NVT stage bills anchors, probes and checks by purpose
    nvt_cost = by_name["nvt"]["result"]["reference_by_purpose"]
    assert nvt_cost.get("anchor") and nvt_cost.get("probe")
    assert by_name["nvt"]["result"]["complete_steps"] == 12
    # provenance chain and the complete-state handoff
    assert by_name["nvt"]["source"]["source_run_id"] == "si-relax"
    assert by_name["nve"]["source"]["source_run_id"] == "si-nvt"
    from pyraimd2.workflows.stages import load_completed_state

    nvt_state = load_completed_state(root / "nvt")
    nve_initial = _rows(root / "nve", "si-nve")[0].toatoms()
    np.testing.assert_array_equal(nve_initial.positions,
                                  nvt_state.atoms.positions)
    np.testing.assert_array_equal(nve_initial.get_momenta(),
                                  nvt_state.atoms.get_momenta())
    assert nve_initial.pbc.all()
    # idempotent: a second invocation computes nothing new
    monkey.setattr(stages_module, "run_workflow", counting)
    try:
        run_serial_recipe(root, _stages(root), verbose=False)
    finally:
        monkey.undo()
    assert calls == ["si-relax", "si-nvt", "si-nve"]
    # finished stage keeps timing marked complete; relax has optimizer steps
    assert by_name["relax"]["result"]["converged"] is True


def test_si_recipe_resume_after_mid_nvt_crash(tmp_path):
    control = _write_si_recipe(tmp_path / "control")
    run_serial_recipe(control, _stages(control), verbose=False)
    crash = _write_si_recipe(tmp_path / "crash")
    monkey = pytest.MonkeyPatch()
    real_append_once = EventLog.append_once

    def crash_step(self, key, event_type, payload):
        if key == "step:si-nvt:5":  # not a checkpoint multiple (interval 4)
            raise RuntimeError("injected crash")
        return real_append_once(self, key, event_type, payload)

    monkey.setattr(EventLog, "append_once", crash_step)
    try:
        with pytest.raises(RuntimeError, match="injected crash"):
            run_serial_recipe(crash, _stages(crash), verbose=False)
    finally:
        monkey.undo()
    manifest = run_serial_recipe(crash, _stages(crash), verbose=False,
                                 force_unlock=True)
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 3
    for stage, run_id in (("nvt", "si-nvt"), ("nve", "si-nve")):
        rows_a = _rows(crash / stage, run_id)
        rows_b = _rows(control / stage, run_id)
        assert len(rows_a) == len(rows_b)
        for row_a, row_b in zip(rows_a, rows_b):
            np.testing.assert_array_equal(row_a.toatoms().positions,
                                          row_b.toatoms().positions)
            np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                          row_b.toatoms().get_momenta())


def test_si_recipe_keeps_user_configs_visibly(tmp_path, capsys):
    # run_recipe.py never silently overwrites an edited output config: it
    # keeps it, says so, and the controller then refuses the change.

    spec = importlib.util.spec_from_file_location(
        "si_run_recipe", EXAMPLE / "run_recipe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def invoke(out):
        monkey = pytest.MonkeyPatch()
        monkey.setattr(sys, "argv",
                       ["run_recipe.py", "--output", str(out)])
        try:
            module.main()
        finally:
            monkey.undo()

    root = _write_si_recipe(tmp_path)
    invoke(root)
    nvt_config = root / "nvt.toml"
    nvt_config.write_text(nvt_config.read_text().replace(
        "transverse_cap = 0.1", "transverse_cap = 0.2"))
    with pytest.raises(WorkflowError, match="different config"):
        invoke(root)
    out, _ = capsys.readouterr()
    assert "keeping existing nvt.toml" in out


def test_validate_defers_controller_materialized_structure(tmp_path):
    # F1: `pyramid validate recipe/nvt.toml` pre-run — the materialized
    # input is absent by convention, so the structure section defers with
    # an explicit scope note; backends and capabilities still validate.
    from pyraimd2.config import load_config
    from pyraimd2.workflows import validate_setup

    root = _write_si_recipe(tmp_path)
    assert not (root / "nvt" / "initial.traj").exists()
    report = validate_setup(load_config(root / "nvt.toml"))
    assert "deferred" in report["structure"]
    assert report["capabilities"]["surrogate"]["energy_kind"]
    assert report["capabilities"]["reference"]["energy_kind"]
    # the probe path needs a real structure and refuses honestly
    with pytest.raises(WorkflowError, match="needs a loadable structure"):
        validate_setup(load_config(root / "nvt.toml"), probe=True)
    # a missing structure outside the controller's convention stays an error
    text = (root / "nvt.toml").read_text().replace(
        "nvt/initial.traj", "missing.extxyz")
    (root / "bad.toml").write_text(text)
    with pytest.raises(WorkflowError, match="structure.file not found"):
        validate_setup(load_config(root / "bad.toml"))
    # once the controller materialized the input, the full check applies
    run_serial_recipe(root, _stages(root)[:1], verbose=False)
    import pyraimd2.workflows.stages as stages_module

    relax_state = stages_module.load_completed_state(root / "relax")
    record = {"source": None, "status": "pending"}
    stages_module._prepare_md_stage(
        record, RecipeStage("nvt", root / "nvt.toml", momenta="initialize"),
        load_config(root / "nvt.toml"), root / "nvt", relax_state)
    report = validate_setup(load_config(root / "nvt.toml"))
    assert "deferred" not in report["structure"]
    assert report["structure"]["n_atoms"] == 8
