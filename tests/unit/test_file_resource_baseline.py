"""Directed checks for the immutable file-resource baseline (T2).

New opt-in runs persist ``file_resources.json`` (schema
``file-resource-baseline-v1``) before the first backend evaluation and
bind its digest into every new checkpoint generation's verified state;
a declaration/option mismatch refuses the run before any compute; a
tampered, missing or corrupted baseline is detected, never re-trusted
from a side file.  Runs without declarations keep their old structure
byte-for-byte.  Analytic test backends only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
from ase.calculators.calculator import Calculator

from pyraimd2.backends import registry
from pyraimd2.config import load_config
from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.engines.ase_resources import (
    FILE_RESOURCE_BASELINE_SCHEMA,
    file_resource_baseline_sha256,
    read_file_sha256,
)
from pyraimd2.surrogate.ase_surrogate import AseSurrogate
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.setup import WorkflowError


class FileBacked(Calculator):
    """Test calculator computing a real linear force from its file."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, model_path):
        super().__init__()
        self.parameters = {"model": str(model_path)}
        # the file genuinely feeds the physics: the stiffness is read
        # from the file content (first line, float)
        self._k = float(Path(model_path).read_text().splitlines()[0])
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=None):
        super().calculate(atoms, properties, system_changes or [])
        self.calls += 1
        dr = self.atoms.positions - 0.9
        self.results = {"energy": 0.5 * self._k * float((dr**2).sum()),
                        "forces": -self._k * dr}


def _write_run(root, *, steps=4, checkpoint=2, backend="file-backed-test",
               options, mode="reference", ensemble="nve"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "structure.extxyz").write_text(
        "2\nH2\nH 0.85 0.9 0.9\nH 0.95 0.9 0.9\n")
    surrogate_block = ""
    reference_block = ""
    if mode == "reference":
        reference_block = f"""[reference]
backend = "{backend}"
{options}
"""
    else:
        surrogate_block = f"""[surrogate]
backend = "{backend}"
{options}
"""
        reference_block = """[reference]
backend = "harmonic-reference"
k = 1.0
r0 = 0.9
"""
    text = f"""\
schema_version = 1
[run]
id = "file-res"
directory = "."
seed = 42
[task]
kind = "md"
mode = "{mode}"
[structure]
file = "structure.extxyz"
{reference_block}{surrogate_block}[dynamics]
ensemble = "{ensemble}"
timestep_fs = 0.5
steps = {steps}
temperature_K = 300.0
velocity_seed = 7
{"friction_per_fs = 0.01" if ensemble == "nvt" else ""}
{"thermostat_seed = 123" if ensemble == "nvt" else ""}
[checkpoint]
interval_steps = {checkpoint}
[output]
trajectory_interval_steps = 1
summary_interval_steps = 2
"""
    if mode == "adaptive":
        text += """\
[policy]
force_budget_eV_A = 0.5
time_cap_fs = 1000.0
transverse_cap = 1.0
[verification]
probability = 0.5
seed = 7
"""
    (root / "run.toml").write_text(text)
    return root / "run.toml"


def _inject(monkeypatch, kind, factory):
    monkeypatch.setitem(
        registry._BUILTINS, "file-backed-test",
        (kind, "fake_file_module", "create_file_backed"))
    import types

    module = types.ModuleType("fake_file_module")
    module.create_file_backed = factory
    monkeypatch.setitem(__import__("sys").modules, "fake_file_module", module)


def _engine_factory(**kwargs):
    return AseEngine(FileBacked(kwargs["model"]),
                     file_parameters={"model": "potential"})


def _surrogate_factory(**kwargs):
    return AseSurrogate(FileBacked(kwargs["model"]),
                        file_parameters={"model": "potential"})


def _checkpoint_states(run_dir):
    states = []
    for path in sorted((Path(run_dir) / "checkpoints").glob("*/state.json")):
        states.append(json.loads(path.read_text()))
    return states


def test_baseline_written_and_bound_into_every_checkpoint(tmp_path,
                                                          monkeypatch):
    _inject(monkeypatch, "engine", _engine_factory)
    model = tmp_path / "inputs" / "model.dat"
    model.parent.mkdir()
    model.write_text("1.5\n")
    config_path = _write_run(tmp_path / "run",
                             options=f'model = "{model}"')
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)

    run_dir = tmp_path / "run"
    baseline = json.loads((run_dir / "file_resources.json").read_text())
    assert baseline["schema"] == FILE_RESOURCE_BASELINE_SCHEMA
    assert baseline["run_id"] == "file-res"
    (resource,) = baseline["resources"].values()
    assert resource["role"] == "potential"
    assert resource["parameter"] == resource["option"] == "model"
    assert resource["backend"] == "file-backed-test"
    assert resource["original_path"] == str(model)
    assert resource["sha256"] == read_file_sha256(model)
    assert resource["identity_format"] == "ase-file-identity-v1"
    # the read-back digest is the checkpoint association's value
    digest = file_resource_baseline_sha256(run_dir)
    assert digest is not None
    states = _checkpoint_states(run_dir)
    assert states  # interval 2 over 4 steps + run end
    assert all(state.get("file_resource_baseline_sha256") == digest
               for state in states)
    # the manifest displays the same digest (display only)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["file_resource_baseline"]["sha256"] == digest


def test_declaration_option_mismatch_refused_before_any_evaluation(
        tmp_path, monkeypatch):
    # the factory builds the calculator on file B while the option names A
    model_a = tmp_path / "a" / "model.dat"
    model_b = tmp_path / "b" / "model.dat"
    model_a.parent.mkdir()
    model_b.parent.mkdir()
    model_a.write_text("1.5\n")
    model_b.write_text("1.5\n")

    def mismatched_factory(**kwargs):
        return AseEngine(FileBacked(model_b),
                         file_parameters={"model": "potential"})

    _inject(monkeypatch, "engine", mismatched_factory)
    config_path = _write_run(tmp_path / "run",
                             options=f'model = "{model_a}"')
    with pytest.raises(WorkflowError, match="same file"):
        run_workflow(load_config(config_path), verbose=False,
                     handle_sigint=False)
    # the refusal happened before any backend evaluation
    assert not (tmp_path / "run" / "events.jsonl").exists()
    assert not (tmp_path / "run" / "file_resources.json").exists()


def test_missing_option_refused_before_any_evaluation(tmp_path,
                                                      monkeypatch):
    model = tmp_path / "model.dat"
    model.write_text("1.5\n")

    def undeclared_option_factory(**kwargs):
        # the adapter declares a parameter the config never named
        calculator = FileBacked(kwargs["model"])
        calculator.parameters = {"weights": str(model)}
        return AseEngine(calculator,
                         file_parameters={"weights": "potential"})

    _inject(monkeypatch, "engine", undeclared_option_factory)
    config_path = _write_run(tmp_path / "run",
                             options=f'model = "{model}"')
    with pytest.raises(WorkflowError, match="no entry"):
        run_workflow(load_config(config_path), verbose=False,
                     handle_sigint=False)
    assert not (tmp_path / "run" / "events.jsonl").exists()


def test_tampered_missing_or_corrupt_baseline_detected(tmp_path,
                                                       monkeypatch):
    _inject(monkeypatch, "engine", _engine_factory)
    model = tmp_path / "model.dat"
    model.write_text("1.5\n")
    config_path = _write_run(tmp_path / "run",
                             options=f'model = "{model}"')
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    run_dir = tmp_path / "run"
    baseline_path = run_dir / "file_resources.json"
    original_bytes = baseline_path.read_bytes()
    digest = file_resource_baseline_sha256(run_dir)
    # replace one byte: the recorded association no longer matches
    payload = json.loads(original_bytes)
    payload["resources"]["reference.potential"]["sha256"] = "0" * 64
    baseline_path.write_text(json.dumps(payload, indent=2) + "\n")
    assert file_resource_baseline_sha256(run_dir) != digest
    # corrupt the file outright: unreadable JSON never passes as a baseline
    baseline_path.write_bytes(b"not json {")
    with pytest.raises(ValueError):
        json.loads(baseline_path.read_text())
    assert file_resource_baseline_sha256(run_dir) != digest
    # delete it: no side-file resurrection — the association reports absent
    baseline_path.unlink()
    assert file_resource_baseline_sha256(run_dir) is None
    # restore and the original association matches again
    baseline_path.write_bytes(original_bytes)
    assert file_resource_baseline_sha256(run_dir) == digest


def test_no_declaration_keeps_the_old_structure(tmp_path):
    config_path = _write_run(tmp_path / "run", backend="harmonic-reference",
                             options="k = 1.0\nr0 = 0.9")
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    run_dir = tmp_path / "run"
    assert not (run_dir / "file_resources.json").exists()
    assert file_resource_baseline_sha256(run_dir) is None
    states = _checkpoint_states(run_dir)
    assert states
    assert all("file_resource_baseline_sha256" not in state
               for state in states)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "file_resource_baseline" not in manifest


def test_adaptive_run_binds_the_baseline_and_resume_keeps_it(
        tmp_path, monkeypatch):
    _inject(monkeypatch, "surrogate", _surrogate_factory)
    model = tmp_path / "model.dat"
    model.write_text("1.5\n")
    config_path = _write_run(tmp_path / "run", mode="adaptive",
                             options=f'model = "{model}"')
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    run_dir = tmp_path / "run"
    digest = file_resource_baseline_sha256(run_dir)
    states = _checkpoint_states(run_dir)
    assert states
    assert all(state.get("file_resource_baseline_sha256") == digest
               for state in states)
    # resume: the new generations carry the association restored from the
    # checkpoint state itself
    resume_workflow(run_dir, 2, verbose=False, handle_sigint=False)
    states_after = _checkpoint_states(run_dir)
    assert len(states_after) >= len(states)
    assert all(state.get("file_resource_baseline_sha256") == digest
               for state in states_after)
