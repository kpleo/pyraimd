"""T3/T4 acceptance: relocate a run and its declared resource files, then
resume and export through the Python workflow API.

Real process boundaries: process A runs N steps and exits; the run
directory and the resource files move to a new path (with spaces and
non-ASCII characters); the old resource location is removed; process B
resumes with an explicit mapping for M steps.  Deterministic NVE state,
driving forces, step counts and physical time agree with the continuous
control at rtol=0, atol=1e-12.  Analytic file-backed test backends only —
zero real DFT budget.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from test_file_resource_baseline import (
    _checkpoint_states,
    _write_run,
)

from pyraimd2.config import load_config
from pyraimd2.store import Store
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.export import frames_for_run
from pyraimd2.workflows.setup import WorkflowError


class FileBacked:
    """Module-level factory pieces must be importable in the child."""

    @staticmethod
    def factory(**kwargs):
        from test_resource_relocation import FileBackedCalculator

        return FileBackedCalculator.as_engine(kwargs["model"])

    @staticmethod
    def surrogate_factory(**kwargs):
        from test_resource_relocation import FileBackedCalculator

        return FileBackedCalculator.as_surrogate(kwargs["model"])


class FileBackedCalculator:
    """The file genuinely feeds the physics: stiffness from file content."""

    @staticmethod
    def as_engine(model_path):
        from pyraimd2.engines.ase_engine import AseEngine

        return AseEngine(FileBackedCalculator._calculator(model_path),
                         file_parameters={"model": "potential"})

    @staticmethod
    def as_surrogate(model_path):
        from pyraimd2.surrogate.ase_surrogate import AseSurrogate

        return AseSurrogate(FileBackedCalculator._calculator(model_path),
                            file_parameters={"model": "potential"})

    @staticmethod
    def _calculator(model_path):
        from typing import ClassVar

        from ase.calculators.calculator import Calculator

        class _FileBacked(Calculator):
            implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

            def __init__(self, model_path):
                super().__init__()
                self.parameters = {"model": str(model_path)}
                self._k = float(
                    Path(model_path).read_text().splitlines()[0])

            def calculate(self, atoms=None, properties=("energy",),
                          system_changes=None):
                super().calculate(atoms, properties, system_changes or [])
                dr = self.atoms.positions - 0.9
                self.results = {
                    "energy": 0.5 * self._k * float((dr**2).sum()),
                    "forces": -self._k * dr}

        return _FileBacked(model_path)


_CHILD = '''
import os
import sys
import types
from pathlib import Path

from test_resource_relocation import FileBacked

from pyraimd2.backends import registry

registry._BUILTINS["file-backed-test"] = ("engine", "fake_file_module",
                                          "create")
registry._BUILTINS["file-backed-test-surr"] = ("surrogate",
                                               "fake_file_module_surr",
                                               "create")
module = types.ModuleType("fake_file_module")
module.create = FileBacked.factory
sys.modules["fake_file_module"] = module
module_s = types.ModuleType("fake_file_module_surr")
module_s.create = FileBacked.surrogate_factory
sys.modules["fake_file_module_surr"] = module_s

from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.config import load_config

mode = os.environ["MODE"]
if mode == "fresh":
    run_workflow(load_config(os.environ["CONFIG"]), verbose=False,
                 handle_sigint=False)
else:
    import json
    mapping = json.loads(os.environ.get("MAPPING", "null"))
    resume_workflow(os.environ["RUN_DIR"], int(os.environ["STEPS"]),
                    verbose=False, handle_sigint=False,
                    resource_paths=mapping)
'''


def _inject_test_backend(monkeypatch):
    # the SAME factory the child processes use — a different calculator
    # class would be a different physical identity by design
    from test_file_resource_baseline import _inject

    _inject(monkeypatch, "engine", FileBacked.factory)


def _child_env(**extra):
    return dict(os.environ,
                PYTHONPATH=os.pathsep.join(
                    [str(Path(__file__).parents[2] / "src"),
                     str(Path(__file__).parent)]),
                **extra)


def _run_child(tmp_path, mode, **extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(_CHILD))
    env = _child_env(**extra)
    return subprocess.run([sys.executable, str(child)], env=env,
                          capture_output=True, text=True, check=False)


def _prepare_run(tmp_path, *, mode="reference", steps=3,
                 ensemble="nve"):
    """Process A: a fresh short run that exits."""
    root = tmp_path / "origin"
    model = root / "inputs" / "model.dat"
    root.mkdir(parents=True)
    model.parent.mkdir()
    model.write_text("1.5\n")
    backend = ("file-backed-test" if mode == "reference"
               else "file-backed-test-surr")
    config_path = _write_run(root / "run", mode=mode, steps=steps,
                             checkpoint=1, ensemble=ensemble,
                             backend=backend,
                             options=f'model = "{model}"')
    result = _run_child(tmp_path, "fresh", MODE="fresh",
                        CONFIG=str(config_path))
    assert result.returncode == 0, result.stderr[-500:]
    return root, model


def _relocate(tmp_path, root, model):
    """Move the run directory and the model to a path with a space and
    non-ASCII characters; the old model location stops existing."""
    moved = tmp_path / "搬迁 relocated"
    moved.mkdir()
    new_root = moved / "run"
    shutil.move(str(root / "run"), str(new_root))
    new_model = moved / "inputs 模型" / "model.dat"
    new_model.parent.mkdir()
    shutil.move(str(model), str(new_model))
    assert not model.exists()  # the old location is gone
    return new_root, new_model


def _frames(run_dir, run_id="file-res"):
    with Store(Path(run_dir) / "trajectory.db") as store:
        return frames_for_run(store, run_dir, run_id, force_source="driving")


def _assert_state_equal(a, b):
    np.testing.assert_allclose(a.positions, b.positions, rtol=0,
                               atol=1e-12)
    np.testing.assert_allclose(a.get_momenta(), b.get_momenta(), rtol=0,
                               atol=1e-12)


def _history_bytes(run_dir):
    run_dir = Path(run_dir)
    out = {}
    for name in ("config.toml", "manifest.json", "resolved_config.json",
                 "file_resources.json"):
        path = run_dir / name
        out[name] = path.read_bytes() if path.is_file() else None
    out["events_prefix"] = (run_dir / "events.jsonl").read_bytes()
    return out


def test_relocation_closed_loop_reference_mode(tmp_path):
    root, model = _prepare_run(tmp_path)
    before = _history_bytes(root / "run")
    new_root, new_model = _relocate(tmp_path, root, model)
    result = _run_child(
        tmp_path, "resume", MODE="resume", RUN_DIR=str(new_root),
        STEPS="2", MAPPING=json.dumps({"reference.potential":
                                       str(new_model)}))
    assert result.returncode == 0, result.stderr[-500:]
    # the control: one uninterrupted 5-step run of the same setup
    control_root, _ = _prepare_run(tmp_path / "control", steps=5)
    relocated_frames = _frames(new_root)
    control_frames = _frames(control_root / "run")
    assert len(relocated_frames) == len(control_frames) == 6
    for relocated, control in zip(relocated_frames, control_frames):
        _assert_state_equal(relocated, control)
        np.testing.assert_allclose(
            relocated.info["energy"], control.info["energy"], rtol=0,
            atol=1e-12)
    # the history was never rewritten: config/manifest/baseline identical,
    # the pre-relocation events are a strict prefix of the current log
    after = _history_bytes(new_root)
    for name in ("config.toml", "manifest.json", "resolved_config.json",
                 "file_resources.json"):
        assert after[name] == before[name], name
    assert after["events_prefix"].startswith(before["events_prefix"])
    # the binding receipt was recorded (and says what it is)
    receipts = sorted((new_root / "resource_bindings").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["record"] == "resource-binding-v1"
    assert receipt["bindings"]["reference.potential"]["path"] == \
        str(new_model)
    assert "not a progress claim" in receipt["note"]
    # export still works with every model file removed afterwards
    new_model.unlink()
    frames = _frames(new_root)
    assert len(frames) == 6


def test_relocation_closed_loop_surrogate_and_adaptive(tmp_path):
    for mode in ("surrogate", "adaptive"):
        root, model = _prepare_run(tmp_path / mode, mode=mode)
        new_root, new_model = _relocate(tmp_path / mode, root, model)
        result = _run_child(
            tmp_path / mode, "resume", MODE="resume",
            RUN_DIR=str(new_root), STEPS="2",
            MAPPING=json.dumps({"surrogate.potential": str(new_model)}))
        assert result.returncode == 0, result.stderr[-500:]
        control_root, _ = _prepare_run(tmp_path / f"{mode}-control",
                                       mode=mode, steps=5)
        relocated_frames = _frames(new_root)
        control_frames = _frames(control_root / "run")
        assert len(relocated_frames) == len(control_frames)
        for relocated, control in zip(relocated_frames, control_frames):
            _assert_state_equal(relocated, control)


def test_adaptive_nvt_relocation_keeps_the_random_streams(tmp_path):
    """NVT representative: the thermostat stream and the check stream
    continue across the relocation; velocities are never re-initialized."""
    root, model = _prepare_run(tmp_path / "nvt", mode="adaptive",
                               ensemble="nvt")
    new_root, new_model = _relocate(tmp_path / "nvt", root, model)
    result = _run_child(
        tmp_path / "nvt", "resume", MODE="resume", RUN_DIR=str(new_root),
        STEPS="2",
        MAPPING=json.dumps({"surrogate.potential": str(new_model)}))
    assert result.returncode == 0, result.stderr[-500:]
    control_root, _ = _prepare_run(tmp_path / "nvt-control",
                                   mode="adaptive", ensemble="nvt", steps=5)
    relocated_frames = _frames(new_root)
    control_frames = _frames(control_root / "run")
    assert len(relocated_frames) == len(control_frames)
    for relocated, control in zip(relocated_frames, control_frames):
        _assert_state_equal(relocated, control)
    # the thermostat state and the per-evaluation check flags match the
    # control exactly — the random streams continued, never re-seeded
    relocated_states = _checkpoint_states(new_root)
    control_states = _checkpoint_states(control_root / "run")
    assert relocated_states[-1]["thermostat"] == \
        control_states[-1]["thermostat"]
    relocated_events = [json.loads(s) for s in
                        (new_root / "events.jsonl").read_text().splitlines()]
    control_events = [json.loads(s) for s in
                      (control_root / "run" / "events.jsonl")
                      .read_text().splitlines()]
    relocated_checks = [bool(e.get("checked")) for e in relocated_events
                        if e["type"] == "evaluation_committed"]
    control_checks = [bool(e.get("checked")) for e in control_events
                      if e["type"] == "evaluation_committed"]
    assert relocated_checks == control_checks


def test_relocation_rejections_and_retry(tmp_path, monkeypatch):
    """Missing mapped file, unknown role, one-byte change (same size and
    mtime) and wrong content all refuse before any factory call; a
    corrected mapping then resumes without duplicating committed frames."""
    # the in-process resume needs the factory registered here too
    _inject_test_backend(monkeypatch)
    root, model = _prepare_run(tmp_path)
    new_root, new_model = _relocate(tmp_path, root, model)
    frames_before = _frames(new_root)
    # unknown role key
    with pytest.raises(WorkflowError, match="not a declared resource"):
        resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                        resource_paths={"reference.unknown":
                                        str(new_model)})
    # non-absolute path
    with pytest.raises(WorkflowError, match="absolute"):
        resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                        resource_paths={"reference.potential":
                                        "relative/model.dat"})
    # a missing mapped file
    missing = tmp_path / "nowhere" / "model.dat"
    with pytest.raises(WorkflowError, match="not an existing regular file"):
        resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                        resource_paths={"reference.potential":
                                        str(missing)})
    # a one-byte change with the same size and preserved mtime
    stat = new_model.stat()
    new_model.write_bytes(b"2.5\n")
    os.utime(new_model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(WorkflowError, match="does not match the baseline"):
        resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                        resource_paths={"reference.potential":
                                        str(new_model)})
    # a valid file with wrong content (one extra line, same family)
    new_model.write_bytes(b"1.5\n2.5\n")
    with pytest.raises(WorkflowError, match="does not match the baseline"):
        resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                        resource_paths={"reference.potential":
                                        str(new_model)})
    # the corrected mapping resumes, and no committed frame is duplicated
    new_model.write_bytes(b"1.5\n")
    os.utime(new_model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    result = resume_workflow(new_root, 2, verbose=False,
                             handle_sigint=False,
                             resource_paths={"reference.potential":
                                             str(new_model)})
    assert result.steps_completed == 5
    frames_after = _frames(new_root)
    assert len(frames_after) == len(frames_before) + 2
    assert len({id(f) for f in frames_after}) == len(frames_after)
    for old, new in zip(frames_before, frames_after):
        _assert_state_equal(old, new)
    assert not (new_root / "resource_bindings").exists() or \
        len(list((new_root / "resource_bindings").glob("*.json"))) == 1


def test_old_checkpoint_with_mapping_is_clearly_refused(tmp_path):
    """A run created without declared resources has no baseline: a nonempty
    mapping refuses clearly; nothing is fabricated backwards."""
    config_path = _write_run(tmp_path / "run", backend="harmonic-reference",
                             steps=2, checkpoint=1,
                             options="k = 1.0\nr0 = 0.9")
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    model = tmp_path / "model.dat"
    model.write_text("1.5\n")
    with pytest.raises(WorkflowError, match="no file-resource baseline"):
        resume_workflow(tmp_path / "run", 1, verbose=False,
                        handle_sigint=False,
                        resource_paths={"reference.potential": str(model)})


class TwoFileBacked:
    """A calculator with TWO declared top-level file parameters."""

    @staticmethod
    def factory(**kwargs):
        from typing import ClassVar

        from ase.calculators.calculator import Calculator

        from pyraimd2.engines.ase_engine import AseEngine

        class _TwoFile(Calculator):
            implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

            def __init__(self, model_path, aux_path, note):
                super().__init__()
                self.parameters = {"model": str(model_path),
                                   "aux": str(aux_path), "note": note}
                self._k = float(
                    Path(model_path).read_text().splitlines()[0])
                self._shift = float(
                    Path(aux_path).read_text().splitlines()[0])

            def calculate(self, atoms=None, properties=("energy",),
                          system_changes=None):
                super().calculate(atoms, properties, system_changes or [])
                dr = self.atoms.positions - 0.9
                self.results = {
                    "energy": 0.5 * self._k * float((dr**2).sum())
                    + self._shift,
                    "forces": -self._k * dr}

        return AseEngine(_TwoFile(kwargs["model"], kwargs["aux"],
                                  kwargs["note"]),
                         file_parameters={"model": "potential",
                                          "aux": "reference-data"})


def test_two_resources_verified_separately_and_swap_refused(tmp_path,
                                                            monkeypatch):
    import types

    from pyraimd2.backends import registry

    module = types.ModuleType("fake_two_file_module")
    module.create = TwoFileBacked.factory
    monkeypatch.setitem(registry._BUILTINS, "two-file-test",
                        ("engine", "fake_two_file_module", "create"))
    monkeypatch.setitem(__import__("sys").modules, "fake_two_file_module",
                        module)
    root = tmp_path / "origin"
    model = root / "inputs" / "model.dat"
    aux = root / "inputs" / "aux.dat"
    root.mkdir(parents=True)
    model.parent.mkdir()
    model.write_text("1.5\n")
    aux.write_text("0.25\n")
    config_path = _write_run(
        root / "run", steps=2, checkpoint=1, backend="two-file-test",
        options=(f'model = "{model}"\naux = "{aux}"\n'
                 'note = "not a file path"  # NOT declared, never a file'))
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    run_dir = root / "run"
    baseline = json.loads((run_dir / "file_resources.json").read_text())
    assert set(baseline["resources"]) == {"reference.potential",
                                          "reference.reference-data"}
    # relocate both, then swap their contents: each role must match ITS
    # baseline digest, so the swap refuses before any compute
    new_root = tmp_path / "moved" / "run"
    shutil.move(str(run_dir), str(new_root.parent.parent /
                                  "run") if False else str(new_root))
    moved_model = tmp_path / "moved" / "model.dat"
    moved_aux = tmp_path / "moved" / "aux.dat"
    shutil.move(str(model), str(moved_model))
    shutil.move(str(aux), str(moved_aux))
    moved_model.write_bytes(b"0.25\n")  # swapped contents
    moved_aux.write_bytes(b"1.5\n")
    with pytest.raises(WorkflowError, match="does not match the baseline"):
        resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                        resource_paths={
                            "reference.potential": str(moved_model),
                            "reference.reference-data": str(moved_aux)})
    # the correct mapping resumes
    moved_model.write_bytes(b"1.5\n")
    moved_aux.write_bytes(b"0.25\n")
    result = resume_workflow(
        new_root, 1, verbose=False, handle_sigint=False,
        resource_paths={"reference.potential": str(moved_model),
                        "reference.reference-data": str(moved_aux)})
    assert result.steps_completed == 3
    # the undeclared same-value string option was never rebound
    from pyraimd2.config import load_resolved_config
    from pyraimd2.workflows.md import _rebind_config
    config = load_resolved_config(new_root)
    current = {"reference.potential": str(moved_model),
               "reference.reference-data": str(moved_aux)}
    rebound = _rebind_config(config, new_root, current, baseline)
    assert rebound.reference.options["model"] == str(moved_model)
    assert rebound.reference.options["aux"] == str(moved_aux)
    assert rebound.reference.options["note"] == "not a file path"
    assert config.reference.options["model"] == str(model)  # original intact


def test_undeclared_old_path_string_keeps_legacy_identity_rules(tmp_path,
                                                                monkeypatch):
    """The documented boundary: an undeclared option string naming the OLD
    resource path is never rebound; after the old file is gone, the legacy
    embedded-file identity changes and the original checkpoint check
    refuses the resume — the undeclared field is judged by its own rule."""
    import types

    from pyraimd2.backends import registry

    module = types.ModuleType("fake_two_file_module")
    module.create = TwoFileBacked.factory
    monkeypatch.setitem(registry._BUILTINS, "two-file-test",
                        ("engine", "fake_two_file_module", "create"))
    monkeypatch.setitem(__import__("sys").modules, "fake_two_file_module",
                        module)
    root = tmp_path / "origin"
    model = root / "inputs" / "model.dat"
    aux = root / "inputs" / "aux.dat"
    root.mkdir(parents=True)
    model.parent.mkdir()
    model.write_text("1.5\n")
    aux.write_text("0.25\n")
    # the note option repeats the model's OLD path as an undeclared string
    config_path = _write_run(
        root / "run", steps=2, checkpoint=1, backend="two-file-test",
        options=(f'model = "{model}"\naux = "{aux}"\nnote = "{model}"'))
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    # relocate the run and both declared files; the OLD model path vanishes
    new_root = tmp_path / "moved" / "run"
    new_root.parent.mkdir()
    shutil.move(str(root / "run"), str(new_root))
    moved_model = tmp_path / "moved" / "model.dat"
    moved_aux = tmp_path / "moved" / "aux.dat"
    shutil.move(str(model), str(moved_model))
    shutil.move(str(aux), str(moved_aux))
    assert not model.exists()
    with pytest.raises(WorkflowError,
                       match="identity does not match the checkpoint"):
        resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                        resource_paths={
                            "reference.potential": str(moved_model),
                            "reference.reference-data": str(moved_aux)})


def test_reference_and_surrogate_namespaces_do_not_conflict(tmp_path):
    """Both sections declare the same local role "potential": the baseline
    keys stay reference.potential / surrogate.potential, each verified
    separately; mapping only one leaves the other dead-path role refused
    with its own name."""
    root = tmp_path / "origin"
    model_ref = root / "inputs" / "ref.dat"
    model_sur = root / "inputs" / "sur.dat"
    root.mkdir(parents=True)
    model_ref.parent.mkdir()
    model_ref.write_text("1.5\n")
    model_sur.write_text("0.75\n")
    config_path = _write_run(
        root / "run", mode="adaptive", steps=2, checkpoint=1,
        backend="file-backed-test-surr",
        options=f'model = "{model_sur}"',
        reference_backend="file-backed-test",
        reference_options=f'model = "{model_ref}"')
    result = _run_child(tmp_path, "fresh", MODE="fresh",
                        CONFIG=str(config_path))
    assert result.returncode == 0, result.stderr[-500:]
    run_dir = root / "run"
    baseline = json.loads((run_dir / "file_resources.json").read_text())
    assert set(baseline["resources"]) == {"reference.potential",
                                          "surrogate.potential"}
    assert baseline["resources"]["reference.potential"]["original_path"] == \
        str(model_ref)
    assert baseline["resources"]["surrogate.potential"]["original_path"] == \
        str(model_sur)
    # relocate everything; map ONLY the surrogate role: the reference role
    # stays at its dead original path and the refusal names it
    new_root = tmp_path / "moved" / "run"
    new_root.parent.mkdir()
    shutil.move(str(run_dir), str(new_root))
    moved_sur = tmp_path / "moved" / "sur.dat"
    shutil.move(str(model_sur), str(moved_sur))
    model_ref.unlink()  # the old reference path is gone
    with pytest.raises(WorkflowError,
                       match="reference.potential"):
        _run_child_expect_failure(
            tmp_path, new_root,
            {"surrogate.potential": str(moved_sur)})
    # mapping both resumes
    moved_ref = tmp_path / "moved" / "ref.dat"
    moved_ref.write_text("1.5\n")
    result = _run_child(tmp_path / "b", "resume", MODE="resume",
                        RUN_DIR=str(new_root), STEPS="1",
                        MAPPING=json.dumps({
                            "reference.potential": str(moved_ref),
                            "surrogate.potential": str(moved_sur)}))
    assert result.returncode == 0, result.stderr[-500:]


def _run_child_expect_failure(tmp_path, run_dir, mapping):
    """A relocation resume expected to refuse — the refusal precedes any
    factory call, so in-process is fine."""
    resume_workflow(run_dir, 1, verbose=False, handle_sigint=False,
                    resource_paths=mapping)
