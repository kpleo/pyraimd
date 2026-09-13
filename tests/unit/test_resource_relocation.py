"""T3/T4 acceptance: relocate a run and its declared resource files, then
resume and export through the Python workflow API.

Real process boundaries: process A runs N steps and exits; the run
directory and the resource files move to a new path (with spaces and
non-ASCII characters); the old resource location is removed; process B
resumes with an explicit mapping for M steps.  Deterministic NVE state,
driving forces, step counts, physical time and the adaptive route/check
decisions agree with the continuous control at rtol=0, atol=1e-12.
Analytic file-backed test backends only — zero real DFT budget.
"""

from __future__ import annotations

import hashlib
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
    _inject,
    _write_run,
)

from pyraimd2.config import load_config, load_resolved_config
from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.store import Store
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.export import export_run, frames_for_run
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
                self.calls = 0

            def calculate(self, atoms=None, properties=("energy",),
                          system_changes=None):
                super().calculate(atoms, properties, system_changes or [])
                self.calls += 1
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


def _inject_test_backend(monkeypatch, factory=None):
    # the SAME factory the child processes use — a different calculator
    # class would be a different physical identity by design
    _inject(monkeypatch, "engine", factory or FileBacked.factory)


class CountingFactory:
    """Counts in-process factory calls and the calculators' compute calls."""

    def __init__(self):
        self.factory_calls = 0
        self.calculators = []

    def engine(self, **kwargs):
        self.factory_calls += 1
        adapter = FileBacked.factory(**kwargs)
        self.calculators.append(adapter.calculator)
        return adapter

    @property
    def compute_calls(self):
        return sum(calculator.calls for calculator in self.calculators)


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


def _events(run_dir):
    return [json.loads(s) for s in
            (Path(run_dir) / "events.jsonl").read_text().splitlines()]


def _commits(run_dir):
    return _commits_from(_events(run_dir))


def _commits_from(events):
    return [e for e in events if e["type"] == "evaluation_committed"]


def _step_boundaries(run_dir):
    return sorted(int(e["step_id"]) for e in _events(run_dir)
                  if e["type"] == "step_completed")


def _snapshot_records(run_dir):
    """Byte-level snapshot of everything a refused resume must not touch:
    the event log, the trajectory database, every checkpoint generation's
    files and the number of binding receipts."""
    run_dir = Path(run_dir)
    checkpoints = {}
    for gen_dir in sorted((run_dir / "checkpoints").iterdir()):
        if gen_dir.is_dir() and gen_dir.name.isdigit():
            checkpoints[gen_dir.name] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(gen_dir.iterdir()) if p.is_file()}
    bindings = run_dir / "resource_bindings"
    return {
        "events": (run_dir / "events.jsonl").read_bytes(),
        "db": (run_dir / "trajectory.db").read_bytes(),
        "checkpoints": checkpoints,
        "receipts": len(list(bindings.glob("*.json"))) if bindings.is_dir()
        else 0,
    }


def _assert_same_trajectory(relocated_dir, control_dir, *, total_steps=5):
    """The relocated 3+2 run and the uninterrupted 5-step control agree on
    every persisted trajectory fact: completed step boundaries, the frame
    sequence's own identifiers (step/evaluation ids and phases — run-local
    sequence numbers, never cross-run random ids), physical times, driving
    forces/energies, positions and momenta."""
    relocated_frames = _frames(relocated_dir)
    control_frames = _frames(control_dir)
    assert len(relocated_frames) == len(control_frames) == total_steps + 1
    step_ids = [f.info["step_id"] for f in relocated_frames]
    assert step_ids == [-1] + list(range(total_steps))
    assert step_ids == [f.info["step_id"] for f in control_frames]
    assert [f.info["evaluation_id"] for f in relocated_frames] == \
        list(range(total_steps + 1))
    assert [f.info["evaluation_id"] for f in relocated_frames] == \
        [f.info["evaluation_id"] for f in control_frames]
    assert [f.info["integration_phase"] for f in relocated_frames] == \
        ["initial_evaluation"] + ["complete_step"] * total_steps
    assert _step_boundaries(relocated_dir) == list(range(total_steps))
    assert _step_boundaries(control_dir) == list(range(total_steps))
    for relocated, control in zip(relocated_frames, control_frames):
        _assert_state_equal(relocated, control)
        np.testing.assert_allclose(
            relocated.arrays["forces"], control.arrays["forces"], rtol=0,
            atol=1e-12)
        np.testing.assert_allclose(
            relocated.info["energy"], control.info["energy"], rtol=0,
            atol=1e-12)
        np.testing.assert_allclose(
            relocated.info["physical_time_fs"],
            control.info["physical_time_fs"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        [f.info["physical_time_fs"] for f in relocated_frames],
        np.arange(total_steps + 1) * 0.5, rtol=0, atol=1e-12)


def _assert_same_decisions(relocated_dir, control_dir, *, total_steps=5):
    """Adaptive only: the discrete route/check decisions of corresponding
    evaluations agree, and every checked commit pairs with the verification
    task that produced its label (run-local ids continue, none restart)."""
    relocated_commits = _commits(relocated_dir)
    control_commits = _commits(control_dir)
    assert [int((e.get("context") or {})["evaluation_id"])
            for e in relocated_commits] == list(range(total_steps + 1))

    def decisions(commits):
        return [(e["route"], bool(e["checked"]), e["reason"],
                 e.get("label_id"), e.get("segment_id"), e.get("model_id"),
                 {k: int(v) for k, v in
                  (e.get("reference_calls_this_evaluation") or {}).items()})
                for e in commits]

    assert decisions(relocated_commits) == decisions(control_commits)
    for run_dir, commits in ((relocated_dir, relocated_commits),
                             (control_dir, control_commits)):
        tasks = [e for e in _events(run_dir) if e["type"] == "task"]
        checks = [e for e in commits if e["checked"]]
        assert checks, "the scenario must contain real checked evaluations"
        for commit in checks:
            evaluation_id = int(commit["context"]["evaluation_id"])
            verification = [t for t in tasks
                            if t.get("operation") == "reference"
                            and t.get("purpose") == "verification"
                            and int(t.get("evaluation_id")) == evaluation_id]
            assert len(verification) == 1
            assert verification[0]["status"] == "success"
            assert verification[0].get("label_id") == commit["label_id"]


def _row_digests(run_dir, run_id="file-res"):
    with Store(Path(run_dir) / "trajectory.db") as store:
        rows = sorted(store._db.select(run_id=run_id), key=lambda r: r.id)
        return {int(row.id): (int(row.key_value_pairs["step"]),
                              Store.row_digest(row)) for row in rows}


def _checkpoint_digests(run_dir):
    out = {}
    for gen_dir in sorted((Path(run_dir) / "checkpoints").iterdir()):
        if gen_dir.is_dir() and gen_dir.name.isdigit():
            out[int(gen_dir.name)] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(gen_dir.iterdir()) if p.is_file()}
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
    _assert_same_trajectory(new_root, control_root / "run")
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
        _assert_same_trajectory(new_root, control_root / "run")
        if mode == "adaptive":
            _assert_same_decisions(new_root, control_root / "run")


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
    _assert_same_trajectory(new_root, control_root / "run")
    _assert_same_decisions(new_root, control_root / "run")
    # the thermostat state, the check stream state and the per-evaluation
    # check flags match the control exactly — the random streams continued,
    # never re-seeded.  check_rng is the persisted generator state itself:
    # a real mapping (bit_generator + position), never default markers.
    relocated_states = _checkpoint_states(new_root)
    control_states = _checkpoint_states(control_root / "run")
    assert relocated_states[-1]["thermostat"] == \
        control_states[-1]["thermostat"]
    relocated_rng = relocated_states[-1]["check_rng"]
    control_rng = control_states[-1]["check_rng"]
    assert relocated_rng == control_rng
    assert relocated_rng["bit_generator"] and relocated_rng["state"]
    relocated_checks = [bool(e.get("checked")) for e in _commits(new_root)]
    control_checks = [bool(e.get("checked"))
                      for e in _commits(control_root / "run")]
    assert relocated_checks == control_checks
    assert any(relocated_checks)  # real checks happened — not an empty
    # or all-default sequence passing for equality


def test_relocation_rejections_and_retry(tmp_path, monkeypatch):
    """Missing mapped file, unknown role, one-byte change (same size and
    mtime) and wrong content all refuse before any factory call — zero new
    factory/compute calls and byte-identical records per refusal; a
    corrected mapping then resumes, adding exactly the new steps' committed
    records without duplicating or rewriting persisted history."""
    counter = CountingFactory()
    # the in-process resume needs the factory registered here too
    _inject_test_backend(monkeypatch, factory=counter.engine)
    root, model = _prepare_run(tmp_path)
    new_root, new_model = _relocate(tmp_path, root, model)
    frames_before = _frames(new_root)
    rows_before = _row_digests(new_root)
    records_before = _snapshot_records(new_root)
    commits_before = _commits(new_root)
    assert records_before["receipts"] == 0

    def assert_refused(mapping, match):
        factory_before = counter.factory_calls
        compute_before = counter.compute_calls
        with pytest.raises(WorkflowError, match=match):
            resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                            resource_paths=mapping)
        # refused before any factory or compute; nothing was appended,
        # rewritten or rebound anywhere in the run directory
        assert counter.factory_calls == factory_before
        assert counter.compute_calls == compute_before
        assert _snapshot_records(new_root) == records_before

    # unknown role key
    assert_refused({"reference.unknown": str(new_model)},
                   "not a declared resource")
    # non-absolute path
    assert_refused({"reference.potential": "relative/model.dat"}, "absolute")
    # a missing mapped file
    missing = tmp_path / "nowhere" / "model.dat"
    assert_refused({"reference.potential": str(missing)},
                   "not an existing regular file")
    # a one-byte change with the same size and preserved mtime
    stat = new_model.stat()
    new_model.write_bytes(b"2.5\n")
    os.utime(new_model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert_refused({"reference.potential": str(new_model)},
                   "does not match the baseline")
    # a valid file with wrong content (one extra line, same family)
    new_model.write_bytes(b"1.5\n2.5\n")
    assert_refused({"reference.potential": str(new_model)},
                   "does not match the baseline")
    # the corrected mapping resumes: exactly one factory call, one compute
    # per new step, and the new content is identified by persisted
    # step/evaluation ids and events — old rows keep their digests
    new_model.write_bytes(b"1.5\n")
    os.utime(new_model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    result = resume_workflow(new_root, 2, verbose=False,
                             handle_sigint=False,
                             resource_paths={"reference.potential":
                                             str(new_model)})
    assert result.steps_completed == 5
    assert counter.factory_calls == 1
    assert counter.compute_calls == 2
    assert _step_boundaries(new_root) == [0, 1, 2, 3, 4]
    commits_after = _commits(new_root)
    assert [int((e.get("context") or {})["evaluation_id"])
            for e in commits_after] == [0, 1, 2, 3, 4, 5]
    # the pre-relocation commits are untouched; the resume appended ev 4, 5
    assert [json.dumps(e, sort_keys=True) for e in
            commits_after[:len(commits_before)]] == \
        [json.dumps(e, sort_keys=True) for e in commits_before]
    rows_after = _row_digests(new_root)
    assert len(rows_after) == len(rows_before) + 2
    for row_id, digest in rows_before.items():
        assert rows_after[row_id] == digest  # history rows byte-identical
    new_row_ids = sorted(set(rows_after) - set(rows_before))
    assert [rows_after[row_id][0] for row_id in new_row_ids] == [3, 4]
    frames_after = _frames(new_root)
    assert len(frames_after) == len(frames_before) + 2
    assert [f.info["step_id"] for f in frames_after] == [-1, 0, 1, 2, 3, 4]
    for old, new in zip(frames_before, frames_after):
        _assert_state_equal(old, new)
        assert old.info["step_id"] == new.info["step_id"]
        assert old.info["evaluation_id"] == new.info["evaluation_id"]
    receipts = list((new_root / "resource_bindings").glob("*.json"))
    assert len(receipts) == 1


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
    # snapshot the pre-relocation checkpoint generations by content: a
    # generation surviving the resume's retention must be byte-identical
    checkpoints_before = _checkpoint_digests(new_root)
    association_before = json.loads(
        (new_root / "checkpoints" / str(max(checkpoints_before)) /
         "state.json").read_text())["file_resource_baseline_sha256"]
    result = resume_workflow(
        new_root, 1, verbose=False, handle_sigint=False,
        resource_paths={"reference.potential": str(moved_model),
                        "reference.reference-data": str(moved_aux)})
    assert result.steps_completed == 3
    checkpoints_after = _checkpoint_digests(new_root)
    retained = set(checkpoints_before) & set(checkpoints_after)
    assert retained  # retention may prune, but this 2+1 case keeps one
    for generation in retained:
        assert checkpoints_after[generation] == \
            checkpoints_before[generation]  # never rewritten
    new_generations = set(checkpoints_after) - set(checkpoints_before)
    assert new_generations
    for generation in new_generations:
        state = json.loads(
            (new_root / "checkpoints" / str(generation) / "state.json")
            .read_text())
        # the new generation keeps the same baseline association
        assert state["file_resource_baseline_sha256"] == association_before
    # the undeclared same-value string option was never rebound
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


def test_non_file_parameter_identity_refusal_short_regression(
        tmp_path, monkeypatch):
    """Short regression of the reviewer's independent three-mode check
    (reviews/N003_identity_check/evidence.json): a non-file parameter
    changed after relocation refuses on physical identity before any
    compute; the restored configuration then resumes and keeps the
    association.  Reference mode, reusing the shared fixtures."""
    from test_file_resource_baseline import FileBacked as BaselineFileBacked

    counts = {"factory": 0, "compute": 0}

    class Scaled(BaselineFileBacked):
        def __init__(self, path, scale):
            super().__init__(path)
            self.parameters["scale"] = float(scale)
            self._k *= float(scale)

        def calculate(self, *args, **kwargs):
            counts["compute"] += 1
            super().calculate(*args, **kwargs)

    def factory(**kwargs):
        counts["factory"] += 1
        return AseEngine(Scaled(kwargs["model"], kwargs["scale"]),
                         file_parameters={"model": "potential"})

    _inject(monkeypatch, "engine", factory)
    root = tmp_path / "origin"
    model = root / "inputs" / "model.dat"
    model.parent.mkdir(parents=True)
    model.write_text("1.5\n")
    config_path = _write_run(root / "run", steps=2, checkpoint=1,
                             options=f'model = "{model}"\nscale = 1.0')
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    new_root, new_model = _relocate(tmp_path, root, model)
    config_file = new_root / "resolved_config.json"
    original_config = config_file.read_bytes()
    edited = json.loads(original_config)
    edited["reference"]["options"]["scale"] = 2.0
    config_file.write_text(json.dumps(edited))
    before = _snapshot_records(new_root)
    counts_before = dict(counts)
    with pytest.raises(WorkflowError, match="identity"):
        resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                        resource_paths={"reference.potential":
                                        str(new_model)})
    assert counts["factory"] == counts_before["factory"] + 1  # rebuilt once
    assert counts["compute"] == counts_before["compute"]  # zero new compute
    assert _snapshot_records(new_root) == before
    # the configuration restored byte-for-byte, the valid mapping resumes
    config_file.write_bytes(original_config)
    result = resume_workflow(new_root, 1, verbose=False, handle_sigint=False,
                             resource_paths={"reference.potential":
                                             str(new_model)})
    assert result.steps_completed == 3
    receipts = list((new_root / "resource_bindings").glob("*.json"))
    assert len(receipts) == 1
    baseline_digest = hashlib.sha256(
        (new_root / "file_resources.json").read_bytes()).hexdigest()
    assert _checkpoint_states(new_root)[-1][
        "file_resource_baseline_sha256"] == baseline_digest
    assert config_file.read_bytes() == original_config


def test_adaptive_export_and_costs_after_relocation(tmp_path):
    """On one adaptive 3+2 relocation: history rows and task/label events
    are preserved; the resume's new costs correspond to real new calls with
    no re-recorded history; after every external model file is deleted the
    driving/reference/base exports stay honest — missing reference labels
    are NaN-marked, never zeros."""
    root, model = _prepare_run(tmp_path, mode="adaptive")
    events_before = _events(root / "run")
    rows_before = _row_digests(root / "run")
    costs_before = summarize_tasks(events_before)
    history_before = _history_bytes(root / "run")
    new_root, new_model = _relocate(tmp_path, root, model)
    result = _run_child(
        tmp_path, "resume", MODE="resume", RUN_DIR=str(new_root),
        STEPS="2", MAPPING=json.dumps({"surrogate.potential":
                                       str(new_model)}))
    assert result.returncode == 0, result.stderr[-500:]
    events_after = _events(new_root)

    # --- persistence: old records preserved, new ones appended ------------
    rows_after = _row_digests(new_root)
    for row_id, digest in rows_before.items():
        assert rows_after[row_id] == digest
    new_row_ids = sorted(set(rows_after) - set(rows_before))
    assert [rows_after[row_id][0] for row_id in new_row_ids] == [3, 4]
    task_ids_before = [e["task_id"] for e in events_before
                       if e["type"] == "task"]
    tasks_after = [e for e in events_after if e["type"] == "task"]
    task_ids_after = [e["task_id"] for e in tasks_after]
    assert len(task_ids_after) == len(set(task_ids_after))  # no duplicates
    assert set(task_ids_before) < set(task_ids_after)
    labels_before = [e["label_id"] for e in _commits_from(events_before)
                     if e.get("label_id")]
    labels_after = [e["label_id"] for e in _commits_from(events_after)
                    if e.get("label_id")]
    assert labels_after[:len(labels_before)] == labels_before
    new_labels = labels_after[len(labels_before):]
    assert len(new_labels) == len(set(new_labels))  # each new label once
    # new events by kind: per resumed step one proposal+attempt+verification
    # +io+commit+boundary; plus the resume bracket (resumed, run_summary)
    new_events = events_after[len(events_before):]
    by_type: dict[str, int] = {}
    for event in new_events:
        by_type[event["type"]] = by_type.get(event["type"], 0) + 1
    assert by_type == {"resumed": 1, "task": 6, "evaluation_proposed": 2,
                       "attempt": 2, "evaluation_committed": 2,
                       "step_completed": 2, "run_summary": 1}
    new_tasks = [e for e in new_events if e["type"] == "task"]
    assert sorted((t["operation"], t["purpose"]) for t in new_tasks) == \
        [("inference", "proposal"), ("inference", "proposal"),
         ("io", "trajectory_append"), ("io", "trajectory_append"),
         ("reference", "verification"), ("reference", "verification")]
    assert all(t["status"] == "success" for t in new_tasks)
    assert not any(t.get("cache_hit") for t in new_tasks)
    assert not any(e["type"] == "label_consumed" for e in new_events)

    # --- costs: the ledger's new entries are exactly the real new calls ---
    costs_after = summarize_tasks(events_after)
    ref_before, ref_after = costs_before["reference"], \
        costs_after["reference"]
    assert ref_after["logical_requests"] == \
        ref_before["logical_requests"] + 2
    assert ref_after["actual_executions"] == \
        ref_before["actual_executions"] + 2
    assert ref_after["successful_executions"] == \
        ref_before["successful_executions"] + 2
    assert ref_after["failed_attempts"] == ref_before["failed_attempts"]
    assert ref_after["cache_hits"] == ref_before["cache_hits"]
    assert ref_after["unresolved_attempts"] == 0
    assert costs_after["cost_record_complete"] is True
    new_check_calls = sum(int((e.get("reference_calls_this_evaluation") or {})
                              .get("check", 0))
                          for e in _commits_from(new_events))
    assert new_check_calls == 2  # the two verification tasks are the calls

    # --- export after every external model file is gone -------------------
    assert not model.exists()  # the old location died at relocation
    new_model.unlink()
    with Store(new_root / "trajectory.db") as store:
        rows = {int(row.key_value_pairs["step"]): row for row in
                store._db.select(run_id="file-res")}
        driving = frames_for_run(store, new_root, "file-res",
                                 force_source="driving")
        reference = frames_for_run(store, new_root, "file-res",
                                   force_source="reference")
        base = frames_for_run(store, new_root, "file-res",
                              force_source="base")
    assert [f.info["step_id"] for f in driving] == [-1, 0, 1, 2, 3, 4]
    # every exported force/energy is the row's stored payload for the
    # requested source, byte value for value
    for frame in driving:
        payload = rows[frame.info["step_id"]].data["driving"]
        np.testing.assert_allclose(frame.arrays["forces"],
                                   payload["forces"], rtol=0, atol=0)
        assert frame.info["energy"] == payload["energy"]
    for frame in base:
        payload = rows[frame.info["step_id"]].data["surrogate"]
        np.testing.assert_allclose(frame.arrays["forces"],
                                   payload["forces"], rtol=0, atol=0)
        assert frame.info["energy"] == payload["energy"]
    # force-source meaning on the dft-routed row: driving IS the reference
    assert rows[-1].key_value_pairs["route"] == "dft"
    initial = rows[-1].data
    np.testing.assert_allclose(initial["driving"]["forces"],
                               initial["engine"]["forces"], rtol=0, atol=0)
    # reference source: the genuinely unlabeled frames (unchecked accepted
    # evaluations at steps 0-2) export NaN forces with forces_available
    # False and no energy/label — missing data, never zeros
    missing, labeled = [], []
    for frame in reference:
        step = frame.info["step_id"]
        engine_payload = rows[step].data.get("engine")
        if engine_payload is None:
            missing.append(frame)
            assert frame.info["forces_available"] is False
            assert np.isnan(frame.arrays["forces"]).all()
            assert "energy" not in frame.info
            assert "reference_label_id" not in frame.info
        else:
            labeled.append(frame)
            np.testing.assert_allclose(frame.arrays["forces"],
                                       engine_payload["forces"], rtol=0,
                                       atol=0)
            assert frame.info["reference_label_id"] == \
                rows[step].data["engine_label_id"]
    assert [f.info["step_id"] for f in missing] == [0, 1, 2]
    assert [f.info["step_id"] for f in labeled] == [-1, 3, 4]
    report = export_run(new_root, force_source="reference",
                        output=tmp_path / "export-reference.extxyz")
    assert report["frames"] == 6
    assert report["missing_forces_frames"] == 3
    # nothing about the run's own metadata/history moved during export
    after = _history_bytes(new_root)
    for name in ("config.toml", "manifest.json", "resolved_config.json",
                 "file_resources.json"):
        assert after[name] == history_before[name], name
    assert after["events_prefix"].startswith(
        history_before["events_prefix"])
    assert after["events_prefix"] == (new_root / "events.jsonl") \
        .read_bytes()  # export appended nothing
