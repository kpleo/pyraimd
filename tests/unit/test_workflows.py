"""Workflow-level tests (WP04): run-directory layout, the three MD modes,
resume semantics, output artifacts and the continuous-vs-resumed identity
of the full config-driven adaptive path.

Everything runs on the builtin analytic harmonic backends — no external
programs, no RNG beyond the configured seeds.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pyraimd2.config import load_config
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.workflows import (
    ExportError,
    WorkflowError,
    export_run,
    resume_workflow,
    run_workflow,
)
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE


def make_config(tmp_path: Path, *, steps: int = 12, mode: str = "adaptive",
                checkpoint_interval: int = 3, trajectory_interval: int = 1,
                summary_interval: int = 2, run_id: str = "wf-test") -> Path:
    """A small harmonic config in tmp_path (fast, deterministic)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = HARMONIC_CONFIG
    text = text.replace('id = "harmonic-demo"', f'id = "{run_id}"')
    text = text.replace('directory = "runs/harmonic-demo"',
                        f'directory = "runs/{run_id}"')
    text = text.replace("steps = 20", f"steps = {steps}")
    text = text.replace("interval_steps = 5",
                        f"interval_steps = {checkpoint_interval}")
    text = text.replace("trajectory_interval_steps = 1",
                        f"trajectory_interval_steps = {trajectory_interval}")
    text = text.replace("summary_interval_steps = 10",
                        f"summary_interval_steps = {summary_interval}")
    if mode != "adaptive":
        text = text.replace('mode = "adaptive"', f'mode = "{mode}"')
        # plain modes use neither the unused backend section nor the
        # adaptive-only policy/verification sections; sections start at
        # line-anchored '[' (array values contain '[' too)
        unused = "[surrogate]" if mode == "reference" else "[reference]"
        for section in (unused, "[policy]", "[verification]"):
            start = text.index(section)
            following = text.index("\n[", start + 1)
            text = text[:start] + text[following + 1:]
    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    config_path = tmp_path / "run.toml"
    config_path.write_text(text)
    return config_path


def run_dir_of(config) -> Path:
    return config.run.directory


def rows(run_dir: Path, run_id: str) -> list:
    store = Store(run_dir / "trajectory.db")
    return sorted(store._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


# ---------------------------------------------------------------------------
# adaptive mode: layout and records


def test_adaptive_run_writes_the_full_layout(tmp_path) -> None:
    config = load_config(make_config(tmp_path))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = result.run_dir
    assert result.steps_completed == 12
    for name in ("config.toml", "resolved_config.json", "manifest.json",
                 "trajectory.db", "events.jsonl", "summary.json",
                 "summary.csv", "trajectory.extxyz"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "checkpoints" / "latest.json").is_file()
    assert (run_dir / "models").is_dir()

    # config.toml is the verbatim original; resolved config carries the
    # effective parameters with absolute paths and units
    assert (run_dir / "config.toml").read_text() == \
        Path(config.source_path).read_text()
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    assert resolved["units"]["force"] == "eV/angstrom"
    assert Path(resolved["run"]["directory"]).is_absolute()
    assert resolved["dynamics"]["steps"] == 12

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "wf-test"
    assert manifest["reference"]["fingerprint"].startswith("harmonic-reference:")
    assert manifest["surrogate"]["fingerprint"].startswith("harmonic-surrogate:")
    assert manifest["structure"]["sha256"]

    info = inspect_run(run_dir)
    assert info["n_complete_steps"] == 12
    assert info["physical_time_fs"] == pytest.approx(12 * 0.5)
    assert info["cost"]["reference"]["actual_executions"] > 0

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_complete_steps"] == info["n_complete_steps"]
    assert (run_dir / "summary.csv").read_text().count("\n") == 14  # header + 13


def test_trajectory_preview_is_complete_and_driving(tmp_path) -> None:
    from ase.io import read as ase_read

    config = load_config(make_config(tmp_path))
    run_workflow(config, verbose=False, handle_sigint=False)
    frames = ase_read(config.run.directory / "trajectory.extxyz", index=":")
    assert len(frames) == 13  # initial evaluation + 12 steps
    for frame in frames:
        assert frame.info["force_source"] == "driving"
        assert np.isfinite(frame.get_forces()).all()


def test_thinned_trajectory_preview(tmp_path) -> None:
    from ase.io import read as ase_read

    config = load_config(make_config(tmp_path, trajectory_interval=3))
    run_workflow(config, verbose=False, handle_sigint=False)
    frames = ase_read(config.run.directory / "trajectory.extxyz", index=":")
    evaluation_ids = [frame.info["evaluation_id"] for frame in frames]
    assert evaluation_ids == [0, 3, 6, 9, 12]


# ---------------------------------------------------------------------------
# resume


def test_resume_adds_steps_and_reports_current_and_target(tmp_path, capsys) -> None:
    config = load_config(make_config(tmp_path, steps=8))
    run_workflow(config, verbose=False, handle_sigint=False)
    result = resume_workflow(config.run.directory, 4, verbose=True)
    output = capsys.readouterr().out
    assert "at complete step 8" in output
    assert "4 additional steps (target 12)" in output
    assert result.steps_completed == 12
    assert result.steps_this_call == 4
    events = [json.loads(line) for line in
              (config.run.directory / "events.jsonl").read_text().splitlines()
              if line.strip()]
    assert any(event.get("type") == "resumed" for event in events)


def test_continuous_vs_stop_resume_are_identical(tmp_path) -> None:
    """The whole config-driven path keeps WP03's identity guarantee."""
    continuous = load_config(make_config(tmp_path / "cont", steps=12))
    run_workflow(continuous, verbose=False, handle_sigint=False)

    split = load_config(make_config(tmp_path / "split", steps=8))
    run_workflow(split, verbose=False, handle_sigint=False)
    resume_workflow(split.run.directory, 4, verbose=False)

    run_id = "wf-test"
    continuous_rows = rows(continuous.run.directory, run_id)
    split_rows = rows(split.run.directory, run_id)
    assert len(continuous_rows) == len(split_rows) == 13
    for first, second in zip(continuous_rows, split_rows, strict=True):
        assert first.key_value_pairs["route"] == second.key_value_pairs["route"]
        np.testing.assert_allclose(first.toatoms().positions,
                                   second.toatoms().positions, atol=1e-12)
        np.testing.assert_allclose(first.toatoms().get_momenta(),
                                   second.toatoms().get_momenta(), atol=1e-12)
        first_meta = first.data.get("metadata") or {}
        second_meta = second.data.get("metadata") or {}
        assert first_meta.get("checked") == second_meta.get("checked")
    first_info = inspect_run(continuous.run.directory)
    second_info = inspect_run(split.run.directory)
    assert (first_info["checks"]["accepted_count"]
            == second_info["checks"]["accepted_count"])


def test_resume_continues_plain_mode_runs(tmp_path) -> None:
    config = load_config(make_config(tmp_path, mode="reference"))
    run_workflow(config, verbose=False, handle_sigint=False)
    result = resume_workflow(config.run.directory, 2, verbose=False,
                             handle_sigint=False)
    assert result.steps_this_call == 2
    assert result.steps_completed == config.dynamics.steps + 2


def test_sigint_stops_at_a_step_boundary_and_resume_continues(tmp_path) -> None:
    config = load_config(make_config(tmp_path, steps=20000,
                                     checkpoint_interval=2))

    def fire_when_running() -> None:
        # send SIGINT only once the runner owns the handler (run_start
        # written); signalling earlier would hit pytest's own handler
        events = config.run.directory / "events.jsonl"
        for _ in range(3000):
            if events.exists():
                time.sleep(0.2)
                signal.raise_signal(signal.SIGINT)
                return
            time.sleep(0.01)

    timer = threading.Thread(target=fire_when_running, daemon=True)
    timer.start()
    result = run_workflow(config, verbose=False, handle_sigint=True)
    timer.join(timeout=5)
    assert result.stopped_early
    assert 0 < result.steps_completed < 20000
    assert (config.run.directory / "checkpoints" / "latest.json").is_file()
    resumed = resume_workflow(config.run.directory, 3, verbose=False)
    assert resumed.steps_completed == result.steps_completed + 3


def test_run_refuses_an_existing_run_directory(tmp_path) -> None:
    config = load_config(make_config(tmp_path))
    run_workflow(config, verbose=False, handle_sigint=False)
    with pytest.raises(WorkflowError, match="already exists"):
        run_workflow(config, verbose=False, handle_sigint=False)


# ---------------------------------------------------------------------------
# plain modes


def test_reference_mode_routes_and_exports(tmp_path) -> None:
    config = load_config(make_config(tmp_path, mode="reference", steps=5))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 5
    stored = rows(config.run.directory, "wf-test")
    assert len(stored) == 6
    assert {str(row.key_value_pairs["route"]) for row in stored} == {"dft"}
    info = inspect_run(config.run.directory)
    assert info["n_complete_steps"] == 5
    assert info["cost"]["reference"]["actual_executions"] == 6

    report = export_run(config.run.directory, force_source="reference",
                        force=True)
    assert report["frames"] == 6
    assert report["missing_forces_frames"] == 0
    report = export_run(config.run.directory, force_source="driving", force=True)
    assert report["frames"] == 6


def test_surrogate_mode_routes_and_missing_reference_marks(tmp_path) -> None:
    config = load_config(make_config(tmp_path, mode="surrogate", steps=5))
    run_workflow(config, verbose=False, handle_sigint=False)
    stored = rows(config.run.directory, "wf-test")
    assert {str(row.key_value_pairs["route"]) for row in stored} == {"ml"}
    # surrogate-only runs have no reference labels at all: every exported
    # reference frame must carry the missing mark, never zeros
    report = export_run(config.run.directory, force_source="reference",
                        force=True)
    assert report["frames"] == 6
    assert report["missing_forces_frames"] == 6


def test_export_refuses_to_overwrite_without_force(tmp_path) -> None:
    config = load_config(make_config(tmp_path, steps=3))
    run_workflow(config, verbose=False, handle_sigint=False)
    output = config.run.directory / "out.extxyz"
    export_run(config.run.directory, output=output)
    with pytest.raises(ExportError, match="--force"):
        export_run(config.run.directory, output=output)
    export_run(config.run.directory, output=output, force=True)
