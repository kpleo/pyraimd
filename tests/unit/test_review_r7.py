"""Review R7 regression: every user-visible output consumes the shared
committed-row view (0.4.2 plan §R7) — regenerate, incremental append,
summary CSV, high- and low-level export agree; raw/legacy mode is
explicit, never mistaken for the authoritative view."""

from __future__ import annotations

import csv
import io

import pytest
from ase.io import read as ase_read
from test_review_c1 import fail_commit_of, make_world, resume

from pyraimd2.runtime.inspect import summary_csv
from pyraimd2.store import Store
from pyraimd2.workflows.export import (
    ExportError,
    export_run,
    frames_for_run,
    frames_from_store,
)
from pyraimd2.workflows.setup import RunOutputs


@pytest.fixture
def orphan_run(tmp_path):
    """row 1 = initial, row 2 = orphan at step 0, row 3 = committed."""
    run_dir = tmp_path / "r7"
    runner, _engine = make_world(run_dir)
    runner.run(0)
    fail_commit_of(runner, 1)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(1)
    del runner
    resumed = resume(run_dir)
    resumed.run(1)
    resumed.close()
    return run_dir


def _events(run_dir):
    import json

    return [json.loads(line) for line in
            (run_dir / "events.jsonl").read_text().splitlines()]


def test_r7_high_level_export_excludes_the_orphan(orphan_run):
    report = export_run(orphan_run)
    frames = ase_read(report["output"], index=":")
    assert [int(f.info["evaluation_id"]) for f in frames] == [0, 1]
    assert [int(f.info["row_id"]) for f in frames] == [1, 3]


def test_r7_frames_for_run_matches_export(orphan_run):
    store = Store(orphan_run / "trajectory.db")
    frames = frames_for_run(store, orphan_run, "run", force_source="driving")
    assert [f.info["evaluation_id"] for f in frames] == [0, 1]
    assert [f.info["row_id"] for f in frames] == [1, 3]


def test_r7_default_trajectory_regenerate_excludes_the_orphan(orphan_run):
    outputs = RunOutputs(orphan_run, "run")
    outputs.regenerate_trajectory()
    frames = ase_read(orphan_run / "trajectory.extxyz", index=":")
    assert [int(f.info["evaluation_id"]) for f in frames] == [0, 1]
    assert [int(f.info["row_id"]) for f in frames] == [1, 3]


def test_r7_incremental_append_selects_the_committed_row(orphan_run):
    outputs = RunOutputs(orphan_run, "run")
    outputs.append_trajectory_step(1)  # the re-executed evaluation
    frames = ase_read(orphan_run / "trajectory.extxyz", index=":")
    assert len(frames) == 1
    assert int(frames[0].info["row_id"]) == 3


def test_r7_summary_csv_uses_committed_rows(orphan_run):
    store = Store(orphan_run / "trajectory.db")
    rows = list(csv.DictReader(io.StringIO(
        summary_csv(store, "run", events=_events(orphan_run)))))
    assert [int(r["evaluation_id"]) for r in rows] == [0, 1]
    legacy = list(csv.DictReader(io.StringIO(summary_csv(store, "run"))))
    assert len(legacy) == 3  # explicit raw mode: all rows, orphans included


def test_r7_complete_steps_without_committed_is_refused(orphan_run):
    store = Store(orphan_run / "trajectory.db")
    with pytest.raises(ExportError, match="authoritative|committed"):
        frames_from_store(store, "run", force_source="driving",
                          complete_steps={0})
    raw = frames_from_store(store, "run", force_source="driving")
    assert len(raw) == 3  # explicit raw/legacy selection still available


def test_r7_initial_frame_momenta_intact(orphan_run):
    store = Store(orphan_run / "trajectory.db")
    frames = frames_for_run(store, orphan_run, "run", force_source="driving")
    initial = next(f for f in frames if f.info["step_id"] == -1)
    assert initial.info["momenta_source"] == "initial_evaluation_record"
