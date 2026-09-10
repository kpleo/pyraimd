"""R4 regression: controlled failures release owned resources (0.5-batch1
§R4) — failed workflow setup closes the event log and store, a failed
EventLog construction releases its lock, read paths take no writer lock,
and the stale-lock crash recovery is unchanged."""

from __future__ import annotations

import pytest

from pyraimd2.runtime.events import EventLog, EventLogError


def test_failed_outputs_construction_releases_log_and_lock(tmp_path):
    from test_review_r4 import _write_config

    from pyraimd2.config import load_config
    from pyraimd2.workflows import md as md_module
    from pyraimd2.workflows import run_workflow

    config = load_config(_write_config(tmp_path / "r4", mode="reference",
                                       steps=2))

    class BrokenOutputs:
        def __init__(self, *args, **kwargs):
            raise OSError("injected outputs failure")

        # the failure path only constructs
    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "RunOutputs", BrokenOutputs)
    try:
        with pytest.raises(OSError, match="injected outputs"):
            run_workflow(config, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()
    # The log this call owned is closed and its lock released: a fresh
    # writer opens without force, and the partial log was removed.
    assert not (config.run.directory / "events.jsonl.lock").exists()
    assert not (config.run.directory / "events.jsonl").exists()


def test_event_log_construct_failure_releases_its_lock(tmp_path):
    # Corrupt a middle (committed) line: construction must fail AND release.
    (tmp_path / "events.jsonl").write_text(
        '{"seq": 1, "type": "run_start"}\n{not json\n{"seq": 3}\n')
    with pytest.raises(EventLogError, match="corrupt event"):
        EventLog(tmp_path)
    assert not (tmp_path / "events.jsonl.lock").exists()
    # A deliberate writer still starts cleanly on an intact log.
    (tmp_path / "events.jsonl").write_text('{"seq": 1, "type": "run_start"}\n')
    with EventLog(tmp_path) as log:
        assert log.last_seq == 1


def test_readonly_paths_create_no_writer_lock(tmp_path):
    with EventLog(tmp_path) as log:
        log.append("run_start", {"run_id": "r"})
    from pyraimd2.runtime.inspect import _read_events
    from pyraimd2.workflows.export import completed_step_ids

    _read_events(tmp_path / "events.jsonl")
    completed_step_ids(tmp_path)
    assert not (tmp_path / "events.jsonl.lock").exists()


def test_crash_lock_reclaim_still_works(tmp_path):
    with EventLog(tmp_path) as log:
        log.append("run_start", {"run_id": "r"})
    # Simulate a crashed writer: lock file left behind.
    (tmp_path / "events.jsonl.lock").write_text("99999")
    with pytest.raises(EventLogError, match="active writer"):
        EventLog(tmp_path)
    with EventLog(tmp_path, force=True) as log:
        assert log.append("task", {"task_id": "t"}) == 2
