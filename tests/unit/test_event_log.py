"""EventLog mechanics: seq/cursor, dedup keys, single-writer lock (hermetic)."""

from __future__ import annotations

import pytest

from pyraimd2.runtime.events import EventLog, EventLogError


def test_append_read_cursor_and_reopen_continues_seq(tmp_path):
    with EventLog(tmp_path) as log:
        assert log.append("run_start", {"run_id": "r"}) == 1
        assert log.append("task", {"operation": "reference", "n": 1}) == 2
        assert log.last_seq == 2
        assert [e["seq"] for e in log.iter_events()] == [1, 2]
        assert [e["seq"] for e in log.iter_events(after_seq=1)] == [2]
        assert list(log.iter_events(after_seq=2)) == []
    # A clean close releases the lock; reopening appends and continues seq.
    with EventLog(tmp_path) as log:
        assert log.last_seq == 2
        assert log.append("run_end", {"status": "success"}) == 3
    with EventLog(tmp_path) as log:
        assert [e["type"] for e in log.iter_events()] == [
            "run_start", "task", "run_end"]


def test_append_once_deduplicates_by_key_across_reopen(tmp_path):
    with EventLog(tmp_path) as log:
        assert log.append_once("label:run-label-1", "task", {"n": 1}) == 1
        assert log.append_once("label:run-label-1", "task", {"n": 2}) is None
        assert log.append_once("label:run-label-2", "task", {"n": 3}) == 2
    # The dedup set survives reopening (committed state, not memory).
    with EventLog(tmp_path) as log:
        assert log.append_once("label:run-label-1", "task", {"n": 4}) is None
    with EventLog(tmp_path) as log:
        events = list(log.iter_events())
    assert [e["n"] for e in events if e["type"] == "task"] == [1, 3]
    assert [e["key"] for e in events] == ["label:run-label-1", "label:run-label-2"]


def test_concurrent_writer_rejected_until_close(tmp_path):
    log = EventLog(tmp_path)
    with pytest.raises(EventLogError, match="active writer"):
        EventLog(tmp_path)
    log.close()
    with EventLog(tmp_path):  # lock released: a deliberate new writer works
        pass
    log.close()  # idempotent


def test_corrupt_line_fails_loud(tmp_path):
    with EventLog(tmp_path) as log:
        log.append("run_start", {"run_id": "r"})
        log.append("task", {"task_id": "t1"})
        log.append("run_end", {"run_id": "r", "status": "success"})
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    lines[1] = "{not json"  # a committed middle event can never be skipped
    (tmp_path / "events.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(EventLogError, match="corrupt event"):
        list(EventLog(tmp_path).iter_events())


def test_torn_tail_is_skipped_and_marked(tmp_path):
    with EventLog(tmp_path) as log:
        log.append("run_start", {"run_id": "r"})
        log.append("task", {"task_id": "t1"})
    with (tmp_path / "events.jsonl").open("a") as fh:
        fh.write('{"seq": 3, "type": "run_end", "statu')  # crash mid-append
    log = EventLog(tmp_path)
    events = list(log.iter_events())
    assert [event["type"] for event in events] == ["run_start", "task"]
    assert log.torn_tail is True
    assert log.last_seq == 2
    # The truncated event never committed: its bytes are dropped on open so
    # the next append starts on a clean line and reuses its seq.
    assert log.append("run_end", {"run_id": "r", "status": "success"}) == 3
    log.close()
    with EventLog(tmp_path) as log2:
        assert [event["type"] for event in log2.iter_events()] == [
            "run_start", "task", "run_end"]
        assert log2.torn_tail is False


def test_clean_log_is_not_marked_torn(tmp_path):
    with EventLog(tmp_path) as log:
        log.append("run_start", {"run_id": "r"})
    with EventLog(tmp_path) as log2:
        assert log2.torn_tail is False


def test_inspect_read_skips_torn_tail_but_not_middle(tmp_path):
    from pyraimd2.runtime.inspect import _read_events

    path = tmp_path / "events.jsonl"
    path.write_text('{"seq": 1, "type": "run_start"}\n{"seq": 2, "ty')
    assert [event["seq"] for event in _read_events(path)] == [1]
    path.write_text('{"seq": 1}\n{bad\n{"seq": 3}\n')
    with pytest.raises(EventLogError, match="corrupt event"):
        _read_events(path)


def test_abandoned_log_releases_the_handle_but_keeps_the_lock(tmp_path):
    """A constructor-failure path can strand an open log (no reference
    returned). GC must close its OS handle — never leak it — while the
    lock file stays put: lock removal remains a deliberate close(), and a
    stranded writer's stale lock stays available as crash evidence."""
    import gc
    import warnings

    log = EventLog(tmp_path)
    log.append("run_start", {"run_id": "r"})
    fh = log._fh
    del log
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        gc.collect()
    assert fh.closed  # the handle was released without leaking
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert (tmp_path / "events.jsonl.lock").exists()  # lock is crash evidence
    with EventLog(tmp_path, force=True):  # deliberate reclaim still works
        pass
