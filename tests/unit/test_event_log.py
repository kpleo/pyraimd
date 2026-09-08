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
    assert [e["type"] for e in EventLog(tmp_path).iter_events()] == [
        "run_start", "task", "run_end"]


def test_append_once_deduplicates_by_key_across_reopen(tmp_path):
    with EventLog(tmp_path) as log:
        assert log.append_once("label:run-label-1", "task", {"n": 1}) == 1
        assert log.append_once("label:run-label-1", "task", {"n": 2}) is None
        assert log.append_once("label:run-label-2", "task", {"n": 3}) == 2
    # The dedup set survives reopening (committed state, not memory).
    with EventLog(tmp_path) as log:
        assert log.append_once("label:run-label-1", "task", {"n": 4}) is None
    events = list(EventLog(tmp_path).iter_events())
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
    with (tmp_path / "events.jsonl").open("a") as fh:
        fh.write("{not json\n")
    with pytest.raises(EventLogError, match="corrupt event"):
        list(EventLog(tmp_path).iter_events())
