"""Authoritative run event log: one JSONL file, one writer, explicit cursor.

``events.jsonl`` is the single authoritative event store of a run directory —
evaluation proposals and commits, task/attempt executions (the cost ledger),
and model updates all live here and nowhere else.  Each appended event gets a
monotonically increasing ``seq``: the committed event number a checkpoint
records as its replay cursor (WP03 reads with ``iter_events(after_seq=...)``).

Concurrency: a run has exactly one writer.  Opening an :class:`EventLog`
takes an exclusive lock file (``<name>.lock``, ``O_CREAT|O_EXCL``); a second
concurrent open of the same run is *rejected*, never serialized silently.
A stale lock after a crash must be removed deliberately by the user — the
log never guesses.

The ledger is append-only: a rolled-back trajectory never removes already
spent physical cost.  Logical events written under a dedup key
(:meth:`append_once`) are idempotent — re-delivering the same label ID or
evaluation ID writes the event once.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Self

EVENT_SCHEMA_VERSION = 1

# Event types (payload shapes are documented on the emitters in
# ``pyraimd2.loop.energetic`` and ``pyraimd2.runtime.inspect``).
RUN_START = "run_start"
EVALUATION_PROPOSED = "evaluation_proposed"
EVALUATION_COMMITTED = "evaluation_committed"
STEP_COMPLETED = "step_completed"
PROBE_COMPLETED = "probe_completed"
LABEL_CONSUMED = "label_consumed"
TASK = "task"
MODEL_UPDATE = "model_update"
RUN_SUMMARY = "run_summary"
RUN_END = "run_end"
RESUMED = "resumed"


class EventLogError(RuntimeError):
    """The event log cannot fulfill its single-writer or format contract."""


class EventLog:
    """Single-writer append-only JSONL event log with an exclusive lock.

    The lock is held for the instance's lifetime; :meth:`close` releases it.
    Re-opening after a clean close appends and continues ``seq`` from the
    last committed event (the resume path WP03 builds on).
    """

    def __init__(self, run_dir: str | Path, name: str = "events.jsonl", *,
                 force: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / name
        self._lock_path = self.run_dir / (name + ".lock")
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            if not force:
                raise EventLogError(
                    f"run at {self.run_dir} already has an active writer "
                    f"({self._lock_path.name} exists); concurrent starts are "
                    "rejected — remove a stale lock deliberately after a crash"
                ) from error
            # Deliberate reclaim: the caller asserts it is the sole writer
            # (e.g. resume after a crash). Never use this to queue writers.
            self._lock_path.unlink()
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        self._closed = False
        self._keys: set[str] = set()
        self._seq = 0
        if self.path.exists():
            for event in self._read_events():
                self._seq = max(self._seq, int(event.get("seq", 0)))
                key = event.get("key")
                if key is not None:
                    self._keys.add(str(key))
        self._fh = self.path.open("a", encoding="utf-8")

    @property
    def last_seq(self) -> int:
        """The last committed event number (WP03's checkpoint cursor)."""
        return self._seq

    def append(self, event_type: str, payload: dict) -> int:
        """Append an event and return its committed sequence number."""
        return self._write(event_type, payload, key=None)

    def append_once(self, key: str, event_type: str, payload: dict) -> int | None:
        """Append unless ``key`` was already committed (label/evaluation ID
        dedup); returns the seq, or None when the event already exists."""
        if key in self._keys:
            return None
        return self._write(event_type, payload, key=key)

    def _write(self, event_type: str, payload: dict, key: str | None) -> int:
        if self._closed:
            raise EventLogError("event log is closed")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type must be a nonempty string")
        event = {"seq": self._seq + 1, "type": event_type, **payload}
        if key is not None:
            event["key"] = key
        line = json.dumps(event, sort_keys=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._seq += 1
        if key is not None:
            self._keys.add(key)
        return self._seq

    def iter_events(self, after_seq: int = 0) -> Iterator[dict]:
        """Yield committed events with ``seq > after_seq`` in original order.

        This is the replay cursor: a checkpoint stores the seq of the last
        committed event it covers, and recovery replays everything after it.
        """
        yield from (event for event in self._read_events()
                    if int(event.get("seq", 0)) > after_seq)

    def _read_events(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise EventLogError(
                        f"corrupt event at {self.path}:{line_number}: {error}"
                    ) from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._fh.close()
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
