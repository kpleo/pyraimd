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

import contextlib
import inspect
import json
import os
import time
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
ATTEMPT = "attempt"
MODEL_UPDATE = "model_update"
RUN_SUMMARY = "run_summary"
RUN_END = "run_end"
RESUMED = "resumed"
UPDATE_REJECTED = "update_rejected"
# Frozen calibration-pacing decisions (0.6 prototype; only present when
# [policy.calibration_pacing] is enabled): one keyed event per refused
# evaluation, persisted before any probe spend so a crash/rebuild never
# re-decides or double-advances the state.
PACING_DECISION = "pacing_decision"

# ``record`` discriminant on attempt events (and on nested physical I/O
# task events): the cost ledger counts physical executions by it.
PHYSICAL_ATTEMPT = "physical_attempt"
PHYSICAL_IO = "physical_io"

# Minimal durable launch receipts (C2): a backend whose launch crosses a
# process boundary writes ``attempt_receipt`` events carrying the attempt's
# stable identity (``request_id`` + ``attempt`` + ``directory``) and a
# ``phase`` — ``prepared`` (staging/input written, before any launch),
# ``started`` (written only from the actual process-creation fact), or
# ``not_launched`` (the engine knows the process never started, e.g. a
# missing executable).  The terminal ``attempt`` event closes the same
# identity.  Receipts carry no timing and are never billed as spans; a
# crash between receipts leaves exactly the evidence that existed — the
# ledger marks the attempt unresolved rather than guessing an outcome, and
# a confirmed ``started`` receipt without a terminal record still counts
# as ONE actual execution (launch certainty), while successful/failed
# stay decided by terminal evidence only (outcome certainty).
ATTEMPT_RECEIPT = "attempt_receipt"
PHYSICAL_ATTEMPT_RECEIPT = "physical_attempt_receipt"

# Terminal statuses of a launched attempt (shared convention).  Every
# launched attempt ends in exactly one of them — never left "running".
# Only "success" means the launch completed successfully; the other three
# are real executions that failed (a failed post-processing step is still
# a real, failed execution — never disguised as not-run).
ATTEMPT_TERMINAL_STATUSES = ("success", "failed", "killed",
                             "post_processing_failed")
ATTEMPT_FAILED_STATUSES = ("failed", "killed", "post_processing_failed")

# RUN_START marker naming the attempt-ledger protocol a log follows.
# summarize_tasks reads NEW logs by this marker — never by whether an
# attempt happened to be recorded (B3: a fresh log whose first request
# failed before any launch has zero attempts and is still new-semantics).
ATTEMPT_LEDGER_PHYSICAL_V1 = "physical_attempt_v1"


def accepts_request_id(backend: object, method: str) -> bool:
    """True when ``backend.<method>`` takes a ``request_id`` keyword — the
    R6 engine convention for self-reporting one attempt event per real
    process launch (see REVIEW_FIXES_BACKEND_20260909.md)."""
    call = getattr(backend, method, None)
    if call is None:
        return False
    try:
        return "request_id" in inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False


# Sink attach points on a self-reporting engine, in preference order.  The
# shared convention (0.4.1): an engine accepting ``request_id`` must expose
# one of these attributes (None when unconnected); a wrapper holding an
# event log connects it for the duration of the call.  Passing request_id
# to an engine with no sink at all would hide internal retries from the
# ledger, so that combination is refused before launch (B2).
_SINK_ATTRIBUTES = ("attempt_sink", "event_log", "_event_log")


def _sink_attribute(backend: object) -> str | None:
    for name in _SINK_ATTRIBUTES:
        if hasattr(backend, name):
            return name
    return None


@contextlib.contextmanager
def physical_attempt(backend: object, event_log: EventLog | None, *,
                     operation: str, request_id: str,
                     purpose: str | None, source: str,
                     method: str = "compute") -> Iterator[dict]:
    """Physical-execution record(s) of one logical backend call.

    A ``task`` event is the *logical* request; an ``attempt`` event
    (``record="physical_attempt"``) is one real launch.  A backend whose
    ``<method>`` accepts ``request_id`` reports every launch itself — the
    context manager yields the keyword to pass through, connecting the
    run's event log as the attempt sink for the call when the engine's own
    sink is unconnected (R2: sink identity is checked — an engine already
    wired to a DIFFERENT log is refused before launch, never silently
    cross-posted).  Any other backend gets exactly one attempt recorded
    around the call.  A self-reporting backend that raises before launching
    anything records no attempt — a precheck failure is not a physical
    execution.  With no event log on the caller side the id is not injected
    at all and no sink is required: the engine's no-log API stays usable.
    """
    if accepts_request_id(backend, method):
        attach = _sink_attribute(backend)
        sink = getattr(backend, attach) if attach is not None else None
        if event_log is None:
            yield {"request_id": request_id} if sink is not None else {}
            return
        if attach is None:
            raise EventLogError(
                f"{type(backend).__name__} accepts request_id but exposes no "
                "attempt sink (attempt_sink/event_log); the combination would "
                "hide internal retries from the ledger — refusing before launch")
        if sink is None:
            # Explicitly connect the run's log as this call's attempt sink.
            setattr(backend, attach, event_log)
            try:
                yield {"request_id": request_id}
            finally:
                setattr(backend, attach, sink)
            return
        if sink is not event_log:
            raise EventLogError(
                f"{type(backend).__name__} is wired to a different attempt "
                "sink than the run's event log; attempts would land in the "
                "other ledger — refusing before launch")
        yield {"request_id": request_id}
        return
    if event_log is None:
        yield {}
        return
    started_unix = time.time()
    start = time.perf_counter()
    status, message = "success", None
    try:
        yield {}
    except Exception as error:
        status, message = "failed", repr(error)
        raise
    finally:
        event_log.append(ATTEMPT, {
            "record": PHYSICAL_ATTEMPT,
            "operation": operation, "purpose": purpose,
            "request_id": request_id, "attempt": 1, "status": status,
            "started_unix": started_unix,
            "elapsed_s": time.perf_counter() - start,
            "returncode": None, "directory": None, "start": None,
            "source": source, "error": message,
        })


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
        self._torn_tail = False
        try:
            if self.path.exists():
                for event in self._read_events():
                    self._seq = max(self._seq, int(event.get("seq", 0)))
                    key = event.get("key")
                    if key is not None:
                        self._keys.add(str(key))
            if self._torn_tail:
                # The torn bytes never committed; drop them so the next
                # append does not glue a new event onto the partial line.
                # Committed events are never rewritten.
                data = self.path.read_bytes()
                keep = data.rstrip(b"\n").rfind(b"\n") + 1
                with self.path.open("r+b") as fh:
                    fh.truncate(keep)
            self._fh = self.path.open("a", encoding="utf-8")
        except Exception:
            # A failed construction must release the lock it just took —
            # the caller keeps the original exception and no live handle.
            self._closed = True
            try:
                self._lock_path.unlink()
            except FileNotFoundError:
                pass
            raise

    @property
    def last_seq(self) -> int:
        """The last committed event number (WP03's checkpoint cursor)."""
        return self._seq

    @property
    def torn_tail(self) -> bool:
        """True when the log's final line was left truncated by a crash
        (skipped on read — it never committed; the file is kept as-is)."""
        return self._torn_tail

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
            lines = [(number, line.strip())
                     for number, line in enumerate(fh, start=1) if line.strip()]
        last_line = lines[-1][0] if lines else None
        for line_number, line in lines:
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                if line_number == last_line:
                    # Torn tail: a crash interrupted the final append before
                    # it committed. Skip it and recover from the last intact
                    # event; the original file is never rewritten.
                    self._torn_tail = True
                    return
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

    def __del__(self) -> None:
        # Last-resort cleanup: an abandoned log (a constructor raised after
        # the log opened, or a caller dropped it mid-failure) must never
        # leak an OS handle. Only the handle is closed here — the lock file
        # stays, because removing it must remain a deliberate act (close());
        # a crashed writer's stale lock is crash evidence, not cleanup.
        try:
            if not self._closed:
                self._fh.close()
        except Exception:  # noqa: BLE001, S110 - GC cleanup must never raise
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
