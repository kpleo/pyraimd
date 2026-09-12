"""Cost ledger aggregation over task and attempt events.

A ``task`` event is one *logical* request (a reference evaluation, a
surrogate inference, training, I/O); an ``attempt`` event with
``record="physical_attempt"`` is one *physical* launch of a backend
(the R6 engine convention: backends whose ``compute`` accepts
``request_id`` self-report one per launch; anything else gets one recorded
by the caller, see ``pyraimd2.runtime.events.physical_attempt``).  Tasks
without attempt children are their own leaves (older logs, and backends
whose call is the launch); tasks with children are spans whose cost is
carried by the children, so aggregation sums leaves only and never double
counts.  Engine I/O task events with ``record="physical_io"`` are nested
inside their parent attempt's span and are never added on top.  Queue time
is its own field (``queue_s``, null when unknown), never folded into
elapsed.  Failed attempts and cache hits are first-class rows:
already-spent cost is append-only and never rolled back.

Task event fields (emitted by ``pyraimd2.loop.energetic``):
``task_id``, ``attempt``, ``operation`` (reference/inference/training/io),
``purpose``, ``status`` (success/failed/cache_hit), ``evaluation_id``,
``started_unix``, ``elapsed_s``, ``cpu_cores``/``gpu`` (null when unknown),
``queue_s`` (null), ``source``, ``label_id`` (reference labels),
``cache_hit`` (bool), optional ``error``.

Attempt event fields: ``record`` ("physical_attempt"), ``operation``,
``purpose``, ``request_id`` (the parent task id), ``attempt``, ``status``
(success/failed), ``started_unix``, ``elapsed_s``, ``returncode``,
``directory``, ``start``, ``source``, optional ``error``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pyraimd2.runtime.events import (
    ATTEMPT,
    ATTEMPT_FAILED_STATUSES,
    ATTEMPT_RECEIPT,
    PHYSICAL_ATTEMPT,
    PHYSICAL_ATTEMPT_RECEIPT,
    PHYSICAL_IO,
    TASK,
)

REFERENCE_PURPOSES = ("anchor", "refusal", "probe", "verification", "diagnostic")


def _attempt_identity(event: dict) -> tuple[str, int, str | None]:
    """The stable attempt identity shared by launch receipts and the
    terminal attempt event (C2): the parent logical request, the attempt
    number within it, and the physical staging directory.

    All three are required: a fresh engine process legitimately reuses the
    same automatic ``request_id``/``attempt`` pair while the directory
    allocation gives each launch its own directory, so a terminal event in
    one directory must never resolve a started-but-unterminated attempt in
    another.  The directory is compared as the recorded string — never
    resolved against the reader's cwd, the filesystem, or a moved run
    tree.  Records without a directory (older or in-process writers)
    compare equal on ``None``, preserving the legacy behavior."""
    directory = event.get("directory")
    return (str(event.get("request_id")),
            int(event.get("attempt") or 0),
            None if directory is None else str(directory))


def summarize_tasks(events: Iterable[dict]) -> dict:
    """Aggregate task/attempt events into a ledger summary.

    Distinguishes logical reference requests (every reference task,
    executed or served from cache), actual physical executions (successful
    and failed attempts), failed attempts, and cache hits.  A reference
    task that failed before any launch (no attempt children, in a log that
    records attempts) counts as a logical request only — a precheck failure
    is not an execution.  ``total_elapsed_s`` sums leaf events only:
    attempts, plus tasks that are neither spans nor nested physical I/O.

    Launch receipts (C2) are deduplicated against terminal attempt events
    by the shared FULL attempt identity (``request_id`` + ``attempt`` +
    ``directory`` — a terminal in one directory never resolves a started
    attempt in another): a receipt identity with no terminal record and no
    ``not_launched`` phase is ONE unresolved attempt, listed under
    ``unresolved_attempts`` with its evidence and counted in
    ``reference['unresolved_attempts']``.  A confirmed ``started`` receipt
    additionally counts once in ``actual_executions`` — process creation
    is confirmed even though the outcome never reached the ledger
    (successful/failed stay terminal-evidence-only); that overlap is
    explicit in ``reference['unresolved_counted_as_executions']`` and the
    entry's ``counted_as_execution`` flag, so ``actual_executions`` plus
    all unresolved is never a disjoint total.  A ``prepared``-only
    identity is launch-unknown and counts zero executions.
    ``cost_record_complete`` is False when any unresolved attempt exists,
    None (unknown) for logs whose external-launch (QE) attempts predate
    the receipt protocol; in-process attempts carry no launch window, so
    their logs are complete when nothing is unresolved.
    """
    events = list(events)
    attempts_by_parent: dict[str, list[dict]] = {}
    terminal_identities: set[tuple[str, int, str | None]] = set()
    receipts: dict[tuple[str, int, str | None], dict[str, dict]] = {}
    for event in events:
        if event.get("type") == ATTEMPT and \
                event.get("record", PHYSICAL_ATTEMPT) == PHYSICAL_ATTEMPT:
            attempts_by_parent.setdefault(
                str(event.get("request_id")), []).append(event)
            terminal_identities.add(_attempt_identity(event))
        elif event.get("type") == ATTEMPT_RECEIPT and \
                event.get("record") == PHYSICAL_ATTEMPT_RECEIPT:
            receipts.setdefault(_attempt_identity(event), {})[
                str(event.get("phase"))] = event
    # New-semantics logs are recognized by the explicit RUN_START marker,
    # never by whether an attempt happened to be recorded (B3): a fresh log
    # whose first request failed before any launch has zero attempts and is
    # still new-semantics.  Attempt events without the marker (mixed or
    # imported logs) also select the new semantics.
    has_attempts = any(
        e.get("type") == ATTEMPT
        and e.get("record", PHYSICAL_ATTEMPT) == PHYSICAL_ATTEMPT
        for e in events) or any(
        e.get("type") == "run_start" and e.get("attempt_ledger") is not None
        for e in events)
    by: dict[tuple[str, str | None], dict] = {}
    reference = {
        "logical_requests": 0,
        "actual_executions": 0,
        "successful_executions": 0,
        "failed_attempts": 0,
        "cache_hits": 0,
        "unresolved_attempts": 0,
        "unresolved_counted_as_executions": 0,
    }
    counts = {"inference": 0, "training": 0, "io": 0}
    total_elapsed = 0.0
    seen_task_ids: set[str] = set()
    for event in events:
        if event.get("type") != TASK:
            continue
        operation = str(event.get("operation"))
        purpose = event.get("purpose")
        status = str(event.get("status"))
        elapsed = float(event.get("elapsed_s") or 0.0)
        task_id = str(event.get("task_id"))
        seen_task_ids.add(task_id)
        children = attempts_by_parent.get(task_id, [])
        if not children and event.get("record") != PHYSICAL_IO:
            # A task with attempt children is a span (the children carry
            # the physical elapsed); a physical-IO task is nested inside
            # its parent attempt's span.  Neither is added on top.
            total_elapsed += elapsed
        bucket = by.setdefault(
            (operation, None if purpose is None else str(purpose)),
            {"operation": operation, "purpose": purpose, "count": 0,
             "elapsed_s": 0.0, "failed": 0, "cache_hits": 0},
        )
        bucket["count"] += 1
        bucket["elapsed_s"] += elapsed
        bucket["failed"] += int(status == "failed")
        bucket["cache_hits"] += int(status == "cache_hit")
        if operation == "reference":
            reference["logical_requests"] += 1
            reference["cache_hits"] += int(status == "cache_hit")
            if children:
                # Every launched attempt is one actual execution, whatever
                # its terminal status; only "success" completes successfully
                # (R3).
                reference["actual_executions"] += len(children)
                reference["successful_executions"] += sum(
                    1 for child in children if child.get("status") == "success")
                reference["failed_attempts"] += sum(
                    1 for child in children
                    if child.get("status") in ATTEMPT_FAILED_STATUSES)
            elif status == "success":
                reference["actual_executions"] += 1
                reference["successful_executions"] += 1
            elif status == "failed" and not has_attempts:
                # Legacy logs (no attempt records anywhere): a failed task
                # was the failed execution itself.  With attempt records, a
                # childless failed task never launched (precheck failure).
                reference["actual_executions"] += 1
                reference["failed_attempts"] += 1
        elif operation in counts:
            counts[operation] += 1
    for parent_id, children in attempts_by_parent.items():
        for child in children:
            total_elapsed += float(child.get("elapsed_s") or 0.0)
            # Orphan attempts (their task event was lost to a crash between
            # the launch and the commit) are still real physical cost, by
            # the same terminal rules as parented attempts (R3).
            if parent_id not in seen_task_ids and \
                    child.get("operation") == "reference":
                reference["actual_executions"] += 1
                reference["successful_executions"] += int(
                    child.get("status") == "success")
                reference["failed_attempts"] += int(
                    child.get("status") in ATTEMPT_FAILED_STATUSES)
    # Launch receipts without their terminal record (C2): each unmatched
    # identity is ONE unresolved attempt, deduplicated against the terminal
    # event of the same FULL identity (request id + attempt + directory),
    # so a terminal record in a second directory never erases a started
    # attempt in the first.  Launch certainty and outcome certainty are
    # separate facts: a confirmed ``started`` receipt counts as one actual
    # execution even with its outcome lost (successful/failed stay decided
    # by terminal evidence only — never guessed); a ``prepared``-only
    # identity is launch-unknown and counts zero executions;
    # ``not_launched`` is the engine's own knowledge of no launch: zero
    # executions, zero unresolved.  A terminal event arriving later for the
    # same full identity resolves the outcome without adding another
    # execution.  Unresolved therefore OVERLAPS actual_executions — never
    # read actual + unresolved as a disjoint total; a conservative upper
    # bound adds only the launch-unknown (``launched`` None) identities.
    unresolved: list[dict] = []
    for identity, phases in sorted(receipts.items()):
        if identity in terminal_identities or "not_launched" in phases:
            continue
        evidence = phases.get("started") or phases.get("prepared") or {}
        started = "started" in phases
        if started and str(evidence.get("operation")) == "reference":
            reference["actual_executions"] += 1
        unresolved.append({
            "request_id": identity[0],
            "attempt": identity[1],
            "directory": identity[2],
            "launched": True if started else None,
            "counted_as_execution": started,
            "evidence": ("process creation confirmed by the started "
                         "receipt; counted as one execution, but no "
                         "terminal record exists — outcome, timing and "
                         "return code unknown"
                         if started else
                         "staging/input prepared but no start confirmation "
                         "survived; whether the process launched is "
                         "unknowable from the record — zero confirmed "
                         "executions for this identity"),
        })
    reference["unresolved_attempts"] = len(unresolved)
    reference["unresolved_counted_as_executions"] = sum(
        1 for u in unresolved if u["launched"] is True)
    # The ledger's completeness: False when unresolved attempts exist;
    # None ("unknown") when EXTERNAL-launch attempts (the QE engines'
    # subprocess protocol) predate launch receipts — such a log alone
    # cannot prove no launch went unrecorded, and the run-directory
    # reconciliation in :func:`inspect_run` (or an external audit)
    # decides.  In-process attempts (the call IS the launch) carry no
    # such window: with no unresolved receipts the ledger is complete.
    legacy_external = any(
        child.get("source") == "qe-engine"
        for children in attempts_by_parent.values()
        for child in children) and not receipts
    return {
        "total_elapsed_s": total_elapsed,
        "by": sorted(by.values(),
                     key=lambda b: (b["operation"], str(b["purpose"]))),
        "reference": reference,
        "counts": counts,
        # Confirmed executions/successes live in ``reference``; unresolved
        # attempts are listed here with their evidence.
        "unresolved_attempts": unresolved,
        "cost_record_complete": (False if unresolved
                                 else None if legacy_external
                                 else True),
    }


def orphan_attempt_directories(run_dir: str | Path,
                               events: Iterable[dict]) -> list[dict]:
    """Attempt staging directories under ``<run_dir>/calculations`` that no
    ledger record of any kind accounts for (C2) — legacy or pre-receipt
    orphans, e.g. the historical crash window between output-file creation
    and process launch.  Each is one unresolved attempt whose launch state
    is unknowable from the filesystem: an empty ``pw.out`` beside a
    ``pw.in`` can sit before OR after a launch, and directory counts or
    mtimes alone never prove a start.  Read-only: nothing is created,
    rewritten or deleted."""
    known: set[str] = set()
    for event in events:
        if event.get("type") in (ATTEMPT, ATTEMPT_RECEIPT):
            directory = event.get("directory")
            if directory:
                known.add(str(directory))
    orphans: list[dict] = []
    calculations = Path(run_dir) / "calculations"
    if not calculations.is_dir():
        return orphans
    for attempt_dir in sorted(calculations.glob("*/attempt-*")):
        if not attempt_dir.is_dir():
            continue
        if str(attempt_dir) in known or str(attempt_dir.resolve()) in known:
            continue
        pw_out = attempt_dir / "pw.out"
        orphans.append({
            "request_id": None,
            "attempt": None,
            "directory": str(attempt_dir),
            "launched": None,
            "evidence": ("no ledger record names this staging directory "
                         "(a pre-receipt-protocol orphan); whether the "
                         "process launched is unknowable from the record"),
            "pw_out_bytes": (pw_out.stat().st_size
                             if pw_out.is_file() else None),
        })
    return orphans
