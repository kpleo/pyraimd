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

from pyraimd2.runtime.events import ATTEMPT, PHYSICAL_ATTEMPT, PHYSICAL_IO, TASK

REFERENCE_PURPOSES = ("anchor", "refusal", "probe", "verification", "diagnostic")


def summarize_tasks(events: Iterable[dict]) -> dict:
    """Aggregate task/attempt events into a ledger summary.

    Distinguishes logical reference requests (every reference task,
    executed or served from cache), actual physical executions (successful
    and failed attempts), failed attempts, and cache hits.  A reference
    task that failed before any launch (no attempt children, in a log that
    records attempts) counts as a logical request only — a precheck failure
    is not an execution.  ``total_elapsed_s`` sums leaf events only:
    attempts, plus tasks that are neither spans nor nested physical I/O.
    """
    events = list(events)
    attempts_by_parent: dict[str, list[dict]] = {}
    for event in events:
        if event.get("type") == ATTEMPT and \
                event.get("record", PHYSICAL_ATTEMPT) == PHYSICAL_ATTEMPT:
            attempts_by_parent.setdefault(
                str(event.get("request_id")), []).append(event)
    has_attempts = bool(attempts_by_parent)
    by: dict[tuple[str, str | None], dict] = {}
    reference = {
        "logical_requests": 0,
        "successful_executions": 0,
        "failed_attempts": 0,
        "cache_hits": 0,
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
                reference["successful_executions"] += sum(
                    1 for child in children if child.get("status") == "success")
                reference["failed_attempts"] += sum(
                    1 for child in children if child.get("status") == "failed")
            elif status == "success":
                reference["successful_executions"] += 1
            elif status == "failed" and not has_attempts:
                # Legacy logs (no attempt records anywhere): a failed task
                # was the failed execution itself.  With attempt records, a
                # childless failed task never launched (precheck failure).
                reference["failed_attempts"] += 1
        elif operation in counts:
            counts[operation] += 1
    for parent_id, children in attempts_by_parent.items():
        for child in children:
            total_elapsed += float(child.get("elapsed_s") or 0.0)
            # Orphan attempts (their task event was lost to a crash between
            # the launch and the commit) are still real physical cost.
            if parent_id not in seen_task_ids and \
                    child.get("operation") == "reference":
                reference["successful_executions"] += int(
                    child.get("status") == "success")
                reference["failed_attempts"] += int(
                    child.get("status") == "failed")
    reference["actual_executions"] = (
        reference["successful_executions"] + reference["failed_attempts"]
    )
    return {
        "total_elapsed_s": total_elapsed,
        "by": sorted(by.values(),
                     key=lambda b: (b["operation"], str(b["purpose"]))),
        "reference": reference,
        "counts": counts,
    }
