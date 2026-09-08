"""Cost ledger aggregation over task events.

Every physical execution is a ``task`` event in the run's event log:
reference evaluations (purposes anchor/refusal/probe/verification/
diagnostic), surrogate inference, training, and I/O.  Task events are the
*leaves* of the timing tree — evaluation and run spans are measured
directly and recorded separately, so aggregation sums task events only and
never double counts.  Queue time is its own field (``queue_s``, null when
unknown), never folded into elapsed.  Failed attempts and cache hits are
first-class rows: already-spent cost is append-only and never rolled back.

Task event fields (emitted by ``pyraimd2.loop.energetic``):
``task_id``, ``attempt``, ``operation`` (reference/inference/training/io),
``purpose``, ``status`` (success/failed/cache_hit), ``evaluation_id``,
``started_unix``, ``elapsed_s``, ``cpu_cores``/``gpu`` (null when unknown),
``queue_s`` (null), ``source``, ``label_id`` (reference labels),
``cache_hit`` (bool), optional ``error``.
"""

from __future__ import annotations

from collections.abc import Iterable

from pyraimd2.runtime.events import TASK

REFERENCE_PURPOSES = ("anchor", "refusal", "probe", "verification", "diagnostic")


def summarize_tasks(events: Iterable[dict]) -> dict:
    """Aggregate task events into a ledger summary.

    Distinguishes logical reference requests (every reference task,
    executed or served from cache), actual physical executions (successful
    and failed), failed attempts, and cache hits.  ``total_elapsed_s`` sums
    leaf task events only — run/evaluation wall times live on their own
    events and are never added on top.
    """
    by: dict[tuple[str, str | None], dict] = {}
    reference = {
        "logical_requests": 0,
        "successful_executions": 0,
        "failed_attempts": 0,
        "cache_hits": 0,
    }
    counts = {"inference": 0, "training": 0, "io": 0}
    total_elapsed = 0.0
    for event in events:
        if event.get("type") != TASK:
            continue
        operation = str(event.get("operation"))
        purpose = event.get("purpose")
        status = str(event.get("status"))
        elapsed = float(event.get("elapsed_s") or 0.0)
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
            reference["successful_executions"] += int(status == "success")
            reference["failed_attempts"] += int(status == "failed")
            reference["cache_hits"] += int(status == "cache_hit")
        elif operation in counts:
            counts[operation] += 1
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
