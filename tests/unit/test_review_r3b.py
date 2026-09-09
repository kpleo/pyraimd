"""Review R3 regression: physical terminal statuses and cost aggregation
(0.4.2 plan §R3) — every launched attempt counts once as actual; failed,
killed and post_processing_failed all count as failed; a missing
executable is zero launches; success means completed successfully."""

from __future__ import annotations

from pyraimd2.runtime.costs import summarize_tasks

RUN_START = {"type": "run_start", "run_id": "r",
             "attempt_ledger": "physical_attempt_v1"}


def _task(task_id, status="success", elapsed=1.0):
    return {"type": "task", "task_id": task_id, "operation": "reference",
            "purpose": "verification", "status": status,
            "elapsed_s": elapsed}


def _attempt(request_id, status, attempt=1, elapsed=1.0):
    return {"type": "attempt", "record": "physical_attempt",
            "operation": "reference", "purpose": "verification",
            "request_id": request_id, "attempt": attempt, "status": status,
            "elapsed_s": elapsed}


def _reference(events):
    summary = summarize_tasks(events)
    reference = summary["reference"]
    attempts = [e for e in events if e["type"] == "attempt"]
    # The aggregation must agree with the raw event counts, not just with
    # the events' own claims.
    assert reference["actual_executions"] == len(attempts)
    return reference


def test_r3_killed_counts_as_actual_and_failed():
    events = [RUN_START, _task("t1"), _attempt("t1", "killed")]
    reference = _reference(events)
    assert (reference["logical_requests"], reference["actual_executions"],
            reference["failed_attempts"]) == (1, 1, 1)
    assert reference["successful_executions"] == 0


def test_r3_post_processing_failed_is_a_real_failed_execution():
    for backend in ("qe-subprocess", "ase-espresso"):
        events = [RUN_START, _task("t1"),
                  _attempt("t1", "post_processing_failed")]
        reference = _reference(events)
        assert (reference["logical_requests"], reference["actual_executions"],
                reference["failed_attempts"]) == (1, 1, 1), backend
        assert reference["successful_executions"] == 0


def test_r3_missing_executable_is_zero_launches():
    events = [RUN_START, _task("t1", status="failed")]
    reference = _reference(events)
    assert (reference["logical_requests"], reference["actual_executions"],
            reference["failed_attempts"]) == (1, 0, 0)


def test_r3_failed_then_succeeded_is_one_two_one():
    events = [RUN_START, _task("t1"), _attempt("t1", "failed", attempt=1),
              _attempt("t1", "success", attempt=2)]
    reference = _reference(events)
    assert (reference["logical_requests"], reference["actual_executions"],
            reference["failed_attempts"],
            reference["successful_executions"]) == (1, 2, 1, 1)


def test_r3_orphan_attempts_follow_the_same_terminal_rules():
    events = [RUN_START, _attempt("lost-parent", "killed"),
              _attempt("lost-parent", "post_processing_failed", attempt=2)]
    reference = _reference(events)
    assert reference["actual_executions"] == 2
    assert reference["failed_attempts"] == 2
    assert reference["successful_executions"] == 0


def test_r3_spans_are_not_double_added():
    # A task span carrying attempt children (staging/process/validation all
    # inside) is settled at its terminal event: only leaf times sum.
    events = [RUN_START, _task("t1", elapsed=9.0),
              _attempt("t1", "success", elapsed=3.0),
              {"type": "task", "task_id": "t1-io-1", "operation": "io",
               "purpose": "density_copy", "record": "physical_io",
               "request_id": "t1", "status": "success", "elapsed_s": 1.0}]
    summary = summarize_tasks(events)
    assert summary["total_elapsed_s"] == 3.0
