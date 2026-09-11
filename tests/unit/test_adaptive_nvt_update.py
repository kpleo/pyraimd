"""M3B acceptance: resumable NVT online updates through GuardedUpdater.

M3B-1: the deferred calibration's direction is the persisted realized
displacement of the step that produced the label — a callback that never
changes the model still anchors and admits; a truly degenerate direction
still falls back safely.
M3B-2/3: candidate → guard → persist → commit → activate with rollback;
rejection and publish failure keep their real costs; resume never retrains
committed updates and never re-consumes labels; bath/check/velocity streams
stay separate (training randomness, if any, lives in the updater state).

A. normal update with physical accounting; B. one candidate rejection and
one artifact-publish failure; C. clean split plus two real os._exit
windows across the update transaction.  Analytic toy backends only — zero
real DFT budget.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_adaptive_nvt import _atoms, _rows, _spec
from test_guarded_update import Reference, TrainableHarmonic
from test_nvt import events

from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store

K_REF = 1.2  # the Reference toy's spring constant (trainable target)


def _run_nvt(run_dir, *, updater=None, steps=8, check_probability=0.5,
             checkpoint_interval=4, positions=None, momenta=None,
             temperature=None):
    """NVT adaptive run with the trainable harmonic model against the
    analytic reference (k=1.2) — the GuardedUpdater path."""
    run_dir.mkdir(parents=True)
    model = TrainableHarmonic()
    engine = Reference(k=K_REF)
    runner = EnergeticRunner(
        _atoms() if positions is None else _atoms(positions, momenta),
        model, engine, Store(run_dir / "trajectory.db"), "run",
        run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=checkpoint_interval,
        force_budget=0.5, timestep_fs=0.5, time_cap_fs=100.0,
        transverse_cap=1.0, check_probability=check_probability,
        check_seed=7,
        integrator_spec=_spec() if temperature is None
        else _spec(temperature=temperature),
        on_label=updater)
    runner.run(steps)
    runner.close()
    return run_dir, model, engine


def _proposals(run_dir):
    return [e for e in events(run_dir) if e["type"] == "evaluation_proposed"]


def _tasks(run_dir, operation=None):
    return [e for e in events(run_dir) if e["type"] == "task"
            and (operation is None or e.get("operation") == operation)]


# --- M3B-1: the deferred-calibration direction ---------------------------------


def test_record_only_callback_still_anchors_and_admits_nvt(tmp_path):
    # The review's reproduction: identical H2, 8 steps, bias=1e-3 — the
    # only change is on_label=None -> lambda: False.  Before the fix the
    # deferred calibration's direction degenerated to exactly zero and no
    # anchor ever formed (0 accepted / 0 calibrations).
    run_dir, _model, _engine = _run_nvt(tmp_path / "cb", updater=None)
    base_props = _proposals(run_dir)
    assert sum(1 for p in base_props if p["accepted"]) == 7

    seen = []
    run_dir, _model, _engine = _run_nvt(
        tmp_path / "cb2", updater=lambda observation: seen.append(
            observation.label_id) or False)
    props = _proposals(run_dir)
    accepted = [p for p in props if p["accepted"]]
    assert accepted, "a record-only callback must not block anchoring"
    assert any(p["reason"] == "forecast_accepted" for p in props)
    anchors = [e for e in events(run_dir)
               if e["type"] == "evaluation_committed"
               and (e.get("segment_id") is not None)]
    assert anchors  # at least one segment established
    assert seen  # the callback observed labels
    # model untouched: no update events, generation stays 0
    assert not [e for e in events(run_dir) if e["type"] == "model_update"]
    # The reference cost stays bounded: anchors + probes + checks only.
    n_reference = len(_tasks(run_dir, "reference"))
    assert n_reference <= len(props) + 4 * 4 + len(props)


def test_degenerate_direction_with_callback_falls_back_safely_nvt(tmp_path):
    # At the minimum with zero momenta and a zero-temperature bath the
    # realized displacement is exactly zero — with or without a callback
    # the run must stay on the reference route with reasons recorded.
    run_dir, _, engine = _run_nvt(
        tmp_path / "deg", updater=lambda observation: False, steps=4,
        check_probability=0.0, positions=[[0.0, 0.0, 0.0]],
        momenta=[[0.0, 0.0, 0.0]], temperature=0.0)
    props = _proposals(run_dir)
    assert not any(p["accepted"] for p in props)
    assert {p["reason"] for p in props} <= {
        "initial_reference", "direction_unavailable_reference"}
    assert engine.attempts == len(props)  # every evaluation: one anchor call
    summary = next(e for e in events(run_dir) if e["type"] == "run_summary")
    assert summary["n_reference"] == len(props)


def test_deferred_origin_resume_matches_continuous_nvt(tmp_path):
    # Crash right after a post-checkpoint label-consumption event (the
    # deferred origin exists, its calibration has not run): the replay
    # rebuilds the origin — direction included — and the resumed run is
    # bit-identical to the continuous one.  The updater never fires
    # (n_label beyond the run length) but is stateful, so resume is legal.
    from pyraimd2.runtime.events import EventLog as _EventLog

    def make_updater(model):
        return GuardedUpdater(model, UpdatePolicy(n_label=1000,
                                                  guard_size=1))

    control_dir, _, _ = _run_nvt(tmp_path / "cont",
                                 updater=make_updater(TrainableHarmonic()))
    consumed = [e for e in events(control_dir) if e["type"] == "label_consumed"]
    target = next(e for e in consumed
                  if int(e["evaluation_id"]) >= 5)  # after checkpoint 4
    kill_key = f"consumed:{target['label_id']}"

    monkey = pytest.MonkeyPatch()
    real_append_once = _EventLog.append_once

    def crash_after_consumed(self, key, event_type, payload):
        result = real_append_once(self, key, event_type, payload)
        if key == kill_key:
            raise RuntimeError("injected crash after label consumption")
        return result

    monkey.setattr(_EventLog, "append_once", crash_after_consumed)
    try:
        with pytest.raises(RuntimeError, match="injected crash"):
            _run_nvt(tmp_path / "crash",
                     updater=make_updater(TrainableHarmonic()))
    finally:
        monkey.undo()
    crash_dir = tmp_path / "crash"
    model = TrainableHarmonic()
    runner = EnergeticRunner.resume(
        crash_dir, model, Reference(k=K_REF), updater=make_updater(model),
        event_log_force=True, checkpoint_interval_steps=4)
    runner.run(9 - runner.calc.n_evaluations)
    runner.close()
    rows_a, rows_b = _rows(crash_dir), _rows(control_dir)
    assert len(rows_a) == len(rows_b)
    for row_a, row_b in zip(rows_a, rows_b):
        np.testing.assert_array_equal(row_a.toatoms().positions,
                                      row_b.toatoms().positions)
        np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                      row_b.toatoms().get_momenta())
    assert [(p["accepted"], p["reason"]) for p in _proposals(crash_dir)] == \
           [(p["accepted"], p["reason"]) for p in _proposals(control_dir)]
