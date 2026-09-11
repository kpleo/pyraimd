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

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from test_adaptive_nvt import _atoms, _complete_boundaries, _rows, _spec
from test_guarded_update import Reference, TrainableHarmonic
from test_nvt import events

from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store

K_REF = 1.2  # the Reference toy's spring constant (trainable target)


def _run_nvt(run_dir, *, updater=None, steps=8, check_probability=0.5,
             checkpoint_interval=4, positions=None, momenta=None,
             temperature=None, model=None, engine=None):
    """NVT adaptive run with the trainable harmonic model against the
    analytic reference (k=1.2) — the GuardedUpdater path.  When an updater
    is given, pass the model it wraps so the runner and the updater share
    one object."""
    run_dir.mkdir(parents=True)
    model = TrainableHarmonic() if model is None else model
    engine = Reference(k=K_REF) if engine is None else engine
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


def _segment_anchors(run_dir):
    """Anchor records keyed by segment id, from both row-metadata channels
    (in-line calibrations land in ``new_anchor``; deferred ones land in
    ``calibration_after_previous_label``)."""
    out = {}
    for row in _rows(run_dir):
        metadata = row.data.get("metadata") or {}
        for record in (metadata.get("new_anchor"),
                       (metadata.get("calibration_after_previous_label")
                        or {}).get("anchor")):
            if record is not None:
                out[int(record["segment_id"])] = record
    return out



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


# --- M3B-3 A: normal update with physical accounting ---------------------------

from pyraimd2.loop.integrators import derive_stream_seed


def _a_policy(model):
    return GuardedUpdater(model, UpdatePolicy(n_label=2, guard_size=1))


def _bath_state_after(steps, n_atoms=2):
    """The bath stream state after `steps` Langevin steps' draws (two
    standard_normal((N,3)) per step, xi then eta) from the role-derived
    seed — nothing else may consume it."""
    from pyraimd2.loop.integrators import derive_stream_seed as derive

    rng = np.random.default_rng(derive(123, "thermostat"))
    for _ in range(steps):
        rng.standard_normal((n_atoms, 3))
        rng.standard_normal((n_atoms, 3))
    return rng.bit_generator.state


def test_a_normal_update_physical_accounting(tmp_path):
    run_dir = tmp_path / "a"
    model = TrainableHarmonic()
    updater = _a_policy(model)
    run_dir, model, engine = _run_nvt(run_dir, updater=updater, model=model,
                                      steps=12)
    evs = events(run_dir)
    updates = [e for e in evs if e["type"] == "model_update"]
    assert updates, "expected at least one published update"
    first_update = updates[0]
    k_update = int(first_update["origin_evaluation_id"])

    # generations, label consumption and external counters agree
    assert [e["generation"] for e in updates] == list(
        range(1, len(updates) + 1))
    consumed_ids = {str(e.get("label_id") or e.get("origin_label_id"))
                    for e in evs
                    if e["type"] in ("label_consumed", "model_update")}
    assert updater.n_consumed == len(consumed_ids)
    reference_tasks = [t for t in _tasks(run_dir, "reference")
                       if t["status"] == "success"]
    assert engine.attempts == len(reference_tasks)
    training_tasks = [t for t in _tasks(run_dir)
                      if t.get("operation") == "training"]
    # the callback is billed once per consumed label; the actual fit runs
    # only when the policy fires — callback count, training attempts and
    # publishes are distinct numbers (M3B-2 accounting)
    assert len(training_tasks) == len(consumed_ids)
    assert model.finetune_calls == len(updates)
    assert updater.n_updates == len(updates) and updater.n_rejected == 0
    assert all(t["status"] == "success" for t in training_tasks)
    assert not [t for t in training_tasks if t.get("recovery")]

    # the update step drives with the OLD frozen model; the new model starts
    # at the next, not-yet-frozen evaluation
    commits = {int((e.get("context") or {})["evaluation_id"]): e
               for e in evs if e["type"] == "evaluation_committed"}
    proposals = {int((e.get("context") or {})["evaluation_id"]): e
                 for e in evs if e["type"] == "evaluation_proposed"}
    old_id = commits[k_update]["model_id"]
    assert first_update["model_id"] != old_id
    artifact = json.loads(
        (run_dir / "models"
         / first_update["model_id"].replace("/", "_")
         / "state.json").read_text())
    assert artifact["parent_model_id"] == old_id
    assert proposals[k_update + 1]["context"]["model_id"] == \
        first_update["model_id"]
    step_event = next(e for e in evs if e["type"] == "step_completed"
                      and e["step_id"] == k_update - 1)
    assert step_event["model_id"] == old_id  # the step's driving identity

    # a new segment under the new generation follows the update
    segments = {(e.get("segment_id"), e["model_id"])
                for e in commits.values() if e.get("segment_id") is not None}
    assert any(seg_model == first_update["model_id"] for _, seg_model in
               segments), "expected a new segment under the new model"

    # training consumed no bath and no check draws: both streams match a
    # fresh generator after exactly the expected number of consumptions
    last_step = [e for e in evs if e["type"] == "step_completed"][-1]
    assert last_step["thermostat_rng"] == _bath_state_after(12)
    from pyraimd2.runtime.checkpoint import CheckpointManager

    checkpoint = CheckpointManager(run_dir).read_latest_valid()
    n_accepted = sum(1 for p in proposals.values() if p["accepted"])
    check_rng = np.random.default_rng(derive_stream_seed(7, "verification"))
    for _ in range(n_accepted):
        check_rng.random()
    assert checkpoint.state["check_rng"] == check_rng.bit_generator.state

    # old-segment records keep the OLD model's analytic values after the
    # publish; the new segments compute with their generations' models —
    # nothing is rewritten retroactively
    from pyraimd2.energetics.work import residual_work
    from pyraimd2.runtime.identity import model_id_for

    k_by_model = {model_id_for(model, 0): 0.8}
    for update in updates:
        artifact = json.loads(
            (run_dir / "models" / update["model_id"].replace("/", "_")
             / "state.json").read_text())
        k_by_model[update["model_id"]] = float(
            artifact["updater_state"]["surrogate"]["k"])
    assert len(set(k_by_model.values())) > 1  # the model really changed
    for row in _rows(run_dir):
        surrogate = row.data.get("surrogate") or {}
        energy = float(surrogate["energy"])
        x2 = float((row.toatoms().positions**2).sum())
        row_model = (row.data.get("metadata") or {}).get("context", {}).get(
            "model_id")
        assert energy == pytest.approx(0.5 * k_by_model[row_model] * x2,
                                       rel=1e-12)

    # the guard evidence: model error on a fixed, never-trained configuration
    # set improves after the update (guard-style regression, not a
    # generalization claim)
    from ase import Atoms as _Atoms

    reference = Reference(k=K_REF)
    guard = [_Atoms("H", positions=[[0.13, 0.07, -0.05]]),
             _Atoms("H", positions=[[-0.21, 0.11, 0.03]]),
             _Atoms("H", positions=[[0.34, -0.18, 0.09]])]

    def max_force_error(k):
        worst = 0.0
        for atoms in guard:
            true = reference.compute(atoms).forces
            worst = max(worst, float(np.abs(-k * atoms.positions
                                              - true).max()))
        return worst

    assert max_force_error(model.k) < max_force_error(0.8) / 4

    # residual-work closure on labeled boundaries, each with its segment's
    # correction and the model of record of that segment's generation, and
    # the ΔH_ref = W_R + ΔH_anchor identity on the same boundaries
    observed = [e for e in commits.values() if e.get("observed") is not None]
    assert observed
    rows = _rows(run_dir)
    boundaries = _complete_boundaries(run_dir)
    masses = rows[0].toatoms().get_masses()[:, None]
    anchors = _segment_anchors(run_dir)
    for commit in observed:
        index = int(commit["context"]["evaluation_id"])
        segment = int(commit["segment_id"])
        anchor = anchors[segment]
        k_seg = k_by_model[model_id_for(model, int(anchor["model_generation"]))]
        anchor_index = int(anchor["evaluation_index"])
        anchor_positions = np.array(anchor["positions_A"], dtype=float)
        correction = np.array(anchor["correction_eV_A"], dtype=float)
        positions = rows[index].toatoms().positions
        endpoint = residual_work(
            anchor_positions, positions,
            0.5 * k_seg * float((anchor_positions**2).sum()),
            0.5 * k_seg * float((positions**2).sum()),
            float(anchor["reference_energy_eV"]),
            float(rows[index].data["engine"]["energy"]),
            correction)
        assert endpoint == pytest.approx(
            commit["observed"]["endpoint_work_eV"], rel=0, abs=1e-12)

        def kinetic(boundary, _boundaries=boundaries, _masses=masses):
            return float(
                (0.5 * _boundaries[boundary][1]**2 / _masses).sum())

        def anchor_energy(pos, _k=k_seg, _c=correction, _a=anchor_positions):
            return (0.5 * _k * float((pos**2).sum())
                    - float((_c * (pos - _a)).sum()))

        delta_h_anchor = (kinetic(index) + anchor_energy(positions)
                          - kinetic(anchor_index)
                          - anchor_energy(anchor_positions))
        delta_h_ref = (kinetic(index)
                       + float(rows[index].data["engine"]["energy"])
                       - kinetic(anchor_index)
                       - float(anchor["reference_energy_eV"]))
        assert delta_h_ref == pytest.approx(endpoint + delta_h_anchor,
                                            rel=0, abs=1e-12)


def test_a_record_only_callback_regression_is_in_place(tmp_path):
    # The M3B-1 counterexample stays pinned here as acceptance A's first
    # gate: on_label=lambda observation: False still anchors and admits.
    seen = []
    run_dir, _, _ = _run_nvt(
        tmp_path / "gate", updater=lambda observation: seen.append(
            observation.label_id) or False)
    props = _proposals(run_dir)
    assert sum(1 for p in props if p["accepted"]) >= 1
    assert any(p["reason"] == "forecast_accepted" for p in props)
    assert seen


# --- M3B-3 B: one candidate rejection and one publish failure ----------------


def test_b_candidate_rejection_rolls_back_and_run_continues(tmp_path):
    # One candidate comes out energy-force inconsistent: the guard rejects
    # and rolls back to the parent (its state restored — the toy's internal
    # call counter included, so a once-only break must live outside the
    # state), the attempt's cost stays billed on the ledger, and the next
    # firing publishes normally.
    class BreakFirstCandidate(TrainableHarmonic):
        def __init__(self):
            super().__init__()
            self._broken_once = False

        def finetune(self, labels):
            report = super().finetune(labels)
            if not self._broken_once:
                self._broken_once = True
                self.bias = 10.0  # force without an energy term
            return report

    run_dir = tmp_path / "rej"
    model = BreakFirstCandidate()
    updater = _a_policy(model)
    run_dir, model, _engine = _run_nvt(run_dir, updater=updater,
                                       model=model, steps=12)
    evs = events(run_dir)
    rejections = [e for e in evs if e["type"] == "update_rejected"]
    assert len(rejections) == 1
    # the uniform bias is the translation zero mode that the guard's
    # energy-force consistency directions exclude by construction (single
    # atoms check it directly) — the force-growth criterion catches it here
    assert rejections[0]["reason"] == "force_growth"
    updates = [e for e in evs if e["type"] == "model_update"]
    assert updates  # a later, consistent candidate publishes
    # the rejected attempt kept the parent generation zero at its commit;
    # the first published generation is 1 — no phantom advance
    assert updates[0]["generation"] == 1
    from pyraimd2.runtime.identity import model_id_for

    assert rejections[0]["model_id"] == model_id_for(model, 0)
    # the rejection's real costs stay on the ledger (one training callback
    # task per consumed label, including the rejected attempt's), while the
    # rejection is not counted as a successful update
    training_tasks = [t for t in _tasks(run_dir)
                      if t.get("operation") == "training"]
    assert len(training_tasks) == updater.n_consumed
    assert updater.n_rejected == 1
    assert updater.n_updates == len(updates)
    # consumption stayed consistent: the rejected attempt's labels are
    # consumed exactly once (durable label ids), so no resume re-delivers
    consumed = [e for e in evs if e["type"] == "label_consumed"]
    assert any(e["label_id"] == rejections[0]["origin_label_id"]
               for e in consumed)
    # and the run completed its physics
    summary = next(e for e in evs if e["type"] == "run_summary")
    assert summary["n_steps"] == 12


def test_b_publish_failure_stops_then_resume_retries_the_update(tmp_path):
    # The artifact publish fails once (after training succeeded): the
    # rollback domain undoes the candidate, the run stops without any
    # half-published model, and the resume re-delivers the uncommitted
    # label — the retry is billed as a marked recovery task while the
    # crashed attempt's own task stays on the ledger.
    crash_dir = tmp_path / "pub"
    model = TrainableHarmonic()
    updater = _a_policy(model)
    runner = None
    run_dir = crash_dir
    run_dir.mkdir()
    from pyraimd2.runtime.models import ModelRegistry

    def fail_first_publish(self, model_id, record):
        raise OSError("injected publish failure")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(ModelRegistry, "publish", fail_first_publish)
    try:
        with pytest.raises(OSError, match="injected publish failure"):
            runner = EnergeticRunner(
                _atoms(), model, Reference(k=K_REF),
                Store(run_dir / "trajectory.db"), "run", run_dir=run_dir,
                event_log=EventLog(run_dir), checkpoint_interval_steps=4,
                force_budget=0.5, timestep_fs=0.5, time_cap_fs=100.0,
                transverse_cap=1.0, check_probability=0.5, check_seed=7,
                integrator_spec=_spec(), on_label=updater)
            runner.run(12)
    finally:
        monkey.undo()
    evs = events(run_dir)
    assert not [e for e in evs if e["type"] == "model_update"]
    # nothing half-published: no model-id artifact directory exists
    assert [p for p in (run_dir / "models").iterdir()
            if p.is_dir() and p.name != "state-arrays"] == []
    # the parent model state survived the rollback
    assert model.k == 0.8
    crashed_training = [t for t in _tasks(run_dir)
                        if t.get("operation") == "training"]
    assert len(crashed_training) >= 1  # the attempt's cost stayed billed

    model2 = TrainableHarmonic()
    updater2 = _a_policy(model2)
    resumed = EnergeticRunner.resume(
        run_dir, model2, Reference(k=K_REF), updater=updater2,
        event_log_force=True, checkpoint_interval_steps=4)
    resumed.run(13 - resumed.calc.n_evaluations)
    resumed.close()
    evs = events(run_dir)
    updates = [e for e in evs if e["type"] == "model_update"]
    assert updates  # the retried consumption published
    recovery_tasks = [t for t in _tasks(run_dir) if t.get("recovery")]
    assert len(recovery_tasks) >= 1  # the retry is billed and marked
    assert all(t["status"] == "success" for t in recovery_tasks)
    # the model lineage is the one the uninterrupted run would have
    assert updates[0]["generation"] == 1

    # final state equals an uninterrupted control bit-for-bit
    control_dir = tmp_path / "control"
    cmodel = TrainableHarmonic()
    _run_nvt(control_dir, updater=_a_policy(cmodel), model=cmodel, steps=12)
    rows_a, rows_b = _rows(run_dir), _rows(control_dir)
    assert len(rows_a) == len(rows_b)
    for row_a, row_b in zip(rows_a, rows_b):
        np.testing.assert_array_equal(row_a.toatoms().positions,
                                      row_b.toatoms().positions)
        np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                      row_b.toatoms().get_momenta())
    assert model2.k == cmodel.k


# --- M3B-3 C: clean split and true cross-process crash windows ----------------

from pyraimd2.engines.base import EngineResult


class CountingReference:
    """The guard-update toy reference (k=1.2 + quartic) counting every real
    compute call in a shared file, with a per-call drift so a silent
    re-execution is numerically distinguishable, and an armed mode whose
    calls raise — proving a recovery phase never touches the live backend."""

    name = "counting-guard-reference"

    def __init__(self, count_file, *, drift_scale=0.0, armed=False):
        from pathlib import Path as _Path

        self.count_file = _Path(count_file)
        self.drift_scale = drift_scale
        self.armed = armed
        self.inner = Reference(k=K_REF)

    @property
    def fingerprint(self):
        return None  # mirrors the unfingerprinted toy reference

    @property
    def capabilities(self):
        return None

    @property
    def n_calls(self):
        return int(self.count_file.read_text()) \
            if self.count_file.exists() else 0

    def compute(self, atoms):
        if self.armed:
            raise AssertionError("the recovery phase touched a live backend")
        n = self.n_calls + 1
        self.count_file.write_text(str(n))
        label = self.inner.compute(atoms)
        drift = self.drift_scale * n
        return EngineResult(label.energy + drift, label.forces + drift,
                            None, 0.0)


_C_UPDATER = "GuardedUpdater(model, UpdatePolicy(n_label=2, guard_size=1))"

_C_CHILD = '''
import os
import sys
from pathlib import Path

from test_adaptive_nvt_update import CountingReference, _run_nvt
from test_guarded_update import TrainableHarmonic
from pyraimd2.loop import GuardedUpdater, UpdatePolicy
from pyraimd2.runtime.events import EventLog

run_dir = Path(os.environ["RUN_DIR"])
steps = int(os.environ["STEPS"])
kill_key = os.environ.get("KILL_KEY")
kill_after = os.environ.get("KILL_AFTER") == "1"

if kill_key:
    original = EventLog.append_once

    def patched(self, key, event_type, payload):
        if key == kill_key and not kill_after:
            os._exit(73)
        result = original(self, key, event_type, payload)
        if key == kill_key and kill_after:
            os._exit(73)
        return result

    EventLog.append_once = patched

model = TrainableHarmonic()
_run_nvt(run_dir,
         updater=GuardedUpdater(model, UpdatePolicy(n_label=2, guard_size=1)),
         model=model, steps=steps,
         engine=CountingReference(os.environ["COUNT_FILE"], drift_scale=1e-6))
'''


def _c_update_control(tmp_path, *, steps=20):
    """Continuous control run with real updates, its own counter file."""
    count_file = tmp_path / "control.calls"
    run_dir = tmp_path / "control"
    model = TrainableHarmonic()
    _run_nvt(run_dir,
             updater=GuardedUpdater(model, UpdatePolicy(n_label=2,
                                                        guard_size=1)),
             model=model, steps=steps,
             engine=CountingReference(count_file, drift_scale=1e-6))
    return run_dir, count_file, model


def _c_update_child(tmp_path, *, steps, kill_key=None, kill_after=False):
    run_dir = tmp_path / "crashed"
    count_file = tmp_path / "crashed.calls"
    child = tmp_path / "child.py"
    tmp_path.mkdir(parents=True, exist_ok=True)
    child.write_text(textwrap.dedent(_C_CHILD))
    env = dict(os.environ, RUN_DIR=str(run_dir), STEPS=str(steps),
               COUNT_FILE=str(count_file),
               PYTHONPATH=os.pathsep.join(
                   [str(Path(__file__).parents[2] / "src"),
                    str(Path(__file__).parent)]))
    if kill_key is not None:
        env["KILL_KEY"] = kill_key
        env["KILL_AFTER"] = "1" if kill_after else "0"
    result = subprocess.run([sys.executable, str(child)],
                            env=env, capture_output=True, text=True,
                            check=False)
    expected = 73 if kill_key is not None else 0
    assert result.returncode == expected, result.stderr[-500:]
    return run_dir, count_file


def _model_chain(run_dir):
    """The model lineage facts: updates (without the digest, which binds
    measured wall times by design) and per-commit driving identities."""
    evs = events(run_dir)
    updates = [(e["generation"], e["model_id"], e["origin_label_id"])
               for e in evs if e["type"] == "model_update"]
    driving = [(int((e.get("context") or {})["evaluation_id"]), e["model_id"],
                e.get("segment_id"))
               for e in evs if e["type"] == "evaluation_committed"]
    labels = [(e.get("type"), str(e.get("label_id")
                                  or e.get("origin_label_id")))
              for e in evs if e.get("type") in ("label_consumed",
                                                "model_update",
                                                "update_rejected")]
    return updates, driving, labels


def _assert_artifact_digests_self_consistent(run_dir):
    """Every committed update's digest recomputes from its artifact (R2) —
    per-run binding; cross-run equality covers the physical content, not
    volatile timing fields."""
    from pyraimd2.runtime.models import ModelRegistry, artifact_digest

    registry = ModelRegistry(run_dir)
    for event in events(run_dir):
        if event["type"] != "model_update":
            continue
        artifact = registry.read(event["model_id"])
        assert artifact is not None
        assert artifact_digest(artifact) == event["artifact_digest"]
        assert artifact["generation"] == event["generation"]
        assert artifact["label_ids"] == event["label_ids"]


def _c_update_compare(crash_dir, count_file, control_dir, control_calls_file,
                      *, remaining, control_model,
                      expected_extra_reference=0, expected_extra_trainings=0):
    """Resume with an armed sentinel, continue, and compare the complete
    state against the control: rows, boundaries, three streams, model
    chain, label ids, artifacts, ledger, export, final checkpoint."""
    from pyraimd2.runtime.checkpoint import CheckpointManager
    from pyraimd2.workflows.export import export_run

    calls_before = int(count_file.read_text())
    sentinel = CountingReference(count_file, drift_scale=1e-6, armed=True)
    model = TrainableHarmonic()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=2, guard_size=1))
    runner = EnergeticRunner.resume(
        crash_dir, model, sentinel, updater=updater, event_log_force=True,
        checkpoint_interval_steps=4)
    assert sentinel.n_calls == calls_before  # recovery: zero live calls
    sentinel.armed = False
    runner.run(remaining)
    runner.close()

    rows_a, rows_b = _rows(crash_dir), _rows(control_dir)
    assert len(rows_a) == len(rows_b)
    for row_a, row_b in zip(rows_a, rows_b):
        np.testing.assert_array_equal(row_a.toatoms().positions,
                                      row_b.toatoms().positions)
        np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                      row_b.toatoms().get_momenta())
    boundaries_a = _complete_boundaries(crash_dir)
    boundaries_b = _complete_boundaries(control_dir)
    for (pos_a, mom_a), (pos_b, mom_b) in zip(boundaries_a, boundaries_b):
        np.testing.assert_array_equal(pos_a, pos_b)
        np.testing.assert_array_equal(mom_a, mom_b)

    steps_a = {e["step_id"]: e for e in events(crash_dir)
               if e["type"] == "step_completed"}
    steps_b = {e["step_id"]: e for e in events(control_dir)
               if e["type"] == "step_completed"}
    assert steps_a.keys() == steps_b.keys()
    for step_id, event in steps_a.items():
        control_event = steps_b[step_id]
        assert event["state_digest"] == control_event["state_digest"]
        assert event["boundary_digest"] == control_event["boundary_digest"]
        assert event["thermostat_rng"] == control_event["thermostat_rng"]
        assert event["model_id"] == control_event["model_id"]
        assert event["segment_id"] == control_event["segment_id"]
    assert [(p["accepted"], p["reason"], p.get("check_draw"))
            for p in _proposals(crash_dir)] == \
           [(p["accepted"], p["reason"], p.get("check_draw"))
            for p in _proposals(control_dir)]
    assert _model_chain(crash_dir) == _model_chain(control_dir)
    assert _segment_anchors(crash_dir) == _segment_anchors(control_dir)
    _assert_artifact_digests_self_consistent(crash_dir)

    # committed training never re-executes; only genuine retries add calls.
    # The crashed window's own attempt may never have run (its task is then
    # absent) or may have completed uncommitted (its task stays billed); the
    # recovery retry is marked — together they account for every callback
    # the control billed, no more and no fewer.
    control_trainings = [t for t in _tasks(control_dir)
                         if t.get("operation") == "training"]
    crash_trainings = [t for t in _tasks(crash_dir)
                       if t.get("operation") == "training"
                       and not t.get("recovery")]
    recovery_trainings = [t for t in _tasks(crash_dir)
                          if t.get("operation") == "training"
                          and t.get("recovery")]
    assert len(recovery_trainings) == expected_extra_trainings
    assert (len(crash_trainings) + len(recovery_trainings)
            == len(control_trainings))
    assert model.finetune_calls == control_model.finetune_calls
    assert (int(count_file.read_text())
            == int(control_calls_file.read_text())
              + expected_extra_reference)

    # export and the final checkpoint name the same complete boundary
    from ase.io import read as ase_read

    report = export_run(crash_dir, force=True)
    frames = ase_read(report["output"], index=":")
    control_report = export_run(control_dir, force=True)
    control_frames = ase_read(control_report["output"], index=":")
    assert len(frames) == len(control_frames) == len(rows_a)
    for frame, control_frame in zip(frames, control_frames):
        np.testing.assert_array_equal(frame.positions,
                                      control_frame.positions)
        np.testing.assert_array_equal(frame.get_momenta(),
                                      control_frame.get_momenta())
    checkpoint = CheckpointManager(crash_dir).read_latest_valid()
    final_positions, final_momenta = boundaries_a[-1]
    np.testing.assert_array_equal(checkpoint.arrays["positions"],
                                  final_positions)
    np.testing.assert_array_equal(checkpoint.arrays["momenta"],
                                  final_momenta)


def test_c_clean_split_with_updates_20_equals_7_plus_13(tmp_path):
    control_dir, control_calls, cmodel = _c_update_control(tmp_path / "ctl")
    updates = [e for e in events(control_dir) if e["type"] == "model_update"]
    assert any(int(e["origin_evaluation_id"]) <= 7 for e in updates), \
        "an update must sit near the split point"
    assert any(int(e["origin_evaluation_id"]) > 7 for e in updates)
    crash_dir, count_file = _c_update_child(tmp_path / "split", steps=7)
    _c_update_compare(crash_dir, count_file, control_dir, control_calls,
                      remaining=13, control_model=cmodel)


def test_c_committed_evaluation_uncommitted_update_redelivers(tmp_path):
    # os._exit right after the commit of the label whose consumption fires
    # an update: the consumption never persisted, so resume re-delivers it
    # once — the pre-crash training attempt keeps its billed task and the
    # retry is marked recovery; the committed step is never recomputed.
    control_dir, control_calls, cmodel = _c_update_control(tmp_path / "ctl")
    first_update = next(e for e in events(control_dir)
                        if e["type"] == "model_update")
    kill_eval = int(first_update["origin_evaluation_id"])
    crash_dir, count_file = _c_update_child(
        tmp_path / "w", steps=20,
        kill_key=f"evaluation:run:{kill_eval}", kill_after=True)
    _c_update_compare(crash_dir, count_file, control_dir, control_calls,
                      remaining=20 - kill_eval, control_model=cmodel,
                      expected_extra_trainings=1)
    recovery = [t for t in _tasks(crash_dir) if t.get("recovery")]
    assert len(recovery) == 1 and recovery[0]["status"] == "success"


def test_c_model_update_committed_step_not(tmp_path):
    # os._exit right after a MODEL_UPDATE lands but before the step
    # completes: resume loads the published artifact — no retraining — and
    # completes the step with the original frozen driving forces and the
    # same bath increments.
    control_dir, control_calls, cmodel = _c_update_control(tmp_path / "ctl")
    updates = [e for e in events(control_dir) if e["type"] == "model_update"]
    assert len(updates) >= 2
    mid = updates[0]
    kill_eval = int(mid["origin_evaluation_id"])
    crash_dir, count_file = _c_update_child(
        tmp_path / "w", steps=20,
        kill_key=f"model-update:{mid['origin_label_id']}:eval-{kill_eval}",
        kill_after=True)
    _c_update_compare(crash_dir, count_file, control_dir, control_calls,
                      remaining=20 - kill_eval, control_model=cmodel)
    # zero recovery retries: everything up to the crash was committed
    assert not [t for t in _tasks(crash_dir) if t.get("recovery")]
