"""Review R1 regression: resume keeps check stream, label consumption and
database/event identity (F01/F02/F03/F07 counterexamples, hermetic)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from conftest import simulate_crash

from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.runtime.events import EVALUATION_COMMITTED, EventLog
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction


class Model:
    """E = 1/2 k |x|^2, F = -k x; k is the updatable parameter."""

    def __init__(self, k=0.8):
        self.k = k

    def predict(self, atoms):
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None,
                                   np.full(len(atoms), np.nan))

    def state_dict(self):
        return {"k": self.k}

    def load_state_dict(self, state):
        self.k = float(state["k"])


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2, fingerprinted=True):
        self.k = k
        self.attempts = 0
        self.fail_next = False
        self._fingerprinted = fingerprinted

    @property
    def fingerprint(self):
        return f"analytic-reference:k={self.k}" if self._fingerprinted else None

    def compute(self, atoms):
        self.attempts += 1
        if self.fail_next:
            self.fail_next = False
            raise EngineError("deliberate reference failure")
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2)), -self.k * x,
                            None, 0.0)


def _direction(atoms):
    direction = np.zeros((len(atoms), 3))
    direction[:, 0] = 1.0
    return direction


def world(tmp_path, *, update_n=None, stationary=False, p=0.5,
          fingerprinted=True):
    run_dir = tmp_path
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.0 if stationary else 0.2, 0, 0]])
    atoms.set_velocities([[0.0 if stationary else 0.1, 0, 0]])
    model = Model()
    engine = Reference(fingerprinted=fingerprinted)
    updater = (None if update_n is None
               else GuardedUpdater(model, UpdatePolicy(n_label=update_n,
                                                       guard_size=1)))
    runner = EnergeticRunner(
        atoms, model, engine, Store(run_dir / "trajectory.db"), "run",
        on_label=updater, run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=100, direction=_direction,
        force_budget=0.08, timestep_fs=0.1, time_cap_fs=2.0,
        check_probability=p, check_seed=2)
    return runner, model, engine, updater


def resume_world(run_dir, update_n=None, model=None, force=False,
                 fingerprinted=True):
    model = Model() if model is None else model
    updater = (None if update_n is None
               else GuardedUpdater(model, UpdatePolicy(n_label=update_n,
                                                       guard_size=1)))
    runner = EnergeticRunner.resume(run_dir, model,
                                    Reference(fingerprinted=fingerprinted),
                                    updater=updater, direction=_direction,
                                    event_log_force=force,
                                    checkpoint_interval_steps=100)
    return runner, model, updater


def events(run_dir):
    import json

    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


def proposal_draws(run_dir):
    return [event["check_draw"] for event in events(run_dir)
            if event["type"] == "evaluation_proposed" and event["accepted"]]


def test_f01_pending_draw_rng_state_survives_failed_check_resume(tmp_path):
    crashed, _, engine, _ = world(tmp_path / "rng-crash")
    crashed.run(0)  # initial evaluation + initial-boundary checkpoint
    engine.fail_next = True
    with pytest.raises(EngineError, match="deliberate"):
        crashed.run(1)  # eval 1: accepted, checked, reference check fails
    simulate_crash(crashed)

    resumed, _, _ = resume_world(tmp_path / "rng-crash", force=True)
    resumed.run(3)
    resumed.close()
    control, _, _, _ = world(tmp_path / "rng-control")
    control.run(3)
    control.close()
    assert proposal_draws(tmp_path / "rng-crash") == \
           proposal_draws(tmp_path / "rng-control")
    np.testing.assert_allclose(proposal_draws(tmp_path / "rng-crash"),
                               [0.2616121342493164, 0.2984911434141233,
                                0.8142257405942803],
                               rtol=0, atol=1e-15)


def test_f02_resume_reuses_durable_label_without_retraining(tmp_path):
    stopped, _, _, _ = world(tmp_path / "cache-training", update_n=2,
                             stationary=True, p=1.0)
    stopped.run(0)  # initial anchor: label consumed once, checkpoint written
    stopped.close()

    resumed, resumed_model, resumed_updater = resume_world(
        tmp_path / "cache-training", update_n=2)
    resumed.run(1)  # same static geometry: accepted + checked, cache hit
    assert resumed_updater.n_consumed == 1
    assert resumed_updater.n_updates == 0
    assert resumed_model.k == pytest.approx(0.8)

    control, control_model, _, control_updater = world(
        tmp_path / "cache-control", update_n=2, stationary=True, p=1.0)
    control.run(0)
    control.run(1)
    assert control_updater.n_consumed == resumed_updater.n_consumed
    assert control_updater.n_updates == resumed_updater.n_updates
    assert control_model.k == resumed_model.k
    resumed.close()
    control.close()


def test_f07_cache_reuse_in_window_is_not_a_consumption_loss(tmp_path):
    stopped, _, _, _ = world(tmp_path / "cache-crash", update_n=100,
                             stationary=True, p=1.0)
    stopped.run(0)
    stopped.run(2)  # two checked accepted evaluations reusing the label
    stopped.close()  # checkpoint only covers the initial evaluation
    hits = [event for event in events(tmp_path / "cache-crash")
            if event.get("type") == "task" and event.get("cache_hit")]
    assert hits  # the window really reused the durable label

    resumed, _, resumed_updater = resume_world(tmp_path / "cache-crash",
                                               update_n=100)
    resumed.run(1)  # resume must not refuse the run as consumption-lost
    assert resumed_updater.n_consumed == 1
    assert resumed_updater.n_updates == 0
    resumed.close()


def test_f03_commit_binds_row_and_orphan_rows_are_not_authoritative(tmp_path):
    runner, _, _, _ = world(tmp_path / "row-crash", fingerprinted=False)
    runner.run(0)  # initial evaluation committed
    log = runner.calc._event_log
    original_append_once = log.append_once

    def fail_commit(key, event_type, payload):
        if (event_type == EVALUATION_COMMITTED
                and payload["context"]["evaluation_id"] == 1):
            raise RuntimeError("injected crash between db write and commit")
        return original_append_once(key, event_type, payload)

    log.append_once = fail_commit
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(1)
    runner.close()  # row for eval 1 written; its commit never happened

    resumed, _, _ = resume_world(tmp_path / "row-crash", force=True,
                                 fingerprinted=False)
    resumed.run(1)  # re-executes the reference for the uncommitted proposal
    resumed.close()

    store = Store(tmp_path / "row-crash" / "trajectory.db")
    step_rows = list(store._db.select(run_id="run", step=0))
    assert len(step_rows) == 2  # orphan row plus the committed re-execution
    commits = [event for event in events(tmp_path / "row-crash")
               if event["type"] == EVALUATION_COMMITTED
               and event["context"]["evaluation_id"] == 1]
    assert len(commits) == 1
    committed = store.committed_row(resumed.calc._event_log, "run", 1)
    assert int(committed.id) == int(commits[0]["row_id"])
    assert committed.data.get("engine_label_id") == commits[0]["label_id"]
    # The orphan row is a different label ID and is never authoritative.
    orphans = [row for row in step_rows if int(row.id) != int(committed.id)]
    assert len(orphans) == 1
    assert orphans[0].data.get("engine_label_id") != \
        committed.data.get("engine_label_id")


def test_f03_orphan_row_is_adopted_when_its_label_is_reused(tmp_path):
    runner, _, _, _ = world(tmp_path / "row-adopt")
    runner.run(0)
    log = runner.calc._event_log
    original_append_once = log.append_once

    def fail_commit(key, event_type, payload):
        if (event_type == EVALUATION_COMMITTED
                and payload["context"]["evaluation_id"] == 1):
            raise RuntimeError("injected crash between db write and commit")
        return original_append_once(key, event_type, payload)

    log.append_once = fail_commit
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(1)
    attempts_at_crash = runner.calc.n_reference
    runner.close()

    resumed, _, _ = resume_world(tmp_path / "row-adopt", force=True)
    resumed.run(1)  # cache-hit check: the orphan label is genuine and reused
    resumed.close()

    store = Store(tmp_path / "row-adopt" / "trajectory.db")
    step_rows = list(store._db.select(run_id="run", step=0))
    # The orphan row is adopted by the commit — no duplicate write, no
    # duplicate physical execution for the check.
    assert len(step_rows) == 1
    commits = [event for event in events(tmp_path / "row-adopt")
               if event["type"] == EVALUATION_COMMITTED
               and event["context"]["evaluation_id"] == 1]
    assert len(commits) == 1
    assert commits[0]["label_id"] == step_rows[0].data.get("engine_label_id")
    assert int(commits[0]["row_id"]) == int(step_rows[0].id)
    # No new physical execution after resume: the orphan attempt stays billed
    # (one successful check execution, one cache hit — never a second SCF),
    # while the logical reference counter only counts committed executions.
    check_tasks = [event for event in events(tmp_path / "row-adopt")
                   if event.get("type") == "task"
                   and event.get("purpose") == "verification"]
    assert sum(1 for event in check_tasks if event["status"] == "success") == 1
    assert sum(1 for event in check_tasks if event["status"] == "cache_hit") == 1
    assert resumed.calc.n_reference == attempts_at_crash - 1


def test_store_dedupe_and_row_digest(tmp_path):
    store = Store(tmp_path / "dedupe.db")
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    label = EngineResult(1.0, np.zeros((1, 3)), None, 0.0)
    first = store.append("run", 0, atoms, "dft", engine=label,
                         driving=label, label_id="run-label-1", dedupe=True)
    again = store.append("run", 0, atoms, "dft", engine=label,
                         driving=label, label_id="run-label-1", dedupe=True)
    assert first == again
    assert len(list(store._db.select(run_id="run"))) == 1
    row = store.row_by_id(first)
    assert store.row_digest(row) == store.row_digest(row)
