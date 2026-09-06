"""Exploration-label protocol tests (referee §2.4; pre-registered knob).

The knob: on an ACCEPTED ("ml") step, with probability ``explore_frac`` the
engine label is computed anyway (shadow label).  It enters the calibration
window/updater like any label — that is its purpose — but the step still
propagates with the surrogate forces, so the certified trajectory is
bit-identical to ``explore_frac = 0``.  Rows keep ``route="ml"`` with
"explore-label" in the reason.

Streak invariant asserted here (T4d): an explore label does NOT reset the
conformal streak.  The streak lives in the decision path
(``ConformalSwitch.assess`` increments on an "ml" decision, resets on a
"dft" decision); ``observe`` is pure window ingestion.  Live, resumed
(``Store.trailing_ml_streak``), and audit-parsed streaks therefore all
count accepted steps since the last DECISION-DRIVEN label.

Setup: w_min=2 so steps -1 and 0 are cold-start "dft" and every step >= 1
accepts (FakeCommittee spread 0.1, bias 0.05 -> qhat*(s+delta) ~ 0.05 <<
eps_acc=1.0).
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import CLUSTER_R0, FakeCommittee, FakeEngine

import ase.db

from pyraimd2.loop import OnlineUpdater, Runner, SwitchingCalculator
from pyraimd2.store import Store
from pyraimd2.switch import ConformalSwitch

W_MIN, WINDOW, EPS = 2, 8, 1.0
SEED = 20250819


def _build(tmp_path, cluster, run_id, explore_frac=0.0, seed=SEED,
           streak_rho=0.0, db_name=None):
    """Runner + wired ConformalSwitch/OnlineUpdater mirroring adaptive_loop."""
    store = Store(tmp_path / (db_name or f"{run_id}.db"))
    surrogate = FakeCommittee(CLUSTER_R0, bias_amplitude=0.05, spread=0.1)
    engine = FakeEngine(CLUSTER_R0)
    switch = ConformalSwitch(
        surrogate, alpha=0.05, eps_acc=EPS, window=WINDOW, w_min=W_MIN,
        streak_rho=streak_rho,
    )
    updater = OnlineUpdater(surrogate, observe=switch.observe, n_label=1000)
    runner = Runner(
        cluster.copy(), surrogate, engine, switch, store,  # Runner mutates atoms
        run_id=run_id, timestep_fs=0.5, temperature_K=300.0,
        on_label=updater,
        explore_frac=explore_frac,
        explore_rng=np.random.default_rng(seed) if explore_frac > 0.0 else None,
        explore_seed=seed,
    )
    return runner, store, engine, surrogate, switch, updater


def _rows(tmp_path, db_name, run_id):
    rows = list(ase.db.connect(str(tmp_path / db_name)).select(run_id=run_id))
    return sorted(rows, key=lambda r: r.key_value_pairs["step"])


def test_explore_off_reproduces_baseline_exactly(tmp_path, cluster) -> None:
    """explore_frac=0.0 (with an rng plumbed in) is the baseline, bit-for-bit:
    same trajectory, same routes, same reasons, same payloads."""
    runner_a, *_ = _build(tmp_path, cluster, "base", db_name="base.db")
    runner_a.run(10)
    runner_b, *_ = _build(tmp_path, cluster, "off", explore_frac=0.0,
                          db_name="off.db")
    runner_b.run(10)

    np.testing.assert_array_equal(runner_b.atoms.get_positions(),
                                  runner_a.atoms.get_positions())
    np.testing.assert_array_equal(runner_b.atoms.get_momenta(),
                                  runner_a.atoms.get_momenta())
    rows_a = _rows(tmp_path, "base.db", "base")
    rows_b = _rows(tmp_path, "off.db", "off")
    assert len(rows_a) == len(rows_b) == 11  # steps -1..9
    for ra, rb in zip(rows_a, rows_b):
        assert ra.key_value_pairs["route"] == rb.key_value_pairs["route"]
        assert ra.data["reason"] == rb.data["reason"]
        assert (ra.data["engine"] is None) == (rb.data["engine"] is None)
        assert "explore-label" not in rb.data["reason"]


def test_explore_one_labels_every_accepted_step(tmp_path, cluster) -> None:
    """explore_frac=1.0: every accepted step carries a shadow engine label;
    route stays "ml", propagation still uses the surrogate forces, and the
    window receives every explore label."""
    runner, store, engine, surrogate, switch, updater = _build(
        tmp_path, cluster, "x1", explore_frac=1.0, db_name="x1.db"
    )
    runner.run(10)  # steps -1..9: dft at -1,0 (cold start), ml at 1..9

    assert engine.calls == 2 + 9  # 2 decision-driven + 9 explore labels
    assert runner.calc.n_dft == 2
    assert runner.calc.n_ml == 9
    assert runner.calc.n_explore == 9
    assert updater.n_observations == 11  # window saw every label
    assert switch.window_size == WINDOW  # 11 pairs, capped at the window

    rows = _rows(tmp_path, "x1.db", "x1")
    for row in rows:
        route = row.key_value_pairs["route"]
        step = int(row.key_value_pairs["step"])
        if step >= 1:
            assert route == "ml"  # the explore label does NOT re-route
            assert row.data["engine"] is not None  # shadow label stored
            assert "explore-label" in row.data["reason"]
            assert f"seed={SEED}" in row.data["reason"]
        else:
            assert route == "dft"
            assert "explore-label" not in row.data["reason"]

    # Certified trajectory unchanged: bit-equal to the explore-off run, and
    # the stored driving forces on explore rows are the surrogate's.
    runner_b, _store_b, *_ = _build(tmp_path, cluster, "x0", db_name="x0.db")
    runner_b.run(10)
    np.testing.assert_array_equal(runner.atoms.get_positions(),
                                  runner_b.atoms.get_positions())
    np.testing.assert_array_equal(runner.atoms.get_momenta(),
                                  runner_b.atoms.get_momenta())
    energy, forces = store.driving_label("x1", 9)  # an explore-label row
    sur = next(r for r in rows if r.key_value_pairs["step"] == 9).data["surrogate"]
    np.testing.assert_array_equal(forces, np.asarray(sur["forces"]))
    eng = next(r for r in rows if r.key_value_pairs["step"] == 9).data["engine"]
    assert not np.allclose(forces, np.asarray(eng["forces"]))  # bias != 0

    # The resume-facing streams see the explore labels exactly as the live
    # switch did (window rebuild + fine-tune label set).
    assert [s for s, _, _ in store.iter_observations("x1")] == list(range(-1, 10))
    assert len(list(store.iter_labels("x1"))) == 11


def test_explore_label_does_not_reset_streak(tmp_path, cluster) -> None:
    """T4d invariant: with streak_rho > 0 the reason strings show k
    incrementing across explore-labeled accepts; live switch, store
    trailing streak, and a resume-time rebuild all agree."""
    runner, store, engine, surrogate, switch, updater = _build(
        tmp_path, cluster, "xs", explore_frac=1.0, streak_rho=0.05,
        db_name="xs.db",
    )
    runner.run(10)  # 9 consecutive accepted steps, each explore-labeled

    assert switch._streak == 9  # NOT reset by the 9 explore labels
    assert store.trailing_ml_streak("xs") == 9  # audit tooling agrees
    ks = []
    for row in _rows(tmp_path, "xs.db", "xs"):
        if row.key_value_pairs["route"] == "ml":
            m = re.search(r"streak k=(\d+)", row.data["reason"])
            assert m is not None
            ks.append(int(m.group(1)))
            assert "explore-label" in row.data["reason"]
    assert ks == list(range(9))  # k = 0..8, incremented per accepted step

    # Resume-time window rebuild replays the explore labels: same window.
    fresh = ConformalSwitch(surrogate, alpha=0.05, eps_acc=EPS, window=WINDOW,
                            w_min=W_MIN, streak_rho=0.05)
    for _, s, e in store.iter_observations("xs"):
        fresh.observe(s, e)
    assert fresh.window_size == switch.window_size
    assert fresh.qhat() == pytest.approx(switch.qhat())


def test_explore_draws_are_deterministic_given_seed(tmp_path, cluster) -> None:
    """Same seed -> same explore schedule; frac 0.5 labels a strict random
    subset of the accepted steps and nothing else."""
    def explore_steps(db_name):
        return [
            int(r.key_value_pairs["step"])
            for r in _rows(tmp_path, db_name, db_name[:-3])
            if "explore-label" in r.data["reason"]
        ]

    _build(tmp_path, cluster, "d1", explore_frac=0.5, db_name="d1.db")[0].run(10)
    _build(tmp_path, cluster, "d2", explore_frac=0.5, db_name="d2.db")[0].run(10)
    steps1, steps2 = explore_steps("d1.db"), explore_steps("d2.db")
    assert steps1 == steps2  # reproducible
    assert 0 < len(steps1) < 9  # a proper subset of the 9 accepts
    assert all(st >= 1 for st in steps1)  # accepted steps only


def test_explore_requires_rng_and_unit_interval(tmp_path, cluster) -> None:
    store = Store(tmp_path / "v.db")
    surrogate = FakeCommittee(CLUSTER_R0, spread=0.1)
    switch = ConformalSwitch(surrogate, eps_acc=EPS, w_min=W_MIN)
    with pytest.raises(ValueError, match="explore_frac"):
        SwitchingCalculator(surrogate, FakeEngine(CLUSTER_R0), switch, store,
                            "v", explore_frac=1.5)
    with pytest.raises(ValueError, match="explore_rng"):
        SwitchingCalculator(surrogate, FakeEngine(CLUSTER_R0), switch, store,
                            "v", explore_frac=0.5)  # no rng -> fail loud


def test_explore_label_calculator_returns_surrogate_forces(tmp_path, cluster) -> None:
    """Direct calculator-level check: on an explore step the forces handed
    to the integrator are the surrogate prediction, not the engine label."""
    store = Store(tmp_path / "c.db")
    surrogate = FakeCommittee(CLUSTER_R0, bias_amplitude=0.05, spread=0.1)
    engine = FakeEngine(CLUSTER_R0)
    switch = ConformalSwitch(surrogate, alpha=0.05, eps_acc=EPS, w_min=1,
                             window=WINDOW)
    switch.observe(0.1, 0.02)  # pre-warmed window -> step 0 accepts
    calc = SwitchingCalculator(
        surrogate, engine, switch, store, "c",
        explore_frac=1.0, explore_rng=np.random.default_rng(7), explore_seed=7,
    )
    atoms = cluster.copy()
    atoms.calc = calc
    forces = atoms.get_forces()
    np.testing.assert_array_equal(forces, surrogate.predict(atoms).forces)
    assert engine.calls == 1 and calc.n_explore == 1 and calc.n_dft == 0
    row = next(iter(ase.db.connect(str(tmp_path / "c.db")).select(run_id="c")))
    assert row.key_value_pairs["route"] == "ml"
    assert "explore-label" in row.data["reason"]
