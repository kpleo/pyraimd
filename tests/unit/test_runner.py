"""Runner tests: route counts, bit-exact restart, engine-failure propagation.

Step-numbering convention (see SwitchingCalculator): the force evaluation at
the initial geometry is logged as step -1, so a fresh ``run(n)`` logs steps
-1 .. n-1 and ScheduledSwitch(5) fires dft at steps 0, 5, 10, 15.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import CLUSTER_R0, FakeEngine, FakeSurrogate

from pyraimd2.engines.base import EngineError
from pyraimd2.loop import Runner
from pyraimd2.store import Store
from pyraimd2.switch import ScheduledSwitch


def _build_runner(tmp_path, cluster, run_id: str, db_name: str, period: int = 5,
                  bias: float = 0.05, fail: bool = False):
    store = Store(tmp_path / db_name)
    engine = FakeEngine(CLUSTER_R0, fail=fail)
    surrogate = FakeSurrogate(CLUSTER_R0, bias_amplitude=bias)
    switch = ScheduledSwitch(period)
    runner = Runner(
        cluster.copy(), surrogate, engine, switch, store,  # Runner mutates atoms in place
        run_id=run_id, timestep_fs=0.5, temperature_K=300.0,
    )
    return runner, store, engine, surrogate


def test_route_counts(tmp_path, cluster) -> None:
    runner, store, engine, surrogate = _build_runner(tmp_path, cluster, "a", "a.db")
    summary = runner.run(20)

    assert summary.n_steps == 20
    assert summary.n_dft == 4  # steps 0, 5, 10, 15
    assert engine.calls == 4
    assert surrogate.calls == 21  # every evaluation, incl. shadow + initial
    assert summary.dft_fraction == pytest.approx(4 / 21)

    import ase.db

    rows = list(ase.db.connect(str(tmp_path / "a.db")).select(run_id="a"))
    steps = sorted(int(r.key_value_pairs["step"]) for r in rows)
    assert steps == list(range(-1, 20))  # every evaluation logged, none duplicated
    dft_steps = sorted(
        int(r.key_value_pairs["step"]) for r in rows if r.key_value_pairs["route"] == "dft"
    )
    assert dft_steps == [0, 5, 10, 15]
    for row in rows:
        assert row.key_value_pairs["route"] in ("ml", "dft")
        assert row.data["reason"]
        assert row.data["surrogate"] is not None  # shadow prediction on dft rows
        if row.key_value_pairs["route"] == "dft":
            assert row.data["engine"] is not None
        else:
            assert row.data["engine"] is None

    # Shadow error: bias amplitude 0.05, |sin| <= 1, 4 atoms, 3 components.
    assert 0.0 < summary.force_mae_ev_a <= 0.05 * np.sqrt(3)
    assert summary.force_max_ev_a >= summary.force_mae_ev_a
    assert np.isfinite(summary.wall_time_s) and summary.wall_time_s >= 0.0

    # DFT labels accumulated in the store (design doc §4.2).
    assert len(list(store.iter_labels("a"))) == 4


def test_restart_equality(tmp_path, cluster) -> None:
    # Uninterrupted reference run.
    runner_a, _store_a, _, _ = _build_runner(tmp_path, cluster, "a", "a.db")
    runner_a.run(20)
    final_positions = runner_a.atoms.get_positions().copy()
    final_momenta = runner_a.atoms.get_momenta().copy()

    # Interrupted run: 8 steps, then resume from the Store with fresh objects
    # (new fakes, new switch, new calculator — the Store is the only carrier).
    runner_b, store_b, engine_b, _ = _build_runner(tmp_path, cluster, "b", "b.db")
    runner_b.run(8)
    assert engine_b.calls == 2  # steps 0, 5

    resumed = Runner.resume(
        store_b, "b",
        FakeSurrogate(CLUSTER_R0, bias_amplitude=0.05),
        FakeEngine(CLUSTER_R0),
        ScheduledSwitch(5),
        timestep_fs=0.5,
    )
    assert resumed.calc.step == 8  # continues where the log ended
    summary_b2 = resumed.run(12)

    np.testing.assert_allclose(resumed.atoms.get_positions(), final_positions, atol=0, rtol=0)
    np.testing.assert_allclose(resumed.atoms.get_momenta(), final_momenta, atol=0, rtol=0)
    assert summary_b2.n_dft == 2  # steps 10, 15 of the continued run

    # The resumed store holds the same logged steps as the uninterrupted one.
    import ase.db

    steps_a = sorted(
        int(r.key_value_pairs["step"])
        for r in ase.db.connect(str(tmp_path / "a.db")).select(run_id="a")
    )
    steps_b = sorted(
        int(r.key_value_pairs["step"])
        for r in ase.db.connect(str(tmp_path / "b.db")).select(run_id="b")
    )
    assert steps_a == steps_b == list(range(-1, 20))


def test_engine_failure_propagates(tmp_path, cluster) -> None:
    # period 1: every evaluation is dft, so the very first one fails.
    runner, _store, engine, _ = _build_runner(
        tmp_path, cluster, "fail", "fail.db", period=1, fail=True
    )
    with pytest.raises(EngineError, match="told to fail"):
        runner.run(5)
    assert engine.calls == 0  # failed calls are not counted
    assert runner.calc.n_dft == 0
    import ase.db

    rows = list(ase.db.connect(str(tmp_path / "fail.db")).select(run_id="fail"))
    assert rows == []  # nothing logged for a failed evaluation


def test_momenta_thermalized_only_when_missing(tmp_path, cluster) -> None:
    from ase import Atoms
    from conftest import CLUSTER_POSITIONS

    def _bare_runner(atoms, run_id, db_name):
        return Runner(
            atoms, FakeSurrogate(CLUSTER_R0), FakeEngine(CLUSTER_R0),
            ScheduledSwitch(5), Store(tmp_path / db_name),
            run_id=run_id, timestep_fs=0.5, temperature_K=300.0,
        )

    # Momenta present (fixture): untouched by the Runner.
    before = cluster.get_momenta().copy()
    _bare_runner(cluster, "keep", "keep.db")
    np.testing.assert_array_equal(cluster.get_momenta(), before)

    # No momenta: thermalized once at init.
    cold = Atoms("H4", positions=CLUSTER_POSITIONS.copy())
    assert "momenta" not in cold.arrays
    _bare_runner(cold, "therm", "therm.db")
    assert "momenta" in cold.arrays
    assert np.any(cold.get_momenta() != 0.0)
