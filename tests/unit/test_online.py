"""OnlineUpdater and the SwitchingCalculator on_label hook (M2 §3)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from conftest import CLUSTER_POSITIONS, CLUSTER_R0, FakeCommittee, FakeEngine

from pyraimd2.engines.base import EngineError
from pyraimd2.loop import OnlineUpdater, Runner
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.switch import ConformalSwitch, LabelObservation, ScheduledSwitch

FRAME = Atoms("H4", positions=CLUSTER_POSITIONS + 0.01)


def _observation(step: int, atoms: Atoms, spread: float = 0.01) -> LabelObservation:
    return LabelObservation(
        step=step,
        atoms=atoms,
        prediction=FakeCommittee(CLUSTER_R0, bias_amplitude=0.05, spread=spread)
        .predict(atoms),
        label=FakeEngine(CLUSTER_R0).compute(atoms),
    )


def test_finetune_fires_every_n_labels() -> None:
    committee = FakeCommittee(CLUSTER_R0)
    observed: list[tuple[float, float]] = []
    updater = OnlineUpdater(
        committee, observe=lambda s, e: observed.append((s, e)), n_label=3
    )
    for step in range(10):
        updater(_observation(step, FRAME))

    assert updater.n_observations == 10
    assert committee.finetune_calls == 3  # at labels 3, 6, 9
    assert committee.finetune_sizes == [3, 6, 9]  # internal accumulation grows
    assert len(updater.reports) == 3
    assert len(observed) == 10  # every label reached the window


def test_checkpoint_hook_fires_after_each_finetune() -> None:
    """Restart safety: the checkpoint callback fires exactly once per
    fine-tune, after the report is recorded."""
    committee = FakeCommittee(CLUSTER_R0)
    checkpoints: list[int] = []
    updater = OnlineUpdater(
        committee,
        observe=lambda s, e: None,
        n_label=2,
        checkpoint=lambda: checkpoints.append(1),
    )
    for step in range(5):
        updater(_observation(step, FRAME))

    assert committee.finetune_calls == 2  # at labels 2 and 4
    assert len(checkpoints) == 2
    assert updater.n_finetunes == 2


def test_observe_receives_correct_s_and_e() -> None:
    committee = FakeCommittee(CLUSTER_R0)
    observed: list[tuple[float, float]] = []
    updater = OnlineUpdater(
        committee, observe=lambda s, e: observed.append((s, e)), n_label=8
    )
    updater(_observation(0, FRAME, spread=0.02))

    prediction = FakeCommittee(CLUSTER_R0, bias_amplitude=0.05, spread=0.02).predict(FRAME)
    label = FakeEngine(CLUSTER_R0).compute(FRAME)
    expected_e = float(np.max(np.linalg.norm(prediction.forces - label.forces, axis=1)))
    assert observed[0][0] == pytest.approx(0.02)
    assert observed[0][1] == pytest.approx(expected_e)


def test_label_source_overrides_internal_accumulation() -> None:
    committee = FakeCommittee(CLUSTER_R0)
    fixed_labels = [("atoms", "label")] * 5  # FakeCommittee only counts them
    updater = OnlineUpdater(
        committee, observe=lambda s, e: None, n_label=2,
        label_source=lambda: fixed_labels,
    )
    for step in range(4):
        updater(_observation(step, FRAME))
    assert committee.finetune_sizes == [5, 5]


def test_updater_rejects_bad_args_and_shape_mismatch() -> None:
    committee = FakeCommittee(CLUSTER_R0)
    with pytest.raises(ValueError, match="n_label"):
        OnlineUpdater(committee, observe=lambda s, e: None, n_label=0)

    updater = OnlineUpdater(committee, observe=lambda s, e: None, n_label=1)
    bad_prediction = SurrogatePrediction(
        energy=0.0,
        forces=np.zeros((3, 3)),
        stress=None,
        uncertainty=np.full(3, 0.01),
    )
    bad = LabelObservation(
        step=0, atoms=FRAME, prediction=bad_prediction,
        label=FakeEngine(CLUSTER_R0).compute(FRAME),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        updater(bad)


def test_calculator_hook_fires_on_dft_steps_only(tmp_path, cluster) -> None:
    observations: list[LabelObservation] = []
    runner = Runner(
        cluster.copy(), FakeCommittee(CLUSTER_R0, spread=0.01), FakeEngine(CLUSTER_R0),
        ScheduledSwitch(2), Store(tmp_path / "hook.db"), run_id="hook",
        timestep_fs=0.5, on_label=observations.append,
    )
    summary = runner.run(10)

    assert summary.n_dft == 5  # steps 0, 2, 4, 6, 8
    assert [obs.step for obs in observations] == [0, 2, 4, 6, 8]
    for obs in observations:
        assert obs.prediction.forces.shape == obs.label.forces.shape
        assert np.isfinite(obs.prediction.uncertainty).all()


def test_hook_not_called_when_engine_fails(tmp_path, cluster) -> None:
    observations: list[LabelObservation] = []
    runner = Runner(
        cluster.copy(), FakeCommittee(CLUSTER_R0), FakeEngine(CLUSTER_R0, fail=True),
        ScheduledSwitch(1), Store(tmp_path / "fail.db"), run_id="fail",
        timestep_fs=0.5, on_label=observations.append,
    )
    with pytest.raises(EngineError, match="told to fail"):
        runner.run(3)
    assert observations == []


def test_live_loop_full_online_wiring(tmp_path, cluster) -> None:
    """The M2 chain in the live loop: ConformalSwitch + OnlineUpdater through
    the calculator hook — every engine call lands in the window, and the
    fine-tune trigger fires on schedule."""
    committee = FakeCommittee(CLUSTER_R0, bias_amplitude=0.05, spread=0.01)
    switch = ConformalSwitch(committee, eps_acc=0.05, window=8, w_min=2)
    updater = OnlineUpdater(committee, observe=switch.observe, n_label=3)
    store = Store(tmp_path / "live.db")
    engine = FakeEngine(CLUSTER_R0)
    runner = Runner(
        cluster.copy(), committee, engine, switch, store, run_id="live",
        timestep_fs=0.5, on_label=updater,
    )
    summary = runner.run(12)

    assert updater.n_observations == summary.n_dft == engine.calls
    assert switch.window_size == min(engine.calls, 8)
    assert committee.finetune_calls == engine.calls // 3
    assert committee.finetune_sizes == sorted(committee.finetune_sizes)
    # First evaluations are cold-start dft.
    routes = [
        row.key_value_pairs["route"]
        for row in sorted(
            store._db.select(run_id="live"),
            key=lambda r: r.key_value_pairs["step"],
        )
    ]
    assert routes[:2] == ["dft", "dft"]
