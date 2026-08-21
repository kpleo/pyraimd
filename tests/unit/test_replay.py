"""Offline replay end-to-end with the fake committee (design-m2.md §5)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from conftest import CLUSTER_POSITIONS, CLUSTER_R0, FakeCommittee, FakeEngine

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import OnlineUpdater
from pyraimd2.switch import ConformalSwitch, ScheduledSwitch, replay

N_FRAMES = 24


def _frames() -> list[tuple[Atoms, EngineResult]]:
    """Deterministic drifting cluster + engine truth per frame."""
    engine = FakeEngine(CLUSTER_R0)
    frames = []
    for step in range(N_FRAMES):
        atoms = Atoms("H4", positions=CLUSTER_POSITIONS + 0.01 * step)
        frames.append((atoms, engine.compute(atoms)))
    return frames


def test_replay_conformal_end_to_end() -> None:
    committee = FakeCommittee(CLUSTER_R0, bias_amplitude=0.05, spread=0.01, improve=0.5)
    switch = ConformalSwitch(committee, alpha=0.05, eps_acc=0.05, window=8, w_min=3)
    updater = OnlineUpdater(committee, observe=switch.observe, n_label=4)

    records, summary = replay(_frames(), committee, switch, updater=updater, eps_acc=0.05)

    assert len(records) == summary.n_frames == N_FRAMES
    # Cold start: the first w_min frames must all be dft.
    assert [r.route for r in records[:3]] == ["dft"] * 3
    assert records[0].qhat == float("inf")  # empty window at frame 0
    assert all(r.spread == pytest.approx(0.01) for r in records)
    # Per-frame error is the max per-atom force error; check frame 0 exactly
    # (no fine-tune has fired yet, so the bias is the initial one).
    dr0 = CLUSTER_POSITIONS - CLUSTER_R0
    expected_e0 = float(np.max(np.linalg.norm(0.05 * np.sin(dr0), axis=1)))
    assert records[0].error == pytest.approx(expected_e0)
    # Violation flags are consistent with (route, error).
    for r in records:
        assert r.violation == (r.route == "ml" and r.error > 0.05)
    # Summary bookkeeping.
    assert summary.n_dft == sum(r.route == "dft" for r in records)
    assert summary.dft_fraction == pytest.approx(summary.n_dft / N_FRAMES)
    assert summary.n_accepted == N_FRAMES - summary.n_dft
    assert summary.n_finetunes == committee.finetune_calls == summary.n_dft // 4
    assert committee.finetune_sizes == sorted(committee.finetune_sizes)
    accepted = [r for r in records if r.route == "ml"]
    expected_alpha = (
        float(np.mean([r.violation for r in accepted])) if accepted else float("nan")
    )
    assert summary.alpha_hat == pytest.approx(expected_alpha)
    assert np.isfinite(summary.wall_time_s)

    # The committee did learn: later frames are mostly accepted.
    assert summary.n_dft < N_FRAMES


def test_replay_scheduled_ablation_path() -> None:
    committee = FakeCommittee(CLUSTER_R0, bias_amplitude=0.05, spread=0.01)
    switch = ScheduledSwitch(3)
    updater = OnlineUpdater(committee, observe=lambda s, e: None, n_label=4)

    records, summary = replay(_frames(), committee, switch, updater=updater, eps_acc=0.05)

    dft_steps = {r.step for r in records if r.route == "dft"}
    assert dft_steps == {0, 3, 6, 9, 12, 15, 18, 21}  # exactly the schedule
    assert summary.n_dft == 8
    assert summary.n_finetunes == 2  # at observations 4 and 8
    assert all(np.isnan(r.qhat) and np.isnan(r.bound) for r in records)
    # Conformal quantities absent, but e and violations are still scored.
    accepted = [r for r in records if r.route == "ml"]
    assert summary.alpha_hat == pytest.approx(
        float(np.mean([r.violation for r in accepted]))
    )


def test_replay_rejects_empty_frames_and_bad_eps() -> None:
    committee = FakeCommittee(CLUSTER_R0, spread=0.01)
    switch = ConformalSwitch(committee, w_min=2)
    with pytest.raises(ValueError, match="at least one frame"):
        replay([], committee, switch)
    with pytest.raises(ValueError, match="eps_acc"):
        replay(_frames(), committee, switch, eps_acc=0.0)


def test_replay_without_updater_makes_no_finetunes() -> None:
    committee = FakeCommittee(CLUSTER_R0, bias_amplitude=0.05, spread=0.01)
    switch = ConformalSwitch(committee, eps_acc=1e-9, window=8, w_min=3)
    # eps_acc ~ 0: everything routes dft, but no updater -> nothing ingested.
    records, summary = replay(_frames(), committee, switch, updater=None)
    assert all(r.route == "dft" for r in records)
    assert summary.n_dft == N_FRAMES
    assert summary.n_finetunes == 0
    assert committee.finetune_calls == 0
    assert switch.window_size == 0  # no observations without an updater
    assert np.isnan(summary.alpha_hat)  # nothing accepted
