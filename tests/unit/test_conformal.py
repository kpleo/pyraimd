"""ConformalSwitch: quantile edge cases, cold start, identical members, and
exchangeable-stream coverage.

The statistical tests use fixed seeds and generous margins: they guard the
machinery (quantile, bound, routing), not a particular random draw.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from conftest import CLUSTER_R0, FakeCommittee, FakeSurrogate

from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.switch import ConformalSwitch, conformal_quantile

ATOMS = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _prediction(spread: float) -> SurrogatePrediction:
    """Synthetic committee prediction with constant per-atom spread."""
    n = len(ATOMS)
    return SurrogatePrediction(
        energy=0.0,
        forces=np.zeros((n, 3)),
        stress=None,
        uncertainty=np.full(n, spread),
    )


def _assess(switch: ConformalSwitch, step: int, spread: float):
    return switch.assess(ATOMS, step, prediction=_prediction(spread))


# -- quantile order statistics ----------------------------------------------


def test_quantile_edge_cases() -> None:
    with pytest.raises(ValueError, match="empty"):
        conformal_quantile([], 0.05)  # n = 0: undefined, caller owns the policy

    assert conformal_quantile([7.0], 0.05) == 7.0  # n = 1: ceil(2*0.95)=2 > n, clamped

    # n = w_min - 1 = 15: ceil(16*0.95) = 16 > 15, clamped to the max.
    assert conformal_quantile(list(range(1, 16)), 0.05) == 15.0

    # In-range index: ceil(101*0.95) = 96 -> 96th order statistic of 1..100.
    assert conformal_quantile(list(range(1, 101)), 0.05) == 96.0


def test_quantile_rejects_bad_inputs() -> None:
    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            conformal_quantile([1.0], bad_alpha)
    with pytest.raises(ValueError, match="finite"):
        conformal_quantile([1.0, np.inf], 0.05)


def test_constructor_validation() -> None:
    surrogate = FakeCommittee(CLUSTER_R0)
    with pytest.raises(ValueError, match="alpha"):
        ConformalSwitch(surrogate, alpha=1.5)
    with pytest.raises(ValueError, match="eps_acc"):
        ConformalSwitch(surrogate, eps_acc=-0.5)
    # eps_acc = 0 is the refuse-everything sentinel (pure-engine control runs)
    switch0 = ConformalSwitch(surrogate, eps_acc=0.0, w_min=2)
    switch0.observe(0.01, 0.05)
    switch0.observe(0.01, 0.05)
    assert _assess(switch0, 2, spread=0.01).route == "dft"
    with pytest.raises(ValueError, match="window"):
        ConformalSwitch(surrogate, window=0)
    with pytest.raises(ValueError, match="w_min"):
        ConformalSwitch(surrogate, window=8, w_min=9)
    with pytest.raises(ValueError, match="delta"):
        ConformalSwitch(surrogate, delta=0.0)


# -- cold start ----------------------------------------------------------------


def test_cold_start_always_dft() -> None:
    switch = ConformalSwitch(FakeCommittee(CLUSTER_R0), w_min=16)
    decision = _assess(switch, 0, spread=0.01)
    assert decision.route == "dft"
    assert decision.score == float("inf")  # empty window: qhat = +inf
    assert "cold start" in decision.reason

    # One below w_min, even with a perfectly calibrated window (e = 0 -> B = 0).
    for _ in range(15):
        switch.observe(0.01, 0.0)
    assert switch.window_size == 15
    decision = _assess(switch, 15, spread=0.01)
    assert decision.route == "dft"
    assert "cold start" in decision.reason

    switch.observe(0.01, 0.0)  # |W| = 16 = w_min, qhat = 0 -> B = 0
    decision = _assess(switch, 16, spread=0.01)
    assert decision.route == "ml"
    assert decision.score == pytest.approx(0.0)


def test_identical_members_zero_spread_behavior() -> None:
    """σ ≡ 0 committee: the δ floor keeps B = q̂·δ — trust iff observed
    errors are also ~zero (spec §5: verify explicitly)."""
    # Real errors present: r = e/δ = 500, B = 500·δ = 0.5 > eps_acc -> never trusts.
    switch = ConformalSwitch(FakeCommittee(CLUSTER_R0, spread=0.0), w_min=16)
    for _ in range(16):
        switch.observe(0.0, 0.5)
    decision = _assess(switch, 16, spread=0.0)
    assert decision.route == "dft"
    assert decision.score == pytest.approx(0.5)  # qhat * delta == e

    # Errors also zero: q̂ = 0 -> B = 0 -> trusts (correctly).
    switch = ConformalSwitch(FakeCommittee(CLUSTER_R0, spread=0.0), w_min=16)
    for _ in range(16):
        switch.observe(0.0, 0.0)
    decision = _assess(switch, 16, spread=0.0)
    assert decision.route == "ml"
    assert decision.score == pytest.approx(0.0)


# -- decision content and validation --------------------------------------------


def test_decision_score_and_reason_carry_the_state() -> None:
    switch = ConformalSwitch(FakeCommittee(CLUSTER_R0), w_min=2, delta=1e-3)
    switch.observe(0.02, 0.04)
    switch.observe(0.02, 0.08)
    decision = _assess(switch, 2, spread=0.03)
    expected_qhat = 0.08 / (0.02 + 1e-3)  # n=2: ceil(3*0.95)=3 > 2 -> max r
    assert decision.score == pytest.approx(expected_qhat * (0.03 + 1e-3))
    assert "s=0.03000" in decision.reason
    assert "qhat=" in decision.reason
    assert "|W|=2" in decision.reason


def test_assess_requires_finite_spread() -> None:
    """A single frozen model reports NaN uncertainty — the switch must fail
    loudly instead of silently trusting it."""
    switch = ConformalSwitch(FakeSurrogate(ATOMS.get_positions()), w_min=1)
    switch.observe(0.01, 0.0)
    with pytest.raises(ValueError, match="finite"):
        switch.assess(ATOMS, 0)  # no prediction -> queries the NaN surrogate


def test_observe_validation() -> None:
    switch = ConformalSwitch(FakeCommittee(CLUSTER_R0))
    for bad_s, bad_e in [(-0.1, 0.0), (np.nan, 0.0), (0.0, -1.0), (0.0, np.inf)]:
        with pytest.raises(ValueError):
            switch.observe(bad_s, bad_e)
    assert switch.window_size == 0


def test_window_evicts_oldest() -> None:
    switch = ConformalSwitch(FakeCommittee(CLUSTER_R0), window=4, w_min=1)
    for _ in range(4):
        switch.observe(0.0, 1.0)  # r = 1000
    assert switch.qhat() == pytest.approx(1000.0)
    for _ in range(4):
        switch.observe(0.0, 0.001)  # r = 1; evicts the old pairs one by one
    assert switch.window_size == 4
    assert switch.qhat() == pytest.approx(1.0)  # old pairs fully evicted


# -- statistical coverage on an exchangeable stream ------------------------------


def test_bound_miscoverage_matches_alpha_on_exchangeable_stream() -> None:
    """Fully-labeled i.i.d. stream: the long-run rate of e > B(s) is ≈ α.

    This is the conformal marginal-coverage property of the bound itself,
    measured on the decisions the switch produces.
    """
    rng = np.random.default_rng(20250819)
    alpha, n_steps, warmup = 0.05, 4000, 64
    switch = ConformalSwitch(FakeCommittee(CLUSTER_R0), alpha=alpha, eps_acc=1e9)
    n_over = 0
    for step in range(n_steps):
        s = rng.uniform(0.5, 1.5)
        e = rng.exponential(1.0) * (s + switch.delta)  # r ~ Exp(1), i.i.d.
        decision = _assess(switch, step, spread=s)
        assert decision.route == "ml" or step < switch.w_min  # huge eps_acc
        if step >= warmup:
            n_over += int(e > decision.score)
        switch.observe(s, e)  # fully labeled stream
    rate = n_over / (n_steps - warmup)
    # binomial std ~ sqrt(0.05*0.95/3936) ~= 0.0035; margin is ~6 sigma.
    assert abs(rate - alpha) < 0.02


def test_accepted_miscoverage_within_tolerance_on_exchangeable_stream() -> None:
    """Realistic path — only "dft" steps update the window.  With the
    acceptance boundary near the error budget, the accepted-step violation
    rate α̂ must land near α (spec §5: generous, seed-fixed margins)."""
    rng = np.random.default_rng(20240209)
    switch = ConformalSwitch(
        FakeCommittee(CLUSTER_R0), alpha=0.05, eps_acc=0.1, window=64, w_min=16
    )
    n_accepted, n_violations = 0, 0
    for step in range(2000):
        s = rng.uniform(0.02, 0.04)
        e = rng.exponential(1.0) * (s + switch.delta)  # r ~ Exp(1)
        decision = _assess(switch, step, spread=s)
        if decision.route == "dft":
            switch.observe(s, e)
        else:
            n_accepted += 1
            n_violations += int(e > switch.eps_acc)
    assert n_accepted > 1000  # the switch actually accepts most steps
    alpha_hat = n_violations / n_accepted
    assert 0.005 < alpha_hat < 0.10


# -- streak inflation (streak-drift finding, audit 7617790) -------------------


def _primed_switch(streak_rho: float, **kw) -> ConformalSwitch:
    """Window primed with two (s=1, e=3) pairs -> qhat = 3/1.001 ~ 2.997."""
    sw = ConformalSwitch(
        FakeCommittee(CLUSTER_R0), alpha=0.05, eps_acc=1.0, w_min=2, window=8,
        streak_rho=streak_rho, **kw,
    )
    sw.observe(1.0, 3.0)
    sw.observe(1.0, 3.0)
    return sw


def test_streak_inflation_flips_at_predicted_k() -> None:
    sw = _primed_switch(0.05)
    routes = [_assess(sw, step, 0.3).route for step in range(4)]
    # k=0..2: 2.997*0.301*(1+0.05k) = 0.902/0.947/0.992 <= 1.0 -> ml
    # k=3: 1.037 > 1.0 -> dft, and a dft route also resets the streak.
    assert routes == ["ml", "ml", "ml", "dft"]


def test_streak_inflation_zero_recovers_plain_bound() -> None:
    sw = _primed_switch(0.0)
    routes = [_assess(sw, step, 0.3).route for step in range(6)]
    assert routes == ["ml"] * 6  # 2.997*0.301 = 0.902 < 1.0 forever


def test_dft_route_resets_streak_and_observe_is_replay_safe() -> None:
    sw = _primed_switch(0.05)
    for step in range(3):
        _assess(sw, step, 0.3)  # streak grows to 3
    assert _assess(sw, 3, 10.0).route == "dft"  # over budget -> resets streak
    d = _assess(sw, 4, 0.3)
    assert d.route == "ml"
    assert d.score == pytest.approx(sw.qhat() * 0.301)  # k back to 0
    # observe() must be replay-safe: it ingests the window pair without
    # clobbering the live streak (resume rebuilds the window from pairs).
    sw2 = _primed_switch(0.05)
    for step in range(2):
        _assess(sw2, step, 0.3)  # streak 2
    sw2.observe(1.0, 3.0)
    d2 = _assess(sw2, 2, 0.3)
    assert d2.route == "ml"
    assert d2.score == pytest.approx(sw2.qhat() * 0.301 * 1.10)  # k=2 survived
    assert "streak k=2" in d2.reason


def test_initial_streak_and_validation() -> None:
    sw = _primed_switch(0.05, initial_streak=3)
    assert _assess(sw, 0, 0.3).route == "dft"  # resumes at k=3: over budget
    with pytest.raises(ValueError, match="streak_rho"):
        ConformalSwitch(FakeCommittee(CLUSTER_R0), streak_rho=-0.1)
    with pytest.raises(ValueError, match="initial_streak"):
        ConformalSwitch(FakeCommittee(CLUSTER_R0), initial_streak=-1)
