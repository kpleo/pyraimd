"""ToyConformalSwitch (experiments/toy/loop.py) is a SEMANTIC MIRROR of
ConformalSwitch (src/pyraimd2/switch/conformal.py): this test drives both
with one scripted (s, e) stream through the live-loop update rule
(observe on "dft" routes only) and asserts byte-identical decisions —
route, score, reason string, streak counter, and window size at every
step.  The stream is hand-built (no RNG) to exercise cold start, warm
accepts, over-budget rejects, streak-note formatting, streaks long enough
for the rho-inflation flip, and window eviction (>64 observations).

Also gates the toy engine: one RK4 macro-step against a dt-converged
reference (< 1e-6, the spec's engine check).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from conftest import CLUSTER_R0, FakeCommittee

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from experiments.toy.loop import ToyConformalSwitch
from experiments.toy.lorenz import RHO_REGIME_A, LorenzEngine, verify_rk4
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.switch import ConformalSwitch

ATOMS = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _prediction(spread: float) -> SurrogatePrediction:
    """Constant-committee prediction, as in test_conformal.py."""
    n = len(ATOMS)
    return SurrogatePrediction(
        energy=0.0,
        forces=np.zeros((n, 3)),
        stress=None,
        uncertainty=np.full(n, spread),
    )


def _scripted_stream() -> list[tuple[float, float]]:
    """170 steps: calibrated margin (alternating under rho>0), a long
    high-error block (permanent rejects, >64 observations total), then a
    quiet block (long streaks + eviction)."""
    seq = []
    for step in range(170):
        if step < 30:
            seq.append((0.020, 0.048))  # r = 2.286: B = 0.048 vs eps 0.05
        elif step < 90:
            seq.append((0.050, 0.300))  # r = 5.88: over budget at any k
        else:
            seq.append((0.005, 0.001))  # r = 0.167: quiet, streaks grow
    return seq


@pytest.mark.parametrize("streak_rho", [0.0, 0.05])
def test_toy_switch_mirrors_conformal_switch(streak_rho: float) -> None:
    kwargs = {"alpha": 0.05, "eps_acc": 0.05, "window": 64, "w_min": 16,
              "delta": 1e-3, "streak_rho": streak_rho}
    md = ConformalSwitch(FakeCommittee(CLUSTER_R0), **kwargs)
    toy = ToyConformalSwitch(**kwargs)

    routes: list[str] = []
    max_streak = 0
    for step, (s, e) in enumerate(_scripted_stream()):
        d_md = md.assess(ATOMS, step, prediction=_prediction(s))
        d_toy = toy.assess(s, step)
        assert d_toy.route == d_md.route, step
        assert d_toy.score == d_md.score, step  # identical arithmetic, bit-exact
        assert d_toy.reason == d_md.reason, step
        assert toy.streak == md._streak, step
        assert toy.window_size == md.window_size, step
        if d_md.route == "dft":
            md.observe(s, e)
            toy.observe(s, e)
        routes.append(d_md.route)
        max_streak = max(max_streak, md._streak)

    # The scripted stream must actually exercise the machinery, or the
    # byte-identity assertions above are vacuous.
    assert routes[:16] == ["dft"] * 16  # cold start
    assert "ml" in routes and routes.count("dft") > 64  # both routes + eviction
    assert max_streak >= 5  # real streaks (streak-note branch under rho>0)


def test_toy_switch_reason_carries_streak_note_like_md() -> None:
    """The rho>0 reason format, one explicit example of the mirrored string."""
    kwargs = {"alpha": 0.05, "eps_acc": 0.05, "window": 64, "w_min": 16,
              "delta": 1e-3, "streak_rho": 0.05}
    md = ConformalSwitch(FakeCommittee(CLUSTER_R0), **kwargs)
    toy = ToyConformalSwitch(**kwargs)
    for _ in range(16):
        md.observe(0.02, 0.048)
        toy.observe(0.02, 0.048)
    d_md = md.assess(ATOMS, 16, prediction=_prediction(0.02))  # k=0 -> ml
    d_toy = toy.assess(0.02, 16)
    assert d_toy.reason == d_md.reason
    d_md = md.assess(ATOMS, 17, prediction=_prediction(0.02))  # k=1 -> over budget
    d_toy = toy.assess(0.02, 17)
    assert d_md.route == "dft"
    assert "streak k=1 rho=0.05" in d_md.reason
    assert d_toy.reason == d_md.reason


def test_toy_switch_validation_and_empty_window() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ToyConformalSwitch(alpha=1.5)
    with pytest.raises(ValueError, match="eps_acc"):
        ToyConformalSwitch(eps_acc=0.0)
    with pytest.raises(ValueError, match="w_min"):
        ToyConformalSwitch(window=8, w_min=9)
    with pytest.raises(ValueError, match="streak_rho"):
        ToyConformalSwitch(streak_rho=-0.1)
    toy = ToyConformalSwitch()
    assert toy.qhat() == float("inf")  # empty-window policy, as in MD
    assert toy.assess(0.01, 0).route == "dft"  # cold start
    with pytest.raises(ValueError, match="finite"):
        toy.assess(float("nan"), 1)


def test_rk4_macro_step_converged() -> None:
    """Engine gate (spec: max|error| vs dt-converged reference < 1e-6 over
    one step of the RK4 integrator at dt=0.005, i.e. one substep), checked
    on attractor points of the deployment regime."""
    states, _ = LorenzEngine(RHO_REGIME_A).attractor_samples(8, seed=123, stride=50)
    assert verify_rk4(states, rho=RHO_REGIME_A, n_sub=1) < 1e-6
