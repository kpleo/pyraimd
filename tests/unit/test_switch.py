"""ScheduledSwitch routing, including step 0 (and the pre-MD step -1)."""

from __future__ import annotations

import pytest
from ase import Atoms

from pyraimd2.switch import ScheduledSwitch

ATOMS = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def test_step_zero_is_dft() -> None:
    decision = ScheduledSwitch(5).assess(ATOMS, 0)
    assert decision.route == "dft"


def test_schedule_over_range() -> None:
    switch = ScheduledSwitch(5)
    routes = {step: switch.assess(ATOMS, step).route for step in range(-1, 13)}
    dft_steps = {step for step, route in routes.items() if route == "dft"}
    assert dft_steps == {0, 5, 10}


def test_period_one_is_always_dft() -> None:
    switch = ScheduledSwitch(1)
    assert all(switch.assess(ATOMS, step).route == "dft" for step in range(-1, 8))


def test_score_is_none_and_reasons_are_informative() -> None:
    switch = ScheduledSwitch(5)
    for step, route in [(0, "dft"), (1, "ml")]:
        decision = switch.assess(ATOMS, step)
        assert decision.route == route
        assert decision.score is None
        assert str(step) in decision.reason
        assert "5" in decision.reason


def test_invalid_period_raises() -> None:
    with pytest.raises(ValueError, match="period"):
        ScheduledSwitch(0)
