"""Scheduled switch: the M1 stand-in for the calibrated switch (M2).

Routes to the engine on a fixed period so the closed loop accumulates DFT
labels at a controlled rate.  No statistics here — committee + conformal
arrive in M2 (design doc §9).
"""

from __future__ import annotations

from ase import Atoms

from pyraimd2.switch.base import Decision


class ScheduledSwitch:
    """Route "dft" when ``step % period == 0`` (step 0 included), else "ml"."""

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period

    def assess(self, atoms: Atoms, step: int) -> Decision:
        if step % self.period == 0:
            return Decision(
                route="dft",
                score=None,
                reason=f"scheduled DFT checkpoint: step {step} % period {self.period} == 0",
            )
        return Decision(
            route="ml",
            score=None,
            reason=(
                f"scheduled ML step: step {step} % period {self.period} "
                f"= {step % self.period} != 0"
            ),
        )
