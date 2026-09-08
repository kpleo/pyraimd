"""Threshold switch: the uncalibrated incumbent baseline.

Trust iff the committee spread is below a fixed threshold — the standard
concurrent-learning heuristic (DP-GEN-style model deviation), with no
window, no quantile, no contract. Used as the tuned-threshold baseline in
the coverage experiments (red-team review 2026-08-27: the scheduled
fallback was a strawman; the fair incumbent is a threshold tuned to the
same label budget).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from pyraimd2.surrogate.base import Surrogate, SurrogatePrediction
from pyraimd2.switch.base import Decision


class ThresholdSwitch:
    """Route "ml" iff max per-atom spread <= threshold (eV/Å)."""

    def __init__(self, surrogate: Surrogate, threshold: float) -> None:
        if threshold <= 0.0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self.surrogate = surrogate
        self.threshold = threshold

    def assess(
        self, atoms: Atoms, step: int, prediction: SurrogatePrediction | None = None
    ) -> Decision:
        if prediction is None:
            prediction = self.surrogate.predict(atoms)
        s = float(np.max(prediction.uncertainty))
        route = "ml" if s <= self.threshold else "dft"
        return Decision(
            route=route,
            score=s,
            reason=(
                f"threshold: s={s:.5f} vs tau={self.threshold} -> {route} "
                f"[step={step}]"
            ),
        )
