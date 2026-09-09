"""Route forces using a fixed threshold on per-atom surrogate uncertainty.

Use the surrogate when its maximum per-atom spread is at or below the
threshold in eV/Å; otherwise request the reference. This heuristic has no
calibration window or quantile model. The spread threshold alone provides
no force-error guarantee.
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
