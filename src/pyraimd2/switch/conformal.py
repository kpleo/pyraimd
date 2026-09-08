"""Empirical committee-spread routing retained for existing workflows.

Recent labeled pairs calibrate the maximum atomic committee spread ``s``
against the realized maximum atomic force error ``e`` through ``e/(s+delta)``.
The rolling quantile is clipped to the sample maximum when its requested
rank exceeds the sample size. An optional streak factor increases the score
between reference labels.

This score is an empirical admission rule for an evolving trajectory. It
is not a split-conformal guarantee conditional on acceptance. Use the
independent-check protocol to measure accepted-force violations, or the
energetic loop to include directional response and signed residual work.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np
from ase import Atoms

from pyraimd2.surrogate.base import Surrogate, SurrogatePrediction
from pyraimd2.switch.base import Decision


def conformal_quantile(nonconformities: Sequence[float], alpha: float) -> float:
    """The ⌈(n+1)(1−α)⌉-th order statistic of the given values.

    Clamped to the n-th order statistic (the max) when the index exceeds n.
    Raises :class:`ValueError` on an empty input or invalid ``alpha`` —
    the empty-window policy (q̂ = +∞) belongs to the switch, not here.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    values = np.sort(np.asarray(list(nonconformities), dtype=float))
    if values.size == 0:
        raise ValueError("conformal quantile of an empty set is undefined")
    if not np.isfinite(values).all():
        raise ValueError("nonconformities must be finite")
    k = min(math.ceil((values.size + 1) * (1.0 - alpha)), values.size)
    return float(values[k - 1])


class ConformalSwitch:
    """Routes by the empirical calibrated score on committee spread.

    Holds a reference to the surrogate so the live loop can call the
    protocol method ``assess(atoms, step)``; callers that already evaluated
    the surrogate (the replay harness) pass ``prediction=`` to avoid a
    second committee evaluation.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        alpha: float = 0.05,
        eps_acc: float = 0.1,
        window: int = 64,
        w_min: int = 16,
        delta: float = 1e-3,
        streak_rho: float = 0.0,
        initial_streak: int = 0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if eps_acc < 0.0:
            raise ValueError(f"eps_acc must be >= 0, got {eps_acc}")
        # eps_acc = 0 is the deliberate refuse-everything sentinel (pure-engine
        # control runs): with qhat > 0 and s + delta > 0 the bound is strictly
        # positive, so every step routes to the engine.
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not 1 <= w_min <= window:
            raise ValueError(f"w_min must be in [1, window], got {w_min} (window {window})")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        if streak_rho < 0.0:
            raise ValueError(f"streak_rho must be >= 0, got {streak_rho}")
        if initial_streak < 0:
            raise ValueError(f"initial_streak must be >= 0, got {initial_streak}")
        self.surrogate = surrogate
        self.alpha = alpha
        self.eps_acc = eps_acc
        self.window = window
        self.w_min = w_min
        self.delta = delta
        self.streak_rho = streak_rho
        self._streak = initial_streak  # consecutive ml steps since last label
        self._pairs: deque[tuple[float, float]] = deque(maxlen=window)

    @property
    def window_size(self) -> int:
        return len(self._pairs)

    def qhat(self) -> float:
        """Current normalized-nonconformity quantile; +∞ on an empty window."""
        if not self._pairs:
            return float("inf")
        return conformal_quantile([e / (s + self.delta) for s, e in self._pairs], self.alpha)

    def bound(self, s: float) -> float:
        """B(s) = q̂ · (s + δ), the predicted max-force-error bound in eV/Å."""
        return self.qhat() * (s + self.delta)

    def observe(self, s: float, e: float) -> None:
        """Ingest one DFT-labeled observation; the oldest pair is evicted
        once the window is full.  Pure window ingestion — replay-safe: the
        acceptance streak is NOT touched here (it belongs to the live
        decision path, ``assess``), so a resume replaying stored (s, e)
        pairs does not clobber the trailing-streak count."""
        if not np.isfinite(s) or s < 0.0:
            raise ValueError(f"spread s must be finite and >= 0, got {s}")
        if not np.isfinite(e) or e < 0.0:
            raise ValueError(f"error e must be finite and >= 0, got {e}")
        self._pairs.append((float(s), float(e)))

    def assess(
        self, atoms: Atoms, step: int, prediction: SurrogatePrediction | None = None
    ) -> Decision:
        if prediction is None:
            prediction = self.surrogate.predict(atoms)
        s = float(np.max(prediction.uncertainty))
        if not np.isfinite(s):
            raise ValueError(
                "ConformalSwitch needs a finite per-atom spread from the surrogate "
                f"(got max uncertainty {s}); pair it with a committee, not a "
                "single frozen model"
            )
        n = len(self._pairs)
        qhat = self.qhat()
        k = self._streak
        bound = qhat * (s + self.delta) * (1.0 + self.streak_rho * k)
        route = "dft" if (n < self.w_min or bound > self.eps_acc) else "ml"
        streak_note = (
            f" streak k={k} rho={self.streak_rho}" if self.streak_rho > 0.0 else ""
        )
        why = (
            f"|W|={n} < w_min={self.w_min} (cold start)"
            if n < self.w_min
            else (
                f"B(s)={bound:.4f} > eps_acc={self.eps_acc} (over budget{streak_note})"
                if bound > self.eps_acc
                else f"B(s)={bound:.4f} <= eps_acc={self.eps_acc} (within budget{streak_note})"
            )
        )
        self._streak = 0 if route == "dft" else k + 1
        return Decision(
            route=route,
            score=bound,
            reason=(
                f"conformal: s={s:.5f} qhat={qhat:.4f} |W|={n} -> {why} "
                f"[alpha={self.alpha}, delta={self.delta}, step={step}]"
            ),
        )
