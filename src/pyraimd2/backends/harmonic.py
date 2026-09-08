"""Builtin analytic harmonic backends for offline runs and examples.

A harmonic reference engine and a biased harmonic surrogate — cheap,
deterministic and dependency-free, so a full adaptive MD workflow can run
(and be tested) without any external quantum code or model weights.  The
surrogate's force bias is included in its reported energy, so both backends
are honestly force-consistent and conservative (WP01 contract).

These are the backends the ``harmonic`` init template uses.  They are toys:
no material realism, no stress.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EnergyKind, EngineCapabilities, EngineResult
from pyraimd2.surrogate.base import SurrogateCapabilities, SurrogatePrediction


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {bound} and finite, got {value!r}")
    return value


class HarmonicReference:
    """Analytic reference: E = 1/2 k |r - r0|^2, F = -k (r - r0)."""

    def __init__(self, k: float = 1.0, r0: float = 0.9) -> None:
        self.k = _positive_finite(k, "k")
        self.r0 = _positive_finite(r0, "r0", allow_zero=True)

    @property
    def name(self) -> str:
        return f"harmonic-reference-k{self.k}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=False,
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(f"{self.k}:{self.r0}".encode()).hexdigest()[:12]
        return f"harmonic-reference:{digest}"

    def compute(self, atoms: Atoms) -> EngineResult:
        dr = atoms.get_positions() - self.r0
        return EngineResult(
            energy=0.5 * self.k * float((dr**2).sum()),
            forces=-self.k * dr,
            stress=None,
            wall_time_s=0.0,
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
        )


class HarmonicSurrogate:
    """Same well plus a deterministic bias, so reference and surrogate differ.

    The bias is conservative and force-consistent:
    E = 1/2 k |dr|^2 + bias * sum(cos(dr)), F = -k dr + bias sin(dr).
    """

    def __init__(self, k: float = 1.0, r0: float = 0.9, bias: float = 0.05) -> None:
        self.k = _positive_finite(k, "k")
        self.r0 = _positive_finite(r0, "r0", allow_zero=True)
        self.bias = _positive_finite(bias, "bias", allow_zero=True)

    @property
    def capabilities(self) -> SurrogateCapabilities:
        return SurrogateCapabilities(
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=False,
            uncertainty_available=False,
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(
            f"{self.k}:{self.r0}:{self.bias}".encode()
        ).hexdigest()[:12]
        return f"harmonic-surrogate:{digest}"

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        dr = atoms.get_positions() - self.r0
        return SurrogatePrediction(
            energy=0.5 * self.k * float((dr**2).sum()) + self.bias * float(np.cos(dr).sum()),
            forces=-self.k * dr + self.bias * np.sin(dr),
            stress=None,
            uncertainty=np.full(len(atoms), np.nan),  # no honest spread
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
        )


def reference_factory(**kwargs) -> HarmonicReference:
    return HarmonicReference(**kwargs)


reference_factory.backend_kind = "engine"


def surrogate_factory(**kwargs) -> HarmonicSurrogate:
    return HarmonicSurrogate(**kwargs)


surrogate_factory.backend_kind = "surrogate"
