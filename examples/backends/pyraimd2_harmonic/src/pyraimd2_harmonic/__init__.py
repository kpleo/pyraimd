"""Example pyraimd2 backend plugin: analytic harmonic potentials.

A minimal third-party backend pair — a harmonic reference engine and a
harmonic surrogate with a deterministic force bias — registered through the
``pyraimd2.backends`` entry-point group. It exists to prove a new backend can
be added without modifying the pyraimd2 core, and to give configs a cheap
builtin-style backend for offline runs. Both backends declare their
capabilities and fingerprints explicitly (WP01 contract): finite molecule /
no stress, energy is force-consistent and conservative.
"""

from __future__ import annotations

import hashlib

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EnergyKind, EngineCapabilities, EngineResult
from pyraimd2.surrogate.base import SurrogateCapabilities, SurrogatePrediction


class HarmonicReference:
    """Analytic reference: E = 1/2 k |r - r0|^2, F = -k (r - r0)."""

    def __init__(self, k: float = 1.0, r0: float = 0.9) -> None:
        self.k = float(k)
        self.r0 = float(r0)

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
    """Same well plus a deterministic bias, so reference and surrogate differ."""

    def __init__(self, k: float = 1.0, r0: float = 0.9, bias: float = 0.05) -> None:
        self.k = float(k)
        self.r0 = float(r0)
        self.bias = float(bias)

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
            energy=0.5 * self.k * float((dr**2).sum()),
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
