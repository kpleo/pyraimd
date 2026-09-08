"""Example pyraimd2 backend plugin: periodic Lennard-Jones backends.

A periodic classical reference and surrogate for the no-external-software CI
recipe (``examples/periodic_lj``).  Both wrap ASE's ``LennardJones``
calculator; the surrogate is a few percent softer in epsilon, giving a
deterministic, tunable shadow error for adaptive runs.  Capabilities and
fingerprints follow the WP01 contract: energy is force-consistent and
conservative, stress is available for periodic systems, and the surrogate
honestly reports no spread (NaN uncertainty).
"""

from __future__ import annotations

import hashlib

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.engines.base import EnergyKind, EngineCapabilities, EngineResult
from pyraimd2.surrogate.base import SurrogateCapabilities, SurrogatePrediction


class LjReference:
    """Periodic Lennard-Jones reference engine (ASE units)."""

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0,
                 rc: float = 3.0) -> None:
        self.epsilon, self.sigma, self.rc = (float(epsilon), float(sigma),
                                             float(rc))
        self._engine = AseEngine(
            LennardJones(epsilon=self.epsilon, sigma=self.sigma, rc=self.rc),
            include_stress=True)

    @property
    def name(self) -> str:
        return "lj-reference"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=True,
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(
            f"{self.epsilon}:{self.sigma}:{self.rc}".encode()).hexdigest()[:12]
        return f"lj-reference:{digest}"

    def compute(self, atoms: Atoms) -> EngineResult:
        return self._engine.compute(atoms)


class LjSurrogate:
    """Same LJ form with a softer epsilon; uncertainty is honestly NaN."""

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0,
                 rc: float = 3.0, softening: float = 0.95) -> None:
        self._reference = LjReference(epsilon=epsilon * float(softening),
                                      sigma=sigma, rc=rc)
        self.softening = float(softening)

    @property
    def capabilities(self) -> SurrogateCapabilities:
        return SurrogateCapabilities(
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=True,
            uncertainty_available=False,
        )

    @property
    def fingerprint(self) -> str:
        inner = self._reference
        digest = hashlib.sha256(
            f"{inner.epsilon}:{inner.sigma}:{inner.rc}".encode()
        ).hexdigest()[:12]
        return f"lj-surrogate:{digest}"

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        result = self._reference.compute(atoms)
        return SurrogatePrediction(
            energy=result.energy,
            forces=result.forces,
            stress=result.stress,
            uncertainty=np.full(len(atoms), np.nan),
            energy_kind=result.energy_kind,
            force_consistent=result.force_consistent,
        )


def reference_factory(**kwargs) -> LjReference:
    return LjReference(**kwargs)


reference_factory.backend_kind = "engine"


def surrogate_factory(**kwargs) -> LjSurrogate:
    return LjSurrogate(**kwargs)


surrogate_factory.backend_kind = "surrogate"
