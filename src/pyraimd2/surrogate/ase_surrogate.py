"""Use a configured ASE interatomic-potential calculator as a surrogate."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.surrogate.base import SurrogateCapabilities, SurrogatePrediction


class AseSurrogate:
    """Model-independent adapter; a single calculator supplies no committee spread."""

    def __init__(self, calculator: Calculator, *, force_consistent: bool = False,
                 include_stress: bool = False) -> None:
        self._engine = AseEngine(calculator, force_consistent=force_consistent,
                                 include_stress=include_stress)

    @property
    def capabilities(self) -> SurrogateCapabilities:
        engine_caps = self._engine.capabilities
        return SurrogateCapabilities(
            energy_kind=engine_caps.energy_kind,
            force_consistent=engine_caps.force_consistent,
            forces_conservative=engine_caps.forces_conservative,
            stress_available=engine_caps.stress_available,
            uncertainty_available=False,  # single calculator: no honest spread
        )

    @property
    def fingerprint(self) -> str:
        return f"ase-surrogate:{self._engine.fingerprint}"

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        result = self._engine.compute(atoms)
        return SurrogatePrediction(result.energy, result.forces, result.stress,
                                   np.full(len(atoms), np.nan),
                                   energy_kind=result.energy_kind,
                                   force_consistent=result.force_consistent)
