"""Use a configured ASE interatomic-potential calculator as a surrogate."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.surrogate.base import SurrogatePrediction


class AseSurrogate:
    """Model-independent adapter; a single calculator supplies no committee spread."""

    def __init__(self, calculator: Calculator, *, force_consistent: bool = False,
                 include_stress: bool = False) -> None:
        self._engine = AseEngine(calculator, force_consistent=force_consistent,
                                 include_stress=include_stress)

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        result = self._engine.compute(atoms)
        return SurrogatePrediction(result.energy, result.forces, result.stress,
                                   np.full(len(atoms), np.nan))
