"""Use a configured ASE interatomic-potential calculator as a surrogate."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.surrogate.base import SurrogateCapabilities, SurrogatePrediction


class AseSurrogate:
    """Model-independent adapter; a single calculator supplies no committee spread.

    The fingerprint is the wrapped engine's (parameters and model-file
    content, never just ``calculator.name``); it is None when the calculator
    state cannot be identified, unless an explicit ``identity`` is given.
    """

    def __init__(self, calculator: Calculator, *, force_consistent: bool = False,
                 include_stress: bool = False,
                 identity: str | None = None) -> None:
        self._engine = AseEngine(calculator, force_consistent=force_consistent,
                                 include_stress=include_stress, identity=identity)

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
    def fingerprint(self) -> str | None:
        engine_fingerprint = self._engine.fingerprint
        if engine_fingerprint is None:
            return None  # unknown identity, honestly undeclared
        return f"ase-surrogate:{engine_fingerprint}"

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        result = self._engine.compute(atoms)
        return SurrogatePrediction(result.energy, result.forces, result.stress,
                                   np.full(len(atoms), np.nan),
                                   energy_kind=result.energy_kind,
                                   force_consistent=result.force_consistent)
