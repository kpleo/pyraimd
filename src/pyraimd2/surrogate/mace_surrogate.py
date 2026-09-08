"""MACE-MP-0 surrogate: frozen foundation model, CPU-first.

A single frozen model has no committee and therefore no honest uncertainty
estimate: ``uncertainty`` is all-NaN by design.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from pyraimd2.surrogate.base import SurrogatePrediction


class MaceSurrogate:
    """Wraps ``mace.calculators.mace_mp``; the calculator is built lazily once
    and reused across predictions (model loading is the expensive part)."""

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        default_dtype: str = "float64",
    ) -> None:
        self.model = model
        self.device = device
        self.default_dtype = default_dtype
        self._calc: Any = None

    def _get_calc(self) -> Any:
        if self._calc is None:
            from mace.calculators import mace_mp  # local import: torch is heavy

            self._calc = mace_mp(
                model=self.model,
                device=self.device,
                default_dtype=self.default_dtype,
            )
        return self._calc

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        calc = self._get_calc()
        work = atoms.copy()
        work.calc = calc
        # Raw physical values, like AseEngine: constraints are applied once
        # by the workflow, never by both sides of the comparison.
        energy = float(work.get_potential_energy(apply_constraint=False))
        forces = np.asarray(work.get_forces(apply_constraint=False), dtype=float)
        stress = (np.asarray(work.get_stress(apply_constraint=False), dtype=float)
                  if np.any(atoms.pbc) else None)
        return SurrogatePrediction(
            energy=energy,
            forces=forces,
            stress=stress,
            uncertainty=np.full(len(atoms), np.nan),
        )
