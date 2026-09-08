"""Use an ASE calculator as a reference engine."""

from __future__ import annotations

import time

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from pyraimd2.engines.base import EngineError, EngineResult


class AseEngine:
    """Adapt an externally configured ASE calculator to the Engine protocol.

    Set ``force_consistent=True`` for calculators whose forces differentiate
    a free energy instead of their default reported energy. Stress is opt-in.
    Each adapter owns its calculator; use separate instances for separate runs.

    The adapter always reports the raw physical energy/forces/stress: any
    constraint adjustment (energy terms included) is left to the workflow,
    which applies constraints exactly once. Mixing a constraint-adjusted
    energy with unprojected forces would pair values from different surfaces.
    """

    def __init__(self, calculator: Calculator, *, force_consistent: bool = False,
                 include_stress: bool = False) -> None:
        self.calculator = calculator
        self.force_consistent = force_consistent
        self.include_stress = include_stress

    @property
    def name(self) -> str:
        return f"ase-{self.calculator.name}"

    def compute(self, atoms: Atoms) -> EngineResult:
        work = atoms.copy()
        work.calc = self.calculator
        start = time.perf_counter()
        try:
            energy = float(work.get_potential_energy(force_consistent=self.force_consistent,
                                                     apply_constraint=False))
            forces = np.array(work.get_forces(apply_constraint=False), dtype=float, copy=True)
            stress = (np.array(work.get_stress(apply_constraint=False), dtype=float, copy=True)
                      if self.include_stress else None)
            if (not np.isfinite(energy) or forces.shape != (len(work), 3)
                    or not np.isfinite(forces).all()):
                raise ValueError("Nonfinite energy or invalid forces")
            if stress is not None and (stress.shape != (6,) or not np.isfinite(stress).all()):
                raise ValueError("Invalid stress")
        except Exception as error:
            self.calculator.reset()
            raise EngineError(f"ASE reference evaluation failed: {error}") from error
        return EngineResult(energy, forces, stress, time.perf_counter() - start)
