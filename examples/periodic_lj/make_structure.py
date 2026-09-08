"""Generate the fcc LJ structure for the periodic CI recipe."""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from ase.io import write

# 2x2x2 conventional fcc cells (32 atoms), a few percent off the LJ minimum
# so the first forces are nonzero; a small deterministic thermal velocity
# comes from the run config (dynamics.temperature_K / velocity_seed).
atoms = bulk("Ar", "fcc", a=1.70, cubic=True) * (2, 2, 2)
rng = np.random.default_rng(7)
atoms.positions += rng.uniform(-0.01, 0.01, size=atoms.positions.shape)
write("structure.extxyz", atoms)
print(f"wrote structure.extxyz: {atoms.get_chemical_formula()}, "
      f"cell {atoms.cell[0, 0]:.3f} A, pbc {list(atoms.pbc)}")
