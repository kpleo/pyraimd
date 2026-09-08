"""Generate the 8-atom bulk Si cell for recipe A."""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from ase.io import write

# Conventional cubic diamond cell (8 atoms) with one atom displaced off the
# ideal site so the first forces are nonzero; momenta come from the config's
# temperature/velocity seed.
atoms = bulk("Si", "diamond", a=5.431, cubic=True)
atoms.positions[0] += np.array([0.05, -0.03, 0.02])
write("structure.extxyz", atoms)
print(f"wrote structure.extxyz: {atoms.get_chemical_formula()}, "
      f"a = {atoms.cell[0, 0]:.4f} A, pbc {list(atoms.pbc)}")
