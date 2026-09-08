"""Generate the Al(111) slab with one adatom for recipe B."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.io import write

# 2x2 Al(111), 4 layers (16 atoms) plus a single adatom on top = 17 atoms;
# the bottom two layers (indices 0-7) are fixed via the run configs'
# [constraints] section. Vacuum padding is added along z.
slab = fcc111("Al", size=(2, 2, 4), a=4.0495, vacuum=0.0)
adatom_z = float(slab.positions[:, 2].max()) + 2.10
adatom = Atoms("Al", positions=[[slab.cell[0, 0] / 2,
                                 slab.cell[1, 1] / 3, adatom_z]])
atoms = slab + adatom
atoms.center(vacuum=12.0, axis=2)
# small deterministic perturbation so initial forces are nonzero
rng = np.random.default_rng(11)
atoms.positions[8:] += rng.uniform(-0.02, 0.02,
                                   size=atoms.positions[8:].shape)
write("structure.extxyz", atoms)
print(f"wrote structure.extxyz: {atoms.get_chemical_formula()}, "
      f"{len(atoms)} atoms, cell z = {atoms.cell[2, 2]:.2f} A, "
      f"pbc {list(atoms.pbc)}")
