#!/usr/bin/env python3
"""Write the fixed initial structure for the mts_nve example (test
particles in the analytic harmonic potential — not a real material).

Two atoms, species Si, masses 28.085 amu, non-periodic:
  center a = (0.9, 0.9, 0.9) A, displacement d = (0.03, -0.02, 0.01) A;
  positions a - d and a + d.  Atom 2 velocity (0.001, -0.0015, 0.002)
  A/fs, atom 1 the opposite; momenta are set with the ASE unit
  conversion, never re-randomized.

Usage: python make_structure.py [output_dir]   (default: this directory)
"""
import sys
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import write as ase_write

MASS = 28.085
CENTER = np.array([0.9, 0.9, 0.9])
DISPLACEMENT = np.array([0.03, -0.02, 0.01])
VELOCITY = np.array([0.001, -0.0015, 0.002])  # A/fs, atom 2


def main() -> Path:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("Si2", positions=[CENTER - DISPLACEMENT,
                                    CENTER + DISPLACEMENT],
                  masses=[MASS, MASS], pbc=False)
    atoms.set_momenta(np.array([-VELOCITY, VELOCITY]) / units.fs * MASS)
    path = out / "structure.extxyz"
    ase_write(path, atoms)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
