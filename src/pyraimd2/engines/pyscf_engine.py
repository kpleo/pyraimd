"""In-process PySCF engine for finite molecules (design doc §8: local dev DFT).

Restricted closed-shell KS-DFT (RKS) only.  Unit conversions happen exactly
here, at the engine boundary (design doc §3, rule 6): PySCF works in
Hartree/Bohr, the rest of the package works in eV/Å via ``ase.units``.
"""

from __future__ import annotations

import time

import numpy as np
from ase import Atoms, units

from pyraimd2.engines.base import EngineError, EngineResult


class PyscfEngine:
    """PySCF RKS single-point engine (molecular, no PBC, no subprocess).

    Raises:
        EngineError: on SCF non-convergence, periodic input, or an
            odd-electron system (RKS closed-shell only).
    """

    def __init__(
        self,
        functional: str = "pbe",
        basis: str = "def2-svp",
        conv_tol: float = 1e-9,
    ) -> None:
        self.functional = functional
        self.basis = basis
        self.conv_tol = conv_tol

    @property
    def name(self) -> str:
        return f"pyscf-rks-{self.functional}-{self.basis}"

    def compute(self, atoms: Atoms) -> EngineResult:
        from pyscf import dft, gto  # local import: keep module import cheap

        if np.any(atoms.pbc):
            raise EngineError("PyscfEngine handles only finite molecules (pbc must be False)")
        charge = round(float(atoms.get_initial_charges().sum()))
        n_electrons = int(atoms.get_atomic_numbers().sum()) - charge
        if n_electrons <= 0:
            raise EngineError(f"non-positive electron count ({n_electrons})")
        if n_electrons % 2 != 0:
            raise EngineError(f"odd electron count ({n_electrons}): RKS closed-shell only")

        symbols = atoms.get_chemical_symbols()
        pos_bohr = atoms.get_positions() / units.Bohr
        atom_spec = [
            f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, pos_bohr)
        ]

        t0 = time.perf_counter()
        mol = gto.M(atom=atom_spec, unit="bohr", basis=self.basis, charge=charge, spin=0, verbose=0)
        mf = dft.RKS(mol)
        mf.xc = self.functional
        mf.conv_tol = self.conv_tol
        energy_ha = mf.kernel()
        if not mf.converged:
            raise EngineError(
                f"PySCF SCF did not converge (conv_tol={self.conv_tol}, "
                f"functional={self.functional}, basis={self.basis})"
            )
        grad_ha_bohr = np.asarray(mf.nuc_grad_method().kernel())
        wall_time_s = time.perf_counter() - t0

        return EngineResult(
            energy=float(energy_ha) * units.Hartree,
            forces=-grad_ha_bohr * (units.Hartree / units.Bohr),
            stress=None,  # finite molecule: no cell, no virial (design doc §3 rule 7 note)
            wall_time_s=wall_time_s,
        )
