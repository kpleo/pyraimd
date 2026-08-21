"""Minimal closed-loop environment verification for PYRAIMD-2 (M0).

Chain under test:  torch -> mace-torch (MACE-MP-0 foundation model) -> ASE MD
                   -> PySCF (DFT reference)

Runs a single-point MACE evaluation on a distorted H2O, a short NVE trajectory,
and a PySCF PBE reference. The DFT check is a self-consistency test (analytic
nuclear gradient vs central finite difference of the SCF energy): it verifies
the units and sign conventions at the quantum-engine boundary — exactly where
PYRAIMD v1 had unit/refactor bugs — without claiming cross-theory accuracy.

Usage:  uv run python examples/minimal_loop.py
"""

from __future__ import annotations

import sys

import numpy as np
from ase import units
from ase.build import molecule
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

MD_STEPS = 30
MD_TIMESTEP_FS = 0.5
DRIFT_TOL_MEV_PER_ATOM = 20.0  # over the full 15 fs NVE run
FD_STEP_BOHR = 5e-4
FD_TOL_EV_PER_A = 1e-3  # analytic-vs-FD force agreement gate


def build_h2o():
    """H2O slightly distorted from equilibrium so forces are clearly nonzero."""
    atoms = molecule("H2O")
    atoms.positions[1, 0] += 0.10  # stretch one O-H bond
    return atoms


def mace_single_point(atoms):
    from mace.calculators import mace_mp

    calc = mace_mp(model="small", default_dtype="float64", device="cpu")
    atoms = atoms.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return energy, forces


def nve_drift_mev_per_atom(atoms) -> float:
    from mace.calculators import mace_mp

    atoms = atoms.copy()
    atoms.calc = mace_mp(model="small", default_dtype="float64", device="cpu")
    thermalize_momenta(atoms, temperature_K=300)
    dyn = VelocityVerlet(atoms, MD_TIMESTEP_FS * units.fs)
    e0 = atoms.get_total_energy()
    dyn.run(MD_STEPS)
    e1 = atoms.get_total_energy()
    return 1e3 * abs(e1 - e0) / len(atoms)


def _pyscf_energy_ha(symbols, positions_bohr) -> float:
    from pyscf import dft, gto

    atom_spec = [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, positions_bohr)]
    mol = gto.M(atom=atom_spec, unit="bohr", basis="def2-svp", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.conv_tol = 1e-10  # tight, so the finite difference is clean
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge")
    return float(energy)


def pyscf_reference(atoms) -> tuple[float, np.ndarray, float]:
    """PBE/def2-SVP energy and analytic forces, plus max |analytic - FD| force deviation."""
    from pyscf import dft, gto

    symbols = atoms.get_chemical_symbols()
    pos_bohr = atoms.positions / units.Bohr
    atom_spec = [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, pos_bohr)]
    mol = gto.M(atom=atom_spec, unit="bohr", basis="def2-svp", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.conv_tol = 1e-10
    energy_ha = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge")
    grad_ha_bohr = np.asarray(mf.nuc_grad_method().kernel())
    forces_ev_a = -grad_ha_bohr * (units.Hartree / units.Bohr)

    max_dev_ha_bohr = 0.0
    for i in range(len(atoms)):
        for d in range(3):
            dp = pos_bohr.copy(); dp[i, d] += FD_STEP_BOHR
            dm = pos_bohr.copy(); dm[i, d] -= FD_STEP_BOHR
            fd = (_pyscf_energy_ha(symbols, dp) - _pyscf_energy_ha(symbols, dm)) / (2 * FD_STEP_BOHR)
            max_dev_ha_bohr = max(max_dev_ha_bohr, abs(fd - grad_ha_bohr[i, d]))
    return energy_ha, forces_ev_a, max_dev_ha_bohr * (units.Hartree / units.Bohr)


def main() -> int:
    atoms = build_h2o()
    checks: list[tuple[str, bool, str]] = []

    e_mace, f_mace = mace_single_point(atoms)
    fmax = float(np.abs(f_mace).max())
    ok = np.isfinite(e_mace) and np.isfinite(f_mace).all() and fmax > 0.01
    checks.append(("MACE-MP-0 single point finite, forces nonzero", ok,
                   f"E = {e_mace:.6f} eV, max|F| = {fmax:.4f} eV/A"))

    drift = nve_drift_mev_per_atom(atoms)
    checks.append((f"NVE {MD_STEPS} steps x {MD_TIMESTEP_FS} fs, drift bounded",
                   drift < DRIFT_TOL_MEV_PER_ATOM,
                   f"drift = {drift:.2f} meV/atom over {MD_STEPS * MD_TIMESTEP_FS:.0f} fs"))

    e_ref, f_ref, fd_dev = pyscf_reference(atoms)
    cos = float(np.dot(f_mace.ravel(), f_ref.ravel())
                / (np.linalg.norm(f_mace) * np.linalg.norm(f_ref)))
    checks.append(("PySCF PBE/def2-SVP: analytic forces match finite difference",
                   np.isfinite(e_ref) and fd_dev < FD_TOL_EV_PER_A,
                   (f"E_PBE = {e_ref:.8f} Ha, max|F_analytic - F_FD| = {fd_dev:.2e} eV/A; "
                    f"informational: cos(F_MACE, F_PBE) = {cos:.3f} (OOD molecule, not gated)")))

    print("\n=== PYRAIMD-2 minimal loop report ===")
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    n_fail = sum(not ok for _, ok, _ in checks)
    print(f"=== {len(checks) - n_fail}/{len(checks)} checks passed ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
