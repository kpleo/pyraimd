"""Generate the QE ecut-convergence campaign inputs for the tungsten bulk box.

Configuration: bcc W, 4x4x4 unit cells = 128 atoms, Gamma only,
PBE (no D3 needed for a metal), mv smearing native. Electronic settings: nbnd headroom, fully converged
Davidson, local-TF mixing. W kjpaw pseudo: 14 valence electrons/atom, suggested
cutoffs 50.4/475.2 Ry (wfc/rho) -> series brackets the suggestion at rho=10x.

Usage:  uv run python hpc/neimeng/scripts/make_w_ecut_series.py
"""

from __future__ import annotations

import os

from pathlib import Path

from ase.build import bulk

from pyraimd2.engines.qe_engine import QeConfig, write_qe_input

HERE = Path(__file__).parents[1]
OUT = HERE / "inputs" / "generated"
A0 = 3.165  # bcc W lattice constant, Angstrom
ECUTS = [40.0, 50.0, 60.0, 70.0]
NBND = 896 + 50  # 128 atoms x 14 valence / 2 = 896 occupied, +50 empty (§3.1)
PSEUDO_DIR = os.environ.get("PYRAMID_PSEUDO_DIR", "pseudopotentials")


def main() -> None:
    atoms = bulk("W", "bcc", a=A0, cubic=True).repeat((4, 4, 4))
    assert len(atoms) == 128
    for ecut in ECUTS:
        cfg = QeConfig(
            pseudo_dir=PSEUDO_DIR,
            ecutwfc=ecut,
            ecutrho=10.0 * ecut,  # W kjpaw suggested rho ratio is 9.4x
            kpts=None,  # Gamma only, matching the production spike box
            nbnd=NBND,
            metallic=True,
            degauss=0.01,
            mixing_ndim=12,
            diago_david_ndim=8,
            diago_full_acc=True,
            conv_thr=1e-6,  # extensive norm, same rationale as §3.1
        )
        target = OUT / f"w_ecut_{int(ecut)}"
        target.mkdir(parents=True, exist_ok=True)
        write_qe_input(target / "pw.in", atoms, cfg)
        print(f"wrote {target / 'pw.in'} (ecutwfc={ecut}, ecutrho={10 * ecut:.0f})")
    print(f"box: {len(atoms)} atoms, {atoms.get_chemical_formula()}, a={A0}")


if __name__ == "__main__":
    main()
