"""Generate the QE ecut-convergence campaign inputs for the bulk electrolyte box.

Reads the Packmol-packed box (hpc/neimeng/inputs/bulk_lp30_288.xyz), assigns the
15 A cubic periodic cell, and writes one pw.x input per ecutwfc into
hpc/neimeng/inputs/generated/ecut_<Ry>/pw.in — using pyraimd2's own
``write_qe_input`` (dogfooding the production code path).

Usage:  uv run python hpc/neimeng/scripts/make_ecut_series.py [--restart]

``--restart`` regenerates the inputs for resuming from the saved charge
densities of a previous (unconverged) run: ``startingpot='file'``, gentler
mixing, longer SCF budget. Campaign history: the 2026-08-19 from-scratch run
plateaued at ~1e-2 Ry density residual after 200 iterations (energy stable to
1e-3 Ry) — the restart phase polishes convergence off.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ase.io import read

from pyraimd2.engines.qe_engine import QeConfig, write_qe_input

HERE = Path(__file__).parents[1]
PACKED = HERE / "inputs" / "bulk_lp30_288.xyz"
OUT = HERE / "inputs" / "generated"
BOX_A = 15.0  # matches packmol_bulk.inp
ECUTS = [40.0, 50.0, 60.0, 70.0]
# 972 electrons -> 486 doubly-occupied KS states; running with zero empty bands
# leaves the HOMO unconstrained (Davidson subspace too tight) and the density
# residual spikes to O(1) Ry every few iterations — observed in arrays 7614891
# and 7615036. +50 empty bands is the standard cure.
NBND = 486 + 50
# Array 7615036 also showed "c_bands: N eigenvalues not converged" — Davidson
# with ndim=4 cannot converge ~500 bands tightly. ndim=8 + diago_full_acc
# converges every band every SCF step (costs more per iteration, buys the
# density-residual descent that mixing alone could not).
#
# conv_thr=1e-6 (not the 1e-8 engine default): the QE accuracy norm is
# extensive, so 1e-6 Ry on 288 atoms is per-atom tighter than 1e-8 Ry on the
# 2-atom Si validation cell — force-worthy, and reachable.
#
# mv smearing (degauss=0.01 Ry ~ 0.14 eV): with exact diagonalization the
# residual still spiked at beta=0.35 and crawled at beta=0.15 (probes 7615060,
# 7615064) — the occupation response at the Fermi level is the stiff mode.
# Cold smearing damps it; energy bias at this width is sub-meV/atom.
CONV_THR = 1e-6
PSEUDO_DIR = "/data/home/df103967/df103967/cloud_projects/pyraimd2/inputs"


def make_cfg(ecut: float, restart: bool) -> QeConfig:
    return QeConfig(
        pseudo_dir=PSEUDO_DIR,
        ecutwfc=ecut,
        ecutrho=8.0 * ecut,  # kjpaw PAW density cutoff
        kpts=None,  # Gamma only: 15 A liquid box
        nbnd=NBND,
        conv_thr=CONV_THR,
        metallic=True,
        degauss=0.01,
        mixing_beta=0.3,
        mixing_ndim=12 if restart else 8,
        diago_david_ndim=8,
        diago_full_acc=True,
        electron_maxstep=400 if restart else 200,
        startpot_file=restart,
    )


def main() -> None:
    restart = "--restart" in sys.argv[1:]
    atoms = read(PACKED)
    atoms.set_cell([BOX_A, BOX_A, BOX_A])
    atoms.set_pbc(True)
    atoms.center()
    for ecut in ECUTS:
        cfg = make_cfg(ecut, restart)
        target = OUT / f"ecut_{int(ecut)}"
        target.mkdir(parents=True, exist_ok=True)
        write_qe_input(target / "pw.in", atoms, cfg)
        print(f"wrote {target / 'pw.in'} (ecutwfc={ecut}, ecutrho={8 * ecut:.0f}, restart={restart})")
    print(f"box: {len(atoms)} atoms, {atoms.get_chemical_formula()}")


if __name__ == "__main__":
    main()
