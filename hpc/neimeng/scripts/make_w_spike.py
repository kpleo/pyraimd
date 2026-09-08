"""Build a tungsten thermal-spike initial condition.

Configurable starting parameters (choose them for the material process):

- bcc W, NCELLS^3 unit cells (7 -> 686 atoms; 6 -> 432 is the documented fallback)
- a spherical core of radius CORE_R_A at the box center is heated to CORE_T K,
  the matrix to MATRIX_T K, by Maxwell-Boltzmann velocity draws (two-temperature
  initialization = the thermal-spike idealization of a cascade core)
- output: XYZ with cell + a JSON sidecar recording every parameter (seeded initial configuration)

The implementation lives in ``pyraimd2.builders.w_irradiation`` (shared with the
finite-size series so all spike configs are numerically identical for a seed);
this script supplies the command-line interface.

Usage:
  uv run python hpc/neimeng/scripts/make_w_spike.py --out w_spike_686.xyz \
      [--cells 7] [--core-radius 5.5] [--core-T 8000] [--matrix-T 300] [--seed 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyraimd2.builders.w_irradiation import build_spike_config, write_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cells", type=int, default=7)
    p.add_argument("--core-radius", type=float, default=5.5, help="Angstrom")
    p.add_argument("--core-T", type=float, default=8000.0, help="Kelvin")
    p.add_argument("--matrix-T", type=float, default=300.0, help="Kelvin")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_spike_config(
        cells=args.cells,
        core_radius_a=args.core_radius,
        core_T_K=args.core_T,
        matrix_T_K=args.matrix_T,
        seed=args.seed,
    )
    _, sidecar = write_config(cfg, args.out)
    n = cfg.sidecar["n_atoms"]
    n_core = cfg.sidecar["n_core"]
    print(f"wrote {args.out} + {sidecar}: {n} atoms, core {n_core} atoms "
          f"at {args.core_T} K in {args.matrix_T} K matrix")


if __name__ == "__main__":
    main()
