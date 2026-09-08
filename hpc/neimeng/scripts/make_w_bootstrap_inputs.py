"""Generate the W (flagship B) bootstrap input structures.

Two inputs, both 7^3 = 686-atom bcc W boxes at the campaign lattice
constant (a0 = 3.165 A, matching the ecut-convergence campaign inputs):

- ``w_bulk_686.xyz``: pristine bulk for the 300/3000/6000 K Langevin
  frame segments (velocities drawn by committee_md at its --temperature).
- ``w_spike_686.xyz``: the two-temperature thermal-spike surrogate of the
  cascade core (core radius 5.5 A at 8000 K in a 300 K matrix, seed 1),
  momenta carried in the xyz; used with committee_md --keep-velocities.

Usage:
  uv run python hpc/neimeng/scripts/make_w_bootstrap_inputs.py \
      --out-dir inputs/generated
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyraimd2.builders.w_irradiation import (
    BuiltConfig,
    build_spike_config,
    build_w_supercell,
    write_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--cells", type=int, default=7)
    p.add_argument("--core-radius", type=float, default=5.5, help="Angstrom")
    p.add_argument("--core-T", type=float, default=8000.0, help="Kelvin")
    p.add_argument("--matrix-T", type=float, default=300.0, help="Kelvin")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bulk_atoms = build_w_supercell(args.cells)
    bulk_cfg = BuiltConfig(
        bulk_atoms,
        {
            "cells": args.cells,
            "n_atoms": len(bulk_atoms),
            "a0": 3.165,
            "kind": "pristine bulk (frame-generation seed; velocities drawn "
                    "by committee_md at its --temperature)",
            "seed": args.seed,
        },
    )
    b_xyz, b_json = write_config(bulk_cfg, args.out_dir / "w_bulk_686.xyz")
    print(f"wrote {b_xyz} + {b_json}: {len(bulk_atoms)} atoms pristine bulk")

    spike_cfg = build_spike_config(
        cells=args.cells,
        core_radius_a=args.core_radius,
        core_T_K=args.core_T,
        matrix_T_K=args.matrix_T,
        seed=args.seed,
    )
    s_xyz, s_json = write_config(spike_cfg, args.out_dir / "w_spike_686.xyz")
    print(f"wrote {s_xyz} + {s_json}: {spike_cfg.sidecar['n_atoms']} atoms, "
          f"core {spike_cfg.sidecar['n_core']} at {args.core_T} K "
          f"in {args.matrix_T} K matrix")


if __name__ == "__main__":
    main()
