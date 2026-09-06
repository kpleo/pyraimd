"""Build a finite-size series of tungsten thermal-spike configurations.

Builds the 4^3/5^3/7^3 bcc W supercells (128/250/686 atoms) at a unified core
energy density (default 1.0 eV per core atom, an approximately 8000 K core).
The core temperature is rescaled per size via
``core_temperature_for_energy_density`` (momentum-zeroing DOF corrected) and
recorded in each sidecar, so the expected deposited energy per core atom is
identical across the series. Same protocol as make_w_spike.py — one shared
implementation.

Usage:
  uv run python hpc/neimeng/scripts/make_w_spike_series.py --out-dir DIR \
      [--cells 4 5 7] [--core-radius 5.5] [--energy-density 1.0] \
      [--matrix-T 300] [--seed 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyraimd2.builders.w_irradiation import (
    build_spike_series_configs,
    write_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--cells", type=int, nargs="+", default=[4, 5, 7])
    p.add_argument("--core-radius", type=float, default=5.5, help="Angstrom")
    p.add_argument("--energy-density", type=float, default=1.0,
                   help="target deposited energy per core atom, eV")
    p.add_argument("--matrix-T", type=float, default=300.0, help="Kelvin")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_spike_series_configs(
        cells_list=tuple(args.cells),
        core_radius_a=args.core_radius,
        energy_density_eV=args.energy_density,
        matrix_T_K=args.matrix_T,
        seed=args.seed,
    )
    for cfg in configs:
        s = cfg.sidecar
        out = args.out_dir / f"w_spike_{s['n_atoms']}.xyz"
        _, sidecar = write_config(cfg, out)
        print(f"wrote {out} + {sidecar}: {s['n_atoms']} atoms, core {s['n_core']} "
              f"atoms at {s['core_T_K']:.1f} K = {s['energy_density_eV_per_core_atom']} "
              f"eV/atom in {s['matrix_T_K']} K matrix")


if __name__ == "__main__":
    main()
