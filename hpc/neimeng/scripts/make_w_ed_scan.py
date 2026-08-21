"""<100> threshold-displacement-energy (E_d) scan stub (design-m3.md v2.1).

Generates the directional E_d scan grid — PKA configurations along <100> at
40-140 eV in 10 eV steps — as configurations plus JSON sidecars only; no
calculations are submitted. Motivation (v2.1 audit): directional MD values
exist (Banisalman et al. 2017) but no DFT-resolved directional E_d for W, so
this is the "<100> E_d from DFT fallback" mini-experiment. Point i uses
seed + i so the 300 K thermal draws are independent across the grid.

Usage:
  uv run python hpc/neimeng/scripts/make_w_ed_scan.py --out-dir DIR \
      [--direction 100] [--e-min 40] [--e-max 140] [--e-step 10] \
      [--cells 7] [--matrix-T 300] [--seed 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyraimd2.builders.w_irradiation import (
    PKA_DIRECTIONS,
    build_ed_scan_configs,
    ed_scan_energies,
    write_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--direction", default="100", choices=sorted(PKA_DIRECTIONS))
    p.add_argument("--e-min", type=float, default=40.0, help="eV")
    p.add_argument("--e-max", type=float, default=140.0, help="eV")
    p.add_argument("--e-step", type=float, default=10.0, help="eV")
    p.add_argument("--cells", type=int, default=7)
    p.add_argument("--matrix-T", type=float, default=300.0, help="Kelvin")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    energies = ed_scan_energies(args.e_min, args.e_max, args.e_step)
    configs = build_ed_scan_configs(
        direction=args.direction,
        energies=energies,
        cells=args.cells,
        matrix_T_K=args.matrix_T,
        seed=args.seed,
    )
    for cfg in configs:
        s = cfg.sidecar
        out = args.out_dir / f"w_ed_{round(s['pka_e_kick_eV'])}eV_{s['pka_direction']}.xyz"
        _, sidecar = write_config(cfg, out)
        print(f"wrote {out} + {sidecar}: seed {s['seed']}, "
              f"{s['ke_deposited_eV']:.2f} eV deposited")
    print(f"{len(configs)} scan points ({energies[0]:.0f}-{energies[-1]:.0f} eV "
          f"along <{args.direction}>); configurations only, nothing submitted")


if __name__ == "__main__":
    main()
