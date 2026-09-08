"""Generate configurations for a tungsten threshold-displacement-energy scan.

The default grid gives a primary knock-on atom 40--140 eV along <100> in
10 eV increments. Each point uses seed + i for its thermal initialization.
Only structures and JSON metadata are written; determining a displacement
threshold requires subsequent dynamics and defect analysis.

Usage:
  uv run python hpc/neimeng/scripts/make_w_ed_scan.py --out-dir DIR
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
