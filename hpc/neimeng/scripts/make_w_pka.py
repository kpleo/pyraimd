"""Build a bcc tungsten configuration with a primary knock-on atom.

One central W atom is kicked along a cubic direction family (<100>/<110>/<111>)
with a prescribed kinetic energy — 200-300 eV for the PKA-vs-spike validation
run — on top of a MATRIX_T Maxwell-Boltzmann matrix. The total momentum is
re-zeroed after the kick without rescaling, and the kinetic-energy ledger
(assigned kick, thermal cross term, COM-momentum removal, realized deposit)
is recorded in the JSON sidecar together with the PKA atom index and direction.

Usage:
  uv run python hpc/neimeng/scripts/make_w_pka.py --out w_pka_250eV_110.xyz \
      [--cells 7] [--direction 110] [--e-kick 250.0] [--matrix-T 300] [--seed 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyraimd2.builders.w_irradiation import (
    PKA_DIRECTIONS,
    build_pka_config,
    write_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cells", type=int, default=7)
    p.add_argument("--direction", default="110", choices=sorted(PKA_DIRECTIONS),
                   help="cubic direction family of the kick")
    p.add_argument("--e-kick", type=float, default=250.0,
                   help="assigned PKA kinetic energy, eV (200-300 for self-proof (a))")
    p.add_argument("--matrix-T", type=float, default=300.0, help="Kelvin")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_pka_config(
        cells=args.cells,
        direction=args.direction,
        e_kick_eV=args.e_kick,
        matrix_T_K=args.matrix_T,
        seed=args.seed,
    )
    _, sidecar = write_config(cfg, args.out)
    s = cfg.sidecar
    print(f"wrote {args.out} + {sidecar}: {s['n_atoms']} atoms, PKA atom "
          f"{s['pka_index']} kicked along <{s['pka_direction']}> with "
          f"{s['pka_e_kick_eV']} eV assigned / {s['ke_deposited_eV']:.3f} eV "
          f"deposited (COM removed {s['ke_com_removed_eV']:.3f} eV)")


if __name__ == "__main__":
    main()
