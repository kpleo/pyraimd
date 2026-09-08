"""Build the bare-Li-slab control initial condition.

Takes the EXACT 36 Li slab positions from the flagship-A production initial
frame (step -1 of the archived loop.db) in the same 10.53x10.53x52.15 A
cell, removes the electrolyte, and draws fresh 300 K Maxwell-Boltzmann
velocities. Running this box with pure-DFT propagation answers the question
the physics check raised: does the (100) slab keep its layering at 300 K
WITHOUT the electrolyte (i.e., is the observed layering loss
adsorption-driven and physical, or a setup artifact)?

Usage:
  uv run python hpc/neimeng/scripts/make_li_slab_control.py \
      --db analysis/flagship_a_prod/loop.db --out /tmp/li_slab_control.xyz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.md.velocitydistribution import Stationary, thermalize_momenta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--step", type=int, default=-1)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    import ase.db

    args = parse_args()
    db = ase.db.connect(args.db)
    rows = [r for r in db.select() if r.key_value_pairs["step"] == args.step]
    if len(rows) != 1:
        raise SystemExit(f"step {args.step}: expected 1 row, got {len(rows)}")
    atoms = rows[0].toatoms()
    mask = [a.symbol == "Li" for a in atoms]
    li = atoms[mask]
    # Keep only the slab: the 36 atoms forming the dense contiguous z-cluster
    # (the 3 electrolyte Li+ sit far off the slab band).
    order = np.argsort(li.positions[:, 2])
    z = li.positions[order, 2]
    best, best_span = 0, np.inf
    for i in range(len(z) - 35):
        span = z[i + 35] - z[i]
        if span < best_span:
            best, best_span = i, span
    slab = li[order[best:best + 36]]
    slab.arrays.pop("momenta", None)  # drop the production half-step momenta
    thermalize_momenta(slab, args.temperature,
                       rng=np.random.default_rng(args.seed))
    Stationary(slab)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    from ase.io import write
    write(args.out, slab)
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "source": f"{args.db} step {args.step} Li sublattice, electrolyte removed",
        "n_atoms": len(slab), "cell_A": [float(x) for x in np.diag(slab.cell)],
        "temperature_K": args.temperature, "seed": args.seed,
    }, indent=2))
    print(f"wrote {args.out} + {sidecar}: {len(slab)} Li, "
          f"cell {np.diag(slab.cell).round(3)}")


if __name__ == "__main__":
    main()
