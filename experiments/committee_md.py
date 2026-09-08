"""Committee-driven NVT MD for bootstrap frame generation (M3 flagship).

Runs Langevin dynamics with the committee mean as the force engine — no
DFT in the loop; the frames are later labeled by the SCF array.  Frames
carry the physically correlated configurations (solvation shells, H-bond
networks) that rattling cannot produce; zero-shot committee accuracy is
sufficient for generation because the labels decide the truth.

Usage:
    python experiments/committee_md.py --atomsXYZ box.xyz \
        --model 0b3.model mpa0.model --temperature 450 --n-steps 400 \
        --sample-every 10 --out frames_450K.extxyz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta

from pyraimd2.surrogate import CommitteeSurrogate


class CommitteeAseCalculator(Calculator):
    """Thin ASE adapter over CommitteeSurrogate (energy + forces only)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, committee: CommitteeSurrogate) -> None:
        super().__init__()
        self.committee = committee

    def calculate(self, atoms, properties, system_changes) -> None:  # noqa: D102
        pred = self.committee.predict(atoms)
        self.results["energy"] = pred.energy
        self.results["forces"] = pred.forces


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atomsXYZ", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path, nargs="+")
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--friction", type=float, default=0.01,
                   help="Langevin friction in 1/fs (gentle thermostat for frame generation)")
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20250822)
    p.add_argument("--keep-velocities", action="store_true",
                   help="use the velocities carried by the xyz (e.g. the "
                        "two-temperature spike initial condition); default "
                        "re-draws Maxwell-Boltzmann at --temperature")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    atoms = read(args.atomsXYZ)
    assert all(atoms.pbc), "frame generation expects a fully periodic box"

    models = [str(m) for m in args.model]
    committee = CommitteeSurrogate(
        model=models if len(models) > 1 else models[0],
        n_members=len(models) if len(models) > 1 else 2,
        seed=args.seed,
        trainable_filters=("readout", "products"),
    )
    atoms.calc = CommitteeAseCalculator(committee)

    if args.keep_velocities:
        if "momenta" not in atoms.arrays:
            raise SystemExit("--keep-velocities set but the xyz carries no "
                             "velocities/momenta")
    else:
        thermalize_momenta(atoms, args.temperature, rng=np.random.default_rng(args.seed))
    dyn = Langevin(
        atoms,
        args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction,
    )

    frames_path = args.out
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    # Incremental append: a wall-clock kill must not lose the frames already
    # produced (campaigns run against partition time limits).
    if frames_path.exists():
        frames_path.unlink()

    def _sample() -> None:
        write(frames_path, atoms.copy(), append=True)

    dyn.attach(_sample, interval=args.sample_every)
    dyn.run(args.n_steps)
    print(f"wrote frames incrementally to {frames_path} "
          f"({args.n_steps} steps at {args.temperature} K, "
          f"~{args.n_steps // args.sample_every + 1} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
