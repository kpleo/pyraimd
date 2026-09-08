"""Tiny H2 energetic-loop smoke with PySCF and a harmonic or explicit MACE model.

Usage: uv run --extra pyscf python examples/energetic_pyscf.py --output results/h2
Add --model MODEL_FILE to use a local MACE checkpoint; no weights are downloaded.
The default analytic bond potential needs no trained weights. This short
trajectory exercises the quantum-engine boundary and makes no accuracy or
speedup claim. The reference is molecular PBE/STO-3G with analytic gradients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from energetic_loop import prepare_output, write_report

from pyraimd2.engines.pyscf_engine import PyscfEngine
from pyraimd2.loop import EnergeticRunner
from pyraimd2.surrogate.base import SurrogatePrediction


class HarmonicBond:
    """Conservative cheap bond model, with energy and force in ASE units."""

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        vector = atoms.positions[1] - atoms.positions[0]
        distance = float(np.linalg.norm(vector))
        if distance <= 0:
            raise ValueError("the H2 bond must have positive length")
        stretch = distance - 0.74
        derivative = 20.0 * stretch * vector / distance
        return SurrogatePrediction(10.0 * stretch**2, np.array([derivative, -derivative]),
                                   None, np.full(2, np.nan))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/h2"),
                        help="directory for trajectory.db and summary.json (default: results/h2)")
    parser.add_argument("--model", type=Path, help="existing local MACE checkpoint; default uses an analytic bond")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.model is not None and not args.model.is_file():
        parser.error("--model must identify an existing local checkpoint")
    from pyscf import lib

    lib.num_threads(args.threads)
    model = HarmonicBond()
    if args.model is not None:
        import torch

        from pyraimd2.surrogate.mace_surrogate import MaceSurrogate

        torch.set_num_threads(args.threads)
        model = MaceSurrogate(model=str(args.model.resolve()), device="cpu", default_dtype="float64")
    atoms = Atoms("H2", positions=[[0, 0, -0.385], [0, 0, 0.385]])
    atoms.set_velocities([[0, 0, -0.02], [0, 0, 0.02]])
    store = prepare_output(args.output)
    runner = EnergeticRunner(atoms, model, PyscfEngine(basis="sto-3g", conv_tol=1e-9), store,
                             "h2", force_budget=0.1, timestep_fs=0.05,
                             probe_steps=(0.002, 0.004), numerical_floor=1e-4,
                             time_cap_fs=0.2, transverse_cap=0.15,
                             check_probability=0.5, check_seed=23)
    write_report(args.output, runner, runner.run(args.steps),
                 "mace-local" if args.model is not None else "harmonic-bond")


if __name__ == "__main__":
    main()
