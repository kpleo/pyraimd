"""Run a small energetic-gated NVE trajectory without external model weights.

Usage: uv run python examples/energetic_loop.py --output results/energetic

The harmonic reference and base potential provide exact energies and forces.
The database records calibration cost, forecasts, checked endpoint work and
the actual driving forces. This is an integration example, not a speed test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import ase.db
import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction


class HarmonicReference:
    name = "harmonic-reference"

    def __init__(self, stiffness: float = 1.2) -> None:
        self.stiffness = stiffness

    def compute(self, atoms: Atoms) -> EngineResult:
        positions = atoms.positions
        return EngineResult(0.5 * self.stiffness * float(np.sum(positions**2)),
                            -self.stiffness * positions, None, 0.0)


class HarmonicSurrogate:
    def __init__(self, stiffness: float = 0.8) -> None:
        self.stiffness = stiffness

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        positions = atoms.positions
        return SurrogatePrediction(0.5 * self.stiffness * float(np.sum(positions**2)),
                                   -self.stiffness * positions, None,
                                   np.full(len(atoms), np.nan))


def prepare_output(output: Path) -> Store:
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("trajectory.db", "summary.json")):
        raise FileExistsError("output contains an existing trajectory or summary; choose a new directory")
    return Store(output / "trajectory.db")


def write_report(output: Path, runner: EnergeticRunner, summary, model: str) -> dict:
    rows = list(ase.db.connect(str(output / "trajectory.db")).select(run_id=runner.calc.run_id))
    if len(rows) != summary.n_steps + 1:
        raise RuntimeError("one stored force evaluation is required at every MD time point")
    observed_work = []
    for row in rows:
        driving = row.data["driving"]
        if not np.isfinite(driving["energy"]) or not np.isfinite(driving["forces"]).all():
            raise RuntimeError("nonfinite driving label")
        metadata = row.data["metadata"]
        for forecast in metadata["forecasts"]:
            if not all(np.isfinite(forecast[key]) for key in
                       ("linear_error_eV_A", "envelope_eV_A", "predicted_work_eV")):
                raise RuntimeError("nonfinite forecast")
        if metadata["observed"] is not None:
            value = metadata["observed"]["endpoint_work_eV"]
            if not np.isfinite(value):
                raise RuntimeError("nonfinite observed residual work")
            observed_work.append(value)
    report = {"model": model, **asdict(summary), "stored_evaluations": len(rows),
              "labeled_segment_endpoints": len(observed_work),
              "max_abs_endpoint_work_eV": max(map(abs, observed_work), default=None),
              "final_positions_A": runner.atoms.positions.tolist(),
              "final_kinetic_energy_eV": float(runner.atoms.get_kinetic_energy())}
    with (output / "summary.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/energetic"),
                        help="directory for trajectory.db and summary.json (default: results/energetic)")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    atoms = Atoms("H2", positions=[[-0.4, 0, 0], [0.4, 0, 0]])
    atoms.set_velocities([[-0.1, 0, 0], [0.1, 0, 0]])
    store = prepare_output(args.output)
    runner = EnergeticRunner(atoms, HarmonicSurrogate(), HarmonicReference(), store,
                             "harmonic", force_budget=0.02, timestep_fs=0.1,
                             time_cap_fs=0.4, transverse_cap=0.1,
                             check_probability=0.5, check_seed=23)
    write_report(args.output, runner, runner.run(args.steps), "harmonic")


if __name__ == "__main__":
    main()
