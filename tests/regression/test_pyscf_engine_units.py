"""PyscfEngine unit/sign conventions: analytic nuclear gradient vs central
finite difference of the SCF energy (design doc §3, rule 6 — conversions are
covered by tests at the engine boundary).

Runs in the default suite (~15 s): it is the cheap guard against the exact
class of unit/refactor bugs that killed PYRAIMD v1.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import units
from ase.build import molecule

from pyraimd2.engines import EngineError, PyscfEngine

FD_STEP_BOHR = 5e-4
FD_TOL_EV_PER_A = 1e-3


def _distorted_h2o():
    atoms = molecule("H2O")
    atoms.positions[1, 0] += 0.10  # stretch one O-H bond: clearly nonzero forces
    return atoms


def test_analytic_forces_vs_finite_difference() -> None:
    atoms = _distorted_h2o()
    engine = PyscfEngine(functional="pbe", basis="def2-svp", conv_tol=1e-10)
    result = engine.compute(atoms)

    assert result.stress is None  # finite molecule
    assert np.isfinite(result.energy)
    assert result.forces.shape == (len(atoms), 3)
    assert result.wall_time_s > 0.0

    max_dev = 0.0
    for i in range(len(atoms)):
        for d in range(3):
            energies = []
            for sign in (+1.0, -1.0):
                displaced = atoms.copy()
                displaced.positions[i, d] += sign * FD_STEP_BOHR * units.Bohr
                energies.append(engine.compute(displaced).energy)
            # F = -dE/dR; energies are in eV, displacement in Bohr.
            fd_force = -(energies[0] - energies[1]) / (2 * FD_STEP_BOHR) / units.Bohr
            max_dev = max(max_dev, abs(fd_force - result.forces[i, d]))

    assert max_dev < FD_TOL_EV_PER_A


def test_periodic_input_raises_engine_error() -> None:
    atoms = _distorted_h2o()
    atoms.pbc = True
    with pytest.raises(EngineError, match="pbc"):
        PyscfEngine().compute(atoms)


def test_odd_electron_count_raises_engine_error() -> None:
    atoms = molecule("H")  # 1 electron: RKS closed-shell cannot handle it
    with pytest.raises(EngineError, match="odd electron count"):
        PyscfEngine().compute(atoms)
