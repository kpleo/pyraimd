"""FixAtomsProjection: projection rules and explicit rejections (hermetic)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLength, Hookean

from pyraimd2.loop.constraints import (
    ConstraintError,
    FixAtomsProjection,
    validate_constraints,
)


def _atoms(n=4):
    return Atoms("H4", positions=[[i * 0.7, 0.0, 0.0] for i in range(n)])


def test_from_atoms_and_projections():
    atoms = _atoms()
    atoms.set_constraint(FixAtoms(indices=[1, 3]))
    projection = validate_constraints(atoms)
    assert projection is not None
    assert projection.indices == (1, 3)
    assert projection.n_free == 2 and projection.n_fixed == 2
    assert projection.as_dict()["kind"] == "fixatoms"
    forces = np.ones((4, 3))
    projected = projection.project_forces(forces)
    np.testing.assert_array_equal(projected[0], np.ones(3))
    np.testing.assert_array_equal(projected[1], np.zeros(3))
    np.testing.assert_array_equal(projected[3], np.zeros(3))
    delta = np.full((4, 3), 0.5)
    np.testing.assert_array_equal(
        projection.project_displacement(delta)[1], np.zeros(3))
    assert projection.max_fixed_displacement(delta) == pytest.approx(
        np.linalg.norm([0.5, 0.5, 0.5]))


def test_direction_projection_drops_fixed_only_and_renormalizes():
    atoms = _atoms()
    atoms.set_constraint(FixAtoms(indices=[0, 1, 2]))
    projection = validate_constraints(atoms)
    directions = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # fixed-only
                           [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
    projected = projection.project_directions(directions)
    assert projected is not None and projected.shape == (1, 4, 3)
    np.testing.assert_allclose(projected[0, 3], [0.0, 1.0, 0.0])
    assert projection.project_directions(directions[:1]) is None


def test_metric_norms_active_vs_all_atoms():
    atoms = _atoms()
    atoms.set_constraint(FixAtoms(indices=[0]))
    projection = validate_constraints(atoms)
    residual = np.array([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]])
    assert projection.metric_norm(residual, "active_dofs_max_atom") == 1.0
    assert projection.metric_norm(residual, "all_atoms_max_atom") == 5.0
    with pytest.raises(ValueError, match="force_metric"):
        projection.metric_norm(residual, "bogus")


def test_merges_multiple_fixatoms_and_rejects_duplicates():
    atoms = _atoms()
    atoms.set_constraint([FixAtoms(indices=[0, 2]), FixAtoms(indices=[1])])
    projection = validate_constraints(atoms)
    assert projection.indices == (0, 1, 2)
    with pytest.raises(ConstraintError, match="out of range"):
        FixAtomsProjection(4, [7])
    with pytest.raises(ValueError, match="duplicate"):
        FixAtomsProjection(4, [1, 1])


def test_unsupported_constraints_rejected_explicitly():
    atoms = _atoms()
    assert validate_constraints(atoms) is None
    atoms.set_constraint(Hookean(a1=0, a2=1, k=5.0, rt=1.2))
    with pytest.raises(ConstraintError, match="FixAtoms only"):
        validate_constraints(atoms)
    atoms2 = _atoms()
    atoms2.set_constraint(FixBondLength(0, 1))
    with pytest.raises(ConstraintError, match="FixAtoms only"):
        validate_constraints(atoms2)
