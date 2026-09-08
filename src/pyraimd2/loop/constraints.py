"""Constraint semantics for dynamics: FixAtoms only in this version.

Backends always return raw physical energy/forces (WP00); constraints are
applied by the workflow exactly once, identically on the reference and the
surrogate path.  Anything richer than ``FixAtoms`` — RATTLE and other
holonomic constraints, energy-carrying constraints (Hookean springs),
moving/deforming constraints, and any variable-cell (NPT) dynamics — is
rejected with an explicit error rather than silently approximated.

The projection rules (plan §5.5):

- probe directions and probe displacements carry zero fixed-DOF components
  on both force paths, so a probe never moves a fixed atom;
- the driving force used for propagation has fixed-DOF components zeroed;
- error norms and the force budget apply to the free coordinates by
  default (``active_dofs_max_atom``); the raw all-atom residual is always
  kept as a diagnostic, and ``all_atoms_max_atom`` is available as an
  explicit alternative metric — never a silent redefinition;
- a fixed atom's displacement is zero and its path work is zero; the
  support layer may carry reaction forces, and no global
  momentum-conservation requirement is imposed on a constrained surface.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from pyraimd2.config import FORCE_METRICS

__all__ = ["FORCE_METRICS", "ConstraintError", "FixAtomsProjection",
           "validate_constraints"]


class ConstraintError(ValueError):
    """A constraint (or ensemble) is outside this version's supported set."""


def _fixatoms_indices(atoms: Atoms) -> list[int]:
    """Merged indices of every FixAtoms constraint; raise on any other kind."""
    indices: list[int] = []
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            name = type(constraint).__name__
            raise ConstraintError(
                f"unsupported constraint {name}: this version supports "
                "FixAtoms only — RATTLE/holonomic, energy-carrying "
                "(Hookean) and moving constraints are rejected explicitly; "
                "remove them or run unconstrained")
        indices.extend(int(i) for i in np.atleast_1d(constraint.index))
    return sorted(set(indices))


class FixAtomsProjection:
    """Fixed-DOF projection shared by the energetic and plain drivers."""

    def __init__(self, n_atoms: int, indices: list[int]) -> None:
        if not indices:
            raise ValueError("FixAtomsProjection needs at least one index")
        if len(set(indices)) != len(indices):
            raise ValueError(f"duplicate FixAtoms indices: {indices}")
        if min(indices) < 0 or max(indices) >= n_atoms:
            raise ConstraintError(
                f"FixAtoms indices {indices} out of range for {n_atoms} atoms")
        self.n_atoms = n_atoms
        self.indices = tuple(sorted(indices))
        self.free_mask = np.ones(n_atoms, dtype=bool)
        self.free_mask[list(self.indices)] = False

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> FixAtomsProjection | None:
        """Projection for the atoms' constraints, or None when unconstrained."""
        indices = _fixatoms_indices(atoms)
        return None if not indices else cls(len(atoms), indices)

    @property
    def n_fixed(self) -> int:
        return len(self.indices)

    @property
    def n_free(self) -> int:
        return self.n_atoms - self.n_fixed

    def as_dict(self) -> dict:
        return {"kind": "fixatoms", "indices": list(self.indices),
                "n_free": self.n_free, "n_fixed": self.n_fixed}

    def project_forces(self, forces: np.ndarray) -> np.ndarray:
        """Forces with fixed-DOF components zeroed (the propagation force)."""
        projected = np.array(forces, dtype=float, copy=True)
        projected[~self.free_mask] = 0.0
        return projected

    def project_displacement(self, displacement: np.ndarray) -> np.ndarray:
        """A displacement with fixed-atom components zeroed."""
        projected = np.array(displacement, dtype=float, copy=True)
        projected[~self.free_mask] = 0.0
        return projected

    def project_directions(self, directions: np.ndarray) -> np.ndarray | None:
        """Directions with fixed components zeroed and unit norm restored.

        Directions that become identically zero (fixed-only motion) are
        dropped; returns None when nothing remains.
        """
        directions = np.array(directions, dtype=float, copy=True)
        directions[..., ~self.free_mask, :] = 0.0
        lengths = np.linalg.norm(directions.reshape(len(directions), -1), axis=1)
        keep = lengths > 0
        if not np.any(keep):
            return None
        directions = directions[keep]
        lengths = lengths[keep]
        return directions / lengths[:, None, None]

    def metric_norm(self, residual: np.ndarray, force_metric: str) -> float:
        """Max per-atom residual norm under the configured metric.

        ``active_dofs_max_atom`` measures only the free coordinates (the
        default the force budget controls); ``all_atoms_max_atom`` includes
        the fixed-DOF reaction residual as an explicit alternative.
        """
        if force_metric not in FORCE_METRICS:
            raise ValueError(
                f"force_metric must be one of {FORCE_METRICS}, got {force_metric!r}")
        norms = np.linalg.norm(np.asarray(residual, dtype=float), axis=1)
        if force_metric == "all_atoms_max_atom":
            return float(norms.max())
        return float(norms[self.free_mask].max()) if self.free_mask.any() else 0.0

    def max_fixed_displacement(self, displacement: np.ndarray) -> float:
        """Largest fixed-atom displacement (must be zero by construction)."""
        return float(np.linalg.norm(
            np.asarray(displacement, dtype=float)[~self.free_mask], axis=1
        ).max()) if self.n_fixed else 0.0


def validate_constraints(atoms: Atoms) -> FixAtomsProjection | None:
    """Single entry point: projection for FixAtoms, None unconstrained,
    an explicit :class:`ConstraintError` for everything else."""
    return FixAtomsProjection.from_atoms(atoms)
