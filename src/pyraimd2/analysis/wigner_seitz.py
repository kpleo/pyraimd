"""Wigner-Seitz point-defect counting for irradiation cascades
(design-m3.md; method: Nordlund et al., PRB 57, 7556 (1998); implementation
follows Stukowski, MSMSE 18, 015012 (2010), the OVITO algorithm).

Every atom is assigned to its nearest perfect-lattice reference site under
the minimum image convention.  A site holding zero atoms is a vacancy; a
site holding ``occ >= 2`` atoms contributes ``occ - 1`` interstitials.  When
the atom count matches the reference (atom-conserving frame) the two tallies
are equal and each is the number of Frenkel pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class DefectCount:
    """Point-defect tally from one Wigner-Seitz occupation pass.

    Attributes:
        n_vacancies: Reference sites with zero assigned atoms.
        n_interstitials: Excess atoms, i.e. ``sum(occ - 1)`` over sites with
            ``occ >= 2``.
        n_frenkel_pairs: ``min(n_vacancies, n_interstitials)`` — equals both
            tallies when the frame conserves atoms; otherwise only paired
            vacancy-interstitial counts are reported.
        site_occupancies: (M,) per-site assigned-atom counts, in the order of
            the ``reference`` array passed to :func:`count_defects`.
    """

    n_vacancies: int
    n_interstitials: int
    n_frenkel_pairs: int
    site_occupancies: np.ndarray


def reference_sites_bcc(
    cell: np.ndarray, lattice_const: float, tol: float = 1e-4
) -> np.ndarray:
    """Perfect bcc lattice sites filling an orthogonal, axis-aligned cell.

    Args:
        cell: (3, 3) cell matrix in Å; must be diagonal with each edge an
            integer multiple of ``lattice_const``.
        lattice_const: bcc lattice constant a in Å.
        tol: Absolute tolerance in Å for the orthogonality and
            integer-multiple checks.

    Returns:
        (M, 3) site positions in Å with M = 2·n_x·n_y·n_z — the two-site bcc
        basis replicated over the n_x×n_y×n_z conventional cells.

    Raises:
        ValueError: If the cell is not diagonal or an edge is not an integer
            multiple of ``lattice_const``.

    For a reference taken from a perfect relaxed structure instead, pass
    ``perfect_atoms.get_positions()`` directly to :func:`count_defects`.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be a (3, 3) matrix, got shape {cell.shape}")
    if lattice_const <= 0.0:
        raise ValueError(f"lattice_const must be positive, got {lattice_const}")
    edges = np.diag(cell)
    if np.any(edges <= 0.0) or not np.allclose(
        cell, np.diag(edges), rtol=0.0, atol=tol
    ):
        raise ValueError("cell must be orthogonal and axis-aligned (diagonal)")
    repeats = np.rint(edges / lattice_const).astype(int)
    if np.any(repeats < 1) or not np.allclose(
        edges, repeats * lattice_const, rtol=0.0, atol=tol
    ):
        raise ValueError(
            "each cell edge must be an integer multiple of lattice_const: "
            f"edges={edges}, a={lattice_const}"
        )
    axes = [lattice_const * np.arange(n) for n in repeats]
    corners = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    return np.concatenate([corners, corners + 0.5 * lattice_const])


def count_defects(atoms: Atoms, reference: np.ndarray) -> DefectCount:
    """Assign atoms to their nearest reference site and tally point defects.

    The assignment is a Voronoi partition of the reference sites: each atom
    goes to the site nearest under the minimum image convention in the cell
    of ``atoms``.  Minimum image is taken in scaled (fractional) coordinates,
    which is exact for orthogonal cells and for moderately skewed ones —
    extremely skewed cells can misassign atoms sitting near a Wigner-Seitz
    boundary of the simulation cell itself.

    Args:
        atoms: Configuration to analyze; must be fully periodic.
        reference: (M, 3) reference site positions in Å, e.g. from
            :func:`reference_sites_bcc` or ``perfect.get_positions()``.

    Returns:
        The :class:`DefectCount` for this frame.

    Raises:
        ValueError: On an empty/malformed reference, a non-periodic or
            singular cell.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) == 0:
        raise ValueError(
            f"reference must be a non-empty (M, 3) array, got {reference.shape}"
        )
    if not all(atoms.pbc):
        raise ValueError("Wigner-Seitz counting needs a fully periodic cell")
    cell = np.asarray(atoms.cell)
    if abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("cell is singular")

    n_sites = len(reference)
    occupancies = np.zeros(n_sites, dtype=np.int64)
    if len(atoms) > 0:
        scaled_ref = (reference @ np.linalg.inv(cell)) % 1.0
        scaled_atoms = atoms.get_scaled_positions() % 1.0
        # Cap the (chunk, n_sites, 3) displacement block at ~2^21 atom–site
        # pairs so huge cascade cells do not exhaust memory.
        chunk = max(1, min(1024, (1 << 21) // n_sites))
        for start in range(0, len(atoms), chunk):
            delta = scaled_atoms[start : start + chunk, None, :] - scaled_ref[None]
            delta -= np.round(delta)  # minimum image: scaled coords in [-1/2, 1/2)
            disp = delta @ cell
            nearest = np.argmin(np.einsum("bij,bij->bi", disp, disp), axis=1)
            occupancies += np.bincount(nearest, minlength=n_sites)

    n_vacancies = int(np.count_nonzero(occupancies == 0))
    crowded = occupancies[occupancies >= 2]
    n_interstitials = int(crowded.sum() - len(crowded))
    return DefectCount(
        n_vacancies=n_vacancies,
        n_interstitials=n_interstitials,
        n_frenkel_pairs=min(n_vacancies, n_interstitials),
        site_occupancies=occupancies,
    )
