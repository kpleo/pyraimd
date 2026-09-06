"""Wigner-Seitz defect counting: perfect lattice, vacancy, Frenkel pair,
thermal jitter, crowdion, and non-cubic periodic cells (Nordlund et al., PRB 57, 7556 (1998)).

Geometries are built straight from the site generator, so the reference is
exact and every expected count is analytic.  No torch, no mace.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.analysis import DefectCount, count_defects, reference_sites_bcc

# bcc tungsten near the experimental lattice constant; all displacements in
# the tests stay far below a/2, so the exact value does not matter.
W_A = 3.16


def _bcc_block(n_cells: tuple[int, int, int]) -> tuple[Atoms, np.ndarray]:
    """Perfect bcc W block plus its reference sites, sharing one cell."""
    cell = np.diag([n * W_A for n in n_cells])
    sites = reference_sites_bcc(cell, W_A)
    atoms = Atoms(symbols=["W"] * len(sites), positions=sites, cell=cell, pbc=True)
    return atoms, sites


# -- reference site generation -------------------------------------------------


def test_reference_sites_fill_bcc_cell() -> None:
    cell = np.diag([2, 3, 4]) * W_A
    sites = reference_sites_bcc(cell, W_A)
    assert sites.shape == (2 * 2 * 3 * 4, 3)
    # Corner and body-center basis of the first conventional cell.
    assert [0.0, 0.0, 0.0] in sites.tolist()
    assert np.any(np.all(np.isclose(sites, 0.5 * W_A), axis=1))
    # Every site lies inside the cell.
    assert np.all(sites >= 0.0) and np.all(sites < np.diag(cell))


def test_reference_sites_reject_bad_cells() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        reference_sites_bcc(np.diag([4.0, 4.0, 4.0]) + 0.1, W_A)  # off-diagonal
    with pytest.raises(ValueError, match="integer multiple"):
        reference_sites_bcc(np.diag([4.5, 4.0, 4.0]) * W_A, W_A)  # half cell
    with pytest.raises(ValueError, match="positive"):
        reference_sites_bcc(np.diag([4.0, 4.0, 4.0]) * W_A, 0.0)


# -- perfect and near-perfect lattices -----------------------------------------


def test_perfect_lattice_has_no_defects() -> None:
    atoms, sites = _bcc_block((7, 7, 7))  # 686 atoms
    result = count_defects(atoms, sites)
    assert isinstance(result, DefectCount)
    assert result.n_vacancies == 0
    assert result.n_interstitials == 0
    assert result.n_frenkel_pairs == 0
    assert np.all(result.site_occupancies == 1)
    assert result.site_occupancies.sum() == len(atoms)


def test_small_random_jitter_keeps_zero_defects() -> None:
    """0.1 Å thermal-scale displacements (<< a/2) must not create defects."""
    atoms, sites = _bcc_block((7, 7, 7))
    rng = np.random.default_rng(19980915)
    atoms.positions += rng.uniform(-0.1, 0.1, size=atoms.positions.shape)
    result = count_defects(atoms, sites)
    assert result.n_vacancies == 0
    assert result.n_interstitials == 0
    assert np.all(result.site_occupancies == 1)


def test_atom_wrapped_across_boundary_stays_home() -> None:
    """An atom sitting slightly outside the cell wraps back to its own site."""
    atoms, sites = _bcc_block((4, 4, 4))
    atoms.positions[0] = [-0.05, 0.02, 0.03]  # site 0 is the origin corner
    result = count_defects(atoms, sites)
    assert result.n_vacancies == 0
    assert result.n_interstitials == 0


# -- vacancies, interstitials, Frenkel pairs ------------------------------------


def test_removed_atom_is_one_vacancy() -> None:
    atoms, sites = _bcc_block((7, 7, 7))
    del atoms[0]
    result = count_defects(atoms, sites)
    assert result.n_vacancies == 1
    assert result.n_interstitials == 0
    assert result.n_frenkel_pairs == 0  # a lone vacancy is not a pair
    assert result.site_occupancies[0] == 0
    assert result.site_occupancies.sum() == len(atoms)


def test_octahedral_interstitial_is_one_frenkel_pair() -> None:
    """Move one atom to the bcc octahedral site (a/2, a/2, 0): its home site
    empties and the atom lands closest to a neighbouring site."""
    atoms, sites = _bcc_block((7, 7, 7))
    atoms.positions[0] = [0.5 * W_A, 0.5 * W_A, 0.0]
    result = count_defects(atoms, sites)
    assert result.n_vacancies == 1
    assert result.n_interstitials == 1
    assert result.n_frenkel_pairs == 1
    assert result.site_occupancies[0] == 0  # home site is the vacancy
    assert result.site_occupancies.max() == 2
    assert result.site_occupancies.sum() == len(atoms)


def test_crowdion_site_has_occupancy_two() -> None:
    """Perfect lattice plus one extra atom sharing a site along <111>."""
    atoms, sites = _bcc_block((5, 5, 5))
    crowdion = sites[0] + 0.4 / np.sqrt(3.0)  # 0.4 Å along [111]
    atoms += Atoms(symbols=["W"], positions=[crowdion], cell=atoms.cell, pbc=True)
    result = count_defects(atoms, sites)
    assert result.site_occupancies[0] == 2
    assert result.site_occupancies.max() == 2
    assert result.n_vacancies == 0
    assert result.n_interstitials == 1
    assert result.n_frenkel_pairs == 0


# -- non-cubic periodic cells ----------------------------------------------------


def test_rectangular_cell_perfect_and_vacancy() -> None:
    atoms, sites = _bcc_block((4, 5, 6))  # 240 atoms, orthogonal but not cubic
    result = count_defects(atoms, sites)
    assert result.n_vacancies == 0
    assert result.n_interstitials == 0
    assert np.all(result.site_occupancies == 1)

    # Remove the atom at the cell corner: the vacancy must be found through
    # the periodic boundary (its neighbours lie across the far faces).
    del atoms[0]
    result = count_defects(atoms, sites)
    assert result.n_vacancies == 1
    assert result.n_interstitials == 0
    assert result.site_occupancies[0] == 0


def test_rectangular_cell_frenkel_pair_near_boundary() -> None:
    """Displace a boundary atom across the -x face to just off a body-centre
    site (offset from the exact Voronoi vertex, so the nearest site is
    unambiguous)."""
    atoms, sites = _bcc_block((3, 4, 5))
    atoms.positions[0] = [-0.5 * W_A, 0.5 * W_A - 0.1, 0.5 * W_A - 0.1]
    result = count_defects(atoms, sites)
    assert result.n_vacancies == 1
    assert result.n_interstitials == 1
    assert result.n_frenkel_pairs == 1
    assert result.site_occupancies[0] == 0


# -- input validation -------------------------------------------------------------


def test_count_defects_rejects_bad_inputs() -> None:
    atoms, sites = _bcc_block((2, 2, 2))
    atoms.pbc = False
    with pytest.raises(ValueError, match="periodic"):
        count_defects(atoms, sites)

    atoms, _ = _bcc_block((2, 2, 2))
    with pytest.raises(ValueError, match="reference"):
        count_defects(atoms, np.empty((0, 3)))
    with pytest.raises(ValueError, match="reference"):
        count_defects(atoms, np.zeros((4, 2)))
