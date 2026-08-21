"""Unit tests for experiments/build_interface_box.py (flagship A builder).

Pure-Python coverage only: composition bookkeeping, box geometry, density
accounting, the PBC overlap checker, packmol input rendering, and the
validation gate on synthetic packings. No mace/torch/packmol/ase — the module
under test needs numpy alone; packmol execution is not exercised here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[2] / "experiments" / "build_interface_box.py"
spec = importlib.util.spec_from_file_location("build_interface_box", MODULE_PATH)
bib = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bib  # dataclasses resolve cls.__module__ via sys.modules
spec.loader.exec_module(bib)


@pytest.fixture
def plan() -> bib.BuildPlan:
    return bib.make_plan()


# --- composition and mass bookkeeping ---------------------------------------


def test_default_composition_element_counts(plan):
    counts = bib.count_elements(plan.composition, plan.surface, plan.layers)
    # From the molecular formulas: 36 slab Li + 3 LiPF6 -> Li 39;
    # EC C3H4O3 x21 + DMC C3H6O3 x17 -> C 114, H 186, O 114; P 3, F 18.
    assert counts == {"Li": 39, "C": 114, "H": 186, "O": 114, "P": 3, "F": 18}
    assert sum(counts.values()) == 474


def test_template_atom_counts_match_formulas():
    for species, (symbols, positions) in bib.TEMPLATES.items():
        assert len(symbols) == len(positions)
        for element, k in bib.FORMULAS[species].items():
            assert symbols.count(element) == k


# --- box geometry and density ------------------------------------------------


def test_box_dimensions(plan):
    g = plan.geometry
    assert g.lx == pytest.approx(3 * 3.51)
    assert g.ly == pytest.approx(3 * 3.51)
    assert g.slab_span == pytest.approx(3 * 3.51 / 2)  # 4 layers, spacing a/2
    # symmetric dual-interface construction
    assert g.region_bottom[1] == pytest.approx(g.t_side)
    assert g.region_top[0] == pytest.approx(g.lz - g.t_side)
    assert g.z_slab_bottom == pytest.approx(g.t_side + g.gap)
    assert g.z_slab_top == pytest.approx(g.t_side + g.gap + g.slab_span)
    assert g.lz == pytest.approx(g.slab_span + 2 * g.gap + 2 * g.t_side)
    # spec: box ~10.5 x 10.5 x ~51-52 A
    assert 51.0 < g.lz < 53.0


def test_liquid_density_on_target(plan):
    g = plan.geometry
    mass = bib.electrolyte_mass_amu(plan.composition)
    rho = bib.liquid_density_g_cm3(mass, g.liquid_volume_a3)
    assert rho == pytest.approx(1.28, abs=1e-9)
    # LP30 consistency: 3 LiPF6 in this volume is ~1 mol/L
    assert bib.salt_molarity(plan.composition["pf6"], g.liquid_volume_a3) == pytest.approx(
        1.0, abs=0.02
    )


def test_lz_override_recomputes_density():
    plan = bib.make_plan(lz=60.0)
    g = plan.geometry
    assert g.lz == 60.0
    assert g.t_side == pytest.approx((60.0 - g.slab_span - 2 * g.gap) / 2)
    mass = bib.electrolyte_mass_amu(plan.composition)
    rho = bib.liquid_density_g_cm3(mass, g.liquid_volume_a3)
    # a 60 A box holds more volume than the derived ~52.15 A -> density drops
    assert rho < 1.28


def test_lz_override_too_small_raises():
    with pytest.raises(ValueError, match="no room for liquid"):
        bib.make_plan(lz=6.0)


# --- Li(100) slab stacking ----------------------------------------------------


def test_slab_stacking_bcc100(plan):
    pos = bib.slab_positions(plan.li_a, plan.surface, plan.layers, z_bottom=0.0)
    assert pos.shape == (36, 3)
    z = np.sort(np.unique(pos[:, 2]))
    np.testing.assert_allclose(z, [0.0, 1.755, 3.51, 5.265], atol=1e-12)
    # ABAB registry, centered in the cell by a uniform (a/4, a/4) shift:
    # even layers on the a/4 grid, odd layers shifted by an extra a/2
    layer0 = pos[pos[:, 2] == 0.0]
    layer1 = pos[pos[:, 2] == 1.755]
    assert set(np.round(layer0[:, 0], 6)) == {0.8775, 4.3875, 7.8975}
    assert set(np.round(layer1[:, 0], 6)) == {2.6325, 6.1425, 9.6525}
    # atom columns keep a/4 clearance to both lateral cell faces
    assert layer0[:, 0].min() == pytest.approx(plan.li_a / 4)
    assert 3 * plan.li_a - layer1[:, 0].max() == pytest.approx(plan.li_a / 4)
    # bcc nearest neighbor = a*sqrt(3)/2
    cell = np.array([10.53, 10.53, 50.0])
    d = bib.minimum_image_distances(pos, cell)
    np.fill_diagonal(d, np.inf)
    assert d.min() == pytest.approx(3.51 * np.sqrt(3) / 2, abs=1e-9)


# --- embedded template sanity --------------------------------------------------


def test_template_bond_lengths_sane():
    bonds = bib.template_bonds()
    ec_co = bonds["ec"]["C-O"]
    assert min(ec_co) == pytest.approx(1.2581, abs=1e-3)  # UFF C=O
    assert max(ec_co) < 1.50  # ring C-O
    assert max(bonds["ec"]["C-H"]) < 1.15
    assert min(bonds["dmc"]["C-O"]) == pytest.approx(1.2626, abs=1e-3)
    assert bonds["pf6"]["F-P"] == [1.6741]
    # no crushed pairs anywhere
    for _, pos in bib.TEMPLATES.values():
        if len(pos) == 1:
            continue
        d = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() > bib.MIN_NN_INTRAMOLECULAR_A


# --- overlap checker -----------------------------------------------------------


def _two_molecule_positions(dz: float, box: float = 30.0):
    symbols, pos = bib.TEMPLATES["ec"]
    p1 = np.array(pos) + np.array([10.0, 10.0, 10.0])
    p2 = np.array(pos) + np.array([10.0, 10.0, 10.0 + dz])
    all_pos = np.vstack([p1, p2])
    mol_ids = np.array([0] * len(symbols) + [1] * len(symbols))
    cell = np.array([box, box, box])
    return all_pos, mol_ids, cell


def test_overlap_check_clean():
    pos, mol_ids, cell = _two_molecule_positions(dz=8.0)
    result = bib.check_overlaps(pos, cell, mol_ids)
    assert result["n_pairs_below_cutoff"] == 0
    assert result["min_intermolecular_A"] > bib.MIN_INTERMOLECULAR_A


def test_overlap_check_flags_clash():
    pos, mol_ids, cell = _two_molecule_positions(dz=1.0)
    result = bib.check_overlaps(pos, cell, mol_ids)
    assert result["n_pairs_below_cutoff"] > 0
    assert result["min_intermolecular_A"] < bib.MIN_INTERMOLECULAR_A


def test_overlap_check_pbc_wrap():
    # Two atoms close only through the periodic z boundary must be flagged.
    pos = np.array([[5.0, 5.0, 0.2], [5.0, 5.0, 29.8]])
    mol_ids = np.array([0, 1])
    cell = np.array([30.0, 30.0, 30.0])
    result = bib.check_overlaps(pos, cell, mol_ids)
    assert result["min_intermolecular_A"] == pytest.approx(0.4, abs=1e-9)
    assert result["n_pairs_below_cutoff"] == 1
    # same-molecule pairs are never flagged
    result_same = bib.check_overlaps(pos, cell, np.array([0, 0]))
    assert result_same["n_pairs_below_cutoff"] == 0


# --- packmol input rendering ---------------------------------------------------


def test_render_packmol_input(plan):
    text = bib.render_packmol_input(plan)
    assert f"tolerance {plan.tolerance:.1f}" in text
    assert f"seed {plan.seed}" in text
    assert "fixed 0. 0. 0. 0. 0. 0." in text
    g = plan.geometry
    m = plan.pack_margin
    # confinement boxes inset by pack_margin from every periodic cell face
    assert (
        f"inside box {m:.6f} {m:.6f} {m:.6f} "
        f"{g.lx - m:.6f} {g.ly - m:.6f} {g.t_side:.6f}" in text
    )
    assert (
        f"inside box {m:.6f} {m:.6f} {g.lz - g.t_side:.6f} "
        f"{g.lx - m:.6f} {g.ly - m:.6f} {g.lz - m:.6f}" in text
    )
    # molecule counts per block: alternating sides keep the mass balance
    numbers = [line.split()[-1] for line in text.splitlines() if line.startswith("  number")]
    assert [int(n) for n in numbers] == [1, 11, 10, 8, 9, 2, 1, 1, 2]


def test_molecule_block_map_and_side_balance(plan):
    mol_ids, _regions = bib.molecule_block_map(plan)
    assert len(mol_ids) == 474
    assert (mol_ids == 0).sum() == 36
    assert mol_ids.max() == 44  # 44 solvent molecules (21+17+3 Li+ +3 PF6-)
    symbols = bib.expected_symbols(plan)
    assert len(symbols) == 474
    assert symbols.count("Li") == 39
    mass = {"bottom": 0.0, "top": 0.0}
    for species, count, region in plan.region_split:
        m = count * sum(bib.ATOMIC_MASSES[e] * k for e, k in bib.FORMULAS[species].items())
        mass[region] += m
    imbalance = abs(mass["bottom"] - mass["top"]) / sum(mass.values())
    assert imbalance < bib.MAX_SIDE_MASS_IMBALANCE


# --- validation gate on a synthetic packing ------------------------------------


def _synthetic_packing(plan: bib.BuildPlan):
    """Place each molecule, unrotated, well separated along z inside its
    region, in the exact atom order of molecule_block_map (region_split
    order); place the slab at its design position. Returns validate() inputs."""
    g = plan.geometry
    symbols = ["Li"] * 36
    positions = [p for p in bib.slab_positions(plan.li_a, plan.surface, plan.layers, g.z_slab_bottom)]
    n_side = {
        r: sum(c for _, c, reg in plan.region_split if reg == r) for r in ("bottom", "top")
    }
    seen = {"bottom": 0, "top": 0}
    for species, count, region in plan.region_split:
        if count == 0:
            continue
        z0, z1 = g.region_bottom if region == "bottom" else g.region_top
        zs = np.linspace(z0 + 8.0, z1 - 8.0, n_side[region])
        s_sym, s_pos = bib.TEMPLATES[species]
        for _ in range(count):
            z = zs[seen[region]]
            seen[region] += 1
            symbols += s_sym
            positions.extend(np.array(s_pos) + np.array([g.lx / 2, g.ly / 2, z]))
    return symbols, np.array(positions)


def test_validate_passes_on_clean_synthetic_packing():
    # Big lz so hand-placed molecules are far apart; density target matched to
    # the resulting geometry so the density check is not the point here.
    # Even, salt-free composition keeps the two sides exactly mass-balanced.
    plan = bib.make_plan(n_ec=2, n_dmc=2, n_lipf6=0, lz=72.0)
    g = plan.geometry
    mass = bib.electrolyte_mass_amu(plan.composition)
    plan.density_target = bib.liquid_density_g_cm3(mass, g.liquid_volume_a3)
    symbols, positions = _synthetic_packing(plan)
    mol_ids, regions = bib.molecule_block_map(plan)
    result = bib.validate(symbols, positions, plan, mol_ids, regions)
    assert result["failures"] == []
    assert result["pass"] is True
    assert result["n_atoms"] == 36 + 2 * 10 + 2 * 12


def test_validate_fails_on_overlap():
    plan = bib.make_plan(n_ec=2, n_dmc=2, n_lipf6=0, lz=72.0)
    mass = bib.electrolyte_mass_amu(plan.composition)
    plan.density_target = bib.liquid_density_g_cm3(mass, plan.geometry.liquid_volume_a3)
    symbols, positions = _synthetic_packing(plan)
    positions[-1] = positions[-2] + 0.5  # crush the last two atoms together
    mol_ids, regions = bib.molecule_block_map(plan)
    result = bib.validate(symbols, positions, plan, mol_ids, regions)
    assert result["pass"] is False
    assert any("intermolecular" in f or "crushed" in f for f in result["failures"])
