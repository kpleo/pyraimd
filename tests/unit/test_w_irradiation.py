"""Tests for the tungsten irradiation builders.

ASE and NumPy tests cover PKA kinetic-energy bookkeeping, size-independent
energy-density conversion, seeded determinism and complete JSON sidecars.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import units
from ase.io import read

from pyraimd2.builders.w_irradiation import (
    build_ed_scan_configs,
    build_pka_config,
    build_spike_config,
    build_spike_series_configs,
    build_w_supercell,
    core_mask,
    core_temperature_for_energy_density,
    ed_scan_energies,
    pka_unit_vector,
    write_config,
)

# Fields every PKA sidecar must carry: the make_w_spike provenance
# conventions plus the mandated PKA ledger (atom index, direction, assigned
# kick energy, realized deposit after the COM-momentum removal).
PKA_MANDATED_FIELDS = {
    "experiment",
    "cells",
    "n_atoms",
    "a0",
    "matrix_T_K",
    "seed",
    "pka_index",
    "pka_direction",
    "pka_direction_unit",
    "pka_e_kick_eV",
    "ke_matrix_eV",
    "kick_thermal_cross_eV",
    "ke_com_removed_eV",
    "ke_total_eV",
    "ke_deposited_eV",
    "pka_ke_final_eV",
    "note",
}

# Legacy spike sidecar set (kept identical to make_w_spike.py) plus the
# finite-size series extras.
SPIKE_MANDATED_FIELDS = {
    "cells",
    "n_atoms",
    "a0",
    "core_radius_A",
    "core_T_K",
    "matrix_T_K",
    "n_core",
    "seed",
    "note",
}
SERIES_MANDATED_FIELDS = SPIKE_MANDATED_FIELDS | {
    "experiment",
    "energy_density_eV_per_core_atom",
    "expected_core_ke_eV",
    "realized_core_ke_eV",
}


def test_pka_kick_kinetic_energy_ledger() -> None:
    cfg = build_pka_config(
        cells=3, direction="110", e_kick_eV=250.0, matrix_T_K=300.0, seed=7
    )
    atoms, s = cfg.atoms, cfg.sidecar
    n = len(atoms)
    # Total momentum is re-zeroed after the kick.
    assert np.linalg.norm(atoms.get_momenta().sum(axis=0)) < 1e-10
    # The sidecar totals are the actual totals.
    assert s["ke_total_eV"] == pytest.approx(atoms.get_kinetic_energy(), abs=1e-12)
    # The ledger closes exactly: deposit = assigned + thermal cross - COM removal.
    delta = s["ke_total_eV"] - s["ke_matrix_eV"]
    assert s["ke_deposited_eV"] == pytest.approx(delta, abs=1e-12)
    ledger = s["pka_e_kick_eV"] + s["kick_thermal_cross_eV"] - s["ke_com_removed_eV"]
    assert delta == pytest.approx(ledger, abs=1e-9)
    # The COM correction is the kick momentum over the total mass: E_kick/N.
    assert s["ke_com_removed_eV"] == pytest.approx(s["pka_e_kick_eV"] / n, rel=1e-9)
    # The realized deposit matches the assigned 250 eV to well within an eV
    # (the thermal cross term is O(sqrt(E_kick * kB * T))).
    assert abs(delta - s["pka_e_kick_eV"]) < 5.0
    # The PKA atom carries essentially the whole deposit.
    p_pka = atoms.get_momenta()[s["pka_index"]]
    m_pka = atoms.get_masses()[s["pka_index"]]
    assert s["pka_ke_final_eV"] == pytest.approx(0.5 * float(p_pka @ p_pka) / m_pka, abs=1e-12)
    assert s["pka_ke_final_eV"] > 0.9 * s["pka_e_kick_eV"]


def test_core_temperature_unifies_energy_density() -> None:
    target = 1.0  # eV per core atom
    for cells, n_atoms in [(4, 128), (5, 250), (7, 686)]:
        atoms = build_w_supercell(cells)
        assert len(atoms) == n_atoms
        n_core = int(core_mask(atoms, 5.5).sum())
        core_T = core_temperature_for_energy_density(n_core, target)
        # The core draw is momentum-zeroed: expected KE = 3/2 (N-1) kB T.
        density = 1.5 * (n_core - 1) * units.kB * core_T / n_core
        assert density == pytest.approx(target, rel=1e-12)


def test_spike_series_records_rescaled_core_temperatures() -> None:
    configs = build_spike_series_configs(seed=3)
    assert [c.sidecar["n_atoms"] for c in configs] == [128, 250, 686]
    for cfg in configs:
        s = cfg.sidecar
        n_core = s["n_core"]
        # The recorded core temperature reproduces the mandated density.
        density = 1.5 * (n_core - 1) * units.kB * s["core_T_K"] / n_core
        assert density == pytest.approx(s["energy_density_eV_per_core_atom"], rel=1e-12)
        assert s["expected_core_ke_eV"] == pytest.approx(
            s["energy_density_eV_per_core_atom"] * n_core
        )
        # The realized core KE matches the atoms and the expectation.
        mask = core_mask(cfg.atoms, s["core_radius_A"])
        realized = float(
            0.5
            * (cfg.atoms.get_momenta()[mask] ** 2 / cfg.atoms.get_masses()[mask, None]).sum()
        )
        assert s["realized_core_ke_eV"] == pytest.approx(realized, abs=1e-12)
        assert realized == pytest.approx(s["expected_core_ke_eV"], rel=0.2)


def test_pka_same_seed_reproduces_bitwise(tmp_path) -> None:
    kwargs = {"cells": 4, "direction": "111", "e_kick_eV": 220.0,
              "matrix_T_K": 300.0, "seed": 42}
    first = build_pka_config(**kwargs)
    second = build_pka_config(**kwargs)
    assert np.array_equal(first.atoms.get_positions(), second.atoms.get_positions())
    assert np.array_equal(first.atoms.get_momenta(), second.atoms.get_momenta())
    path_a, _ = write_config(first, tmp_path / "a.xyz")
    path_b, _ = write_config(second, tmp_path / "b.xyz")
    assert path_a.read_bytes() == path_b.read_bytes()
    other = build_pka_config(**{**kwargs, "seed": 43})
    assert not np.array_equal(first.atoms.get_momenta(), other.atoms.get_momenta())


def test_spike_same_seed_reproduces_bitwise() -> None:
    kwargs = {"cells": 4, "core_radius_a": 5.5, "core_T_K": 8000.0,
              "matrix_T_K": 300.0, "seed": 5}
    first = build_spike_config(**kwargs)
    second = build_spike_config(**kwargs)
    assert np.array_equal(first.atoms.get_momenta(), second.atoms.get_momenta())
    assert first.sidecar == second.sidecar


def test_xyz_roundtrip_preserves_momenta(tmp_path) -> None:
    cfg = build_pka_config(
        cells=3, direction="100", e_kick_eV=100.0, matrix_T_K=300.0, seed=1
    )
    out, _ = write_config(cfg, tmp_path / "pka.xyz")
    back = read(out)
    assert np.allclose(back.get_positions(), cfg.atoms.get_positions(), atol=1e-7)
    assert np.allclose(back.get_momenta(), cfg.atoms.get_momenta(), atol=1e-7)


def test_pka_sidecar_mandated_fields(tmp_path) -> None:
    cfg = build_pka_config(
        cells=3, direction="100", e_kick_eV=200.0, matrix_T_K=300.0, seed=11
    )
    _, sidecar_path = write_config(cfg, tmp_path / "pka.xyz")
    s = json.loads(sidecar_path.read_text())
    assert PKA_MANDATED_FIELDS <= s.keys()
    assert s["n_atoms"] == len(cfg.atoms)
    assert 0 <= s["pka_index"] < s["n_atoms"]
    assert s["pka_direction"] == "100"
    assert np.linalg.norm(s["pka_direction_unit"]) == pytest.approx(1.0)
    # The serialized ledger closes on its own.
    assert s["ke_deposited_eV"] == pytest.approx(
        s["pka_e_kick_eV"] + s["kick_thermal_cross_eV"] - s["ke_com_removed_eV"],
        abs=1e-9,
    )


def test_series_sidecar_mandated_fields() -> None:
    for cfg in build_spike_series_configs(seed=2):
        assert SERIES_MANDATED_FIELDS <= cfg.sidecar.keys()


def test_ed_scan_grid_and_configs() -> None:
    assert ed_scan_energies() == [40.0 + 10.0 * i for i in range(11)]
    configs = build_ed_scan_configs(energies=[40.0, 50.0], cells=3, seed=4)
    assert [c.sidecar["pka_e_kick_eV"] for c in configs] == [40.0, 50.0]
    assert [c.sidecar["seed"] for c in configs] == [4, 5]  # independent draws
    for cfg in configs:
        assert PKA_MANDATED_FIELDS <= cfg.sidecar.keys()
        assert cfg.sidecar["experiment"] == "ed_scan"
        assert cfg.sidecar["pka_direction"] == "100"
    assert not np.array_equal(
        configs[0].atoms.get_momenta(), configs[1].atoms.get_momenta()
    )


def test_direction_parsing_and_validation() -> None:
    assert np.allclose(pka_unit_vector("100"), [1.0, 0.0, 0.0])
    assert np.allclose(pka_unit_vector("<110>"), [2**-0.5, 2**-0.5, 0.0])
    assert np.allclose(pka_unit_vector("[111]"), [3**-0.5, 3**-0.5, 3**-0.5])
    with pytest.raises(ValueError, match="unknown PKA direction"):
        pka_unit_vector("211")
