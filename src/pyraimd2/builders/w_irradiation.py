"""Builders for tungsten irradiation initial configurations.

Single implementation behind the CLI scripts in ``hpc/neimeng/scripts/``:

- :func:`build_spike_config` — the two-temperature thermal-spike surrogate of
  the cascade core (naming discipline: never "cascade simulation"). The draw
  sequence is exactly the legacy ``make_w_spike.py`` one, so identical seeds
  reproduce seeded configurations bitwise.
- :func:`build_pka_config` — PKA initialization for self-proof (a)
  (PKA-vs-spike validation): one central W atom kicked along a cubic
  direction family (<100>/<110>/<111>) on top of a 300 K Maxwell-Boltzmann
  matrix, with the total momentum re-zeroed after the kick.
- :func:`build_spike_series_configs` — self-proof (b): finite-size
  convergence series (4^3/5^3/7^3 = 128/250/686 atoms) at a unified core
  energy density; the core temperature is rescaled per size.
- :func:`build_ed_scan_configs` — the <100> E_d (threshold displacement
  energy) scan grid, 40-140 eV in 10 eV steps; configurations and sidecars
  only, nothing is submitted.

Every random draw comes from a seeded ``numpy.random.Generator`` and every
parameter is recorded in a JSON sidecar (restart-complete provenance).
ASE units throughout: energies in eV, lengths in Angstrom.

PKA kinetic-energy ledger (all recorded in the sidecar). The kick adds
momentum ``p_kick`` (|p_kick|^2 / 2m = ``pka_e_kick_eV``) to the PKA atom's
thermal momentum, then ``Stationary(..., preserve_temperature=False)``
removes the center-of-mass motion without rescaling, so::

    ke_total_eV = ke_matrix_eV + pka_e_kick_eV + kick_thermal_cross_eV
                  - ke_com_removed_eV

with ``kick_thermal_cross_eV = p_kick . v_pka_thermal`` (vanishes in
expectation, O(sqrt(E_kick * kB * T)) for a single draw) and
``ke_com_removed_eV = |P|^2 / (2 M)``, which equals ``E_kick / N`` here
because the matrix is thermalized stationary and all masses are equal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import units
from ase.atoms import Atoms
from ase.build import bulk
from ase.io import write
from ase.md.velocitydistribution import Stationary, thermalize_momenta

A0_W = 3.165  # bcc W lattice constant, Angstrom (rounded from 3.1648, CRC)

# Canonical members of the cubic direction families offered on the CLI.
PKA_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "100": (1.0, 0.0, 0.0),
    "110": (1.0, 1.0, 0.0),
    "111": (1.0, 1.0, 1.0),
}


@dataclass
class BuiltConfig:
    """One built initial condition: the atoms plus their provenance sidecar."""

    atoms: Atoms
    sidecar: dict


def pka_unit_vector(family: str) -> np.ndarray:
    """Unit vector along a cubic direction family ("100"/"110"/"111").

    Brackets are tolerated ("<110>", "[100]"). Cubic symmetry makes every
    member of a family equivalent for the kick, so the canonical positive
    member is used.
    """
    key = family.strip().strip("<>[]")
    if key not in PKA_DIRECTIONS:
        raise ValueError(
            f"unknown PKA direction {family!r}; choose from {sorted(PKA_DIRECTIONS)}"
        )
    vec = np.array(PKA_DIRECTIONS[key], dtype=float)
    return vec / np.linalg.norm(vec)


def build_w_supercell(cells: int, a0: float = A0_W) -> Atoms:
    """Centered bcc W cubic supercell of ``cells``^3 unit cells (2*cells^3 atoms)."""
    atoms = bulk("W", "bcc", a=a0, cubic=True).repeat((cells, cells, cells))
    atoms.center()
    return atoms


def core_mask(atoms: Atoms, core_radius_a: float) -> np.ndarray:
    """Boolean mask of atoms within ``core_radius_a`` of the box center."""
    center = np.diag(atoms.cell) / 2.0
    r = np.linalg.norm(atoms.positions - center, axis=1)
    return r <= core_radius_a


def central_atom_index(atoms: Atoms) -> int:
    """Index of the atom nearest the box center (the PKA atom)."""
    center = np.diag(atoms.cell) / 2.0
    return int(np.argmin(np.linalg.norm(atoms.positions - center, axis=1)))


def apply_two_temperature_spike(
    atoms: Atoms,
    core_radius_a: float,
    core_T_K: float,
    matrix_T_K: float,
    rng: np.random.Generator,
) -> int:
    """Two-temperature spike init in place; returns the core atom count.

    Exactly the legacy ``make_w_spike.py`` draw sequence (thermalize the
    matrix, redraw the core from ``rng``, zero the core net momentum, then
    ``Stationary`` on the whole box) so seeds stay reproducible across the
    refactor.
    """
    core = core_mask(atoms, core_radius_a)
    thermalize_momenta(atoms, temperature_K=matrix_T_K, rng=rng)
    masses = atoms.get_masses()[core, None]
    kT = units.kB * core_T_K  # eV
    v_core = rng.normal(0.0, np.sqrt(kT / masses), size=(int(core.sum()), 3))
    v_core -= v_core.mean(axis=0)  # zero net momentum inside the core
    velocities = atoms.get_velocities()
    velocities[core] = v_core
    atoms.set_velocities(velocities)
    Stationary(atoms)  # no net drift after the core swap
    return int(core.sum())


def core_temperature_for_energy_density(n_core: int, energy_density_eV: float) -> float:
    """Core temperature depositing ``energy_density_eV`` per core atom.

    The core draw is momentum-zeroed, so its expected kinetic energy is
    3/2 (N_core - 1) kB T; solving for T makes the expected deposited
    energy density exactly ``energy_density_eV`` per core atom at any size.
    """
    if n_core < 2:
        raise ValueError(f"need at least 2 core atoms, got {n_core}")
    return 2.0 * energy_density_eV * n_core / (3.0 * (n_core - 1) * units.kB)


def build_spike_config(
    cells: int,
    core_radius_a: float,
    core_T_K: float,
    matrix_T_K: float,
    seed: int,
    a0: float = A0_W,
) -> BuiltConfig:
    """One thermal-spike configuration with the legacy sidecar field set."""
    atoms = build_w_supercell(cells, a0)
    rng = np.random.default_rng(seed)
    n_core = apply_two_temperature_spike(atoms, core_radius_a, core_T_K, matrix_T_K, rng)
    sidecar = {
        "cells": cells,
        "n_atoms": len(atoms),
        "a0": a0,
        "core_radius_A": core_radius_a,
        "core_T_K": core_T_K,
        "matrix_T_K": matrix_T_K,
        "n_core": n_core,
        "seed": seed,
        "note": "two-temperature spike init; numbers pending literature anchoring",
    }
    return BuiltConfig(atoms, sidecar)


def build_spike_series_configs(
    cells_list: tuple[int, ...] = (4, 5, 7),
    core_radius_a: float = 5.5,
    energy_density_eV: float = 1.0,
    matrix_T_K: float = 300.0,
    seed: int = 1,
    a0: float = A0_W,
) -> list[BuiltConfig]:
    """Finite-size convergence series at a unified core energy density.

    Same seed for every size (documented in the sidecar); the core
    temperature is rescaled per size via
    :func:`core_temperature_for_energy_density` so the expected deposited
    energy per core atom is identical across the series.
    """
    configs = []
    for cells in cells_list:
        atoms = build_w_supercell(cells, a0)
        n_core = int(core_mask(atoms, core_radius_a).sum())
        core_T_K = core_temperature_for_energy_density(n_core, energy_density_eV)
        cfg = build_spike_config(
            cells=cells,
            core_radius_a=core_radius_a,
            core_T_K=core_T_K,
            matrix_T_K=matrix_T_K,
            seed=seed,
            a0=a0,
        )
        mask = core_mask(cfg.atoms, core_radius_a)
        core_ke = float(
            0.5
            * (
                cfg.atoms.get_momenta()[mask] ** 2
                / cfg.atoms.get_masses()[mask, None]
            ).sum()
        )
        cfg.sidecar.update(
            {
                "experiment": "finite_size_convergence",
                "energy_density_eV_per_core_atom": energy_density_eV,
                "expected_core_ke_eV": energy_density_eV * n_core,
                "realized_core_ke_eV": core_ke,
                "note": (
                    "finite-size convergence spike (self-proof b); core T rescaled "
                    "per size so the expected deposited energy density per core "
                    "atom is identical (momentum-zeroing DOF corrected)"
                ),
            }
        )
        configs.append(cfg)
    return configs


def apply_pka_kick(
    atoms: Atoms,
    direction: str,
    e_kick_eV: float,
    pka_index: int | None = None,
) -> dict:
    """Kick one atom along ``direction`` with ``e_kick_eV`` kinetic energy.

    Adds the kick momentum on top of the atom's thermal momentum, then
    re-zeroes the total momentum without rescaling
    (``Stationary(..., preserve_temperature=False)``) so the kinetic-energy
    ledger closes exactly — see the module docstring. Returns the ledger.
    """
    if e_kick_eV <= 0.0:
        raise ValueError(f"e_kick_eV must be positive, got {e_kick_eV}")
    if pka_index is None:
        pka_index = central_atom_index(atoms)
    unit = pka_unit_vector(direction)
    masses = atoms.get_masses()
    m_pka = float(masses[pka_index])
    total_mass = float(masses.sum())

    ke_before = float(atoms.get_kinetic_energy())
    momenta = atoms.get_momenta()
    p_kick = np.sqrt(2.0 * m_pka * e_kick_eV) * unit
    cross_eV = float(p_kick @ momenta[pka_index] / m_pka)  # p_kick . v_thermal
    momenta[pka_index] += p_kick
    atoms.set_momenta(momenta)

    total_p = atoms.get_momenta().sum(axis=0)
    com_removed_eV = float(total_p @ total_p / (2.0 * total_mass))
    Stationary(atoms, preserve_temperature=False)  # drop COM motion, no rescale

    ke_after = float(atoms.get_kinetic_energy())
    p_pka = atoms.get_momenta()[pka_index]
    return {
        "pka_index": int(pka_index),
        "pka_direction": direction.strip().strip("<>[]"),
        "pka_direction_unit": unit.tolist(),
        "pka_e_kick_eV": e_kick_eV,
        "ke_matrix_eV": ke_before,
        "kick_thermal_cross_eV": cross_eV,
        "ke_com_removed_eV": com_removed_eV,
        "ke_total_eV": ke_after,
        "ke_deposited_eV": ke_after - ke_before,
        "pka_ke_final_eV": float(0.5 * (p_pka @ p_pka) / m_pka),
    }


def build_pka_config(
    cells: int,
    direction: str,
    e_kick_eV: float,
    matrix_T_K: float,
    seed: int,
    a0: float = A0_W,
    pka_index: int | None = None,
) -> BuiltConfig:
    """One PKA configuration: 300 K stationary matrix + a central-atom kick."""
    atoms = build_w_supercell(cells, a0)
    rng = np.random.default_rng(seed)
    thermalize_momenta(atoms, temperature_K=matrix_T_K, rng=rng)
    Stationary(atoms)  # zero thermal COM drift before the kick
    ledger = apply_pka_kick(atoms, direction, e_kick_eV, pka_index=pka_index)
    sidecar = {
        "experiment": "pka",
        "cells": cells,
        "n_atoms": len(atoms),
        "a0": a0,
        "matrix_T_K": matrix_T_K,
        "seed": seed,
        **ledger,
        "note": (
            "PKA init (self-proof a); kick added to the thermal velocity, "
            "total momentum re-zeroed without rescaling — deposited = assigned "
            "+ thermal cross - COM removal"
        ),
    }
    return BuiltConfig(atoms, sidecar)


def ed_scan_energies(
    e_min: float = 40.0, e_max: float = 140.0, e_step: float = 10.0
) -> list[float]:
    """The E_d scan grid: 40-140 eV inclusive in 10 eV steps by default."""
    if e_step <= 0.0 or e_max < e_min:
        raise ValueError(f"bad scan grid: min={e_min}, max={e_max}, step={e_step}")
    n = round((e_max - e_min) / e_step) + 1
    return [e_min + i * e_step for i in range(n)]


def build_ed_scan_configs(
    direction: str = "100",
    energies: list[float] | None = None,
    cells: int = 7,
    matrix_T_K: float = 300.0,
    seed: int = 1,
    a0: float = A0_W,
) -> list[BuiltConfig]:
    """The directional E_d scan: one PKA configuration per energy point.

    Configurations and sidecars only — nothing is submitted. Point ``i``
    uses ``seed + i`` so the thermal draws are independent across the grid.
    """
    if energies is None:
        energies = ed_scan_energies()
    configs = []
    for i, energy in enumerate(energies):
        cfg = build_pka_config(
            cells=cells,
            direction=direction,
            e_kick_eV=energy,
            matrix_T_K=matrix_T_K,
            seed=seed + i,
            a0=a0,
        )
        cfg.sidecar.update(
            {
                "experiment": "ed_scan",
                "scan_point": i,
                "note": (
                    "E_d scan initial condition; subsequent dynamics and "
                    "defect analysis required; configuration only, not submitted"
                ),
            }
        )
        configs.append(cfg)
    return configs


def write_config(cfg: BuiltConfig, out: Path) -> tuple[Path, Path]:
    """Write the XYZ (with momenta) plus the JSON sidecar; return both paths."""
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, cfg.atoms)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(cfg.sidecar, indent=2))
    return out, sidecar
