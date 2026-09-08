"""Quantum ESPRESSO through ASE's Espresso calculator: the ASE-native QE path.

Programs ASE can drive should be reached through a small factory rather than
another handwritten subprocess layer. This module builds an
``ase.calculators.espresso.Espresso`` calculator from the same :class:`QeConfig`
the handwritten :class:`~pyraimd2.engines.qe_engine.QeEngine` uses, so the two
QE paths share one scientific-settings source (XC, dispersion, cutoffs,
k-points, smearing, pseudopotentials).

Result conventions are identical to the handwritten path by construction —
energy in eV, forces in eV/Å, stress as a 6-component ASE Voigt vector with
the ASE (compression-negative) sign — and are verified against each other on
a real pw.x fixture in tests/unit/test_ase_qe.py. ASE's espresso reader
converts Rydbergs with CODATA-2006 constants while the handwritten parser
uses the current ``ase.units`` defaults, so energies can differ at the 1e-7
relative level; that scale difference is documented, not silently absorbed.

Interface note: ``EspressoProfile(command, pseudo_dir)`` and
``Espresso(profile=..., directory=..., input_data=..., pseudopotentials=...,
kpts=...)`` follow the ASE version locked in uv.lock (3.29), verified by
tests — not copied from older blog examples. ASE takes the launch command as
a single string for its own FileIO machinery; our authoritative runner
(:class:`QeEngine`) keeps the command as an argv array and never goes through
a shell.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile

from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.engines.base import EnergyKind, EngineCapabilities
from pyraimd2.engines.qe_engine import (
    QE_PREFIX,
    QeConfig,
    _settings_digest,
    recipe_name,
)


def espresso_input_data(config: QeConfig) -> dict:
    """Map a QeConfig onto ASE Espresso ``input_data`` namelists.

    This is the same physics mapping as
    :func:`~pyraimd2.engines.qe_engine.write_qe_input`: both paths must
    produce equivalent pw.x inputs for the same config.
    """
    control = {
        "calculation": "scf",
        "prefix": QE_PREFIX,
        "pseudo_dir": config.pseudo_dir,
        "outdir": "./tmp",
        "tprnfor": True,
        "tstress": True,
    }
    system: dict = {
        "ibrav": 0,
        "ecutwfc": config.ecutwfc,
        "ecutrho": config.ecutrho,
        "input_dft": config.xc,
    }
    if config.dispersion is not None:
        system["vdw_corr"] = config.dispersion
    if config.nbnd is not None:
        system["nbnd"] = config.nbnd
    if config.metallic:
        system.update(
            occupations="smearing",
            smearing=config.smearing,
            degauss=config.degauss,
        )
    electrons: dict = {
        "conv_thr": config.conv_thr,
        "mixing_beta": config.mixing_beta,
        "mixing_mode": config.mixing_mode,
        "mixing_ndim": config.mixing_ndim,
        "diago_david_ndim": config.diago_david_ndim,
        "electron_maxstep": config.electron_maxstep,
    }
    if config.diago_full_acc:
        electrons["diago_full_acc"] = True
    if config.startpot_file:
        electrons["startingpot"] = "file"
    return {"control": control, "system": system, "electrons": electrons}


def make_espresso_calculator(config: QeConfig, *, directory: str | Path,
                             command: str | None = None) -> Espresso:
    """Construct ASE's Espresso calculator for ``config``.

    ``command`` overrides the launch string (ASE's FileIO layer takes one
    string); the default joins the config's argv. The calculator writes its
    input as ``<directory>/espresso.pwi`` when ASE calculates.
    """
    profile = EspressoProfile(
        command=command if command is not None else shlex.join(config.pw_cmd),
        pseudo_dir=config.pseudo_dir,
    )
    return Espresso(
        profile=profile,
        directory=str(directory),
        input_data=espresso_input_data(config),
        pseudopotentials=dict(config.pseudos),
        kpts=config.kpts,
    )


class AseQeEngine(AseEngine):
    """Engine-protocol QE reference backed by ASE's Espresso calculator.

    Same Engine contract as :class:`QeEngine`: each compute runs in its own
    ``run_root/<label>-NNNNNN`` directory, capabilities state the energy
    convention honestly (smearing → free energy), and the fingerprint covers
    the full reference settings including pseudopotential content hashes,
    under the ``ase-qe-...`` name so labels record which path produced them.
    """

    def __init__(self, config: QeConfig, run_root: str | Path, *,
                 command: str | None = None) -> None:
        self.config = config
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._call_counter = 0
        self._command = command
        calculator = make_espresso_calculator(
            config, directory=self.run_root / "pending", command=command
        )
        super().__init__(calculator, include_stress=True)

    @property
    def name(self) -> str:
        return f"ase-{recipe_name(self.config)}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            energy_kind=EnergyKind.FREE_ENERGY if self.config.metallic else EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=True,
        )

    @property
    def fingerprint(self) -> str:
        digest = _settings_digest(self.config, path_kind="ase-espresso")
        return f"{self.name}:{digest}"

    def fresh_directory(self, label: str | None = None) -> Path:
        """The next unique per-call directory (consecutive calls never share one)."""
        base = "eval" if label is None else str(label).replace("/", "_")
        directory = self.run_root / f"{base}-{self._call_counter:06d}"
        self._call_counter += 1
        return directory

    def compute(self, atoms: Atoms, label: str | None = None):
        directory = self.fresh_directory(label)
        directory.mkdir(parents=True, exist_ok=False)
        self.calculator.directory = directory
        return super().compute(atoms)


def create_ase_qe_engine(*, run_root: str | Path, command: str | None = None,
                         **config_kwargs) -> AseQeEngine:
    """Registry factory: build an AseQeEngine from plain keyword settings."""
    return AseQeEngine(QeConfig(**config_kwargs), run_root, command=command)
