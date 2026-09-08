"""ASE Espresso path: construction/input-level factory tests plus
result-convention consistency with the handwritten QE parser.

No pw.x exists on this machine, so the ASE calculator is exercised up to
construction and input writing — the interface signature is locked to the
ASE version in uv.lock (3.29: ``EspressoProfile(command, pseudo_dir)``,
``Espresso(profile=..., directory=..., input_data=..., pseudopotentials=...,
kpts=...)``). Convention consistency is checked at the parser/units level:
the same pw.x output parsed by ASE's espresso reader and by the handwritten
parser must agree on energy/force units and signs and on stress Voigt order
and sign. ASE's reader converts Rydbergs with CODATA-2006 constants while
the handwritten parser uses current ase.units, so magnitudes agree to ~1e-7
relative — a documented scale difference, not a convention difference.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io.espresso import read_espresso_out

from pyraimd2.engines.ase_qe import (
    AseQeEngine,
    espresso_input_data,
    make_espresso_calculator,
)
from pyraimd2.engines.base import EnergyKind
from pyraimd2.engines.qe_engine import (
    QeConfig,
    QeEngine,
    parse_qe_output,
    write_qe_input,
)

FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"


def _si() -> Atoms:
    return Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                 cell=[5.43] * 3, pbc=True)


def test_espresso_interface_signature_is_as_locked() -> None:
    """Guard the ASE-version interface this factory is written against."""
    profile_params = list(inspect.signature(EspressoProfile.__init__).parameters)
    assert profile_params[:3] == ["self", "command", "pseudo_dir"]
    espresso_params = list(inspect.signature(Espresso.__init__).parameters)
    for name in ("profile", "directory", "kwargs"):
        assert name in espresso_params


def test_input_data_maps_the_same_physics_as_the_handwritten_writer(
    tmp_path: Path,
) -> None:
    cfg = QeConfig(pseudo_dir="/pseudo", ecutwfc=60.0, ecutrho=480.0,
                   kpts=(2, 2, 2), nbnd=16, startpot_file=True)
    data = espresso_input_data(cfg)
    assert data["system"]["input_dft"] == "pbe"
    assert data["system"]["vdw_corr"] == "grimme-d3"
    assert data["system"]["ecutwfc"] == 60.0 and data["system"]["ecutrho"] == 480.0
    assert data["system"]["nbnd"] == 16
    assert data["electrons"]["startingpot"] == "file"
    assert data["electrons"]["conv_thr"] == cfg.conv_thr

    # The handwritten writer and the ASE calculator must put the same
    # scientific settings into the pw.x input.
    calc = make_espresso_calculator(cfg, directory=tmp_path)
    calc.write_inputfiles(_si(), properties=["energy"])
    ase_text = next(tmp_path.glob("*.pwi")).read_text()
    ours = tmp_path / "pw.in"
    write_qe_input(ours, _si(), cfg)
    our_text = ours.read_text()
    for needle in (
        "input_dft", "pbe", "grimme-d3", "60.0", "480.0",
        "Si.pbe-n-kjpaw_psl.1.0.0.UPF", "CELL_PARAMETERS", "ATOMIC_POSITIONS",
    ):
        assert needle in ase_text, needle
        assert needle in our_text, needle
    assert "K_POINTS automatic" in ase_text and "K_POINTS automatic" in our_text
    assert "2 2 2" in ase_text and "2 2 2" in our_text
    assert "startingpot" in ase_text and "startingpot" in our_text


def test_pbe_only_recipe_drops_vdw_corr_on_the_ase_path_too(tmp_path: Path) -> None:
    calc = make_espresso_calculator(QeConfig(pseudo_dir="/pseudo", dispersion=None),
                                    directory=tmp_path)
    calc.write_inputfiles(_si(), properties=["energy"])
    text = next(tmp_path.glob("*.pwi")).read_text()
    assert "input_dft" in text and "vdw_corr" not in text


def test_profile_carries_command_and_pseudo_dir(tmp_path: Path) -> None:
    cfg = QeConfig(pseudo_dir="/pseudo", pw_cmd=("mpirun", "-np", "4", "pw.x"))
    calc = make_espresso_calculator(cfg, directory=tmp_path)
    assert calc.profile.pseudo_dir == "/pseudo"
    assert calc.profile.command == "mpirun -np 4 pw.x"
    override = make_espresso_calculator(cfg, directory=tmp_path, command="pw.x -npool 2")
    assert override.profile.command == "pw.x -npool 2"


def _ase_parse(text: str):
    import io

    with io.StringIO(text) as fd:
        frames = list(read_espresso_out(fd, index=slice(None)))
    return frames[-1].calc.results


def test_result_conventions_match_the_handwritten_parser() -> None:
    """Same output, two parsers: same units, same signs, same Voigt order."""
    ours = parse_qe_output(FIXTURE.read_text())
    ase = _ase_parse(FIXTURE.read_text())
    # ASE's espresso reader uses CODATA-2006 Rydbergs (Ry differs from the
    # current ase.units default at ~8e-8 relative); magnitudes agree well
    # inside 1e-6 — conventions (units/signs/order) are identical.
    assert ase["energy"] == pytest.approx(ours.energy, rel=1e-6)
    np.testing.assert_allclose(ase["forces"], ours.forces, rtol=1e-6, atol=1e-10)
    ase_stress = np.asarray(ase["stress"])
    assert ase_stress.shape == (6,)
    np.testing.assert_allclose(ase_stress, ours.stress, rtol=1e-6, atol=1e-10)
    # Sign and Voigt order exactly: isotropic compression is negative in both.
    assert np.sign(ase_stress[0]) == np.sign(ours.stress[0]) == -1.0
    assert ase_stress[3:] == pytest.approx(ours.stress[3:], abs=1e-12)


def test_force_conventions_match_on_nonzero_forces() -> None:
    """Both parsers convert Ry/bohr to eV/Å with the same sign on the same
    block (the fixture's zero forces cannot catch a sign or unit flip)."""
    text = FIXTURE.read_text()
    # Give atom 1 a nonzero force; keep the block structure intact.
    old = "force =     0.00000000    0.00000000    0.00000000"
    new = "force =     0.01000000   -0.02000000    0.03000000"
    assert text.count(old) >= 1
    modified = text.replace(old, new, 1)
    ours = parse_qe_output(modified)
    ase = _ase_parse(modified)
    expected = np.array([0.01, -0.02, 0.03]) * (units.Hartree / 2.0) / units.Bohr
    np.testing.assert_allclose(ours.forces[0], expected, rtol=1e-6)
    np.testing.assert_allclose(ase["forces"][0], ours.forces[0], rtol=1e-6)
    np.testing.assert_allclose(ase["forces"][1], ours.forces[1], rtol=1e-6, atol=1e-10)


def test_ase_qe_engine_identity_and_directories(tmp_path: Path) -> None:
    cfg = QeConfig(pseudo_dir="/pseudo")
    engine = AseQeEngine(cfg, run_root=tmp_path / "runs")
    assert engine.name == "ase-qe-pbe-d3"
    caps = engine.capabilities
    assert caps.energy_kind == EnergyKind.ENERGY
    assert caps.force_consistent is True and caps.forces_conservative is True
    assert caps.stress_available is True
    assert engine.fingerprint.startswith("ase-qe-pbe-d3:")
    # The two QE paths share settings but not identity: labels record the path.
    handwritten = QeEngine(cfg, run_root=tmp_path / "hw")
    assert engine.fingerprint != handwritten.fingerprint
    metallic = AseQeEngine(QeConfig(pseudo_dir="/pseudo", metallic=True),
                           run_root=tmp_path / "m")
    assert metallic.capabilities.energy_kind == EnergyKind.FREE_ENERGY
    first = engine.fresh_directory("step")
    second = engine.fresh_directory("step")
    assert first != second
    assert first.name == "step-000000" and second.name == "step-000001"
