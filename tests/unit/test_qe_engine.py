"""QeEngine tests: output parsing against a real pw.x fixture, input writing,
and the full compute() path with a fake pw.x binary (hermetic — no QE needed).

Fixture: tests/data/qe_si_scf.out — the Si bulk validation run on Neimeng A
(job 7614854, QE 7.5): E = -93.43942921 Ry, zero forces, isotropic stress
0.00017602 Ry/bohr^3 (P = 25.89 kbar).
"""

from __future__ import annotations

import stat
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import molecule

from pyraimd2.engines.base import EngineError
from pyraimd2.engines.qe_engine import (
    QeConfig,
    QeEngine,
    parse_qe_output,
    write_qe_input,
)

FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"
SI_ENERGY_RY = -93.43942921
SI_STRESS_RY_BOHR3 = 0.00017602


def test_parse_si_fixture() -> None:
    result = parse_qe_output(FIXTURE.read_text())
    assert result.energy == pytest.approx(SI_ENERGY_RY * units.Hartree / 2.0, abs=1e-6)
    assert result.forces.shape == (2, 3)
    assert np.abs(result.forces).max() < 1e-8
    assert result.stress is not None
    assert result.stress.shape == (6,)
    # ASE sign convention is compression-negative: stress = -QE value.
    assert result.stress[0] == pytest.approx(
        -SI_STRESS_RY_BOHR3 * (units.Hartree / 2.0) / units.Bohr**3, rel=1e-6
    )
    # Cross-check against the printed pressure: P = -tr(stress)/3 in eV/A^3
    # must equal 25.89 kbar (1 eV/A^3 = 1602.18 kbar).
    p_kbar = -float(np.trace(np.diag(result.stress[:3])) / 3.0) * 1602.18
    assert p_kbar == pytest.approx(25.89, abs=0.05)


def test_parse_missing_energy_raises() -> None:
    with pytest.raises(EngineError):
        parse_qe_output("this is not a pw.x output")


def _water_box() -> Atoms:
    atoms = molecule("H2O")
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.set_pbc(True)
    return atoms


def test_write_input(tmp_path: Path) -> None:
    cfg = QeConfig(pseudo_dir="/pseudo", ecutwfc=60.0, ecutrho=480.0)
    out = tmp_path / "pw.in"
    write_qe_input(out, _water_box(), cfg)
    text = out.read_text()
    assert "vdw_corr = 'grimme-d3'" in text
    assert "ecutwfc = 60.0" in text
    assert "nat = 3" in text and "ntyp = 2" in text
    assert "CELL_PARAMETERS angstrom" in text and "ATOMIC_POSITIONS angstrom" in text
    assert "K_POINTS gamma" in text
    assert "H.pbe-kjpaw_psl.1.0.0.UPF" in text


def test_write_input_metallic_smearing(tmp_path: Path) -> None:
    cfg = QeConfig(pseudo_dir="/pseudo", metallic=True, kpts=(2, 2, 1))
    out = tmp_path / "pw.in"
    write_qe_input(out, _water_box(), cfg)
    text = out.read_text()
    assert "smearing = 'mv'" in text
    assert "2 2 1 0 0 0" in text


def _fake_pwx(tmp_path: Path, body: str) -> tuple[str, ...]:
    script = tmp_path / "fake_pwx.sh"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return ("bash", str(script))


def test_compute_with_fake_pwx(tmp_path: Path) -> None:
    body = f"#!/bin/bash\ncat {FIXTURE.resolve()}\n"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=[5.43] * 3, pbc=True)
    result = engine.compute(si, label="t0")
    assert result.energy == pytest.approx(SI_ENERGY_RY * units.Hartree / 2.0, abs=1e-6)
    assert result.forces.shape == (2, 3)
    assert result.wall_time_s >= 0.0
    assert (tmp_path / "runs" / "t0" / "pw.in").exists()


def test_compute_engine_error_on_failure(tmp_path: Path) -> None:
    body = "#!/bin/bash\necho 'garbage output'; exit 3\n"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="exited with code 3"):
        engine.compute(_water_box(), label="boom")


def test_compute_engine_error_on_nonconvergence(tmp_path: Path) -> None:
    body = f"#!/bin/bash\nsed 's/convergence has been achieved/convergence NOT achieved/' {FIXTURE.resolve()}\n"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="did not converge"):
        engine.compute(_water_box(), label="nc")


def test_compute_engine_error_on_timeout(tmp_path: Path) -> None:
    """A hung pw.x must surface as EngineError, never raw TimeoutExpired."""
    body = "#!/bin/bash\nsleep 30\n"
    engine = QeEngine(
        QeConfig(
            pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), timeout_s=0.5
        ),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="timed out"):
        engine.compute(_water_box(), label="hang")
