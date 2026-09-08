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
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    assert (tmp_path / "runs" / "t0-000000" / "attempt-1" / "pw.in").exists()


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


def test_compute_startpot_retry_after_failure(tmp_path: Path) -> None:
    """A failed first attempt is retried within the retry budget.

    Regression: the wipe path crashed with NameError (missing shutil import)
    in production (W bootstrap labels, 2026-09-03), masking the real error.
    """
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 1 ]; then echo garbage; exit 3; fi\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                 startpot_file=True),
        run_root=tmp_path / "runs",
    )
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
               cell=[5.43] * 3, pbc=True)
    result = engine.compute(si, label="retry")
    assert result.energy == pytest.approx(SI_ENERGY_RY * units.Hartree / 2.0,
                                          abs=1e-6)
    assert counter.read_text().strip() == "2"


_READ_INPUT = (
    'in=""; while [ $# -gt 0 ]; do '
    'if [ "$1" = "-in" ]; then in="$2"; shift 2; else shift; fi; done\n'
    '[ -f "$in" ] || { echo "missing input: $in"; exit 7; }\n'
)

# Fake pw.x that also leaves a charge-density tree behind, like a real run
# writing <outdir>/<prefix>.save into its working directory.
_MAKE_SAVE = "mkdir -p tmp/pyraimd2.save && echo fake-density > tmp/pyraimd2.save/charge-density.dat\n"


def test_compute_unique_directories_and_absolute_input(tmp_path: Path, monkeypatch) -> None:
    """Consecutive calls must not share a work directory, and the input path
    handed to pw.x must stay readable from inside the run directory even when
    run_root itself is relative."""
    body = "#!/bin/bash\n" + _READ_INPUT + f"cat {FIXTURE.resolve()}\n"
    monkeypatch.chdir(tmp_path)
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root="runs",  # deliberately relative: catches cwd-coupled paths
    )
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=[5.43] * 3, pbc=True)
    first = engine.compute(si)
    second = engine.compute(si)
    assert first.energy == pytest.approx(second.energy)
    run_dirs = sorted(p.name for p in (tmp_path / "runs").iterdir() if p.is_dir())
    assert len(run_dirs) == 2


def test_density_start_fallback_keeps_failed_attempt(tmp_path: Path) -> None:
    """A failed chained-density start retries once with an explicit
    atomic-start input (a different input, not the same startpot='file'
    against a missing density), and keeps the failed attempt's diagnostics."""
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(
            tmp_path, "#!/bin/bash\n" + _MAKE_SAVE + f"cat {FIXTURE.resolve()}\n"),
                 startpot_file=True),
        run_root=tmp_path / "runs",
    )
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
               cell=[5.43] * 3, pbc=True)
    engine.compute(si, label="seed")  # successful attempt -> reusable density

    failing = QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_fake_pwx(
                     tmp_path / "failing",
                     "#!/bin/bash\n" + _READ_INPUT
                     + "if grep -q \"startingpot = 'file'\" \"$in\"; then"
                     " echo 'bad density'; exit 3; fi\n"
                     + _MAKE_SAVE + f"cat {FIXTURE.resolve()}\n"),
                 startpot_file=True,
                 density_source=str(tmp_path / "runs" / "seed-000000" / "attempt-1")),
        run_root=tmp_path / "runs",
    )
    result = failing.compute(si, label="chain")
    assert result.energy == pytest.approx(SI_ENERGY_RY * units.Hartree / 2.0,
                                          abs=1e-6)
    run_dir = tmp_path / "runs" / "chain-000000"
    attempt_1 = (run_dir / "attempt-1" / "pw.in").read_text()
    attempt_2 = (run_dir / "attempt-2" / "pw.in").read_text()
    assert "startingpot = 'file'" in attempt_1
    assert "startingpot" not in attempt_2
    # The failed attempt's output stays on disk for diagnosis.
    assert "bad density" in (run_dir / "attempt-1" / "pw.out").read_text()


def test_parse_groups_last_scf_block() -> None:
    """Energy, forces and stress must come from the same (last complete) SCF
    block — not last energy + every force line + first stress."""
    block_a = (
        "!    total energy              =      -1.00000000 Ry\n"
        "     Forces acting on atoms (cartesian axes, Ry/au):\n\n"
        "     atom    1 type  1   force =     0.10000000    0.00000000    0.00000000\n"
        "     atom    2 type  1   force =    -0.10000000    0.00000000    0.00000000\n\n"
        "          total   stress  (Ry/bohr**3)                   (kbar)     P=        1.00\n"
        "   0.00001000   0.00000000   0.00000000            1.00        0.00        0.00\n"
        "   0.00000000   0.00001000   0.00000000            0.00        1.00        0.00\n"
        "   0.00000000   0.00000000   0.00001000            0.00        0.00        1.00\n"
    )
    block_b = block_a.replace("-1.00000000", "-2.00000000").replace("0.10000000", "0.20000000").replace("0.00001000", "0.00002000")
    result = parse_qe_output(block_a + "\n" + block_b)
    assert result.energy == pytest.approx(-2.0 * units.Hartree / 2.0, abs=1e-6)
    assert result.forces.shape == (2, 3)
    assert result.forces[0, 0] == pytest.approx(0.2 * (units.Hartree / 2.0) / units.Bohr, rel=1e-6)
    assert result.stress is not None
    assert result.stress[0] == pytest.approx(
        -0.00002 * (units.Hartree / 2.0) / units.Bohr**3, rel=1e-6
    )


def test_parse_fortran_d_exponents() -> None:
    """pw.x can print Fortran D exponents; they must parse, not crash float()."""
    text = (
        "!    total energy              =     -0.93439429D+02 Ry\n"
        "     Forces acting on atoms (cartesian axes, Ry/au):\n\n"
        "     atom    1 type  1   force =     0.10000000D+00    0.00000000D+00    0.00000000D+00\n\n"
    )
    result = parse_qe_output(text)
    assert result.energy == pytest.approx(-93.439429 * units.Hartree / 2.0, rel=1e-6)
    assert result.forces.shape == (1, 3)
    assert result.forces[0, 0] == pytest.approx(0.1 * (units.Hartree / 2.0) / units.Bohr, rel=1e-6)


def test_parse_missing_force_block_raises() -> None:
    """An energy line without a following force block is not a label."""
    text = "!    total energy              =      -1.00000000 Ry\n"
    with pytest.raises(EngineError, match="no force block"):
        parse_qe_output(text)


def test_parse_empty_force_block_raises() -> None:
    """A force header with no atom lines (e.g. killed mid-write) is not a label."""
    text = (
        "!    total energy              =      -1.00000000 Ry\n"
        "     Forces acting on atoms (cartesian axes, Ry/au):\n\n"
        "     The total force is     0.000\n"
    )
    with pytest.raises(EngineError, match="force block.*empty"):
        parse_qe_output(text)


def test_compute_engine_error_on_atom_count_mismatch(tmp_path: Path) -> None:
    """A force block for the wrong number of atoms must be rejected, never
    truncated or padded into a label."""
    one_atom = (
        "!    total energy              =      -1.00000000 Ry\n"
        "     Forces acting on atoms (cartesian axes, Ry/au):\n\n"
        "     atom    1 type  1   force =     0.10000000    0.00000000    0.00000000\n\n"
    )
    body = f"#!/bin/bash\ncat <<'EOF'\n{one_atom}   JOB DONE.\nEOF\n"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=[5.43] * 3, pbc=True)
    with pytest.raises(EngineError, match=r"force shape.*!= \(2, 3\)"):
        engine.compute(si, label="nat")
