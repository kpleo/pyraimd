"""QeEngine tests: output parsing against a real pw.x fixture, input writing,
and the full compute() path with a fake pw.x binary (hermetic — no QE needed).

Fixture: tests/data/qe_si_scf.out — a QE 7.5 Si bulk parsing fixture: E = -93.43942921 Ry, zero forces, isotropic stress
0.00017602 Ry/bohr^3 (P = 25.89 kbar).
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import molecule

from pyraimd2.engines.base import EngineError
from pyraimd2.engines.qe_engine import (
    QeConfig,
    QeEngine,
    _pseudo_sha256,
    load_density_source,
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


def test_write_input_pseudo_inside_pseudo_dir_uses_basename(tmp_path: Path) -> None:
    """A pseudopotential living inside pseudo_dir must be named by basename:
    QE parses ATOMIC_SPECIES lines with a limited buffer, and a long
    absolute path silently truncates into an unreadable filename."""
    pseudo_dir = tmp_path / "deep" / "nested" / "pseudo" / "library" / "dir"
    pseudo_dir.mkdir(parents=True)
    pseudo = pseudo_dir / "Si.pbe-n-kjpaw_psl.1.0.0.UPF"
    pseudo.write_text("UPF")
    cfg = QeConfig(pseudo_dir=str(pseudo_dir),
                   pseudos={"H": str(pseudo), "O": str(pseudo)})
    out = tmp_path / "pw.in"
    write_qe_input(out, _water_box(), cfg)
    lines = out.read_text().splitlines()
    species_lines = lines[lines.index("ATOMIC_SPECIES") + 1:
                          lines.index("CELL_PARAMETERS angstrom")]
    for line in species_lines:
        assert len(line) <= 80, line
        assert line.endswith("Si.pbe-n-kjpaw_psl.1.0.0.UPF"), line
        assert str(pseudo_dir) not in line


def _fake_pwx(tmp_path: Path, body: str) -> tuple[str, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_pwx.sh"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return ("bash", str(script))


def test_pseudo_sha256_same_tick_same_size_rewrite(tmp_path: Path) -> None:
    """UPF identity must track content even when a coarse-mtime filesystem
    makes (path, mtime, size) identical across a same-size rewrite."""
    pseudo = tmp_path / "X.pbe-n.UPF"
    pseudo.write_text("pseudo-v1")
    tick = time.time_ns()
    os.utime(pseudo, ns=(tick, tick))
    first = _pseudo_sha256(pseudo)
    pseudo.write_text("pseudo-v2")
    os.utime(pseudo, ns=(tick, tick))  # same path, same size, same mtime tick
    second = _pseudo_sha256(pseudo)
    assert first is not None
    assert second is not None
    assert first != second


def test_pseudo_sha256_old_mtime_same_size_rewrite(tmp_path: Path) -> None:
    """mtime pinned old, then a same-size rewrite preserving the old mtime:
    the content identity must change (no stat tuple is trusted)."""
    pseudo = tmp_path / "X.pbe-n.UPF"
    pseudo.write_text("pseudo-v1")
    old = (1_600_000_000, 1_600_000_000)
    os.utime(pseudo, old)
    first = _pseudo_sha256(pseudo)
    pseudo.write_text("pseudo-v2")
    os.utime(pseudo, old)
    second = _pseudo_sha256(pseudo)
    assert first is not None
    assert second is not None
    assert first != second


def test_pseudo_sha256_atomic_replace(tmp_path: Path) -> None:
    """Atomic same-path replacement: identity follows content even with the
    mtime pinned old."""
    pseudo = tmp_path / "X.pbe-n.UPF"
    pseudo.write_text("pseudo-v1")
    old = (1_600_000_000, 1_600_000_000)
    os.utime(pseudo, old)
    first = _pseudo_sha256(pseudo)
    staging = tmp_path / "staging.UPF"
    staging.write_text("pseudo-v2")
    os.utime(staging, old)
    os.replace(staging, pseudo)
    second = _pseudo_sha256(pseudo)
    assert first is not None
    assert second is not None
    assert first != second


def _large_bytes(seed_byte: int, size: int) -> bytes:
    block = bytes((seed_byte + i) % 256 for i in range(256))
    return (block * (size // 256 + 1))[:size]


def test_pseudo_sha256_large_file_same_size_rewrite(tmp_path: Path) -> None:
    """A file just above the former 16 MiB cache threshold: same path, same
    size, mtime pinned identical — the identity must still follow the real
    bytes (the threshold is gone; large files are re-read too)."""
    size = 16 * 1024 * 1024 + 1
    pseudo = tmp_path / "X.pbe-n.UPF"
    old = (1_600_000_000, 1_600_000_000)
    pseudo.write_bytes(_large_bytes(0, size))
    os.utime(pseudo, old)
    first = _pseudo_sha256(pseudo)
    pseudo.write_bytes(_large_bytes(1, size))
    os.utime(pseudo, old)
    second = _pseudo_sha256(pseudo)
    assert first is not None
    assert second is not None
    assert first != second


def test_write_input_disk_io_opt_in(tmp_path: Path) -> None:
    """disk_io is an execution knob: absent by default, written verbatim when set."""
    out = tmp_path / "pw.in"
    write_qe_input(out, _water_box(), QeConfig(pseudo_dir="/pseudo"))
    assert "disk_io" not in out.read_text()
    write_qe_input(out, _water_box(), QeConfig(pseudo_dir="/pseudo", disk_io="nowf"))
    assert "disk_io = 'nowf'" in out.read_text()


def test_disk_io_rejects_unsupported_value(tmp_path: Path) -> None:
    """An unsupported disk_io fails at construction — before any launch —
    and Python None is not the string 'none'."""
    QeEngine(QeConfig(pseudo_dir="/pseudo", disk_io=None), run_root=tmp_path / "a")
    QeEngine(QeConfig(pseudo_dir="/pseudo", disk_io="none"), run_root=tmp_path / "b")
    with pytest.raises(EngineError, match="disk_io"):
        QeEngine(QeConfig(pseudo_dir="/pseudo", disk_io="banana"),
                 run_root=tmp_path / "c")


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


def test_compute_directory_counter_continues_across_processes(tmp_path: Path) -> None:
    """A fresh engine on an existing run root (resume) continues directory
    numbering instead of colliding with earlier attempts."""
    body = f"#!/bin/bash\ncat {FIXTURE.resolve()}\n"
    first = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=[5.43] * 3,
               pbc=True)
    first.compute(si)
    assert (tmp_path / "runs" / "eval-000000").exists()
    second = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    second.compute(si)
    assert (tmp_path / "runs" / "eval-000001" / "attempt-1" / "pw.in").exists()
    # Labels do not reset the global numbering either.
    second.compute(si, label="chain")
    assert (tmp_path / "runs" / "chain-000002" / "attempt-1" / "pw.in").exists()


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
    # A second engine on the same run root continues the global numbering.
    run_dir = tmp_path / "runs" / "chain-000001"
    attempt_1 = (run_dir / "attempt-1" / "pw.in").read_text()
    attempt_2 = (run_dir / "attempt-2" / "pw.in").read_text()
    assert "startingpot = 'file'" in attempt_1
    assert "startingpot" not in attempt_2
    # The failed attempt's output stays on disk for diagnosis.
    assert "bad density" in (run_dir / "attempt-1" / "pw.out").read_text()


def test_nowf_keeps_the_density_chain(tmp_path: Path) -> None:
    """disk_io='nowf' still writes the converged charge density (only the
    wavefunctions are skipped), so the chain must keep reusing it."""
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(
            tmp_path, "#!/bin/bash\n" + _MAKE_SAVE + f"cat {FIXTURE.resolve()}\n"),
            startpot_file=True, disk_io="nowf"),
        run_root=tmp_path / "runs",
    )
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
               cell=[5.43] * 3, pbc=True)
    engine.compute(si, label="first")
    engine.compute(si, label="second")
    assert engine.last_density_decision["start"] == "density"
    manifest = json.loads(
        (tmp_path / "runs" / "first-000000" / "attempt-1"
         / "density_manifest.json").read_text())
    assert manifest["density_available"] is True


def test_disk_io_none_claims_no_reusable_density(tmp_path: Path) -> None:
    """disk_io='none' writes no charge density: the manifest must not claim
    one, the attempt must not become a chained-start source, and a later
    evaluation falls back to an atomic start with the reason recorded."""
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", disk_io="none",
                 # no _MAKE_SAVE: like a real disk_io='none' run, nothing is
                 # left in tmp/pyraimd2.save
                 pw_cmd=_fake_pwx(tmp_path, f"#!/bin/bash\ncat {FIXTURE.resolve()}\n"),
                 startpot_file=True),
        run_root=tmp_path / "runs",
    )
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
               cell=[5.43] * 3, pbc=True)
    engine.compute(si, label="first")
    attempt_dir = tmp_path / "runs" / "first-000000" / "attempt-1"
    manifest = json.loads((attempt_dir / "density_manifest.json").read_text())
    assert manifest["density_available"] is False
    source, reason = load_density_source(attempt_dir, engine=engine, atoms=si)
    assert source is None
    assert "charge density" in reason
    engine.compute(si, label="second")
    # the non-producing attempt was never promoted, so the chain stays atomic
    assert engine.last_density_decision["start"] == "atomic"
    assert engine.last_density_decision["reason"] == "no density source configured"


def _si() -> Atoms:
    return Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                 cell=[5.43] * 3, pbc=True)


def _density_engine(work: Path, *, policy: str = "latest",
                    source: str | None = None) -> QeEngine:
    body = "#!/bin/bash\n" + _MAKE_SAVE + f"cat {FIXTURE.resolve()}\n"
    return QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(work, body),
                 startpot_file=True, density_source=source,
                 density_source_policy=policy),
        run_root=work / "runs")


def test_density_chain_prefers_latest_successful(tmp_path: Path) -> None:
    """config.density_source only initializes: a continuous chain reuses the
    most recent successful density, not the original external source."""
    seed = _density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    assert seed.last_density_decision["start"] == "atomic"
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000" / "attempt-1")
    engine = _density_engine(tmp_path / "b", source=source_dir)
    engine.compute(_si(), label="first")
    assert engine.last_density_decision["via"] == "config.density_source"
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "previous attempt"
    assert engine.last_density_decision["origin"] != source_dir


def test_density_chain_fixed_policy_keeps_config_source(tmp_path: Path) -> None:
    """The legacy fixed order stays available explicitly: every evaluation
    re-seeds from the configured external source."""
    seed = _density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000" / "attempt-1")
    engine = _density_engine(tmp_path / "b", policy="fixed", source=source_dir)
    engine.compute(_si(), label="first")
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "config.density_source"
    assert engine.last_density_decision["origin"] == source_dir


def test_density_latest_invalid_falls_back_to_config(tmp_path: Path) -> None:
    """When the most recent density is unusable, the chain falls back to the
    configured external source instead of reusing the broken one."""
    seed = _density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000" / "attempt-1")
    engine = _density_engine(tmp_path / "b", source=source_dir)
    engine.compute(_si(), label="first")
    manifest = Path(engine._last_density_dir) / "density_manifest.json"
    assert manifest.exists()
    manifest.unlink()
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "config.density_source"


def test_density_resume_fresh_engine_uses_config_source(tmp_path: Path) -> None:
    """A fresh process (resume) has no in-memory density: the configured
    external source initializes it, and the record says so honestly."""
    seed = _density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000" / "attempt-1")
    resumed = _density_engine(tmp_path / "b", source=source_dir)
    resumed.compute(_si(), label="resume")
    assert resumed.last_density_decision["via"] == "config.density_source"
    assert "fresh process" in resumed.last_density_decision["note"]


def test_density_source_policy_rejects_unknown(tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="density_source_policy"):
        _density_engine(tmp_path, policy="sometimes")


def test_density_failed_attempt_is_not_promoted(tmp_path: Path) -> None:
    """A failed evaluation must not become the chain's latest density: the
    pointer stays on the last attempt that actually succeeded."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 2 ]; then echo "transient crash"; exit 139; fi\n'
        + _MAKE_SAVE + f"cat {FIXTURE.resolve()}\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                 startpot_file=True, max_retries=0),
        run_root=tmp_path / "runs",
    )
    engine.compute(_si(), label="first")
    latest = engine._last_density_dir
    assert latest is not None
    with pytest.raises(EngineError):
        engine.compute(_si(), label="crash")
    assert engine._last_density_dir == latest  # unchanged by the failure
    engine.compute(_si(), label="third")
    assert engine._last_density_dir != latest
    assert engine.last_density_decision["via"] == "previous attempt"
    assert engine.last_density_decision["origin"] == str(latest)


def test_density_mismatched_source_is_not_used(tmp_path: Path) -> None:
    """A density from different reference settings is rejected, not chained."""
    seed = _density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000" / "attempt-1")
    other = QeEngine(
        QeConfig(pseudo_dir="/pseudo", ecutwfc=99.0,  # different fingerprint
                 pw_cmd=_fake_pwx(tmp_path / "b",
                                  "#!/bin/bash\n" + _MAKE_SAVE
                                  + f"cat {FIXTURE.resolve()}\n"),
                 startpot_file=True, density_source=source_dir),
        run_root=tmp_path / "b" / "runs",
    )
    other.compute(_si(), label="first")
    assert other.last_density_decision["start"] == "atomic"
    assert "reference settings differ" in other.last_density_decision["reason"]


# HDF5-build QE writes charge-density.hdf5 instead of .dat.  This fixture is
# a NAMED MARKER only: it validates the wrapper's file recognition, never
# HDF5 parsing or a real pw.x run.
_MAKE_SAVE_HDF5 = ("mkdir -p tmp/pyraimd2.save && echo fake-density-hdf5 "
                   "> tmp/pyraimd2.save/charge-density.hdf5\n")


def test_hdf5_density_source_loads_and_chains(tmp_path: Path) -> None:
    """An HDF5-build density (charge-density.hdf5, no .dat) must be a valid
    warm-start source and a promotable product — the recognition rule covers
    both formats QE 7.5 writes."""
    body = "#!/bin/bash\n" + _MAKE_SAVE_HDF5 + f"cat {FIXTURE.resolve()}\n"
    seed = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path / "a", body),
                 startpot_file=True),
        run_root=tmp_path / "a" / "runs")
    seed.compute(_si(), label="seed")
    attempt_dir = tmp_path / "a" / "runs" / "seed-000000" / "attempt-1"
    manifest = json.loads((attempt_dir / "density_manifest.json").read_text())
    assert manifest["density_available"] is True
    source, reason = load_density_source(attempt_dir, engine=seed, atoms=_si())
    assert source is not None, reason
    source_dir = str(attempt_dir)
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path / "b", body),
                 startpot_file=True, density_source=source_dir),
        run_root=tmp_path / "b" / "runs")
    engine.compute(_si(), label="first")
    assert engine.last_density_decision["via"] == "config.density_source"
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "previous attempt"


def _seeded_no_output_engine(work: Path, *, disk_io: str,
                             source: str) -> QeEngine:
    """A disk_io mode that writes no new density; the fake leaves only the
    staged input copy behind (exactly what QE's none/minimal punch does)."""
    return QeEngine(
        QeConfig(pseudo_dir="/pseudo", disk_io=disk_io,
                 pw_cmd=_fake_pwx(work, f"#!/bin/bash\ncat {FIXTURE.resolve()}\n"),
                 startpot_file=True, density_source=source),
        run_root=work / "runs")


def test_seeded_disk_io_none_keeps_true_origin(tmp_path: Path) -> None:
    """disk_io='none' with a staged seed: the copy still sits in the save
    tree after the run, but it is the INPUT, not this attempt's output —
    no false density claim, no promotion, and the chain keeps the true
    origin."""
    seed = _density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000" / "attempt-1")
    engine = _seeded_no_output_engine(tmp_path / "b", disk_io="none",
                                      source=source_dir)
    engine.compute(_si(), label="first")
    attempt_dir = tmp_path / "b" / "runs" / "first-000000" / "attempt-1"
    # the staged copy really survives — the fix is in the claim, not the copy
    assert (attempt_dir / "tmp" / "pyraimd2.save" / "charge-density.dat").is_file()
    manifest = json.loads((attempt_dir / "density_manifest.json").read_text())
    assert manifest["density_available"] is False
    assert manifest["source"]["from"] == source_dir  # true input origin kept
    assert engine.last_attempt_records[-1]["density_available"] is False
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "config.density_source"
    assert engine.last_density_decision["origin"] == source_dir


def test_seeded_disk_io_minimal_keeps_true_origin(tmp_path: Path) -> None:
    """Same contract for disk_io='minimal' (XML only, no new density)."""
    seed = _density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000" / "attempt-1")
    engine = _seeded_no_output_engine(tmp_path / "b", disk_io="minimal",
                                      source=source_dir)
    engine.compute(_si(), label="first")
    attempt_dir = tmp_path / "b" / "runs" / "first-000000" / "attempt-1"
    manifest = json.loads((attempt_dir / "density_manifest.json").read_text())
    assert manifest["density_available"] is False
    assert manifest["source"]["from"] == source_dir
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "config.density_source"
    assert engine.last_density_decision["origin"] == source_dir


def test_seeded_nowf_produces_and_promotes_new_density(tmp_path: Path) -> None:
    """nowf still writes the converged density: a seeded nowf attempt is a
    genuine new product and becomes the chain's next source."""
    body = "#!/bin/bash\n" + _MAKE_SAVE_HDF5 + f"cat {FIXTURE.resolve()}\n"
    seed = _density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000" / "attempt-1")
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", disk_io="nowf",
                 pw_cmd=_fake_pwx(tmp_path / "b", body),
                 startpot_file=True, density_source=source_dir),
        run_root=tmp_path / "b" / "runs")
    engine.compute(_si(), label="first")
    attempt_dir = tmp_path / "b" / "runs" / "first-000000" / "attempt-1"
    manifest = json.loads((attempt_dir / "density_manifest.json").read_text())
    assert manifest["density_available"] is True
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "previous attempt"
    assert engine.last_density_decision["origin"] == str(attempt_dir)


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
