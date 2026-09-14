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
import json
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


# --- compute-path contract (fake pw.x through ASE's real FileIO machinery) --

import stat as _stat

from pyraimd2.engines.base import EngineError as _EngineError
from pyraimd2.runtime.events import EventLog as _EventLog


def _fake_pwx(tmp_path: Path, body: str) -> tuple[str, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_pwx.sh"
    script.write_text(body)
    script.chmod(script.stat().st_mode | _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH)
    return ("bash", str(script))


def _fixture_cat() -> str:
    return f"#!/bin/bash\ncat {FIXTURE.resolve()}\n"


def test_missing_job_done_rejected_like_the_handwritten_path(tmp_path: Path) -> None:
    """rc=0 with values but no completion marker: not a label (review R5.4 —
    the ASE path used to accept exactly this)."""
    text = FIXTURE.read_text().replace("JOB DONE.", "")
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_fake_pwx(tmp_path, f"#!/bin/bash\ncat <<'EOF'\n{text}\nEOF\n"),
                 max_retries=0),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(_EngineError, match="JOB DONE"):
        engine.compute(_si(), label="trunc")


def test_nonconverged_output_rejected_like_the_handwritten_path(tmp_path: Path) -> None:
    text = FIXTURE.read_text().replace(
        "JOB DONE.", "convergence NOT achieved after 200 iterations\nJOB DONE.")
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_fake_pwx(tmp_path, f"#!/bin/bash\ncat <<'EOF'\n{text}\nEOF\n"),
                 max_retries=0),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(_EngineError, match="did not converge"):
        engine.compute(_si(), label="nc")
    assert engine.last_attempt_records[0]["retryable"] is False


def test_qe_error_banner_rejected_like_the_handwritten_path(tmp_path: Path) -> None:
    body = (
        "#!/bin/bash\n"
        "echo '     Error in routine read_namelists (1):'\n"
        "echo '     bad line in namelist &control'\n"
        "exit 1\n"
    )
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), max_retries=3),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(_EngineError):
        engine.compute(_si(), label="bad")
    assert len(engine.last_attempt_records) == 1  # deterministic: never retried


def test_good_output_accepted_with_matching_metadata(tmp_path: Path) -> None:
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat())),
        run_root=tmp_path / "runs",
    )
    result = engine.compute(_si(), label="ok")
    assert result.energy == pytest.approx(-1271.308, abs=1e-2)
    # Label metadata matches the declared capabilities (review R5.13).
    assert result.force_consistent is True
    assert result.energy_kind == EnergyKind.ENERGY
    assert result.stress is not None and result.stress.shape == (6,)
    metallic = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", metallic=True,
                 pw_cmd=_fake_pwx(tmp_path / "m", _fixture_cat())),
        run_root=tmp_path / "mruns",
    )
    assert metallic.compute(_si(), label="m").energy_kind == EnergyKind.FREE_ENERGY


def test_fresh_instance_continues_directories(tmp_path: Path) -> None:
    """A second instance (resume, another workflow) must not restart at
    eval-000000 and collide (review R5.5)."""
    first = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat())),
        run_root=tmp_path / "runs",
    )
    first.compute(_si())
    assert (tmp_path / "runs" / "eval-000000").is_dir()
    second = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat())),
        run_root=tmp_path / "runs",
    )
    result = second.compute(_si())
    assert result.energy == pytest.approx(-1271.308, abs=1e-2)
    assert (tmp_path / "runs" / "eval-000001" / "espresso.pwi").exists()


def test_transient_failure_retried_deterministic_not(tmp_path: Path) -> None:
    counter = tmp_path / "calls.txt"
    flaky = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 1 ]; then echo "launcher hiccup"; exit 139; fi\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, flaky), max_retries=1),
        run_root=tmp_path / "runs",
    )
    result = engine.compute(_si(), label="flaky")
    assert result.energy == pytest.approx(-1271.308, abs=1e-2)
    assert counter.read_text().strip() == "2"
    assert [r["status"] for r in engine.last_attempt_records] == ["failed", "success"]
    # The failed attempt keeps its directory; the retry gets a fresh one.
    assert (tmp_path / "runs" / "flaky-000000").is_dir()
    assert (tmp_path / "runs" / "flaky-000001" / "espresso.pwi").exists()


def test_warm_start_without_density_goes_atomic_before_launch(tmp_path: Path) -> None:
    """startpot_file=True with no compatible density must not write
    startingpot='file' (review R5.10): the first attempt is atomic."""
    reader = (
        "#!/bin/bash\n"
        "if grep -q \"startingpot = 'file'\" espresso.pwi; then\n"
        "  echo 'Error in routine read_rho: no density file'; exit 2\n"
        "fi\n"
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, reader),
                 startpot_file=True),
        run_root=tmp_path / "runs",
    )
    result = engine.compute(_si(), label="cold")
    assert result.energy == pytest.approx(-1271.308, abs=1e-2)
    assert engine.last_density_decision["start"] == "atomic"
    assert len(engine.last_attempt_records) == 1


def test_warm_start_stages_density_and_records_provenance(tmp_path: Path) -> None:
    with_save = (
        "#!/bin/bash\n"
        "mkdir -p tmp/pyraimd2.save && echo fake-density > tmp/pyraimd2.save/charge-density.dat\n"
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, with_save),
                 startpot_file=True),
        run_root=tmp_path / "runs",
    )
    engine.compute(_si(), label="first")
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["start"] == "density"
    source_save = tmp_path / "runs" / "first-000000" / "tmp" / "pyraimd2.save"
    copied_save = tmp_path / "runs" / "second-000001" / "tmp" / "pyraimd2.save"
    assert copied_save.is_dir() and source_save.is_dir()
    assert (copied_save / "charge-density.dat").read_text() == \
        (source_save / "charge-density.dat").read_text()
    record = engine.last_attempt_records[-1]
    assert record["start"] == "density" and record["density_copy_bytes"] > 0
    # The warm-started input actually asks for the staged density.
    pwi = (tmp_path / "runs" / "second-000001" / "espresso.pwi").read_text()
    assert "startingpot" in pwi


def test_nondefault_timeout_rejected_at_construction(tmp_path: Path) -> None:
    """ASE's FileIO layer has no timeout: a non-default value must fail
    loudly instead of pretending to bound the run (review R5.10)."""
    with pytest.raises(_EngineError, match="timeout_s"):
        AseQeEngine(QeConfig(pseudo_dir="/pseudo", timeout_s=10.0),
                    run_root=tmp_path / "runs")


def test_espresso_input_data_maps_disk_io(tmp_path: Path) -> None:
    """The ASE path passes disk_io through to &CONTROL exactly like the
    handwritten writer: absent by default, verbatim when set."""
    assert "disk_io" not in espresso_input_data(QeConfig(pseudo_dir="/pseudo"))["control"]
    data = espresso_input_data(QeConfig(pseudo_dir="/pseudo", disk_io="nowf"))
    assert data["control"]["disk_io"] == "nowf"
    calc = make_espresso_calculator(QeConfig(pseudo_dir="/pseudo", disk_io="nowf"),
                                    directory=tmp_path)
    calc.write_inputfiles(_si(), properties=["energy"])
    assert "disk_io" in next(tmp_path.glob("*.pwi")).read_text()


def test_disk_io_rejects_unsupported_value(tmp_path: Path) -> None:
    """Same vocabulary check as the handwritten path, before any launch."""
    AseQeEngine(QeConfig(pseudo_dir="/pseudo", disk_io="none"),
                run_root=tmp_path / "a")
    with pytest.raises(_EngineError, match="disk_io"):
        AseQeEngine(QeConfig(pseudo_dir="/pseudo", disk_io="banana"),
                    run_root=tmp_path / "b")


def test_disk_io_none_claims_no_reusable_density(tmp_path: Path) -> None:
    """disk_io='none' writes no charge density: the manifest must not claim
    one and the chain must stay atomic (same contract as the handwritten
    path)."""
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", disk_io="none",
                 pw_cmd=_fake_pwx(tmp_path, _fixture_cat()),  # no .save tree
                 startpot_file=True),
        run_root=tmp_path / "runs",
    )
    engine.compute(_si(), label="first")
    manifest = json.loads(
        (tmp_path / "runs" / "first-000000" / "density_manifest.json").read_text())
    assert manifest["density_available"] is False
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["start"] == "atomic"


_WITH_SAVE = (
    "mkdir -p tmp/pyraimd2.save && echo fake-density > tmp/pyraimd2.save/charge-density.dat\n"
)


def _ase_density_engine(work: Path, *, policy: str = "latest",
                        source: str | None = None) -> AseQeEngine:
    body = "#!/bin/bash\n" + _WITH_SAVE + f"cat {FIXTURE.resolve()}\n"
    return AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(work, body),
                 startpot_file=True, density_source=source,
                 density_source_policy=policy),
        run_root=work / "runs")


def test_density_chain_prefers_latest_successful(tmp_path: Path) -> None:
    """Same policy semantics as the handwritten path: the configured source
    initializes, then the chain reuses the most recent successful density."""
    seed = _ase_density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    assert seed.last_density_decision["start"] == "atomic"
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000")
    engine = _ase_density_engine(tmp_path / "b", source=source_dir)
    engine.compute(_si(), label="first")
    assert engine.last_density_decision["via"] == "config.density_source"
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "previous attempt"
    assert engine.last_density_decision["origin"] != source_dir


def test_density_chain_fixed_policy_keeps_config_source(tmp_path: Path) -> None:
    seed = _ase_density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000")
    engine = _ase_density_engine(tmp_path / "b", policy="fixed", source=source_dir)
    engine.compute(_si(), label="first")
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "config.density_source"
    assert engine.last_density_decision["origin"] == source_dir


def test_density_latest_invalid_falls_back_to_config(tmp_path: Path) -> None:
    seed = _ase_density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000")
    engine = _ase_density_engine(tmp_path / "b", source=source_dir)
    engine.compute(_si(), label="first")
    manifest = Path(engine._last_density_dir) / "density_manifest.json"
    assert manifest.exists()
    manifest.unlink()
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "config.density_source"


def test_density_resume_fresh_engine_uses_config_source(tmp_path: Path) -> None:
    seed = _ase_density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000")
    resumed = _ase_density_engine(tmp_path / "b", source=source_dir)
    resumed.compute(_si(), label="resume")
    assert resumed.last_density_decision["via"] == "config.density_source"
    assert "fresh process" in resumed.last_density_decision["note"]


def test_density_source_policy_rejects_unknown(tmp_path: Path) -> None:
    with pytest.raises(_EngineError, match="density_source_policy"):
        _ase_density_engine(tmp_path, policy="sometimes")


def test_density_failed_attempt_is_not_promoted(tmp_path: Path) -> None:
    """Same contract as the handwritten path: a failed evaluation never
    becomes the chain's latest density."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 2 ]; then echo "transient crash"; exit 139; fi\n'
        + _WITH_SAVE + f"cat {FIXTURE.resolve()}\n"
    )
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                 startpot_file=True, max_retries=0),
        run_root=tmp_path / "runs",
    )
    engine.compute(_si(), label="first")
    latest = engine._last_density_dir
    assert latest is not None
    with pytest.raises(_EngineError):
        engine.compute(_si(), label="crash")
    assert engine._last_density_dir == latest  # unchanged by the failure
    engine.compute(_si(), label="third")
    assert engine._last_density_dir != latest
    assert engine.last_density_decision["via"] == "previous attempt"
    assert engine.last_density_decision["origin"] == str(latest)


# HDF5-build QE writes charge-density.hdf5 instead of .dat.  This fixture is
# a NAMED MARKER only: it validates the wrapper's file recognition, never
# HDF5 parsing or a real pw.x run.
_WITH_SAVE_HDF5 = ("mkdir -p tmp/pyraimd2.save && echo fake-density-hdf5 "
                   "> tmp/pyraimd2.save/charge-density.hdf5\n")


def test_hdf5_density_source_loads_and_chains(tmp_path: Path) -> None:
    """Same recognition rule as the handwritten path: an HDF5-only density
    is a valid warm-start source and a promotable product."""
    body = "#!/bin/bash\n" + _WITH_SAVE_HDF5 + f"cat {FIXTURE.resolve()}\n"
    seed = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path / "a", body),
                 startpot_file=True),
        run_root=tmp_path / "a" / "runs")
    seed.compute(_si(), label="seed")
    seed_dir = tmp_path / "a" / "runs" / "seed-000000"
    manifest = json.loads((seed_dir / "density_manifest.json").read_text())
    assert manifest["density_available"] is True
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path / "b", body),
                 startpot_file=True, density_source=str(seed_dir)),
        run_root=tmp_path / "b" / "runs")
    engine.compute(_si(), label="first")
    assert engine.last_density_decision["via"] == "config.density_source"
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "previous attempt"


def _seeded_no_output_engine(work: Path, *, disk_io: str,
                             source: str) -> AseQeEngine:
    """A disk_io mode that writes no new density; the fake leaves only the
    staged input copy behind (exactly what QE's none/minimal punch does)."""
    return AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", disk_io=disk_io,
                 pw_cmd=_fake_pwx(work, _fixture_cat()),
                 startpot_file=True, density_source=source),
        run_root=work / "runs")


def test_seeded_disk_io_none_keeps_true_origin(tmp_path: Path) -> None:
    """Same contract as the handwritten path: the staged input copy survives
    the run but is never claimed as this attempt's product."""
    seed = _ase_density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000")
    engine = _seeded_no_output_engine(tmp_path / "b", disk_io="none",
                                      source=source_dir)
    engine.compute(_si(), label="first")
    attempt_dir = tmp_path / "b" / "runs" / "first-000000"
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
    seed = _ase_density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000")
    engine = _seeded_no_output_engine(tmp_path / "b", disk_io="minimal",
                                      source=source_dir)
    engine.compute(_si(), label="first")
    attempt_dir = tmp_path / "b" / "runs" / "first-000000"
    manifest = json.loads((attempt_dir / "density_manifest.json").read_text())
    assert manifest["density_available"] is False
    assert manifest["source"]["from"] == source_dir
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "config.density_source"
    assert engine.last_density_decision["origin"] == source_dir


def test_seeded_nowf_produces_and_promotes_new_density(tmp_path: Path) -> None:
    """nowf still writes the converged density: a seeded nowf attempt is a
    genuine new product and becomes the chain's next source."""
    body = "#!/bin/bash\n" + _WITH_SAVE_HDF5 + f"cat {FIXTURE.resolve()}\n"
    seed = _ase_density_engine(tmp_path / "a")
    seed.compute(_si(), label="seed")
    source_dir = str(tmp_path / "a" / "runs" / "seed-000000")
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", disk_io="nowf",
                 pw_cmd=_fake_pwx(tmp_path / "b", body),
                 startpot_file=True, density_source=source_dir),
        run_root=tmp_path / "b" / "runs")
    engine.compute(_si(), label="first")
    attempt_dir = tmp_path / "b" / "runs" / "first-000000"
    manifest = json.loads((attempt_dir / "density_manifest.json").read_text())
    assert manifest["density_available"] is True
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["via"] == "previous attempt"
    assert engine.last_density_decision["origin"] == str(attempt_dir)


def test_spin_and_charge_reach_the_written_input(tmp_path: Path) -> None:
    capture = "#!/bin/bash\ncp espresso.pwi captured.pwi\ncat " f"{FIXTURE.resolve()}\n"
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, capture)),
        run_root=tmp_path / "runs",
    )
    atoms = _si()
    atoms.set_initial_magnetic_moments([1.0, -1.0])
    atoms.set_initial_charges([0.5, -0.25])
    engine.compute(atoms, label="spin")
    text = (tmp_path / "runs" / "spin-000000" / "espresso.pwi").read_text()
    assert "nspin" in text and "2" in text
    assert "starting_magnetization(1)" in text
    assert "starting_magnetization(2)" in text
    assert "tot_charge" in text


def test_noncollinear_magmoms_rejected_before_launch(tmp_path: Path) -> None:
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat())),
        run_root=tmp_path / "runs",
    )
    atoms = _si()
    atoms.set_initial_magnetic_moments([[1.0, 0, 0], [0, 1.0, 0]])
    with pytest.raises(_EngineError, match="noncollinear"):
        engine.compute(atoms, label="nc")
    assert not (tmp_path / "runs" / "nc-000000" / "espresso.pwi").exists()


def test_attempt_events_emitted_per_execution(tmp_path: Path) -> None:
    counter = tmp_path / "calls.txt"
    flaky = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 1 ]; then echo "launcher hiccup"; exit 139; fi\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    with _EventLog(tmp_path / "run") as log:
        engine = AseQeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, flaky),
                     max_retries=1),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        engine.compute(_si(), label="flaky", request_id="run-task-3")
    with _EventLog(tmp_path / "run") as log:
        events = [e for e in log.iter_events() if e.get("type") == "attempt"]
    assert [e["status"] for e in events] == ["failed", "success"]
    assert {e["request_id"] for e in events} == {"run-task-3"}
    assert all(e["record"] == "physical_attempt" for e in events)


def test_identical_geometry_still_executes_every_compute(tmp_path: Path) -> None:
    """ASE's geometry cache must never turn a reference execution into a
    silent cache hit: two computes of the same state launch two processes."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    first = engine.compute(_si())
    second = engine.compute(_si())
    assert counter.read_text().strip() == "2"
    assert first.energy == pytest.approx(second.energy)


# --- attempt sink discipline and execution boundaries (review B2/B3) --------


def test_request_id_without_sink_rejected_before_any_launch(tmp_path: Path) -> None:
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = AseQeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(_EngineError, match="attempt sink"):
        engine.compute(_si(), label="x", request_id="run-task-1")
    assert not counter.exists()
    assert not (tmp_path / "runs" / "x-000000").exists()


def test_missing_stress_executes_exactly_once(tmp_path: Path) -> None:
    """Review F3 evidence A: a complete-but-stressless output used to make
    ASE re-execute under the stress getter — two launches, one record. One
    explicit execution now reads the whole result; the label is rejected
    with exactly one launch and one failed attempt."""
    text = FIXTURE.read_text()
    cut = text[: text.index("total   stress  (Ry/bohr**3)")] + "\nJOB DONE.\n"
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        f"cat <<'EOF'\n{cut}\nEOF\n"
    )
    with _EventLog(tmp_path / "run") as log:
        engine = AseQeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                     max_retries=0),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        with pytest.raises(_EngineError, match="stress"):
            engine.compute(_si(), label="nostress")
    assert counter.read_text().strip() == "1"  # one real launch, not two
    with _EventLog(tmp_path / "run") as log:
        events = [e for e in log.iter_events() if e.get("type") == "attempt"]
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["failure_kind"] == "parse"


def test_missing_executable_is_zero_launches(tmp_path: Path) -> None:
    """Review F3 evidence B: a nonexistent executable never starts a
    process — no attempt event, terminal record, contract EngineError."""
    with _EventLog(tmp_path / "run") as log:
        engine = AseQeEngine(
            QeConfig(pseudo_dir="/pseudo",
                     pw_cmd=(str(tmp_path / "does-not-exist"),), max_retries=0),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        with pytest.raises(_EngineError, match="executable not found"):
            engine.compute(_si(), label="noexe")
    with _EventLog(tmp_path / "run") as log:
        events = [e for e in log.iter_events() if e.get("type") == "attempt"]
    assert events == []
    record = engine.last_attempt_records[-1]
    assert record["status"] == "failed"
    assert record["failure_kind"] == "executable_missing"


def test_manifest_write_failure_ends_post_processing_failed(tmp_path: Path) -> None:
    """Review F5 (ASE path): process and parse succeeded, the provenance
    sidecar could not be written — the attempt ends post_processing_failed,
    not running, and the event exists."""
    body = (
        "#!/bin/bash\n"
        "mkdir density_manifest.json\n"
        f"cat {FIXTURE.resolve()}\n"
    )
    with _EventLog(tmp_path / "run") as log:
        engine = AseQeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                     max_retries=0),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        with pytest.raises(_EngineError, match="density manifest"):
            engine.compute(_si(), label="pp")
    with _EventLog(tmp_path / "run") as log:
        events = [e for e in log.iter_events() if e.get("type") == "attempt"]
    assert len(events) == 1
    assert events[0]["status"] == "post_processing_failed"
    assert engine.last_attempt_records[-1]["status"] == "post_processing_failed"
    assert engine.last_attempt_records[-1]["error"]


def test_density_copy_interval_is_nested_inside_the_attempt_span(tmp_path: Path) -> None:
    with_save = (
        "#!/bin/bash\n"
        "mkdir -p tmp/pyraimd2.save && echo fake-density > tmp/pyraimd2.save/charge-density.dat\n"
        f"cat {FIXTURE.resolve()}\n"
    )
    with _EventLog(tmp_path / "run") as log:
        engine = AseQeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, with_save),
                     startpot_file=True),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        engine.compute(_si(), label="first")
        engine.compute(_si(), label="second")
    with _EventLog(tmp_path / "run") as log:
        events = list(log.iter_events())
    io = next(e for e in events if e.get("record") == "physical_io")
    attempt = next(e for e in events if e.get("type") == "attempt"
                   and e.get("request_id") == io["request_id"])
    io_end = io["started_unix"] + io["elapsed_s"]
    attempt_end = attempt["started_unix"] + attempt["elapsed_s"]
    assert attempt["started_unix"] <= io["started_unix"] <= io_end
    assert io_end <= attempt_end + 1e-6
    assert attempt["process_elapsed_s"] is not None


def test_manifest_write_time_is_settled_once_in_the_attempt_span(
    tmp_path: Path, monkeypatch,
) -> None:
    """The attempt span claims staging + process + validation: a slow
    density-manifest write must be counted in it exactly once (on both the
    success and the post-processing-failure exits), never frozen out before
    the write and never added twice downstream."""
    import time

    import pyraimd2.engines.ase_qe as ase_qe_module

    real_write = ase_qe_module.write_density_manifest
    delay_s = 0.15

    def slow_manifest(*args, **kwargs):
        time.sleep(delay_s)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(ase_qe_module, "write_density_manifest", slow_manifest)
    with _EventLog(tmp_path / "run") as log:
        engine = AseQeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat())),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        result = engine.compute(_si(), label="slow")
    with _EventLog(tmp_path / "run") as log:
        events = [e for e in log.iter_events() if e.get("type") == "attempt"]
    assert len(events) == 1 and events[0]["status"] == "success"
    span = events[0]["elapsed_s"]
    process = events[0]["process_elapsed_s"]
    assert span >= delay_s
    # The write is inside the span, on top of the pure process time — once.
    assert span >= process + 0.9 * delay_s
    assert result.wall_time_s == pytest.approx(span, rel=1e-6)
