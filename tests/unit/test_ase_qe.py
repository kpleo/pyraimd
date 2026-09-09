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
