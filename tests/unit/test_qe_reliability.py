"""QeEngine reliability: explicit recipe, reference identity, combined SCF
success checks, classified bounded retries, process-group cleanup, and
density warm start — all hermetic (fake pw.x scripts, no real QE)."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest
from ase import Atoms, units

from pyraimd2.engines.base import EnergyKind, EngineError
from pyraimd2.engines.qe_engine import (
    DENSITY_MANIFEST,
    QeConfig,
    QeEngine,
    load_density_source,
    pseudo_identities,
    write_qe_input,
)
from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import EventLog

FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"
SI_ENERGY_RY = -93.43942921


def _si() -> Atoms:
    return Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                 cell=[5.43] * 3, pbc=True)


def _fake_pwx(tmp_path: Path, body: str) -> tuple[str, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_pwx.sh"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return ("bash", str(script))


def _fixture_cat() -> str:
    return f"#!/bin/bash\ncat {FIXTURE.resolve()}\n"


def _fixture_cat_with_save() -> str:
    return (
        "#!/bin/bash\n"
        "mkdir -p tmp/pyraimd2.save && echo fake-density > tmp/pyraimd2.save/charge-density.dat\n"
        f"cat {FIXTURE.resolve()}\n"
    )


# --- explicit XC / dispersion recipe -------------------------------------


def test_recipe_is_explicit_in_input_and_name(tmp_path: Path) -> None:
    cfg = QeConfig(pseudo_dir="/pseudo")
    assert cfg.xc == "pbe" and cfg.dispersion == "grimme-d3"
    engine = QeEngine(cfg, run_root=tmp_path / "runs")
    assert engine.name == "qe-pbe-d3"  # default recipe unchanged
    out = tmp_path / "pw.in"
    write_qe_input(out, _si(), cfg)
    text = out.read_text()
    assert "input_dft = 'pbe'" in text
    assert "vdw_corr = 'grimme-d3'" in text


def test_pbe_only_recipe_must_drop_dispersion_explicitly(tmp_path: Path) -> None:
    cfg = QeConfig(pseudo_dir="/pseudo", dispersion=None)
    engine = QeEngine(cfg, run_root=tmp_path / "runs")
    assert engine.name == "qe-pbe"  # plainly PBE, no hidden D3
    out = tmp_path / "pw.in"
    write_qe_input(out, _si(), cfg)
    text = out.read_text()
    assert "input_dft = 'pbe'" in text
    assert "vdw_corr" not in text


def test_other_recipes_are_named_plainly(tmp_path: Path) -> None:
    engine = QeEngine(QeConfig(pseudo_dir="/pseudo", xc="blyp", dispersion=None),
                      run_root=tmp_path / "runs")
    assert engine.name == "qe-blyp"
    out = tmp_path / "pw.in"
    write_qe_input(out, _si(), engine.config)
    assert "input_dft = 'blyp'" in out.read_text()
    assert engine.fingerprint != QeEngine(
        QeConfig(pseudo_dir="/pseudo"), run_root=tmp_path / "r2").fingerprint


# --- reference identity: pseudopotential content ---------------------------


def test_pseudo_identity_includes_content_hash(tmp_path: Path) -> None:
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    upf = pseudo_dir / "Si.fake.UPF"
    upf.write_text("<PP_HEADER z_valence=\"4.0\"/>\n")
    cfg = QeConfig(pseudo_dir=str(pseudo_dir), pseudos={"Si": "Si.fake.UPF"})
    identity = pseudo_identities(cfg)
    assert identity["Si"]["file"] == "Si.fake.UPF"
    assert isinstance(identity["Si"]["sha256"], str) and identity["Si"]["sha256"]
    first = QeEngine(cfg, run_root=tmp_path / "r1").fingerprint
    upf.write_text("<PP_HEADER z_valence=\"4.0\"/>\n<!-- changed -->\n")
    second = QeEngine(cfg, run_root=tmp_path / "r2").fingerprint
    assert first != second  # same file name, different content -> different identity


def test_pseudo_identity_degrades_honestly_when_unreadable(tmp_path: Path) -> None:
    cfg = QeConfig(pseudo_dir="/nonexistent")
    identity = pseudo_identities(cfg)
    assert all(entry["sha256"] is None for entry in identity.values())
    engine = QeEngine(cfg, run_root=tmp_path / "r1")
    # Degraded is still stable and recorded — never invented.
    assert engine.fingerprint == QeEngine(cfg, run_root=tmp_path / "r2").fingerprint
    assert engine.fingerprint != QeEngine(
        QeConfig(pseudo_dir="/nonexistent", pseudos={"Si": "Other.UPF"}),
        run_root=tmp_path / "r3").fingerprint


def test_platform_fields_do_not_change_reference_identity(tmp_path: Path) -> None:
    base = QeConfig(pseudo_dir="/pseudo")
    same_physics = QeConfig(pseudo_dir="/pseudo", pw_cmd=("mpirun", "-np", "4", "pw.x"),
                            timeout_s=10.0, max_retries=5, startpot_file=True,
                            density_source="/somewhere")
    assert QeEngine(base, run_root=tmp_path / "a").fingerprint == \
        QeEngine(same_physics, run_root=tmp_path / "b").fingerprint


# --- energy convention identity -------------------------------------------


def test_metallic_config_declares_free_energy(tmp_path: Path) -> None:
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat()),
                 metallic=True),
        run_root=tmp_path / "runs",
    )
    assert engine.capabilities.energy_kind == EnergyKind.FREE_ENERGY
    assert engine.capabilities.force_consistent is True
    result = engine.compute(_si(), label="metal")
    assert result.energy_kind == EnergyKind.FREE_ENERGY
    assert result.force_consistent is True


def test_insulator_config_declares_total_energy(tmp_path: Path) -> None:
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat())),
        run_root=tmp_path / "runs",
    )
    result = engine.compute(_si(), label="bulk")
    assert result.energy_kind == EnergyKind.ENERGY
    assert result.force_consistent is True


# --- combined SCF success determination ------------------------------------


def test_clean_exit_without_job_done_is_not_a_success(tmp_path: Path) -> None:
    """Exit code 0 alone never makes a label: a truncated output (killed
    job, full disk) must be rejected even when pw.x exits cleanly."""
    body = (
        "#!/bin/bash\n"
        f"head -n 300 {FIXTURE.resolve()}\n"  # energy+forces, no JOB DONE.
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), max_retries=0),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="JOB DONE"):
        engine.compute(_si(), label="trunc")


def test_qe_error_banner_is_not_retried(tmp_path: Path) -> None:
    """A deterministic input/physics error (QE error banner) fails once —
    rerunning the identical input would only burn cost."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        "echo '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'\n"
        "echo '     Error in routine read_namelists (1):'\n"
        "echo '     bad line in namelist &control'\n"
        "echo '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'\n"
        "exit 1\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), max_retries=3),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="exited with code 1"):
        engine.compute(_si(), label="badinput")
    assert counter.read_text().strip() == "1"


def test_nonconvergence_of_identical_input_is_not_retried(tmp_path: Path) -> None:
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        f"sed 's/convergence has been achieved/convergence NOT achieved/' {FIXTURE.resolve()}\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), max_retries=3),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="did not converge"):
        engine.compute(_si(), label="nc")
    assert counter.read_text().strip() == "1"


def test_transient_crash_is_retried_within_budget(tmp_path: Path) -> None:
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 1 ]; then echo "mpirun noticed that process rank 3 exited"; exit 139; fi\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), max_retries=1),
        run_root=tmp_path / "runs",
    )
    result = engine.compute(_si(), label="flaky")
    assert result.energy == pytest.approx(SI_ENERGY_RY * units.Hartree / 2.0, abs=1e-6)
    assert counter.read_text().strip() == "2"
    attempts = engine.last_attempt_records
    assert [a["status"] for a in attempts] == ["failed", "success"]
    assert attempts[0]["retryable"] is True


def test_retry_budget_is_bounded(tmp_path: Path) -> None:
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        "echo segfault; exit 139\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), max_retries=1),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="exited with code 139"):
        engine.compute(_si(), label="always")
    assert counter.read_text().strip() == "2"  # first attempt + one bounded retry


# --- timeout: the whole process group is cleaned up -------------------------


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    """A hung run must not leave orphaned children: the grandchild spawned
    by pw.x (mpirun ranks in production) is SIGKILLed with the group."""
    pid_file = tmp_path / "child.pid"
    body = (
        "#!/bin/bash\n"
        f"sleep 60 & echo $! > {pid_file}\n"
        "wait\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), timeout_s=0.5),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="timed out"):
        engine.compute(_si(), label="hang")
    child_pid = int(pid_file.read_text().strip())
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild process {child_pid} survived the timeout kill")


# --- density warm start ------------------------------------------------------


def _read_manifest(attempt_dir: Path) -> dict:
    return json.loads((attempt_dir / DENSITY_MANIFEST).read_text())


def test_successful_attempt_writes_density_manifest(tmp_path: Path) -> None:
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat_with_save())),
        run_root=tmp_path / "runs",
    )
    engine.compute(_si(), label="seed")
    manifest = _read_manifest(tmp_path / "runs" / "seed-000000" / "attempt-1")
    assert manifest["reference_fingerprint"] == engine.fingerprint
    assert manifest["nat"] == 2 and manifest["species"] == ["Si"]
    assert manifest["source"] == {"kind": "atomic"}
    assert manifest["save_dir"] == "tmp/pyraimd2.save"


def test_warm_start_copies_density_and_records_provenance(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat_with_save()),
                 startpot_file=True),
        run_root=run_root,
    )
    si = _si()
    engine.compute(si, label="first")
    result = engine.compute(si, label="second")
    assert result.energy == pytest.approx(SI_ENERGY_RY * units.Hartree / 2.0, abs=1e-6)

    decision = engine.last_density_decision
    assert decision["start"] == "density"
    source_save = run_root / "first-000000" / "attempt-1" / "tmp" / "pyraimd2.save"
    copied_save = run_root / "second-000001" / "attempt-1" / "tmp" / "pyraimd2.save"
    assert decision["origin"] == str(run_root / "first-000000" / "attempt-1")
    # The copy is a real, separate tree: no shared writable .save.
    assert copied_save.is_dir() and source_save.is_dir()
    assert copied_save != source_save
    assert (copied_save / "charge-density.dat").read_text() == \
        (source_save / "charge-density.dat").read_text()
    (copied_save / "charge-density.dat").write_text("mutated")
    assert (source_save / "charge-density.dat").read_text() == "fake-density\n"
    # Provenance: the attempt record and the new manifest say where it came from.
    record = engine.last_attempt_records[0]
    assert record["start"] == "density"
    assert record["density_from"] == str(run_root / "first-000000" / "attempt-1")
    assert record["density_copy_bytes"] > 0
    manifest = _read_manifest(run_root / "second-000001" / "attempt-1")
    assert manifest["source"]["kind"] == "copied"
    assert manifest["source"]["from"] == str(run_root / "first-000000" / "attempt-1")
    assert manifest["source"]["from_fingerprint"] == engine.fingerprint


def test_warm_start_without_density_goes_atomic_before_launch(tmp_path: Path) -> None:
    """startpot_file=True with no compatible density must not launch a
    startpot='file' run that is known to fail: the first attempt is already
    an atomic start."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                 startpot_file=True),
        run_root=tmp_path / "runs",
    )
    engine.compute(_si(), label="cold")
    assert counter.read_text().strip() == "1"  # no deliberate first failure
    pw_in = (tmp_path / "runs" / "cold-000000" / "attempt-1" / "pw.in").read_text()
    assert "startingpot" not in pw_in
    assert not (tmp_path / "runs" / "cold-000000" / "attempt-2").exists()
    assert engine.last_density_decision["start"] == "atomic"
    assert engine.last_attempt_records[0]["start"] == "atomic"


def test_incompatible_density_is_declined_with_reason(tmp_path: Path) -> None:
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, _fixture_cat_with_save())),
        run_root=tmp_path / "runs",
    )
    engine.compute(_si(), label="seed")
    other = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path / "other",
                                                        _fixture_cat_with_save()),
                 ecutwfc=99.0,  # different reference settings
                 startpot_file=True,
                 density_source=str(tmp_path / "runs" / "seed-000000" / "attempt-1")),
        run_root=tmp_path / "runs",
    )
    source, reason = load_density_source(
        tmp_path / "runs" / "seed-000000" / "attempt-1", engine=other, atoms=_si())
    assert source is None and "reference settings differ" in reason
    other.compute(_si(), label="cold")
    assert other.last_density_decision["start"] == "atomic"
    assert "reference settings differ" in other.last_density_decision["reason"]


def test_density_copy_is_charged_to_the_cost_ledger(tmp_path: Path) -> None:
    with EventLog(tmp_path / "run") as log:
        engine = QeEngine(
            QeConfig(pseudo_dir="/pseudo",
                     pw_cmd=_fake_pwx(tmp_path, _fixture_cat_with_save()),
                     startpot_file=True),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        engine.compute(_si(), label="first")
        engine.compute(_si(), label="second")
    with EventLog(tmp_path / "run") as log:
        events = list(log.iter_events())
    io_tasks = [e for e in events
                if e.get("type") == "task" and e.get("purpose") == "density_copy"]
    assert len(io_tasks) == 1
    task = io_tasks[0]
    assert task["operation"] == "io" and task["status"] == "success"
    assert task["elapsed_s"] >= 0.0
    assert task["provenance"]["bytes"] > 0
    assert "first-000000" in task["provenance"]["from"]
    summary = summarize_tasks(events)
    assert summary["counts"]["io"] == 1
