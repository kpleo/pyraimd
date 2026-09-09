"""QeEngine reliability: explicit recipe, reference identity, combined SCF
success checks, classified bounded retries, process-group cleanup, and
density warm start — all hermetic (fake pw.x scripts, no real QE)."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import numpy as np
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


# --- electronic state: mapped explicitly or rejected before launch --------


def test_collinear_spin_mapped_like_the_ase_path(tmp_path: Path) -> None:
    """Initial magmoms must reach the input: nspin=2, one species per
    (element, magmom) group, starting_magnetization per species — the same
    mapping ASE's espresso writer produces for the same Atoms."""
    atoms = _si()
    atoms.set_initial_magnetic_moments([1.0, -1.0])
    out = tmp_path / "pw.in"
    write_qe_input(out, atoms, QeConfig(pseudo_dir="/pseudo"))
    text = out.read_text()
    assert "nspin = 2" in text
    assert "ntyp = 2" in text
    assert "starting_magnetization(1) = 1.0" in text
    assert "starting_magnetization(2) = -1.0" in text
    species = text.split("ATOMIC_SPECIES\n")[1].split("CELL_PARAMETERS")[0]
    assert "Si " in species and "Si2 " in species
    position_lines = [line.strip() for line in
                      text.split("ATOMIC_POSITIONS angstrom\n")[1].splitlines()]
    assert position_lines[0].startswith("Si ")
    assert position_lines[1].startswith("Si2 ")


def test_zero_magmoms_keep_single_species(tmp_path: Path) -> None:
    atoms = _si()
    atoms.set_initial_magnetic_moments([0.0, 0.0])
    out = tmp_path / "pw.in"
    write_qe_input(out, atoms, QeConfig(pseudo_dir="/pseudo"))
    text = out.read_text()
    assert "nspin" not in text and "ntyp = 1" in text


def test_noncollinear_magmoms_rejected_before_launch(tmp_path: Path) -> None:
    """An electronic state this input path cannot express must fail before
    any subprocess, never degrade into a different physical system."""
    counter = tmp_path / "calls.txt"
    body = ("#!/bin/bash\n"
            f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
            f"cat {FIXTURE.resolve()}\n")
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    atoms = _si()
    atoms.set_initial_magnetic_moments([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(EngineError, match="noncollinear"):
        engine.compute(atoms, label="nc")
    assert not counter.exists()  # nothing was ever launched


def test_net_charge_mapped_to_tot_charge(tmp_path: Path) -> None:
    atoms = _si()
    atoms.set_initial_charges([0.5, -0.25])
    out = tmp_path / "pw.in"
    write_qe_input(out, atoms, QeConfig(pseudo_dir="/pseudo"))
    assert "tot_charge = 0.25" in out.read_text()


# --- stricter parser / success contract -------------------------------------


def test_duplicate_force_atom_index_rejected(tmp_path: Path) -> None:
    """A force block repeating atom 1 and skipping atom 2 (right line count,
    wrong indices) does not describe this system: reject, never relabel."""
    import re as _re

    text = _re.sub(r"(atom\s+)2(\s+type)", r"\g<1>1\2", FIXTURE.read_text())
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_fake_pwx(tmp_path, f"#!/bin/bash\ncat <<'EOF'\n{text}\nEOF\n"),
                 max_retries=0),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="indices"):
        engine.compute(_si(), label="dup")


def test_overflow_stress_rejected_as_engine_error(tmp_path: Path) -> None:
    """********** overflow markers must fail as an invalid label — never a
    raw ValueError past the engine boundary."""
    import re as _re

    text, n = _re.subn(r"(total   stress[^\n]*\n\s*)\S+", r"\g<1>**********",
                       FIXTURE.read_text(), count=1)
    assert n == 1
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_fake_pwx(tmp_path, f"#!/bin/bash\ncat <<'EOF'\n{text}\nEOF\n"),
                 max_retries=0),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="unparseable|parse"):
        engine.compute(_si(), label="badstress")
    assert engine.last_attempt_records[-1]["status"] == "failed"


def test_missing_stress_block_rejected(tmp_path: Path) -> None:
    """tstress is always requested and capabilities declare stress: a
    completed run without the stress block is incomplete, not stress=None."""
    text = FIXTURE.read_text()
    cut = text[: text.index("total   stress  (Ry/bohr**3)")] + "\nJOB DONE.\n"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_fake_pwx(tmp_path, f"#!/bin/bash\ncat <<'EOF'\n{cut}\nEOF\n"),
                 max_retries=0),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="stress block"):
        engine.compute(_si(), label="nostress")


# --- retry classification ----------------------------------------------------


def test_nonconvergence_with_nonzero_exit_is_not_retried(tmp_path: Path) -> None:
    """convergence NOT achieved + exit 1 is still deterministic: the exit
    code must not reclassify it as transient."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        "echo 'convergence NOT achieved after 200 iterations'\n"
        "exit 1\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body), max_retries=3),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="did not converge"):
        engine.compute(_si(), label="nc")
    assert counter.read_text().strip() == "1"
    assert engine.last_attempt_records[0]["retryable"] is False


# --- path normalization -------------------------------------------------------


def test_relative_pseudo_dir_resolved_once_at_construction(tmp_path: Path,
                                                           monkeypatch) -> None:
    """The fingerprint hashes pseudo files in the parent; pw.x reads them in
    the attempt directory. A relative pseudo_dir must be resolved once, at
    construction, so both see the same directory."""
    pseudo_dir = tmp_path / "project" / "pseudos"
    pseudo_dir.mkdir(parents=True)
    (pseudo_dir / "Si.UPF").write_text("fake UPF content")
    reader = (
        "#!/bin/bash\n"
        'in=""; while [ $# -gt 0 ]; do '
        'if [ "$1" = "-in" ]; then in="$2"; shift 2; else shift; fi; done\n'
        "dir=$(sed -n \"s/.*pseudo_dir = '\\([^']*\\)'.*/\\1/p\" \"$in\")\n"
        '[ -f "$dir/Si.UPF" ] || { echo "pseudo unreadable in subprocess"; exit 9; }\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    monkeypatch.chdir(tmp_path / "project")
    engine = QeEngine(
        QeConfig(pseudo_dir="pseudos", pseudos={"Si": "Si.UPF"},
                 pw_cmd=_fake_pwx(tmp_path / "fake", reader)),
        run_root=tmp_path / "runs",
    )
    assert Path(engine.config.pseudo_dir).is_absolute()
    result = engine.compute(_si(), label="rel")
    assert result.forces.shape == (2, 3)
    # The hashed identity and the launched input point at the same files.
    assert pseudo_identities(engine.config)["Si"]["sha256"] is not None
    pw_in = next((tmp_path / "runs").glob("rel-*/attempt-1/pw.in"))
    assert str(pseudo_dir.resolve()) in pw_in.read_text()


# --- attempt events: every real launch, nothing pre-launch --------------------


def _attempt_events(log_dir: Path) -> list[dict]:
    with EventLog(log_dir) as log:
        return [e for e in log.iter_events() if e.get("type") == "attempt"]


def test_attempt_event_per_real_launch_grouped_by_request(tmp_path: Path) -> None:
    """First attempt fails transiently, second succeeds: two physical
    executions under one logical request (ledger raw material: actual=2,
    failed=1 alongside the workflow's logical span=1)."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 1 ]; then echo "launcher hiccup"; exit 139; fi\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    with EventLog(tmp_path / "run") as log:
        engine = QeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                     max_retries=1),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        result = engine.compute(_si(), label="flaky", request_id="run-task-7")
    events = _attempt_events(tmp_path / "run")
    assert len(events) == 2
    assert [e["status"] for e in events] == ["failed", "success"]
    assert {e["request_id"] for e in events} == {"run-task-7"}
    assert [e["attempt"] for e in events] == [1, 2]
    assert all(e["record"] == "physical_attempt" for e in events)
    assert events[0]["returncode"] == 139
    assert all(e["elapsed_s"] is not None and e["elapsed_s"] >= 0 for e in events)
    assert events[0]["error"] and events[1]["error"] is None
    # wall_time_s is the physical total across attempts, not only the last.
    total = sum(e["elapsed_s"] for e in events)
    assert result.wall_time_s == pytest.approx(total, rel=1e-6)


def test_pre_launch_rejection_emits_no_attempt_event(tmp_path: Path) -> None:
    """A failure rejected before any process starts (unknown species here)
    is not a physical execution and must not be counted as one."""
    with EventLog(tmp_path / "run") as log:
        engine = QeEngine(
            QeConfig(pseudo_dir="/pseudo", pseudos={"Si": "Si.UPF"},
                     pw_cmd=_fake_pwx(tmp_path, _fixture_cat())),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        with pytest.raises(EngineError, match="no pseudopotential"):
            engine.compute(Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]],
                                 cell=[8.0] * 3, pbc=True))
    assert _attempt_events(tmp_path / "run") == []
    assert engine.last_attempt_records[-1]["status"] == "failed"


def test_density_io_event_carries_request_and_nesting_marker(tmp_path: Path) -> None:
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
    io_events = [e for e in events if e.get("type") == "task"
                 and e.get("purpose") == "density_copy"]
    assert len(io_events) == 1
    assert io_events[0]["record"] == "physical_io"
    assert io_events[0]["request_id"]
    attempts = [e for e in events if e.get("type") == "attempt"]
    assert len(attempts) == 2
    # The io event is nested inside the second attempt's request span.
    assert io_events[0]["request_id"] == attempts[1]["request_id"]


def test_failed_attempt_record_never_left_running(tmp_path: Path) -> None:
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_fake_pwx(tmp_path, "#!/bin/bash\necho garbage; exit 3\n"),
                 max_retries=0),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError):
        engine.compute(_si(), label="x")
    record = engine.last_attempt_records[-1]
    assert record["status"] == "failed" and record["error"]


def test_logical_vs_physical_aggregation_convention(tmp_path: Path) -> None:
    """The ledger contract the core consumer aggregates: outer task spans
    (type "task", operation "reference") are logical requests; engine
    attempt events (type "attempt", record "physical_attempt") are physical
    executions. Fail-then-succeed must read logical=1, actual=2, failed=1 —
    the engine emits exactly the raw material for it."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 1 ]; then echo "launcher hiccup"; exit 139; fi\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    with EventLog(tmp_path / "run") as log:
        engine = QeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                     max_retries=1),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        # What a workflow does around one logical request (outer span).
        started = time.time()
        result = engine.compute(_si(), label="x", request_id="run-task-1")
        log.append("task", {
            "task_id": "run-task-1", "attempt": 1, "operation": "reference",
            "purpose": "singlepoint", "status": "success",
            "started_unix": started, "elapsed_s": 0.0,
            "cpu_cores": None, "gpu": None, "queue_s": None,
            "source": "workflow", "evaluation_id": 0,
            "label_id": None, "cache_hit": False})
    with EventLog(tmp_path / "run") as log:
        events = list(log.iter_events())
    logical = [e for e in events if e.get("type") == "task"
               and e.get("operation") == "reference"]
    physical = [e for e in events if e.get("type") == "attempt"
                and e.get("record") == "physical_attempt"]
    assert len(logical) == 1
    assert len(physical) == 2
    assert sum(e["status"] == "failed" for e in physical) == 1
    assert all(e["request_id"] == logical[0]["task_id"] for e in physical)
    assert result.wall_time_s == pytest.approx(
        sum(e["elapsed_s"] for e in physical), rel=1e-6)


# --- attempt sink discipline (review B2) ------------------------------------


def test_request_id_without_sink_rejected_before_any_launch(tmp_path: Path) -> None:
    """A caller naming a parent request expects per-launch events to reach
    its ledger; without a sink they would vanish — reject pre-launch."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body)),
        run_root=tmp_path / "runs",
    )
    with pytest.raises(EngineError, match="attempt sink"):
        engine.compute(_si(), label="x", request_id="run-task-1")
    assert not counter.exists()  # nothing launched
    assert not (tmp_path / "runs" / "x-000000").exists()  # no side effects


def test_runner_log_connects_sink_to_sinkless_engine(tmp_path: Path) -> None:
    """The review's B2 combination: EnergeticRunner with an event log around
    a QeEngine constructed without one used to undercount silently. The
    runner now explicitly connects its log as the attempt sink for each
    call (and restores it afterwards), so internal retries stay visible."""
    from pyraimd2.loop import EnergeticRunner
    from pyraimd2.store import Store
    from pyraimd2.surrogate.base import SurrogatePrediction

    class TinySurrogate:
        def predict(self, atoms):
            return SurrogatePrediction(
                0.5 * float(np.sum(atoms.positions**2)), -atoms.positions, None,
                np.full(len(atoms), np.nan))

    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 1 ]; then echo garbage; exit 3; fi\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = QeEngine(  # no event_log of its own: sink connected per call
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                 startpot_file=True),  # enables the atomic-start retry
        run_root=tmp_path / "qe",
    )
    atoms = _si()
    atoms.set_velocities(np.zeros((2, 3)))
    store = Store(tmp_path / "run.db")
    with EventLog(tmp_path / "run") as log:
        runner = EnergeticRunner(atoms, TinySurrogate(), engine, store, "run",
                                 force_budget=0.1, timestep_fs=0.1,
                                 event_log=log)
        runner.run(1)
    # Zero velocities take the reference route without probes: evaluation 0
    # launches twice (first launch failed, atomic-start retry), evaluation 1
    # launches once. Every launch lands on the ledger.
    assert counter.read_text().strip() == "3"
    attempts = [event for event in log.iter_events()
                if event.get("record") == "physical_attempt"]
    assert [a["status"] for a in attempts] == ["failed", "success", "success"]
    assert all(a["request_id"] for a in attempts)
    assert engine._event_log is None  # sink restored after the call


def test_local_records_consumable_without_sink(tmp_path: Path) -> None:
    """Without any ledger the engine still keeps a complete per-launch
    record (started/status/elapsed/returncode) a caller can consume."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        'if [ "$n" -eq 1 ]; then echo "launcher hiccup"; exit 139; fi\n'
        f"cat {FIXTURE.resolve()}\n"
    )
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                 max_retries=1),
        run_root=tmp_path / "runs",
    )
    engine.compute(_si())
    records = engine.last_attempt_records
    assert [r["status"] for r in records] == ["failed", "success"]
    assert all(r["started_unix"] > 0 for r in records)
    assert all(r["wall_time_s"] >= 0 for r in records)
    assert records[0]["returncode"] == 139 and records[1]["returncode"] == 0
    assert records[0]["failure_kind"] == "process"


# --- zero launches and terminal states (review B3 / F3-F5) -------------------


def test_missing_executable_is_zero_launches(tmp_path: Path) -> None:
    """A nonexistent executable never starts a process: no attempt event,
    and the failure surfaces as a contract EngineError, not a raw
    FileNotFoundError (review F4's engine side)."""
    with EventLog(tmp_path / "run") as log:
        engine = QeEngine(
            QeConfig(pseudo_dir="/pseudo",
                     pw_cmd=(str(tmp_path / "does-not-exist"),), max_retries=3),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        with pytest.raises(EngineError, match="executable not found"):
            engine.compute(_si(), label="noexe")
    assert _attempt_events(tmp_path / "run") == []
    record = engine.last_attempt_records[-1]
    assert record["status"] == "failed"
    assert record["failure_kind"] == "executable_missing"
    assert record["retryable"] is False  # a missing binary is deterministic


def test_timeout_attempt_ends_killed(tmp_path: Path) -> None:
    body = "#!/bin/bash\nsleep 30\n"
    with EventLog(tmp_path / "run") as log:
        engine = QeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                     timeout_s=0.5, max_retries=0),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        with pytest.raises(EngineError, match="timed out"):
            engine.compute(_si(), label="hang")
    events = _attempt_events(tmp_path / "run")
    assert len(events) == 1
    assert events[0]["status"] == "killed"
    assert events[0]["failure_kind"] == "timeout"
    assert engine.last_attempt_records[-1]["status"] == "killed"


def test_manifest_write_failure_ends_post_processing_failed(tmp_path: Path) -> None:
    """The process succeeded and parsed; the provenance sidecar could not be
    written. The attempt must still be terminated and recorded — distinctly
    from a process/parse failure (review F5)."""
    body = (
        "#!/bin/bash\n"
        "mkdir density_manifest.json\n"  # a directory: writing the file fails
        f"cat {FIXTURE.resolve()}\n"
    )
    with EventLog(tmp_path / "run") as log:
        engine = QeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                     max_retries=3),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        with pytest.raises(EngineError, match="density manifest"):
            engine.compute(_si(), label="pp")
    events = _attempt_events(tmp_path / "run")
    assert len(events) == 1  # one real launch, one terminated record
    assert events[0]["status"] == "post_processing_failed"
    assert events[0]["failure_kind"] == "post_processing"
    record = engine.last_attempt_records[-1]
    assert record["status"] == "post_processing_failed"
    assert record["error"]  # original OSError preserved


def test_density_copy_interval_is_nested_inside_the_attempt_span(tmp_path: Path) -> None:
    """The copy happens inside the attempt span by construction; the io
    event's real interval must lie within the attempt's (review F7)."""
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
    io = next(e for e in events if e.get("record") == "physical_io")
    attempt = next(e for e in events if e.get("type") == "attempt"
                   and e.get("request_id") == io["request_id"])
    io_end = io["started_unix"] + io["elapsed_s"]
    attempt_end = attempt["started_unix"] + attempt["elapsed_s"]
    assert attempt["started_unix"] <= io["started_unix"] <= io_end
    assert io_end <= attempt_end + 1e-6  # nested: never a missing interval
    assert attempt["process_elapsed_s"] is not None


def test_unreadable_output_is_a_terminal_traceable_failure(tmp_path: Path) -> None:
    """The fake starts, replaces its stdout path with a directory, writes
    the fixture into the still-open fd and exits 0. The engine's output
    read then fails — and the launched attempt must still end in a
    terminal, traceable state: one launch, one failed attempt event with
    the read failure as cause, no running record, no UnboundLocalError."""
    counter = tmp_path / "calls.txt"
    body = (
        "#!/bin/bash\n"
        f'n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}\n'
        "rm -f pw.out && mkdir pw.out\n"
        f"cat {FIXTURE.resolve()}\n"
    )
    with EventLog(tmp_path / "run") as log:
        engine = QeEngine(
            QeConfig(pseudo_dir="/pseudo", pw_cmd=_fake_pwx(tmp_path, body),
                     max_retries=0),
            run_root=tmp_path / "runs",
            event_log=log,
        )
        with pytest.raises(EngineError) as excinfo:
            engine.compute(_si(), label="unreadable")
    assert counter.read_text().strip() == "1"  # exactly one launch
    error = excinfo.value
    assert isinstance(error, EngineError)  # contract error, never UnboundLocalError
    assert "read" in str(error)
    assert isinstance(error.__cause__, IsADirectoryError)  # cause preserved
    events = _attempt_events(tmp_path / "run")
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["failure_kind"] == "read"
    assert events[0]["error"]  # never an empty error on a terminated attempt
    record = engine.last_attempt_records[-1]
    assert record["status"] == "failed" and record["failure_kind"] == "read"
    assert record["error"] and record["returncode"] == 0
