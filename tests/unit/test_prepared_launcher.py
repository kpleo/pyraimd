"""The prepared-QE-launcher example (examples/prepared_qe_launcher/):
fake mpirun/pw.x shell stubs only — no real QE, no Slurm, no network.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.engines.base import EngineError
from pyraimd2.engines.qe_engine import QeConfig, QeEngine

LAUNCHER = (Path(__file__).parents[2] / "examples" / "prepared_qe_launcher"
            / "launcher.py")
QE_FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"
JOB_ID = "test-job-42"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("prepared_launcher",
                                                  LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
               | stat.S_IXOTH)


def _fake_bin(tmp_path: Path, *, pw_body: str | None = None) -> Path:
    """bin/ with a fake mpirun (execs the resolved pw.x) and a fake pw.x
    that records its argv and environment to $PW_RECORD{,.env}."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    _write_stub(bin_dir / "mpirun", 'exec "${@:3}"\n')  # skip -np N
    _write_stub(bin_dir / "pw.x", pw_body if pw_body is not None else (
        'printf "argv:" >> "$PW_RECORD"; for a in "$@"; do '
        'printf " <%s>" "$a" >> "$PW_RECORD"; done; printf "\\n" >> '
        '"$PW_RECORD"\n'
        'env | sort > "$PW_RECORD.env"\n'
        'exit "${PW_EXIT:-0}"\n'))
    return bin_dir


def _setup_script(tmp_path: Path, bin_dir: Path, *, extra: str = "") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "setup.sh"
    script.write_text(
        f'export PATH="{bin_dir}:$PATH"\n'
        'export OMP_NUM_THREADS=2\n'
        'unset LD_LIBRARY_PATH\n'          # a removal the run must honor
        'echo "setup chatter on stdout"\n'  # must never enter the env data
        + extra)
    return script


def _prepare(tmp_path: Path, monkeypatch, *, extra: str = "") -> Path:
    """In-process prepare: strongest proof the parent env is untouched."""
    bin_dir = _fake_bin(tmp_path)
    setup = _setup_script(tmp_path, bin_dir, extra=extra)
    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/original/lib")
    launcher = _load_launcher()
    state = tmp_path / "state"
    code = launcher.main(["prepare", "--setup", str(setup),
                          "--state", str(state), "--mpi", "mpirun",
                          "--pw", "pw.x", "--ranks", "4"])
    assert code == 0
    return state


def _run(state: Path, digest: str, *extra_argv: str,
         job_id: str = JOB_ID, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = job_id
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(LAUNCHER), "run", "--state", str(state),
         "--expected-sha256", digest, "--", *extra_argv],
        capture_output=True, text=True, env=env, check=False)


def _digest(state: Path) -> str:
    return json.loads((state / "state.json").read_text())["payload_sha256"]


def test_prepare_once_two_runs_succeed(tmp_path, monkeypatch) -> None:
    state = _prepare(tmp_path, monkeypatch)
    digest = _digest(state)
    record = tmp_path / "record"
    for name in ("first.in", "second.in"):
        result = _run(state, digest, "-in", name,
                      env_extra={"PW_RECORD": str(record)})
        assert result.returncode == 0, result.stderr
    lines = record.read_text().splitlines()
    assert len(lines) == 2                    # one prepare, two runs
    assert lines[0] == "argv: <-in> <first.in>"
    assert lines[1] == "argv: <-in> <second.in>"
    payload = json.loads((state / "state.json").read_text())["payload"]
    assert payload["argv"][1:3] == ["-np", "4"]
    assert Path(payload["argv"][0]).name == "mpirun"
    assert Path(payload["argv"][3]).name == "pw.x"
    assert payload["slurm_job_id"] == JOB_ID


def test_removed_var_absent_in_child_and_parent_unchanged(tmp_path,
                                                          monkeypatch) -> None:
    state = _prepare(tmp_path, monkeypatch)
    # the parent process environment was never modified by prepare
    assert os.environ["LD_LIBRARY_PATH"] == "/original/lib"
    assert "OMP_NUM_THREADS" not in os.environ
    record = tmp_path / "record"
    result = _run(state, _digest(state), "-in", "pw.in",
                  env_extra={"PW_RECORD": str(record)})
    assert result.returncode == 0, result.stderr
    child_env = (Path(str(record) + ".env")).read_text()
    assert "LD_LIBRARY_PATH" not in child_env       # the removal was honored
    assert "OMP_NUM_THREADS=2" in child_env         # the prepared value applied
    assert "setup chatter" not in (state / "state.json").read_text()


def test_argv_and_paths_with_spaces_pass_through(tmp_path, monkeypatch) -> None:
    state = _prepare(tmp_path, monkeypatch)
    record = tmp_path / "record"
    spaced = "my input file.in"
    result = _run(state, _digest(state), "-in", spaced,
                  env_extra={"PW_RECORD": str(record)})
    assert result.returncode == 0, result.stderr
    # one argv word, verbatim — no shell re-splitting
    assert record.read_text().strip() == f"argv: <-in> <{spaced}>"


def test_wrong_job_or_changed_hash_refuses_before_spawn(tmp_path,
                                                        monkeypatch) -> None:
    state = _prepare(tmp_path, monkeypatch)
    digest = _digest(state)
    marker = tmp_path / "record"
    # wrong job id: refused before spawning; the error names ids only,
    # never environment values
    result = _run(state, digest, "-in", "pw.in", job_id="someone-elses-job",
                  env_extra={"PW_RECORD": str(marker)})
    assert result.returncode != 0
    assert "SLURM_JOB_ID" in result.stderr
    assert "/original/lib" not in result.stderr
    # wrong expected hash
    result = _run(state, "0" * 64, "-in", "pw.in",
                  env_extra={"PW_RECORD": str(marker)})
    assert result.returncode != 0
    assert "hash" in result.stderr
    # a tampered state file no longer matches its own recorded hash
    state_file = state / "state.json"
    record = json.loads(state_file.read_text())
    record["payload"]["argv"][3] = "/opt/other/pw.x"
    state_file.write_text(json.dumps(record))
    result = _run(state, digest, "-in", "pw.in",
                  env_extra={"PW_RECORD": str(marker)})
    assert result.returncode != 0
    assert "modified" in result.stderr
    assert not marker.exists()                    # the child never ran


def test_child_exit_code_and_sigint_propagate(tmp_path, monkeypatch) -> None:
    state = _prepare(tmp_path, monkeypatch)
    digest = _digest(state)
    result = _run(state, digest, "-in", "pw.in",
                  env_extra={"PW_RECORD": str(tmp_path / "r"),
                             "PW_EXIT": "3"})
    assert result.returncode == 3                 # no supervisor translation
    # SIGINT: a pw.x that dies by SIGINT shows as a signal death, not a
    # translated exit code
    bin_dir = tmp_path / "sigint-bin"
    bin_dir.mkdir()
    _write_stub(bin_dir / "mpirun", 'exec "${@:3}"\n')
    _write_stub(bin_dir / "pw.x", "kill -INT $$\n")
    setup = _setup_script(tmp_path, bin_dir)
    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    launcher = _load_launcher()
    sigint_state = tmp_path / "sigint-state"
    assert launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(sigint_state), "--mpi", "mpirun", "--pw",
                          "pw.x", "--ranks", "2"]) == 0
    result = _run(sigint_state, _digest(sigint_state), "-in", "pw.in")
    assert result.returncode == -2  # killed by signal 2 (SIGINT)


def test_prepare_refusals(tmp_path, monkeypatch, capsys) -> None:
    launcher = _load_launcher()
    bin_dir = _fake_bin(tmp_path)
    setup = _setup_script(tmp_path, bin_dir)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    code = launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(tmp_path / "nojob"), "--mpi", "mpirun",
                          "--pw", "pw.x", "--ranks", "1"])
    assert code == 2
    assert "SLURM_JOB_ID" in capsys.readouterr().err

    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    state = _prepare(tmp_path / "sub", monkeypatch)
    code = launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(state), "--mpi", "mpirun", "--pw", "pw.x",
                          "--ranks", "1"])
    assert code == 2                              # never overwrites a state
    assert "already exists" in capsys.readouterr().err

    # a credential-class variable added by the setup aborts by NAME ONLY
    secret_setup = _setup_script(tmp_path, bin_dir,
                                 extra='export AWS_SECRET_KEY="s3cret"\n')
    code = launcher.main(["prepare", "--setup", str(secret_setup),
                          "--state", str(tmp_path / "cred"), "--mpi",
                          "mpirun", "--pw", "pw.x", "--ranks", "1"])
    assert code == 2
    err = capsys.readouterr().err
    assert "AWS_SECRET_KEY" in err
    assert "s3cret" not in err                    # the value never leaks
    assert not (tmp_path / "cred" / "state.json").exists()


def test_qe_engine_through_prepared_launcher(tmp_path, monkeypatch) -> None:
    """QeEngine's existing pw_cmd path appends -in correctly through the
    prepared launcher and propagates return codes — fake pw.x only."""
    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    launcher = _load_launcher()
    si = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
               cell=[5.43] * 3, pbc=True)

    # success path: the fake pw.x records its argv and prints the QE
    # fixture; the engine parses it (fixture energy, not a real run)
    monkeypatch.setenv("PW_RECORD", str(tmp_path / "engine-record"))
    bin_dir = _fake_bin(tmp_path / "ok")
    _write_stub(bin_dir / "pw.x",
                'printf "%s\\n" "$@" >> "$PW_RECORD.argv"\n'
                f"cat {QE_FIXTURE.resolve()}\n")
    ok_state = tmp_path / "ok-state"
    ok_setup = _setup_script(tmp_path / "ok-setup", bin_dir)
    assert launcher.main(["prepare", "--setup", str(ok_setup), "--state",
                          str(ok_state), "--mpi", "mpirun", "--pw", "pw.x",
                          "--ranks", "4"]) == 0
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=(sys.executable, str(LAUNCHER), "run", "--state",
                         str(ok_state), "--expected-sha256",
                         _digest(ok_state), "--")),
        run_root=tmp_path / "runs")
    result = engine.compute(si, label="t0")
    assert np.isfinite(result.energy)
    argv_lines = (tmp_path / "engine-record.argv").read_text().splitlines()
    # the engine's own argv contract: QE's input flag appended after `--`
    assert argv_lines == ["-in", str(tmp_path / "runs" / "t0-000000"
                                     / "attempt-1" / "pw.in")]

    # failure path: the child's non-zero exit surfaces as EngineError
    bad_state = _prepare(tmp_path / "bad", monkeypatch)
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=(sys.executable, str(LAUNCHER), "run", "--state",
                         str(bad_state), "--expected-sha256",
                         _digest(bad_state), "--")),
        run_root=tmp_path / "runs2")
    monkeypatch.setenv("PW_EXIT", "3")
    monkeypatch.setenv("PW_RECORD", str(tmp_path / "bad-record"))
    with pytest.raises(EngineError, match="exited with code 3"):
        engine.compute(si, label="boom")


# --- F1/F2/F3: reviewer usage fixes -------------------------------------------


def test_unallowed_setup_change_refused_and_allow_var_works(
        tmp_path, monkeypatch) -> None:
    launcher = _load_launcher()
    bin_dir = _fake_bin(tmp_path)
    setup = _setup_script(tmp_path, bin_dir,
                          extra="export OMPI_MCA_btl=self,tcp\n")
    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    # an unallowed real change fails BEFORE any state is written,
    # reported by NAME with the --allow-var remedy
    state = tmp_path / "state"
    code = launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(state), "--mpi", "mpirun", "--pw", "pw.x",
                          "--ranks", "1"])
    assert code == 2
    assert not state.exists()
    # explicitly allowed: the variable is recorded and reaches the child
    code = launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(state), "--mpi", "mpirun", "--pw", "pw.x",
                          "--ranks", "1", "--allow-var", "OMPI_MCA_btl"])
    assert code == 0
    record = tmp_path / "record"
    result = _run(state, _digest(state), "-in", "pw.in",
                  env_extra={"PW_RECORD": str(record)})
    assert result.returncode == 0, result.stderr
    child_env = Path(str(record) + ".env").read_text()
    assert "OMPI_MCA_btl=self,tcp" in child_env

    # an explicitly allowed REMOVAL is honored too: the parent exports it,
    # the setup unsets it, the child must not see it
    monkeypatch.setenv("CUSTOM_PRUNE", "present")
    prune_setup = _setup_script(tmp_path, bin_dir, extra="unset CUSTOM_PRUNE\n")
    prune_state = tmp_path / "prune-state"
    code = launcher.main(["prepare", "--setup", str(prune_setup),
                          "--state", str(prune_state), "--mpi", "mpirun",
                          "--pw", "pw.x", "--ranks", "1",
                          "--allow-var", "CUSTOM_PRUNE"])
    assert code == 0
    record2 = tmp_path / "record2"
    result = _run(prune_state, _digest(prune_state), "-in", "pw.in",
                  env_extra={"PW_RECORD": str(record2),
                             "CUSTOM_PRUNE": "still-present-at-run"})
    assert result.returncode == 0, result.stderr
    assert "CUSTOM_PRUNE" not in Path(str(record2) + ".env").read_text()

    # credential-class names fail even via --allow-var
    code = launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(tmp_path / "cred"), "--mpi", "mpirun",
                          "--pw", "pw.x", "--ranks", "1",
                          "--allow-var", "MY_SECRET_VAR"])
    assert code == 2
    assert not (tmp_path / "cred").exists()


def test_unallowed_change_reports_names_not_values(tmp_path, monkeypatch,
                                                   capsys) -> None:
    launcher = _load_launcher()
    bin_dir = _fake_bin(tmp_path)
    setup = _setup_script(tmp_path, bin_dir,
                          extra="export OMPI_MCA_btl=self,tcp\n")
    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    code = launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(tmp_path / "state"), "--mpi", "mpirun",
                          "--pw", "pw.x", "--ranks", "1"])
    assert code == 2
    err = capsys.readouterr().err
    assert "OMPI_MCA_btl" in err
    assert "self,tcp" not in err              # names only, never values
    assert "--allow-var" in err


def test_prepared_managed_set_wins_over_changed_run_env(tmp_path,
                                                        monkeypatch) -> None:
    # prepare with OMP_NUM_THREADS=1 present and UNTOUCHED by the setup;
    # run later with 6: the child gets the PREPARED value under the same
    # hash (verified in the actual child environment, not just fields)
    launcher = _load_launcher()
    bin_dir = _fake_bin(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    setup = tmp_path / "setup.sh"
    setup.write_text(f'export PATH="{bin_dir}:$PATH"\n')  # no OMP change
    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    state = tmp_path / "state"
    assert launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(state), "--mpi", "mpirun", "--pw", "pw.x",
                          "--ranks", "1"]) == 0
    digest = _digest(state)
    record = tmp_path / "record"
    result = _run(state, digest, "-in", "pw.in",
                  env_extra={"PW_RECORD": str(record),
                             "OMP_NUM_THREADS": "6",
                             "MKL_VERBOSE": "1"})
    assert result.returncode == 0, result.stderr
    child_env = Path(str(record) + ".env").read_text()
    assert "OMP_NUM_THREADS=1" in child_env
    assert "OMP_NUM_THREADS=6" not in child_env
    # a managed variable NOT present at prepare is unset even when set now
    assert "MKL_VERBOSE" not in child_env
    # unmanaged variables keep inheriting
    assert "PW_RECORD=" in child_env


def test_racing_and_preexisting_state_dirs(tmp_path, monkeypatch) -> None:
    # a pre-existing state dir is refused and its contents stay untouched
    launcher = _load_launcher()
    bin_dir = _fake_bin(tmp_path)
    setup = _setup_script(tmp_path, bin_dir)
    monkeypatch.setenv("SLURM_JOB_ID", JOB_ID)
    preexisting = tmp_path / "taken"
    preexisting.mkdir()
    (preexisting / "user.txt").write_text("keep me\n")
    code = launcher.main(["prepare", "--setup", str(setup), "--state",
                          str(preexisting), "--mpi", "mpirun", "--pw",
                          "pw.x", "--ranks", "1"])
    assert code == 2
    assert (preexisting / "user.txt").read_text() == "keep me\n"

    # two racing prepares: exactly one succeeds, the loser is refused and
    # the winner's state is complete
    raced = tmp_path / "raced"
    env = dict(os.environ)
    argv = [sys.executable, str(LAUNCHER), "prepare", "--setup",
            str(setup), "--state", str(raced), "--mpi", "mpirun", "--pw",
            "pw.x", "--ranks", "1"]
    first = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    second = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    rc = sorted(p.wait() for p in (first, second))
    assert rc == [0, 2], [p.communicate() for p in (first, second)]
    assert (raced / "state.json").is_file()
    assert (raced / "pw_cmd.json").is_file()
    # and the winning state actually launches
    record = tmp_path / "race-record"
    result = _run(raced, _digest(raced), "-in", "pw.in",
                  env_extra={"PW_RECORD": str(record)})
    assert result.returncode == 0, result.stderr


# --- the complete spooled-template fake end-to-end ------------------------------

SBATCH_TEMPLATE = (LAUNCHER.parent / "job_template.sbatch")


def _fake_submit_dir(tmp_path: Path) -> Path:
    """The example as a submit/work dir (with spaces), fake QE stubs, and
    a run.toml.template whose rendered run goes through the REAL pyramid
    parse/run path with the fake pw.x."""
    work = tmp_path / "submit dir"
    work.mkdir()
    import shutil
    shutil.copy(LAUNCHER, work / "launcher.py")
    bin_dir = work / "bin"
    bin_dir.mkdir()
    _write_stub(bin_dir / "mpirun", 'exec "${@:3}"\n')
    _write_stub(bin_dir / "pw.x",
                'printf "argv:" >> "$PW_RECORD"; for a in "$@"; do '
                'printf " <%s>" "$a" >> "$PW_RECORD"; done; '
                'printf "\\n" >> "$PW_RECORD"\n'
                'env | sort > "$PW_RECORD.env"\n'
                f"cat {QE_FIXTURE.resolve()}\n"
                'exit "${PW_EXIT:-0}"\n')
    (work / "setup.sh").write_text(
        f'export PATH="{bin_dir}:$PATH"\n'
        'export OMP_NUM_THREADS=3\n'
        'unset LD_LIBRARY_PATH\n')
    (work / "structure.extxyz").write_text(
        '2\n'
        'Lattice="0.0 2.715 2.715 2.715 0.0 2.715 2.715 2.715 0.0" '
        'Properties=species:S:1:pos:R:3:momenta:R:3 pbc="T T T"\n'
        'Si 0.0 0.0 0.0 -0.28591950 0.42887925 -0.57183900\n'
        'Si 1.3575 1.3575 1.3575 0.28591950 -0.42887925 0.57183900\n')
    (work / "pseudos").mkdir()
    (work / "pseudos" / "Si.UPF").write_text("dummy pseudo for existence checks\n")
    (work / "run.toml.template").write_text(
        'schema_version = 1\n'
        '[run]\n'
        'id = "fake-prepared-demo"\n'
        'directory = "runs/demo"\n'
        '[task]\n'
        'kind = "md"\n'
        'mode = "reference"\n'
        '[structure]\n'
        'file = "structure.extxyz"\n'
        '[dynamics]\n'
        'ensemble = "nve"\n'
        'integrator = "verlet"\n'
        'timestep_fs = 1.0\n'
        'steps = 2\n'
        '[checkpoint]\n'
        'interval_steps = 2\n'
        '[reference]\n'
        'backend = "qe"\n'
        'pseudo_dir = "pseudos"\n'
        'xc = "pbe"\n'
        'pseudos = { Si = "Si.UPF" }\n'
        'pw_cmd = @PREPARED_PW_CMD@\n')
    return work


def _slurm_env(tmp_path: Path, work: Path, record: Path) -> dict:
    venv_bin = str(Path(sys.executable).parent)
    slurm_tmp = tmp_path / "slurmtmp"
    slurm_tmp.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PATH": venv_bin + os.pathsep + env.get("PATH", ""),
        "SLURM_JOB_ID": "fake-alloc-7",
        "SLURM_SUBMIT_DIR": str(work),
        "SLURM_NTASKS": "2",
        "SLURM_TMPDIR": str(slurm_tmp),
        "PW_RECORD": str(record),
    })
    env.pop("PYRAMID_EXAMPLE_DIR", None)
    return env


def test_spooled_template_end_to_end(tmp_path) -> None:
    work = _fake_submit_dir(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    import shutil
    script = spool / "slurm_script"
    shutil.copy(SBATCH_TEMPLATE, script)   # Slurm's spool copy, no launcher
    record = tmp_path / "record"
    template_text = (work / "run.toml.template").read_text()
    result = subprocess.run(["bash", str(script)], cwd=spool,
                            env=_slurm_env(tmp_path, work, record),
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    # no manual hash pasting: the prepared state was actually used, with
    # QE's -in appended through the real engine path
    lines = record.read_text().splitlines()
    assert len(lines) == 3                       # initial + 2 MD steps
    for line in lines:
        assert line.startswith("argv: <-in> <")
        assert str(work) in line
        assert line.endswith("pw.in>")
    child_env = Path(str(record) + ".env").read_text()
    assert "OMP_NUM_THREADS=3" in child_env      # the prepared value won
    assert "LD_LIBRARY_PATH" not in child_env    # the setup's removal held
    # relative paths preserved: structure/pseudos/run.directory resolved
    # next to the template, and the run results are KEPT
    assert (work / "runs" / "demo" / "events.jsonl").is_file()
    assert (work / "runs" / "demo" / "trajectory.db").is_file()
    # the user's template and the spool copy are unmodified
    assert (work / "run.toml.template").read_text() == template_text
    assert "@PREPARED_PW_CMD@" in (work / "run.toml.template").read_text()
    assert script.read_text() == SBATCH_TEMPLATE.read_text()
    # temp artifacts cleaned: no rendered config, no job-private state
    assert not list(work.glob("run.job-*.toml"))
    assert [p.name for p in (tmp_path / "slurmtmp").iterdir()] == []
    # the launcher was found via SLURM_SUBMIT_DIR, not the spool copy
    assert not (spool / "launcher.py").exists()


def test_template_failure_cleans_only_own_artifacts(tmp_path) -> None:
    work = _fake_submit_dir(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    import shutil
    script = spool / "slurm_script"
    shutil.copy(SBATCH_TEMPLATE, script)
    record = tmp_path / "record"
    env = _slurm_env(tmp_path, work, record)
    env["PW_EXIT"] = "3"                        # every QE call fails
    result = subprocess.run(["bash", str(script)], cwd=spool, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    # own temp config and state are gone; the failed run's data and every
    # pre-existing file stay untouched
    assert not list(work.glob("run.job-*.toml"))
    assert [p.name for p in (tmp_path / "slurmtmp").iterdir()] == []
    assert (work / "runs" / "demo").is_dir()     # the failure record stays
    assert (work / "pseudos" / "Si.UPF").is_file()
    assert (work / "run.toml.template").is_file()


# --- F3a/F3b: config ownership, no-clobber publish, actual-template base -------


def test_prepare_failure_preserves_every_preexisting_file(tmp_path) -> None:
    # a failing setup must delete NOTHING — above all not a user-owned
    # config that happens to carry the old-style name
    work = _fake_submit_dir(tmp_path)
    decoy = work / "run.job-fake-alloc-7.toml"
    decoy_content = "user-owned configuration; preserve exactly\n"
    decoy.write_text(decoy_content)
    (work / "setup.sh").write_text("return 7\n")
    spool = tmp_path / "spool"
    spool.mkdir()
    import shutil
    script = spool / "slurm_script"
    shutil.copy(SBATCH_TEMPLATE, script)
    result = subprocess.run(["bash", str(script)], cwd=spool,
                            env=_slurm_env(tmp_path, work,
                                           tmp_path / "record"),
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert decoy.read_text() == decoy_content     # byte-identical
    assert not (tmp_path / "record").exists()     # fakeQE never started
    # nothing was created at all: exactly the original files remain
    assert sorted(p.name for p in work.iterdir()) == sorted(
        ["launcher.py", "bin", "setup.sh", "structure.extxyz", "pseudos",
         "run.toml.template", "run.job-fake-alloc-7.toml"])


def _render(state: Path, template: Path, output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(LAUNCHER), "render-config", "--template",
         str(template), "--state", str(state), "--output", str(output)],
        capture_output=True, text=True, check=False)


def test_render_config_no_clobber_file_symlink_and_race(tmp_path,
                                                        monkeypatch) -> None:
    state = _prepare(tmp_path / "s1", monkeypatch)
    template = tmp_path / "s1" / "run.toml.template"
    template.write_text('x = @PREPARED_PW_CMD@\n')
    output = tmp_path / "s1" / "out.toml"

    # an existing regular file is refused and left byte-identical
    output.write_text("user content\n")
    result = _render(state, template, output)
    assert result.returncode == 2
    assert output.read_text() == "user content\n"

    # an existing symlink is refused too — never dereferenced, target safe
    link = tmp_path / "s1" / "link.toml"
    link_target = tmp_path / "s1" / "link-target.toml"
    link_target.write_text("link target content\n")
    link.symlink_to(link_target)
    result = _render(state, template, link)
    assert result.returncode == 2
    assert link.is_symlink() and link.readlink() == link_target
    assert link_target.read_text() == "link target content\n"

    # two concurrent renders to one fresh target: exactly one creator,
    # the loser deletes nothing of the winner's
    state2 = _prepare(tmp_path / "s2", monkeypatch)
    raced = tmp_path / "s1" / "raced.toml"
    env = dict(os.environ)
    argv = [sys.executable, str(LAUNCHER), "render-config", "--template",
            str(template), "--output", str(raced)]
    first = subprocess.Popen(argv + ["--state", str(state)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    second = subprocess.Popen(argv + ["--state", str(state2)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
    rc = sorted(p.wait() for p in (first, second))
    assert rc == [0, 2], [p.communicate() for p in (first, second)]
    winner_hash = (_digest(state) if first.wait() == 0 else _digest(state2))
    assert winner_hash in raced.read_text()
    # a successful render parses as TOML and binds the prepared argv
    import tomllib
    parsed = tomllib.loads(raced.read_text())
    assert parsed["x"][-1] == "--"
    assert template.read_text() == 'x = @PREPARED_PW_CMD@\n'  # template intact


def test_invalid_toml_never_publishes(tmp_path, monkeypatch) -> None:
    state = _prepare(tmp_path, monkeypatch)
    broken = tmp_path / "state" / "broken.toml.template"
    broken.write_text('[unclosed\npw_cmd = @PREPARED_PW_CMD@\n')
    output = tmp_path / "state" / "broken.out.toml"
    result = _render(state, broken, output)
    assert result.returncode == 2
    assert "not valid TOML" in result.stderr
    assert not output.exists()
    # a template with no (or two) placeholders is refused as well
    plain = tmp_path / "state" / "plain.toml.template"
    plain.write_text("[run]\nid = 'x'\n")
    result = _render(state, plain, output)
    assert result.returncode == 2
    assert not output.exists()


def _external_project(tmp_path: Path, work: Path) -> Path:
    """Move template+structure+pseudos into a separate space-carrying
    project dir and point RUN_CONFIG_TEMPLATE at it (the reviewer's
    scenario)."""
    import shutil
    project = tmp_path / "separate scientific project"
    project.mkdir()
    for name in ("run.toml.template", "structure.extxyz", "pseudos"):
        shutil.move(str(work / name), str(project / name))
    return project


def test_alternate_template_dir_end_to_end(tmp_path) -> None:
    work = _fake_submit_dir(tmp_path)
    project = _external_project(tmp_path, work)
    spool = tmp_path / "spool"
    spool.mkdir()
    import shutil
    script = spool / "slurm_script"
    shutil.copy(SBATCH_TEMPLATE, script)
    record = tmp_path / "record"
    env = _slurm_env(tmp_path, work, record)
    env["RUN_CONFIG_TEMPLATE"] = str(project / "run.toml.template")
    template_text = (project / "run.toml.template").read_text()
    result = subprocess.run(["bash", str(script)], cwd=spool, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    # the prepared state drove the real engine path with -in args inside
    # the ACTUAL project (relative semantics of the template's own dir)
    lines = record.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert line.startswith("argv: <-in> <")
        assert str(project) in line
        assert line.endswith("pw.in>")
    # results land in the actual template's project, not the example dir
    assert (project / "runs" / "demo" / "events.jsonl").is_file()
    assert not (work / "runs").exists()
    # the actual template and the user's files are unchanged
    assert (project / "run.toml.template").read_text() == template_text
    assert not list(project.glob("run.job-*.toml"))
    assert [p.name for p in (tmp_path / "slurmtmp").iterdir()] == []


def test_alternate_template_failure_keeps_records(tmp_path) -> None:
    work = _fake_submit_dir(tmp_path)
    project = _external_project(tmp_path, work)
    decoy = project / "run.job-someone-elses.toml"
    decoy.write_text("pre-existing user config\n")
    spool = tmp_path / "spool"
    spool.mkdir()
    import shutil
    script = spool / "slurm_script"
    shutil.copy(SBATCH_TEMPLATE, script)
    env = _slurm_env(tmp_path, work, tmp_path / "record")
    env["RUN_CONFIG_TEMPLATE"] = str(project / "run.toml.template")
    env["PW_EXIT"] = "3"
    result = subprocess.run(["bash", str(script)], cwd=spool, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    # this run's temp config and state are cleaned; the failure record
    # and every pre-existing user file survive byte-identically
    assert not list(project.glob("run.job-fake-alloc-7-*.toml"))
    assert [p.name for p in (tmp_path / "slurmtmp").iterdir()] == []
    assert (project / "runs" / "demo").is_dir()
    assert decoy.read_text() == "pre-existing user config\n"
    assert (project / "pseudos" / "Si.UPF").is_file()
    assert "@PREPARED_PW_CMD@" in (project / "run.toml.template").read_text()


def test_relative_template_anchors_at_submit_dir_across_depths(
        tmp_path) -> None:
    """The reviewer scenario: submit dir `root/submit dir`, template value
    `../separate scientific project/run.toml.template`, run cwd a spool
    tree at a DIFFERENT depth — the relative template resolves against
    SLURM_SUBMIT_DIR, never the process cwd."""
    root = tmp_path / "root"
    root.mkdir()
    work = _fake_submit_dir(root)               # -> root/"submit dir"
    project = _external_project(root, work)
    spool = root / "spool" / "job123"          # different, deeper tree
    spool.mkdir(parents=True)
    import shutil
    script = spool / "slurm_script"
    shutil.copy(SBATCH_TEMPLATE, script)
    record = root / "record"
    env = _slurm_env(root, work, record)
    env["SLURM_SUBMIT_DIR"] = str(work)
    env["RUN_CONFIG_TEMPLATE"] = ("../separate scientific project"
                                  "/run.toml.template")
    template_text = (project / "run.toml.template").read_text()
    result = subprocess.run(["bash", str(script)], cwd=spool, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    lines = record.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert line.startswith("argv: <-in> <")
        assert str(project) in line
        assert line.endswith("pw.in>")
    # results land under the template's own project; temp state and this
    # run's config are cleaned; every user file is untouched
    assert (project / "runs" / "demo" / "events.jsonl").is_file()
    assert not list(project.glob("run.job-*.toml"))
    assert [p.name for p in (root / "slurmtmp").iterdir()] == []
    assert (project / "run.toml.template").read_text() == template_text
    assert (project / "pseudos" / "Si.UPF").is_file()

    # a relative template with NO submit directory errors clearly instead
    # of silently depending on cwd
    env = {key: value for key, value in env.items()
           if key != "SLURM_SUBMIT_DIR"}
    env["PYRAMID_EXAMPLE_DIR"] = str(work)     # resources still found
    result = subprocess.run(["bash", str(script)], cwd=spool, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "RUN_CONFIG_TEMPLATE" in result.stderr
    assert "SLURM_SUBMIT_DIR" in result.stderr
