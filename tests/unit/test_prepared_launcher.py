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
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
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
    assert "never overwritten" in capsys.readouterr().err

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
