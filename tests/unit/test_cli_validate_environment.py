"""``pyramid validate --check-environment`` / ``--json`` acceptance tests.

Scope: the static, read-only environment preflight and the structured
report.  No backend is started, no command is executed, nothing is
downloaded; a fake executable doubles as an execution sentinel (it
writes a marker file the moment anything runs it — the marker must
never appear).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from pyraimd2.cli import main as cli_main
from pyraimd2.config import load_config
from pyraimd2.workflows import preflight
from pyraimd2.workflows.setup import build_backends
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE


def write_config(tmp_path: Path, text: str) -> Path:
    config = tmp_path / "run dir with space" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    (config.parent / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    config.write_text(text)
    return config


def _plain_mode_config(mode: str, backend_block: str) -> str:
    text = HARMONIC_CONFIG.replace('mode = "adaptive"', f'mode = "{mode}"')
    keep, drop = ("[reference]", "[surrogate]") if mode == "reference" \
        else ("[surrogate]", "[reference]")
    start = text.index(keep)
    following = text.index("\n[", start + 1)
    text = text[:start] + backend_block + "\n\n" + text[following + 1:]
    for section in (drop, "[policy]", "[verification]"):
        start = text.index(section)
        following = text.index("\n[", start + 1)
        text = text[:start] + text[following + 1:]
    return text


def qe_config(tmp_path: Path, *, pw_cmd: str, with_pseudos: bool = True,
              extra: str = "") -> Path:
    block = (f'[reference]\nbackend = "qe"\npseudo_dir = "pseudos"\n'
             f'ecutwfc = 30.0\npseudos = {{ H = "H.upf", O = "O.upf" }}\n'
             f'pw_cmd = "{pw_cmd}"' + extra)
    text = _plain_mode_config("reference", block)
    if with_pseudos:
        pseudo_dir = tmp_path / "run dir with space" / "pseudos"
        pseudo_dir.mkdir(parents=True, exist_ok=True)
        for species in ("H", "O"):
            (pseudo_dir / f"{species}.upf").write_text(
                f'<UPF version="2.0.1"><PP_HEADER element="{species}"/></UPF>\n')
    return write_config(tmp_path, text)


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root)).encode()
        if path.is_symlink():
            h.update(b"L" + rel + str(path.readlink()).encode())
        elif path.is_dir():
            h.update(b"D" + rel)
        else:
            h.update(b"F" + rel + path.read_bytes())
    return h.hexdigest()


# --- acceptance 1: default validate never checks the environment ----------


def test_default_validate_marks_environment_not_checked(tmp_path, capsys):
    config = qe_config(tmp_path, pw_cmd="definitely-no-such-pw-executable")
    code = cli_main(["validate", str(config)])
    out = capsys.readouterr()
    assert code == 0
    assert "configuration valid" in out.out
    assert "environment NOT checked" in out.out


# --- acceptance 2: missing pw.x blocked; fake executable pass, never run ---


def test_environment_missing_pw_cmd_blocked_with_json(tmp_path, capsys):
    config = qe_config(tmp_path, pw_cmd="definitely-no-such-pw-executable")
    code = cli_main(["validate", str(config), "--check-environment", "--json"])
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert code == 1
    assert report["validation_scope"] == "environment"
    assert report["configuration_valid"] is True
    assert report["readiness"] == "blocked"
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "fail"
    assert "remedy" in pw and "Quantum ESPRESSO" in pw["remedy"]


def test_environment_fake_executable_passes_and_is_never_executed(
        tmp_path, capsys):
    marker = tmp_path / "executed.marker"
    fake = tmp_path / "fake pw.x"
    fake.write_text(f"#!/bin/sh\ntouch {marker}\n")
    fake.chmod(0o755)
    config = qe_config(tmp_path, pw_cmd=str(fake))
    before = _tree_digest(tmp_path)
    code = cli_main(["validate", str(config), "--check-environment", "--json"])
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert code == 0, out.err + out.out
    assert report["readiness"] == "ready"
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "pass"
    assert "never executed" in pw["message"]
    assert not marker.exists()  # the executable was resolved, not run
    assert _tree_digest(tmp_path) == before  # strictly read-only


def test_environment_pw_cmd_on_path(tmp_path, capsys, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "pw.x"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    config = qe_config(tmp_path, pw_cmd="pw.x")
    code = cli_main(["validate", str(config), "--check-environment", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "pass"


# --- acceptance 3: wrappers / model names / optional packages --------------


def test_environment_wrapper_command_is_unverified(tmp_path, capsys,
                                                   monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "srun"
    launcher.write_text("#!/bin/sh\nexit 99\n")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH",
                       f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    # launcher present on PATH: the solver layer behind it stays
    # unconfirmed statically (an ABSENT launcher would be blocked)
    config = qe_config(tmp_path, pw_cmd="srun -n 4 pw.x")
    code = cli_main(["validate", str(config), "--check-environment", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["readiness"] == "unverified"
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "unverified"
    assert "srun" in pw["message"] and "solver layer" in pw["message"]


def test_environment_mace_name_unverified_and_missing_package_blocked(
        tmp_path, capsys, monkeypatch):
    block = ('[surrogate]\nbackend = "mace"\nmodel = "small"\n'
             'device = "cpu"\ndefault_dtype = "float64"')
    text = _plain_mode_config("surrogate", block)
    config = write_config(tmp_path, text)

    real_find_spec = preflight._find_spec
    monkeypatch.setattr(preflight, "_find_spec",
                        lambda name: None if name == "mace"
                        else real_find_spec(name))
    code = cli_main(["validate", str(config), "--check-environment", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["readiness"] == "blocked"
    pkg = next(c for c in report["checks"]
               if c["id"] == "surrogate.mace")
    assert pkg["status"] == "fail"
    model = next(c for c in report["checks"] if c["id"] == "surrogate.model")
    assert model["status"] == "unverified"
    assert "cache" in model["message"]


def test_environment_harmonic_offline_example_is_ready(tmp_path, capsys):
    config = write_config(tmp_path, HARMONIC_CONFIG)
    code = cli_main(["validate", str(config), "--check-environment", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["readiness"] == "ready"
    assert all(c["status"] == "pass" for c in report["checks"])


# --- acceptance 6: JSON across scopes; exit codes ---------------------------


def test_json_configuration_scope_success(tmp_path, capsys):
    config = write_config(tmp_path, HARMONIC_CONFIG)
    code = cli_main(["validate", str(config), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["validation_scope"] == "configuration"
    assert report["configuration_valid"] is True
    assert report["readiness"] == "not_checked"
    assert report["structure"]["formula"] == "H2O"


def test_json_configuration_error_is_machine_readable(tmp_path, capsys):
    config = write_config(
        tmp_path, HARMONIC_CONFIG.replace("timestep_fs", "timestep"))
    code = cli_main(["validate", str(config), "--json"])
    out = capsys.readouterr()
    assert code == 2
    report = json.loads(out.out)
    assert report["configuration_valid"] is False
    assert "error" in report and "message" in report["error"]


def test_json_probe_scope_runs_existing_probe(tmp_path, capsys):
    config = write_config(tmp_path, HARMONIC_CONFIG)
    code = cli_main(["validate", str(config), "--probe-backends", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["validation_scope"] == "probe"
    assert set(report["probes"]) == {"reference", "surrogate"}


def test_check_environment_and_probe_are_mutually_exclusive(tmp_path, capsys):
    config = write_config(tmp_path, HARMONIC_CONFIG)
    code = cli_main(["validate", str(config), "--check-environment",
                     "--probe-backends"])
    assert code == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_text_environment_mode_reports_checks_and_remedy(tmp_path, capsys):
    config = qe_config(tmp_path, pw_cmd="no-such-pw-anywhere")
    code = cli_main(["validate", str(config), "--check-environment"])
    out = capsys.readouterr()
    assert code == 1
    assert "environment   : blocked" in out.out
    assert "[fail] reference.pw_cmd" in out.out
    assert "remedy:" in out.out


# ---------------------------------------------------------------------------
# N058-A regressions (reviews/N055-r2 F1-F6)


def _fake_pw(tmp_path: Path, name: str = "fake pw.x") -> Path:
    fake = tmp_path / name
    fake.write_text("#!/bin/sh\nexit 99\n")
    fake.chmod(0o755)
    return fake


def _validate_env_json(config: Path, capsys):
    code = cli_main(["validate", str(config), "--check-environment", "--json"])
    out = capsys.readouterr()
    return code, json.loads(out.out)


def _wrapper_block(wrapper: str, base_spec: str, extra: str = "") -> str:
    return (f'[surrogate]\nbackend = "{wrapper}"\n'
            f'base = {{ name = {base_spec} }}\n' + extra)


# F1: wrapper backends recurse into the actual base spec --------------------


def test_wrapper_scaled_missing_mace_base_blocked(tmp_path, capsys,
                                                  monkeypatch):
    block = ('[surrogate]\nbackend = "scaled"\nscale = 1.1\n'
             'base = { name = "mace", kwargs = { model = "small" } }')
    config = write_config(tmp_path, _plain_mode_config("surrogate", block))
    real_find_spec = preflight._find_spec
    monkeypatch.setattr(preflight, "_find_spec",
                        lambda name: None if name == "mace"
                        else real_find_spec(name))
    code, report = _validate_env_json(config, capsys)
    assert code == 1
    assert report["readiness"] == "blocked"
    own = next(c for c in report["checks"] if c["id"] == "surrogate.scaled")
    assert own["status"] == "pass"
    assert "base backend is checked separately" in own["message"]
    nested = next(c for c in report["checks"]
                  if c["id"] == "surrogate.scaled.base.mace")
    assert nested["status"] == "fail"
    assert "no optional dependencies" not in json.dumps(report["checks"])
    assert "mace" not in sys.modules  # never imported to answer the check


def test_wrapper_quadratic_corrected_missing_mace_base_blocked(
        tmp_path, capsys, monkeypatch):
    zeros3x3 = ", ".join(["[0.0, 0.0, 0.0]"] * 3)
    eye9_rows = ", ".join(
        "[" + ", ".join("1.0" if i == j else "0.0" for j in range(9)) + "]"
        for i in range(9))
    block = ('[surrogate]\nbackend = "quadratic-corrected"\n'
             'species = ["O", "H", "H"]\n'
             f'q0 = [{zeros3x3}]\n'
             f'delta_f0 = [{zeros3x3}]\n'
             f'delta_h = [{eye9_rows}]\n'
             'base = { name = "mace", kwargs = { model = "small" } }')
    config = write_config(tmp_path, _plain_mode_config("surrogate", block))
    real_find_spec = preflight._find_spec
    monkeypatch.setattr(preflight, "_find_spec",
                        lambda name: None if name == "mace"
                        else real_find_spec(name))
    code, report = _validate_env_json(config, capsys)
    assert code == 1
    assert report["readiness"] == "blocked"
    nested = next(c for c in report["checks"]
                  if c["id"] == "surrogate.quadratic-corrected.base.mace")
    assert nested["status"] == "fail"


def test_wrapper_unknown_plugin_base_unverified(tmp_path, capsys):
    block = ('[surrogate]\nbackend = "scaled"\nscale = 1.1\n'
             'base = { name = "no-such-plugin-backend" }')
    config = write_config(tmp_path, _plain_mode_config("surrogate", block))
    # the unknown base cannot even be constructed -> configuration invalid
    code = cli_main(["validate", str(config), "--check-environment",
                     "--json"])
    out = capsys.readouterr()
    report = json.loads(out.out)
    if report["configuration_valid"] is True:
        # if the plugin resolves lazily in some environment, the check
        # must still never upgrade an unknown backend to ready
        assert report["readiness"] == "unverified"
        assert code == 1
    else:
        assert code == 2


def test_wrapper_unknown_plugin_base_unverified_preflight_direct(tmp_path):
    checks = preflight._backend_checks(
        "surrogate", "scaled",
        {"scale": 1.1, "base": {"name": "no-such-plugin-backend"}}, ())
    nested = checks[-1]
    assert nested["id"] == "surrogate.scaled.base.no-such-plugin-backend"
    assert nested["status"] == "unverified"


def test_wrapper_scaled_harmonic_base_ready(tmp_path, capsys):
    block = ('[surrogate]\nbackend = "scaled"\nscale = 1.1\n'
             'base = { name = "harmonic-surrogate", '
             'kwargs = { k = 1.0, r0 = 0.9, bias = 0.05 } }')
    config = write_config(tmp_path, _plain_mode_config("surrogate", block))
    code, report = _validate_env_json(config, capsys)
    assert code == 0
    assert report["readiness"] == "ready"
    ids = [c["id"] for c in report["checks"]]
    assert "surrogate.scaled.base.harmonic-surrogate" in ids


# F2: one shared pw_cmd argv contract ---------------------------------------


def test_string_pw_cmd_quoted_path_matches_execution_tuple(
        tmp_path, capsys):
    fake = _fake_pw(tmp_path)
    config = qe_config(tmp_path, pw_cmd=f"'{fake}'")
    code, report = _validate_env_json(config, capsys)
    assert code == 0, json.dumps(report)
    assert report["readiness"] == "ready"
    engine, _ = build_backends(load_config(config))
    assert engine.config.pw_cmd == (str(fake),)  # same argv as execution


def test_string_pw_cmd_existing_path_is_literal(tmp_path, capsys):
    fake = _fake_pw(tmp_path)
    config = qe_config(tmp_path, pw_cmd=str(fake))  # unquoted, has spaces
    code, report = _validate_env_json(config, capsys)
    assert code == 0
    assert report["readiness"] == "ready"
    engine, _ = build_backends(load_config(config))
    assert engine.config.pw_cmd == (str(fake),)
    assert not next(tmp_path.glob("executed*"), None), "solver executed"


def test_pw_cmd_invalid_forms_refused_at_configuration(tmp_path, capsys):
    qe_config(tmp_path, pw_cmd="42")  # a TOML string "42" is fine;
    # but a TOML integer is not a command:
    bad = tmp_path / "run dir with space" / "config.toml"
    bad.write_text(bad.read_text().replace('pw_cmd = "42"', "pw_cmd = 42"))
    code = cli_main(["validate", str(bad), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["configuration_valid"] is False
    assert "pw_cmd" in report["error"]["message"]
    assert "list of strings" in report["error"]["message"]

    bad.write_text(bad.read_text().replace("pw_cmd = 42",
                                           "pw_cmd = \"'unterminated\""))
    code = cli_main(["validate", str(bad), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "quote" in report["error"]["message"]


# F3: qe-ase checks the effective command source ----------------------------


def _write_qe_ase(tmp_path, block):
    text = _plain_mode_config("reference", block)
    path = write_config(tmp_path, text)
    pseudo_dir = tmp_path / "run dir with space" / "pseudos"
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    for species in ("H", "O"):
        (pseudo_dir / f"{species}.upf").write_text(
            f'<UPF version="2.0.1"><PP_HEADER element="{species}"/></UPF>\n')
    return path


def test_qe_ase_missing_command_blocked_despite_valid_pw_cmd(
        tmp_path, capsys):
    fake = _fake_pw(tmp_path)
    config = _write_qe_ase(
        tmp_path,
        '[reference]\nbackend = "qe-ase"\npseudo_dir = "pseudos"\n'
        'pseudos = { H = "H.upf", O = "O.upf" }\n'
        f'pw_cmd = ["{fake}"]\n'
        'command = "no-such-overridden-solver"\n')
    code, report = _validate_env_json(config, capsys)
    assert code == 1
    assert report["readiness"] == "blocked"
    check = next(c for c in report["checks"] if c["id"] ==
                 "reference.command")
    assert check["status"] == "fail"
    assert "overrides pw_cmd" in check["message"]
    # the adapter really prefers command over pw_cmd
    engine, _ = build_backends(load_config(config))
    assert engine._command == "no-such-overridden-solver"


def test_qe_ase_valid_command_overrides_missing_pw_cmd(tmp_path, capsys):
    fake = _fake_pw(tmp_path)
    config = _write_qe_ase(
        tmp_path,
        '[reference]\nbackend = "qe-ase"\npseudo_dir = "pseudos"\n'
        'pseudos = { H = "H.upf", O = "O.upf" }\n'
        'pw_cmd = ["no-such-unused-pw"]\n'
        f'command = "\'{fake}\'"\n')
    code, report = _validate_env_json(config, capsys)
    assert code == 0, json.dumps(report)
    assert report["readiness"] == "ready"
    check = next(c for c in report["checks"] if c["id"] ==
                 "reference.command")
    assert check["status"] == "pass"


def test_qe_ase_without_command_uses_pw_cmd(tmp_path, capsys):
    fake = _fake_pw(tmp_path)
    config = _write_qe_ase(
        tmp_path,
        '[reference]\nbackend = "qe-ase"\npseudo_dir = "pseudos"\n'
        'pseudos = { H = "H.upf", O = "O.upf" }\n'
        f'pw_cmd = ["{fake}"]\n')
    code, report = _validate_env_json(config, capsys)
    assert code == 0
    assert report["readiness"] == "ready"
    check = next(c for c in report["checks"] if c["id"] ==
                 "reference.pw_cmd")
    assert check["status"] == "pass"


# F4: direct argv, literal paths, launchers ---------------------------------


def test_direct_argv_with_arguments_is_ready(tmp_path, capsys):
    fake = _fake_pw(tmp_path)
    config = _write_qe_ase(
        tmp_path,
        '[reference]\nbackend = "qe"\npseudo_dir = "pseudos"\n'
        'pseudos = { H = "H.upf", O = "O.upf" }\n'
        f'pw_cmd = ["{fake}", "-nk", "2"]\n')
    code, report = _validate_env_json(config, capsys)
    assert code == 0, json.dumps(report)
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "pass"
    assert "verbatim" in pw["message"]


def test_literal_metachar_path_is_not_a_wrapper(tmp_path, capsys):
    special_dir = tmp_path / "QE(7.5)"
    special_dir.mkdir()
    fake = _fake_pw(special_dir, name="pw.x")
    config = qe_config(tmp_path, pw_cmd=str(fake))
    code, report = _validate_env_json(config, capsys)
    assert code == 0, json.dumps(report)
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "pass"


def test_launcher_missing_is_blocked(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
    config = qe_config(tmp_path, pw_cmd="unused")
    cfg = tmp_path / "run dir with space" / "config.toml"
    cfg.write_text(cfg.read_text().replace(
        'pw_cmd = "unused"',
        'pw_cmd = ' + json.dumps(["mpirun", "-np", "2", "pw.x"])))
    code, report = _validate_env_json(config, capsys)
    assert code == 1
    assert report["readiness"] == "blocked"
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "fail"
    assert "launcher" in pw["message"]


def test_launcher_present_is_unverified(tmp_path, capsys, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "mpirun"
    launcher.write_text("#!/bin/sh\nexit 99\n")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH",
                       f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    config = qe_config(tmp_path, pw_cmd="unused")
    cfg = tmp_path / "run dir with space" / "config.toml"
    cfg.write_text(cfg.read_text().replace(
        'pw_cmd = "unused"', 'pw_cmd = ["mpirun", "-np", "2", "pw.x"]'))
    code, report = _validate_env_json(config, capsys)
    assert code == 1
    assert report["readiness"] == "unverified"
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "unverified"
    assert "solver layer" in pw["message"]


def test_shell_operator_form_is_unverified_not_executed(tmp_path, capsys):
    config = qe_config(tmp_path, pw_cmd="unused")
    cfg = tmp_path / "run dir with space" / "config.toml"
    cfg.write_text(cfg.read_text().replace('pw_cmd = "unused"',
                                           'pw_cmd = "pw.x > out.log"'))
    code, report = _validate_env_json(config, capsys)
    assert code == 1
    pw = next(c for c in report["checks"] if c["id"] == "reference.pw_cmd")
    assert pw["status"] == "unverified"
    assert "shell" in pw["message"]
    assert not (tmp_path / "run dir with space" / "out.log").exists()


# F5: validate never creates or modifies the run store ----------------------


def _run_dir_config(tmp_path: Path) -> Path:
    text = HARMONIC_CONFIG.replace('directory = "runs/harmonic-demo"',
                                   'directory = "run"')
    return write_config(tmp_path, text)


def test_validate_existing_empty_db_refused_without_writes(
        tmp_path, capsys):
    config = _run_dir_config(tmp_path)
    run_dir = tmp_path / "run dir with space" / "run"
    run_dir.mkdir(parents=True)
    db = run_dir / "trajectory.db"
    db.write_bytes(b"")
    mtime = db.stat().st_mtime_ns
    digest = _tree_digest(tmp_path)
    code = cli_main(["validate", str(config), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["configuration_valid"] is False
    assert "not readable as a trajectory database" in \
        report["error"]["message"]
    assert db.read_bytes() == b""
    assert db.stat().st_mtime_ns == mtime
    assert not list(run_dir.glob("trajectory.db-*"))  # no WAL/SHM/journal
    assert _tree_digest(tmp_path) == digest


def test_validate_existing_valid_db_refused_readonly(tmp_path, capsys):
    config = _run_dir_config(tmp_path)
    run_dir = tmp_path / "run dir with space" / "run"
    run_dir.mkdir(parents=True)
    db = run_dir / "trajectory.db"
    import ase.db
    with ase.db.connect(db) as con:
        con.write(None, run_id="somebody-elses-run")
    before = db.read_bytes()
    mtime = db.stat().st_mtime_ns
    digest = _tree_digest(tmp_path)
    code = cli_main(["validate", str(config), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "somebody-elses-run" in report["error"]["message"]
    assert db.read_bytes() == before
    assert db.stat().st_mtime_ns == mtime
    assert _tree_digest(tmp_path) == digest


def test_validate_existing_corrupt_db_refused_readonly(tmp_path, capsys):
    config = _run_dir_config(tmp_path)
    run_dir = tmp_path / "run dir with space" / "run"
    run_dir.mkdir(parents=True)
    db = run_dir / "trajectory.db"
    db.write_bytes(b"\x00\x01\x02\x03 not a sqlite database" * 64)
    before = db.read_bytes()
    digest = _tree_digest(tmp_path)
    code = cli_main(["validate", str(config), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "not readable as a trajectory database" in \
        report["error"]["message"]
    assert db.read_bytes() == before
    assert _tree_digest(tmp_path) == digest


# F6: the validate mode conflict is one JSON error object -------------------


def test_validate_mode_conflict_json_is_machine_readable(tmp_path, capsys):
    config = write_config(tmp_path, HARMONIC_CONFIG)
    code = cli_main(["validate", str(config), "--check-environment",
                     "--probe-backends", "--json"])
    out = capsys.readouterr()
    assert code == 2
    report = json.loads(out.out)  # stdout is one parseable object
    assert report["configuration_valid"] is False
    assert report["readiness"] == "not_checked"
    assert report["error"]["code"] == "UsageError"
    assert "mutually exclusive" in report["error"]["message"]
