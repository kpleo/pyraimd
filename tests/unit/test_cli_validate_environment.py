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
from pathlib import Path

import pytest

from pyraimd2.cli import main as cli_main
from pyraimd2.workflows import preflight
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


def test_environment_wrapper_command_is_unverified(tmp_path, capsys):
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
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["validate", str(config), "--check-environment",
                  "--probe-backends"])
    assert excinfo.value.code == 2


def test_text_environment_mode_reports_checks_and_remedy(tmp_path, capsys):
    config = qe_config(tmp_path, pw_cmd="no-such-pw-anywhere")
    code = cli_main(["validate", str(config), "--check-environment"])
    out = capsys.readouterr()
    assert code == 1
    assert "environment   : blocked" in out.out
    assert "[fail] reference.pw_cmd" in out.out
    assert "remedy:" in out.out
