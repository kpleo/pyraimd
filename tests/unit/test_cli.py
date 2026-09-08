"""CLI tests (WP04): every command, offline, with the acceptance flow
init -> validate -> run -> inspect -> export -> resume running end to end
on builtin backends in a temporary directory whose name contains a space —
no Python source edited, exactly the user experience.

``cli.main`` returns exit codes; argparse handles --help/--version by
printing and exiting 0 (caught here as SystemExit).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pyraimd2.cli import main as cli_main
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE


def run_cli(*argv: str) -> int:
    return cli_main(list(argv))


def help_text(capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as excinfo:
        cli_main(list(argv))
    assert excinfo.value.code == 0
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# offline basics


def test_version(capsys) -> None:
    output = help_text(capsys, "--version")
    assert "0.3.0" in output


def test_help_offline_for_all_commands(capsys) -> None:
    assert "init" in help_text(capsys, "--help")
    for command in ("init", "validate", "run", "resume", "inspect", "export",
                    "backends"):
        assert "usage:" in help_text(capsys, command, "--help")


def test_help_and_imports_stay_light(capsys) -> None:
    """--help must not drag in optional heavy dependencies."""
    for module in ("torch", "pyscf"):
        sys.modules.pop(module, None)
    help_text(capsys, "--help")
    import pyraimd2.cli
    import pyraimd2.workflows  # noqa: F401
    assert "torch" not in sys.modules
    assert "pyscf" not in sys.modules


def test_backends_lists_builtins(capsys) -> None:
    output = run_cli_and_out(capsys, "backends")
    for name in ("harmonic-reference", "harmonic-surrogate", "qe", "qe-ase",
                 "pyscf", "mace"):
        assert name in output
    assert "torch" not in sys.modules


def run_cli_and_out(capsys, *argv: str) -> str:
    code = run_cli(*argv)
    assert code == 0
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# init / validate


def test_init_writes_template_and_refuses_overwrite(tmp_path, capsys) -> None:
    output = run_cli_and_out(capsys, "init", "--template", "harmonic",
                             "--output", str(tmp_path / "proj"))
    assert "run.toml" in output
    assert (tmp_path / "proj" / "run.toml").is_file()
    assert (tmp_path / "proj" / "structure.extxyz").is_file()
    assert run_cli("init", "--output", str(tmp_path / "proj")) == 2
    assert "--force" in capsys.readouterr().err
    assert run_cli("init", "--output", str(tmp_path / "proj"), "--force") == 0


def test_init_rejects_unknown_template(tmp_path, capsys) -> None:
    assert run_cli("init", "--template", "diamond", "--output",
                   str(tmp_path / "x")) == 2
    assert "unknown template" in capsys.readouterr().err


def write_config(tmp_path: Path, text: str) -> Path:
    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def test_validate_ok(tmp_path, capsys) -> None:
    config = write_config(tmp_path, HARMONIC_CONFIG)
    output = run_cli_and_out(capsys, "validate", str(config))
    assert "validate: OK" in output
    assert "harmonic-reference" in output


def test_validate_probe_backends(tmp_path, capsys) -> None:
    config = write_config(tmp_path, HARMONIC_CONFIG)
    output = run_cli_and_out(capsys, "validate", str(config), "--probe-backends")
    assert "probe reference" in output
    assert "probe surrogate" in output


def test_validate_names_unknown_field(tmp_path, capsys) -> None:
    config = write_config(tmp_path, HARMONIC_CONFIG.replace("timestep_fs", "timestep"))
    assert run_cli("validate", str(config)) == 2
    error = capsys.readouterr().err
    assert "dynamics.timestep" in error and "timestep_fs" in error


def test_validate_names_negative_timestep(tmp_path, capsys) -> None:
    config = write_config(tmp_path, HARMONIC_CONFIG.replace("timestep_fs = 0.5",
                                                            "timestep_fs = -0.5"))
    assert run_cli("validate", str(config)) == 2
    assert "dynamics.timestep_fs" in capsys.readouterr().err


def test_validate_names_zero_check_combination(tmp_path, capsys) -> None:
    config = write_config(tmp_path, HARMONIC_CONFIG.replace("probability = 0.1",
                                                            "probability = 0.0"))
    assert run_cli("validate", str(config)) == 2
    error = capsys.readouterr().err
    assert "verification" in error and "probability is 0" in error


def test_validate_names_missing_model_file(tmp_path, capsys) -> None:
    text = HARMONIC_CONFIG.replace(
        '[surrogate]\nbackend = "harmonic-surrogate"',
        '[surrogate]\nbackend = "mace"\nmodel = "models/mace.model"')
    config = write_config(tmp_path, text)
    assert run_cli("validate", str(config)) == 2
    error = capsys.readouterr().err
    assert "surrogate.model" in error and "file not found" in error


def test_validate_names_unknown_backend(tmp_path, capsys) -> None:
    config = write_config(tmp_path, HARMONIC_CONFIG.replace(
        'backend = "harmonic-reference"', 'backend = "vasp"'))
    assert run_cli("validate", str(config)) == 2
    error = capsys.readouterr().err
    assert "'vasp' is not registered" in error and "harmonic-reference" in error


def test_validate_mace_name_config_without_torch(tmp_path, capsys) -> None:
    """A MACE foundation-model *name* (not a path) validates without torch:
    construction is lazy, capability/contract checks need no weights."""
    block = ('[surrogate]\nbackend = "mace"\nmodel = "small"\n'
             'device = "cpu"\ndefault_dtype = "float64"')
    config = write_config(tmp_path, _plain_mode_config("surrogate", block))
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("needs a torch-free environment")
    output = run_cli_and_out(capsys, "validate", str(config))
    assert "surrogate   : mace" in output
    assert "torch" not in sys.modules


def test_validate_refuses_singlepoint_and_relax(tmp_path, capsys) -> None:
    for kind in ("singlepoint", "relax"):
        target = tmp_path / kind
        target.mkdir()
        config = write_config(target, HARMONIC_CONFIG.replace(
            'kind = "md"', f'kind = "{kind}"'))
        assert run_cli("validate", str(config)) == 2
        assert "WP07" in capsys.readouterr().err


def _plain_mode_config(mode: str, backend_block: str) -> str:
    """The harmonic template turned into a plain-mode config whose backend
    section is `backend_block` (including its [section] header)."""
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


def qe_config(tmp_path: Path, *, with_pseudos: bool) -> Path:
    block = ('[reference]\nbackend = "qe"\npseudo_dir = "pseudos"\n'
             'ecutwfc = 30.0\npseudos = { H = "H.upf", O = "O.upf" }')
    text = _plain_mode_config("reference", block)
    assert 'backend = "qe"' in text
    if with_pseudos:
        (tmp_path / "pseudos").mkdir()
        for species in ("H", "O"):
            (tmp_path / "pseudos" / f"{species}.upf").write_text(
                f'<UPF version="2.0.1"><PP_HEADER element="{species}"/></UPF>\n')
    return write_config(tmp_path, text)


def test_validate_qe_backend_names_missing_pseudo_dir(tmp_path, capsys) -> None:
    config = qe_config(tmp_path, with_pseudos=False)
    assert run_cli("validate", str(config)) == 2
    error = capsys.readouterr().err
    assert "reference.pseudo_dir" in error and "directory not found" in error


def test_validate_qe_backend_constructs_without_pw(tmp_path, capsys) -> None:
    """QE configs are validated through create_backend (parameters, paths,
    capabilities) without pw.x on this machine — no SCF is executed."""
    config = qe_config(tmp_path, with_pseudos=True)
    output = run_cli_and_out(capsys, "validate", str(config))
    assert "reference   : qe" in output
    assert "validate: OK" in output


def test_validate_probe_reports_missing_optional_dependency(tmp_path, capsys) -> None:
    block = ('[reference]\nbackend = "pyscf"\nfunctional = "pbe"\n'
             'basis = "def2-svp"')
    config = write_config(tmp_path, _plain_mode_config("reference", block))
    if importlib.util.find_spec("pyscf") is not None:
        pytest.skip("needs a pyscf-free environment")
    assert run_cli("validate", str(config), "--probe-backends") == 2
    error = capsys.readouterr().err
    assert "optional dependency" in error and "pyraimd2[pyscf]" in error


# ---------------------------------------------------------------------------
# the full acceptance flow (directory name contains a space; cwd changes
# between commands must not move the results)


def test_full_flow_space_directory_and_cwd_independence(tmp_path, capsys,
                                                        monkeypatch) -> None:
    project = tmp_path / "my run"
    config_path = project / "cfg" / "run.toml"

    assert run_cli("init", "--template", "harmonic", "--output",
                   str(project / "cfg")) == 0
    capsys.readouterr()

    # validate/run/inspect from an unrelated cwd: paths resolve against the
    # configuration file, not the caller's cwd
    monkeypatch.chdir(tmp_path)
    assert run_cli("validate", str(config_path)) == 0
    assert run_cli("run", str(config_path)) == 0
    run_output = capsys.readouterr().out
    run_dir = project / "cfg" / "runs" / "harmonic-demo"
    assert str(run_dir) in run_output
    assert (run_dir / "trajectory.db").is_file()
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "resolved_config.json").is_file()
    assert (run_dir / "checkpoints" / "latest.json").is_file()

    # duplicate run is refused with the field name and the remedy — and so
    # is re-validating a configuration whose run already exists
    assert run_cli("run", str(config_path)) == 2
    error = capsys.readouterr().err
    assert "run.id 'harmonic-demo' already exists" in error
    assert "resume" in error
    assert run_cli("validate", str(config_path)) == 2
    assert "already exists" in capsys.readouterr().err

    # inspect: human and JSON renderings come from the same source
    human = run_cli_and_out(capsys, "inspect", str(run_dir))
    assert "complete steps        : 20" in human
    json_output = run_cli_and_out(capsys, "inspect", str(run_dir), "--json")
    parsed = json.loads(json_output)
    assert parsed == json.loads(json.dumps(inspect_run(run_dir), default=str))
    assert parsed["n_complete_steps"] == 20

    # export: all three sources; missing reference labels are marked
    for source in ("driving", "reference", "base"):
        output = run_dir / f"{source}.extxyz"
        text = run_cli_and_out(capsys, "export", str(run_dir),
                               "--force-source", source, "--output",
                               str(output))
        assert "21 frames" in text
    reference_text = run_cli_and_out(capsys, "export", str(run_dir),
                                     "--force-source", "reference", "--force")
    assert "NaN" in reference_text and "never zero-filled" in reference_text

    # resume: --steps N means N additional steps; current and target printed
    resume_output = run_cli_and_out(capsys, "resume", str(run_dir),
                                    "--steps", "5")
    assert "at complete step 20" in resume_output
    assert "5 additional steps (target 25)" in resume_output
    final = run_cli_and_out(capsys, "inspect", str(run_dir))
    assert "complete steps        : 25" in final


def test_export_marks_missing_reference_labels(tmp_path) -> None:
    from ase.io import read as ase_read

    config_path = write_config(tmp_path, HARMONIC_CONFIG)
    assert run_cli("run", str(config_path)) == 0
    run_dir = tmp_path / "runs" / "harmonic-demo"
    output = run_dir / "ref.extxyz"
    assert run_cli("export", str(run_dir), "--force-source", "reference",
                   "--output", str(output)) == 0
    frames = ase_read(output, index=":")
    labeled = [f for f in frames if f.info["forces_available"]]
    missing = [f for f in frames if not f.info["forces_available"]]
    assert labeled and missing  # the adaptive run produces both kinds
    for frame in labeled:
        assert np.isfinite(frame.get_forces()).all()
    for frame in missing:
        forces = frame.get_forces()
        assert np.isnan(forces).all()  # missing marker, never zeros
        assert "energy" not in frame.calc.results


def test_resume_rejects_a_plain_mode_run(tmp_path, capsys) -> None:
    block = ('[surrogate]\nbackend = "harmonic-surrogate"\n'
             'k = 1.0\nr0 = 0.9\nbias = 0.05')
    config_path = write_config(tmp_path, _plain_mode_config("surrogate", block))
    assert run_cli("run", str(config_path)) == 0
    capsys.readouterr()
    run_dir = tmp_path / "runs" / "harmonic-demo"
    assert run_cli("resume", str(run_dir), "--steps", "2") == 2
    assert "adaptive" in capsys.readouterr().err


def test_resume_stale_lock_requires_deliberate_force_unlock(tmp_path, capsys) -> None:
    config_path = write_config(tmp_path, HARMONIC_CONFIG)
    assert run_cli("run", str(config_path)) == 0
    capsys.readouterr()
    run_dir = tmp_path / "runs" / "harmonic-demo"
    # a crashed writer leaves its lock behind; resume must refuse to share it
    (run_dir / "events.jsonl.lock").write_text("999999")
    assert run_cli("resume", str(run_dir), "--steps", "2") == 2
    assert "active writer" in capsys.readouterr().err
    output = run_cli_and_out(capsys, "resume", str(run_dir), "--steps", "2",
                             "--force-unlock")
    assert "now at complete step 22" in output
