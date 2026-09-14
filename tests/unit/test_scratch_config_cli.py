"""[scratch] configuration, workflow injection and CLI entries."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from ase import Atoms

from pyraimd2.cli import main as cli_main
from pyraimd2.config import ConfigError, load_config, load_resolved_config
from pyraimd2.engines.qe_engine import QeConfig, QeEngine
from pyraimd2.workflows.setup import build_backends

FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"


def _fake_pwx(tmp_path: Path, body: str) -> tuple[str, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_pwx.sh"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                 | stat.S_IXOTH)
    return ("bash", str(script))


def _success_script(tmp_path: Path) -> tuple[str, ...]:
    return _fake_pwx(
        tmp_path,
        "#!/bin/bash\n"
        "mkdir -p tmp/pyraimd2.save && echo fake-density > "
        "tmp/pyraimd2.save/charge-density.dat\n"
        f"cat {FIXTURE.resolve()}\n")


def _write_config(root: Path, scratch: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "structure.extxyz").write_text(
        "2\nH2\nH 0.85 0.9 0.9\nH 0.95 0.9 0.9\n")
    path = root / "run.toml"
    path.write_text(
        "schema_version = 1\n[run]\nid = \"t\"\ndirectory = \".\"\n"
        "seed = 42\n[task]\nkind = \"md\"\nmode = \"reference\"\n"
        "[structure]\nfile = \"structure.extxyz\"\n"
        "[reference]\nbackend = \"harmonic-reference\"\nk = 1.0\nr0 = 0.9\n"
        "[dynamics]\nensemble = \"nve\"\ntimestep_fs = 0.5\nsteps = 1\n"
        "temperature_K = 300.0\nvelocity_seed = 7\n" + scratch)
    return path


def test_scratch_section_resolves_against_config_file(tmp_path, monkeypatch):
    project = tmp_path / "proj" / "deep"
    project.mkdir(parents=True)
    config_path = _write_config(project, '[scratch]\nroot = "./tmp"\n'
                                'retention = "results"\n')
    monkeypatch.chdir(tmp_path / "proj")  # a different caller cwd
    config = load_config(config_path)
    assert config.scratch is not None
    assert config.scratch.root == (project / "tmp").resolve()
    assert config.scratch.retention == "results"


def test_scratch_section_validation(tmp_path):
    bad = _write_config(tmp_path / "a", '[scratch]\nroot = "./tmp"\n'
                        'retention = "everything"\n')
    with pytest.raises(ConfigError, match="retention"):
        load_config(bad)
    bad2 = _write_config(tmp_path / "b", '[scratch]\nroot = "./tmp"\n'
                         'extra = 1\n')
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(bad2)


def test_resolved_round_trip_and_legacy_bytes(tmp_path):
    plain = load_config(_write_config(tmp_path / "plain"))
    assert "scratch" not in plain.resolved_dict()
    with_scratch = load_config(_write_config(
        tmp_path / "s", '[scratch]\nroot = "./tmp"\n'))
    document = with_scratch.resolved_dict()
    assert document["scratch"] == {
        "root": str((tmp_path / "s" / "tmp").resolve()),
        "retention": "all"}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.json").write_text(
        json.dumps(document, indent=2) + "\n")
    rebuilt = load_resolved_config(run_dir)
    assert rebuilt.scratch is not None
    assert rebuilt.scratch.root == with_scratch.scratch.root
    assert rebuilt.scratch.retention == "all"


def test_build_backends_injects_only_declared_factories(tmp_path):
    config_path = tmp_path / "qe.toml"
    config_path.write_text(
        "schema_version = 1\n[run]\nid = \"t\"\ndirectory = \".\"\n"
        "seed = 42\n[task]\nkind = \"md\"\nmode = \"reference\"\n"
        "[structure]\nfile = \"structure.extxyz\"\n"
        "[reference]\nbackend = \"qe\"\npseudo_dir = \"/pseudo\"\n"
        "[dynamics]\nensemble = \"nve\"\ntimestep_fs = 0.5\nsteps = 1\n"
        "temperature_K = 300.0\nvelocity_seed = 7\n"
        "[scratch]\nroot = \"./tmp\"\nretention = \"results\"\n")
    (tmp_path / "structure.extxyz").write_text(
        "2\nH2\nH 0.85 0.9 0.9\nH 0.95 0.9 0.9\n")
    config = load_config(config_path)
    engine, _ = build_backends(config, run_dir=tmp_path / "run")
    assert engine.config.scratch_root == \
        str((tmp_path / "tmp").resolve())
    assert engine.config.retention == "results"
    # a backend without the scratch declaration is never offered the
    # parameters
    harmonic = load_config(_write_config(
        tmp_path / "h", '[scratch]\nroot = "./tmp"\n'))
    engine2, _ = build_backends(harmonic, run_dir=tmp_path / "run2")
    assert getattr(getattr(engine2, "config", None), "scratch_root",
                   None) is None


def test_cli_scratch_inspect_and_dry_run_clean(tmp_path, capsys):
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")
    engine.compute(Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                         cell=[5.43] * 3, pbc=True), label="si")
    assert cli_main(["scratch", "inspect", "--root",
                     str(tmp_path / "tmp")]) == 0
    out = capsys.readouterr().out
    assert "cleaned" in out
    assert cli_main(["scratch", "clean", "--root", str(tmp_path / "tmp"),
                     "--dry-run"]) == 0
    assert "reclaimable" in capsys.readouterr().out


def _qe_workflow_config(root: Path, kind: str, scratch: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "structure.extxyz").write_text(
        "2\n"
        'Lattice="5.43 0.0 0.0 0.0 5.43 0.0 0.0 0.0 5.43" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Si 0.0 0.0 0.0\nSi 1.36 1.36 1.36\n")
    # validate_setup requires an existing pseudo_dir; the fake pw.x never
    # reads real UPFs, so an empty directory satisfies the path check
    (root / "pseudos").mkdir(exist_ok=True)
    script = _success_script(root)
    path = root / "run.toml"
    path.write_text(
        "schema_version = 1\n[run]\nid = \"t\"\ndirectory = \"run\"\n"
        "seed = 42\n[task]\nkind = \"" + kind + "\"\nmode = \"reference\"\n"
        "[structure]\nfile = \"structure.extxyz\"\n"
        "[reference]\nbackend = \"qe\"\npseudo_dir = \"" + str(root / "pseudos") + "\"\n"
        "pw_cmd = [\"bash\", \"" + str(script[1]) + "\"]\n"
        "[dynamics]\nensemble = \"nve\"\ntimestep_fs = 0.5\nsteps = 1\n"
        "temperature_K = 300.0\nvelocity_seed = 7\n" + scratch)
    return path


def _assert_scratch_used(root: Path) -> None:
    run_dir = root / "run"
    # attempts ran under the managed root, durable results archived into
    # the run directory, records kept outside the scratch root
    assert list((root / "sc").rglob("pw.out")), "no attempt ran in scratch"
    assert list((run_dir / "calculations").rglob("pw.out")), \
        "no archived pw.out in the run directory"
    assert list((run_dir / "scratch_records").rglob("*.json")), \
        "no scratch records written"


def test_plain_singlepoint_honours_scratch_section(tmp_path):
    """run_workflow's plain path routes [scratch] into the QE engine
    (regression: _plain_backend bypassed the build_backends merge)."""
    from pyraimd2.workflows.md import run_workflow

    root = tmp_path / "singlepoint"
    config = load_config(_qe_workflow_config(
        root, "singlepoint", '[scratch]\nroot = "./sc"\nretention = "all"\n'))
    run_workflow(config, verbose=False, handle_sigint=False)
    _assert_scratch_used(root)


def test_plain_md_honours_scratch_section(tmp_path):
    root = tmp_path / "md"
    config = load_config(_qe_workflow_config(
        root, "md", '[scratch]\nroot = "./sc"\nretention = "all"\n'))
    run_workflow(config, verbose=False, handle_sigint=False)
    _assert_scratch_used(root)
