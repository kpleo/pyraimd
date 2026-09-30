"""The ``qe-mace-compare`` init template: five embedded files, static
validation with placeholder resources (no QE/MACE installed here), honest
missing-resource reporting without any execution, path handling from
another cwd, and the drift guard against examples/qe_mace_mts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyraimd2.cli import main as cli_main
from pyraimd2.config import load_config
from pyraimd2.workflows import validate_setup
from pyraimd2.workflows.setup import WorkflowError
from pyraimd2.workflows.templates import write_template

TEMPLATE_FILES = {"run.toml", "run_mts.toml", "run_mts_scaled.toml",
                  "structure.extxyz", "README.md"}
CONFIGS = ("run.toml", "run_mts.toml", "run_mts_scaled.toml")
EXAMPLE = Path(__file__).parents[2] / "examples" / "qe_mace_mts"


def test_emits_exactly_five_files_and_static_validate(tmp_path,
                                                      capsys) -> None:
    out = tmp_path / "qe compare demo"          # space-carrying project dir
    code = cli_main(["init", "--template", "qe-mace-compare",
                     "--output", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    assert {p.name for p in out.iterdir()} == TEMPLATE_FILES
    assert "prepare resources first" in printed
    assert "README.md" in printed

    configs = [load_config(out / name) for name in CONFIGS]
    assert [c.task.mode for c in configs] == ["reference", "mts", "mts"]
    ids = [c.run.id for c in configs]
    assert len(set(ids)) == 3
    for config in configs:
        assert Path(config.run.directory).parent == out.resolve() / "runs"
        assert Path(config.structure.file) == out.resolve() \
            / "structure.extxyz"
    reference, mts, mts_scaled = configs
    # same reference recipe, same timestep and physical span; the
    # reference's 1 fs grid contains every complete candidate boundary
    assert mts.reference.options == reference.reference.options
    assert mts_scaled.reference.options == reference.reference.options
    assert reference.dynamics.timestep_fs == mts.dynamics.timestep_fs == 1.0
    assert reference.dynamics.steps == mts.dynamics.steps == 16
    assert mts.dynamics.outer_ratio == 4
    boundary_times = [i * 4.0 for i in range(5)]
    assert all(t <= reference.dynamics.steps * reference.dynamics.timestep_fs
               and t % reference.dynamics.timestep_fs == 0
               for t in boundary_times)

    # static validate passes with PLACEHOLDER resources present — no real
    # QE binary, model weights or torch/mace package anywhere in this env
    (out / "pseudos").mkdir()
    (out / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF").write_text(
        "placeholder pseudo\n")
    (out / "models").mkdir()
    (out / "models" / "user.model").write_bytes(b"placeholder model\n")
    for config in configs:
        report = validate_setup(config)
        assert report["structure"]["n_atoms"] == 2
    for module in ("mace", "torch", "pyscf"):
        assert module not in sys.modules


def test_missing_resources_reported_without_executing(tmp_path,
                                                      capsys) -> None:
    out = tmp_path / "proj"
    write_template("qe-mace-compare", out)
    code = cli_main(["validate", str(out / "run.toml"),
                     "--check-environment"])
    captured = capsys.readouterr()
    assert code == 2
    # the concrete missing items are named as to-prepare items
    assert "pseudopotential not found" in captured.err
    assert "pseudo_dir" in captured.err
    # nothing executed, constructed or downloaded: no run artifacts, no
    # backend modules imported, no computation directories
    assert not (out / "runs").exists()
    for module in ("mace", "torch", "pyscf"):
        assert module not in sys.modules
    # the --json form is one parseable structured error object
    code = cli_main(["validate", str(out / "run_mts.toml"), "--json"])
    payload = __import__("json").loads(capsys.readouterr().out)
    assert code == 2
    assert payload["configuration_valid"] is False
    assert "pseudopotential not found" in payload["error"]["message"]


def test_space_paths_other_cwd_and_conflict_semantics(tmp_path,
                                                      monkeypatch) -> None:
    out = tmp_path / "my project"
    write_template("qe-mace-compare", out)
    pristine = {name: (out / name).read_bytes() for name in TEMPLATE_FILES}
    # configs load from a DIFFERENT cwd; the structure resolves inside the
    # project (relative paths bind to each TOML's own directory)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for name in CONFIGS:
        config = load_config(out / name)
        assert Path(config.structure.file).is_file()
        assert Path(config.structure.file) == out.resolve() \
            / "structure.extxyz"
    # overwrite refusal: ANY existing target writes NOTHING
    marker = tmp_path / "second"
    marker.mkdir()
    (marker / "run_mts.toml").write_text("# user's own\n")
    with pytest.raises(WorkflowError, match="--force"):
        write_template("qe-mace-compare", marker)
    assert {p.name for p in marker.iterdir()} == {"run_mts.toml"}
    # --force touches only the five template files; user resources survive
    (marker / "models").mkdir()
    (marker / "models" / "user.model").write_bytes(b"user model bytes")
    (marker / "runs").mkdir()
    (marker / "runs" / "keep.txt").write_text("existing run data")
    write_template("qe-mace-compare", marker, force=True)
    assert (marker / "run_mts.toml").read_bytes() == pristine["run_mts.toml"]
    assert (marker / "models" / "user.model").read_bytes() == \
        b"user model bytes"
    assert (marker / "runs" / "keep.txt").read_text() == \
        "existing run data"


def test_template_matches_qe_mace_mts_example(tmp_path) -> None:
    """Drift guard: the embedded template must stay semantically identical
    to examples/qe_mace_mts/ (run ids/dirs/file names intentionally
    differ; every physical and backend setting is shared)."""
    import os

    def _re_relativize(value, base: Path):
        """Resolved absolute paths back to their config-relative form."""
        if isinstance(value, dict):
            return {key: _re_relativize(item, base)
                    for key, item in value.items()}
        if isinstance(value, str) and os.path.isabs(value):
            return os.path.relpath(value, base)
        return value

    out = tmp_path / "demo"
    write_template("qe-mace-compare", out)
    # the structure is byte-identical (same initial state incl. momenta)
    assert (out / "structure.extxyz").read_bytes() == \
        (EXAMPLE / "si_diamond_2atom.extxyz").read_bytes()
    example_configs = [load_config(EXAMPLE / name) for name in
                       ("run_reference.toml", "run_mts.toml",
                        "run_mts_scaled.toml")]
    template_configs = [load_config(out / name) for name in CONFIGS]
    for example, template, name in zip(example_configs, template_configs,
                                       CONFIGS):
        assert example.task.kind == template.task.kind
        assert example.task.mode == template.task.mode
        for field in ("ensemble", "integrator", "timestep_fs", "steps",
                      "outer_ratio"):
            assert getattr(example.dynamics, field) == \
                getattr(template.dynamics, field), field
        assert (_re_relativize(example.reference.options,
                               (EXAMPLE / name).parent)
                == _re_relativize(template.reference.options,
                                  template.source_path.parent))
        assert (example.surrogate is None) == (template.surrogate is None)
        if example.surrogate is not None:
            assert example.surrogate.name == template.surrogate.name
            assert (_re_relativize(example.surrogate.options,
                                   EXAMPLE)
                    == _re_relativize(template.surrogate.options,
                                      template.source_path.parent))
        assert example.checkpoint.interval_steps == \
            template.checkpoint.interval_steps
