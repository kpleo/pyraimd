"""The ``harmonic-compare`` init template: five embedded files, atomic
conflict semantics, and the full offline verify chain (validate -> three
runs -> two comparisons) invoked from a different working directory —
the wheel-only user path, with analytic backends only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyraimd2.cli import main as cli_main
from pyraimd2.config import load_config
from pyraimd2.workflows import compare_runs, run_workflow
from pyraimd2.workflows.setup import WorkflowError
from pyraimd2.workflows.templates import write_template

TEMPLATE_FILES = {"run.toml", "run_mts.toml", "run_mts_scaled.toml",
                  "structure.extxyz", "README.md"}
CONFIGS = ("run.toml", "run_mts.toml", "run_mts_scaled.toml")


def test_emits_exactly_five_files(tmp_path, capsys) -> None:
    out = tmp_path / "demo"
    code = cli_main(["init", "--template", "harmonic-compare",
                     "--output", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    assert {p.name for p in out.iterdir()} == TEMPLATE_FILES
    assert "run_mts.toml" in printed and "README.md" in printed
    # the primary config keeps write_template's return contract
    assert write_template("harmonic-compare", tmp_path / "again",
                          force=True) == tmp_path / "again" / "run.toml"


def test_configs_validate_and_share_one_initial_state(tmp_path) -> None:
    out = tmp_path / "demo"
    write_template("harmonic-compare", out)
    configs = [load_config(out / name) for name in CONFIGS]
    # schema validation passes, three distinct runs inside the output dir
    ids = [config.run.id for config in configs]
    assert len(set(ids)) == 3
    for config in configs:
        directory = Path(config.run.directory)
        assert directory.parent == out.resolve() / "runs"
        assert directory.is_relative_to(out.resolve())
        # the shared structure resolves next to each config file
        assert Path(config.structure.file) == out.resolve() / "structure.extxyz"
    modes = [config.task.mode for config in configs]
    assert modes == ["reference", "mts", "mts"]


def test_full_chain_runs_from_a_different_cwd(tmp_path, monkeypatch,
                                              capsys) -> None:
    out = tmp_path / "demo"
    write_template("harmonic-compare", out)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)   # relative paths bind to the TOML's dir
    for name in CONFIGS:
        run_workflow(load_config(out / name), verbose=False,
                     handle_sigint=False)
    reference = out / "runs" / "reference"
    for candidate_name in ("mts", "mts-scaled"):
        candidate = out / "runs" / candidate_name
        report = compare_runs(reference, candidate)
        assert report["criteria_status"] == "not_requested"
        assert report["time_axis"]["matched_points"] == 5
        assert report["time_axis"]["start_fs"] == 0.0
        assert report["time_axis"]["end_fs"] == 16.0
        assert report["time_axis"]["reference_complete_states"] == 17
        json.dumps(report, allow_nan=False)
        code = cli_main(["compare", str(reference), str(candidate),
                         "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["criteria_status"] == "not_requested"


@pytest.mark.parametrize("target", ["run_mts.toml", "run_mts_scaled.toml"])
def test_conflict_without_force_writes_nothing(tmp_path, target) -> None:
    out = tmp_path / "demo"
    out.mkdir()
    (out / target).write_text("# user's own file\n")
    with pytest.raises(WorkflowError, match="--force"):
        write_template("harmonic-compare", out)
    # the pre-existing file is intact and NOT ONE template file was written
    assert (out / target).read_text() == "# user's own file\n"
    assert {p.name for p in out.iterdir()} == {target}


def test_force_overwrites_only_the_template_files(tmp_path) -> None:
    out = tmp_path / "demo"
    write_template("harmonic-compare", out)
    pristine = {name: (out / name).read_text() for name in TEMPLATE_FILES}
    # user edits plus unrelated files and existing run data
    for name in TEMPLATE_FILES:
        (out / name).write_text("# edited\n")
    (out / "keep.txt").write_text("unrelated\n")
    run_data = out / "runs" / "reference"
    run_data.mkdir(parents=True)
    (run_data / "trajectory.db").write_bytes(b"existing run data")
    write_template("harmonic-compare", out, force=True)
    for name in TEMPLATE_FILES:
        assert (out / name).read_text() == pristine[name]
    assert (out / "keep.txt").read_text() == "unrelated\n"
    assert (run_data / "trajectory.db").read_bytes() == b"existing run data"


def test_existing_templates_keep_their_two_file_contract(tmp_path) -> None:
    for template in ("harmonic", "harmonic-nvt", "harmonic-adaptive-nvt",
                     "harmonic-mts"):
        out = tmp_path / template
        assert write_template(template, out) == out / "run.toml"
        assert {p.name for p in out.iterdir()} == {"run.toml",
                                                   "structure.extxyz"}
