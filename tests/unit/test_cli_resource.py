"""CLI --resource tests (E3): argument parsing and side effects, a real
two-process relocation closed loop driven through the actual CLI from a
different working directory, and refusal/compatibility checks.

The file-backed analytic backends are shared with
test_resource_relocation (the stiffness genuinely feeds energy and
forces); the closed loop compares against a continuous control at
rtol=0, atol=1e-12.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from test_file_resource_baseline import _checkpoint_states, _write_run
from test_resource_relocation import (
    CountingFactory,
    _assert_same_decisions,
    _assert_same_trajectory,
    _child_env,
    _inject_test_backend,
    _prepare_run,
    _relocate,
    _run_child,
    _snapshot_records,
)

from pyraimd2.cli import main as cli_main
from pyraimd2.engines.ase_resources import file_resource_baseline_sha256


def _recorder(monkeypatch):
    calls = []

    def fake_resume(run_dir, extra_steps, **kwargs):
        calls.append({"run_dir": run_dir, "extra_steps": extra_steps,
                      **kwargs})

    monkeypatch.setattr("pyraimd2.workflows.resume_workflow", fake_resume)
    return calls


def test_resource_parse_rules_and_zero_workflow_calls(tmp_path, monkeypatch,
                                                      capsys):
    """Malformed --resource values refuse with exit 2 BEFORE any workflow
    call; well-formed ones are handed over absolute, split at the first
    '=', relative paths based on the caller's cwd."""
    calls = _recorder(monkeypatch)
    run_dir = str(tmp_path / "run")  # the workflow is mocked; no real dir
    monkeypatch.chdir(tmp_path)

    bad = [
        ["reference.potential"],                          # no '='
        ["=./x.dat"],                                     # empty key
        ["reference.potential="],                         # empty path
        ["reference.potential=./a", "reference.potential=./a"],  # dup, same
        ["reference.potential=./a", "reference.potential=./b"],  # dup
    ]
    for values in bad:
        argv = ["resume", run_dir, "--steps", "1"]
        for value in values:
            argv += ["--resource", value]
        assert cli_main(argv) == 2, values
        assert calls == []  # the workflow was never called
        assert "--resource" in capsys.readouterr().err

    # multiple roles; a path containing a later '=', spaces, CJK and a
    # non-BMP character — all preserved verbatim after the first '='
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "models").mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    odd = "mod 模型😀=v2.dat"
    assert cli_main(["resume", run_dir, "--steps", "2",
                     "--resource", "reference.potential=./models/ref.dat",
                     "--resource", f"surrogate.potential=./models/{odd}"]) == 0
    assert len(calls) == 1
    handed = calls[0]["resource_paths"]
    assert handed == {
        "reference.potential": str(elsewhere / "models" / "ref.dat"),
        "surrogate.potential": str(elsewhere / "models" / odd)}
    assert Path(handed["surrogate.potential"]).is_absolute()

    # without --resource the resume keeps its original semantics
    assert cli_main(["resume", run_dir, "--steps", "1"]) == 0
    assert calls[1]["resource_paths"] is None


_CLI_CHILD = '''
import sys
import types

from test_resource_relocation import FileBacked

from pyraimd2.backends import registry

registry._BUILTINS["file-backed-test"] = ("engine", "fake_file_module",
                                          "create")
registry._BUILTINS["file-backed-test-surr"] = ("surrogate",
                                               "fake_file_module_surr",
                                               "create")
module = types.ModuleType("fake_file_module")
module.create = FileBacked.factory
sys.modules["fake_file_module"] = module
module_s = types.ModuleType("fake_file_module_surr")
module_s.create = FileBacked.surrogate_factory
sys.modules["fake_file_module_surr"] = module_s

from pyraimd2.cli import main

sys.exit(main(sys.argv[1:]))
'''


def _run_cli_child(tmp_path, argv, *, cwd):
    """One real CLI process (argv through the actual argparse/main path)
    with the test backends registered, run from the given cwd."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    child = tmp_path / "cli_child.py"
    child.write_text(textwrap.dedent(_CLI_CHILD))
    return subprocess.run([sys.executable, str(child), *argv],
                          env=_child_env(), cwd=str(cwd),
                          capture_output=True, text=True, check=False)


def _prepare_two_file_adaptive(tmp_path, *, steps=3):
    """Process A: a fresh fixed-model adaptive run with TWO declared file
    resources (reference + surrogate), then exit."""
    root = tmp_path / "origin"
    model_ref = root / "inputs" / "ref.dat"
    model_sur = root / "inputs" / "sur.dat"
    root.mkdir(parents=True)
    model_ref.parent.mkdir()
    model_ref.write_text("1.5\n")
    model_sur.write_text("0.75\n")
    config_path = _write_run(
        root / "run", mode="adaptive", steps=steps, checkpoint=1,
        backend="file-backed-test-surr",
        options=f'model = "{model_sur}"',
        reference_backend="file-backed-test",
        reference_options=f'model = "{model_ref}"')
    result = _run_child(tmp_path, "fresh", MODE="fresh",
                        CONFIG=str(config_path))
    assert result.returncode == 0, result.stderr[-500:]
    return root, model_ref, model_sur


def test_cli_relocation_closed_loop_from_another_cwd(tmp_path, capsys):
    """Fixed-model adaptive, two declared files: fresh 3 steps, relocate
    (paths with a space and CJK), old paths gone, then resume +2 through
    the REAL CLI from an unrelated cwd with RELATIVE --resource paths —
    trajectory and adaptive decisions identical to the continuous control.
    """
    root, model_ref, model_sur = _prepare_two_file_adaptive(tmp_path)
    moved = tmp_path / "搬迁 relocated"
    moved.mkdir()
    new_root = moved / "run"
    import shutil
    shutil.move(str(root / "run"), str(new_root))
    moved_ref = moved / "ref.dat"
    moved_sur = moved / "sur.dat"
    shutil.move(str(model_ref), str(moved_ref))
    shutil.move(str(model_sur), str(moved_sur))
    assert not model_ref.exists() and not model_sur.exists()

    # the CLI runs from an unrelated cwd; the models sit under it, so the
    # RELATIVE --resource paths only resolve by the cwd rule
    cli_cwd = tmp_path / "cli work"
    (cli_cwd / "models").mkdir(parents=True)
    shutil.copy(moved_ref, cli_cwd / "models" / "ref.dat")
    shutil.copy(moved_sur, cli_cwd / "models" / "sur.dat")
    result = _run_cli_child(
        tmp_path / "cli",
        ["resume", str(new_root), "--steps", "2",
         "--resource", "reference.potential=models/ref.dat",
         "--resource", "surrogate.potential=models/sur.dat"],
        cwd=cli_cwd)
    assert result.returncode == 0, result.stderr[-500:]

    # inspect and export through the CLI read the complete result
    assert cli_main(["inspect", str(new_root)]) == 0
    assert "complete steps        : 5" in capsys.readouterr().out
    export_path = tmp_path / "out" / "traj.extxyz"
    assert cli_main(["export", str(new_root), "--force-source", "driving",
                     "--output", str(export_path)]) == 0
    assert "6 frames" in capsys.readouterr().out

    # the control: one uninterrupted 5-step run of the same setup
    control_root, _, _ = _prepare_two_file_adaptive(
        tmp_path / "control", steps=5)
    _assert_same_trajectory(new_root, control_root / "run")
    _assert_same_decisions(new_root, control_root / "run")

    # one binding receipt; the new checkpoint keeps the baseline
    # association; exported frames carry real driving forces
    receipts = sorted((new_root / "resource_bindings").glob("*.json"))
    assert len(receipts) == 1
    assert _checkpoint_states(new_root)[-1][
        "file_resource_baseline_sha256"] == \
        file_resource_baseline_sha256(new_root)


def test_cli_refusal_and_legacy_compat(tmp_path, monkeypatch, capsys):
    """C: wrong content / missing mapped file through the CLI refuse with
    exit 2 and zero new compute, records byte-identical, no receipts; the
    corrected mapping then continues.  A run without declared resources
    resumed without --resource keeps the legacy CLI behavior."""
    counter = CountingFactory()
    _inject_test_backend(monkeypatch, factory=counter.engine)
    root, model = _prepare_run(tmp_path / "ref")
    new_root, new_model = _relocate(tmp_path / "ref", root, model)
    before = _snapshot_records(new_root)
    # wrong content (baseline bytes differ)
    new_model.write_bytes(b"2.5\n")
    code = cli_main(["resume", str(new_root), "--steps", "1",
                     "--resource",
                     f"reference.potential={new_model}"])
    assert code == 2
    assert "does not match the baseline" in capsys.readouterr().err
    # a missing mapped file
    code = cli_main(["resume", str(new_root), "--steps", "1",
                     "--resource",
                     f"reference.potential={tmp_path / 'ref' / 'none.dat'}"])
    assert code == 2
    assert "not an existing regular file" in capsys.readouterr().err
    # both refusals: zero factory/compute, records untouched, no receipts
    assert counter.factory_calls == 0
    assert counter.compute_calls == 0
    assert _snapshot_records(new_root) == before
    # the corrected mapping continues through the CLI
    new_model.write_bytes(b"1.5\n")
    code = cli_main(["resume", str(new_root), "--steps", "2",
                     "--resource",
                     f"reference.potential={new_model}"])
    assert code == 0
    assert "now at complete step 5" in capsys.readouterr().out
    assert len(list((new_root / "resource_bindings").glob("*.json"))) == 1

    # legacy compatibility: no declared resources, no --resource
    config_path = _write_run(tmp_path / "plain" / "run",
                             backend="harmonic-reference", steps=2,
                             checkpoint=1, options="k = 1.0\nr0 = 0.9")
    from pyraimd2.config import load_config
    from pyraimd2.workflows import run_workflow
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    code = cli_main(["resume", str(tmp_path / "plain" / "run"),
                     "--steps", "1"])
    assert code == 0
    assert "now at complete step 3" in capsys.readouterr().out
