"""QE density chain across processes: the checkpoint carries the run's own
last successful density and a resumed process re-anchors its first SCF on
it (fake pw.x, plain reference MD — same fixture style as
test_scratch_config_cli.py / test_workflows.py continuity tests)."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from pyraimd2.config import load_config
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.workflows import resume_workflow, run_workflow

FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"


def _fake_pwx(tmp_path: Path, body: str) -> tuple[str, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_pwx.sh"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                 | stat.S_IXOTH)
    return ("bash", str(script))


def _write_qe_config(root: Path, *, steps: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "structure.extxyz").write_text(
        "2\n"
        'Lattice="5.43 0.0 0.0 0.0 5.43 0.0 0.0 0.0 5.43" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Si 0.0 0.0 0.0\nSi 1.36 1.36 1.36\n")
    # validate_setup requires an existing pseudo_dir and the mapped UPF
    # files; the fake pw.x never reads them, placeholders satisfy the check
    (root / "pseudos").mkdir(exist_ok=True)
    (root / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF").write_text(
        "placeholder UPF for a fake-pwx run\n")
    script = _fake_pwx(
        root,
        "#!/bin/bash\n"
        "mkdir -p tmp/pyraimd2.save && echo fake-density > "
        "tmp/pyraimd2.save/charge-density.dat\n"
        f"cat {FIXTURE.resolve()}\n")
    path = root / "run.toml"
    path.write_text(
        "schema_version = 1\n[run]\nid = \"t\"\ndirectory = \"run\"\n"
        "seed = 42\n[task]\nkind = \"md\"\nmode = \"reference\"\n"
        "[structure]\nfile = \"structure.extxyz\"\n"
        "[reference]\nbackend = \"qe\"\npseudo_dir = \"" + str(root / "pseudos") + "\"\n"
        "pw_cmd = [\"bash\", \"" + str(script[1]) + "\"]\n"
        "startpot_file = true\n"
        "[dynamics]\nensemble = \"nve\"\ntimestep_fs = 0.5\n"
        f"steps = {steps}\n"
        "temperature_K = 300.0\nvelocity_seed = 7\n"
        # a checkpoint per step, so the resume checkpoint names the run's
        # own last evaluation
        "[checkpoint]\ninterval_steps = 1\n")
    return path


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in
            (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]


def _checkpoint_state(run_dir: Path) -> dict:
    checkpoint = CheckpointManager(run_dir).read_latest_valid()
    assert checkpoint is not None
    return checkpoint.state


def _resumed_split(events: list[dict]) -> tuple[dict, list[dict]]:
    resumed = [e for e in events if e.get("type") == "resumed"][-1]
    after = [e for e in events if int(e["seq"]) > int(resumed["seq"])]
    return resumed, after


def test_resume_reanchors_on_the_runs_own_last_density(tmp_path) -> None:
    """Run 2 steps, resume for 1 in a fresh process: the first SCF after
    the resume starts from this trajectory's own last density, not from
    an atomic guess."""
    root = tmp_path / "chain"
    config = load_config(_write_qe_config(root, steps=2))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = root / "run"
    chain = _checkpoint_state(run_dir).get("density_chain")
    assert chain is not None
    assert (Path(chain["density_dir"]) / "density_manifest.json").is_file()

    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3

    resumed, after = _resumed_split(_events(run_dir))
    assert resumed["density_chain"]["adopted"] is True
    assert resumed["density_chain"]["density_dir"] == chain["density_dir"]
    attempt = next(e for e in after if e.get("type") == "attempt")
    assert attempt["start"] == "density"
    copy = next(e for e in after if e.get("record") == "physical_io")
    assert copy["provenance"]["from"] == chain["density_dir"]


def test_resume_refuses_a_tampered_density_manifest(tmp_path) -> None:
    """A manifest modified after the checkpoint was written no longer
    matches the recorded sha256: the chain is refused (resume does not
    fail) and the first SCF falls back to an atomic start, with the
    refusal reason on the resumed event."""
    root = tmp_path / "tampered"
    config = load_config(_write_qe_config(root, steps=2))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = root / "run"
    chain = _checkpoint_state(run_dir)["density_chain"]
    manifest_path = Path(chain["density_dir"]) / "density_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_unix"] = 0.0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3

    resumed, after = _resumed_split(_events(run_dir))
    refusal = resumed["density_chain"]
    assert refusal["adopted"] is False
    assert refusal["density_dir"] == chain["density_dir"]
    assert "manifest" in refusal["reason"]
    attempt = next(e for e in after if e.get("type") == "attempt")
    assert attempt["start"] == "atomic"
    assert not [e for e in after if e.get("record") == "physical_io"]


def test_resume_from_a_checkpoint_without_density_chain(tmp_path) -> None:
    """Old checkpoints carry no density_chain key: resume behaves exactly
    as before (no restore attempt, atomic start with no configured
    source) and records no density_chain on the resumed event."""
    root = tmp_path / "legacy"
    config = load_config(_write_qe_config(root, steps=2))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = root / "run"
    # strip the key and re-checksum, simulating a pre-feature checkpoint
    checkpoint = CheckpointManager(run_dir).read_latest_valid()
    state_path = checkpoint.directory / "state.json"
    state = json.loads(state_path.read_text())
    assert state.pop("density_chain") is not None
    state_path.write_text(json.dumps(state, sort_keys=True))
    manifest_path = checkpoint.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["state.json"] = hashlib.sha256(
        state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3

    resumed, after = _resumed_split(_events(run_dir))
    assert "density_chain" not in resumed
    attempt = next(e for e in after if e.get("type") == "attempt")
    assert attempt["start"] == "atomic"
