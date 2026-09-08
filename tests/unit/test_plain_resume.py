"""WP07 plain-mode (reference/surrogate-only) checkpoint/resume identity."""

from __future__ import annotations

import numpy as np

from pyraimd2.config import load_config
from pyraimd2.store import Store
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE


def _write_config(tmp_path, *, mode: str, steps: int,
                  checkpoint_interval: int = 5) -> object:
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = HARMONIC_CONFIG
    unused = "[surrogate]" if mode == "reference" else "[reference]"
    for section in (unused, "[policy]", "[verification]"):
        start = text.index(section)
        following = text.index("\n[", start + 1)
        text = text[:start] + text[following + 1:]
    text = text.replace('mode = "adaptive"', f'mode = "{mode}"')
    text = text.replace("steps = 20", f"steps = {steps}")
    text = text.replace("interval_steps = 5",
                        f"interval_steps = {checkpoint_interval}")
    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def _rows(run_dir, run_id="harmonic-demo"):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def _assert_same_run(rows_a, rows_b):
    assert len(rows_a) == len(rows_b) > 0
    for a, b in zip(rows_a, rows_b):
        assert int(a.key_value_pairs["step"]) == int(b.key_value_pairs["step"])
        assert a.key_value_pairs["route"] == b.key_value_pairs["route"]
        assert a.data.get("engine_label_id") == b.data.get("engine_label_id")
        np.testing.assert_allclose(a.toatoms().positions, b.toatoms().positions,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.toatoms().get_momenta(),
                                   b.toatoms().get_momenta(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(a.data["driving"]["forces"], dtype=float),
            np.asarray(b.data["driving"]["forces"], dtype=float),
            rtol=0, atol=1e-12)


def test_plain_reference_resume_matches_continuous(tmp_path):
    continuous = load_config(_write_config(tmp_path / "continuous",
                                           mode="reference", steps=40))
    run_workflow(continuous, verbose=False, handle_sigint=False)

    stopped = load_config(_write_config(tmp_path / "resumed", mode="reference",
                                        steps=30))
    run_workflow(stopped, verbose=False, handle_sigint=False)
    result = resume_workflow(stopped.run.directory, 10, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 40
    _assert_same_run(_rows(continuous.run.directory),
                     _rows(stopped.run.directory))


def test_plain_surrogate_resume_matches_continuous(tmp_path):
    continuous = load_config(_write_config(tmp_path / "continuous-s",
                                           mode="surrogate", steps=24))
    run_workflow(continuous, verbose=False, handle_sigint=False)

    stopped = load_config(_write_config(tmp_path / "resumed-s",
                                        mode="surrogate", steps=14))
    run_workflow(stopped, verbose=False, handle_sigint=False)
    result = resume_workflow(stopped.run.directory, 10, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 24
    _assert_same_run(_rows(continuous.run.directory),
                     _rows(stopped.run.directory))


def test_plain_resume_uses_latest_checkpoint_and_appends(tmp_path):
    config = load_config(_write_config(tmp_path / "window", mode="reference",
                                       steps=17, checkpoint_interval=10))
    run_workflow(config, verbose=False, handle_sigint=False)
    # checkpoint at step 10; steps 11..17 replay from events/trajectory
    result = resume_workflow(config.run.directory, 3, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 20
    assert len(_rows(config.run.directory)) == 21


def test_plain_resume_with_fixatoms_keeps_layer_fixed(tmp_path):
    directory = tmp_path / "fix"
    path = _write_config(directory, mode="surrogate", steps=8)
    text = path.read_text() + '\n[constraints]\nfix_atoms_indices = [0]\n'
    path.write_text(text)
    stopped = load_config(path)
    run_workflow(stopped, verbose=False, handle_sigint=False)
    before = _rows(stopped.run.directory)[0].toatoms().positions[0].copy()
    result = resume_workflow(stopped.run.directory, 4, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 12
    for row in _rows(stopped.run.directory):
        np.testing.assert_array_equal(row.toatoms().positions[0], before)
