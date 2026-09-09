"""Review R2 regression: model artifact integrity (F04) and plain surrogate
resume identity checks."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from test_guarded_update import TrainableHarmonic

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.runtime import ResumeError
from pyraimd2.runtime.events import EventLog
from pyraimd2.runtime.models import artifact_digest
from pyraimd2.store import Store
from pyraimd2.workflows import WorkflowError, resume_workflow, run_workflow
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2, quartic=0.05):
        self.k, self.quartic = k, quartic
        self.attempts = 0

    def compute(self, atoms):
        self.attempts += 1
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2
                                         + 0.25 * self.quartic * x**4)),
                            -self.k * x - self.quartic * x**3, None, 0.0)


POLICY = {"force_budget": 0.08, "timestep_fs": 0.1, "check_probability": 1.0,
          "check_seed": 2, "time_cap_fs": 2.0}


def world(tmp_path, *, update_n=2):
    run_dir = tmp_path
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    log = EventLog(run_dir)
    model = TrainableHarmonic()
    engine = Reference()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=update_n,
                                                 guard_size=1))
    runner = EnergeticRunner(atoms, model, engine, store, "run",
                             event_log=log, on_label=updater,
                             run_dir=run_dir, checkpoint_interval_steps=100,
                             **POLICY)
    return runner, model, engine, updater


def resume_world(tmp_path, *, update_n=2):
    model = TrainableHarmonic()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=update_n,
                                                 guard_size=1))
    runner = EnergeticRunner.resume(tmp_path, model, Reference(),
                                    updater=updater, event_log_force=False,
                                    checkpoint_interval_steps=100)
    return runner, model, updater


def events(run_dir):
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


def _artifact_path(run_dir, model_id):
    return run_dir / "models" / model_id.replace("/", "_") / "state.json"


def test_f04_tampered_artifact_rejected_clean_artifact_loads(tmp_path):
    runner, _, _, _ = world(tmp_path / "artifact")
    runner.run(3)  # labels accumulate; one published update
    runner.close()
    updates = [event for event in events(tmp_path / "artifact")
               if event["type"] == "model_update"]
    assert updates, "expected a published update"
    update = updates[0]
    assert update.get("artifact_digest")
    artifact_path = _artifact_path(tmp_path / "artifact", update["model_id"])
    record = json.loads(artifact_path.read_text())
    assert artifact_digest(record) == update["artifact_digest"]

    # Tamper with the artifact's payload: the recorded digest no longer
    # matches the file, and resume must refuse rather than load it.
    record["updater_state"]["surrogate"]["k"] = 9.9
    artifact_path.write_text(json.dumps(record))
    with pytest.raises(ResumeError, match="digest|tampered|verification"):
        resume_world(tmp_path / "artifact")

    # The untouched sibling run resumes and loads the genuine state.
    clean, clean_model, _, clean_updater = world(tmp_path / "clean")
    clean.run(3)
    clean.close()
    resumed, resumed_model, resumed_updater = resume_world(tmp_path / "clean")
    assert resumed_model.k == clean_model.k
    assert resumed_updater.n_updates == clean_updater.n_updates
    resumed.close()


def test_f04_parent_chain_mismatch_rejected(tmp_path):
    runner, _, _, _ = world(tmp_path / "parent")
    runner.run(3)
    runner.close()
    updates = [event for event in events(tmp_path / "parent")
               if event["type"] == "model_update"]
    assert updates
    artifact_path = _artifact_path(tmp_path / "parent",
                                   updates[0]["model_id"])
    record = json.loads(artifact_path.read_text())
    record["parent_model_id"] = "forged-parent#g0"
    artifact_path.write_text(json.dumps(record))
    # The digest bound into the commit mismatches, and the parent chain
    # check fails too — either way resume is refused.
    with pytest.raises(ResumeError):
        resume_world(tmp_path / "parent")


def _plain_config(tmp_path, *, steps=8, k=1.0, timestep=0.5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = HARMONIC_CONFIG
    for section in ("[reference]", "[policy]", "[verification]"):
        start = text.index(section)
        following = text.index("\n[", start + 1)
        text = text[:start] + text[following + 1:]
    text = text.replace('mode = "adaptive"', 'mode = "surrogate"')
    text = text.replace("steps = 20", f"steps = {steps}")
    text = text.replace("timestep_fs = 0.5", f"timestep_fs = {timestep}")
    text = text.replace("k = 1.0", f"k = {k}")
    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def test_plain_surrogate_resume_checks_model_identity_and_timestep(tmp_path):
    from pyraimd2.config import load_config

    config = load_config(_plain_config(tmp_path / "run"))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory

    def edit_resolved(**changes):
        resolved = json.loads((run_dir / "resolved_config.json").read_text())
        resolved["surrogate"]["options"].update(
            {key: value for key, value in changes.items()
             if key in resolved["surrogate"]["options"]})
        if "timestep_fs" in changes:
            resolved["dynamics"]["timestep_fs"] = changes["timestep_fs"]
        (run_dir / "resolved_config.json").write_text(json.dumps(resolved))

    # Physical setting change inside the run record: resume must refuse.
    edit_resolved(k=2.0)
    with pytest.raises(WorkflowError, match="surrogate identity"):
        resume_workflow(run_dir, 2, verbose=False, handle_sigint=False)

    edit_resolved(k=1.0, timestep_fs=0.25)
    with pytest.raises(WorkflowError, match="timestep_fs"):
        resume_workflow(run_dir, 2, verbose=False, handle_sigint=False)

    # Same model and settings: resume continues normally.
    edit_resolved(timestep_fs=0.5)
    result = resume_workflow(run_dir, 2, verbose=False, handle_sigint=False)
    assert result.steps_completed == 10
