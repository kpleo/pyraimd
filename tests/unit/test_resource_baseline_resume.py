"""Behavior checks for the resume-time resource-baseline gate (N002/R1).

A run whose valid checkpoint carries the file-resource association is
verified against the current baseline file before any backend factory,
evaluation or new step; the association is inherited from the checkpoint
only — never recomputed from the current side file, and a baseline file
never upgrades an old run.  Plain reference, plain surrogate and adaptive
modes share the one small gate.  Analytic file-backed test backends only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_file_resource_baseline import (
    _checkpoint_states,
    _engine_factory,
    _inject,
    _surrogate_factory,
    _write_run,
)

from pyraimd2.config import load_config
from pyraimd2.engines.ase_resources import file_resource_baseline_sha256
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.setup import WorkflowError


class FactoryCounter:
    """Counts backend factory calls and the calculators' compute calls."""

    def __init__(self):
        self.factory_calls = 0
        self.calculators = []

    def engine(self, **kwargs):
        self.factory_calls += 1
        adapter = _engine_factory(**kwargs)
        self.calculators.append(adapter.calculator)
        return adapter

    def surrogate(self, **kwargs):
        self.factory_calls += 1
        adapter = _surrogate_factory(**kwargs)
        self.calculators.append(adapter._engine.calculator)
        return adapter

    @property
    def compute_calls(self):
        return sum(calculator.calls for calculator in self.calculators)


def _records(run_dir):
    run_dir = Path(run_dir)
    return {
        "events": (run_dir / "events.jsonl").read_bytes()
        if (run_dir / "events.jsonl").is_file() else None,
        "db": (run_dir / "trajectory.db").read_bytes()
        if (run_dir / "trajectory.db").is_file() else None,
        "checkpoints": sorted(p.name for p in
                              (run_dir / "checkpoints").iterdir())
        if (run_dir / "checkpoints").is_dir() else None,
    }


def _fresh_run(tmp_path, monkeypatch, *, mode="reference", steps=2):
    counter = FactoryCounter()
    _inject(monkeypatch, "engine" if mode == "reference" else "surrogate",
            counter.engine if mode == "reference" else counter.surrogate)
    model = tmp_path / "inputs" / "model.dat"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("1.5\n")
    config_path = _write_run(tmp_path / "run", mode=mode, steps=steps,
                             checkpoint=1, options=f'model = "{model}"')
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    return counter, tmp_path / "run"


def test_in_place_baseline_resume_succeeds_all_modes(tmp_path,
                                                     monkeypatch):
    for mode in ("reference", "surrogate", "adaptive"):
        counter, run_dir = _fresh_run(tmp_path / mode, monkeypatch,
                                      mode=mode)
        digest = file_resource_baseline_sha256(run_dir)
        before = _records(run_dir)
        factory_before = counter.factory_calls
        result = resume_workflow(run_dir, 1, verbose=False,
                                 handle_sigint=False)
        assert result.steps_completed == 3
        assert counter.factory_calls == factory_before + 1  # the resume's
        states = _checkpoint_states(run_dir)
        assert all(state.get("file_resource_baseline_sha256") == digest
                   for state in states)
        assert _records(run_dir) != before  # the resume genuinely advanced


def _tamper_payloads(run_dir):
    """Tamper variants built from the run's actual baseline."""
    original = (Path(run_dir) / "file_resources.json").read_bytes()
    payload = json.loads(original)
    payload["run_id"] = "wrong-run-id"
    return [("modified", json.dumps(payload, indent=2).encode()),
            ("corrupt", b"not json {"),
            ("missing", None)]


def test_tampered_or_missing_baseline_refused_before_any_backend(
        tmp_path, monkeypatch):
    for mode in ("reference", "adaptive"):
        for index in range(3):
            counter, run_dir = _fresh_run(
                tmp_path / f"{mode}-{index}", monkeypatch, mode=mode)
            label, content = _tamper_payloads(run_dir)[index]
            baseline = run_dir / "file_resources.json"
            original = baseline.read_bytes()
            before = _records(run_dir)
            factory_before = counter.factory_calls
            compute_before = counter.compute_calls
            if content is None:
                baseline.unlink()
            else:
                baseline.write_bytes(content)
            try:
                with pytest.raises(WorkflowError,
                                   match="baseline"):
                    resume_workflow(run_dir, 1, verbose=False,
                                    handle_sigint=False)
                # the heal-only form is not a bypass either
                with pytest.raises(WorkflowError, match="baseline"):
                    resume_workflow(run_dir, 0, verbose=False,
                                    handle_sigint=False)
            finally:
                baseline.write_bytes(original)
            # zero new factory calls, zero new compute, zero record growth
            assert counter.factory_calls == factory_before
            assert counter.compute_calls == compute_before
            assert _records(run_dir) == before, f"{mode}/{label}"


def test_surrogate_mode_tampered_baseline_refused(tmp_path, monkeypatch):
    counter, run_dir = _fresh_run(tmp_path / "surrogate", monkeypatch,
                                  mode="surrogate")
    baseline = run_dir / "file_resources.json"
    payload = json.loads(baseline.read_bytes())
    payload["resources"]["surrogate.potential"]["sha256"] = "0" * 64
    before = _records(run_dir)
    factory_before = counter.factory_calls
    compute_before = counter.compute_calls
    baseline.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(WorkflowError, match="baseline"):
        resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert counter.factory_calls == factory_before
    assert counter.compute_calls == compute_before
    assert _records(run_dir) == before


def test_old_run_without_baseline_never_upgrades(tmp_path):
    """Old record: a no-baseline run resumes with the old behavior; a
    side file appearing later is ignored — never adopted, no new field."""
    config_path = _write_run(tmp_path / "run", backend="harmonic-reference",
                             steps=2, checkpoint=1,
                             options="k = 1.0\nr0 = 0.9")
    run_workflow(load_config(config_path), verbose=False,
                 handle_sigint=False)
    run_dir = tmp_path / "run"
    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3
    assert all("file_resource_baseline_sha256" not in state
               for state in _checkpoint_states(run_dir))
    # a planted side file is never adopted: the resume continues and the
    # new checkpoints still carry no association
    (run_dir / "file_resources.json").write_text(json.dumps({
        "schema": "file-resource-baseline-v1", "run_id": "file-res",
        "resources": {}}) + "\n")
    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 4
    assert all("file_resource_baseline_sha256" not in state
               for state in _checkpoint_states(run_dir))
