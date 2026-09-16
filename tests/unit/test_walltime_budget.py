"""Plain-MD walltime budget (dynamics.max_wall_hours): the driver stops at
the last complete boundary instead of being killed mid-SCF, and a resume
in the next process finishes the run.

The clock is fake: ``pyraimd2.workflows.md.time.perf_counter`` is patched
to a controllable value and the harmonic reference's compute() advances
it by a fixed per-step cost, so the budget arithmetic is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyraimd2.backends.harmonic import HarmonicReference
from pyraimd2.config import ConfigError, load_config, load_resolved_config
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE

STEP_COST_S = 7000.0


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _patch_clock(monkeypatch, step_cost_s: float = STEP_COST_S) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr("pyraimd2.workflows.md.time.perf_counter",
                        clock.perf_counter)
    original = HarmonicReference.compute

    def compute(self, atoms):
        clock.advance(step_cost_s)
        return original(self, atoms)

    monkeypatch.setattr(HarmonicReference, "compute", compute)
    return clock


def _write_config(tmp_path: Path, *, steps: int = 5,
                  checkpoint_interval: int = 100,
                  extra_dynamics: str = "") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = HARMONIC_CONFIG
    for section in ("[surrogate]", "[policy]", "[verification]"):
        start = text.index(section)
        following = text.index("\n[", start + 1)
        text = text[:start] + text[following + 1:]
    text = text.replace('mode = "adaptive"', 'mode = "reference"')
    text = text.replace("steps = 20", f"steps = {steps}")
    text = text.replace("interval_steps = 5",
                        f"interval_steps = {checkpoint_interval}")
    text = text.replace("velocity_seed = 7", "velocity_seed = 7" + extra_dynamics)
    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in
            (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]


def test_budget_sufficient_runs_to_completion(tmp_path, monkeypatch):
    _patch_clock(monkeypatch)
    config = load_config(_write_config(tmp_path / "enough", steps=3,
                                       extra_dynamics="\nmax_wall_hours = 24.0"))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3
    assert result.stopped_early is False
    assert not [e for e in _events(config.run.directory)
                if e.get("type") == "run_end"]


def test_budget_stops_at_the_boundary_and_resume_completes(tmp_path, monkeypatch):
    _patch_clock(monkeypatch)
    # 7.2 h budget, 7000 s steps: steps 1-3 fit the reserve, step 4 does
    # not (remaining 4920 s < 10800 s); the checkpoint at step 3 is written
    # although the interval (100) never fires.
    config = load_config(_write_config(tmp_path / "budget",
                                       extra_dynamics="\nmax_wall_hours = 7.2"))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    assert result.steps_completed == 3
    assert result.stopped_early is True

    checkpoint = CheckpointManager(run_dir).read_latest_valid()
    assert checkpoint is not None
    assert checkpoint.state["nsteps"] == 3

    ends = [e for e in _events(run_dir) if e.get("type") == "run_end"]
    assert len(ends) == 1
    assert ends[0]["status"] == "stopped"
    assert "walltime" in ends[0]["reason"]
    assert ends[0]["step"] == 3
    assert ends[0]["remaining_s"] == pytest.approx(4920.0)
    assert ends[0]["reserve_s"] == pytest.approx(10800.0)
    assert ends[0]["last_step_wall_s"] == pytest.approx(STEP_COST_S)

    # the budget round-trips through resolved_config.json into the resume
    assert load_resolved_config(run_dir).dynamics.max_wall_hours == 7.2

    # the next process re-arms the budget: the remaining steps complete
    resumed = resume_workflow(run_dir, 2, verbose=False, handle_sigint=False)
    assert resumed.steps_completed == 5
    assert resumed.stopped_early is False
    assert len([e for e in _events(run_dir) if e.get("type") == "run_end"]) == 1


def test_budget_without_a_measured_step_reserves_the_scf_ceiling(tmp_path,
                                                                 monkeypatch):
    _patch_clock(monkeypatch)
    # 10000 s budget < the 3 h first-step reserve: no step starts, the
    # initial evaluation is already committed and checkpointed at step 0.
    config = load_config(_write_config(
        tmp_path / "tight",
        extra_dynamics=f"\nmax_wall_hours = {10000.0 / 3600.0}"))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 0
    assert result.stopped_early is True
    checkpoint = CheckpointManager(config.run.directory).read_latest_valid()
    assert checkpoint is not None
    assert checkpoint.state["nsteps"] == 0
    ends = [e for e in _events(config.run.directory)
            if e.get("type") == "run_end"]
    assert len(ends) == 1
    assert ends[0]["step"] == 0
    assert ends[0]["last_step_wall_s"] is None
    assert ends[0]["reserve_s"] == pytest.approx(10800.0)


def test_max_wall_hours_validation(tmp_path):
    for i, bad in enumerate(("0.0", "-1.0", '"soon"')):
        path = _write_config(tmp_path / f"bad-{i}",
                             extra_dynamics=f"\nmax_wall_hours = {bad}")
        with pytest.raises(ConfigError, match="max_wall_hours"):
            load_config(path)
    config = load_config(_write_config(tmp_path / "ok",
                                       extra_dynamics="\nmax_wall_hours = 1.5"))
    assert config.dynamics.max_wall_hours == 1.5
    assert load_config(_write_config(tmp_path / "off")).dynamics.max_wall_hours is None
