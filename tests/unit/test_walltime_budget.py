"""Plain-MD walltime budget (dynamics.max_wall_hours): the driver stops at
the last complete boundary instead of being killed mid-SCF, and a resume
in the next process finishes the run.

The clock is fake: ``pyraimd2.workflows.md.time.perf_counter`` is patched
to a controllable value and the harmonic reference's compute() advances
it by a fixed per-step cost, so the budget arithmetic is exact.

Semantics under test (the 0.7.4 adaptation): the budget anchors at the
workflow entry — backend construction and the initial evaluation count
against it; the reserve for starting one more step is 1.5x the last
measured step wall plus a 300 s I/O margin (the initial evaluation is
the first measurement; a resumed process with no measurement yet follows
the backend's declared timeout x retries when available, else the
margin); a running step is never interrupted; a resumed process re-arms
the budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyraimd2.backends.harmonic import HarmonicReference
from pyraimd2.config import ConfigError, load_config, load_resolved_config
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.store import Store
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.setup import WorkflowError
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE

STEP_COST_S = 7000.0


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _patch_clock(monkeypatch, costs: list[float] | None = None,
                 step_cost_s: float = STEP_COST_S) -> _FakeClock:
    """Patch the driver clock; the harmonic compute advances it by the next
    entry of ``costs`` (evaluations 0..N in order) or the fixed cost."""
    clock = _FakeClock()
    monkeypatch.setattr("pyraimd2.workflows.md.time.perf_counter",
                        clock.perf_counter)
    original = HarmonicReference.compute
    table = list(costs) if costs is not None else None
    calls = {"n": 0}

    def compute(self, atoms):
        cost = table[min(calls["n"], len(table) - 1)] if table else step_cost_s
        calls["n"] += 1
        clock.advance(cost)
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


def _rows(run_dir, run_id="t"):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def test_budget_sufficient_runs_to_completion(tmp_path, monkeypatch):
    _patch_clock(monkeypatch)
    config = load_config(_write_config(tmp_path / "enough", steps=3,
                                       extra_dynamics="\nmax_wall_hours = 24.0"))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3
    assert result.stopped_early is False
    assert not [e for e in _events(config.run.directory)
                if e.get("type") == "run_end"]


def test_budget_stops_at_the_boundary_and_resume_completes(tmp_path,
                                                           monkeypatch):
    _patch_clock(monkeypatch)
    # 7.2 h budget anchored at the workflow entry; 7000 s evaluations.  The
    # initial evaluation (7000 s) counts: the reserve for one more step is
    # 1.5 x 7000 + 300 = 10800 s, so step 3 does not start (remaining
    # 4920 s); the checkpoint at step 2 is written although the interval
    # (100) never fires.
    config = load_config(_write_config(tmp_path / "budget",
                                       extra_dynamics="\nmax_wall_hours = 7.2"))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    assert result.steps_completed == 2
    assert result.stopped_early is True

    checkpoint = CheckpointManager(run_dir).read_latest_valid()
    assert checkpoint is not None
    assert checkpoint.state["nsteps"] == 2

    ends = [e for e in _events(run_dir) if e.get("type") == "run_end"]
    assert len(ends) == 1
    assert ends[0]["status"] == "stopped"
    assert "walltime" in ends[0]["reason"]
    assert ends[0]["step"] == 2
    assert ends[0]["remaining_s"] == pytest.approx(4920.0)
    assert ends[0]["reserve_s"] == pytest.approx(10800.0)
    assert ends[0]["last_step_wall_s"] == pytest.approx(STEP_COST_S)

    # the budget round-trips through resolved_config.json into the resume
    assert load_resolved_config(run_dir).dynamics.max_wall_hours == 7.2

    # the next process re-arms the budget: the remaining steps complete
    resumed = resume_workflow(run_dir, 3, verbose=False, handle_sigint=False)
    assert resumed.steps_completed == 5
    assert resumed.stopped_early is False
    assert len([e for e in _events(run_dir) if e.get("type") == "run_end"]) == 1


def test_budget_without_a_measured_step_checkpoints_step_zero(tmp_path,
                                                              monkeypatch):
    _patch_clock(monkeypatch)
    # 10000 s budget: the initial evaluation costs 7000 s and counts, so
    # the first step's reserve (10800 s) exceeds the remainder — no MD
    # step starts; the initial boundary is committed AND checkpointed.
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
    assert ends[0]["last_step_wall_s"] == pytest.approx(STEP_COST_S)
    assert ends[0]["reserve_s"] == pytest.approx(10800.0)
    # and the zero-step boundary resumes
    resumed = resume_workflow(config.run.directory, 1, verbose=False,
                              handle_sigint=False)
    assert resumed.steps_completed == 1


def test_cheap_steps_progress_under_a_modest_budget(tmp_path, monkeypatch):
    _patch_clock(monkeypatch, step_cost_s=60.0)
    config = load_config(_write_config(
        tmp_path / "cheap", steps=6, extra_dynamics="\nmax_wall_hours = 0.5"))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 6
    assert result.stopped_early is False


def test_a_sudden_slow_step_is_never_interrupted(tmp_path, monkeypatch):
    # costs: eval 0 and steps 1-3 at 100 s, step 4 suddenly 5000 s.  The
    # estimate never predicts it (1.5 x 100 + 300 = 450 s reserve), the
    # slow step runs to completion intact, and the NEXT boundary check
    # stops the run: the budget is a soft between-step scheduler, never a
    # mid-step kill.
    _patch_clock(monkeypatch,
                 costs=[100.0, 100.0, 100.0, 100.0, 5000.0, 100.0])
    config = load_config(_write_config(tmp_path / "sudden", steps=6,
                                       extra_dynamics="\nmax_wall_hours = 1.5"))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    assert result.steps_completed == 4
    assert result.stopped_early is True
    ends = [e for e in _events(run_dir) if e.get("type") == "run_end"]
    assert len(ends) == 1
    assert ends[0]["step"] == 4
    assert ends[0]["last_step_wall_s"] == pytest.approx(5000.0)
    assert ends[0]["reserve_s"] == pytest.approx(7800.0)
    # the slow step's boundary is intact and the resume completes the run
    resumed = resume_workflow(run_dir, 2, verbose=False, handle_sigint=False)
    assert resumed.steps_completed == 6


def test_stopped_then_resumed_matches_continuous(tmp_path, monkeypatch):
    """Committed forces/momenta survive the budgeted stop: the stopped +
    resumed run reproduces the continuous one row by row."""
    _patch_clock(monkeypatch)
    continuous = load_config(_write_config(tmp_path / "continuous", steps=5))
    run_workflow(continuous, verbose=False, handle_sigint=False)
    stopped = load_config(_write_config(tmp_path / "stopped", steps=3,
                                        extra_dynamics="\nmax_wall_hours = 7.2"))
    result = run_workflow(stopped, verbose=False, handle_sigint=False)
    assert result.steps_completed == 2 and result.stopped_early
    resumed = resume_workflow(stopped.run.directory, 3, verbose=False,
                              handle_sigint=False)
    assert resumed.steps_completed == 5
    rows_a = _rows(continuous.run.directory, run_id="harmonic-demo")
    rows_b = _rows(stopped.run.directory, run_id="harmonic-demo")
    assert len(rows_a) == len(rows_b) > 0
    for a, b in zip(rows_a, rows_b):
        np.testing.assert_allclose(a.toatoms().positions,
                                   b.toatoms().positions, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.toatoms().get_momenta(),
                                   b.toatoms().get_momenta(), rtol=0,
                                   atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(a.data["driving"]["forces"], dtype=float),
            np.asarray(b.data["driving"]["forces"], dtype=float),
            rtol=0, atol=1e-12)


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


def test_max_wall_hours_refused_outside_plain_md(tmp_path):
    # adaptive mode: the energetic runner does not honor the budget —
    # refused at parse time, never silently ignored
    text = HARMONIC_CONFIG.replace("velocity_seed = 7",
                                   "velocity_seed = 7\nmax_wall_hours = 1.0")
    root = tmp_path / "adaptive"
    root.mkdir()
    (root / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    (root / "run.toml").write_text(text)
    with pytest.raises(ConfigError, match="max_wall_hours"):
        load_config(root / "run.toml")
    # a serial-recipe stage (controller-materialized input): refused at run
    stage = load_config(_write_config(
        tmp_path / "stage", extra_dynamics="\nmax_wall_hours = 1.0"))
    path = stage.source_path
    path.write_text(path.read_text().replace('file = "structure.extxyz"',
                                             'file = "run/initial.traj"'))
    config = load_config(path)
    with pytest.raises(WorkflowError, match="max_wall_hours"):
        run_workflow(config, verbose=False, handle_sigint=False)


def test_step_reserve_before_any_measurement():
    """Unmeasured first step (a resumed process): the reserve follows the
    backend's declared timeout x (retries + 1) when available, else just
    the I/O margin — never a hardcoded SCF ceiling."""
    from types import SimpleNamespace

    from pyraimd2.workflows.md import _PlainDriver

    driver = object.__new__(_PlainDriver)
    driver.backend = SimpleNamespace(
        config=SimpleNamespace(timeout_s=100.0, max_retries=2))
    assert driver._step_reserve(None) == pytest.approx(100.0 * 3 + 300.0)
    driver.backend = SimpleNamespace(config=SimpleNamespace(timeout_s=None))
    assert driver._step_reserve(None) == pytest.approx(300.0)
    driver.backend = SimpleNamespace(config=None)  # no declared config at all
    assert driver._step_reserve(None) == pytest.approx(300.0)
    # once measured, the estimate follows the measurement
    assert driver._step_reserve(7000.0) == pytest.approx(10800.0)
