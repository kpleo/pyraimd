"""R2/R3 regression: resume consumes committed state — no implicit backend
recomputation, no orphan propagation, digest and bath-stream binding
verified on use (0.5-batch1-fixes §R2/R3 minimal acceptance group)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from test_nvt import events, rows, write_nvt

from pyraimd2.config import load_config
from pyraimd2.loop.integrators import state_digest
from pyraimd2.store import Store
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.export import export_run


class CountingReference:
    """Harmonic reference counting every real compute call in a file, with
    a tiny per-call drift so a re-execution is numerically distinguishable."""

    name = "counting-reference"

    def __init__(self, counter_path, fingerprint):
        self.counter_path = counter_path
        self._fingerprint = fingerprint

    @property
    def fingerprint(self):
        return self._fingerprint

    @property
    def n_calls(self):
        return int(self.counter_path.read_text()) \
            if self.counter_path.exists() else 0

    def compute(self, atoms):
        n = self.n_calls + 1
        self.counter_path.write_text(str(n))
        x = atoms.positions
        drift = 1e-6 * n
        from pyraimd2.engines.base import EngineResult

        return EngineResult(float(np.sum(0.5 * 1.0 * x**2)) + drift,
                            -1.0 * x - drift, None, 0.0)


def _harmonic_fingerprint():
    from pyraimd2.backends import backend_factory

    return backend_factory("harmonic-reference")(k=1.0, r0=0.9).fingerprint


def _run_with_backend(tmp_path, engine, *, steps, resume=None):
    from pyraimd2.workflows import md as md_module

    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: engine)
    try:
        config = load_config(write_nvt(tmp_path, steps=steps))
        run_workflow(config, verbose=False, handle_sigint=False)
        if resume is not None:
            resume_workflow(config.run.directory, resume, verbose=False,
                            handle_sigint=False, force_unlock=True)
        return config
    finally:
        monkey.undo()


def test_r2_continuous_vs_split_resume_calls_and_ledger_match(tmp_path):
    continuous = _run_with_backend(
        tmp_path / "cont", CountingReference(tmp_path / "cont.calls", _harmonic_fingerprint()),
        steps=20)
    cont_calls = int((tmp_path / "cont.calls").read_text())
    cont_attempts = sum(1 for e in events(continuous.run.directory)
                        if e["type"] == "attempt")
    assert cont_calls == 21 == cont_attempts  # initial + 20 steps

    stopped = _run_with_backend(
        tmp_path / "split", CountingReference(tmp_path / "split.calls", _harmonic_fingerprint()),
        steps=7, resume=13)
    split_calls = int((tmp_path / "split.calls").read_text())
    split_attempts = sum(1 for e in events(stopped.run.directory)
                         if e["type"] == "attempt")
    # No completed step is recomputed: calls match the continuous run and
    # the ledger agrees (0.5-batch1 R1/R2 defect was 24 calls / 21 records).
    assert split_calls == cont_calls
    assert split_attempts == cont_attempts
    for a, b in zip(rows(continuous.run.directory),
                    rows(stopped.run.directory)):
        np.testing.assert_allclose(a.toatoms().positions,
                                   b.toatoms().positions, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.toatoms().get_momenta(),
                                   b.toatoms().get_momenta(), rtol=0,
                                   atol=1e-12)


_CHILD = """
import os, sys
import numpy as np
from pyraimd2.config import load_config
from pyraimd2.runtime.events import EventLog
from pyraimd2.workflows import run_workflow

kill_key = os.environ["KILL_KEY"]
count_file = os.environ.get("COUNT_FILE")
original = EventLog.append_once

def patched(self, key, event_type, payload):
    if key == kill_key:
        os._exit(73)
    return original(self, key, event_type, payload)

EventLog.append_once = patched

if count_file:
    from pathlib import Path
    from pyraimd2.engines.base import EngineResult
    import pyraimd2.workflows.md as md_module

    class CountingReference:
        name = "counting-reference"
        @property
        def fingerprint(self):
            from pyraimd2.backends import backend_factory
            return backend_factory("harmonic-reference")(k=1.0, r0=0.9).fingerprint
        @property
        def n_calls(self):
            path = Path(count_file)
            return int(path.read_text()) if path.exists() else 0
        def compute(self, atoms):
            n = self.n_calls + 1
            Path(count_file).write_text(str(n))
            x = atoms.positions
            drift = 1e-6 * n
            return EngineResult(float(np.sum(0.5 * x**2)) + drift,
                                -1.0 * x - drift, None, 0.0)

    md_module._plain_backend = lambda config, run_dir, **kwargs: CountingReference()

run_workflow(load_config(sys.argv[1]), verbose=False, handle_sigint=False)
"""


def _crash_window(tmp_path, kill_key, *, steps=8, count_file=None):
    config = load_config(write_nvt(tmp_path, steps=steps))
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(_CHILD))
    env = dict(os.environ, KILL_KEY=kill_key,
               PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    if count_file is not None:
        env["COUNT_FILE"] = str(count_file)
    result = subprocess.run([sys.executable, str(child),
                             str(config.source_path)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 73, result.stderr[-500:]
    return config


def test_r3_committed_evaluation_uncommitted_step_heals_without_recompute(
        tmp_path):
    # Hard exit after the evaluation commit of evaluation 5, before the
    # step commit (true subprocess, os._exit — not force_unlock).
    crashed = _crash_window(tmp_path / "w", "step:nvt-demo:4",
                            count_file=tmp_path / "shared.calls")
    # The resume continues the same counter file: the per-call drift makes
    # any recomputed label numerically distinguishable.
    engine = CountingReference(tmp_path / "shared.calls",
                               _harmonic_fingerprint())
    # resume the crashed directory with the counting backend
    from pyraimd2.workflows import md as md_module

    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: engine)
    try:
        resumed = resume_workflow(crashed.run.directory, 3, verbose=False,
                                  handle_sigint=False, force_unlock=True)
    finally:
        monkey.undo()
    assert resumed.steps_completed == 8
    # The healing consumed the committed evaluation 5: zero recomputation
    # (the child's 6 calls + the resume's 3 = the control's 9).
    assert engine.n_calls == 9
    store = Store(crashed.run.directory / "trajectory.db")
    evs = events(crashed.run.directory)
    # No orphan row was produced; the step-4 event now exists and its
    # digest recomputes from the authoritative row.
    step4 = next(e for e in evs if e["type"] == "step_completed"
                 and e["step_id"] == 4)
    row5 = store.committed_row(evs, "nvt-demo", 5)
    assert state_digest(row5.toatoms().positions,
                        row5.toatoms().get_momenta()) == step4["state_digest"]
    n_rows = len(list(store._db.select(run_id="nvt-demo")))
    assert n_rows == resumed.steps_completed + 1
    report = export_run(crashed.run.directory)
    from ase.io import read as ase_read

    frames = ase_read(report["output"], index=":")
    assert [int(f.info["evaluation_id"]) for f in frames] == list(range(9))
    control = _run_with_backend(tmp_path / "ctl",
                                CountingReference(tmp_path / "ctl.calls", _harmonic_fingerprint()),
                                steps=8)
    for a, b in zip(rows(crashed.run.directory),
                    rows(control.run.directory)):
        np.testing.assert_allclose(a.toatoms().positions,
                                   b.toatoms().positions, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.toatoms().get_momenta(),
                                   b.toatoms().get_momenta(), rtol=0,
                                   atol=1e-12)


def test_r3_row_written_commit_lost_retries_once_and_bills_it(tmp_path):
    # Hard exit at the evaluation commit of evaluation 5: the row exists,
    # the commit does not — the retry is necessary and billed.
    crashed = _crash_window(tmp_path / "w", "evaluation:nvt-demo:5")
    engine = CountingReference(tmp_path / "retry.calls", _harmonic_fingerprint())
    from pyraimd2.workflows import md as md_module

    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: engine)
    try:
        resumed = resume_workflow(crashed.run.directory, 3, verbose=False,
                                  handle_sigint=False, force_unlock=True)
    finally:
        monkey.undo()
    assert resumed.steps_completed == 7  # 4 committed steps + 3
    # The re-execution of the uncommitted evaluation is the first resumed
    # step itself — one necessary retry, billed like any step.
    assert engine.n_calls == 3
    # The pre-crash evaluation's cost stays visible in the ledger.
    pre_crash_tasks = [e for e in events(crashed.run.directory)
                       if e["type"] == "task"
                       and e.get("evaluation_id") == 5]
    assert len(pre_crash_tasks) == 2  # crashed attempt + the billed retry
    store = Store(crashed.run.directory / "trajectory.db")
    evs = events(crashed.run.directory)
    row5 = store.committed_row(evs, "nvt-demo", 5)
    step4 = next(e for e in evs if e["type"] == "step_completed"
                 and e["step_id"] == 4)
    # The retried evaluation's committed row is what the step digest names,
    # what propagation used, and what export selects — one state throughout.
    assert state_digest(row5.toatoms().positions,
                        row5.toatoms().get_momenta()) == step4["state_digest"]
    report = export_run(crashed.run.directory)
    from ase.io import read as ase_read

    frames = ase_read(report["output"], index=":")
    assert [int(f.info["evaluation_id"]) for f in frames] == list(range(8))
    frame5 = frames[5]
    np.testing.assert_allclose(frame5.get_momenta(),
                               row5.toatoms().get_momenta(), rtol=0, atol=1e-8)


def test_r3_sentinel_backend_resume_reads_never_call_compute(tmp_path):
    config = load_config(write_nvt(tmp_path / "s", steps=4))
    run_workflow(config, verbose=False, handle_sigint=False)
    from pyraimd2.backends import backend_factory

    original_fingerprint = backend_factory("harmonic-reference")(
        k=1.0, r0=0.9).fingerprint

    class Sentinel:
        fingerprint = original_fingerprint

        def compute(self, atoms):
            raise AssertionError("backend called during resume read phase")

    from pyraimd2.workflows import md as md_module

    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: Sentinel())
    try:
        with pytest.raises(AssertionError, match="read phase"):
            resume_workflow(config.run.directory, 2, verbose=False,
                            handle_sigint=False, force_unlock=True)
    finally:
        monkey.undo()
    # The read phase completed and emitted RESUMED without any backend call:
    # between the initial run's summary and RESUMED there is no task/attempt.
    log = events(config.run.directory)
    resumed_seq = next(e["seq"] for e in log if e["type"] == "resumed")
    summary_seq = max(e["seq"] for e in log
                      if e["type"] == "run_summary" and e["seq"] < resumed_seq)
    work_between = [e for e in log
                    if e["type"] in ("task", "attempt")
                    and summary_seq < e["seq"] < resumed_seq]
    assert work_between == []
    # And a working backend resumes the same directory normally.
    resumed = resume_workflow(config.run.directory, 2, verbose=False,
                              handle_sigint=False, force_unlock=True)
    assert resumed.steps_completed == 6
