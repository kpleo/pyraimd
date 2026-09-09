"""Review A1-A4 regression: the shared commit-reading interface for export
and inspect (INDEPENDENT_REVIEW_040_20260909 — initial-frame half-kick,
relax filtered out, orphan rows exported, tail-frame observables)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import units
from test_review_c1 import fail_commit_of, make_world, resume
from test_review_r1 import world

from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.workflows.export import export_run, frames_from_store


def test_a1_initial_frame_keeps_its_original_momenta(tmp_path):
    run_dir = tmp_path / "a1"
    runner, _, _, _ = world(run_dir)
    initial_momenta = runner.atoms.get_momenta().copy()
    runner.run(0)
    runner.run(2)
    runner.close()

    store = Store(run_dir / "trajectory.db")
    frames = frames_from_store(
        store, "run", force_source="driving",
        complete_steps={0, 1},
        committed=list(store.iter_committed(
            _events_list(run_dir), "run")))
    by_step = {f.info["step_id"]: f for f in frames}
    initial = by_step[-1]
    # The initial evaluation stores the complete initial momenta: no half
    # kick may be added (the A1 defect moved them by 0.5*dt*F).
    np.testing.assert_allclose(initial.get_momenta(), initial_momenta,
                               rtol=0, atol=1e-15)
    assert initial.info["momenta_source"] == "initial_evaluation_record"
    assert initial.info["integration_phase"] == "initial_evaluation"
    # A real mid-step row is still reconstructed to the complete step.
    row0 = store._row_at_step("run", 0)
    driving = np.asarray(row0.data["driving"]["forces"], dtype=float)
    expected = (row0.toatoms().get_momenta()
                + 0.5 * 0.1 * units.fs * driving)
    np.testing.assert_allclose(by_step[0].get_momenta(), expected,
                               rtol=0, atol=1e-12)
    assert by_step[0].info["momenta_source"] == "complete_step_reconstructed"


def _events_list(run_dir):
    import json

    path = run_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a2_relax_and_singlepoint_export_their_committed_rows(tmp_path):
    from test_review_r4 import _write_config

    from pyraimd2.config import load_config
    from pyraimd2.workflows import run_workflow

    relax = load_config(_write_config(tmp_path / "relax", kind="relax",
                                      mode="reference",
                                      positions=((1.3, 0.9, 0.9),
                                                 (0.92, 0.9, 0.9))))
    result = run_workflow(relax, verbose=False)
    report = export_run(result.run_dir)
    n_rows = len(list(Store(result.run_dir / "trajectory.db")
                     ._db.select(run_id="r4-demo")))
    assert report["frames"] >= 2
    assert report["frames"] <= n_rows

    sp = load_config(_write_config(tmp_path / "sp", kind="singlepoint",
                                   mode="reference"))
    result = run_workflow(sp, verbose=False)
    assert export_run(result.run_dir)["frames"] == 1


def test_a2_plain_md_keeps_complete_step_filtering(tmp_path):
    from test_review_r4 import _write_config

    from pyraimd2.config import load_config
    from pyraimd2.workflows import run_workflow

    config = load_config(_write_config(tmp_path / "md", mode="reference",
                                       steps=3, momenta=[[0.05, 0.0, 0.0]]))
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert export_run(result.run_dir)["frames"] == 4  # initial + 3 steps


def test_a3_export_never_includes_orphan_rows(tmp_path):
    run_dir = tmp_path / "a3"
    runner, _engine = make_world(run_dir)
    runner.run(0)
    fail_commit_of(runner, 1)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(1)  # orphan row written, commit lost
    runner.close()
    resumed = resume(run_dir)
    resumed.run(1)  # committed re-execution at the same step
    resumed.close()

    store = Store(run_dir / "trajectory.db")
    events = _events_list(run_dir)
    frames = frames_from_store(
        store, "run", force_source="driving",
        committed=list(store.iter_committed(events, "run")))
    assert [f.info["evaluation_id"] for f in frames] == [0, 1]
    expected_rows = [int(store.committed_row(events, "run", eid).id)
                     for eid in (0, 1)]
    assert [f.info["row_id"] for f in frames] == expected_rows
    # The orphan stays in the store as an audit record, not a frame.
    assert len(list(store._db.select(run_id="run"))) == 3


def test_a4_inspect_separates_complete_state_from_unfinished_tail(tmp_path):
    run_dir = tmp_path / "a4"
    runner, _, _, _ = world(run_dir)
    runner.run(0)
    calc = runner.calc
    from pyraimd2.runtime.events import STEP_COMPLETED

    original = calc._emit_once

    def fail_boundary(key, event_type, **payload):
        if event_type == STEP_COMPLETED:
            raise RuntimeError("injected crash before the last half-kick")
        return original(key, event_type, **payload)

    calc._emit_once = fail_boundary
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(1)  # evaluation committed; the step boundary never was
    runner.close()

    info = inspect_run(run_dir)
    assert info["n_complete_steps"] == 0
    assert info["physical_time_fs"] == 0.0  # the initial evaluation at t=0
    assert info["trajectory"]["last_step"] == -1
    tail = info["last_evaluation"]
    assert tail["step_id"] == 0
    assert tail["phase"] == "md_step"
    assert tail["physical_time_fs"] == pytest.approx(0.1)
    assert tail["complete"] is False
    assert tail["energy_eV"] is not None
    # Complete-state temperature comes from the initial frame only.
    store = Store(run_dir / "trajectory.db")
    initial = store.complete_step_frame(store._row_at_step("run", -1), 0.1)
    from pyraimd2.runtime.inspect import _temperature_K

    assert info["trajectory"]["last_temperature_K"] == pytest.approx(
        _temperature_K(initial))
