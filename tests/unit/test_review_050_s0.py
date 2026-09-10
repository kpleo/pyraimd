"""S0 regression for the 0.5-M3A handoff.

S0a: every Store closes on its owner's controlled return — workflow-created
stores close in the workflow, driver-owned stores close with the driver,
and a caller-passed store outlives the runner (the three paths the
store-lifecycle probes reproduced).

S0b: step-summary formats are versioned.  New records carry an explicit
``digest_format`` marker and bind the complete boundary (bath stream
included); records of the previous development batch (one complete-boundary
JSON digest, no marker) and 0.4.x records (no step summary) are verified by
their own recognizable semantics on resume — verified, never rewritten.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_review_r4 import _write_config

from pyraimd2.config import load_config
from pyraimd2.loop.integrators import DIGEST_FORMAT
from pyraimd2.runtime.events import STEP_COMPLETED, EventLog
from pyraimd2.store import Store
from pyraimd2.workflows import md as md_module
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows import setup as wf_setup


def _tracking_stores(monkey):
    stores = []
    for module in (md_module, wf_setup):
        real = module.Store

        def track(*args, _real=real, **kwargs):
            store = _real(*args, **kwargs)
            stores.append(store)
            return store

        monkey.setattr(module, "Store", track)
    return stores


class _BrokenOutputs:
    def __init__(self, *args, **kwargs):
        raise OSError("injected outputs failure")


def test_plain_setup_failure_closes_driver_store(tmp_path):
    config = load_config(_write_config(tmp_path / "s0a", mode="reference",
                                       steps=2))
    monkey = pytest.MonkeyPatch()
    stores = _tracking_stores(monkey)
    monkey.setattr(md_module, "RunOutputs", _BrokenOutputs)
    monkey.setattr(wf_setup, "RunOutputs", _BrokenOutputs)
    try:
        with pytest.raises(OSError, match="injected outputs"):
            run_workflow(config, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()
    assert stores
    for store in stores:
        assert store._db is None  # closed before the error returned


def test_resume_setup_failure_closes_driver_store(tmp_path):
    config = load_config(_write_config(tmp_path / "s0a-r", mode="reference",
                                       steps=4))
    run_workflow(config, verbose=False, handle_sigint=False)
    monkey = pytest.MonkeyPatch()
    stores = _tracking_stores(monkey)
    monkey.setattr(md_module, "RunOutputs", _BrokenOutputs)
    monkey.setattr(wf_setup, "RunOutputs", _BrokenOutputs)
    try:
        with pytest.raises(OSError, match="injected outputs"):
            resume_workflow(config.run.directory, 2, verbose=False,
                            handle_sigint=False, force_unlock=True)
    finally:
        monkey.undo()
    assert stores
    for store in stores:
        assert store._db is None


def test_adaptive_workflow_closes_its_own_store(tmp_path):
    from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE

    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    text = HARMONIC_CONFIG.replace("steps = 20", "steps = 3")
    (tmp_path / "run.toml").write_text(text)
    config = load_config(tmp_path / "run.toml")
    monkey = pytest.MonkeyPatch()
    stores = _tracking_stores(monkey)
    try:
        run_workflow(config, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()
    assert stores
    for store in stores:
        assert store._db is None


def test_caller_passed_store_outlives_the_runner(tmp_path):
    from ase import Atoms
    from test_review_r1 import Model, Reference, _direction

    from pyraimd2.loop import EnergeticRunner
    from pyraimd2.runtime.events import EventLog
    from pyraimd2.store import Store

    run_dir = tmp_path / "lib"
    run_dir.mkdir()
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(run_dir / "trajectory.db")
    runner = EnergeticRunner(
        atoms, Model(), Reference(), store, "run", run_dir=run_dir,
        event_log=EventLog(run_dir), checkpoint_interval_steps=100,
        direction=_direction, force_budget=0.08, timestep_fs=0.1,
        time_cap_fs=2.0, check_probability=0.0, check_seed=2)
    runner.run(1)
    runner.close()
    assert store._db is not None  # the caller's store stays usable
    assert len(list(store._db.select(run_id="run"))) == 2
    store.close()


# --- S0b: step-summary format versioning -------------------------------------


def _read_events(run_dir):
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()]


def _step_events(run_dir):
    return [e for e in _read_events(run_dir) if e.get("type") == STEP_COMPLETED]


def _rewrite_events(run_dir, events):
    path = run_dir / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def _as_legacy_record(event, store, events, run_id):
    """Rewrite a current step event into the previous batch's format: the
    single complete-boundary JSON digest (no bath stream), no marker."""
    evaluation_id = int(event["step_id"]) + 1
    row = store.committed_row(events, run_id, evaluation_id)
    commit = next(
        e for e in events if e.get("type") == "evaluation_committed"
        and int((e.get("context") or {})["evaluation_id"]) == evaluation_id)
    boundary = md_module._boundary_from_event(row, event, commit)
    rewritten = {key: value for key, value in event.items()
                 if key not in ("digest_format", "boundary_digest",
                                "thermostat_rng")}
    rewritten["state_digest"] = boundary.legacy_digest()
    return rewritten


def test_step_records_carry_boundary_v2_marker_and_verify(tmp_path):
    config = load_config(_write_config(tmp_path / "s0b", mode="reference",
                                       steps=5, checkpoint_interval=4))
    run_workflow(config, verbose=False, handle_sigint=False)
    steps = _step_events(config.run.directory)
    assert len(steps) == 5
    for event in steps:
        assert event["digest_format"] == DIGEST_FORMAT
        assert event["boundary_digest"]
        assert "thermostat_rng" not in event  # NVE has no bath stream
    # Resume verifies the boundary record through the v2 branch.
    result = resume_workflow(config.run.directory, 2, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 7
    for event in _step_events(config.run.directory):
        assert event["digest_format"] == DIGEST_FORMAT


def test_legacy_step_records_verify_under_legacy_semantics(tmp_path):
    config = load_config(_write_config(tmp_path / "s0b-legacy",
                                       mode="reference", steps=5,
                                       checkpoint_interval=4))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    events = _read_events(run_dir)
    store = Store(run_dir / "trajectory.db")
    rewritten = []
    for event in events:
        if event.get("type") == STEP_COMPLETED:
            legacy = _as_legacy_record(event, store, events, config.run.id)
            # the formats genuinely differ — the branch is not a no-op
            assert legacy["state_digest"] != event["state_digest"]
            rewritten.append(legacy)
        else:
            rewritten.append(event)
    store.close()
    _rewrite_events(run_dir, rewritten)

    result = resume_workflow(run_dir, 2, verbose=False, handle_sigint=False)
    assert result.steps_completed == 7
    steps = _step_events(run_dir)
    # The old records are verified, never rewritten to the new format …
    for event in steps[:5]:
        assert "digest_format" not in event
        assert "boundary_digest" not in event
    # … while the continuation steps are written in the current format.
    for event in steps[5:]:
        assert event["digest_format"] == DIGEST_FORMAT
        assert event["boundary_digest"]


def test_tampered_legacy_digest_is_refused(tmp_path):
    config = load_config(_write_config(tmp_path / "s0b-leg-tamper",
                                       mode="reference", steps=5,
                                       checkpoint_interval=4))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    events = _read_events(run_dir)
    store = Store(run_dir / "trajectory.db")
    rewritten = [_as_legacy_record(e, store, events, config.run.id)
                 if e.get("type") == STEP_COMPLETED else e for e in events]
    store.close()
    boundary = rewritten[[i for i, e in enumerate(rewritten)
                          if e.get("type") == STEP_COMPLETED][-1]]
    digest = boundary["state_digest"]
    boundary["state_digest"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    _rewrite_events(run_dir, rewritten)
    with pytest.raises(Exception, match="no known unmarked format"):
        resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)


def test_tampered_boundary_v2_digest_is_refused(tmp_path):
    config = load_config(_write_config(tmp_path / "s0b-v2-tamper",
                                       mode="reference", steps=5,
                                       checkpoint_interval=4))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    events = _read_events(run_dir)
    boundary = [e for e in events if e.get("type") == STEP_COMPLETED][-1]
    digest = boundary["boundary_digest"]
    boundary["boundary_digest"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    _rewrite_events(run_dir, events)
    with pytest.raises(Exception, match="boundary digest"):
        resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)


def test_healed_step_record_carries_the_complete_boundary(tmp_path):
    # Crash between the evaluation commit and the step commit of step 5
    # (the checkpoint at step 4 already exists): the heal window.  Resume
    # must re-emit that step with exactly the normal-commit fields (marker,
    # array digest, boundary digest, bath stream) and reproduce the
    # continuous run bit-for-bit.
    from test_nvt import rows, write_nvt

    config = load_config(write_nvt(tmp_path / "s0b-heal", steps=10,
                                   checkpoint_interval=4))
    run_id = config.run.id
    monkey = pytest.MonkeyPatch()
    real_append_once = EventLog.append_once

    def crash_before_step_commit(self, key, event_type, payload):
        if key == f"step:{run_id}:5":
            raise RuntimeError("injected crash before the step commit")
        return real_append_once(self, key, event_type, payload)

    monkey.setattr(EventLog, "append_once", crash_before_step_commit)
    try:
        with pytest.raises(RuntimeError, match="injected crash"):
            run_workflow(config, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()

    result = resume_workflow(config.run.directory, 4, verbose=False,
                             handle_sigint=False, force_unlock=True)
    assert result.steps_completed == 10
    healed = next(e for e in _step_events(config.run.directory)
                  if int(e["step_id"]) == 5)
    assert healed["digest_format"] == DIGEST_FORMAT
    assert healed["state_digest"]
    assert healed["boundary_digest"]
    assert healed["thermostat_rng"]  # the bath stream is part of the record
    # A second resume verifies the healed record through the v2 branch.
    result = resume_workflow(config.run.directory, 1, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 11

    control = load_config(write_nvt(tmp_path / "s0b-heal-control", steps=11,
                                    checkpoint_interval=4))
    run_workflow(control, verbose=False, handle_sigint=False)
    healed_rows = rows(config.run.directory)
    control_rows = rows(control.run.directory)
    assert len(healed_rows) == len(control_rows) == 12
    for healed_row, control_row in zip(healed_rows, control_rows):
        np.testing.assert_allclose(healed_row.toatoms().positions,
                                   control_row.toatoms().positions,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(healed_row.toatoms().get_momenta(),
                                   control_row.toatoms().get_momenta(),
                                   rtol=0, atol=1e-12)


# --- S1: the 182cc8d unmarked dual-digest records ------------------------------
#
# The 182cc8d plain driver wrote step records with an array state_digest and
# a JSON boundary_digest that does NOT cover the bath stream, and no format
# marker; old healed records of that lineage may carry only the array digest.
# The fixtures below reproduce a real 182cc8d wheel's production bit-for-bit
# (same template config and seeds) and pin its actual recorded digests as
# numbers — no private paths, artifacts or packages are included.


def _write_182_config(tmp_path, *, ensemble):
    from pyraimd2.workflows.templates import write_template

    path = write_template("harmonic-nvt", tmp_path)
    text = path.read_text().replace("steps = 20", "steps = 5")
    if ensemble == "nve":
        text = text.replace(
            'ensemble = "nvt"\nintegrator = "langevin"\n',
            'ensemble = "nve"\n').replace(
            'friction_per_fs = 0.01            # bath coupling (required for NVT)\n'
            'thermostat_seed = 123             # new-run thermostat stream (resume restores it)\n',
            '')
    path.write_text(text)
    return load_config(path)


def _as_182cc8d_record(event, store, events, run_id):
    """Rewrite a current step event into 182cc8d's semantics: array
    state_digest (unchanged), JSON boundary_digest without the bath stream,
    no format marker; the NVT thermostat_rng field is kept, as then."""
    rewritten = {key: value for key, value in event.items()
                 if key != "digest_format"}
    evaluation_id = int(event["step_id"]) + 1
    row = store.committed_row(events, run_id, evaluation_id)
    commit = next(
        e for e in events if e.get("type") == "evaluation_committed"
        and int((e.get("context") or {})["evaluation_id"]) == evaluation_id)
    boundary = md_module._boundary_from_event(row, event, commit)
    rewritten["boundary_digest"] = boundary.legacy_digest()
    return rewritten


# The digests the real 182cc8d wheel recorded for these two 5-step runs.
_182_NVE_DIGESTS = ("9da390babde44b3a5610136b", "0f790128f0b5dcff3c943e15")
_182_NVT_DIGESTS = ("b1937e74ebd0eb91ab98c16b", "3858508f7212259f77203f35")


@pytest.mark.parametrize("ensemble,expected",
                         [("nve", _182_NVE_DIGESTS), ("nvt", _182_NVT_DIGESTS)])
def test_182cc8d_dual_format_records_verify_on_resume(tmp_path, ensemble,
                                                      expected):
    config = _write_182_config(tmp_path / ensemble, ensemble=ensemble)
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    events = _read_events(run_dir)
    store = Store(run_dir / "trajectory.db")
    rewritten = [_as_182cc8d_record(e, store, events, config.run.id)
                 if e.get("type") == STEP_COMPLETED else e for e in events]
    store.close()
    # the fixture reproduces the real 182cc8d production bit-for-bit
    boundary = [e for e in rewritten if e.get("type") == STEP_COMPLETED][-1]
    assert (boundary["state_digest"], boundary["boundary_digest"]) == expected
    _rewrite_events(run_dir, rewritten)

    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 6
    steps = _step_events(run_dir)
    # old records verified, never rewritten; the continuation is v2
    assert all("digest_format" not in e for e in steps[:5])
    assert steps[5]["digest_format"] == DIGEST_FORMAT
    # and the continuation is the same trajectory a v2 run produces
    control = _write_182_config(tmp_path / f"{ensemble}-control",
                                ensemble=ensemble)
    text_path = control.source_path
    text_path.write_text(text_path.read_text().replace(
        "steps = 5", "steps = 6"))
    control = load_config(text_path)
    run_workflow(control, verbose=False, handle_sigint=False)
    from test_nvt import rows as _nvt_rows

    for row_a, row_b in zip(_nvt_rows(run_dir, config.run.id),
                            _nvt_rows(control.run.directory, config.run.id)):
        np.testing.assert_array_equal(row_a.toatoms().positions,
                                      row_b.toatoms().positions)
        np.testing.assert_array_equal(row_a.toatoms().get_momenta(),
                                      row_b.toatoms().get_momenta())


def test_182cc8d_dual_format_tamper_is_refused(tmp_path):
    for field in ("state_digest", "boundary_digest"):
        config = _write_182_config(tmp_path / f"nvt-{field}", ensemble="nvt")
        run_workflow(config, verbose=False, handle_sigint=False)
        run_dir = config.run.directory
        events = _read_events(run_dir)
        store = Store(run_dir / "trajectory.db")
        rewritten = [_as_182cc8d_record(e, store, events, config.run.id)
                     if e.get("type") == STEP_COMPLETED else e
                     for e in events]
        store.close()
        boundary = [e for e in rewritten
                    if e.get("type") == STEP_COMPLETED][-1]
        value = boundary[field]
        boundary[field] = ("0" if value[0] != "0" else "1") + value[1:]
        _rewrite_events(run_dir, rewritten)
        with pytest.raises(Exception, match="does not match the committed row"):
            resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)


def test_array_only_unmarked_record_verifies_and_tamper_refused(tmp_path):
    # Old healed records may carry only the array digest (plus the NVT
    # thermostat field): verified by array semantics, not silently skipped.
    config = _write_182_config(tmp_path / "nvt-heal", ensemble="nvt")
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    events = _read_events(run_dir)
    rewritten = []
    for event in events:
        if event.get("type") == STEP_COMPLETED:
            event = {key: value for key, value in event.items()
                     if key not in ("digest_format", "boundary_digest")}
        rewritten.append(event)
    _rewrite_events(run_dir, rewritten)
    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 6

    config = _write_182_config(tmp_path / "nvt-heal-tamper", ensemble="nvt")
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    events = _read_events(run_dir)
    rewritten = []
    for event in events:
        if event.get("type") == STEP_COMPLETED:
            event = {key: value for key, value in event.items()
                     if key not in ("digest_format", "boundary_digest")}
        rewritten.append(event)
    boundary = [e for e in rewritten if e.get("type") == STEP_COMPLETED][-1]
    value = boundary["state_digest"]
    boundary["state_digest"] = ("0" if value[0] != "0" else "1") + value[1:]
    _rewrite_events(run_dir, rewritten)
    with pytest.raises(Exception, match="no known unmarked format"):
        resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
