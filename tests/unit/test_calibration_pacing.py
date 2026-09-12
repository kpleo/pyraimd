"""0.6 calibration-pacing prototype acceptance (opt-in, default off).

1. Consecutive accepts under one favourable anchor: pacing on but never
   triggering is bitwise identical to pacing off (the anchor is never
   cleared by an accept) — covers the original simulator counterexample.
2. A sterile regime turns fertile: an engineered position-dependent
   surrogate spends its early phase out of the forecast envelope, pacing
   defers after the sterile streak, the bounded retry fires, and accepts
   resume — the run never parks in reference mode permanently.
3. Error growth during a wait: the retained anchor's correction against
   the step's own reference label exceeds the budget and cancels the wait
   immediately (recomputed against the RETAINED anchor, never read from a
   stale record).
4. Si-recipe analytic stub regression: pacing reproduces the recorded
   routing exactly — trajectories bitwise identical to the off control,
   reference tasks 49 (K=3/W0=1) and 45 (K=3/W0=2), both accepts kept.
5. Resume: a hard exit right after a defer decision, and one mid-probes
   of a forced-retry calibration — each resumed run matches its
   continuous control on q/p, rule state, decision order and RNG; probes
   persisted before the crash are never re-executed.

Analytic harmonic stand-ins only — zero real DFT budget.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from test_adaptive_nvt import _atoms, _spec
from test_nvt import events

from pyraimd2.backends.harmonic import HarmonicReference, HarmonicSurrogate
from pyraimd2.config import ConfigError, load_config
from pyraimd2.loop import EnergeticRunner
from pyraimd2.loop.integrators import IntegratorSpec
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogateCapabilities, SurrogatePrediction

K_REF, R0 = 1.0, 0.9

_PACING = {"failure_streak_limit": 3, "wait_initial": 1, "wait_max": 8}


@pytest.fixture
def _si_stub_backends(monkeypatch):
    """The Si recipe's QE/MACE stand-ins through the real factory path
    (analytic, hermetic) — same injection as test_si_recipe."""
    from test_si_recipe import _fake_mace_factory, _fake_qe_factory

    from pyraimd2.backends import registry

    monkeypatch.setitem(
        registry._BUILTINS, "qe",
        (registry._BUILTINS["qe"][0], "fake_qe_module", "create_qe"))
    monkeypatch.setitem(
        registry._BUILTINS, "mace",
        (registry._BUILTINS["mace"][0], "fake_mace_module", "create_mace"))
    import types

    qe_module = types.ModuleType("fake_qe_module")
    qe_module.create_qe = _fake_qe_factory
    mace_module = types.ModuleType("fake_mace_module")
    mace_module.create_mace = _fake_mace_factory
    sys.modules["fake_qe_module"] = qe_module
    sys.modules["fake_mace_module"] = mace_module
    yield


def _nve_spec():
    return IntegratorSpec(algorithm="velocity_verlet", ensemble="nve",
                          timestep_fs=0.5, temperature_K=300.0,
                          friction_per_fs=None, thermostat_seed=None)


def _runner(run_dir, surrogate, *, pacing=None, steps=None, atoms=None,
            check_probability=0.5, spec=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return EnergeticRunner(
        _atoms() if atoms is None else atoms, surrogate,
        HarmonicReference(k=K_REF, r0=R0),
        Store(run_dir / "trajectory.db"), "run",
        run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=10, force_budget=0.5, timestep_fs=0.5,
        time_cap_fs=1000.0, transverse_cap=1.0,
        check_probability=check_probability, check_seed=7,
        integrator_spec=_nve_spec() if spec is None else spec,
        calibration_pacing=pacing)


def _decisions(run_dir):
    return [(e["evaluation_id"], e["decision"], e["reason"])
            for e in events(run_dir) if e["type"] == "pacing_decision"]


def _routes(run_dir):
    return [e["route"] for e in events(run_dir)
            if e["type"] == "evaluation_committed"]


def _summary(run_dir):
    return [e for e in events(run_dir) if e["type"] == "run_summary"][-1]


# --- configuration surface ----------------------------------------------------


def _write_config(tmp_path, pacing_block=""):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "structure.extxyz").write_text(
        "2\nH2\nH 0.85 0.9 0.9\nH 0.95 0.9 0.9\n")
    (tmp_path / "run.toml").write_text(f"""\
schema_version = 1
[run]
id = "pacing-cfg"
directory = "."
seed = 42
[task]
kind = "md"
mode = "adaptive"
[structure]
file = "structure.extxyz"
[reference]
backend = "harmonic-reference"
k = 1.0
r0 = 0.9
[surrogate]
backend = "harmonic-surrogate"
k = 1.01
r0 = 0.9
[dynamics]
ensemble = "nve"
timestep_fs = 0.5
steps = 4
temperature_K = 300.0
velocity_seed = 7
[policy]
force_budget_eV_A = 0.5
time_cap_fs = 1000.0
transverse_cap = 1.0
[verification]
probability = 0.5
seed = 7
[checkpoint]
interval_steps = 10
[output]
trajectory_interval_steps = 1
summary_interval_steps = 10
{pacing_block}""")
    return tmp_path / "run.toml"


def test_config_defaults_off_and_validates(tmp_path):
    config = load_config(_write_config(tmp_path))
    assert config.policy.calibration_pacing is None

    enabled = """\

[policy.calibration_pacing]
enabled = true
"""
    config = load_config(_write_config(tmp_path / "on", enabled))
    pacing = config.policy.calibration_pacing
    assert pacing.enabled is True
    assert (pacing.failure_streak_limit, pacing.wait_initial,
            pacing.wait_max) == (3, 1, 8)

    # the resolved identity carries the section only when configured
    assert "calibration_pacing" in config.resolved_dict()["policy"]
    assert "calibration_pacing" not in \
        load_config(_write_config(tmp_path / "off")).resolved_dict()["policy"]


def test_config_rejects_bad_pacing_fields(tmp_path):
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(_write_config(tmp_path / "a", """\

[policy.calibration_pacing]
enabled = true
streak = 3
"""))
    with pytest.raises(ConfigError, match=">="):
        load_config(_write_config(tmp_path / "b", """\

[policy.calibration_pacing]
enabled = true
wait_initial = 4
wait_max = 2
"""))
    with pytest.raises(ConfigError, match="pretend parameters"):
        load_config(_write_config(tmp_path / "c", """\

[policy.calibration_pacing]
enabled = false
wait_max = 4
"""))


def test_pacing_refuses_unsupported_combinations_before_compute(tmp_path):
    def direction_callback(atoms):
        return np.ones((len(atoms), 3))

    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    # the guards fire at construction — before any backend compute
    with pytest.raises(ValueError, match="does not compose with on_label"):
        EnergeticRunner(
            _atoms(), HarmonicSurrogate(k=1.01, r0=R0),
            HarmonicReference(k=K_REF, r0=R0),
            Store(tmp_path / "b" / "trajectory.db"), "run",
            run_dir=tmp_path / "b", event_log=EventLog(tmp_path / "b"),
            force_budget=0.5, timestep_fs=0.5, integrator_spec=_nve_spec(),
            on_label=lambda label: False,
            calibration_pacing=dict(_PACING))
    with pytest.raises(ValueError, match="single-direction"):
        EnergeticRunner(
            _atoms(), HarmonicSurrogate(k=1.01, r0=R0),
            HarmonicReference(k=K_REF, r0=R0),
            Store(tmp_path / "c" / "trajectory.db"), "run",
            run_dir=tmp_path / "c", event_log=EventLog(tmp_path / "c"),
            force_budget=0.5, timestep_fs=0.5, integrator_spec=_nve_spec(),
            direction=direction_callback,
            calibration_pacing=dict(_PACING))


# --- 1. consecutive accepts under one favourable anchor -------------------------


def test_consecutive_accepts_identical_with_pacing_on_and_off(tmp_path):
    """A favourable anchor keeps accepting across consecutive steps; pacing
    on (never triggering) and off are bitwise identical, and no refusal
    ever clears the anchor."""
    surrogate = lambda: HarmonicSurrogate(k=1.01, r0=R0)
    runner_off = _runner(tmp_path / "off", surrogate(), pacing=None,
                         spec=_spec())
    summary_off = runner_off.run(8)
    runner_on = _runner(tmp_path / "on", surrogate(), pacing=dict(_PACING),
                        spec=_spec())
    summary_on = runner_on.run(8)
    assert summary_off.n_accepted >= 2  # consecutive accepts happened
    assert summary_on.n_accepted == summary_off.n_accepted
    # only the initial mandatory fallbacks were recorded; nothing deferred
    assert _decisions(tmp_path / "on") == [
        (0, "calibrate", "no_anchor"), (1, "calibrate", "no_anchor")]
    assert _summary(tmp_path / "on")["pacing"] == {
        "deferred": 0, "retried": 0, "safety_exits": 0}
    # the off run carries no pacing surface at all
    assert not [e for e in events(tmp_path / "off")
                if e["type"] == "pacing_decision"]
    assert _summary(tmp_path / "off").get("pacing") is None
    # identical driving forces and streams: bitwise-identical states
    for step, run_id in ((7, "run"),):
        rows_off = _rows(tmp_path / "off", run_id)
        rows_on = _rows(tmp_path / "on", run_id)
        assert len(rows_off) == len(rows_on)
        for row_off, row_on in zip(rows_off, rows_on):
            np.testing.assert_array_equal(row_off.toatoms().positions,
                                          row_on.toatoms().positions)
            np.testing.assert_array_equal(row_off.toatoms().get_momenta(),
                                          row_on.toatoms().get_momenta())


def _rows(run_dir, run_id):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


# --- 2/3. sterile regime turns fertile; safety exit -----------------------------

_BOUNDARY = 0.30   # |dr| below this (the fast central region) is bad
_FAR_ERR = 4.0     # 300% surrogate error in the fast central region
_NEAR_ERR = 1.01   # 1% error near the slow turning points
_AMPL = 0.70


class RegionSurrogate:
    """Fixed analytic predictor whose error is large in the fast central
    region and small near the turning points — a fixed physical
    construction, fixed seeds, no tuning per outcome."""

    name = "region-surrogate"
    fingerprint = "region-surrogate-v1"

    @property
    def capabilities(self):
        return SurrogateCapabilities()

    def predict(self, atoms):
        dr = atoms.positions - R0
        energy = 0.5 * K_REF * float((dr**2).sum())
        central = float(np.abs(dr).max()) < _BOUNDARY
        scale = _FAR_ERR if central else _NEAR_ERR
        return SurrogatePrediction(energy * scale, -K_REF * dr * scale, None,
                                   np.full(len(atoms), np.nan))


def _region_run(run_dir, *, pacing, budget=0.05, steps=90):
    positions = [[R0 - _AMPL, 0.9, 0.9], [R0 + _AMPL, 0.9, 0.9]]
    atoms = Atoms("H2", positions=positions, cell=[10.0] * 3, pbc=False)
    atoms.set_momenta([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = EnergeticRunner(
        atoms, RegionSurrogate(), HarmonicReference(k=K_REF, r0=R0),
        Store(run_dir / "trajectory.db"), "run",
        run_dir=run_dir, event_log=EventLog(run_dir),
        checkpoint_interval_steps=10, force_budget=budget, timestep_fs=0.5,
        time_cap_fs=1000.0, transverse_cap=1.0,
        check_probability=0.5, check_seed=7,
        integrator_spec=_nve_spec(), calibration_pacing=pacing)
    summary = runner.run(steps)
    runner.close()
    return summary


def test_sterile_regime_recovers_into_a_fertile_one(tmp_path):
    """Early centre crossings refuse (envelope over budget) and the pacing
    wait engages; the bounded retry recalibrates, and once the trajectory
    reaches a favourable region accepts resume in a sustained run — the
    controller never parks in reference-direct mode."""
    summary = _region_run(tmp_path / "on", pacing=dict(_PACING))
    decisions = _decisions(tmp_path / "on")
    defers = [d for d in decisions if d[1] == "defer"]
    retries = [d for d in decisions if d[2] == "forced_retry"]
    assert defers and retries
    first_defer = defers[0][0]
    retry_after = next(d for d in retries if d[0] > first_defer)
    routes = _routes(tmp_path / "on")
    # accepts resume after the bounded retry and become sustained
    later = routes[retry_after[0]:]
    assert "ml" in later
    longest_streak = max((len(list(group)) for key, group in
                          __import__("itertools").groupby(later)
                          if key == "ml"), default=0)
    assert longest_streak >= 20  # a sustained fertile phase, not a flicker
    assert summary.pacing["deferred"] == len(defers)
    assert summary.pacing["retried"] >= 1
    # the pacing-off control on the same construction calibrates at every
    # refusal — more probes, same physical governance
    summary_off = _region_run(tmp_path / "off", pacing=None)
    assert summary_off.n_probe > summary.n_probe
    assert summary_off.pacing is None


def test_error_growth_during_a_wait_cancels_it_with_retained_anchor(tmp_path):
    """During a wait the retained anchor's correction against the step's
    own reference label crosses the budget: the wait is cancelled at once
    (safety_exit) and the recomputation is against the RETAINED anchor —
    never a stale record's observed value."""
    _region_run(tmp_path / "on", pacing=dict(_PACING))
    decision_events = [e for e in events(tmp_path / "on")
                       if e["type"] == "pacing_decision"]
    exit_event = next(e for e in decision_events
                      if e["reason"] == "safety_exit")
    evaluation_id = int(exit_event["evaluation_id"])
    assert exit_event["state_before"]["wait"] > 0
    assert exit_event["state_after"]["wait"] == 0
    # the previous evaluation deferred and recorded its retained anchor
    with Store(tmp_path / "on" / "trajectory.db") as store:
        rows = {int(r.key_value_pairs["step"]): r
                for r in store._db.select(run_id="run")}
    previous = rows[evaluation_id - 2].data.get("metadata") or {}
    retained = previous.get("retained_anchor")
    assert retained is not None
    assert retained["open_prefix"] == [False] * len(retained["open_prefix"])
    row = rows[evaluation_id - 1]
    metadata = row.data.get("metadata") or {}
    observed = metadata["observed"]["max_force_error_eV_A"]
    assert observed > metadata["force_budget_eV_A"]
    # recompute the same error from the retained anchor's correction — the
    # live safety signal is exactly this recomputation
    residual = (np.asarray(row.data["surrogate"]["forces"], dtype=float)
                + np.asarray(retained["correction_eV_A"], dtype=float)
                - np.asarray(row.data["engine"]["forces"], dtype=float))
    recomputed = float(np.linalg.norm(residual, axis=1).max())
    assert recomputed == pytest.approx(observed, rel=1e-12)
    assert retained["segment_id"] == exit_event["segment"]


# --- 4. Si-recipe analytic stub regression --------------------------------------


def _si_stages_with_pacing(root, pacing):
    import test_si_recipe as si

    if pacing is not None:
        nvt = root / "nvt.toml"
        nvt.write_text(nvt.read_text() + f"""\
[policy.calibration_pacing]
enabled = true
failure_streak_limit = {pacing[0]}
wait_initial = {pacing[1]}
wait_max = {pacing[2]}
""")
    return si._stages(root)


def _si_reference_tasks(root):
    return [e for e in events(root / "nvt")
            if e["type"] == "task" and e.get("operation") == "reference"]


def test_si_stub_recipe_pacing_regression(tmp_path, _si_stub_backends):
    """The Si analytic stub (which reproduces the B record's routing) under
    pacing: trajectories stay bitwise identical to the off control, both
    accepts are kept, and the reference totals land on the conditional
    counts — now measured by real runs, not projected."""
    import test_si_recipe as si

    from pyraimd2.workflows.stages import load_completed_state, run_serial_recipe

    runs = {}
    for name, pacing in (("off", None), ("k3w1", (3, 1, 8)),
                         ("k3w2", (3, 2, 8))):
        root = si._write_si_recipe(tmp_path / name)
        manifest = run_serial_recipe(root, _si_stages_with_pacing(root, pacing),
                                     verbose=False)
        assert [s["status"] for s in manifest["stages"]] == ["done"] * 3
        runs[name] = root
    tasks_off = _si_reference_tasks(runs["off"])
    assert len(tasks_off) == 53  # the recorded B split, reproduced
    assert len(_si_reference_tasks(runs["k3w1"])) == 49
    assert len(_si_reference_tasks(runs["k3w2"])) == 45
    for name in ("k3w1", "k3w2"):
        decisions = _decisions(runs[name] / "nvt")
        assert (11, "defer", "sterile_streak") in decisions
        off_state = load_completed_state(runs["off"] / "nvt")
        on_state = load_completed_state(runs[name] / "nvt")
        np.testing.assert_array_equal(off_state.atoms.positions,
                                      on_state.atoms.positions)
        np.testing.assert_array_equal(off_state.atoms.get_momenta(),
                                      on_state.atoms.get_momenta())
        routes_off = _routes(runs["off"] / "nvt")
        assert _routes(runs[name] / "nvt") == routes_off  # accepts kept
    # K=3/W0=1 retries at eval 12; K=3/W0=2 defers it (end of record)
    assert (12, "calibrate", "forced_retry") in _decisions(runs["k3w1"] / "nvt")
    assert (12, "defer", "waiting") in _decisions(runs["k3w2"] / "nvt")
    # the ledger bills only what ran; nothing invented
    from pyraimd2.runtime.inspect import inspect_run

    info = inspect_run(runs["k3w2"] / "nvt", run_id="si-nvt")
    assert info["cost"]["reference"]["actual_executions"] == 45
    assert info["pacing"]["decisions"] == {"calibrate": 9, "defer": 2}


# --- 5. recovery windows --------------------------------------------------------


_PACING_RECIPE_CHILD = '''
import os
import sys
import types
from pyraimd2.backends import registry
from pyraimd2.runtime.events import EventLog
from test_calibration_pacing import _si_stages_with_pacing
from test_si_recipe import _fake_mace_factory, _fake_qe_factory, _write_si_recipe
from pyraimd2.workflows.stages import run_serial_recipe

# the analytic QE/MACE stand-ins through the real entry-point path (the
# parent's monkeypatch does not cross the process boundary)
registry._BUILTINS["qe"] = (registry._BUILTINS["qe"][0], "fake_qe_module", "create_qe")
registry._BUILTINS["mace"] = (registry._BUILTINS["mace"][0], "fake_mace_module", "create_mace")
qe_module = types.ModuleType("fake_qe_module")
qe_module.create_qe = _fake_qe_factory
mace_module = types.ModuleType("fake_mace_module")
mace_module.create_mace = _fake_mace_factory
sys.modules["fake_qe_module"] = qe_module
sys.modules["fake_mace_module"] = mace_module

kill_key = os.environ["KILL_KEY"]
original = EventLog.append_once


def patched(self, key, event_type, payload):
    result = original(self, key, event_type, payload)
    if key == kill_key:
        os._exit(73)
    return result


EventLog.append_once = patched
root = os.environ["ROOT"]
from pathlib import Path
run_serial_recipe(Path(root), _si_stages_with_pacing(Path(root), (3, 1, 8)),
                  verbose=False)
'''


def _pacing_crash_root(tmp_path, kill_key):
    import test_si_recipe as si

    crash_root = si._write_si_recipe(tmp_path / "crash")
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(_PACING_RECIPE_CHILD))
    env = dict(os.environ, ROOT=str(crash_root), KILL_KEY=kill_key,
               PYTHONPATH=os.pathsep.join(
                   [str(Path(__file__).parents[2] / "src"),
                    str(Path(__file__).parent)]))
    result = subprocess.run([sys.executable, str(child)],
                            env=env, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 73, result.stderr[-500:]
    return crash_root


def _continuous_pacing_control(tmp_path):
    import test_si_recipe as si

    from pyraimd2.workflows.stages import run_serial_recipe

    root = si._write_si_recipe(tmp_path / "control")
    run_serial_recipe(root, _si_stages_with_pacing(root, (3, 1, 8)),
                      verbose=False)
    return root


def _pacing_state(run_dir):
    with Store(run_dir / "trajectory.db") as store:
        row = max(store._db.select(run_id="si-nvt"),
                  key=lambda r: int(r.key_value_pairs["step"]))
    return (row.data.get("metadata") or {}).get("pacing")


def test_resume_after_a_defer_decision_hard_exit(tmp_path, _si_stub_backends):
    """Hard exit with the defer decision durable but the evaluation
    uncommitted: the resume replays the frozen decision verbatim — no
    re-decision, no double-advanced wait, identical trajectory."""
    import test_si_recipe as si

    from pyraimd2.workflows.stages import load_completed_state, run_serial_recipe

    control = _continuous_pacing_control(tmp_path)
    crash_root = _pacing_crash_root(tmp_path, "pacing:si-nvt:11")
    manifest = run_serial_recipe(crash_root, si._stages(crash_root),
                                 verbose=False, force_unlock=True)
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 3
    crash_ev = events(crash_root / "nvt")
    control_ev = events(control / "nvt")
    # the defer decision exists exactly once — replayed, never re-emitted
    assert [e["seq"] for e in crash_ev if e["type"] == "pacing_decision"
            and e["decision"] == "defer"] == \
        [e["seq"] for e in control_ev if e["type"] == "pacing_decision"
         and e["decision"] == "defer"]
    assert [(e["evaluation_id"], e["decision"], e["reason"]) for e in crash_ev
            if e["type"] == "pacing_decision"] == _decisions(control / "nvt")
    np.testing.assert_array_equal(
        load_completed_state(crash_root / "nvt").atoms.positions,
        load_completed_state(control / "nvt").atoms.positions)
    np.testing.assert_array_equal(
        load_completed_state(crash_root / "nve").atoms.get_momenta(),
        load_completed_state(control / "nve").atoms.get_momenta())
    assert _pacing_state(crash_root / "nvt") == _pacing_state(control / "nvt")
    # the crashed evaluation's reference label re-executed and is billed —
    # real re-runs are never free
    assert len(_si_reference_tasks(crash_root)) == \
        len(_si_reference_tasks(control)) + 1


def test_resume_mid_probes_of_a_forced_retry_calibration(tmp_path, _si_stub_backends):
    """Hard exit mid-probes of the forced-retry calibration: the replayed
    decision calibrates, the probes persisted before the crash are reused
    (never re-executed), and the trajectory matches the control."""
    import test_si_recipe as si

    from pyraimd2.workflows.stages import load_completed_state, run_serial_recipe

    control = _continuous_pacing_control(tmp_path)
    # the forced retry calibrates at eval 12; kill after its second probe
    crash_root = _pacing_crash_root(tmp_path, "probe:si-nvt:12:0:0:-1")
    manifest = run_serial_recipe(crash_root, si._stages(crash_root),
                                 verbose=False, force_unlock=True)
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 3
    crash_ev = events(crash_root / "nvt")
    # four distinct probes for eval 12 — persisted ones were reused, the
    # rest ran fresh exactly once
    probes = [e for e in crash_ev if e["type"] == "probe_completed"
              and e["evaluation_id"] == 12]
    assert len(probes) == 4
    probe_tasks = [e for e in crash_ev if e["type"] == "task"
                   and e.get("operation") == "reference"
                   and e.get("purpose") == "probe"
                   and e.get("evaluation_id") == 12]
    assert len(probe_tasks) == 4  # 2 before the crash + 2 after, not 4+4
    assert [(e["evaluation_id"], e["decision"], e["reason"]) for e in crash_ev
            if e["type"] == "pacing_decision"] == _decisions(control / "nvt")
    np.testing.assert_array_equal(
        load_completed_state(crash_root / "nvt").atoms.positions,
        load_completed_state(control / "nvt").atoms.positions)
    np.testing.assert_array_equal(
        load_completed_state(crash_root / "nvt").atoms.get_momenta(),
        load_completed_state(control / "nvt").atoms.get_momenta())
    assert _pacing_state(crash_root / "nvt") == _pacing_state(control / "nvt")
