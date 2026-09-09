"""Real-MACE tensor persistence integration (review section 6 item 5).

One genuine update on the Al(111) slab: CommitteeSurrogate (MACE-MPA-0
medium, CPU) + GuardedUpdater inside the energetic runner with the QE
metallic reference.  Process 1 runs the initial evaluation plus two MD
steps (n_label=2 -> one accepted update publishes a tensor-artifact-v1
model artifact), records the boundary prediction, and closes.  Process 2
(a fresh python) resumes from the checkpoint/artifact and re-predicts the
same boundary: member weights, consumption counts and predictions must
match exactly.  Run: process1 | process2 in two invocations.

SCF budget (process 1): anchor + 2 calibration probes + 2 step checks = 5
(one re-anchor with probes would add 3; the run stops at budget either
way).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

work = Path(sys.argv[1]).resolve()
phase = sys.argv[2]
run_dir = work / "runs" / "al-committee"


def make_backends(event_log=None):
    from pyraimd2.engines.qe_engine import QeConfig, QeEngine
    from pyraimd2.loop import GuardedUpdater, UpdatePolicy
    from pyraimd2.surrogate.committee import CommitteeSurrogate

    config = QeConfig(
        pseudo_dir="/tmp/ps",
        pw_cmd=("mpirun", "-np", "28", "pw.x"),
        xc="pbe", dispersion="grimme-d3",
        ecutwfc=50.0, ecutrho=400.0, kpts=(4, 4, 1),
        metallic=True, smearing="mv", degauss=0.02, conv_thr=1e-8,
        pseudos={"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
    )
    engine = QeEngine(config, run_root=run_dir / "calculations",
                      event_log=event_log)
    surrogate = CommitteeSurrogate(
        model=str(work / "mace-mpa-0-medium.model"),
        n_members=2, epochs=2, seed=0, lr=1e-3)
    updater = GuardedUpdater(surrogate, UpdatePolicy(n_label=2, guard_size=1))
    return engine, surrogate, updater


if phase == "process1":
    from pyraimd2.loop import EnergeticRunner
    from pyraimd2.runtime.events import EventLog
    from pyraimd2.store import Store

    atoms = ase_read(work / "structure.extxyz")
    from ase.md.velocitydistribution import thermalize_momenta

    thermalize_momenta(atoms, 300.0, rng=np.random.default_rng(7))
    run_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(run_dir)
    engine, surrogate, updater = make_backends(event_log=log)
    runner = EnergeticRunner(
        atoms, surrogate, engine, Store(run_dir / "trajectory.db"),
        "al-committee", on_label=updater, run_dir=run_dir, event_log=log,
        checkpoint_interval_steps=2, force_budget=0.15,
        probe_steps=(0.02, 0.04), numerical_floor=0.001, time_cap_fs=2.0,
        transverse_cap=0.1, timestep_fs=1.0,
        check_probability=1.0, check_seed=19)
    runner.run(0)
    runner.run(2)
    from pyraimd2.store import Store as _Store

    def _listify(value):
        detach = getattr(value, "detach", None)
        if callable(detach):
            return np.asarray(detach().cpu().numpy()).tolist()
        if isinstance(value, dict):
            return {k: _listify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_listify(v) for v in value]
        return value

    store = _Store(run_dir / "trajectory.db")
    row = store.committed_row(log, "al-committee", runner.calc.n_evaluations - 1)
    boundary = row.toatoms()
    prediction = surrogate.predict(boundary)
    evidence = {
        "n_evaluations": runner.calc.n_evaluations,
        "n_reference": runner.calc.n_reference,
        "n_consumed": updater.n_consumed,
        "n_updates": updater.n_updates,
        "n_rejected": updater.n_rejected,
        "model_generation": runner.calc._model_generation,
        "model_id": runner.calc.model_id,
        "boundary_step": int(row.key_value_pairs["step"]),
        "prediction_energy_eV": float(prediction.energy),
        "prediction_forces": np.asarray(prediction.forces).tolist(),
        "member_state_dicts": _listify(
            surrogate.state_dict()["member_state_dicts"]),
        "energy_shifts": surrogate.state_dict()["energy_shifts"],
    }
    runner.close()
    (work / "committee_process1.json").write_text(json.dumps(evidence))
    print(json.dumps({k: v for k, v in evidence.items()
                      if k not in ("prediction_forces", "member_state_dicts")},
                     indent=2))
elif phase == "process2":
    from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
    from pyraimd2.runtime.models import ModelRegistry
    from pyraimd2.store import Store as _Store

    previous = json.loads((work / "committee_process1.json").read_text())
    engine, surrogate, updater = make_backends()
    runner = EnergeticRunner.resume(
        run_dir, surrogate, engine, updater=updater,
        checkpoint_interval_steps=2, event_log_force=True)
    store = _Store(run_dir / "trajectory.db")
    rows = list(store._db.select(run_id="al-committee"))
    row = sorted(rows, key=lambda r: int(r.key_value_pairs["step"]))[-1]
    boundary = row.toatoms()
    prediction = surrogate.predict(boundary)
    restored = surrogate.state_dict()
    registry = ModelRegistry(run_dir)
    artifact_raw = registry.read(previous["model_id"], resolve=False)
    artifact_resolved = registry.read(previous["model_id"], resolve=True)
    guard_payload = updater.state_dict()["guard"][0]
    checks = {
        "artifact_exists": artifact_raw is not None,
        "artifact_arrays_npz": (run_dir / "models"
                                / previous["model_id"].replace("/", "_")
                                / "state_arrays.npz").is_file(),
        "artifact_has_placeholders": "__ndarray__" in json.dumps(
            artifact_raw["updater_state"]) if artifact_raw else False,
        "n_consumed": updater.n_consumed,
        "n_updates": updater.n_updates,
        "guard_payload_has_cell": bool(np.asarray(
            guard_payload["cell"]).any()),
        "guard_payload_pbc": guard_payload["pbc"] == [True, True, False],
        "prediction_energy_match": float(np.isclose(
            prediction.energy, previous["prediction_energy_eV"],
            rtol=0, atol=1e-10)),
        "prediction_forces_max_abs_diff": float(np.abs(
            np.asarray(prediction.forces)
            - np.asarray(previous["prediction_forces"])).max()),
        "member_weights_match": all(
            (np.asarray(a).size == 0 and np.array_equal(a, b))
            or float(np.abs(np.asarray(a) - np.asarray(b)).max()) == 0.0
            for a, b in zip(
                [t for member in restored["member_state_dicts"]
                 for t in member.values()],
                [t for member in previous["member_state_dicts"]
                 for t in member.values()])),
        "energy_shifts_match": bool(np.allclose(
            restored["energy_shifts"], previous["energy_shifts"],
            rtol=0, atol=0.0)),
    }
    boolean_checks = {k: bool(v) for k, v in checks.items()
                      if k not in ("n_consumed", "n_updates",
                                   "prediction_forces_max_abs_diff",
                                   "passed")}
    checks["passed"] = all(boolean_checks.values()) \
        and checks["prediction_forces_max_abs_diff"] <= 1e-10 \
        and checks["n_consumed"] == previous["n_consumed"] \
        and checks["n_updates"] == previous["n_updates"]
    runner.close()
    (work / "committee_process2.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    sys.exit(0 if checks["passed"] else 1)
else:
    raise SystemExit(f"unknown phase {phase!r}")
