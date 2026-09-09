"""Small, offline reproductions for the 0.4.0rc1 independent review.

Run from the repository with: uv run --no-sync python <this file>
All simulations and fake executable outputs live in temporary directories.
No actual electronic-structure calculation, model download, or code edit.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import ase
import numpy as np
from ase import Atoms, units
from ase.io import read

from pyraimd2.backends.harmonic import HarmonicReference, HarmonicSurrogate
from pyraimd2.config import load_config
from pyraimd2.engines.qe_engine import QeConfig, QeEngine
from pyraimd2.loop import EnergeticRunner
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.runtime.events import EventLog
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.workflows import export_run, run_workflow
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE


def configuration_and_export(root: Path) -> dict:
    root.mkdir()
    (root / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    text = HARMONIC_CONFIG.replace("steps = 20", "steps = 3")
    text = text.replace("interval_steps = 5", "interval_steps = 1")
    text = text.replace('name = "energetic"',
                        'name = "energetic"\nforce_metric = "all_atoms_max_atom"')
    (root / "run.toml").write_text(text)
    config = load_config(root / "run.toml")
    run_workflow(config, verbose=False, handle_sigint=False)
    checkpoint = CheckpointManager(config.run.directory).read_latest_valid()
    frame = read(export_run(config.run.directory)["output"], index=-1)
    full_momenta = checkpoint.arrays["momenta"]
    correction = (0.5 * config.dynamics.timestep_fs * units.fs
                  * checkpoint.arrays["driving_forces"])
    complete_frame = frame.copy()
    complete_frame.set_momenta(full_momenta)
    return {
        "force_metric": {
            "requested": config.policy.force_metric,
            "executed": checkpoint.state["policy"]["force_metric"],
        },
        "export": {
            "checkpoint_step": int(checkpoint.manifest["nsteps"]),
            "export_evaluation_id": int(frame.info["evaluation_id"]),
            "max_momentum_error_ASE_units": float(
                np.max(np.abs(frame.get_momenta() - full_momenta))),
            "error_equals_missing_half_kick": bool(np.allclose(
                full_momenta - frame.get_momenta(), correction,
                rtol=1e-6, atol=1e-8)),
            "reported_temperature_K": inspect_run(config.run.directory)[
                "trajectory"]["last_temperature_K"],
            "complete_step_temperature_K": float(complete_frame.get_temperature()),
        },
    }


def incomplete_step(root: Path) -> dict:
    root.mkdir()
    atoms = Atoms("H", positions=[[0.8, 0.9, 0.9]])
    atoms.set_velocities([[0.05, 0, 0]])

    def fail_label(observation):
        if observation.step >= 0:
            raise RuntimeError("review failure before second half-kick")
        return False

    runner = EnergeticRunner(
        atoms, HarmonicSurrogate(), HarmonicReference(),
        Store(root / "trajectory.db"), "failed-step",
        force_budget=0.1, timestep_fs=0.5, check_probability=1,
        time_cap_fs=3, on_label=fail_label, event_log=EventLog(root),
        run_dir=root, checkpoint_interval_steps=5)
    error = None
    try:
        runner.run(2)
    except RuntimeError as exc:
        error = str(exc)
    finally:
        runner.close()
    events = [json.loads(line) for line in
              (root / "events.jsonl").read_text().splitlines()]
    info = inspect_run(root)
    return {
        "injected_error": error,
        "completed_step_events": sum(e["type"] == "step_completed" for e in events),
        "inspect_completed_steps": info["n_complete_steps"],
        "exported_frames_including_initial": export_run(root)["frames"],
        "checkpoint_step": CheckpointManager(root).read_latest_valid().manifest["nsteps"],
    }


def retry_accounting(root: Path, repository: Path) -> dict:
    root.mkdir()
    (root / "Si.UPF").write_text("placeholder: fake executable only")
    fixture = repository / "tests/data/qe_si_scf.out"
    counter = root / "calls.txt"
    script = root / "fake_pw.py"
    script.write_text(
        "from pathlib import Path\nimport sys\n"
        f"p=Path({str(counter)!r})\n"
        "n=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\n"
        "if n==1:\n print('transient launcher failure')\n sys.exit(1)\n"
        f"print(Path({str(fixture)!r}).read_text())\n")
    config = QeConfig(pseudo_dir=str(root), pseudos={"Si": "Si.UPF"},
                      pw_cmd=(sys.executable, str(script)), max_retries=1)
    run_dir = root / "run"
    run_dir.mkdir()
    engine = QeEngine(config, run_root=run_dir / "calculations")
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                  cell=[5.43] * 3, pbc=True)
    atoms.set_velocities(np.zeros((2, 3)))
    runner = EnergeticRunner(
        atoms, HarmonicSurrogate(), engine, Store(run_dir / "trajectory.db"),
        "retry-cost", force_budget=0.1, event_log=EventLog(run_dir),
        check_probability=0, temperature_K=0)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(0)
    finally:
        runner.close()
    return {
        "fake_executable_invocations": int(counter.read_text()),
        "backend_attempt_statuses": [x["status"] for x in engine.last_attempt_records],
        "ledger_reference": inspect_run(run_dir)["cost"]["reference"],
    }


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    with TemporaryDirectory(prefix="pyramid-independent-review-") as temporary:
        temporary = Path(temporary)
        results = {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
            "environment": {"python": sys.version.split()[0],
                            "ase": ase.__version__, "numpy": np.__version__},
            "real_DFT_executions": 0,
            "configuration_and_export": configuration_and_export(temporary / "normal"),
            "incomplete_step": incomplete_step(temporary / "failed"),
            "retry_accounting": retry_accounting(temporary / "retry", repository),
        }
    print(json.dumps(results, indent=2))
