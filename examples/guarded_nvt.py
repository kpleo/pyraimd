"""Guarded online-updated adaptive NVT on analytic toy potentials.

Usage:
  uv run python examples/guarded_nvt.py --output results/guarded-nvt --steps 20
  uv run python examples/guarded_nvt.py --output results/guarded-nvt --resume --extra-steps 13

A trainable harmonic model (least-squares stiffness fit) drives Langevin NVT
under the energetic force-error policy with anchoring and independent
reference checks; reference labels enter a GuardedUpdater queue, and
candidate models are validated on a fixed guard set before they publish or
roll back.  A run can be stopped and resumed in a new process with the same
dynamics, model chain and cost ledger.

The toy potentials exist to demonstrate the update contract and its
accounting — they are not a material validation, and the analytic wall clock
says nothing about DFT speed-ups.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.loop.integrators import IntegratorSpec, derive_stream_seed
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction, TrainReport

REFERENCE_K = 1.2
REFERENCE_QUARTIC = 0.05
INITIAL_K = 0.8


class QuarticReference:
    """Analytic reference: E = 1/2 k x^2 + 1/4 q x^4 (per component)."""

    name = "quartic-reference"

    def __init__(self):
        self.calls = 0

    def compute(self, atoms: Atoms) -> EngineResult:
        self.calls += 1
        x = atoms.positions
        return EngineResult(
            float(np.sum(0.5 * REFERENCE_K * x**2
                         + 0.25 * REFERENCE_QUARTIC * x**4)),
            -REFERENCE_K * x - REFERENCE_QUARTIC * x**3, None, 0.0)


class LeastSquaresHarmonic:
    """Trainable model: E = 1/2 k x^2, F = -k x; k fits by least squares.

    Deterministic fitting — the updater state carries every piece of
    training state (there is no training RNG to persist).  The fingerprint
    names the settings lineage; fitted parameters live in the state.
    """

    def __init__(self, k: float = INITIAL_K) -> None:
        self.k = float(k)
        self.finetune_calls = 0

    @property
    def fingerprint(self) -> str:
        return "least-squares-harmonic"

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        x = atoms.positions
        return SurrogatePrediction(0.5 * self.k * float(np.sum(x**2)),
                                   -self.k * x, None,
                                   np.full(len(atoms), np.nan))

    def finetune(self, labels) -> TrainReport:
        items = list(labels)
        if not items:
            raise ValueError("finetune needs at least one (atoms, label) pair")
        started = time.perf_counter()

        def force_mse() -> float:
            # Per-component force mean squared error over the TRAINING
            # labels, in eV^2/angstrom^2 — a training-set measure of the
            # fit, never an independent test error.
            errors = []
            for atoms, result in items:
                delta = -self.k * atoms.positions - result.forces
                errors.append(float((delta**2).mean()))
            return float(np.mean(errors))

        initial = force_mse()
        numerator = denominator = 0.0
        for atoms, result in items:
            x = atoms.positions
            numerator -= float((result.forces * x).sum())
            denominator += float((x**2).sum())
        self.k = numerator / denominator
        self.finetune_calls += 1
        final = force_mse()
        # The measured wall time of the fit itself; the GuardedUpdater's
        # outer task record bills the whole attempt (fit + guard) instead.
        return TrainReport(n_labels=len(items), n_epochs=1,
                           initial_loss=initial, final_loss=final,
                           member_losses=(final,),
                           wall_time_s=time.perf_counter() - started)

    def state_dict(self) -> dict:
        return {"k": self.k, "finetune_calls": self.finetune_calls}

    def load_state_dict(self, state: dict) -> None:
        self.k = float(state["k"])
        self.finetune_calls = int(state["finetune_calls"])


def build(output: Path, *, resume: bool, force_unlock: bool,
          checkpoint_interval: int):
    """Assemble runner pieces from public imports; returns ``(runner,
    store)`` — the store of a fresh run is caller-owned and must be closed
    by the caller (a resumed runner owns and closes its own)."""
    if resume:
        model = LeastSquaresHarmonic()
        updater = GuardedUpdater(model, UpdatePolicy(n_label=2,
                                                     guard_size=1))
        return (EnergeticRunner.resume(
            output, model, QuarticReference(), updater=updater,
            event_log_force=force_unlock,
            checkpoint_interval_steps=checkpoint_interval), None)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trajectory.db").exists():
        raise FileExistsError(
            f"{output} already contains a trajectory; choose a new directory "
            "or pass --resume")
    atoms = Atoms("H2", positions=[[0.85, 0.9, 0.9], [0.95, 0.9, 0.9]])
    atoms.set_momenta([[0.05, 0.02, 0.0], [-0.03, 0.01, 0.0]])
    model = LeastSquaresHarmonic()
    updater = GuardedUpdater(model, UpdatePolicy(n_label=2, guard_size=1))
    store = Store(output / "trajectory.db")
    # A Langevin NVT spec: the bath stream is role-derived from the seed,
    # so it never shares a generator with the velocity or check streams.
    spec = IntegratorSpec(algorithm="langevin", ensemble="nvt",
                          timestep_fs=0.5, temperature_K=300.0,
                          friction_per_fs=0.01,
                          thermostat_seed=derive_stream_seed(123,
                                                             "thermostat"))
    return (EnergeticRunner(
        atoms, model, QuarticReference(), store, "guarded-nvt",
        run_dir=output, event_log=EventLog(output),
        checkpoint_interval_steps=checkpoint_interval,
        force_budget=0.5, timestep_fs=0.5, time_cap_fs=100.0,
        transverse_cap=1.0, check_probability=0.5, check_seed=7,
        integrator_spec=spec, on_label=updater), store)


def report(output: Path) -> dict:
    """Costs and outcomes from the authoritative event log and store."""
    events = [json.loads(line)
              for line in (output / "events.jsonl").read_text().splitlines()]
    tasks = [e for e in events if e["type"] == "task"]
    reference = [t for t in tasks if t.get("operation") == "reference"
                 and t["status"] == "success"]
    by_purpose = {}
    for task in reference:
        by_purpose[task["purpose"]] = by_purpose.get(task["purpose"], 0) + 1
    trainings = [t for t in tasks if t.get("operation") == "training"]
    updates = [e for e in events if e["type"] == "model_update"]
    rejections = [e for e in events if e["type"] == "update_rejected"]
    steps = [e for e in events if e["type"] == "step_completed"]
    committed = [e for e in events if e["type"] == "evaluation_committed"]
    works = [(int(e["context"]["evaluation_id"]), e["segment_id"],
              e["observed"]["endpoint_work_eV"])
             for e in committed if e.get("observed") is not None]
    return {
        "complete_steps": len(steps),
        "evaluations": len(committed),
        "reference_calls": dict(sorted(by_purpose.items())),
        "reference_total": len(reference),
        "inference_calls": len([t for t in tasks
                                if t.get("operation") == "inference"
                                and t["status"] == "success"]),
        "training_callbacks": len(trainings),
        "training_callbacks_retried_on_recovery": len(
            [t for t in trainings if t.get("recovery")]),
        "updates_published": len(updates),
        "updates_rejected": len(rejections),
        "model_generations": [e["generation"] for e in updates],
        "segment_residual_work_eV": [
            {"evaluation": index, "segment": segment, "work_eV": work}
            for index, segment, work in works],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="the run directory (user-owned output)")
    parser.add_argument("--steps", type=int, default=20,
                        help="steps for a new run")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run in --output (new process)")
    parser.add_argument("--extra-steps", type=int, default=13,
                        help="additional steps when resuming")
    parser.add_argument("--force-unlock", action="store_true",
                        help="reclaim the writer lock of a crashed run")
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    args = parser.parse_args()

    runner, store = build(args.output, resume=args.resume,
                          force_unlock=args.force_unlock,
                          checkpoint_interval=args.checkpoint_interval)
    try:
        summary = runner.run(args.extra_steps if args.resume else args.steps)
    finally:
        runner.close()  # the event log and (for a resume) the owned store
        if store is not None:
            store.close()  # a fresh run's store is this example's own
    result = report(args.output)
    result["steps_this_call"] = summary.n_steps
    result["accepted_fraction"] = summary.accepted_fraction
    mode = "resumed" if args.resume else "new"
    print(f"guarded NVT ({mode}): {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
