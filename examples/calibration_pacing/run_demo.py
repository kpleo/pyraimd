"""Calibration pacing on a smooth analytic toy — enable, observe, compare.

Usage:
  uv run python examples/calibration_pacing/run_demo.py --output results/pacing-off
  uv run python examples/calibration_pacing/run_demo.py --output results/pacing-on --pacing
  uv run python examples/calibration_pacing/run_demo.py --output results/pacing-on --pacing \
      --resume --extra-steps 20

Two analytically-defined potentials drive a fixed-cell NVE run of H2:

- reference:  U_r = k s / 2 with s = sum_i |r_i - r0|^2 (k = 1.0, r0 = 0.9)
- surrogate:  U_s = U_r + A exp(-s / (2 l^2)), forces taken analytically

The surrogate error is smooth and position-dependent — largest near the
error peak, negligible in the tails.  With calibration pacing enabled,
a run of refused forecasts defers recalibration probes after
`failure_streak_limit` sterile calibrations and retries them on a bounded
backoff; `pacing_decision` events record every calibrate/defer choice, and
the resumed run replays them.  This toy is not a material, and its wall
clock says nothing about DFT speed — the point is the accounting: every
reference evaluation is billed by what actually ran, and each accepted
step's true driving-force error is checked below against the analytic
reference (offline verification, never used online).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms

from pyraimd2.backends.harmonic import HarmonicReference
from pyraimd2.loop import EnergeticRunner
from pyraimd2.loop.integrators import IntegratorSpec
from pyraimd2.runtime.events import EventLog
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogateCapabilities, SurrogatePrediction

K_REF, R0 = 1.0, 0.9
A_SURR, ELL = 0.1, 0.12  # eV / angstrom — the surrogate error bump
DR0, P0 = 0.02, 0.2      # initial offsets / momenta
BUDGET_EV_A = 0.05       # the acceptance force budget
DT_FS = 0.5


def smooth_energy(dr: np.ndarray) -> float:
    s = float((dr**2).sum())
    return 0.5 * K_REF * s + A_SURR * np.exp(-s / (2 * ELL**2))


def smooth_force(dr: np.ndarray) -> np.ndarray:
    s = float((dr**2).sum())
    return -K_REF * dr + (A_SURR / ELL**2) * np.exp(-s / (2 * ELL**2)) * dr


class SmoothSurrogate:
    """U_s with analytic forces (energy/force consistent)."""

    name = "smooth-surrogate"
    fingerprint = "smooth-surrogate-v1"

    @property
    def capabilities(self) -> SurrogateCapabilities:
        return SurrogateCapabilities()

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        dr = atoms.positions - R0
        return SurrogatePrediction(smooth_energy(dr), smooth_force(dr), None,
                                   np.full(len(atoms), np.nan))


def demo_atoms() -> Atoms:
    atoms = Atoms("H2",
                  positions=[[R0 - DR0, 0.9, 0.9], [R0 + DR0, 0.9, 0.9]],
                  cell=[10.0] * 3, pbc=False)
    atoms.set_momenta([[P0, 0.0, 0.0], [-P0, 0.0, 0.0]])
    return atoms


def build(output: Path, *, resume: bool, force_unlock: bool,
          pacing: bool) -> EnergeticRunner:
    """A fresh runner, or a resumed one through the ordinary protocol."""
    if resume:
        return EnergeticRunner.resume(
            output, SmoothSurrogate(), HarmonicReference(k=K_REF, r0=R0),
            event_log=EventLog(output, force=force_unlock))
    output.mkdir(parents=True, exist_ok=True)
    settings = (None if not pacing else
                {"failure_streak_limit": 3, "wait_initial": 1,
                 "wait_max": 8})
    return EnergeticRunner(
        demo_atoms(), SmoothSurrogate(), HarmonicReference(k=K_REF, r0=R0),
        Store(output / "trajectory.db"), "pacing-demo",
        run_dir=output, event_log=EventLog(output),
        checkpoint_interval_steps=10, force_budget=BUDGET_EV_A,
        timestep_fs=DT_FS, time_cap_fs=1000.0, transverse_cap=1.0,
        check_probability=0.5, check_seed=7,
        integrator_spec=IntegratorSpec(algorithm="velocity_verlet",
                                       ensemble="nve", timestep_fs=DT_FS,
                                       temperature_K=300.0,
                                       friction_per_fs=None,
                                       thermostat_seed=None),
        calibration_pacing=settings)


def report(output: Path) -> None:
    """Decisions, costs, accepted-step true errors, Hamiltonian drift."""
    from pyraimd2.runtime.costs import summarize_tasks

    events = [json.loads(line) for line in
              (output / "events.jsonl").read_text().splitlines()]
    decisions = [(e["evaluation_id"], e["decision"], e["reason"])
                 for e in events if e["type"] == "pacing_decision"]
    # Costs use the ledger's own semantics: actual physical executions by
    # purpose, never raw logical-task counts.
    cost = summarize_tasks(events)
    by_purpose = {b["purpose"]: b["count"] for b in cost["by"]
                  if b["operation"] == "reference"}
    n_reference = cost["reference"]["actual_executions"]
    complete_steps = {int(e["step_id"]) for e in events
                      if e["type"] == "step_completed"}
    errors = []
    hamiltonian = []
    with Store(output / "trajectory.db") as store:
        # The authoritative committed (commit event, row) pairs only —
        # never raw store scans, and orphan rows stay out.
        for commit, row in store.iter_committed(events, "pacing-demo"):
            step = int(row.key_value_pairs["step"])
            if step != -1 and step not in complete_steps:
                continue
            metadata = row.data.get("metadata") or {}
            context = metadata.get("context") or {}
            timestep = (float(context["physical_time_fs"])
                        / int(context["evaluation_id"])
                        if context.get("evaluation_id") else DT_FS)
            # The complete-step frame reconstructs full-step momenta (the
            # stored row is a mid-step record); the initial evaluation
            # carries the complete initial momenta.
            frame = store.complete_step_frame(row, timestep, commit=commit)
            # E_ref + K: the plain harmonic reference energy read directly,
            # kinetic energy from the complete-step momenta.
            dr = frame.positions - R0
            hamiltonian.append(float(0.5 * K_REF * (dr**2).sum())
                               + float(frame.get_kinetic_energy()))
            if row.key_value_pairs["route"] != "ml":
                continue
            driving = row.data.get("driving")
            if driving is None:
                continue
            error = float(np.linalg.norm(
                np.asarray(driving["forces"], dtype=float)
                - (-K_REF * (frame.positions - R0)), axis=1).max())
            errors.append(error)
    hamiltonian = np.asarray(hamiltonian)
    errors = np.asarray(errors)
    print(f"run directory: {output}")
    print(f"  reference evaluations: {n_reference} ({by_purpose})")
    print(f"  pacing decisions: {decisions if decisions else '— (feature off)'}")
    if len(errors):
        print(f"  accepted-step true force errors (analytic oracle): "
              f"max {errors.max():.4f} eV/A, RMS "
              f"{float(np.sqrt((errors**2).mean())):.4f} eV/A, "
              f"over budget ({BUDGET_EV_A}): {int((errors > BUDGET_EV_A).sum())}"
              f" of {len(errors)}")
    print(f"  reference Hamiltonian (E_ref + K) drift over complete steps: "
          f"{float(hamiltonian[-1] - hamiltonian[0]):.3e} eV "
          f"(range {float(np.ptp(hamiltonian)):.3e} eV)")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="run directory (user-owned)")
    parser.add_argument("--pacing", action="store_true",
                        help="enable calibration pacing (default: off)")
    parser.add_argument("--steps", type=int, default=40,
                        help="steps for a fresh run (default 40)")
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing run directory")
    parser.add_argument("--extra-steps", type=int, default=20,
                        help="additional steps when resuming (default 20)")
    parser.add_argument("--force-unlock", action="store_true",
                        help="reclaim a stale writer lock deliberately (only "
                             "after confirming no live writer exists)")
    args = parser.parse_args()
    if not args.resume and (args.output / "events.jsonl").exists():
        parser.error(f"{args.output} already has a run; use --resume to "
                     "continue it or choose a new output directory")
    runner = build(args.output, resume=args.resume,
                   force_unlock=args.force_unlock, pacing=args.pacing)
    if args.resume:
        summary = runner.run(args.extra_steps)
        print(f"resumed for {args.extra_steps} steps: "
              f"{summary.n_accepted} accepted, {summary.n_reference} "
              f"reference evaluations this call")
    else:
        summary = runner.run(args.steps)
        print(f"ran {args.steps} steps: {summary.n_accepted} accepted, "
              f"{summary.n_reference} reference evaluations")
    if summary.pacing is not None:
        print(f"  pacing this call: {summary.pacing['deferred']} deferred, "
              f"{summary.pacing['retried']} retried, "
              f"{summary.pacing['safety_exits']} safety exits")
    runner.close()
    report(args.output)


if __name__ == "__main__":
    main()
