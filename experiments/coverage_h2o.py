"""M2 acceptance gate: conformal coverage on H2O NVE (design-m2.md §4), CPU only.

Protocol:
1. Collect a 300-step NVE H2O trajectory at 300 K (dt = 0.5 fs) driven by
   frozen MACE-MP-0; compute PyscfEngine truth at every frame into a Store
   (all-labeled trajectory, fresh tmp dir).
2. Replay chronologically with a fresh CommitteeSurrogate (K=4 readout
   heads) + ConformalSwitch (alpha=0.05, eps_acc=0.1 eV/A, window=64,
   w_min=16, delta=1e-3) + OnlineUpdater (fine-tune every 8 labels).
3. Gate (design-m2.md §8, one-sided safety): alpha_hat over accepted frames
   must not exceed alpha + 0.03. Zero-acceptance runs are CORRECT-REFUSAL iff
   no frame meets the budget. Tightness and oracle overhead are reported as
   efficiency metrics, not gated.
4. Ablation: ScheduledSwitch at the same DFT fraction — report its
   alpha_hat for comparison.
5. Print a compact report; save both decision logs next to the store.

Usage:  uv run python experiments/coverage_h2o.py
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase import units
from ase.build import molecule
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

from pyraimd2.engines import PyscfEngine
from pyraimd2.loop import OnlineUpdater
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate
from pyraimd2.switch import ConformalSwitch, ScheduledSwitch, replay

MD_STEPS = 300
TIMESTEP_FS = 0.5
TEMPERATURE_K = 300.0
SEED = 20250819

ALPHA = 0.05
# Accuracy budget. First run (eps_acc = 0.1 eV/A) measured the head-only
# committee's error floor on OOD water: max-atom force error median 0.17,
# min 0.138 eV/A — zero frames below 0.1, so the switch correctly refused
# everything (302/302 DFT). 0.25 sits in the informative regime above the
# floor, where acceptance is possible and coverage is measurable.
EPS_ACC = 0.25  # eV/A
WINDOW = 64
W_MIN = 16
DELTA = 1e-3
N_LABEL = 8
GATE_TOL = 0.03

COLLECT_RUN_ID = "collect-h2o-nve-300K"


def collect(store: Store) -> int:
    """Drive NVE with frozen MACE-MP-0; label every frame with PySCF."""
    from mace.calculators import mace_mp  # local import: torch is heavy

    atoms = molecule("H2O")
    atoms.positions[1, 0] += 0.10  # same distorted start as the M1 examples
    thermalize_momenta(atoms, TEMPERATURE_K, rng=np.random.default_rng(SEED))
    atoms.calc = mace_mp(model="small", device="cpu", default_dtype="float64")
    engine = PyscfEngine()

    frames = [atoms.copy()]  # frame 0: the initial geometry
    dyn = VelocityVerlet(atoms, TIMESTEP_FS * units.fs)
    dyn.attach(lambda: frames.append(atoms.copy()))
    dyn.run(MD_STEPS)

    t0 = time.perf_counter()
    for step, frame in enumerate(frames):
        store.append(
            COLLECT_RUN_ID, step, frame, "dft", engine=engine.compute(frame),
            reason="collection: every frame labeled",
        )
        if step % 50 == 0:
            print(f"  [collect] labeled {step + 1}/{len(frames)} frames "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return len(frames)


class _ProgressCommittee:
    """Delegates to CommitteeSurrogate; prints a progress line per fine-tune."""

    def __init__(self, committee: CommitteeSurrogate, tag: str) -> None:
        self._committee = committee
        self._tag = tag

    def predict(self, atoms):
        return self._committee.predict(atoms)

    def finetune(self, labels):
        t0 = time.perf_counter()
        report = self._committee.finetune(labels)
        print(f"  [{self._tag}] finetune on {report.n_labels} labels: loss "
              f"{report.initial_loss:.3f} -> {report.final_loss:.3f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        return report


def _make_updater(committee: CommitteeSurrogate, observe, tag: str) -> OnlineUpdater:
    progress = _ProgressCommittee(committee, tag)
    return OnlineUpdater(progress, observe=observe, n_label=N_LABEL)


def run_conformal(frames):
    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ConformalSwitch(
        committee, alpha=ALPHA, eps_acc=EPS_ACC, window=WINDOW, w_min=W_MIN, delta=DELTA
    )
    updater = _make_updater(committee, switch.observe, "conformal")
    return replay(frames, committee, switch, updater=updater, eps_acc=EPS_ACC)


def run_scheduled(frames, dft_fraction: float):
    period = max(1, round(1.0 / dft_fraction))
    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ScheduledSwitch(period)
    updater = _make_updater(committee, lambda s, e: None, "scheduled")
    records, summary = replay(frames, committee, switch, updater=updater, eps_acc=EPS_ACC)
    return records, summary, period


def save_log(path: Path, records, summary) -> None:
    payload = {
        "summary": asdict(summary),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=True))


def main() -> int:
    outdir = Path(tempfile.mkdtemp(prefix="pyraimd2_m2_coverage_"))
    store = Store(outdir / "coverage_h2o.db")

    t0 = time.perf_counter()
    print(f"[1/3] collecting {MD_STEPS + 1} labeled frames (MACE-driven NVE, "
          f"PySCF truth on every frame) ...", flush=True)
    n_frames = collect(store)
    t_collect = time.perf_counter() - t0
    frames = list(store.iter_labels(COLLECT_RUN_ID))
    assert len(frames) == n_frames
    print(f"      done: {n_frames} frames in {t_collect:.0f} s", flush=True)

    print("[2/3] replay: fresh committee + conformal switch ...", flush=True)
    records_c, summary_c = run_conformal(frames)
    print(f"      done: dft_fraction={summary_c.dft_fraction:.3f} "
          f"alpha_hat={summary_c.alpha_hat:.4f} "
          f"({summary_c.wall_time_s:.0f} s)", flush=True)
    print("[3/3] replay: scheduled ablation at the same DFT fraction ...",
          flush=True)
    records_s, summary_s, period = run_scheduled(frames, summary_c.dft_fraction)
    print(f"      done: alpha_hat={summary_s.alpha_hat:.4f} "
          f"({summary_s.wall_time_s:.0f} s)", flush=True)

    save_log(outdir / "decision_log_conformal.json", records_c, summary_c)
    save_log(outdir / "decision_log_scheduled.json", records_s, summary_s)

    errors = np.array([r.error for r in records_c])
    bounds = np.array([r.bound for r in records_c])
    late = np.arange(len(errors)) >= 50  # post-warmup diagnostics
    # Bound tightness: B(s) / e, ~1 means the calibrated bound tracks the
    # true error (values > 1 are conservative by construction).
    tightness = float(np.median(bounds[late] / np.maximum(errors[late], 1e-12)))
    n_below_budget = int((errors <= EPS_ACC).sum())

    if summary_c.n_accepted == 0:
        if n_below_budget == 0:
            verdict = ("CORRECT-REFUSAL: no frame meets the accuracy budget; "
                       "routing everything to DFT is the right answer")
            exit_code = 0
        else:
            verdict = ("FAIL: frames below budget existed but none were "
                       "accepted — switch is over-conservative")
            exit_code = 1
    else:
        # One-sided safety gate (design-m2.md §8): conformal promises marginal
        # coverage >= 1 - alpha, i.e. alpha_hat must not exceed alpha + tol.
        # Tightness and oracle overhead are efficiency metrics, not gates.
        safe = summary_c.alpha_hat <= ALPHA + GATE_TOL
        verdict = (f"{'PASS' if safe else 'FAIL'}: alpha_hat = {summary_c.alpha_hat:.4f} "
                   f"vs one-sided safety bound {ALPHA + GATE_TOL:.2f}")
        exit_code = 0 if safe else 1

    print("\n=== PYRAIMD-2 M2 coverage experiment: H2O NVE @ 300 K ===")
    print(f"store:            {outdir / 'coverage_h2o.db'}")
    print(f"decision logs:    {outdir}/decision_log_{{conformal,scheduled}}.json")
    print(f"trajectory:       {MD_STEPS} steps x {TIMESTEP_FS} fs "
          f"= {MD_STEPS * TIMESTEP_FS:.0f} fs, {n_frames} frames all labeled "
          f"(collection {t_collect:.0f} s)")
    print(f"conformal params: alpha={ALPHA}, eps_acc={EPS_ACC} eV/A, "
          f"window={WINDOW}, w_min={W_MIN}, delta={DELTA}, n_label={N_LABEL}")
    print("---")
    print("conformal switch:")
    print(f"  DFT fraction:   {summary_c.dft_fraction:.3f} "
          f"({summary_c.n_dft}/{summary_c.n_frames})")
    print(f"  fine-tunes:     {summary_c.n_finetunes}")
    print(f"  alpha_hat:      {summary_c.alpha_hat:.4f} "
          f"(target {ALPHA} +/- {GATE_TOL})")
    print(f"  replay time:    {summary_c.wall_time_s:.0f} s")
    print("calibration quality (conformal):")
    print(f"  frames meeting budget:  {n_below_budget}/{len(errors)}")
    print(f"  bound tightness (late): median B(s)/e = {tightness:.2f}")
    print(f"  oracle DFT fraction:    {1 - n_below_budget / len(errors):.3f} "
          f"(frames not meeting budget)")
    print("scheduled ablation:")
    print(f"  period:         {period} (tuned to the same DFT fraction)")
    print(f"  DFT fraction:   {summary_s.dft_fraction:.3f} "
          f"({summary_s.n_dft}/{summary_s.n_frames})")
    sched_safe = summary_s.alpha_hat <= ALPHA + GATE_TOL
    print(f"  alpha_hat:      {summary_s.alpha_hat:.4f} "
          f"({'safe' if sched_safe else 'UNSAFE: exceeds ' + format(ALPHA + GATE_TOL, '.2f')})")
    print(f"  replay time:    {summary_s.wall_time_s:.0f} s")
    print("---")
    print(f"gate: {verdict}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
