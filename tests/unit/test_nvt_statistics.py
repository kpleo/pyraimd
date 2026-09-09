"""M2 ensemble acceptance: a harmonic oscillator under plain Langevin NVT
samples position variance and kinetic energy with the right means, with
tolerances derived from block-averaged standard errors (never a single
instantaneous temperature).  Units: eV, angstrom, fs, K throughout.

Statistics notes (kept with the test):
- Theory: one H atom in E = 1/2 k x^2 (k = 1 eV/A^2) at T = 300 K;
  per-coordinate <x^2> = kB T / k and mean kinetic per free DOF = kB T / 2.
- Sampling: 6000 steps after 1000 equilibration steps, friction 0.04/fs
  (velocity correlation time ~25 fs = 50 steps).  Blocks of 120 steps
  (60 fs) exceed twice the correlation time, so 25 block means are
  treated as independent; tolerances are 4x the block standard error of
  the mean (SEM).  No seed is chosen for "passing": the seed is fixed in
  the config.
- Resolution: at this budget all three estimators (position variance,
  mean kinetic, total energy) resolve to ~6% of their measured quantity;
  the 7% resolution guards below are recorded with the assertions, and
  the theory comparisons stay at 4x SEM (~20-25%) — stable, yet tight
  enough to catch a wrong temperature factor, a wrong DOF count or a
  misapplied bath coupling (all of which are 2x effects or larger).
- Bussi-Parrinello Langevin (BAOAB family) samples the harmonic marginals
  without a timestep bias at these steps, so two timesteps must both agree
  with theory within their joint error — that is the dt-trend check.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import units
from test_nvt import rows, write_nvt

from pyraimd2.config import load_config
from pyraimd2.workflows import run_workflow

EQUILIBRATION = 1000
SAMPLES = 6000
N_BLOCKS = 25


def _run_oscillator(tmp_path, *, dt, seed):
    config = load_config(write_nvt(
        tmp_path, dt=dt, steps=EQUILIBRATION + SAMPLES, temperature=300.0,
        friction=0.04, thermostat_seed=seed, k=1.0,
        positions=((0.9, 0.9, 0.9),),
        momenta=[[0.0, 0.0, 0.0]],
        checkpoint_interval=2000,
        trajectory_interval=2000, summary_interval=2000))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_rows = rows(config.run.directory)[EQUILIBRATION + 1:]
    assert len(run_rows) == SAMPLES
    displacements = np.array([r.toatoms().positions[0] - 0.9
                              for r in run_rows])
    masses = run_rows[0].toatoms().get_masses()[0]
    kinetics = np.array([float((r.toatoms().get_momenta() ** 2
                                / (2.0 * masses)).sum())
                         for r in run_rows])
    return displacements, kinetics


def _block_mean_sem(samples, n_blocks=N_BLOCKS):
    blocks = np.array_split(np.asarray(samples, dtype=float), n_blocks)
    means = np.array([block.mean() for block in blocks])
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(n_blocks))


def test_harmonic_oscillator_position_variance_and_kinetic_mean(tmp_path):
    displacements, kinetics = _run_oscillator(tmp_path / "osc", dt=0.5,
                                              seed=20240910)
    kbt = units.kB * 300.0  # eV

    # Position variance per coordinate: <x^2> = kB T / k  (k = 1 eV/A^2).
    # The oscillator is isotropic, so the three iid coordinate samples are
    # pooled inside each time block.  At this sample budget the estimator's
    # resolution is about 6% of kB T (see the module docstring); the theory
    # check itself uses 4x the block SEM.
    variance, sem = _block_mean_sem((displacements ** 2).reshape(-1))
    assert sem < 0.07 * kbt
    assert variance == pytest.approx(kbt, rel=0, abs=4 * sem)

    # Mean kinetic energy: 3 free DOF of one atom, 1/2 kB T each.
    kinetic, sem_k = _block_mean_sem(kinetics)
    assert sem_k < 0.07 * (1.5 * kbt)
    assert kinetic == pytest.approx(1.5 * kbt, rel=0, abs=4 * sem_k)

    # Total energy E = T + 1/2 k|x|^2 = 3 kB T — the third independent
    # view of the same thermostat temperature.
    total = kinetics + 0.5 * (displacements ** 2).sum(axis=1)
    e_total, sem_e = _block_mean_sem(total)
    assert sem_e < 0.07 * (3.0 * kbt)
    assert e_total == pytest.approx(3.0 * kbt, rel=0, abs=4 * sem_e)


def test_dt_trend_is_consistent_with_theory_at_two_stepsizes(tmp_path):
    _, kinetics_small = _run_oscillator(tmp_path / "dt05", dt=0.5,
                                        seed=20240911)
    _, kinetics_large = _run_oscillator(tmp_path / "dt10", dt=1.0,
                                        seed=20240911)
    kbt = units.kB * 300.0
    mean_small, sem_small = _block_mean_sem(kinetics_small)
    mean_large, sem_large = _block_mean_sem(kinetics_large)
    assert mean_small == pytest.approx(1.5 * kbt, rel=0, abs=4 * sem_small)
    assert mean_large == pytest.approx(1.5 * kbt, rel=0, abs=4 * sem_large)
    # The two estimates agree within their joint (independent) error —
    # no systematic drift of the mean with dt at these stepsizes.
    joint = 4 * float(np.hypot(sem_small, sem_large))
    assert abs(mean_small - mean_large) <= joint
