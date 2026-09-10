"""Ensemble acceptance for plain Langevin NVT on a harmonic oscillator.

Three layers, with error bars derived from the actual discrete propagator
rather than a guessed precision:

1. A deterministic covariance oracle: four basis calls per timestep recover
   the adapter's exact one-coordinate linear update z' = A z + B xi, and a
   2x2 discrete Lyapunov solve gives the stationary covariance — no
   trajectory, no seed lottery.  ASE's Langevin is the
   Vanden-Eijnden–Ciccotti propagator (per the ASE source note), not BAOAB;
   its harmonic marginals carry a small O(dt^2) timestep bias, measured
   here: at dt=0.5 fs the position variance is +0.020% and the mean kinetic
   energy -0.042% off canonical; at dt=1.0 fs, +0.080% and -0.166%.  Small
   and shrinking with dt is correct wiring evidence; "unbiased" would be
   wrong, so the assertions carry the measured bias bounds.
2. A short end-to-end sampling-chain check (400 warmup + 1200 retained
   steps, a few seconds) whose tolerances come from the same oracle at the
   same budget: exact estimator SEMs via the stationary autocovariance, no
   fixed "7%" gate.  The two timesteps run with different seeds — same-seed
   kinetic estimates are positively correlated (the oracle measures ~0.24),
   so a shared seed would understate the joint error.
3. The long 6000-sample run stays as a coarse sanity check, marked ``slow``
   (runs under ``--runslow``; the CI statistical job executes it).  Its
   gates likewise use oracle SEMs at its own budget (about 8.0%, 7.4% and
   7.6% for the three estimators — block averaging at 25 blocks reads ~10%
   low, so the gates are not 7%).

Notes on what is NOT independent evidence: mean(E) = mean(K) + 1.5 k mean(x^2)
for this oscillator, so the total energy is a dependent consistency view of
the same data, never a third independent estimator.  Block SEM measures
estimator uncertainty, not the physical fluctuation amplitude; canonical
variance targets (Var(K) = 1.5 (kB T)^2, Var(E) = 3 (kB T)^2) are not
claimed here.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from test_nvt import rows, write_nvt

from pyraimd2.config import load_config
from pyraimd2.loop.integrators import IntegratorSpec, LangevinAdapter
from pyraimd2.workflows import run_workflow

TEMPERATURE = 300.0
KBT = units.kB * TEMPERATURE  # eV
FRICTION = 0.04  # per fs
SPRING_K = 1.0  # eV/A^2
H_MASS = float(Atoms("H").get_masses()[0])


class _HarmonicCalc(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": float(0.5 * (atoms.positions**2).sum()),
                        "forces": -atoms.positions.copy()}


class _BasisRNG:
    """Feeds one basis value into each of the two noise draws per step."""

    def __init__(self, xi, eta):
        self.values = iter((xi, eta))

    def standard_normal(self, size):
        out = np.zeros(size)
        out[0, 0] = next(self.values)
        return out


def _linear_update(dt_fs):
    """The adapter's exact one-coordinate update z' = A z + B xi, recovered
    from basis calls to the real step (not a reimplementation of its
    formulas)."""
    spec = IntegratorSpec("langevin", "nvt", dt_fs, TEMPERATURE, FRICTION, 123)

    def advance(x, v, xi, eta):
        atoms = Atoms("H", positions=[[x, 0, 0]])
        atoms.set_velocities([[v, 0, 0]])
        atoms.calc = _HarmonicCalc()
        adapter = LangevinAdapter(atoms, spec)
        adapter.dyn.rng = _BasisRNG(xi, eta)
        adapter.step(atoms.get_forces())
        return np.array([atoms.positions[0, 0], atoms.get_velocities()[0, 0]])

    a = np.column_stack((advance(1, 0, 0, 0), advance(0, 1, 0, 0)))
    b = np.column_stack((advance(0, 0, 1, 0), advance(0, 0, 0, 1)))
    return a, b


def _stationary_covariance(a, b):
    return np.linalg.solve(np.eye(4) - np.kron(a, a), (b @ b.T).ravel()).reshape(2, 2)


def _observable_stats(a, cov, n_samples, n_blocks):
    """Exact estimator statistics for one coordinate's x^2, the kinetic
    energy, and the total energy, from the stationary autocovariance of the
    actual discrete process (Gaussian quadratic-moment identities)."""
    weights = {
        "position_squared": np.diag([1.0, 0.0]),
        "kinetic": np.diag([0.0, H_MASS / 2]),
        "total_energy": np.diag([SPRING_K / 2, H_MASS / 2]),
    }
    block_length = n_samples // n_blocks
    out = {}
    lagged = cov.copy()
    autocov = {key: [] for key in weights}
    for _lag in range(n_samples):
        for key, w in weights.items():
            autocov[key].append(2 * np.trace(w @ lagged @ w @ lagged.T))
        lagged = a @ lagged
    for key, ac in autocov.items():
        ac = np.array(ac)
        var_mean = (ac[0] + 2 * np.dot(1 - np.arange(1, n_samples) / n_samples,
                                       ac[1:])) / n_samples
        var_block = (ac[0] + 2 * np.dot(1 - np.arange(1, block_length) / block_length,
                                        ac[1:block_length])) / block_length
        expected_block_sem2 = (var_block - var_mean) / (n_blocks - 1)
        mean = np.trace(weights[key] @ cov)
        # Three Cartesian coordinates are pooled (x^2) or summed (K, E).
        exact_sem = float(np.sqrt(var_mean / 3) / mean)
        out[key] = {
            "mean": float(mean),
            "exact_rel_sem_3coords": exact_sem,
            "expected_block_rel_sem": float(np.sqrt(expected_block_sem2 / 3) / mean),
        }
    return out


# --- 1. deterministic covariance oracle -------------------------------------


def test_langevin_stationary_covariance_and_timestep_bias():
    """The propagator's stationary covariance by discrete Lyapunov: small,
    measured, dt-shrinking timestep bias — bounded, not absent."""
    biases = {}
    for dt_fs in (0.5, 1.0):
        a, b = _linear_update(dt_fs)
        cov = _stationary_covariance(a, b)
        biases[dt_fs] = (
            cov[0, 0] / (KBT / SPRING_K),          # Var(x) / canonical
            H_MASS * cov[1, 1] / KBT,              # <K>/coord / canonical
        )
    # Reference values measured from this oracle on ASE 3.29.0; they pin the
    # wiring, and the bounds state the bias instead of denying it.
    assert biases[0.5][0] == pytest.approx(1.0001995409, rel=1e-9)
    assert biases[0.5][1] == pytest.approx(0.9995847434, rel=1e-9)
    assert biases[1.0][0] == pytest.approx(1.0007996672, rel=1e-9)
    assert biases[1.0][1] == pytest.approx(0.9983402076, rel=1e-9)
    # Bias magnitude grows with dt (consistency trend), staying sub-0.1%
    # at dt=0.5 fs for both marginals.
    assert abs(biases[1.0][0] - 1) > abs(biases[0.5][0] - 1)
    assert abs(biases[0.5][0] - 1) < 1e-3 and abs(biases[0.5][1] - 1) < 1e-3
    assert abs(biases[1.0][0] - 1) < 3e-3 and abs(biases[1.0][1] - 1) < 3e-3


def test_oracle_sem_explains_the_sampling_budget():
    """The exact SEM at the long budget is ~7.4-8.0%, not the 7% gate the
    long test once carried; the oracle is the tolerance source everywhere."""
    a, b = _linear_update(0.5)
    cov = _stationary_covariance(a, b)
    stats = _observable_stats(a, cov, 6000, 25)
    assert stats["position_squared"]["exact_rel_sem_3coords"] == pytest.approx(
        0.0802319, rel=1e-5)
    assert stats["kinetic"]["exact_rel_sem_3coords"] == pytest.approx(
        0.0742300, rel=1e-5)
    assert stats["total_energy"]["exact_rel_sem_3coords"] == pytest.approx(
        0.0757608, rel=1e-5)
    # The 25-block SEM estimate itself reads ~10% low at this budget.
    for key in stats:
        ratio = (stats[key]["expected_block_rel_sem"]
                 / stats[key]["exact_rel_sem_3coords"])
        assert ratio == pytest.approx(0.89, abs=0.03)


# --- 2. short end-to-end sampling chain -------------------------------------

SHORT_WARMUP = 400
SHORT_SAMPLES = 1200
SHORT_BLOCKS = 10


def _run_oscillator(tmp_path, *, dt, seed, warmup, samples, n_blocks_out):
    config = load_config(write_nvt(
        tmp_path, dt=dt, steps=warmup + samples, temperature=TEMPERATURE,
        friction=FRICTION, thermostat_seed=seed, k=SPRING_K,
        positions=((0.9, 0.9, 0.9),),
        momenta=[[0.0, 0.0, 0.0]],
        checkpoint_interval=2000,
        trajectory_interval=2000, summary_interval=2000))
    run_workflow(config, verbose=False, handle_sigint=False)
    run_rows = rows(config.run.directory)[warmup + 1:]
    assert len(run_rows) == samples
    displacements = np.array([r.toatoms().positions[0] - 0.9
                              for r in run_rows])
    mass = run_rows[0].toatoms().get_masses()[0]
    kinetics = np.array([float((r.toatoms().get_momenta() ** 2
                                / (2.0 * mass)).sum())
                         for r in run_rows])
    return displacements, kinetics


def _block_mean_sem(samples, n_blocks):
    blocks = np.array_split(np.asarray(samples, dtype=float), n_blocks)
    means = np.array([block.mean() for block in blocks])
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(n_blocks))


def _block_sem_band(expected_sem, n_blocks):
    """3-sigma band of the block-SEM estimator itself: a variance estimate
    from B blocks carries relative std ~1/sqrt(2(B-1)) (chi-square with
    B-1 dof), so the measured SEM is compared within expected*(1 +/- 3x).
    This catches gross miscalibration (the old fixed 7% gate) without
    punishing the estimator's own fluctuation."""
    spread = 3.0 / np.sqrt(2 * (n_blocks - 1))
    return expected_sem * (1.0 - spread), expected_sem * (1.0 + spread)


def test_short_sampling_chain_matches_oracle_gates(tmp_path):
    """One short run per timestep (independent seeds — same-seed estimates
    are positively correlated).  Gates: 4x the oracle's exact SEM at this
    budget, with the measured block SEM inside a band around its exact
    expectation (~10% low bias included), not a hardcoded precision."""
    a, b = _linear_update(0.5)
    cov = _stationary_covariance(a, b)
    oracle = _observable_stats(a, cov, SHORT_SAMPLES, SHORT_BLOCKS)

    displacements, kinetics = _run_oscillator(
        tmp_path / "short", dt=0.5, seed=20240910,
        warmup=SHORT_WARMUP, samples=SHORT_SAMPLES, n_blocks_out=SHORT_BLOCKS)

    variance, sem = _block_mean_sem((displacements ** 2).reshape(-1),
                                    SHORT_BLOCKS)
    exact = oracle["position_squared"]
    assert variance == pytest.approx(exact["mean"], rel=0,
                                     abs=4 * exact["mean"] * exact["exact_rel_sem_3coords"])
    lo, hi = _block_sem_band(
        exact["expected_block_rel_sem"] * exact["mean"], SHORT_BLOCKS)
    assert lo < sem < hi

    kinetic, sem_k = _block_mean_sem(kinetics, SHORT_BLOCKS)
    exact_k = oracle["kinetic"]
    assert kinetic == pytest.approx(3 * exact_k["mean"], rel=0,
                                    abs=4 * 3 * exact_k["mean"]
                                    * exact_k["exact_rel_sem_3coords"])
    lo, hi = _block_sem_band(
        exact_k["expected_block_rel_sem"] * 3 * exact_k["mean"], SHORT_BLOCKS)
    assert lo < sem_k < hi

    # Dependent consistency view (mean E = mean K + 1.5 k mean x^2 here):
    # same data, never counted as a third independent estimator.
    total = kinetics + 0.5 * (displacements ** 2).sum(axis=1)
    e_total, _ = _block_mean_sem(total, SHORT_BLOCKS)
    exact_e = oracle["total_energy"]
    assert e_total == pytest.approx(3 * exact_e["mean"], rel=0,
                                    abs=4 * 3 * exact_e["mean"]
                                    * exact_e["exact_rel_sem_3coords"])

    # dt comparison with a *different* seed: independent streams, so the
    # joint error is the quadrature sum (no hidden positive covariance).
    _, kinetics_large = _run_oscillator(
        tmp_path / "short-dt10", dt=1.0, seed=20240912,
        warmup=SHORT_WARMUP, samples=SHORT_SAMPLES, n_blocks_out=SHORT_BLOCKS)
    a2, b2 = _linear_update(1.0)
    oracle2 = _observable_stats(a2, _stationary_covariance(a2, b2),
                                SHORT_SAMPLES, SHORT_BLOCKS)["kinetic"]
    mean_large, _ = _block_mean_sem(kinetics_large, SHORT_BLOCKS)
    assert mean_large == pytest.approx(3 * oracle2["mean"], rel=0,
                                       abs=4 * 3 * oracle2["mean"]
                                       * oracle2["exact_rel_sem_3coords"])
    joint = 4 * 3 * float(np.hypot(
        exact_k["mean"] * exact_k["exact_rel_sem_3coords"],
        oracle2["mean"] * oracle2["exact_rel_sem_3coords"]))
    assert abs(kinetic - mean_large) <= joint


# --- 3. long coarse sanity run (slow: CI statistical job) --------------------

LONG_WARMUP = 1000
LONG_SAMPLES = 6000
LONG_BLOCKS = 25


@pytest.mark.slow
def test_long_harmonic_sampling_coarse_sanity(tmp_path):
    """The retained long run: 6000 samples, 25 blocks of 240 steps (120 fs
    at dt=0.5 fs).  Gates use the oracle's exact SEM at this budget
    (~8%); no seed was selected for passing."""
    a, b = _linear_update(0.5)
    cov = _stationary_covariance(a, b)
    oracle = _observable_stats(a, cov, LONG_SAMPLES, LONG_BLOCKS)

    displacements, kinetics = _run_oscillator(
        tmp_path / "long", dt=0.5, seed=20240910,
        warmup=LONG_WARMUP, samples=LONG_SAMPLES, n_blocks_out=LONG_BLOCKS)

    variance, sem = _block_mean_sem((displacements ** 2).reshape(-1), LONG_BLOCKS)
    exact = oracle["position_squared"]
    assert variance == pytest.approx(exact["mean"], rel=0,
                                     abs=4 * exact["mean"] * exact["exact_rel_sem_3coords"])
    lo, hi = _block_sem_band(
        exact["expected_block_rel_sem"] * exact["mean"], LONG_BLOCKS)
    assert lo < sem < hi

    kinetic, sem_k = _block_mean_sem(kinetics, LONG_BLOCKS)
    exact_k = oracle["kinetic"]
    assert kinetic == pytest.approx(3 * exact_k["mean"], rel=0,
                                    abs=4 * 3 * exact_k["mean"]
                                    * exact_k["exact_rel_sem_3coords"])
    lo, hi = _block_sem_band(
        exact_k["expected_block_rel_sem"] * 3 * exact_k["mean"], LONG_BLOCKS)
    assert lo < sem_k < hi

    # Dependent consistency view only (see module docstring).
    total = kinetics + 0.5 * (displacements ** 2).sum(axis=1)
    e_total, sem_e = _block_mean_sem(total, LONG_BLOCKS)
    exact_e = oracle["total_energy"]
    assert e_total == pytest.approx(3 * exact_e["mean"], rel=0,
                                    abs=4 * 3 * exact_e["mean"]
                                    * exact_e["exact_rel_sem_3coords"])
    lo, hi = _block_sem_band(
        exact_e["expected_block_rel_sem"] * 3 * exact_e["mean"], LONG_BLOCKS)
    assert lo < sem_e < hi


@pytest.mark.slow
def test_long_dt_trend_at_two_stepsizes(tmp_path):
    """Both timesteps agree with the oracle within their exact errors; the
    two runs use different seeds (independent streams)."""
    _, kinetics_small = _run_oscillator(
        tmp_path / "dt05", dt=0.5, seed=20240911,
        warmup=LONG_WARMUP, samples=LONG_SAMPLES, n_blocks_out=LONG_BLOCKS)
    _, kinetics_large = _run_oscillator(
        tmp_path / "dt10", dt=1.0, seed=20240913,
        warmup=LONG_WARMUP, samples=LONG_SAMPLES, n_blocks_out=LONG_BLOCKS)
    means = {}
    sems = {}
    for dt, series in ((0.5, kinetics_small), (1.0, kinetics_large)):
        a, b = _linear_update(dt)
        oracle = _observable_stats(a, _stationary_covariance(a, b),
                                   LONG_SAMPLES, LONG_BLOCKS)["kinetic"]
        mean, _ = _block_mean_sem(series, LONG_BLOCKS)
        assert mean == pytest.approx(3 * oracle["mean"], rel=0,
                                     abs=4 * 3 * oracle["mean"]
                                     * oracle["exact_rel_sem_3coords"])
        means[dt], sems[dt] = mean, 3 * oracle["mean"] * oracle["exact_rel_sem_3coords"]
    joint = 4 * float(np.hypot(sems[0.5], sems[1.0]))
    assert abs(means[0.5] - means[1.0]) <= joint
