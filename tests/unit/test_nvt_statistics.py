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
   fixed "7%" gate.  The block-SEM self-consistency check asserts on
   R = sem^2/E[sem^2] with acceptance edges taken as empirical 1e-3
   quantiles of R under 4000 fixed-seed replicas of the exact discrete
   process (an iid-block chi^2 reference under-spreads here — adjacent
   block means correlate and block means of x^2 are skewed).  The two
   timesteps run with different seeds: same-seed kinetic estimates are
   positively correlated (the oracle measures ~0.24), and
   Var(A-B) = Var(A) + Var(B) - 2Cov(A,B) — dropping that positive
   covariance would overstate the difference's uncertainty, so the
   comparison uses independent streams whose quadrature error model is
   exact.
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


def _sem_acceptance_ratio_band(a, b, cov, w, *, warmup, n_samples, n_blocks,
                               alpha=1e-3, n_rep=4000, mc_seed=7):
    """Acceptance band for the block-SEM of one trajectory, stated on
    R = sem_hat^2 / E[sem_hat^2] (the variance dimension — the estimator is
    a variance ratio, so the interval is not written on the SD).

    ``w`` is the 2x2 weight of the quadratic observable q = z' w z per
    coordinate (x^2, kinetic m v^2/2, or total energy); R is invariant to
    the pooling-vs-sum layout of the three coordinates, so the band is
    computed for the pooled layout.

    Reference distribution: the exact discrete propagator itself.  E[sem^2]
    comes from its stationary autocovariance; the band edges are empirical
    alpha/2 and 1-alpha/2 quantiles of R over 4000 fixed-seed replicas of
    that process in the same warmup/block layout (a few seconds,
    deterministic).  An iid-block chi^2_{B-1} reference would under-spread:
    adjacent block means correlate and block means of x^2 are skewed —
    measured rel. std of R is ~0.37 at the long budget vs chi^2_24's 0.29,
    and the true 0.9995 upper quantile of R is ~3.2 vs chi^2's 2.23.  The
    nominal two-sided false-alarm budget is alpha=1e-3 (binomial resolution
    at 4000 replicas ~5e-4 per tail — empirical replica quantiles are not
    an exact coverage guarantee).  Resolution: a 2x SEM miscalibration
    multiplies a realization's R by 4, far beyond the band's bulk, so the
    check catches gross error-model breakage; it is not a calibrated
    detection guarantee at that factor.  Physical resolution lives in the
    4x-exact-SEM mean gates.
    """
    rng = np.random.default_rng(mc_seed)
    L = n_samples // n_blocks
    z = rng.multivariate_normal(np.zeros(2), cov, size=(n_rep, 3))
    block_sums = np.zeros((n_rep, 3, n_blocks))
    for i in range(warmup + n_samples):
        z = z @ a.T + rng.standard_normal((n_rep, 3, 2)) @ b.T
        if i >= warmup:
            q = w[0, 0] * z[:, :, 0] ** 2 + w[1, 1] * z[:, :, 1] ** 2
            block_sums[:, :, (i - warmup) // L] += q
    # The test pools the three coordinates inside each time block
    # (reshape(-1) layout): pool first, then take the across-block spread.
    pooled_means = block_sums.mean(axis=1) / L
    sem2 = (pooled_means.std(axis=1, ddof=1) / np.sqrt(n_blocks)) ** 2
    # E[sem^2] from the exact autocovariance of q (single-coordinate
    # formula, then pooled over the three independent coordinates).
    lagged = cov.copy()
    ac = np.empty(n_samples)
    for lag in range(n_samples):
        ac[lag] = 2 * np.trace(w @ lagged @ w @ lagged.T)
        lagged = a @ lagged
    var_mean = (ac[0] + 2 * np.dot(1 - np.arange(1, n_samples) / n_samples,
                                   ac[1:])) / n_samples
    var_block = (ac[0] + 2 * np.dot(1 - np.arange(1, L) / L, ac[1:L])) / L
    e_sem2 = (var_block - var_mean) / (n_blocks - 1) / 3.0
    ratio = sem2 / e_sem2
    lo, hi = np.quantile(ratio, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), float(e_sem2)


def test_short_sampling_chain_matches_oracle_gates(tmp_path):
    """One short run per timestep (independent seeds — same-seed estimates
    are positively correlated, see the module note).  Gates: 4x the
    oracle's exact SEM at this budget, with the measured block SEM inside a
    band around its exact expectation, not a hardcoded precision.  At this
    short budget the block SEM reads ~18-22% low across the three
    estimators — the long-budget ~10% note does not transfer; the band's
    exact E[sem^2] normalization already absorbs that bias."""
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
    lo_r, hi_r, e_sem2 = _sem_acceptance_ratio_band(
        a, b, cov, np.diag([1.0, 0.0]), warmup=SHORT_WARMUP,
        n_samples=SHORT_SAMPLES, n_blocks=SHORT_BLOCKS)
    assert lo_r <= sem**2 / e_sem2 <= hi_r

    kinetic, sem_k = _block_mean_sem(kinetics, SHORT_BLOCKS)
    exact_k = oracle["kinetic"]
    assert kinetic == pytest.approx(3 * exact_k["mean"], rel=0,
                                    abs=4 * 3 * exact_k["mean"]
                                    * exact_k["exact_rel_sem_3coords"])
    # The kinetic series is the per-step sum over 3 coordinates; the
    # summed estimator's E[sem^2] is 9x the pooled one used for the band.
    lo_r, hi_r, e_sem2_k = _sem_acceptance_ratio_band(
        a, b, cov, np.diag([0.0, H_MASS / 2]), warmup=SHORT_WARMUP,
        n_samples=SHORT_SAMPLES, n_blocks=SHORT_BLOCKS)
    assert lo_r <= sem_k**2 / (9 * e_sem2_k) <= hi_r

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
    lo_r, hi_r, e_sem2 = _sem_acceptance_ratio_band(
        a, b, cov, np.diag([1.0, 0.0]), warmup=LONG_WARMUP,
        n_samples=LONG_SAMPLES, n_blocks=LONG_BLOCKS)
    assert lo_r <= sem**2 / e_sem2 <= hi_r

    kinetic, sem_k = _block_mean_sem(kinetics, LONG_BLOCKS)
    exact_k = oracle["kinetic"]
    assert kinetic == pytest.approx(3 * exact_k["mean"], rel=0,
                                    abs=4 * 3 * exact_k["mean"]
                                    * exact_k["exact_rel_sem_3coords"])
    lo_r, hi_r, e_sem2_k = _sem_acceptance_ratio_band(
        a, b, cov, np.diag([0.0, H_MASS / 2]), warmup=LONG_WARMUP,
        n_samples=LONG_SAMPLES, n_blocks=LONG_BLOCKS)
    assert lo_r <= sem_k**2 / (9 * e_sem2_k) <= hi_r

    # Dependent consistency view only (see module docstring).
    total = kinetics + 0.5 * (displacements ** 2).sum(axis=1)
    e_total, sem_e = _block_mean_sem(total, LONG_BLOCKS)
    exact_e = oracle["total_energy"]
    assert e_total == pytest.approx(3 * exact_e["mean"], rel=0,
                                    abs=4 * 3 * exact_e["mean"]
                                    * exact_e["exact_rel_sem_3coords"])
    lo_r, hi_r, e_sem2_e = _sem_acceptance_ratio_band(
        a, b, cov, np.diag([SPRING_K / 2, H_MASS / 2]), warmup=LONG_WARMUP,
        n_samples=LONG_SAMPLES, n_blocks=LONG_BLOCKS)
    assert lo_r <= sem_e**2 / (9 * e_sem2_e) <= hi_r


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
