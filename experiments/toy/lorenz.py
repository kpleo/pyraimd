"""Lorenz-63 oracle engine for the PYRAIMD-2 toy port (T1-T5 spec:
docs/research-log.md, 玩具系统运输判据预注册, 2026-08-29).

The engine plays the DFT role: an exact (microsecond) oracle for the true
vector field of Lorenz-63 (σ=10, β=8/3, ρ set by the regime).  One toy
"MD step" is one RK4 macro-step of 0.01 time units (2 substeps of
dt=0.005); the drift dial (T3) multiplies the substep count of *accepted*
steps, scaling the unlabeled integration time per accepted decision while
the per-substep integration accuracy is unchanged.

``verify_rk4`` checks the integrator the way an engine-convergence test
checks a DFT protocol: max |error| of one macro-step against a
dt-converged reference (64x substeps) must stay < 1e-6.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SIGMA = 10.0
BETA = 8.0 / 3.0
RHO_REGIME_A = 28.0  # training distribution (committee prior lives here)
RHO_REGIME_B = 35.0  # deployment shift (online labels only, no re-pretrain)

DT_SUB = 0.005  # RK4 substep in Lorenz time units
N_SUB_BASE = 2  # substeps per macro-step at dial 1x -> 0.01 time units/"MD step"

# Attractor warm-up for initial states / pretraining trajectories: 20 time
# units is several lobe-switch timescales, so transients are gone.
BURN_IN_MACRO_STEPS = 2000


def lorenz_rhs(state: np.ndarray, rho: float) -> np.ndarray:
    """The exact Lorenz-63 vector field at ``state`` = (x, y, z)."""
    x, y, z = state
    return np.array(
        [SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z], dtype=float
    )


def rk4_step(
    rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray, dt: float
) -> np.ndarray:
    """One classical RK4 step of ``dt`` under the vector field ``rhs``."""
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class LorenzEngine:
    """The oracle: exact Lorenz-63 field evaluations and RK4 integration.

    ``calls`` counts label evaluations, mirroring the engine call counters
    of the MD drivers (oracle cost accounting for the speedup narrative).
    """

    name = "lorenz63-oracle"

    def __init__(self, rho: float) -> None:
        if rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {rho}")
        self.rho = float(rho)
        self.calls = 0

    def rhs(self, state: np.ndarray) -> np.ndarray:
        return lorenz_rhs(np.asarray(state, dtype=float), self.rho)

    def label(self, state: np.ndarray) -> np.ndarray:
        """The oracle label: the true vector field at ``state``."""
        self.calls += 1
        return self.rhs(state)

    def macro_step(self, state: np.ndarray, n_sub: int = N_SUB_BASE) -> np.ndarray:
        """Advance ``n_sub`` RK4 substeps of DT_SUB (n_sub=2 -> one "MD step")."""
        if n_sub < 1:
            raise ValueError(f"n_sub must be >= 1, got {n_sub}")
        state = np.asarray(state, dtype=float)
        for _ in range(n_sub):
            state = rk4_step(self.rhs, state, DT_SUB)
        return state

    def on_attractor_state(self, seed: int, burn_in: int = BURN_IN_MACRO_STEPS) -> np.ndarray:
        """A deterministic point on this regime's attractor: (1,1,1) plus a
        seeded perturbation, burned in for ``burn_in`` macro-steps."""
        rng = np.random.default_rng(seed)
        state = np.array([1.0, 1.0, 1.0]) + rng.normal(0.0, 0.1, size=3)
        for _ in range(burn_in):
            state = self.macro_step(state)
        return state

    def attractor_samples(
        self, n: int, seed: int, stride: int = 2, burn_in: int = BURN_IN_MACRO_STEPS
    ) -> tuple[np.ndarray, np.ndarray]:
        """``n`` (state, true field) pairs sampled along the attractor.

        Used for the committee's foundation prior (regime A, rho=28) and for
        offline calibration rolls.  ``stride`` thins the trajectory so the
        samples cover the attractor instead of one short arc.
        """
        states = np.empty((n, 3))
        fields = np.empty((n, 3))
        state = self.on_attractor_state(seed, burn_in)
        for i in range(n):
            for _ in range(stride):
                state = self.macro_step(state)
            states[i] = state
            fields[i] = self.rhs(state)  # not a "label" — offline sampling
        return states, fields


def verify_rk4(
    states: np.ndarray, rho: float = RHO_REGIME_A, n_sub: int = N_SUB_BASE, ref_mult: int = 64
) -> float:
    """Max |error| of one macro-step vs a dt-converged RK4 reference.

    Both integrations cover the SAME total time (n_sub x DT_SUB); the
    reference refines the substep by ``ref_mult`` (dt = 0.005/64), whose own
    error is ~ref_mult^4 smaller than the coarse one — negligible next to
    it.  Spec gate: < 1e-6 over one macro-step.
    """
    engine = LorenzEngine(rho)
    worst = 0.0
    for state in np.asarray(states, dtype=float).reshape(-1, 3):
        coarse = np.asarray(state, dtype=float)
        reference = np.asarray(state, dtype=float)
        for _ in range(n_sub):
            coarse = rk4_step(engine.rhs, coarse, DT_SUB)
        for _ in range(n_sub * ref_mult):
            reference = rk4_step(engine.rhs, reference, DT_SUB / ref_mult)
        worst = max(worst, float(np.max(np.abs(coarse - reference))))
    return worst
