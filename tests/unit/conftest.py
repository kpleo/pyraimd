"""Hermetic fakes driving all loop/switch/store unit logic.

No quantum code, no torch, no RNG: the fakes are deterministic analytic
potentials, so restart equality can be asserted with atol=0.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.surrogate.base import SurrogatePrediction, TrainReport

# Fixed 4-atom cluster geometry (Å) and small initial momenta — no RNG
# anywhere in the unit tests.
CLUSTER_POSITIONS = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.92, 0.08, 0.01],
        [0.10, 0.88, 0.21],
        [0.18, 0.12, 0.94],
    ]
)
CLUSTER_MOMENTA = np.array(
    [
        [0.010, -0.020, 0.015],
        [-0.012, 0.008, 0.011],
        [0.006, 0.014, -0.009],
        [-0.004, -0.002, -0.017],
    ]
)
# Harmonic reference point, offset from the cluster so forces are nonzero.
CLUSTER_R0 = CLUSTER_POSITIONS * 0.9 + 0.05


class FakeEngine:
    """Analytic harmonic label source: E = 1/2 k |r - r0|², F = -k (r - r0).

    Deterministic, counts successful calls, and can be told to fail — in
    which case :class:`EngineError` is raised and the call is not counted.
    """

    name = "fake-engine"

    def __init__(self, r0: np.ndarray, k: float = 1.0, fail: bool = False) -> None:
        self.r0 = np.asarray(r0, dtype=float)
        self.k = float(k)
        self.fail = fail
        self.calls = 0

    def compute(self, atoms: Atoms) -> EngineResult:
        if self.fail:
            raise EngineError("FakeEngine was told to fail")
        self.calls += 1
        dr = atoms.get_positions() - self.r0
        return EngineResult(
            energy=0.5 * self.k * float((dr**2).sum()),
            forces=-self.k * dr,
            stress=None,
            wall_time_s=0.0,
        )


class FakeSurrogate:
    """Same harmonic potential plus a deterministic sinusoidal force bias,
    so the surrogate-vs-engine shadow error is nonzero and controllable."""

    def __init__(self, r0: np.ndarray, k: float = 1.0, bias_amplitude: float = 0.0) -> None:
        self.r0 = np.asarray(r0, dtype=float)
        self.k = float(k)
        self.bias_amplitude = float(bias_amplitude)
        self.calls = 0

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        self.calls += 1
        dr = atoms.get_positions() - self.r0
        forces = -self.k * dr + self.bias_amplitude * np.sin(dr)
        return SurrogatePrediction(
            energy=0.5 * self.k * float((dr**2).sum()),
            forces=forces,
            stress=None,
            uncertainty=np.full(len(atoms), np.nan),  # honest: no spread available
        )


class FakeCommittee(FakeSurrogate):
    """Committee analog: controllable constant per-atom spread, a finetune
    counter, and an ``improve`` factor that shrinks the force bias after
    every fine-tune so tests can simulate a committee that learns."""

    def __init__(
        self,
        r0: np.ndarray,
        k: float = 1.0,
        bias_amplitude: float = 0.0,
        spread: float = 0.0,
        improve: float = 1.0,
    ) -> None:
        super().__init__(r0, k=k, bias_amplitude=bias_amplitude)
        self.spread = float(spread)
        self.improve = float(improve)
        self.finetune_calls = 0
        self.finetune_sizes: list[int] = []

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        prediction = super().predict(atoms)
        return SurrogatePrediction(
            energy=prediction.energy,
            forces=prediction.forces,
            stress=prediction.stress,
            uncertainty=np.full(len(atoms), self.spread),  # honest, finite spread
        )

    def finetune(self, labels) -> TrainReport:
        label_list = list(labels)
        if not label_list:
            raise ValueError("finetune needs at least one (atoms, label) pair")
        self.finetune_calls += 1
        self.finetune_sizes.append(len(label_list))
        self.bias_amplitude *= self.improve
        return TrainReport(
            n_labels=len(label_list),
            n_epochs=50,
            initial_loss=1.0,
            final_loss=0.5,
            member_losses=(0.5,),
            wall_time_s=0.0,
        )


@pytest.fixture
def cluster() -> Atoms:
    """A fresh 4-atom cluster with momenta already set (no thermalization,
    so trajectories are exactly reproducible)."""
    atoms = Atoms("H4", positions=CLUSTER_POSITIONS.copy())
    atoms.set_momenta(CLUSTER_MOMENTA.copy())
    return atoms
