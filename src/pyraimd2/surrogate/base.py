"""Surrogate protocol: the foundation-model slot (design doc §4.1).

The surrogate is model-agnostic; Phase A will host a learned energy functional
behind this same interface (design doc §11).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult


@dataclass(frozen=True)
class SurrogatePrediction:
    """One surrogate prediction, ASE units throughout.

    Attributes:
        energy: Total energy in eV.
        forces: (N, 3) forces in eV/Å.
        stress: (6,) stress in eV/Å³ in ASE Voigt order, or None when the
            system is non-periodic.
        uncertainty: (N,) per-atom uncertainty estimate.  NaN when the model
            cannot provide one — a single frozen model has no honest spread,
            so it must say so rather than invent a number.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    uncertainty: np.ndarray


class Surrogate(Protocol):
    """A fast, differentiable stand-in for the engine."""

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        """Return the prediction for ``atoms``."""
        ...


@dataclass(frozen=True)
class TrainReport:
    """Outcome of one :meth:`TrainableSurrogate.finetune` call.

    Attributes:
        n_labels: Number of (atoms, label) pairs fine-tuned on.
        n_epochs: Epochs each member was trained for.
        initial_loss: Mean over members of the pre-update (epoch 0) loss.
        final_loss: Mean over members of the last-epoch loss.
        member_losses: Per-member last-epoch losses.
        wall_time_s: Wall-clock seconds spent fine-tuning.
    """

    n_labels: int
    n_epochs: int
    initial_loss: float
    final_loss: float
    member_losses: tuple[float, ...]
    wall_time_s: float


class TrainableSurrogate(Surrogate, Protocol):
    """A surrogate that can be fine-tuned online on accumulated labels
    (design doc §4.1: ``finetune(batch) -> TrainReport``)."""

    def finetune(self, labels: Iterable[tuple[Atoms, EngineResult]]) -> TrainReport:
        """Fine-tune on ``labels`` = (atoms, engine result) pairs; raise on
        failure — never silently train nothing."""
        ...
