"""Surrogate protocol: the foundation-model slot.

The surrogate interface accepts different models that supply compatible energies
and forces.  Predictions carry the same energy-convention metadata as engine
labels (:class:`~pyraimd2.engines.base.EnergyKind`, ``force_consistent``), and
surrogates declare :class:`SurrogateCapabilities` — including whether an honest
uncertainty estimate exists at all.  Undeclared capabilities read as
unknown/unavailable, never as supported.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import (
    CapabilityMismatchError,
    EnergyKind,
    EngineCapabilities,
    EngineResult,
    _energy_kind,
    _tri_state,
)


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
        energy_kind: Which quantity ``energy`` reports ("energy",
            "free_energy" or "unknown"); defaults to "unknown".
        force_consistent: True when ``forces`` are the negative gradient of
            the reported ``energy``, False when explicitly not, None (default)
            when the model does not say.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    uncertainty: np.ndarray
    energy_kind: str = EnergyKind.UNKNOWN
    force_consistent: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_kind", _energy_kind(self.energy_kind))
        object.__setattr__(
            self, "force_consistent", _tri_state(self.force_consistent, "force_consistent")
        )


@dataclass(frozen=True)
class SurrogateCapabilities(EngineCapabilities):
    """What a surrogate can deliver; undeclared means unknown/unavailable.

    Adds ``uncertainty_available`` to the shared backend contract.  False
    means the model cannot provide an honest spread (or does not declare
    one) — spread-dependent policies must treat it as absent.
    """

    uncertainty_available: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "uncertainty_available", bool(self.uncertainty_available))


def surrogate_capabilities(surrogate: object) -> SurrogateCapabilities:
    """The surrogate's declared capabilities, all-unknown when undeclared."""
    caps = getattr(surrogate, "capabilities", None)
    if caps is None:
        return SurrogateCapabilities()
    if not isinstance(caps, SurrogateCapabilities):
        raise TypeError(
            f"surrogate capabilities must be SurrogateCapabilities, "
            f"got {type(caps).__name__}"
        )
    return caps


def assert_compatible_energy_contract(
    engine_caps: EngineCapabilities, surrogate_caps: SurrogateCapabilities
) -> None:
    """Fail fast when declared energy conventions cannot be combined.

    The energetic anchor bookkeeping mixes surrogate and reference energies
    and forces.  It is only sound when both sides report the same energy
    quantity and that quantity is consistent with the reported forces.  Any
    *declared* conflict raises :class:`CapabilityMismatchError` here — before
    the first SCF or inference call.  Undeclared (unknown) fields do not
    prove a mismatch, so they pass and stay recorded as unknown.
    """
    problems: list[str] = []
    if (
        engine_caps.energy_kind != EnergyKind.UNKNOWN
        and surrogate_caps.energy_kind != EnergyKind.UNKNOWN
        and engine_caps.energy_kind != surrogate_caps.energy_kind
    ):
        problems.append(
            f"engine reports energy_kind={engine_caps.energy_kind.value!r} but the "
            f"surrogate reports {surrogate_caps.energy_kind.value!r}; anchored "
            "energies would mix different energy quantities"
        )
    for side, caps in (("engine", engine_caps), ("surrogate", surrogate_caps)):
        if caps.force_consistent is False:
            problems.append(
                f"{side} declares force_consistent=False: its reported energy is "
                "not the quantity its forces differentiate, so anchored energy "
                "and endpoint-work bookkeeping would be unsound"
            )
        if caps.forces_conservative is False:
            problems.append(
                f"{side} declares forces_conservative=False: endpoint work and "
                "directional coefficients require conservative forces"
            )
    if problems:
        raise CapabilityMismatchError("incompatible energy contract: " + "; ".join(problems))


class Surrogate(Protocol):
    """A fast, differentiable stand-in for the engine.

    ``capabilities`` is optional at runtime; consumers must read it through
    :func:`surrogate_capabilities`, which maps an undeclared attribute to
    all-unknown rather than assuming support.
    """

    capabilities: SurrogateCapabilities

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
    """A surrogate that can be fine-tuned online on accumulated labels."""

    def finetune(self, labels: Iterable[tuple[Atoms, EngineResult]]) -> TrainReport:
        """Fine-tune on ``labels`` = (atoms, engine result) pairs; raise on
        failure — never silently train nothing."""
        ...
