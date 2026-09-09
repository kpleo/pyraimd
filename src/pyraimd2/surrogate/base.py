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
) -> str:
    """Fail fast when declared energy conventions cannot be combined.

    The energetic anchor bookkeeping needs the *reference* scalar potential
    to be consistent with the reference forces, and the surrogate to be
    consistent with its own potential — it does not need the two
    ``energy_kind`` strings to be equal (INDEPENDENT_REVIEW_20260909 §5:
    d/dt[K+Phi_R] = R_dot·(F_drive−F_R) puts a per-side consistency
    requirement, not a naming requirement).  Accordingly:

    - Any declared ``force_consistent=False`` or ``forces_conservative=False``
      is rejected, on either side, always.
    - Both sides declaring *different* energy kinds (e.g. QE metallic
      ``free_energy`` with a MACE ``energy``) are allowed only when both
      sides strictly declare ``force_consistent=True`` and
      ``forces_conservative=True``: a cross-kind combination may not rest on
      an unknown ("不得把 unknown 当作已验证").
    - Undeclared (unknown) fields otherwise pass and stay recorded as
      unknown; they prove nothing either way and no support claim may rest
      on them.

    Any declared conflict raises :class:`CapabilityMismatchError` before the
    first SCF or inference call.  Returns the combination mode:
    ``"same_kind"``, ``"cross_kind"`` (both declared, different kinds, both
    strictly consistent) or ``"unknown"`` (something undeclared).
    """
    problems: list[str] = []
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
    kinds = {engine_caps.energy_kind, surrogate_caps.energy_kind}
    cross_kind = (
        EnergyKind.UNKNOWN not in kinds
        and engine_caps.energy_kind != surrogate_caps.energy_kind
    )
    if cross_kind:
        for side, caps in (("engine", engine_caps), ("surrogate", surrogate_caps)):
            if caps.force_consistent is not True \
                    or caps.forces_conservative is not True:
                problems.append(
                    f"{side} does not strictly declare force/energy consistency "
                    f"(force_consistent={caps.force_consistent!r}, "
                    f"forces_conservative={caps.forces_conservative!r}): a "
                    f"cross-kind combination ({engine_caps.energy_kind.value!r} "
                    f"x {surrogate_caps.energy_kind.value!r}) requires both "
                    "sides verified, not merely un-rejected"
                )
    if problems:
        raise CapabilityMismatchError("incompatible energy contract: " + "; ".join(problems))
    if cross_kind:
        return "cross_kind"
    if EnergyKind.UNKNOWN in kinds:
        return "unknown"
    return "same_kind"


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
