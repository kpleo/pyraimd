"""Engine protocol: the supervised quantum-label boundary.

An engine turns an ``ase.Atoms`` into a label (energy, forces, stress) in ASE
units.  Engines raise :class:`EngineError` on any failure — they never return
partial or stale results.

Result metadata and capabilities (WP01): every label states which energy
quantity it reports (:class:`EnergyKind`) and whether the reported forces are
the negative gradient of that energy (``force_consistent``).  Backends declare
their :class:`EngineCapabilities`; anything a backend does not know must read
as unknown/unavailable, never as supported.  A unit or energy-convention
mismatch raises :class:`CapabilityMismatchError` before any expensive
computation (SCF or inference), never after.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from ase import Atoms


class EngineError(RuntimeError):
    """Raised when an engine cannot deliver a valid label."""


class CapabilityMismatchError(ValueError):
    """Declared backend contracts cannot be combined into one calculation.

    Raised before any expensive computation so a unit or energy-convention
    mismatch never burns a reference call or a model inference.
    """


class EnergyKind(StrEnum):
    """Which scalar a backend reports as its "energy"."""

    ENERGY = "energy"  # potential/total energy
    FREE_ENERGY = "free_energy"  # force-consistent free energy (DFT smearing)
    UNKNOWN = "unknown"  # not declared — must not be treated as either


def _energy_kind(value: str) -> EnergyKind:
    try:
        return EnergyKind(value)
    except ValueError:
        raise ValueError(
            f"energy_kind must be one of {[kind.value for kind in EnergyKind]}, "
            f"got {value!r}"
        ) from None


def _tri_state(value: bool | None, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True, False or None (unknown)")
    return value


@dataclass(frozen=True)
class EngineResult:
    """One engine label, ASE units throughout.

    Attributes:
        energy: Total energy in eV.
        forces: (N, 3) forces in eV/Å.
        stress: (6,) stress in eV/Å³ in ASE Voigt order (xx, yy, zz, yz, xz,
            xy), or None when the system is non-periodic.
        wall_time_s: Wall-clock seconds spent inside the engine.
        energy_kind: Which quantity ``energy`` reports ("energy",
            "free_energy" or "unknown").  Defaults to "unknown" so legacy
            positional construction never claims a convention it did not
            declare.
        force_consistent: True when ``forces`` are the negative gradient of
            the reported ``energy``, False when explicitly not, None (default)
            when the backend does not say.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    wall_time_s: float
    energy_kind: str = EnergyKind.UNKNOWN
    force_consistent: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_kind", _energy_kind(self.energy_kind))
        object.__setattr__(
            self, "force_consistent", _tri_state(self.force_consistent, "force_consistent")
        )


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine can deliver; undeclared means unknown/unavailable.

    Attributes:
        energy_kind: Energy convention of reported energies.
        force_consistent: Tri-state — True/False/None(unknown): whether the
            reported forces differentiate the reported energy.
        forces_conservative: Tri-state: whether forces derive from a
            conservative potential.  Non-conservative forces disqualify
            endpoint work and directional-coefficient interpretations.
        stress_available: Whether the engine returns a stress tensor.  False
            means unavailable *or undeclared* — plan as if absent.
    """

    energy_kind: str = EnergyKind.UNKNOWN
    force_consistent: bool | None = None
    forces_conservative: bool | None = None
    stress_available: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_kind", _energy_kind(self.energy_kind))
        object.__setattr__(
            self, "force_consistent", _tri_state(self.force_consistent, "force_consistent")
        )
        object.__setattr__(
            self,
            "forces_conservative",
            _tri_state(self.forces_conservative, "forces_conservative"),
        )
        object.__setattr__(self, "stress_available", bool(self.stress_available))


def engine_capabilities(engine: object) -> EngineCapabilities:
    """The engine's declared capabilities, all-unknown when undeclared.

    A backend without a ``capabilities`` attribute is not assumed to support
    anything: every field reads unknown/unavailable.
    """
    caps = getattr(engine, "capabilities", None)
    if caps is None:
        return EngineCapabilities()
    if not isinstance(caps, EngineCapabilities):
        raise TypeError(
            f"engine capabilities must be EngineCapabilities, got {type(caps).__name__}"
        )
    return caps


class Engine(Protocol):
    """A swappable quantum-chemistry backend.

    ``capabilities`` is optional at runtime; consumers must read it through
    :func:`engine_capabilities`, which maps an undeclared attribute to
    all-unknown rather than assuming support.
    """

    name: str
    capabilities: EngineCapabilities

    def compute(self, atoms: Atoms) -> EngineResult:
        """Return the label for ``atoms`` or raise :class:`EngineError`."""
        ...
