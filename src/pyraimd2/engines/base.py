"""Engine protocol: the supervised quantum-label boundary (design doc §4.1).

An engine turns an ``ase.Atoms`` into a label (energy, forces, stress) in ASE
units.  Engines raise :class:`EngineError` on any failure — they never return
partial or stale results (design doc §3, rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ase import Atoms


class EngineError(RuntimeError):
    """Raised when an engine cannot deliver a valid label."""


@dataclass(frozen=True)
class EngineResult:
    """One engine label, ASE units throughout.

    Attributes:
        energy: Total energy in eV.
        forces: (N, 3) forces in eV/Å.
        stress: (6,) stress in eV/Å³ in ASE Voigt order (xx, yy, zz, yz, xz,
            xy), or None when the system is non-periodic.
        wall_time_s: Wall-clock seconds spent inside the engine.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    wall_time_s: float


class Engine(Protocol):
    """A swappable quantum-chemistry backend (design doc §4.1)."""

    name: str

    def compute(self, atoms: Atoms) -> EngineResult:
        """Return the label for ``atoms`` or raise :class:`EngineError`."""
        ...
