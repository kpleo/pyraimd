"""Numeric label cache keys (§5.4) — deliberately separate from decision caching.

The numeric label cache answers: *the same configuration under the same
backend and reference settings may reuse an energy/forces label.*  It never
reuses a decision, never draws a check, and is consulted only where the
contract allows (independent verification of an accepted evaluation whose
reference settings, geometry and energy convention match exactly).

Keys use exact normalized values and a hash — never geometric proximity.
Included: atomic numbers, positions, cell, PBC, initial charges and initial
magnetic moments, the reference-settings identity (compute parameters and
pseudopotential/model identity via the WP01 fingerprint), the reported
energy convention, and the requested properties.  Masses are isotope-
independent for electronic labels and velocities are irrelevant for
conservative forces, so neither enters the key.

Backends whose external state is opaque and cannot be reliably identified
(no declared fingerprint) get NO cross-evaluation label cache by default.
"""

from __future__ import annotations

import hashlib

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EnergyKind, EngineResult

_LABEL_KEY_PREFIX = b"pyraimd2-label-v1\0"
_INPUT_HASH_PREFIX = b"pyraimd2-input-v1\0"


def _feed(hasher: hashlib._Hash, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(name.encode())
    hasher.update(str(contiguous.dtype).encode())
    hasher.update(str(contiguous.shape).encode())
    hasher.update(contiguous.tobytes())


def label_key(
    atoms: Atoms,
    reference_id: str,
    energy_kind: str = EnergyKind.UNKNOWN,
    properties: tuple[str, ...] = ("energy", "forces"),
) -> str:
    """Exact-match key for one numeric label (geometry + settings + convention)."""
    hasher = hashlib.sha256()
    hasher.update(_LABEL_KEY_PREFIX)
    _feed(hasher, "numbers", atoms.numbers)
    _feed(hasher, "positions", atoms.positions)
    _feed(hasher, "cell", atoms.cell.array)
    _feed(hasher, "pbc", np.asarray(atoms.pbc))
    _feed(hasher, "initial_charges", atoms.get_initial_charges())
    _feed(hasher, "initial_magmoms", atoms.get_initial_magnetic_moments())
    hasher.update(str(reference_id).encode())
    hasher.update(str(EnergyKind(energy_kind)).encode())
    hasher.update(",".join(properties).encode())
    return hasher.hexdigest()


def atoms_input_hash(atoms: Atoms) -> str:
    """Content hash of the full physical input, including masses and momenta.

    Recorded in run identity (manifest/run_start records) — unlike a label
    key this must distinguish everything that makes the *dynamical* input
    different.
    """
    hasher = hashlib.sha256()
    hasher.update(_INPUT_HASH_PREFIX)
    _feed(hasher, "numbers", atoms.numbers)
    _feed(hasher, "positions", atoms.positions)
    _feed(hasher, "cell", atoms.cell.array)
    _feed(hasher, "pbc", np.asarray(atoms.pbc))
    _feed(hasher, "masses", atoms.get_masses())
    _feed(hasher, "momenta", atoms.get_momenta())
    _feed(hasher, "initial_charges", atoms.get_initial_charges())
    _feed(hasher, "initial_magmoms", atoms.get_initial_magnetic_moments())
    return hasher.hexdigest()


class LabelCache:
    """In-memory cross-evaluation label cache for one run.

    Disabled unless the reference backend declares a fingerprint: an engine
    whose settings cannot be reliably identified must not share labels across
    evaluations.  Lookups and insertions are by exact key only.
    """

    def __init__(self, reference_id: str | None, *, enabled: bool = True) -> None:
        self.reference_id = reference_id
        self.enabled = bool(enabled) and reference_id is not None
        self._entries: dict[str, tuple[EngineResult, str]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self,
        atoms: Atoms,
        energy_kind: str = EnergyKind.UNKNOWN,
        properties: tuple[str, ...] = ("energy", "forces"),
    ) -> tuple[EngineResult, str] | None:
        """Return ``(label, label_id)`` on an exact hit, else None."""
        if not self.enabled:
            return None
        return self._entries.get(label_key(atoms, self.reference_id, energy_kind, properties))

    def put(
        self,
        atoms: Atoms,
        result: EngineResult,
        label_id: str,
        properties: tuple[str, ...] = ("energy", "forces"),
    ) -> None:
        if not self.enabled:
            return
        key = label_key(atoms, self.reference_id, result.energy_kind, properties)
        self._entries.setdefault(key, (result, label_id))
