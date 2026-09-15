"""Static conservative corrections wrapped around a base surrogate (D1).

Two wrappers adapt a frozen base model to reference-level data without
retraining, and both return an energy and a force set that stay mutually
consistent, so the wrapped surrogate plugs into the ordinary NVE/NVT
workflows like any other :class:`~pyraimd2.surrogate.base.Surrogate`:

- :class:`ScaledSurrogate` multiplies energy and forces by one frozen
  scalar: ``U_c = c U_b``, ``F_c = c F_b``.  ``c`` is fixed at
  construction (calibration time) and is never updated while the
  corrected model is in use; scaling forces without scaling the energy
  (or vice versa) would break force consistency.
- :class:`QuadraticCorrectedSurrogate` adds the static quadratic Taylor
  correction of the reference-minus-base difference at one fixed center
  ``q0``: with the Cartesian displacement ``u = q - q0`` in a fixed atom
  order,

  ``U_c = U_b - dF0·u + 1/2 uᵀ dH u + E0``,
  ``F_c = F_b + dF0 - dH u``,

  where ``dF0 = F_r(q0) - F_b(q0)`` (eV/angstrom) and
  ``dH = H_r - H_b`` (eV/angstrom^2) are the force and Hessian
  differences at the reference center.

Units are ASE units throughout: coordinates in angstrom, energies in eV,
forces in eV/angstrom, correction Hessians in eV/angstrom^2.  ``dH`` is
a plain Cartesian force-constant difference — atomic masses are NOT
folded into it; mass weighting happens only when an analysis builds the
dynamical matrix, never inside the wrapper.

Stress is not implemented for either wrapper: capabilities declare
``stress_available=False`` and predictions return ``stress=None``.  The
base model's stress is never passed through as if it were a corrected
stress.
"""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike

from pyraimd2.runtime.identity import fingerprint_of
from pyraimd2.surrogate.base import (
    SurrogateCapabilities,
    SurrogatePrediction,
    surrogate_capabilities,
)


def _finite_array(value: ArrayLike, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    result = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positions(value: ArrayLike, name: str) -> np.ndarray:
    result = _finite_array(value, name)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3) with N > 0")
    return result


def _immutable(value: np.ndarray) -> np.ndarray:
    # A bytes-backed array owns its data and cannot be made writable again.
    return np.frombuffer(value.tobytes(order="C"), dtype=float).reshape(value.shape)


def _base_identity(base: object) -> str:
    return fingerprint_of(base) or type(base).__qualname__


def _resolve_base(base: object) -> object:
    """A surrogate instance, or a ``{"name": ..., "kwargs": {...}}`` registry
    spec naming the base surrogate backend."""
    if callable(getattr(base, "predict", None)):
        return base
    if isinstance(base, dict):
        unknown = set(base) - {"name", "kwargs"}
        if unknown:
            raise TypeError(
                f"unknown base spec fields {sorted(unknown)}; a base spec has "
                "'name' and optional 'kwargs' only"
            )
        name = base.get("name")
        if not isinstance(name, str):
            raise TypeError("a base spec requires the backend 'name' string")
        kwargs = base.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise TypeError("base spec 'kwargs' must be a mapping of factory options")
        from pyraimd2.backends.registry import create_backend  # local: lazy loading

        return create_backend(name, kind="surrogate", **kwargs)
    raise TypeError(
        f"base must be a surrogate instance or a {{'name', 'kwargs'}} backend "
        f"spec, got {type(base).__name__}"
    )


class ScaledSurrogate:
    """Energy and forces of a base surrogate rescaled by one frozen scalar.

    ``U_c = c U_b``, ``F_c = c F_b``.  ``c`` must be positive and finite;
    it is frozen at construction and there is deliberately no update path.
    The scale and its provenance string (``calibration_note`` — where the
    number came from, e.g. the calibration set and fit) enter the
    fingerprint together with the base model's identity.

    Scaling by a positive constant preserves the base model's
    energy/force consistency and conservativeness exactly, so those
    declarations are mirrored; an honest base uncertainty spread scales
    with the forces (``sigma_c = c sigma_b``).  Stress is not exposed:
    ``stress_available=False`` and predictions return ``stress=None``.
    """

    def __init__(self, base: object, scale: float, *,
                 calibration_note: str = "") -> None:
        if not callable(getattr(base, "predict", None)):
            raise TypeError(
                f"base must provide the surrogate protocol (predict), got "
                f"{type(base).__name__}"
            )
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale must be > 0 and finite, got {scale!r}")
        self._base = base
        self.scale = scale
        self.calibration_note = str(calibration_note)

    @property
    def capabilities(self) -> SurrogateCapabilities:
        base_caps = surrogate_capabilities(self._base)
        return SurrogateCapabilities(
            energy_kind=base_caps.energy_kind,
            force_consistent=base_caps.force_consistent,
            forces_conservative=base_caps.forces_conservative,
            stress_available=False,  # no corrected stress; never impersonated
            uncertainty_available=base_caps.uncertainty_available,
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack("<d", self.scale))
        digest.update(self.calibration_note.encode("utf-8"))
        return f"scaled:{digest.hexdigest()[:16]}:{_base_identity(self._base)}"

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        base = self._base.predict(atoms)
        return SurrogatePrediction(
            energy=self.scale * base.energy,
            forces=self.scale * base.forces,
            stress=None,  # scaled stress is not exposed in this version
            uncertainty=self.scale * base.uncertainty,
            energy_kind=base.energy_kind,
            force_consistent=base.force_consistent,
        )


class QuadraticCorrectedSurrogate:
    """Static quadratic reference-minus-base correction around a fixed center.

    With the displacement ``u = q - q0`` from the fixed reference center
    (flattened atom-major: ``u[3*i + alpha]`` is Cartesian component
    ``alpha`` of atom ``i``, matching the (3N, 3N) ``delta_h`` indexing):

    ``U_c = U_b - dF0·u + 1/2 uᵀ dH u + E0``
    ``F_c = F_b + dF0 - dH u``

    ``delta_h`` is symmetrized at construction; the Frobenius residuals
    before and after symmetrization are recorded
    (``symmetrization_residual_before`` / ``_after``).  The translation
    constraint is *checked*, never silently enforced:
    ``translation_hessian_residuals`` holds ``|dH·t|`` for the three unit
    global-translation vectors ``t`` and ``translation_force_residuals``
    the per-component magnitude of the net force ``sum(dF0)`` — both are
    ~0 for a translationally consistent correction.

    The displacement uses the fixed-reference Cartesian expansion.  For
    periodic systems the per-atom periodic image is chosen once — by the
    fractional minimum-image convention at the first ``predict`` call —
    and frozen together with the cell for the rest of the wrapper's life;
    images are never re-selected per step (a jumping nearest image would
    break the expansion), and a changed cell raises.  Callers must pass
    continuous (unwrapped) positions on one consistent branch, as the
    dynamics loops already guarantee, and the same fixed atom order as
    ``q0``; only the atom count is validated.

    ``energy_offset`` is the recorded constant zero point in eV: use
    ``U_r(q0) - U_b(q0)`` to anchor the corrected energy to the reference
    at ``q0``; the default 0 keeps the base model's value there
    (``U_c(q0) = U_b(q0)``).  A constant changes no forces.

    The correction terms are exactly force-consistent by construction, so
    the base model's energy/force consistency declarations carry over
    unchanged; the deterministic, member-independent correction leaves an
    honest base spread unchanged, so the base uncertainty is forwarded
    as-is.  Stress is not implemented: ``stress_available=False`` and
    predictions return ``stress=None``, never the base model's stress.
    The content hashes of ``q0``, ``delta_f0``, the symmetrized
    ``delta_h``, the energy offset and the calibration note enter the
    fingerprint together with the base model's identity.
    """

    def __init__(
        self,
        base: object,
        q0: ArrayLike,
        delta_f0: ArrayLike,
        delta_h: ArrayLike,
        *,
        energy_offset: float = 0.0,
        calibration_note: str = "",
    ) -> None:
        if not callable(getattr(base, "predict", None)):
            raise TypeError(
                f"base must provide the surrogate protocol (predict), got "
                f"{type(base).__name__}"
            )
        q0_checked = _positions(q0, "q0")
        n_atoms = len(q0_checked)
        delta_f0_checked = _positions(delta_f0, "delta_f0")
        if delta_f0_checked.shape != q0_checked.shape:
            raise ValueError(
                f"delta_f0 must have shape {q0_checked.shape}, got "
                f"{delta_f0_checked.shape}"
            )
        dof = 3 * n_atoms
        delta_h_checked = _finite_array(delta_h, "delta_h")
        if delta_h_checked.shape != (dof, dof):
            raise ValueError(
                f"delta_h must have shape ({dof}, {dof}) for {n_atoms} atoms, "
                f"got {delta_h_checked.shape}"
            )
        energy_offset = float(energy_offset)
        if not math.isfinite(energy_offset):
            raise ValueError(f"energy_offset must be finite, got {energy_offset!r}")
        self._base = base
        self.energy_offset = energy_offset
        self.calibration_note = str(calibration_note)
        self.symmetrization_residual_before = float(
            np.linalg.norm(delta_h_checked - delta_h_checked.T)
        )
        symmetrized = 0.5 * (delta_h_checked + delta_h_checked.T)
        self.symmetrization_residual_after = float(
            np.linalg.norm(symmetrized - symmetrized.T)
        )
        translation_hessian_residuals = []
        for alpha in range(3):
            translation = np.zeros((n_atoms, 3))
            translation[:, alpha] = 1.0 / math.sqrt(n_atoms)  # unit 3N vector
            translation_hessian_residuals.append(
                float(np.linalg.norm(symmetrized @ translation.reshape(-1)))
            )
        self.translation_hessian_residuals = tuple(translation_hessian_residuals)
        self.translation_force_residuals = tuple(
            float(abs(component)) for component in delta_f0_checked.sum(axis=0)
        )
        self._q0 = _immutable(q0_checked)
        self._delta_f0 = _immutable(delta_f0_checked)
        self._delta_h = _immutable(symmetrized)
        self._image_offsets: np.ndarray | None = None
        self._chart_cell: np.ndarray | None = None
        self._chart_inv_cell: np.ndarray | None = None

    @property
    def q0(self) -> np.ndarray:
        """The frozen reference center, (N, 3) angstrom, read-only."""
        return self._q0

    @property
    def delta_f0(self) -> np.ndarray:
        """The frozen force difference ``F_r(q0) - F_b(q0)``, (N, 3) eV/angstrom."""
        return self._delta_f0

    @property
    def delta_h(self) -> np.ndarray:
        """The symmetrized Hessian difference, (3N, 3N) eV/angstrom^2,
        atom-major Cartesian, no mass weighting, read-only."""
        return self._delta_h

    @property
    def capabilities(self) -> SurrogateCapabilities:
        base_caps = surrogate_capabilities(self._base)
        return SurrogateCapabilities(
            energy_kind=base_caps.energy_kind,
            force_consistent=base_caps.force_consistent,
            forces_conservative=base_caps.forces_conservative,
            stress_available=False,  # corrected stress not implemented; never impersonated
            uncertainty_available=base_caps.uncertainty_available,
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, content in (
            ("q0", self._q0),
            ("delta_f0", self._delta_f0),
            ("delta_h", self._delta_h),
        ):
            digest.update(name.encode("utf-8"))
            digest.update(str(content.shape).encode("utf-8"))
            digest.update(content.tobytes(order="C"))
        digest.update(struct.pack("<d", self.energy_offset))
        digest.update(self.calibration_note.encode("utf-8"))
        return (
            f"quadratic-corrected:{digest.hexdigest()[:16]}:"
            f"{_base_identity(self._base)}"
        )

    def _displacement(self, atoms: Atoms) -> np.ndarray:
        delta = np.asarray(atoms.get_positions(), dtype=float) - self._q0
        if not np.any(atoms.pbc):
            return delta
        cell = np.asarray(atoms.cell, dtype=float)
        if self._image_offsets is None:
            # Freeze the coordinate chart once: this call's fractional
            # minimum image fixes every atom's periodic image (and the cell
            # the expansion lives on) for the wrapper's remaining life.
            try:
                inv_cell = np.linalg.inv(cell)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "periodic correction chart needs an invertible cell, got "
                    f"{cell.tolist()}"
                ) from None
            self._chart_cell = cell.copy()
            self._chart_inv_cell = inv_cell
            self._image_offsets = np.round(delta @ inv_cell)
        elif not np.allclose(cell, self._chart_cell, rtol=0.0, atol=1e-10):
            raise ValueError(
                "the correction chart assumes a fixed cell; the cell changed "
                "after the image offsets were frozen"
            )
        fractional = delta @ self._chart_inv_cell - self._image_offsets
        return fractional @ self._chart_cell

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        if len(atoms) != len(self._q0):
            raise ValueError(
                f"the correction map is defined for {len(self._q0)} atoms in "
                f"one fixed order, got {len(atoms)}"
            )
        base = self._base.predict(atoms)
        u = self._displacement(atoms).reshape(-1)  # atom-major, matching delta_h
        correction_force = (self._delta_h @ u).reshape(-1, 3)
        return SurrogatePrediction(
            energy=(
                base.energy
                - float(self._delta_f0.reshape(-1) @ u)
                + 0.5 * float(u @ self._delta_h @ u)
                + self.energy_offset
            ),
            forces=base.forces + self._delta_f0 - correction_force,
            stress=None,  # corrected stress not implemented; never impersonated
            uncertainty=base.uncertainty,
            energy_kind=base.energy_kind,
            force_consistent=base.force_consistent,
        )


def scaled_factory(*, base: object, scale: float,
                   calibration_note: str = "") -> ScaledSurrogate:
    """Registry factory for :class:`ScaledSurrogate`.

    ``base`` is a surrogate instance or a ``{"name": ..., "kwargs": {...}}``
    backend spec; unknown fields are rejected by the signature itself.
    """
    return ScaledSurrogate(
        _resolve_base(base), scale, calibration_note=calibration_note
    )


scaled_factory.backend_kind = "surrogate"


def quadratic_corrected_factory(
    *,
    base: object,
    q0: ArrayLike,
    delta_f0: ArrayLike,
    delta_h: ArrayLike,
    energy_offset: float = 0.0,
    calibration_note: str = "",
) -> QuadraticCorrectedSurrogate:
    """Registry factory for :class:`QuadraticCorrectedSurrogate`.

    ``base`` is a surrogate instance or a ``{"name": ..., "kwargs": {...}}``
    backend spec; unknown fields are rejected by the signature itself.
    """
    return QuadraticCorrectedSurrogate(
        _resolve_base(base),
        q0,
        delta_f0,
        delta_h,
        energy_offset=energy_offset,
        calibration_note=calibration_note,
    )


quadratic_corrected_factory.backend_kind = "surrogate"
