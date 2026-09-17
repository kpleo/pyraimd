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

Periodic chart contract: the pbc mask and the cell are recorded at the
first ``predict`` call (a non-periodic first call included) and every
later call must match them — a pbc change in either direction or a cell
change is refused.  The local validity domain is the half-cell
neighborhood around ``q0`` on each periodic axis: a rigid integer-cell
shift of the whole configuration is explicitly supported (uniform
per-atom offsets map it back identically), while any atom wrapping
relative to the others — a genuine exit from the local domain — raises
:class:`CorrectionDomainError` for the upper layer to handle (re-center
or refresh).  The wrapper never silently rounds onto another periodic
branch: there the correction energy would jump while its force stayed
smooth, an inconsistent energy/force pair under a conservative
declaration.  Nothing about the chart depends on call history or process
lifetime beyond the recorded environment, so a fresh-process resume
reproduces the same corrections under the same fingerprint.
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
from pathlib import Path

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


def _species_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise TypeError(
            "species must be a sequence of element symbols in the run's "
            f"fixed atom order, got {type(value).__name__}")
    species = tuple(str(item) for item in value)
    if not species or not all(species):
        raise ValueError("species must be a nonempty sequence of element "
                         "symbols in the run's fixed atom order")
    return species


def _species_digest(species: tuple[str, ...]) -> bytes:
    """The species identity bytes for the fingerprint: JSON-joined, so
    adjacent symbols can never alias (['S','e'] vs ['Se'])."""
    import json

    return json.dumps(list(species)).encode("utf-8")


def _translation_vectors(n_atoms: int) -> np.ndarray:
    """The three uniform translation directions of the 3N configuration
    space as (3N, 3) unit-norm columns."""
    translations = np.zeros((3 * n_atoms, 3))
    for alpha in range(3):
        translations[alpha::3, alpha] = 1.0 / math.sqrt(n_atoms)
    return translations


def _translation_residuals(
    hessian: np.ndarray, translations: np.ndarray
) -> tuple[float, float, float]:
    """``|dH·t|`` for each of the three unit translation vectors."""
    return tuple(
        float(np.linalg.norm(hessian @ translations[:, alpha])) for alpha in range(3)
    )


def _base_identity(base: object) -> str:
    return fingerprint_of(base) or type(base).__qualname__


class CorrectionDomainError(ValueError):
    """The configuration left the correction's local validity domain (an
    atom crossed the half-cell boundary around q0 on a periodic axis
    relative to the others).  A controlled refusal for the upper layer to
    handle — re-center or refresh the correction; the wrapper never
    silently rounds onto another periodic branch and returns an
    inconsistent energy/force pair."""


def _load_parameters_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read the q0/delta_f0/delta_h arrays of one .npz file, plus the file
    provenance (path and content sha256) recorded on the instance."""
    file_path = Path(path)
    try:
        content = file_path.read_bytes()
    except OSError as error:
        raise ValueError(f"parameters_npz {path} cannot be read: {error}") from error
    try:
        archive = np.load(io.BytesIO(content))
    except Exception as error:
        raise ValueError(
            f"parameters_npz {path} is not a readable .npz archive: {error}"
        ) from error
    with archive:
        missing = [key for key in ("q0", "delta_f0", "delta_h") if key not in archive]
        if missing:
            raise ValueError(
                f"parameters_npz {path} lacks the required array(s) "
                f"{', '.join(missing)} (expected q0, delta_f0, delta_h)")
        q0 = archive["q0"]
        delta_f0 = archive["delta_f0"]
        delta_h = archive["delta_h"]
    provenance = {"path": str(file_path),
                  "sha256": hashlib.sha256(content).hexdigest()}
    return q0, delta_f0, delta_h, provenance


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
    it is frozen at construction and there is deliberately no update path
    (the read-only property has no setter — the frozen model identity
    cannot be bypassed by mutation).  The scale and its provenance string
    (``calibration_note`` — where the number came from, e.g. the
    calibration set and fit) enter the fingerprint together with the base
    model's identity.

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
        self._scale = scale
        self._calibration_note = str(calibration_note)

    @property
    def scale(self) -> float:
        """The frozen scale factor (read-only; no update path exists)."""
        return self._scale

    @property
    def calibration_note(self) -> str:
        """The frozen provenance note (read-only)."""
        return self._calibration_note

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
        digest.update(struct.pack("<d", self._scale))
        digest.update(self._calibration_note.encode("utf-8"))
        return f"scaled:{digest.hexdigest()[:16]}:{_base_identity(self._base)}"

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        base = self._base.predict(atoms)
        return SurrogatePrediction(
            energy=self._scale * base.energy,
            forces=self._scale * base.forces,
            stress=None,  # scaled stress is not exposed in this version
            uncertainty=self._scale * base.uncertainty,
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

    ``delta_h`` is symmetrized at construction and then projected onto the
    translation-invariant subspace: P dH P with
    ``P = I - T (TᵀT)⁻¹ Tᵀ``, T the three uniform translation vectors of
    the 3N configuration space, so the enforced matrix satisfies the
    acoustic sum rule by construction.  Every step is recorded: the
    Frobenius residuals before/after symmetrization
    (``symmetrization_residual_before`` / ``_after``), the ``|dH·t|``
    residuals for the three unit translation vectors before/after the
    projection (``translation_hessian_residuals_before`` / ``_after``),
    and the projection's matrix-difference norm
    (``translation_projection_norm``).  The projected matrix is what
    enters the energy, the forces and the fingerprint.  The constraint
    applies to the matrix only: ``delta_f0`` is never projected — its net
    force is just recorded (``translation_force_residuals``, the
    per-component magnitude of ``sum(dF0)``), ~0 for a translationally
    consistent correction.  Projecting the matrix constrains the
    correction Hessian; it does NOT make the corrected model
    translation-invariant as a whole (a nonzero net ``delta_f0`` still
    produces a net correction force and a translation-dependent energy),
    and user data is never silently altered.

    ``species`` (required) pins the correction map to one fixed atom order
    and element list; every ``predict`` validates the atoms against it
    (order-sensitive), and it enters the fingerprint.  The periodic chart
    is the stateless contract documented in the module docstring: the
    environment (pbc mask + cell) is recorded at the first call and
    verified unchanged on every later one; inside the local domain
    (half-cell around ``q0`` on periodic axes) the correction is exactly
    force-consistent; a rigid integer-cell shift of the whole
    configuration is explicitly supported; a genuine domain exit raises
    :class:`CorrectionDomainError` instead of returning a discontinuous
    energy/force pair.  Nothing about the chart depends on call history
    or process lifetime, so a resume in a fresh process reproduces the
    same corrections under the same fingerprint.

    ``energy_offset`` is the recorded constant zero point in eV: use
    ``U_r(q0) - U_b(q0)`` to anchor the corrected energy to the reference
    at ``q0``; the default 0 keeps the base model's value there
    (``U_c(q0) = U_b(q0)``).  A constant changes no forces.  Both
    ``energy_offset`` and ``calibration_note`` are frozen read-only
    properties — the frozen identity cannot be bypassed by mutation.

    The correction terms are exactly force-consistent by construction, so
    the base model's energy/force consistency declarations carry over
    unchanged; the deterministic, member-independent correction leaves an
    honest base spread unchanged, so the base uncertainty is forwarded
    as-is — it remains the BASE model's quantity: no recalibrated
    confidence of the corrected potential exists (the wrapper never
    recalibrates).  Stress is not implemented: ``stress_available=False``
    and predictions return ``stress=None``, never the base model's stress.
    The content hashes of ``species``, ``q0``, ``delta_f0``, the enforced
    ``delta_h`` (symmetrized and translation-projected), the energy
    offset and the calibration note enter the fingerprint together with
    the base model's identity.  When the parameters were loaded from an
    .npz file by the registry factory, ``parameters_provenance`` records
    the path and the file's content sha256 — documentation only, never
    fingerprinted (the content already is); inline construction leaves it
    None.
    """

    def __init__(
        self,
        base: object,
        q0: ArrayLike,
        delta_f0: ArrayLike,
        delta_h: ArrayLike,
        *,
        species: object,
        energy_offset: float = 0.0,
        calibration_note: str = "",
        parameters_provenance: dict | None = None,
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
        species_checked = _species_list(species)
        if len(species_checked) != n_atoms:
            raise ValueError(
                f"species must name every atom in the fixed order "
                f"({n_atoms} entries), got {len(species_checked)}"
            )
        energy_offset = float(energy_offset)
        if not math.isfinite(energy_offset):
            raise ValueError(f"energy_offset must be finite, got {energy_offset!r}")
        self._base = base
        self._species = species_checked
        self._energy_offset = energy_offset
        self._calibration_note = str(calibration_note)
        self._parameters_provenance = parameters_provenance
        self.symmetrization_residual_before = float(
            np.linalg.norm(delta_h_checked - delta_h_checked.T)
        )
        symmetrized = 0.5 * (delta_h_checked + delta_h_checked.T)
        self.symmetrization_residual_after = float(
            np.linalg.norm(symmetrized - symmetrized.T)
        )
        translations = _translation_vectors(n_atoms)
        self.translation_hessian_residuals_before = _translation_residuals(
            symmetrized, translations
        )
        # Enforce the translation constraint on the matrix: P δH P with
        # P = I - T (TᵀT)⁻¹ Tᵀ, T the three uniform translation vectors.
        # Two rank-3 downdates; P itself is never formed.
        gram_inv_t = np.linalg.solve(translations.T @ translations, translations.T)
        projected = symmetrized - translations @ (gram_inv_t @ symmetrized)
        projected = projected - (projected @ translations) @ gram_inv_t
        self.translation_projection_norm = float(
            np.linalg.norm(projected - symmetrized)
        )
        self.translation_hessian_residuals_after = _translation_residuals(
            projected, translations
        )
        self.translation_force_residuals = tuple(
            float(abs(component)) for component in delta_f0_checked.sum(axis=0)
        )
        self._q0 = _immutable(q0_checked)
        self._delta_f0 = _immutable(delta_f0_checked)
        self._delta_h = _immutable(projected)
        # The periodic chart record of this process: (cell, pbc) of the
        # first periodic call; every later call must match it exactly.
        self._chart_cell: np.ndarray | None = None
        self._chart_pbc: np.ndarray | None = None

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
        """The enforced Hessian difference — symmetrized and
        translation-projected — (3N, 3N) eV/angstrom^2, atom-major
        Cartesian, no mass weighting, read-only."""
        return self._delta_h

    @property
    def species(self) -> tuple[str, ...]:
        """The fixed atom order and elements the correction map is
        defined for (read-only)."""
        return self._species

    @property
    def energy_offset(self) -> float:
        """The frozen constant zero point in eV (read-only)."""
        return self._energy_offset

    @property
    def calibration_note(self) -> str:
        """The frozen provenance note (read-only)."""
        return self._calibration_note

    @property
    def parameters_provenance(self) -> dict | None:
        """The .npz file path/content digest when parameters came from a
        file (documentation only, never fingerprinted); read-only."""
        return self._parameters_provenance

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
            ("species", self._species),
            ("q0", self._q0),
            ("delta_f0", self._delta_f0),
            ("delta_h", self._delta_h),
        ):
            digest.update(name.encode("utf-8"))
            if name == "species":
                digest.update(_species_digest(content))
            else:
                digest.update(str(content.shape).encode("utf-8"))
                digest.update(content.tobytes(order="C"))
        digest.update(struct.pack("<d", self._energy_offset))
        digest.update(self._calibration_note.encode("utf-8"))
        return (
            f"quadratic-corrected:{digest.hexdigest()[:16]}:"
            f"{_base_identity(self._base)}"
        )

    def _displacement(self, atoms: Atoms) -> np.ndarray:
        delta = np.asarray(atoms.get_positions(), dtype=float) - self._q0
        pbc = np.asarray(atoms.pbc, dtype=bool)
        cell = np.asarray(atoms.cell, dtype=float)
        if self._chart_pbc is None:
            # The environment contract holds from the FIRST call, a
            # non-periodic one included: the pbc mask and the cell are
            # recorded once, and every later call must match them.
            if np.any(pbc):
                try:
                    np.linalg.inv(cell)
                except np.linalg.LinAlgError:
                    raise ValueError(
                        "periodic correction chart needs an invertible "
                        f"cell, got {cell.tolist()}") from None
            self._chart_pbc = pbc.copy()
            self._chart_cell = cell.copy()
        else:
            if not np.array_equal(pbc, self._chart_pbc):
                raise ValueError(
                    "the correction chart fixes the pbc mask at the first "
                    "call; the periodicity changed (either direction) — "
                    "that is a different model state, not a resume")
            if np.any(self._chart_pbc) and not np.allclose(
                    cell, self._chart_cell, rtol=0.0, atol=1e-10):
                raise ValueError(
                    "the correction chart assumes a fixed cell; the cell "
                    "changed after the chart was recorded")
        if not np.any(pbc):
            return delta
        inv_cell = np.linalg.inv(self._chart_cell)
        fractional = delta @ inv_cell
        offsets = np.zeros_like(fractional)
        for axis in range(3):
            if self._chart_pbc[axis]:
                offsets[:, axis] = np.round(fractional[:, axis])
        # The local validity domain is the half-cell neighborhood around
        # q0: a rigid integer-cell shift of the WHOLE configuration is the
        # same physical state and maps back identically (uniform offsets
        # per axis — explicitly supported, never confused with a real
        # crossing); any atom wrapping relative to the others means the
        # trajectory left the local domain.  Refuse that in a controlled
        # way for the upper layer to handle — never silently round onto
        # another branch, where the correction energy would jump while
        # its force stayed smooth (an inconsistent energy/force pair).
        for axis in range(3):
            if self._chart_pbc[axis] and np.any(offsets[:, axis]
                                              != offsets[0, axis]):
                raise CorrectionDomainError(
                    f"an atom crossed the correction's local domain on "
                    f"periodic axis {axis} (more than half the cell from "
                    "q0 relative to the others): the static quadratic "
                    "correction is only valid on the recorded chart — "
                    "re-center or refresh the correction instead of "
                    "continuing on a wrapped branch")
        return (fractional - offsets) @ self._chart_cell

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        if tuple(atoms.get_chemical_symbols()) != self._species:
            raise ValueError(
                f"the correction map is defined for {self._species} in one "
                f"fixed order, got {list(atoms.get_chemical_symbols())}"
            )
        base = self._base.predict(atoms)
        u = self._displacement(atoms).reshape(-1)  # atom-major, matching delta_h
        correction_force = (self._delta_h @ u).reshape(-1, 3)
        return SurrogatePrediction(
            energy=(
                base.energy
                - float(self._delta_f0.reshape(-1) @ u)
                + 0.5 * float(u @ self._delta_h @ u)
                + self._energy_offset
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
    q0: ArrayLike | None = None,
    delta_f0: ArrayLike | None = None,
    delta_h: ArrayLike | None = None,
    species: object = None,
    energy_offset: float = 0.0,
    calibration_note: str = "",
    parameters_npz: str | None = None,
) -> QuadraticCorrectedSurrogate:
    """Registry factory for :class:`QuadraticCorrectedSurrogate`.

    ``base`` is a surrogate instance or a ``{"name": ..., "kwargs": {...}}``
    backend spec; unknown fields are rejected by the signature itself.
    ``species`` (the fixed atom order/elements) is required.  The
    correction parameters come either inline (``q0``/``delta_f0``/
    ``delta_h``) or from ``parameters_npz`` — one .npz holding exactly
    those three arrays, so large matrices stay out of the configuration
    file; the two sources are mutually exclusive.  File origin rides on
    the instance as ``parameters_provenance`` and never enters the
    fingerprint (the content hashes already do).
    """
    if parameters_npz is not None:
        if q0 is not None or delta_f0 is not None or delta_h is not None:
            raise ValueError(
                "parameters_npz and inline q0/delta_f0/delta_h are mutually "
                "exclusive; pass the correction parameters once")
        q0, delta_f0, delta_h, provenance = _load_parameters_npz(parameters_npz)
    else:
        provenance = None
    if q0 is None or delta_f0 is None or delta_h is None:
        raise ValueError(
            "quadratic-corrected needs q0, delta_f0 and delta_h — inline or "
            "via parameters_npz")
    if species is None:
        raise ValueError(
            "quadratic-corrected needs species: the fixed atom order and "
            "elements the correction map is defined for")
    return QuadraticCorrectedSurrogate(
        _resolve_base(base),
        q0,
        delta_f0,
        delta_h,
        species=species,
        energy_offset=energy_offset,
        calibration_note=calibration_note,
        parameters_provenance=provenance,
    )


quadratic_corrected_factory.backend_kind = "surrogate"
