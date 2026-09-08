"""Signed residual work for a fixed base potential and constant correction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._arrays import FloatArray, array, configuration, finite, immutable, scalar


def residual_work(
    origin_positions: ArrayLike,
    positions: ArrayLike,
    origin_base_energy: float,
    base_energy: float,
    origin_reference_energy: float,
    reference_energy: float,
    correction: ArrayLike,
) -> float:
    """Return W = delta(U_reference) - delta(U_base) + c.(X-X_origin).

    Energies are in eV; positions and correction have shape (N, 3), in
    angstrom and eV/angstrom respectively. Use conservative paired energies
    at the same reference settings and the same base model. Both positions
    must be unwrapped on a common branch with identical atom ordering;
    coordinates alone cannot establish either of these physical conditions.
    """
    origin = configuration(origin_positions, "origin_positions")
    position = configuration(positions, "positions")
    c = configuration(correction, "correction")
    if position.shape != origin.shape or c.shape != origin.shape:
        raise ValueError("Positions and correction shapes must agree")
    eb0 = scalar(origin_base_energy, "origin_base_energy")
    eb = scalar(base_energy, "base_energy")
    er0 = scalar(origin_reference_energy, "origin_reference_energy")
    er = scalar(reference_energy, "reference_energy")
    with np.errstate(over="raise", invalid="raise"):
        work = er - er0 - (eb - eb0) + float(np.sum(c * (position - origin)))
    return finite(work, "residual work")


def integrate_residual_work(positions: ArrayLike, residuals: ArrayLike) -> FloatArray:
    """Return immutable cumulative trapezoidal work per atom, shape (T, N).

    Inputs share shape (T, N, 3), with T, N >= 1. Positions (angstrom)
    must be unwrapped, in path order and with fixed atom identities.
    Residuals (eV/angstrom) are F_base+c-F_reference at those same states.
    The first row is zero. Sum the atom axis for total signed work (eV).
    No time factor is needed because integration uses coordinate increments.
    """
    position = array(positions, "positions")
    residual = array(residuals, "residuals")
    if position.ndim != 3 or min(position.shape[:2]) == 0 or position.shape[-1] != 3:
        raise ValueError("positions must have shape (T, N, 3) with T, N > 0")
    if residual.shape != position.shape:
        raise ValueError("positions and residuals shapes must agree")
    with np.errstate(over="raise", invalid="raise"):
        increments = np.sum(
            (0.5 * residual[:-1] + 0.5 * residual[1:]) * np.diff(position, axis=0),
            axis=-1,
        )
        work = np.zeros(position.shape[:2], dtype=np.float64)
        work[1:] = np.cumsum(increments, axis=0)
    return immutable(array(work, "integrated residual work"))
