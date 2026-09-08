"""Directional residual response and empirical force-error forecasts.

Positions use angstrom, energies eV, forces eV/angstrom and times fs.
Directions have unit Euclidean norm over the entire configuration. All
residuals are F_base + correction - F_reference, with the same frozen base
model and constant correction throughout a calibration and its forecast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ._arrays import (
    FloatArray,
    array,
    atom_norm,
    configuration,
    finite,
    immutable,
    length,
    scalar,
)


class DegenerateResponseError(ValueError):
    """The retained directional response is zero; C_rw is undefined."""


@dataclass(frozen=True, slots=True)
class Forecast:
    """One empirical forecast; the caller must enforce the accepted prefix."""

    linear_error: float
    envelope: float
    predicted_work: float
    transverse_fraction: float
    in_domain: bool
    admitted: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True, eq=False)
class DirectionalResponse:
    """A frozen local response with independently owned, immutable arrays.

    ``response`` is q in eV/angstrom^2. Curvature is u.q; coefficient is
    C_rw in angstrom^2/eV. Its factorization is participation times signed
    inverse curvature. Negative and zero curvature are valid when q != 0.
    ``eta`` and the transverse coefficient have units eV/angstrom^2;
    the remainder coefficient has units eV/angstrom^3. Finite-probe values
    give empirical envelopes, not certified bounds between probes.
    """

    direction: FloatArray
    response: FloatArray
    eta: float
    transverse_coefficient: float
    remainder_coefficient: float
    curvature: float = field(init=False)
    coefficient: float = field(init=False)
    participation: float = field(init=False)
    signed_inverse_curvature: float = field(init=False)

    def __post_init__(self) -> None:
        u = configuration(self.direction, "direction")
        q = configuration(self.response, "response")
        if q.shape != u.shape:
            raise ValueError("response and direction shapes must agree")
        if not np.isclose(length(u), 1.0, rtol=0, atol=1e-8):
            raise ValueError("direction must have unit full-configuration norm")
        maximum = float(atom_norm(q))
        if maximum == 0:
            raise DegenerateResponseError(
                "Directional response is zero; coefficient is undefined"
            )
        # Scaling avoids squaring extremely small or large response amplitudes.
        scaled = q / maximum
        participation = float(np.sum(scaled * scaled))
        coefficient = float(np.sum(u * scaled)) / maximum
        derived = {
            "curvature": float(np.sum(u * q)),
            "coefficient": coefficient,
            "participation": participation,
            "signed_inverse_curvature": coefficient / participation,
        }
        for name in ("eta", "transverse_coefficient", "remainder_coefficient"):
            value = scalar(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        for name, value in derived.items():
            object.__setattr__(self, name, finite(value, name))
        object.__setattr__(self, "direction", immutable(u))
        object.__setattr__(self, "response", immutable(q))

    def forecast(
        self,
        displacement: ArrayLike,
        elapsed_fs: float,
        force_budget: float,
        numerical_floor: float,
        time_cap_fs: float,
        transverse_cap: float,
    ) -> Forecast:
        """Predict at one unwrapped displacement from the calibration origin.

        B = ||alpha*q||_infinity,2 + floor + eta*|alpha| + K*||z||
        + M*||d||^2/2, and W_pred = alpha^2*(u.q)/2.
        This call is stateless: once any step fails, the runtime must stop
        the accepted prefix, even if a later point would pass individually.
        """
        d = configuration(displacement, "displacement")
        if d.shape != self.direction.shape:
            raise ValueError("displacement and direction shapes must agree")
        elapsed = scalar(elapsed_fs, "elapsed_fs")
        budget = scalar(force_budget, "force_budget")
        floor = scalar(numerical_floor, "numerical_floor")
        cap = scalar(time_cap_fs, "time_cap_fs")
        transverse_cap = scalar(transverse_cap, "transverse_cap")
        if elapsed < 0 or budget <= 0 or floor < 0 or cap <= 0:
            raise ValueError(
                "elapsed and floor must be nonnegative; budget and time cap positive"
            )
        if not 0 <= transverse_cap <= 1:
            raise ValueError("transverse_cap must be in [0, 1]")
        alpha = float(np.sum(self.direction * d))
        distance = length(d)
        transverse = length(d - alpha * self.direction)
        fraction = transverse / distance if distance else 0.0
        linear = abs(alpha) * float(atom_norm(self.response))
        envelope = (
            linear
            + floor
            + self.eta * abs(alpha)
            + self.transverse_coefficient * transverse
            + 0.5 * self.remainder_coefficient * distance * distance
        )
        work = 0.5 * alpha * alpha * self.curvature
        for name, value in (
            ("linear_error", linear),
            ("envelope", envelope),
            ("predicted_work", work),
            ("transverse_fraction", fraction),
        ):
            finite(value, name)
        domain = elapsed <= cap and fraction <= transverse_cap
        return Forecast(
            linear, envelope, work, fraction, domain, domain and envelope <= budget
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible snapshot, with field names matching this API."""
        return {
            "direction": self.direction.tolist(),
            "response": self.response.tolist(),
            "curvature": self.curvature,
            "coefficient": self.coefficient,
            "participation": self.participation,
            "signed_inverse_curvature": self.signed_inverse_curvature,
            "eta": self.eta,
            "transverse_coefficient": self.transverse_coefficient,
            "remainder_coefficient": self.remainder_coefficient,
        }


def estimate_responses(
    unit_directions: ArrayLike,
    probe_steps: ArrayLike,
    plus_displacements: ArrayLike,
    minus_displacements: ArrayLike,
    plus_residuals: ArrayLike,
    minus_residuals: ArrayLike,
) -> tuple[DirectionalResponse, ...]:
    """Estimate all directions at one force-matched origin from two scales.

    Directions have shape (D, N, 3), steps (2,), and probe arrays
    (D, 2, N, 3). The two positive steps must be in increasing order.
    Saved displacements must agree with +/-h*u within 1e-7 angstrom;
    their actual values enter M. The larger scale supplies q. K is shared
    across every direction and both scales; eta and M are directional.
    Zero retained q in any direction raises DegenerateResponseError.
    """
    u = array(unit_directions, "unit_directions")
    steps = array(probe_steps, "probe_steps")
    if u.ndim != 3 or min(u.shape[:2]) == 0 or u.shape[-1] != 3:
        raise ValueError("unit_directions must have shape (D, N, 3) with D, N > 0")
    if not np.allclose(
        np.hypot.reduce(u.reshape(len(u), -1), axis=1), 1, rtol=0, atol=1e-8
    ):
        raise ValueError("unit_directions must have unit full-configuration norm")
    if steps.shape != (2,) or not 0 < steps[0] < steps[1]:
        raise ValueError("probe_steps must contain two increasing positive steps")
    plus_d = array(plus_displacements, "plus_displacements")
    minus_d = array(minus_displacements, "minus_displacements")
    plus_r = array(plus_residuals, "plus_residuals")
    minus_r = array(minus_residuals, "minus_residuals")
    expected = (len(u), 2, *u.shape[1:])
    if any(a.shape != expected for a in (plus_d, minus_d, plus_r, minus_r)):
        raise ValueError("All probe arrays must have shape (D, 2, N, 3)")
    ideal = steps[None, :, None, None] * u[:, None]
    if not (
        np.allclose(plus_d, ideal, rtol=0, atol=1e-7)
        and np.allclose(minus_d, -ideal, rtol=0, atol=1e-7)
    ):
        raise ValueError(
            "Probe displacements must agree with the signed steps and directions"
        )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        q = (plus_r - minus_r) / (2 * steps[None, :, None, None])
        shared_transverse = float(2 * atom_norm(q).max())
        results = []
        for i, direction in enumerate(u):
            response = q[i, 1]
            remainder = 0.0
            for displacements, residuals in (
                (plus_d[i], plus_r[i]),
                (minus_d[i], minus_r[i]),
            ):
                for d, r in zip(displacements, residuals, strict=True):
                    distance = length(d)
                    if distance == 0:
                        raise ValueError("A probe displacement is zero")
                    alpha = float(np.sum(direction * d))
                    defect = float(atom_norm(r - alpha * response))
                    remainder = max(remainder, 4 * defect / distance / distance)
            results.append(
                DirectionalResponse(
                    direction,
                    response,
                    float(2 * atom_norm(q[i, 0] - response)),
                    shared_transverse,
                    remainder,
                )
            )
    return tuple(results)
