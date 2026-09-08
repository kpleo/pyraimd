"""Reusable energetics of force errors: local forecasts, work and checks.

This NumPy-only core performs no reference calculations and controls no
trajectory. The runtime owns calibration origins, frozen-model intervals,
atom identities, independent checking and the accepted forecast prefix.
"""

from .response import (
    DegenerateResponseError,
    DirectionalResponse,
    Forecast,
    estimate_responses,
)
from .verification import IndependentCheckBound
from .work import integrate_residual_work, residual_work

__all__ = [
    "DegenerateResponseError",
    "DirectionalResponse",
    "Forecast",
    "IndependentCheckBound",
    "estimate_responses",
    "integrate_residual_work",
    "residual_work",
]
