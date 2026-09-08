"""Sequential accepted-error bounds from independent reference checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._arrays import scalar


@dataclass(frozen=True, slots=True, eq=False)
class IndependentCheckBound:
    """Fixed-p, fixed-tilt bound on the accepted-violation fraction.

    At each accepted step the force proposal and admission must be fixed
    before a fresh Bernoulli(p) check is drawn independently of the current
    latent error. Checked labels may affect later decisions only. The
    caller supplies these checks; this counter cannot verify independence.
    A failed reference check must be resolved before updating its event.

    Settings are fixed at construction and counts change only via update.
    The simultaneous confidence level is 1-failure_probability. No bound
    is reported before the first acceptance. At p=1 every acceptance must
    be checked and the bound is the exact observed fraction. Units and the
    violation budget are defined by the caller and must remain consistent
    within this record.
    """

    probability: float
    failure_probability: float = 0.05
    tilt: float = math.log(2)
    accepted_count: int = field(default=0, init=False)
    detected_count: int = field(default=0, init=False)
    _denominator: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("probability", "failure_probability", "tilt"):
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        if not 0 < self.probability <= 1 or not 0 < self.failure_probability < 1:
            raise ValueError(
                "probability must be in (0, 1] and failure_probability in (0, 1)"
            )
        if self.tilt <= 0:
            raise ValueError("tilt must be positive")
        denominator = (
            self.tilt
            if self.probability == 1
            else -math.log1p(self.probability * math.expm1(-self.tilt))
        )
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError(
                "The probability and tilt give an unrepresentable denominator"
            )
        object.__setattr__(self, "_denominator", denominator)

    @property
    def bound(self) -> float | None:
        if self.accepted_count == 0:
            return None
        if self.probability == 1:
            return self.detected_count / self.accepted_count
        return min(
            1.0,
            (self.tilt * self.detected_count - math.log(self.failure_probability))
            / (self.accepted_count * self._denominator),
        )

    def update(
        self, accepted: bool, checked: bool, violation: bool | None
    ) -> float | None:
        """Record one event and return its bound, without filtering acceptances.

        Only an accepted proposal can be checked, and a checked proposal
        requires its observed boolean outcome. Unchecked outcomes, if supplied
        by a fully labeled record, are ignored. Invalid events do not change
        either count. A detection retains the original pre-check acceptance.
        """
        if not isinstance(accepted, (bool, np.bool_)) or not isinstance(
            checked, (bool, np.bool_)
        ):
            raise TypeError("accepted and checked must be booleans")
        if violation is not None and not isinstance(violation, (bool, np.bool_)):
            raise TypeError("violation must be a boolean or None")
        if checked and (not accepted or violation is None):
            raise ValueError(
                "A checked event must be accepted and have an observed violation outcome"
            )
        if self.probability == 1 and accepted and not checked:
            raise ValueError(
                "probability=1 requires every accepted proposal to be checked"
            )
        object.__setattr__(self, "accepted_count", self.accepted_count + int(accepted))
        object.__setattr__(
            self,
            "detected_count",
            self.detected_count + int(checked and bool(violation)),
        )
        return self.bound

    def as_dict(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "failure_probability": self.failure_probability,
            "tilt": self.tilt,
            "accepted_count": self.accepted_count,
            "detected_count": self.detected_count,
            "bound": self.bound,
        }
