"""Evaluation identity: physical time and evaluation counting are separate.

An :class:`EvaluationContext` names *which* logical evaluation is running and
*when* it happens in physical time.  ``step_id`` counts complete integration
steps, ``evaluation_id`` counts logical force evaluations and
``physical_time_fs`` comes from the real integration clock — never from a call
counter.  The three are related but must not be mixed:

- probes, single points and optimizer trial points share their parent
  evaluation's identity and do not advance MD time;
- re-reading a committed evaluation's energy or forces creates no new
  context, no new check draw and no new reference request;
- a new integration step whose coordinates happen to be unchanged is still a
  new time event: new ``step_id`` and advanced ``physical_time_fs``.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, replace
from enum import StrEnum


class EvaluationPhase(StrEnum):
    """What kind of work an evaluation belongs to."""

    INITIAL = "initial"  # pre-MD evaluation of the start configuration
    MD_STEP = "md_step"  # force evaluation completing an integration step
    PROBE = "probe"  # off-trajectory calibration probe (never advances MD time)
    SINGLE_POINT = "single_point"
    OPTIMIZATION_TRIAL = "optimization_trial"


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        integer = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {integer}")
    return int(integer)


@dataclass(frozen=True)
class EvaluationContext:
    """Identity of one logical evaluation, provided by the workflow/integrator.

    Attributes:
        run_id: Non-empty run identifier.
        step_id: Integration-step boundary this evaluation belongs to; -1 for
            the pre-MD initial evaluation (no complete step exists yet).
        evaluation_id: Logical force-evaluation counter, starting at 0.
        phase: One of :class:`EvaluationPhase` (plain strings accepted).
        physical_time_fs: Real integration time in fs.  Probes carry their
            parent evaluation's time.
        model_id: Model generation identity (see
            :mod:`pyraimd2.runtime.identity`); part of the decision-cache key.
    """

    run_id: str
    step_id: int
    evaluation_id: int
    phase: str
    physical_time_fs: float
    model_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a nonempty string")
        object.__setattr__(self, "step_id", _integer(self.step_id, "step_id", -1))
        object.__setattr__(
            self, "evaluation_id", _integer(self.evaluation_id, "evaluation_id", 0)
        )
        try:
            phase = EvaluationPhase(self.phase)
        except ValueError:
            raise ValueError(
                f"phase must be one of {[p.value for p in EvaluationPhase]}, "
                f"got {self.phase!r}"
            ) from None
        object.__setattr__(self, "phase", phase)
        time_fs = float(self.physical_time_fs)
        if not math.isfinite(time_fs) or time_fs < 0:
            raise ValueError(f"physical_time_fs must be finite and >= 0, got {time_fs}")
        object.__setattr__(self, "physical_time_fs", time_fs)
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a nonempty string")

    def for_probe(self) -> EvaluationContext:
        """Context of an off-trajectory probe belonging to this evaluation.

        Same identity and same physical time, only the phase changes: probes
        never advance the MD clock or the evaluation counter.
        """
        return replace(self, phase=EvaluationPhase.PROBE)

    def as_dict(self) -> dict:
        """JSON-safe record form for run metadata."""
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "evaluation_id": self.evaluation_id,
            "phase": str(self.phase),
            "physical_time_fs": self.physical_time_fs,
            "model_id": self.model_id,
        }
