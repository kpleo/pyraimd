"""Switch protocol: the routing decision is a first-class, logged object
(design doc §3, rule 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate.base import SurrogatePrediction

Route = Literal["ml", "dft"]


@dataclass(frozen=True)
class Decision:
    """The switch's verdict for one force evaluation.

    Attributes:
        route: "ml" (surrogate) or "dft" (engine).
        score: Optional continuous score behind the decision (None for a
            purely scheduled switch).
        reason: Human-readable rationale, logged to the Store with every step.
    """

    route: Route
    score: float | None
    reason: str


class Switch(Protocol):
    """Decides, per force evaluation, who provides the forces."""

    def assess(self, atoms: Atoms, step: int) -> Decision:
        """Return the :class:`Decision` for ``atoms`` at logged ``step``."""
        ...


@dataclass(frozen=True)
class LabelObservation:
    """One DFT-labeled step, handed to online-adaptation hooks (M2 §3).

    Produced by the SwitchingCalculator on every "dft" route (and by the
    offline replay on every revealed frame): the engine label plus the
    surrogate's shadow prediction at the same geometry.

    Attributes:
        step: The logged step index.
        atoms: Snapshot of the evaluated configuration.
        prediction: The surrogate's shadow prediction.
        label: The engine's label.
    """

    step: int
    atoms: Atoms
    prediction: SurrogatePrediction
    label: EngineResult
