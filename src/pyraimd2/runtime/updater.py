"""Stateful updater contract for resumable runs.

A plain ``on_label`` callback is a side effect with no explicit state and
therefore cannot be replayed or resumed.  A resumable run uses an updater
that is callable like a callback and additionally exports its complete
continuation state: after ``load_state_dict(state)`` the updater (and the
surrogate it drives) must behave exactly as it would have without a stop.
The state must be JSON-serializable; it is persisted after every consumed
label (and, for updates, as the immutable model artifact) before the run
commits the corresponding event.
"""

from __future__ import annotations

from typing import Protocol

from pyraimd2.switch.base import LabelObservation


class StatefulUpdater(Protocol):
    """Callable label consumer with an exportable, restorable state."""

    def __call__(self, observation: LabelObservation) -> bool | None:
        """Consume one label; return exactly ``False`` to declare the model
        unchanged (any other return announces a model update)."""
        ...

    def state_dict(self) -> dict:
        """Complete JSON-safe continuation state (counters, queue position,
        and everything needed to reproduce the surrogate's parameters)."""
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restore a state exported by :meth:`state_dict`; raise on mismatch
        rather than partially mutating the updater or its surrogate."""
        ...
