"""Online adaptation: one object ingests every DFT label.

After each DFT-labeled step the :class:`OnlineUpdater`:

1. feeds the (s, e) pair of the shadow evaluation to the switch's
   calibration window (``observe``), and
2. fires ``committee.finetune(...)`` on the accumulated labels every
   ``n_label`` new labels.

The same updater is wired into the live loop (via the SwitchingCalculator's
``on_label`` hook) and into the offline replay (``pyraimd2.switch.replay``),
so the fine-tune trigger behaves identically in both.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate.base import TrainableSurrogate, TrainReport
from pyraimd2.switch.base import LabelObservation


class OnlineUpdater:
    """Conformal window update + periodic committee fine-tune.

    Args:
        committee: The trainable surrogate.
        observe: Sink for (s, e) pairs — ``ConformalSwitch.observe`` in the
            calibrated loop; a no-op for uncalibrated ablations.
        n_label: Fine-tune period in new DFT labels (default: 8).
        label_source: Supplier of the full current label set for fine-tuning
            (e.g. ``lambda: store.iter_labels(run_id)``).  When None, the
            updater accumulates (atoms, label) pairs from the observations
            themselves — equivalent, and the only option under replay.
        checkpoint: Optional callback invoked after every fine-tune (e.g.
            persisting the committee for restart-safe campaigns).  Exceptions
            propagate — a campaign that cannot checkpoint is not restart-safe.
    """

    def __init__(
        self,
        committee: TrainableSurrogate,
        observe: Callable[[float, float], None],
        n_label: int = 8,
        label_source: Callable[[], Iterable[tuple[Atoms, EngineResult]]] | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        if n_label < 1:
            raise ValueError(f"n_label must be >= 1, got {n_label}")
        self.committee = committee
        self.observe = observe
        self.n_label = n_label
        self.label_source = label_source
        self.checkpoint = checkpoint
        self.labels: list[tuple[Atoms, EngineResult]] = []
        self.reports: list[TrainReport] = []
        self.n_observations = 0
        self.n_finetunes = 0

    def __call__(self, observation: LabelObservation) -> TrainReport | None:
        """Ingest one label; returns the TrainReport iff a fine-tune fired."""
        prediction, label = observation.prediction, observation.label
        if np.asarray(prediction.forces).shape != np.asarray(label.forces).shape:
            raise ValueError(
                f"prediction/label force shape mismatch: "
                f"{np.asarray(prediction.forces).shape} vs {np.asarray(label.forces).shape}"
            )
        s = float(np.max(prediction.uncertainty))
        e = float(np.max(np.linalg.norm(prediction.forces - label.forces, axis=1)))
        self.observe(s, e)  # validation of finiteness lives in the switch
        self.labels.append((observation.atoms, label))
        self.n_observations += 1

        if self.n_observations % self.n_label != 0:
            return None
        if self.label_source is not None:
            labels = list(self.label_source())
        else:
            labels = list(self.labels)
        report = self.committee.finetune(labels)
        self.reports.append(report)
        self.n_finetunes += 1
        if self.checkpoint is not None:
            self.checkpoint()
        return report
