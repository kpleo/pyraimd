"""Online adaptation: one object ingests every DFT label.

After each DFT-labeled step the :class:`OnlineUpdater`:

1. feeds the (s, e) pair of the shadow evaluation to the switch's
   calibration window (``observe``), and
2. fires ``committee.finetune(...)`` on the accumulated labels every
   ``n_label`` new labels.

The same updater is wired into the live loop (via the SwitchingCalculator's
``on_label`` hook) and into the offline replay (``pyraimd2.switch.replay``),
so the fine-tune trigger behaves identically in both.

For the energetic runner, :class:`GuardedUpdater` implements the resumable
candidate → validate → publish-or-rollback update protocol (WP06), and
:class:`LegacyCallbackAdapter` bridges the legacy ``TrainReport | None``
callback convention onto the False/True contract.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

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


@dataclass(frozen=True)
class UpdatePolicy:
    """Configuration of one :class:`GuardedUpdater` (WP06).

    Attributes:
        n_label: Fire one update attempt every this many new labels.
        guard_size: Size of the fixed guard set — the first labels ever
            consumed, never future acceptance-testing labels.
        train_on: ``"pending"`` trains on the labels since the last attempt;
            ``"full_history"`` explicitly reuses every consumed label
            (recipe-scheduled reuse, exempt from consumption dedup).
        max_force_growth: Reject when the candidate's worst guard force
            error exceeds this factor times the parent's.
        force_growth_floor_eV_A: Absolute floor of the growth criterion.
        max_energy_drift_eV: Optional per-config |E_candidate - E_parent|
            cap; None disables it.
        energy_force_fd_step_A: Finite-difference step for the candidate's
            energy-force consistency check.
        energy_force_atol_eV: Absolute tolerance of that consistency check.
        energy_force_rtol: Relative tolerance (against |F·u|).
    """

    n_label: int = 4
    guard_size: int = 2
    train_on: str = "pending"
    max_force_growth: float = 3.0
    force_growth_floor_eV_A: float = 0.05
    max_energy_drift_eV: float | None = None
    energy_force_fd_step_A: float = 1e-3
    energy_force_atol_eV: float = 1e-3
    energy_force_rtol: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.n_label, int) or self.n_label < 1:
            raise ValueError(f"n_label must be a positive integer, got {self.n_label}")
        if not isinstance(self.guard_size, int) or self.guard_size < 1:
            raise ValueError(f"guard_size must be a positive integer, got {self.guard_size}")
        if self.train_on not in ("pending", "full_history"):
            raise ValueError(f"train_on must be 'pending' or 'full_history', got {self.train_on!r}")
        if self.max_force_growth <= 0:
            raise ValueError("max_force_growth must be positive")
        if self.force_growth_floor_eV_A < 0:
            raise ValueError("force_growth_floor_eV_A must be nonnegative")
        if self.max_energy_drift_eV is not None and self.max_energy_drift_eV <= 0:
            raise ValueError("max_energy_drift_eV must be positive or None")
        if self.energy_force_fd_step_A <= 0:
            raise ValueError("energy_force_fd_step_A must be positive")
        if self.energy_force_atol_eV < 0 or self.energy_force_rtol < 0:
            raise ValueError("energy-force tolerances must be nonnegative")

    def recipe(self) -> dict:
        """The update recipe recorded in every model artifact."""
        return asdict(self)


def _label_payload(observation: LabelObservation) -> dict:
    return {
        "label_id": observation.label_id,
        "numbers": observation.atoms.numbers.tolist(),
        "positions": observation.atoms.positions.tolist(),
        "reference_energy_eV": float(observation.label.energy),
        "reference_forces_eV_A": observation.label.forces.tolist(),
    }


def _payload_atoms(item: dict) -> tuple[Atoms, EngineResult]:
    atoms = Atoms(numbers=np.array(item["numbers"], dtype=int),
                  positions=np.array(item["positions"], dtype=float))
    return atoms, EngineResult(float(item["reference_energy_eV"]),
                               np.array(item["reference_forces_eV_A"], dtype=float),
                               None, 0.0)


class GuardedUpdater:
    """Candidate → guard validation → publish or roll back (WP06).

    Wraps a trainable surrogate with ``state_dict``/``load_state_dict``.
    Every ``policy.n_label`` new labels it snapshots the parent state, runs
    one fine-tune, and validates the candidate on a fixed guard set (finite
    outputs, energy-force consistency by finite difference, anomalous
    error-growth and optional energy drift against the parent). A rejected
    candidate is rolled back to the exact parent state and the rejection is
    recorded; the parent model carries on. A lower training loss alone never
    counts as evidence for the candidate.

    Label consumption is deduplicated by the durable label ID: a
    redelivered label is never consumed or trained twice. ``train_on =
    "full_history"`` reuses history only because the recipe says so.
    """

    def __init__(self, surrogate: TrainableSurrogate, policy: UpdatePolicy) -> None:
        if not (hasattr(surrogate, "state_dict")
                and hasattr(surrogate, "load_state_dict")):
            raise ValueError(
                "GuardedUpdater requires a surrogate with state_dict/"
                "load_state_dict; without an explicit state interface the "
                "run is not resumable and is refused, not silently degraded")
        self.surrogate = surrogate
        self.policy = policy
        self._consumed_ids: list[str] = []
        self._consumed_set: set[str] = set()
        self._pending: list[dict] = []
        self._history: list[dict] = []
        self._guard: list[dict] = []
        self.n_updates = 0
        self.n_rejected = 0
        self.reports: list[dict] = []
        self.rejections: list[dict] = []
        self._last_update: dict | None = None
        self._last_rejection: dict | None = None

    @property
    def n_consumed(self) -> int:
        return len(self._consumed_ids)

    def __call__(self, observation: LabelObservation) -> bool:
        """Consume one label; True on a published update, False otherwise."""
        label_id = observation.label_id
        if label_id is not None and label_id in self._consumed_set:
            return False  # redelivered label: never consume or train twice
        if label_id is not None:
            self._consumed_set.add(label_id)
            self._consumed_ids.append(label_id)
        self._pending.append(_label_payload(observation))
        self._history.append(_label_payload(observation))
        if len(self._guard) < self.policy.guard_size:
            self._guard.append(_label_payload(observation))
        if len(self._pending) < self.policy.n_label:
            return False
        return self._attempt_update()

    # -- candidate lifecycle ---------------------------------------------

    def _evaluate(self, guard: list[dict]) -> list[dict]:
        evaluations = []
        for item in guard:
            atoms, _label = _payload_atoms(item)
            prediction = self.surrogate.predict(atoms)
            evaluations.append({
                "numbers": atoms.numbers.copy(),
                "energy_eV": float(prediction.energy),
                "forces_eV_A": np.asarray(prediction.forces, dtype=float),
                "reference_energy_eV": item["reference_energy_eV"],
                "reference_forces_eV_A": np.array(item["reference_forces_eV_A"],
                                                  dtype=float),
                "positions": np.array(item["positions"], dtype=float),
            })
        return evaluations

    def _energy_force_consistency(self, evaluation: dict) -> float:
        """|FD derivative + F·u| of the candidate along a fixed direction."""
        positions = evaluation["positions"]
        numbers = np.array(evaluation["numbers"], dtype=int)
        direction = np.ones_like(positions) / np.sqrt(positions.size)
        step = self.policy.energy_force_fd_step_A
        energies = []
        for sign in (1.0, -1.0):
            probe = Atoms(numbers=numbers,
                          positions=positions + sign * step * direction)
            energies.append(float(self.surrogate.predict(probe).energy))
        fd = (energies[0] - energies[1]) / (2.0 * step)
        directional = -float(np.sum(evaluation["forces_eV_A"] * direction))
        return abs(fd - directional)

    def _validate(self, parent_evals: list[dict],
                  candidate_evals: list[dict]) -> tuple[bool, str, dict]:
        metrics: dict = {"per_config": []}
        for index, (parent, candidate) in enumerate(zip(parent_evals, candidate_evals)):
            forces = candidate["forces_eV_A"]
            if not np.isfinite(candidate["energy_eV"]) or not np.isfinite(forces).all():
                return False, "non_finite_output", metrics
            consistency = self._energy_force_consistency(candidate)
            scale = abs(float(np.sum(forces**2)) ** 0.5)
            if consistency > (self.policy.energy_force_atol_eV
                              + self.policy.energy_force_rtol * scale):
                metrics["energy_force_consistency_eV"] = consistency
                return False, "energy_force_inconsistent", metrics
            error_parent = float(np.linalg.norm(
                parent["forces_eV_A"] - parent["reference_forces_eV_A"], axis=1).max())
            error_candidate = float(np.linalg.norm(
                forces - candidate["reference_forces_eV_A"], axis=1).max())
            metrics["per_config"].append({
                "guard_index": index,
                "parent_max_force_error_eV_A": error_parent,
                "candidate_max_force_error_eV_A": error_candidate,
            })
            if error_candidate > max(self.policy.max_force_growth * error_parent,
                                     self.policy.force_growth_floor_eV_A):
                metrics.update(guard_index=index,
                               parent_max_force_error_eV_A=error_parent,
                               candidate_max_force_error_eV_A=error_candidate)
                return False, "force_growth", metrics
            if self.policy.max_energy_drift_eV is not None:
                drift = abs(candidate["energy_eV"] - parent["energy_eV"])
                if drift > self.policy.max_energy_drift_eV:
                    metrics.update(guard_index=index, energy_drift_eV=drift)
                    return False, "energy_drift", metrics
        return True, "", metrics

    def _attempt_update(self) -> bool:
        policy = self.policy
        label_ids = [item["label_id"] for item in self._pending]
        training_items = (self._pending if policy.train_on == "pending"
                          else self._history)
        parent_state = self.surrogate.state_dict()
        parent_evals = self._evaluate(self._guard)
        start = time.perf_counter()
        try:
            report = self.surrogate.finetune(_payload_atoms(item)
                                             for item in training_items)
        except Exception as error:  # noqa: BLE001 — any training failure rolls
            # back to the parent state and is recorded, never swallowed.
            self.surrogate.load_state_dict(parent_state)  # back to the known version
            self._record_rejection(label_ids, "training_failed",
                                   {"error": repr(error)})
            return False
        wall = time.perf_counter() - start
        candidate_evals = self._evaluate(self._guard)
        accepted, reason, metrics = self._validate(parent_evals, candidate_evals)
        if not accepted:
            self.surrogate.load_state_dict(parent_state)  # back to the known version
            self._record_rejection(label_ids, reason, metrics)
            return False
        self.n_updates += 1
        report_dict = asdict(report)
        self.reports.append(report_dict)
        self._pending = []
        self._last_update = {
            "label_ids": label_ids,
            "recipe": policy.recipe(),
            "training": report_dict,
            "wall_time_s": wall,
        }
        return True

    def _record_rejection(self, label_ids: list[str], reason: str,
                          metrics: dict) -> None:
        self.n_rejected += 1
        self._pending = []
        rejection = {"label_ids": label_ids, "reason": reason,
                     "metrics": metrics, "recipe": self.policy.recipe()}
        self.rejections.append(rejection)
        self._last_rejection = rejection

    # -- protocol hooks used by the energetic loop -------------------------

    def update_record(self) -> dict | None:
        """Metadata of the last published update for the model artifact."""
        return self._last_update

    def pop_rejection(self) -> dict | None:
        """Return and clear the last rolled-back attempt, if any."""
        rejection = self._last_rejection
        self._last_rejection = None
        return rejection

    def state_dict(self) -> dict:
        return {
            "recipe": self.policy.recipe(),
            "consumed_ids": list(self._consumed_ids),
            "pending": [dict(item) for item in self._pending],
            "history": [dict(item) for item in self._history],
            "guard": [dict(item) for item in self._guard],
            "n_updates": self.n_updates,
            "n_rejected": self.n_rejected,
            "reports": [dict(report) for report in self.reports],
            "rejections": [dict(rejection) for rejection in self.rejections],
            "last_update": self._last_update,
            "surrogate": self.surrogate.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore completely or not at all: every check runs before any
        mutation of the updater or its surrogate."""
        if dict(state.get("recipe", {})) != self.policy.recipe():
            raise ValueError(
                f"updater recipe mismatch: {state.get('recipe')!r} != "
                f"{self.policy.recipe()!r}")
        for key in ("consumed_ids", "pending", "history", "guard", "n_updates",
                    "n_rejected", "reports", "rejections", "surrogate"):
            if key not in state:
                raise ValueError(f"updater state is missing {key!r}")
        self._consumed_ids = [str(v) for v in state["consumed_ids"]]
        self._consumed_set = set(self._consumed_ids)
        self._pending = [dict(item) for item in state["pending"]]
        self._history = [dict(item) for item in state["history"]]
        self._guard = [dict(item) for item in state["guard"]]
        self.n_updates = int(state["n_updates"])
        self.n_rejected = int(state["n_rejected"])
        self.reports = [dict(report) for report in state["reports"]]
        self.rejections = [dict(rejection) for rejection in state["rejections"]]
        self._last_update = state.get("last_update")
        self._last_rejection = None
        self.surrogate.load_state_dict(state["surrogate"])


class LegacyCallbackAdapter:
    """Bridge the legacy ``TrainReport | None`` callback convention onto the
    False/True contract the energetic loop expects.

    Legacy callbacks return ``None`` when nothing changed and a
    :class:`TrainReport` (or True) after an update. Mapped faithfully:
    ``None``/``False`` → ``False`` (no change — no generation bump, no
    recalibration beyond the baseline), anything else → ``True``. A
    record-only legacy callback therefore never triggers unnecessary
    re-probes. If the wrapped callback exports a state, the adapter proxies
    it; otherwise runs consuming labels through it are not resumable.
    """

    def __init__(self, callback: Callable) -> None:
        self.callback = callback

    def __call__(self, observation: LabelObservation) -> bool:
        result = self.callback(observation)
        return not (result is None or result is False)

    def state_dict(self) -> dict | None:
        state_dict = getattr(self.callback, "state_dict", None)
        return None if state_dict is None else state_dict()

    def load_state_dict(self, state: dict) -> None:
        load_state_dict = getattr(self.callback, "load_state_dict", None)
        if load_state_dict is None:
            raise ValueError("wrapped callback has no state restore")
        load_state_dict(state)
