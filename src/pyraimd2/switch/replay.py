"""Chronological offline replay over an all-labeled trajectory (M2 §4).

The replay harness is the M2 measurement instrument: given a stored
trajectory where every frame carries engine truth, a fresh committee, and a
switch, it iterates the frames in order — committee.predict → switch.assess
→ on "dft": reveal the stored label, hand it to the online updater (window
update + fine-tune trigger, the same code path as the live loop).

The harness is omniscient for scoring only: the per-frame error ``e``
against the hidden truth is recorded for every frame, but the switch's
window only ever ingests the revealed "dft" frames.  Effective miscoverage
α̂ is then the fraction of *accepted* ("ml") frames with ``e > ε_acc``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate.base import Surrogate
from pyraimd2.switch.base import LabelObservation, Route, Switch
from pyraimd2.switch.conformal import ConformalSwitch


@dataclass(frozen=True)
class ReplayRecord:
    """One replayed frame.

    Attributes:
        step: Frame index in replay order.
        route: The switch's decision ("ml" or "dft").
        spread: Committee score s = max per-atom force spread (eV/Å).
        qhat: Conformal quantile at decision time (NaN for non-conformal
            switches, +inf on an empty window).
        bound: B(s) at decision time (NaN for non-conformal switches).
        error: e = max per-atom |F̄ − F_true| against the hidden truth (eV/Å).
        violation: Accepted step whose true error exceeded ε_acc.
        finetuned: Whether revealing this frame's label fired a fine-tune.
    """

    step: int
    route: Route
    spread: float
    qhat: float
    bound: float
    error: float
    violation: bool
    finetuned: bool


@dataclass(frozen=True)
class ReplaySummary:
    """Aggregate of one replay.

    Attributes:
        n_frames: Frames replayed.
        n_dft: Frames routed to the engine (labels revealed).
        dft_fraction: n_dft / n_frames.
        n_accepted: Frames routed to the surrogate.
        alpha_hat: Effective miscoverage over accepted frames (NaN if none).
        n_finetunes: Fine-tunes fired by the updater.
        wall_time_s: Wall-clock seconds of the replay.
    """

    n_frames: int
    n_dft: int
    dft_fraction: float
    n_accepted: int
    alpha_hat: float
    n_finetunes: int
    wall_time_s: float


def replay(
    frames: Iterable[tuple[Atoms, EngineResult]],
    surrogate: Surrogate,
    switch: Switch,
    updater: Callable[[LabelObservation], object] | None = None,
    eps_acc: float = 0.1,
) -> tuple[list[ReplayRecord], ReplaySummary]:
    """Replay ``frames`` chronologically; return the decision log + summary.

    ``updater`` (typically an :class:`~pyraimd2.loop.online.OnlineUpdater`)
    is called with each revealed label, exactly like the live loop's
    ``on_label`` hook; a non-None return value marks a fired fine-tune.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("replay needs at least one frame")
    if eps_acc <= 0.0:
        raise ValueError(f"eps_acc must be > 0, got {eps_acc}")

    t0 = time.perf_counter()
    records: list[ReplayRecord] = []
    for step, (atoms, truth) in enumerate(frame_list):
        prediction = surrogate.predict(atoms)
        if isinstance(switch, ConformalSwitch):
            decision = switch.assess(atoms, step, prediction=prediction)
            qhat = switch.qhat()
            bound = float(decision.score)
        else:
            decision = switch.assess(atoms, step)
            qhat = float("nan")
            bound = float("nan")

        spread = float(np.max(prediction.uncertainty))
        error = float(np.max(np.linalg.norm(prediction.forces - truth.forces, axis=1)))
        violation = decision.route == "ml" and error > eps_acc

        finetuned = False
        if decision.route == "dft" and updater is not None:
            finetuned = (
                updater(
                    LabelObservation(
                        step=step, atoms=atoms, prediction=prediction, label=truth
                    )
                )
                is not None
            )
        records.append(
            ReplayRecord(
                step=step,
                route=decision.route,
                spread=spread,
                qhat=qhat,
                bound=bound,
                error=error,
                violation=violation,
                finetuned=finetuned,
            )
        )

    n_dft = sum(record.route == "dft" for record in records)
    accepted = [record for record in records if record.route == "ml"]
    alpha_hat = (
        float(np.mean([record.violation for record in accepted]))
        if accepted
        else float("nan")
    )
    summary = ReplaySummary(
        n_frames=len(records),
        n_dft=n_dft,
        dft_fraction=n_dft / len(records),
        n_accepted=len(accepted),
        alpha_hat=alpha_hat,
        n_finetunes=sum(record.finetuned for record in records),
        wall_time_s=time.perf_counter() - t0,
    )
    return records, summary
