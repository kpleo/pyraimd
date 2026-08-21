"""SwitchingCalculator: the per-step contract of design doc §4.2 as an ASE
calculator.

Every force evaluation:

1. asks the Switch for a :class:`~pyraimd2.switch.base.Decision`,
2. routes to the surrogate ("ml") or to the engine ("dft") — on "dft" the
   surrogate is *also* evaluated (shadow prediction) so the surrogate-vs-label
   force error is logged on every labelled step,
3. is appended to the Store with the step counter, route, and reason.

Step numbering: the counter starts at -1 on a fresh run.  VelocityVerlet
evaluates the initial geometry x0 once before the first MD step, so that
evaluation is logged as step -1 and MD step k >= 1 is driven by the
evaluation logged as step k-1.  Hence ``run(n_steps)`` logs steps
-1 .. n_steps-1 and every evaluation is audited (design doc §4.1: "logs
everything").

Engine failures propagate as :class:`~pyraimd2.engines.base.EngineError` —
the calculator never falls back to the surrogate silently (§3, rule 2), and
nothing is logged or counted for a failed evaluation.

Online adaptation (M2 §3): the optional ``on_label`` hook is called with a
:class:`~pyraimd2.switch.base.LabelObservation` after every successfully
logged "dft" step (engine label + shadow prediction).  Hook exceptions
propagate — the label is already durably in the Store at that point.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pyraimd2.engines.base import Engine
from pyraimd2.store.store import Store
from pyraimd2.surrogate.base import Surrogate
from pyraimd2.switch.base import LabelObservation, Switch


class SwitchingCalculator(Calculator):
    """Routes each force evaluation through the Switch; logs to the Store."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        surrogate: Surrogate,
        engine: Engine,
        switch: Switch,
        store: Store,
        run_id: str,
        start_step: int = -1,
        on_label: Callable[[LabelObservation], None] | None = None,
    ) -> None:
        super().__init__()
        self.surrogate = surrogate
        self.engine = engine
        self.switch = switch
        self.store = store
        self.run_id = run_id
        self.step = start_step  # next step index to log; set on resume
        self.on_label = on_label
        self.n_dft = 0  # engine evaluations (this calculator's lifetime)
        self.n_ml = 0  # surrogate-only evaluations (this calculator's lifetime)
        # Per-atom |F_surrogate - F_engine| on dft steps, one (N,) array each.
        self.shadow_force_errors: list[np.ndarray] = []

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] = ("energy", "forces"),
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        step = self.step
        decision = self.switch.assess(self.atoms, step)

        engine_result = None
        if decision.route == "dft":
            # EngineError propagates to the caller before anything is logged,
            # counted, or advanced — never fall back to the surrogate silently.
            engine_result = self.engine.compute(self.atoms)
            prediction = self.surrogate.predict(self.atoms)  # shadow evaluation
            energy = engine_result.energy
            forces = engine_result.forces
            self.shadow_force_errors.append(
                np.linalg.norm(prediction.forces - engine_result.forces, axis=1)
            )
            self.n_dft += 1
        else:
            prediction = self.surrogate.predict(self.atoms)
            energy = prediction.energy
            forces = prediction.forces
            self.n_ml += 1

        self.store.append(
            self.run_id,
            step,
            self.atoms,
            decision.route,
            surrogate=prediction,
            engine=engine_result,
            reason=decision.reason,
        )
        if engine_result is not None and self.on_label is not None:
            self.on_label(
                LabelObservation(
                    step=step,
                    atoms=self.atoms.copy(),
                    prediction=prediction,
                    label=engine_result,
                )
            )
        self.results["energy"] = float(energy)
        self.results["forces"] = np.asarray(forces, dtype=float)
        self.step = step + 1
