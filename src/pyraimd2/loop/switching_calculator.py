"""SwitchingCalculator: route force evaluations through an ASE calculator.

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
-1 .. n_steps-1 and every evaluation is recorded.

Engine failures propagate as :class:`~pyraimd2.engines.base.EngineError` —
the calculator never falls back to the surrogate silently, and
nothing is logged or counted for a failed evaluation.

Online adaptation: the optional ``on_label`` hook is called with a
:class:`~pyraimd2.switch.base.LabelObservation` after every successfully
logged "dft" step (engine label + shadow prediction).  Hook exceptions
propagate — the label is already durably in the Store at that point.

Randomized exploration labels (off by default): with ``explore_frac`` = p > 0,
each *accepted* ("ml") step draws
``u ~ U[0,1)`` from a dedicated, deterministically seeded stream, and on
``u < p`` the engine label is computed ANYWAY — a shadow label.  The step
still propagates with the surrogate forces (the current force remains fixed; labels may change later
decisions and model updates), the row keeps ``route="ml"`` with the engine
payload attached and "explore-label" in its reason string, and the label
flows through ``on_label`` into the calibration window/updater like any
other label — that is its purpose (drift audit on accepted steps).

Streak semantics: an explore label does NOT reset the
conformal streak.  The streak belongs to the *decision* path:
``ConformalSwitch.assess`` increments it on an "ml" decision and resets it
only on a "dft" decision, while ``observe`` is pure window ingestion and
never touches it.  The explore path feeds the window through ``observe``
only, so live, resumed (``Store.trailing_ml_streak`` counts route=="ml"
rows), and audit-parsed streaks all agree: the streak counts accepted steps
since the last DECISION-DRIVEN label.
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
        explore_frac: float = 0.0,
        explore_rng: np.random.Generator | None = None,
        explore_seed: int | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= explore_frac <= 1.0:
            raise ValueError(f"explore_frac must be in [0, 1], got {explore_frac}")
        if explore_frac > 0.0 and explore_rng is None:
            raise ValueError(
                "explore_frac > 0 requires an explicit explore_rng "
                "(np.random.default_rng(seed)) — the stream must be "
                "deterministic and documented, never ambient entropy"
            )
        self.surrogate = surrogate
        self.engine = engine
        self.switch = switch
        self.store = store
        self.run_id = run_id
        self.step = start_step  # next step index to log; set on resume
        self.on_label = on_label
        self.explore_frac = explore_frac
        self.explore_rng = explore_rng
        self.explore_seed = explore_seed  # echoed into explore reason strings
        self.n_dft = 0  # engine evaluations (this calculator's lifetime)
        self.n_ml = 0  # surrogate-only evaluations (this calculator's lifetime)
        # Shadow engine labels on accepted steps (subset of n_ml; NOT added
        # to n_dft — that counter tracks decision-driven engine evaluations).
        self.n_explore = 0
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
        explore_label = False
        explore_draw = float("nan")
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
            if self.explore_frac > 0.0:
                explore_draw = float(self.explore_rng.random())
                if explore_draw < self.explore_frac:
                    # Explore label (see module docstring): shadow engine
                    # label on an ACCEPTED step.  Same failure contract as a
                    # decision-driven label — EngineError propagates before
                    # anything is logged or counted.  energy/forces stay the
                    # surrogate's for this evaluation.
                    engine_result = self.engine.compute(self.atoms)
                    explore_label = True
                    self.shadow_force_errors.append(
                        np.linalg.norm(
                            prediction.forces - engine_result.forces, axis=1
                        )
                    )
                    self.n_explore += 1

        reason = decision.reason
        if explore_label:
            reason += (
                f" [explore-label: shadow engine label on accepted step "
                f"(p={self.explore_frac}, seed={self.explore_seed}, "
                f"draw={explore_draw:.4f}); surrogate forces propagated]"
            )
        self.store.append(
            self.run_id,
            step,
            self.atoms,
            decision.route,
            surrogate=prediction,
            engine=engine_result,
            reason=reason,
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
