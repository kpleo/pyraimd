"""Store: append-only run log on top of ``ase.db`` (SQLite).

Every force evaluation is one row: the atoms snapshot (positions *and*
momenta), the route, the switch's reason, the surrogate prediction, and the
engine label when present.  The loop can be killed and resumed from the Store
alone (design doc §3, rule 3).

Note: use a ``.db`` file suffix — ASE maps it to its SQLite3 backend.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import ase.db
import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate.base import SurrogatePrediction


class Store:
    """Append-only wrapper around an ASE SQLite database."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._db = ase.db.connect(self.path)

    def append(
        self,
        run_id: str,
        step: int,
        atoms: Atoms,
        route: str,
        surrogate: SurrogatePrediction | None = None,
        engine: EngineResult | None = None,
        reason: str = "",
    ) -> int:
        """Append one logged step; momenta on ``atoms`` are preserved by ase.db."""
        data = {
            "reason": reason,
            "surrogate": None if surrogate is None else _prediction_to_dict(surrogate),
            "engine": None if engine is None else _result_to_dict(engine),
        }
        return int(
            self._db.write(atoms, run_id=run_id, step=int(step), route=route, data=data)
        )

    def latest_state(self, run_id: str) -> tuple[Atoms, int]:
        """Return ``(atoms, step)`` of the last logged MD step of ``run_id``.

        The atoms carry the momenta exactly as logged.  Note that the
        SwitchingCalculator logs the atoms at force-evaluation time, so for
        step >= 0 these are the velocity-Verlet half-step momenta;
        :meth:`driving_label` provides what is needed to reconstruct the
        on-step momenta.
        """
        row = self._latest_row(run_id)
        step = int(row.key_value_pairs["step"])
        if step < 0:
            raise RuntimeError(
                f"run {run_id!r} has only the pre-MD initial evaluation "
                "(step -1); nothing to resume from"
            )
        return row.toatoms(), step

    def driving_label(self, run_id: str, step: int) -> tuple[float, np.ndarray]:
        """Return ``(energy, forces)`` that actually drove the MD at ``step``:
        the engine label on "dft" rows, the surrogate prediction on "ml" rows.

        ``Runner.resume`` uses this — instead of re-invoking engine and
        switch — so a restart reproduces the run bit-for-bit (§3, rule 3).
        """
        row = self._row_at_step(run_id, step)
        route = row.key_value_pairs["route"]
        payload = row.data["engine"] if route == "dft" else row.data["surrogate"]
        if payload is None:
            raise RuntimeError(f"run {run_id!r} step {step}: no {route!r} payload stored")
        return float(payload["energy"]), np.asarray(payload["forces"], dtype=float)

    def iter_labels(self, run_id: str) -> Iterator[tuple[Atoms, EngineResult]]:
        """Yield ``(atoms, EngineResult)`` for every "dft" row, in step order."""
        rows = [r for r in self._db.select(run_id=run_id) if r.key_value_pairs["route"] == "dft"]
        for row in sorted(rows, key=lambda r: r.key_value_pairs["step"]):
            eng = row.data["engine"]
            yield row.toatoms(), EngineResult(
                energy=float(eng["energy"]),
                forces=np.asarray(eng["forces"], dtype=float),
                stress=None if eng["stress"] is None else np.asarray(eng["stress"], dtype=float),
                wall_time_s=float(eng["wall_time_s"]),
            )

    def _latest_row(self, run_id: str) -> ase.db.row.AtomsRow:
        rows = list(self._db.select(run_id=run_id))
        if not rows:
            raise RuntimeError(f"no rows stored for run_id={run_id!r}")
        # rows come out in insertion (id) order; max() keeps the first
        # occurrence on ties, i.e. the originally logged row.
        return max(rows, key=lambda r: r.key_value_pairs["step"])

    def _row_at_step(self, run_id: str, step: int) -> ase.db.row.AtomsRow:
        for row in self._db.select(run_id=run_id):
            if int(row.key_value_pairs["step"]) == step:
                return row
        raise RuntimeError(f"no row stored for run_id={run_id!r} at step {step}")


def _prediction_to_dict(prediction: SurrogatePrediction) -> dict:
    return {
        "energy": float(prediction.energy),
        "forces": np.asarray(prediction.forces, dtype=float),
        "stress": None if prediction.stress is None else np.asarray(prediction.stress, dtype=float),
        "uncertainty": np.asarray(prediction.uncertainty, dtype=float),
    }


def _result_to_dict(result: EngineResult) -> dict:
    return {
        "energy": float(result.energy),
        "forces": np.asarray(result.forces, dtype=float),
        "stress": None if result.stress is None else np.asarray(result.stress, dtype=float),
        "wall_time_s": float(result.wall_time_s),
    }
