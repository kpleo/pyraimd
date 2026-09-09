"""Store: append-only run log on top of ``ase.db`` (SQLite).

Every force evaluation is one row: the atoms snapshot (positions *and*
momenta), the route, the switch's reason, the surrogate prediction, and the
engine label when present. Legacy runners use these records for restart;
energetic restart additionally needs calibration and check state and is not
implemented yet.

Row formats are versioned: rows written by this version carry
``data["schema_version"] == STORE_SCHEMA_VERSION`` (old fields kept verbatim,
new fields additive); rows from older libraries simply lack the marker and
stay readable — ``latest_state``, ``driving_label``, ``iter_labels`` and
``iter_observations`` work on both formats without conversion.

Note: use a ``.db`` file suffix — ASE maps it to its SQLite3 backend.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import ase.db
import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate.base import SurrogatePrediction

STORE_SCHEMA_VERSION = 2


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
        *,
        metadata: dict | None = None,
        driving: SurrogatePrediction | EngineResult | None = None,
        label_id: str | None = None,
        dedupe: bool = False,
    ) -> int:
        """Append a step, preserving momenta and optional actual driving label.

        ``metadata`` holds additional method-specific records. ``driving``
        distinguishes corrected forces from the uncorrected prediction and
        a reference label acquired only for checking. ``label_id`` is the
        durable identity of this row's engine label (WP02); it dedups label
        consumption and model-update events. Legacy callers omit all three
        keyword arguments and retain their original storage format, stamped
        with the current ``STORE_SCHEMA_VERSION``.

        With ``dedupe=True`` and a ``label_id``, an existing row at the same
        (run_id, step) carrying the same ``engine_label_id`` is returned
        instead of appended: the write is idempotent under a committed
        label identity, so a crash between the database write and the
        authoritative commit event never forks the evaluation into two
        divergent rows. A *different* label at the same step is a genuine
        re-execution and always appends; the commit event then binds its
        row explicitly (see :meth:`row_digest`).
        """
        if dedupe and label_id is not None:
            for row in self._db.select(run_id=run_id):
                if (int(row.key_value_pairs["step"]) == step
                        and row.data.get("engine_label_id") == str(label_id)):
                    return int(row.id)
        data = {
            "schema_version": STORE_SCHEMA_VERSION,
            "reason": reason,
            "surrogate": None if surrogate is None else _prediction_to_dict(surrogate),
            "engine": None if engine is None else _result_to_dict(engine),
        }
        if label_id is not None:
            data["engine_label_id"] = str(label_id)
        if metadata is not None:
            data["metadata"] = metadata
        if driving is not None:
            data["driving"] = {
                "energy": float(driving.energy),
                "forces": np.array(driving.forces, dtype=float, copy=True),
            }
        return int(
            self._db.write(atoms, run_id=run_id, step=int(step), route=route, data=data)
        )

    def row_by_id(self, row_id: int) -> ase.db.row.AtomsRow:
        """Fetch one row by its database id (the commit-bound identity)."""
        return self._db.get(id=int(row_id))

    @staticmethod
    def row_digest(row: ase.db.row.AtomsRow) -> str:
        """Content digest of a row's identity payload, bound into commits.

        Covers run/step/route, the durable label ID and the driving label
        values — enough to detect a commit pointing at different content
        than the row it names.
        """
        data = row.data
        driving = data.get("driving") or {}
        payload = {
            "run_id": row.key_value_pairs.get("run_id"),
            "step": int(row.key_value_pairs["step"]),
            "route": row.key_value_pairs.get("route"),
            "engine_label_id": data.get("engine_label_id"),
            "driving_energy": driving.get("energy"),
            "driving_forces": np.asarray(driving.get("forces", []),
                                         dtype=float).tolist(),
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    def committed_row(self, event_log: object | None, run_id: str,
                      evaluation_id: int) -> ase.db.row.AtomsRow:
        """Resolve the row for a committed evaluation through its commit.

        The authoritative commit event binds ``row_id`` (and ``row_digest``);
        rows from before that binding existed fall back to the legacy
        step lookup. An orphan row at the same step (a crash between the
        database write and the commit) is never returned by this path.
        """
        if event_log is not None:
            for event in event_log.iter_events():
                if event.get("type") != "evaluation_committed":
                    continue
                context = event.get("context") or {}
                if int(context.get("evaluation_id", -2)) != evaluation_id:
                    continue
                row_id = event.get("row_id")
                if row_id is not None:
                    row = self.row_by_id(int(row_id))
                    expected = event.get("row_digest")
                    if expected is not None and self.row_digest(row) != expected:
                        raise RuntimeError(
                            f"commit for evaluation {evaluation_id} binds row "
                            f"{row_id} whose content no longer matches its "
                            "recorded digest; the store looks tampered with")
                    return row
                break
        return self._row_at_step(run_id, evaluation_id - 1)

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
        if row.data.get("driving") is not None:
            payload = row.data["driving"]
            return float(payload["energy"]), np.asarray(payload["forces"], dtype=float)
        route = row.key_value_pairs["route"]
        payload = row.data["engine"] if route == "dft" else row.data["surrogate"]
        if payload is None:
            raise RuntimeError(f"run {run_id!r} step {step}: no {route!r} payload stored")
        return float(payload["energy"]), np.asarray(payload["forces"], dtype=float)

    def iter_labels(self, run_id: str) -> Iterator[tuple[Atoms, EngineResult]]:
        """Yield ``(atoms, EngineResult)`` for every row carrying an engine
        label, in step order.

        That is every "dft" row plus every explore-label row (an accepted
        "ml" step whose shadow engine label was computed anyway — see
        SwitchingCalculator): explore labels enter the live calibration
        window and fine-tune counting, so the resume-time label set must
        contain exactly them too.
        """
        rows = [
            r
            for r in self._db.select(run_id=run_id)
            if r.key_value_pairs["route"] == "dft" or r.data.get("engine") is not None
        ]
        for row in sorted(rows, key=lambda r: r.key_value_pairs["step"]):
            eng = row.data["engine"]
            yield row.toatoms(), EngineResult(
                energy=float(eng["energy"]),
                forces=np.asarray(eng["forces"], dtype=float),
                stress=None if eng["stress"] is None else np.asarray(eng["stress"], dtype=float),
                wall_time_s=float(eng["wall_time_s"]),
            )

    def trailing_ml_streak(self, run_id: str) -> int:
        """Length of the current trailing run of accepted ("ml") steps.

        The conformal streak-inflation needs the true streak length even
        across a resume: the switch's window rebuilds from (s, e) pairs,
        which only exist on "dft" rows, so the trailing "ml" tail is
        counted here from the stored routes.
        """
        rows = sorted(
            self._db.select(run_id=run_id),
            key=lambda r: r.key_value_pairs["step"],
        )
        streak = 0
        for row in reversed(rows):
            if row.key_value_pairs["route"] != "ml":
                break
            streak += 1
        return streak

    def iter_observations(self, run_id: str) -> Iterator[tuple[int, float, float]]:
        """Yield ``(step, s, e)`` for every row carrying an engine label, in
        step order — the exact spread/error stream the switch observed at
        label time.

        ``s`` is the stored max per-atom committee spread and ``e`` the
        realized max per-atom force error of the shadow prediction, both in
        eV/Å.  Used to rebuild the conformal window on resume (the window
        lives in memory; the labels live here).  Explore-label rows (route
        "ml" with an engine payload) are included: they were observed by the
        live switch, so the rebuilt window must replay them or qhat would
        silently diverge after a resume.
        """
        rows = [
            r
            for r in self._db.select(run_id=run_id)
            if r.key_value_pairs["route"] == "dft" or r.data.get("engine") is not None
        ]
        for row in sorted(rows, key=lambda r: r.key_value_pairs["step"]):
            eng = row.data.get("engine")
            sur = row.data.get("surrogate")
            if eng is None or sur is None:
                continue
            err = np.asarray(sur["forces"], dtype=float) - np.asarray(eng["forces"], dtype=float)
            e = float(np.linalg.norm(err, axis=1).max())
            s = float(np.max(np.asarray(sur["uncertainty"], dtype=float)))
            yield int(row.key_value_pairs["step"]), s, e

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
        "energy_kind": str(prediction.energy_kind),
        "force_consistent": prediction.force_consistent,
    }


def _result_to_dict(result: EngineResult) -> dict:
    return {
        "energy": float(result.energy),
        "forces": np.asarray(result.forces, dtype=float),
        "stress": None if result.stress is None else np.asarray(result.stress, dtype=float),
        "wall_time_s": float(result.wall_time_s),
        "energy_kind": str(result.energy_kind),
        "force_consistent": result.force_consistent,
    }
