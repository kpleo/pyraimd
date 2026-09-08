"""Energetic verification replay: rebuild check counts and the bound offline.

``Store.iter_observations`` computes the raw base–reference difference and
cannot reconstruct violations of the *accepted corrected* force.  This
reader instead recomputes, per stored energetic evaluation, the error of
the **frozen driving force** against the reference force acquired under the
same definition, then replays the independent-check stream step by step:
``p=0`` runs report disabled, ``p=1`` runs give the exact observed fraction.
"""

from __future__ import annotations

import numpy as np

from pyraimd2.energetics import IndependentCheckBound
from pyraimd2.store.store import Store


def _row_error_and_violation(row: object, force_budget: float) -> tuple[float, bool]:
    """Max per-atom |driving - reference| and its budget verdict.

    The driving force is the frozen accepted force (base prediction plus the
    anchor correction); the reference force is the check label acquired at
    the same geometry — the same definition the online loop used.
    """
    data = row.data
    driving = np.asarray(data["driving"]["forces"], dtype=float)
    reference = np.asarray(data["engine"]["forces"], dtype=float)
    error = float(np.linalg.norm(driving - reference, axis=1).max())
    return error, bool(error > force_budget)


def replay_verification(store: Store, run_id: str) -> dict:
    """Replay the independent-check stream of ``run_id`` from stored rows.

    Returns a dict with the rebuilt counts/bound, the online-recorded final
    counts, and a per-step mismatch list.  ``matches_online`` is True only
    when every stored evaluation's recomputed violation verdict and the
    running (accepted, detected) counts equal what the online loop recorded.
    With ``check_probability=0`` the run recorded no verification settings
    and the replay reports ``enabled=False``.
    """
    rows = sorted(store._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))
    bound: IndependentCheckBound | None = None
    mismatches: list[dict] = []
    n_evaluations = 0
    last_recorded: dict | None = None
    for row in rows:
        metadata = row.data.get("metadata") or {}
        if metadata.get("method") != "energetic_force_error":
            continue
        n_evaluations += 1
        step = int(row.key_value_pairs["step"])
        recorded = metadata.get("verification")
        if recorded is None:
            if bound is not None:
                mismatches.append({"step": step,
                                   "problem": "missing verification settings "
                                              "in an enabled run"})
            continue
        if bound is None:
            bound = IndependentCheckBound(
                float(recorded["probability"]),
                float(recorded["failure_probability"]),
                float(recorded["tilt"]),
            )
        accepted = bool(metadata["accepted"])
        checked = bool(metadata["checked"])
        violation: bool | None = None
        if checked:
            _, violation = _row_error_and_violation(
                row, float(metadata["force_budget_eV_A"])
            )
            if violation != metadata["violation"]:
                mismatches.append({"step": step,
                                   "problem": "violation verdict differs",
                                   "recorded": metadata["violation"],
                                   "recomputed": violation})
        bound.update(accepted, checked, violation)
        recorded_now = recorded
        last_recorded = recorded_now
        for key in ("accepted_count", "detected_count"):
            if int(recorded_now[key]) != int(getattr(bound, key)):
                mismatches.append({"step": step,
                                   "problem": f"{key} differs",
                                   "recorded": recorded_now[key],
                                   "recomputed": getattr(bound, key)})
    result = {
        "enabled": bound is not None,
        "n_evaluations": n_evaluations,
        "mismatches": mismatches,
        "recorded": last_recorded,
    }
    if bound is not None:
        result.update(bound.as_dict())
        result["matches_online"] = (
            not mismatches
            and last_recorded is not None
            and int(last_recorded["accepted_count"]) == bound.accepted_count
            and int(last_recorded["detected_count"]) == bound.detected_count
        )
    else:
        result["matches_online"] = not mismatches
    return result
