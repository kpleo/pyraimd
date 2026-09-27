"""Offline closed-form force least-squares scale calibration.

One narrow tool: given reference forces and fast-model forces already
paired over the SAME configurations in the SAME atom order, the scalar
minimizing the sum of squared force residuals ``|F_ref - c F_fast|^2`` is
the closed form

    alpha = sum(F_fast · F_ref) / sum(F_fast · F_fast)

which is exactly the frozen scale the ``scaled`` correction wrapper
(:class:`~pyraimd2.surrogate.corrections.ScaledSurrogate`) applies to BOTH
the energy and the forces in production.  There is deliberately no
intercept, no Hessian, no alternative loss and no train/validation split:
all frames enter the fit, the reported residual RMS values are TRAINING
metrics over the fitted set, and the result carries no generalization
bound, no confidence interval and no recommended MTS outer ratio.

:func:`fit_force_scale` is pure array math and never touches backends,
files or model packages.  The ``calibrate-scale`` CLI's NPZ input protocol
and JSON output record live in this module too (the thin adapter the CLI
wires): :func:`read_pairs_npz` validates the pairing/identity declarations
and :func:`write_report` emits the record atomically.  The record is meant
for users to copy ``scale`` into ``[surrogate] backend = "scaled"``; its
content hash can be noted in the wrapper's ``calibration_note``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPORT_SCHEMA_VERSION = 1
FORCE_UNIT = "eV/angstrom"
#: smallest accepted denominator (eV^2/angstrom^2) — below this the closed
#: form divides by numerical noise
MIN_DENOMINATOR = 1e-20


class CalibrationError(ValueError):
    """The calibration inputs or protocol are not usable; a controlled
    refusal (never clipped, windowed or defaulted to scale 1)."""


@dataclass(frozen=True)
class ForceScaleFit:
    """The closed-form scale and its training-set metrics (schema 1)."""

    schema_version: int
    status: str
    scale: float
    numerator: float            # sum(F_fast · F_ref), (eV/angstrom)^2
    denominator: float          # sum(F_fast · F_fast), (eV/angstrom)^2
    n_frames: int
    n_atoms: int
    # per-ATOM vector RMS of the force residuals over the fitted set:
    # sqrt(sum(residual^2) / (n_frames * n_atoms)), in force_unit
    force_residual_rms_before_eV_A: float
    force_residual_rms_after_eV_A: float
    force_unit: str = FORCE_UNIT


def _checked_forces(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind == "c":
        raise CalibrationError(f"{name} must be real, got a complex dtype")
    if raw.dtype.kind not in "iuf":
        raise CalibrationError(
            f"{name} must be a real numeric array, got dtype {raw.dtype}")
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise CalibrationError(
            f"{name} must have shape (n_frames, n_atoms, 3), got "
            f"{raw.shape}")
    if raw.shape[0] < 1 or raw.shape[1] < 1:
        raise CalibrationError(
            f"{name} must have n_frames >= 1 and n_atoms >= 1, got "
            f"{raw.shape}")
    result = np.array(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise CalibrationError(f"{name} must contain only finite values")
    return result


def _rms(residual: np.ndarray) -> float:
    n = residual.shape[0] * residual.shape[1]  # frames * atoms (per atom)
    return float(np.sqrt((residual**2).sum() / n))


def fit_force_scale(reference_forces: object,
                    fast_forces: object) -> ForceScaleFit:
    """Closed-form scale ``alpha = sum(F_fast·F_ref) / sum(F_fast·F_fast)``.

    Both arrays must share exactly the shape ``(n_frames, n_atoms, 3)``
    with ``n_frames >= 1`` and ``n_atoms >= 1``, real and finite
    (broadcastable-but-different shapes, complex dtypes and empty axes
    are refused, as are a vanishing denominator, a non-positive or
    non-finite alpha and overflowed products).  Every frame enters the
    fit; the returned RMS values are training metrics over that set.
    """
    reference = _checked_forces(reference_forces, "reference_forces")
    fast = _checked_forces(fast_forces, "fast_forces")
    if reference.shape != fast.shape:
        raise CalibrationError(
            "reference_forces and fast_forces must share one exact shape "
            f"(n_frames, n_atoms, 3); got {reference.shape} vs "
            f"{fast.shape} — pairing is declared by the caller, never "
            "broadcast or inferred")
    with np.errstate(over="ignore", invalid="ignore"):
        numerator = float((fast * reference).sum())
        denominator = float((fast * fast).sum())
    if not np.isfinite(denominator) or not np.isfinite(numerator):
        raise CalibrationError(
            "the force products overflowed float64; the inputs are not "
            "usable as given (no NaN/Infinity result is ever produced)")
    if denominator <= MIN_DENOMINATOR:
        raise CalibrationError(
            f"the denominator sum(F_fast^2) = {denominator:.3e} "
            f"(eV/angstrom)^2 is not > {MIN_DENOMINATOR:g}; the closed "
            "form would divide by numerical noise")
    scale = numerator / denominator
    if not np.isfinite(scale) or scale <= 0.0:
        raise CalibrationError(
            f"the fitted scale is {scale!r}; only a finite, positive "
            "scale is usable in the scaled wrapper")
    with np.errstate(over="ignore", invalid="ignore"):
        rms_before = _rms(reference - fast)
        rms_after = _rms(reference - scale * fast)
    for name, value in (("force_residual_rms_before", rms_before),
                        ("force_residual_rms_after", rms_after)):
        if not np.isfinite(value):
            raise CalibrationError(f"{name} overflowed float64")
    return ForceScaleFit(
        schema_version=REPORT_SCHEMA_VERSION,
        status="ok",
        scale=scale,
        numerator=numerator,
        denominator=denominator,
        n_frames=int(reference.shape[0]),
        n_atoms=int(reference.shape[1]),
        force_residual_rms_before_eV_A=rms_before,
        force_residual_rms_after_eV_A=rms_after,
    )


# ---------------------------------------------------------------------------
# CLI adapter: the NPZ pairing protocol and the JSON output record

_REQUIRED_NPZ_KEYS = ("reference_forces_eV_A", "fast_forces_eV_A",
                      "frame_ids", "reference_id", "fast_model_id",
                      "force_unit")


def _unicode_scalar(value: np.ndarray, key: str) -> str:
    array = np.asarray(value)
    if array.dtype.kind != "U" or array.ndim != 0:
        raise CalibrationError(
            f"npz key {key!r} must be a Unicode string scalar")
    return str(array)


def read_pairs_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and validate one calibration pairs file (``allow_pickle=False``).

    Returns ``(reference_forces, fast_forces, provenance)`` where
    provenance carries the file basename + content sha256, the declared
    ``frame_ids`` (unique Unicode strings, one per frame — the caller's
    declaration that both sides are paired in identical configurations
    and atom order) and the ``reference_id`` / ``fast_model_id`` /
    ``force_unit`` scalars.  No data is discovered, split or inferred.
    """
    path = Path(path)
    try:
        content = path.read_bytes()
    except OSError as error:
        raise CalibrationError(f"pairs file cannot be read: {error}") from error
    try:
        archive = np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as error:
        raise CalibrationError(
            f"{path.name} is not a readable .npz archive: {error}") from error
    try:
        keys = set(archive.files)
    finally:
        archive.close()
    missing = [key for key in _REQUIRED_NPZ_KEYS if key not in keys]
    if missing:
        raise CalibrationError(
            f"{path.name} lacks the required key(s) {missing}; the "
            f"protocol expects {list(_REQUIRED_NPZ_KEYS)}")
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        try:
            reference = archive["reference_forces_eV_A"]
            fast = archive["fast_forces_eV_A"]
            frame_ids = archive["frame_ids"]
            reference_id = archive["reference_id"]
            fast_model_id = archive["fast_model_id"]
            force_unit = archive["force_unit"]
        except Exception as error:  # e.g. pickled payload blocked above
            raise CalibrationError(
                f"{path.name} is not a readable calibration pairs file: "
                f"{error}") from error
        reference = _checked_forces(reference, "reference_forces_eV_A")
        fast = _checked_forces(fast, "fast_forces_eV_A")
        ids = np.asarray(frame_ids)
        if ids.dtype.kind != "U" or ids.ndim != 1:
            raise CalibrationError(
                "npz key 'frame_ids' must be a 1-D array of Unicode "
                "strings, one per frame")
        if len(ids) != reference.shape[0]:
            raise CalibrationError(
                f"frame_ids has {len(ids)} entries but the force arrays "
                f"have {reference.shape[0]} frames")
        if len({str(item) for item in ids}) != len(ids):
            raise CalibrationError("frame_ids must be unique")
        reference_id = _unicode_scalar(reference_id, "reference_id")
        fast_model_id = _unicode_scalar(fast_model_id, "fast_model_id")
        force_unit = _unicode_scalar(force_unit, "force_unit")
    if not reference_id.strip() or not fast_model_id.strip():
        raise CalibrationError(
            "reference_id and fast_model_id must be non-empty")
    if force_unit != FORCE_UNIT:
        raise CalibrationError(
            f"force_unit must be exactly {FORCE_UNIT!r}, got "
            f"{force_unit!r}")
    provenance = {
        "file": path.name,  # basename only — never an absolute path
        "sha256": hashlib.sha256(content).hexdigest(),
        "frame_ids": [str(item) for item in ids],
        "reference_id": reference_id,
        "fast_model_id": fast_model_id,
        "force_unit": force_unit,
    }
    return reference, fast, provenance


def report_dict(fit: ForceScaleFit, provenance: dict) -> dict:
    """The calibrate-scale output record (schema 1)."""
    return {**asdict(fit),
            "input": {"file": provenance["file"],
                      "sha256": provenance["sha256"]},
            "frame_ids": provenance["frame_ids"],
            "reference_id": provenance["reference_id"],
            "fast_model_id": provenance["fast_model_id"]}


def write_report(report: dict, output: str | Path, *,
                 force: bool = False) -> Path:
    """Write the report atomically (temp file, then replace).  An existing
    output is kept unless ``force``; a failed fit never reaches here, so a
    pre-existing output file is never corrupted."""
    output = Path(output)
    if output.exists() and not force:
        raise CalibrationError(
            f"output file exists: {output}; pass --force to overwrite or "
            "choose a different --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, output)
    return output
