"""Validation and immutable storage for configuration-space arrays."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def array(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def configuration(value: ArrayLike, name: str) -> FloatArray:
    result = array(value, name)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3) with N > 0")
    return result


def scalar(value: float, name: str) -> float:
    result = array(value, name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(result)


def immutable(value: FloatArray) -> FloatArray:
    # Own a copy of the buffer. Unlike setflags(write=False) on an owning
    # ndarray, a bytes-backed array cannot be made writable again by a caller.
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(
        value.shape
    )


def atom_norm(value: FloatArray) -> FloatArray | np.float64:
    """Maximum atomic Euclidean norm; preserve any leading batch axes."""
    return np.hypot.reduce(value, axis=-1).max(axis=-1)


def length(value: FloatArray) -> float:
    return float(np.hypot.reduce(value.ravel()))


def finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} is not representable as a finite float")
    return value
