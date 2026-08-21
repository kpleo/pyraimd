"""Surrogate protocol and implementations (design doc §6)."""

from pyraimd2.surrogate.base import (
    Surrogate,
    SurrogatePrediction,
    TrainableSurrogate,
    TrainReport,
)
from pyraimd2.surrogate.committee import CommitteeSurrogate
from pyraimd2.surrogate.mace_surrogate import MaceSurrogate

__all__ = [
    "CommitteeSurrogate",
    "MaceSurrogate",
    "Surrogate",
    "SurrogatePrediction",
    "TrainReport",
    "TrainableSurrogate",
]
