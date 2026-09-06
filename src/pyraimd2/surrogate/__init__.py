"""Surrogate protocol and implementations."""

from pyraimd2.surrogate.base import (
    Surrogate,
    SurrogatePrediction,
    TrainableSurrogate,
    TrainReport,
)
from pyraimd2.surrogate.committee import CommitteeSurrogate
from pyraimd2.surrogate.ase_surrogate import AseSurrogate
from pyraimd2.surrogate.mace_surrogate import MaceSurrogate

__all__ = [
    "AseSurrogate",
    "CommitteeSurrogate",
    "MaceSurrogate",
    "Surrogate",
    "SurrogatePrediction",
    "TrainReport",
    "TrainableSurrogate",
]
