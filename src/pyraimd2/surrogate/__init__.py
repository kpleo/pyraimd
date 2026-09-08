"""Surrogate protocol and implementations."""

from pyraimd2.surrogate.ase_surrogate import AseSurrogate
from pyraimd2.surrogate.base import (
    Surrogate,
    SurrogateCapabilities,
    SurrogatePrediction,
    TrainableSurrogate,
    TrainReport,
    assert_compatible_energy_contract,
    surrogate_capabilities,
)
from pyraimd2.surrogate.committee import CommitteeSurrogate
from pyraimd2.surrogate.mace_surrogate import MaceSurrogate

__all__ = [
    "AseSurrogate",
    "CommitteeSurrogate",
    "MaceSurrogate",
    "Surrogate",
    "SurrogateCapabilities",
    "SurrogatePrediction",
    "TrainReport",
    "TrainableSurrogate",
    "assert_compatible_energy_contract",
    "surrogate_capabilities",
]
