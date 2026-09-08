"""Runtime contracts: evaluation context and model/reference identity."""

from pyraimd2.runtime.context import EvaluationContext, EvaluationPhase
from pyraimd2.runtime.identity import fingerprint_of, model_id_for

__all__ = [
    "EvaluationContext",
    "EvaluationPhase",
    "fingerprint_of",
    "model_id_for",
]
