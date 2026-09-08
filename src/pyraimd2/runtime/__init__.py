"""Runtime contracts and records: evaluation context, identities, event log,
label cache, checkpoints, updater protocol, cost aggregation and inspection."""

from pyraimd2.runtime.checkpoint import (
    CheckpointError,
    CheckpointManager,
    ResumeError,
)
from pyraimd2.runtime.context import EvaluationContext, EvaluationPhase
from pyraimd2.runtime.costs import summarize_tasks
from pyraimd2.runtime.events import EventLog, EventLogError
from pyraimd2.runtime.identity import fingerprint_of, model_id_for
from pyraimd2.runtime.inspect import format_inspection, inspect_run, summary_csv
from pyraimd2.runtime.labels import LabelCache, atoms_input_hash, label_key
from pyraimd2.runtime.models import ModelRegistry, ModelRegistryError
from pyraimd2.runtime.updater import StatefulUpdater

__all__ = [
    "CheckpointError",
    "CheckpointManager",
    "EvaluationContext",
    "EvaluationPhase",
    "EventLog",
    "EventLogError",
    "LabelCache",
    "ModelRegistry",
    "ModelRegistryError",
    "ResumeError",
    "StatefulUpdater",
    "atoms_input_hash",
    "fingerprint_of",
    "format_inspection",
    "inspect_run",
    "label_key",
    "model_id_for",
    "summarize_tasks",
    "summary_csv",
]
