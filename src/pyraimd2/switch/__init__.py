"""Switch protocol and implementations (design doc §6)."""

from pyraimd2.switch.base import Decision, LabelObservation, Route, Switch
from pyraimd2.switch.conformal import ConformalSwitch, conformal_quantile
from pyraimd2.switch.replay import ReplayRecord, ReplaySummary, replay
from pyraimd2.switch.scheduled import ScheduledSwitch

__all__ = [
    "ConformalSwitch",
    "Decision",
    "LabelObservation",
    "ReplayRecord",
    "ReplaySummary",
    "Route",
    "ScheduledSwitch",
    "Switch",
    "conformal_quantile",
    "replay",
]
