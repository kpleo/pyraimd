"""Guarded updates on a real MACE committee (slow integration).

Not executed in the minimal development environment (no torch): the markers
below keep it out of the default suite; run with ``--runslow`` where MACE is
installed. Capability note: ``CommitteeSurrogate.state_dict`` covers member
weights, energy shifts and the recipe echo — enough to publish and restore a
model artifact. Optimizer state (Adam moments, trainer RNG) is NOT saved, so
a fine-tune continuation restarts the optimizer rather than resuming it.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.loop import GuardedUpdater, UpdatePolicy
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.switch.base import LabelObservation

pytestmark = pytest.mark.slow


def _observation(label_id, position):
    atoms = Atoms("H", positions=[[position, 0, 0]])
    label = EngineResult(0.6 * position**2, [[-1.2 * position, 0, 0]], None, 0.0)
    prediction = SurrogatePrediction(0.0, np.zeros((1, 3)), None,
                                     np.full(1, np.nan))
    return LabelObservation(0, atoms, prediction, label, label_id=label_id)


def test_committee_guarded_update_publish_and_state_roundtrip():
    pytest.importorskip("mace")
    from pyraimd2.surrogate import CommitteeSurrogate

    committee = CommitteeSurrogate(model="small", n_members=2, epochs=2)
    updater = GuardedUpdater(committee, UpdatePolicy(n_label=2, guard_size=1))
    updater(_observation("L1", 0.20))
    updater(_observation("L2", 0.24))
    assert updater.n_updates + updater.n_rejected == 1
    state = updater.state_dict()
    # A full roundtrip: restore into a fresh updater and continue consuming.
    fresh = GuardedUpdater(CommitteeSurrogate(model="small", n_members=2,
                                              epochs=2),
                           UpdatePolicy(n_label=2, guard_size=1))
    fresh.load_state_dict(state)
    assert fresh.n_consumed == updater.n_consumed == 2
    fresh(_observation("L3", 0.28))
    assert fresh.n_consumed == 3
