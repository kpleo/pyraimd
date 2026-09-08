"""CheckpointManager mechanics: atomic publish, pruning, corrupt fallback."""

from __future__ import annotations

import json

import numpy as np

from pyraimd2.runtime.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManager,
    rng_state_to_json,
)


def _payload(tag: int):
    state = {"run_id": "run", "tag": tag,
             "check_rng": rng_state_to_json(np.random.default_rng(1).bit_generator.state)}
    arrays = {"positions": np.full((2, 3), float(tag)),
              "momenta": np.ones((2, 3)) * tag}
    manifest = {"run_id": "run", "nsteps": tag, "last_event_seq": tag * 10}
    return state, arrays, manifest


def test_write_read_roundtrip_and_latest_pointer(tmp_path):
    manager = CheckpointManager(tmp_path)
    state, arrays, manifest = _payload(1)
    manager.write(1, state, arrays, manifest)
    checkpoint = manager.read_latest_valid()
    assert checkpoint is not None and checkpoint.generation == 1
    assert checkpoint.manifest["last_event_seq"] == 10
    assert checkpoint.manifest["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint.state["tag"] == 1
    np.testing.assert_array_equal(checkpoint.arrays["positions"], np.ones((2, 3)))
    assert isinstance(checkpoint.state["check_rng"]["state"]["state"], int)


def test_keep_two_generations_and_prune(tmp_path):
    manager = CheckpointManager(tmp_path)
    for generation in range(1, 5):
        manager.write(generation, *_payload(generation))
    remaining = sorted(p.name for p in (tmp_path / "checkpoints").iterdir()
                       if p.name.isdigit())
    assert remaining == ["3", "4"]
    assert manager.read_latest_valid().generation == 4


def test_truncated_checkpoint_falls_back_to_previous(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.write(1, *_payload(1))
    manager.write(2, *_payload(2))
    arrays = tmp_path / "checkpoints" / "2" / "arrays.npz"
    with arrays.open("r+b") as fh:  # tear the payload: checksum must fail
        fh.truncate(16)
    checkpoint = manager.read_latest_valid()
    assert checkpoint is not None and checkpoint.generation == 1


def test_tampered_state_detected(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.write(1, *_payload(1))
    state_path = tmp_path / "checkpoints" / "1" / "state.json"
    state_path.write_text(json.dumps({"run_id": "run", "tag": 999}))
    assert manager.read_latest_valid() is None


def test_atomic_publish_leaves_no_temp_dirs(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.write(1, *_payload(1))
    manager.write(2, *_payload(2))
    names = [p.name for p in (tmp_path / "checkpoints").iterdir()]
    assert not any(name.startswith(".tmp") for name in names)
    assert json.loads((tmp_path / "checkpoints" / "latest.json").read_text()) == {
        "generation": 2}
