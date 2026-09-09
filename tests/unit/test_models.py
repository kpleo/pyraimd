"""ModelRegistry: immutable model artifacts (hermetic)."""

from __future__ import annotations

import pytest

from pyraimd2.runtime.models import ModelRegistry, ModelRegistryError


def _record(tag: int) -> dict:
    return {
        "generation": tag,
        "parent_model_id": f"model#g{tag - 1}",
        "label_ids": [f"run-label-{tag}"],
        "recipe": {"n_label": 3},
        "training": {"wall_time_s": 0.0},
        "updater_state": {"k": 1.0 + tag},
    }


def test_publish_read_roundtrip(tmp_path):
    registry = ModelRegistry(tmp_path)
    payload = registry.publish("model#g1", _record(1))
    artifact = registry.read("model#g1")
    assert artifact["parent_model_id"] == "model#g0"
    assert artifact["label_ids"] == ["run-label-1"]
    assert artifact["updater_state"] == {"k": 2.0}
    assert artifact["model_id"] == "model#g1"
    assert payload == artifact  # publish returns the stored payload
    assert (tmp_path / "models" / "model#g1" / "state.json").is_file()
    assert not list((tmp_path / "models").rglob("*.tmp"))


def test_republish_identical_is_idempotent_but_different_content_rejected(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.publish("model#g1", _record(1))
    registry.publish("model#g1", _record(1))  # identical: no-op
    with pytest.raises(ModelRegistryError, match="immutable"):
        registry.publish("model#g1", _record(2))


def test_missing_or_corrupt_artifact_reads_none(tmp_path):
    registry = ModelRegistry(tmp_path)
    assert registry.read("model#g9") is None
    directory = tmp_path / "models" / "model#g9"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text("{not json")
    assert registry.read("model#g9") is None


def test_model_id_with_slash_is_sanitized(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.publish("a/b#g1", _record(1))
    assert (tmp_path / "models" / "a_b#g1" / "state.json").exists()
    assert registry.read("a/b#g1")["generation"] == 1
