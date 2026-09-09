"""Review C2 regression: array sidecar identity covers dtype, byte order,
shape and content; scalar (0-D) states keep their dimension
(INDEPENDENT_REVIEW_040_20260909 §C2)."""

from __future__ import annotations

import numpy as np
import pytest

from pyraimd2.runtime.models import (
    ModelRegistry,
    ModelRegistryError,
    array_placeholder,
    content_array_sink,
    content_array_source,
    dict_array_source,
    dump_state_arrays,
    load_state_arrays,
)


def test_same_bytes_different_shape_cannot_cross_load(tmp_path):
    sink = content_array_sink(tmp_path / "store")
    vector = dump_state_arrays({"w": np.zeros(4)}, sink)
    matrix = dump_state_arrays({"w": np.zeros((2, 2))}, sink)
    files = sorted((tmp_path / "store").glob("*.npz"))
    assert len(files) == 2  # identical bytes, different shape: two entries
    source = content_array_source(tmp_path / "store")
    with pytest.raises(ModelRegistryError):
        load_state_arrays({"w": dict(vector["w"],
                                     __ndarray__=matrix["w"]["__ndarray__"])},
                          source)


def test_same_bytes_different_dtype_cannot_cross_load(tmp_path):
    sink = content_array_sink(tmp_path / "store")
    floats = dump_state_arrays({"w": np.array([1.0])}, sink)
    ints = dump_state_arrays({"w": np.array([4607182418800017408],
                                            dtype=np.uint64)}, sink)
    files = sorted((tmp_path / "store").glob("*.npz"))
    assert len(files) == 2
    source = content_array_source(tmp_path / "store")
    restored = load_state_arrays(floats, source)["w"]
    assert restored.dtype == np.float64
    assert restored[0] == 1.0
    with pytest.raises(ModelRegistryError):
        load_state_arrays({"w": ints["w"] | {}}  # placeholder of the ints
                          | {"w": {**ints["w"],
                                   "sha256": floats["w"]["sha256"]}},
                          source)


def test_zero_d_state_keeps_scalar_shape(tmp_path):
    sink = content_array_sink(tmp_path / "store")
    scalar = np.array(2.5)
    converted = dump_state_arrays({"scale": scalar}, sink)
    assert converted["scale"]["shape"] == []
    assert converted["scale"]["dtype"] == "float64"
    restored = load_state_arrays(converted,
                                 content_array_source(tmp_path / "store"))
    assert restored["scale"].shape == ()
    assert float(restored["scale"]) == 2.5


def test_placeholder_identity_covers_shape_dtype_and_content(tmp_path):
    ph = array_placeholder("arr0", np.zeros((2, 2), dtype=">f8"))
    assert ph["shape"] == [2, 2]
    assert ph["dtype"] == ">f8"
    arrays = {"arr0": np.zeros((2, 2), dtype=">f8"),
              "other": np.zeros(4)}
    assert load_state_arrays({"w": ph}, dict_array_source(arrays))["w"].shape == (2, 2)
    with pytest.raises(ModelRegistryError):
        load_state_arrays(
            {"w": dict(ph, shape=[4])}, dict_array_source(arrays))
    with pytest.raises(ModelRegistryError):
        load_state_arrays(
            {"w": dict(ph, dtype="<f8")}, dict_array_source(arrays))
    with pytest.raises(ModelRegistryError):
        load_state_arrays(
            {"w": dict(ph, __ndarray__="other")}, dict_array_source(arrays))


def test_artifact_sidecar_enforces_identity(tmp_path):
    registry = ModelRegistry(tmp_path)
    state = {"surrogate": {"weights": np.zeros(4)}}
    registry.publish("m#g1", {"generation": 1, "parent_model_id": "m#g0",
                              "updater_state": state})
    resolved = registry.read("m#g1", resolve=True)
    assert resolved["updater_state"]["surrogate"]["weights"].shape == (4,)
    # A matrix payload under the same key cannot be swapped in: publish of a
    # different-shaped state under the same model id is not the same content.
    with pytest.raises(ModelRegistryError, match="immutable"):
        registry.publish("m#g1", {"generation": 1, "parent_model_id": "m#g0",
                                  "updater_state":
                                  {"surrogate": {"weights": np.zeros((2, 2))}}})
