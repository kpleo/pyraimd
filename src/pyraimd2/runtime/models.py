"""Immutable model artifacts under ``models/<model_id>/``.

Every published update produces one artifact: parent model, the durable
label IDs it trained on, the recipe, the training cost, and the updater's
continuation state. Artifacts are written once and never edited — a second
publish under the same model ID must carry identical content, otherwise it
is an error (history is never clobbered). The write is atomic: temporary
file, fsync, then rename, so readers never see a half-written artifact.

Integrity: :func:`artifact_digest` is computed over the artifact's canonical
content (everything but the volatile write time) and is bound into the
commit event and the checkpoint by the caller — outside the rewritable
artifact file itself. Loading verifies that digest, the schema version and
the parent/generation chain before any state is applied.

Tensor persistence ("tensor-artifact-v1"): updater states may carry real
arrays (NumPy, or torch tensors via a duck-typed ``detach().cpu().numpy()``
conversion — torch is never imported here). :func:`dump_state_arrays`
replaces every array with a JSON placeholder
``{"__ndarray__": key, "sha256", "dtype", "shape"}``; the array bytes live
in a sidecar:

- model artifacts: ``models/<model_id>/state_arrays.npz`` (keys arr0..arrN,
  written by :meth:`ModelRegistry.publish` before ``state.json``);
- checkpoints: the updater-state arrays ride in the checkpoint's own
  ``arrays.npz`` under ``updater_state:arrN`` keys (wired by the caller);
- event payloads (label consumption without a publish): a content-addressed
  store at ``models/state-arrays/<sha256>.npz`` — one file per unique
  array, so repeated identical states cost nothing and every reference is
  integrity-checked by its own digest.

:func:`load_state_arrays` resolves placeholders back to arrays and verifies
each ``sha256`` before returning, so a tampered sidecar fails loud.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

MODEL_ARTIFACT_FORMAT_VERSION = 1

ARRAY_PLACEHOLDER = "__ndarray__"


class ModelRegistryError(RuntimeError):
    """A model artifact cannot be published or read honestly."""


def artifact_digest(record: dict) -> str:
    """Content digest of an artifact record, excluding volatile metadata."""
    comparable = {key: value for key, value in record.items()
                  if key != "written_unix"}
    canonical = json.dumps(comparable, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:24]


def _as_array(value: object) -> np.ndarray | None:
    """Duck-typed array conversion: NumPy, or a torch tensor (without
    importing torch). Anything else returns None."""
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            return np.asarray(detach().cpu().numpy())
        except Exception:  # noqa: BLE001 — not actually a tensor
            return None
    return None


def array_placeholder(key: str, array: np.ndarray) -> dict:
    """Placeholder JSON dict for one sidecar-stored array."""
    return {ARRAY_PLACEHOLDER: key, "sha256": _array_digest(array),
            "dtype": str(array.dtype), "shape": [int(v) for v in array.shape]}


def dump_state_arrays(state: object, sink: object) -> object:
    """JSON-safe copy of ``state`` with every array replaced by the
    placeholder dict returned by ``sink(array)``."""
    array = _as_array(state)
    if array is not None:
        return sink(array)
    if isinstance(state, dict):
        return {str(key): dump_state_arrays(value, sink)
                for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return [dump_state_arrays(value, sink) for value in state]
    return state


def load_state_arrays(state: object, source: object) -> object:
    """Inverse of :func:`dump_state_arrays`: ``source(placeholder)`` must
    return the verified array for each placeholder."""
    if isinstance(state, dict):
        if state.get(ARRAY_PLACEHOLDER):
            return source(state)
        return {key: load_state_arrays(value, source)
                for key, value in state.items()}
    if isinstance(state, list):
        return [load_state_arrays(value, source) for value in state]
    return state


def content_array_sink(directory: str | Path):
    """Sink writing one content-addressed ``<sha256>.npz`` per unique array
    into ``directory`` (append-only; existing files are never rewritten)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def sink(array: np.ndarray) -> dict:
        array = np.ascontiguousarray(array)
        digest = _array_digest(array)
        path = directory / f"{digest}.npz"
        if not path.exists():
            tmp = directory / f".{digest}.tmp"
            np.savez(tmp, array=array)  # np.savez appends .npz
            with tmp.with_suffix(".tmp.npz").open("rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp.with_suffix(".tmp.npz"), path)
        return array_placeholder(digest, array)

    return sink


def content_array_source(directory: str | Path):
    """Source resolving content-addressed placeholders, verifying digests."""
    directory = Path(directory)

    def source(placeholder: dict) -> np.ndarray:
        path = directory / f"{placeholder[ARRAY_PLACEHOLDER]}.npz"
        try:
            array = np.load(path)["array"]
        except (OSError, KeyError, ValueError) as error:
            raise ModelRegistryError(
                f"state array {path} is missing or unreadable; the run "
                "directory is incomplete") from error
        if _array_digest(array) != placeholder.get("sha256"):
            raise ModelRegistryError(
                f"state array {path} does not match the digest bound into "
                "its reference; the store looks tampered with")
        return array

    return source


def dict_array_source(arrays: dict) -> object:
    """Source resolving placeholders from an in-memory arrays dict (the
    checkpoint's own ``arrays.npz`` content), verifying digests."""

    def source(placeholder: dict) -> np.ndarray:
        key = placeholder[ARRAY_PLACEHOLDER]
        if key not in arrays:
            raise ModelRegistryError(
                f"checkpoint arrays are missing {key!r}; the checkpoint is "
                "incomplete")
        array = np.asarray(arrays[key])
        if _array_digest(array) != placeholder.get("sha256"):
            raise ModelRegistryError(
                f"checkpoint array {key!r} does not match the digest bound "
                "into its reference; the checkpoint looks tampered with")
        return array

    return source


def write_arrays_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomic ``arrays.npz`` write (temporary file, fsync, rename)."""
    tmp = path.with_suffix(".tmp")
    np.savez(tmp, **arrays)  # writes <name>.tmp.npz
    tmp_npz = path.with_suffix(".tmp.npz")
    with tmp_npz.open("rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_npz, path)


def resolve_artifact_state(state: object, artifact_dir: str | Path) -> object:
    """Resolve placeholders against an artifact's ``state_arrays.npz``."""
    path = Path(artifact_dir) / "state_arrays.npz"
    store = dict(np.load(path)) if path.exists() else {}

    def source(placeholder: dict) -> np.ndarray:
        key = placeholder[ARRAY_PLACEHOLDER]
        if key not in store:
            raise ModelRegistryError(
                f"artifact arrays at {path} are missing {key!r}; the "
                "artifact is incomplete")
        array = store[key]
        if _array_digest(array) != placeholder.get("sha256"):
            raise ModelRegistryError(
                f"artifact array {key!r} does not match the digest bound "
                "into its reference; the artifact looks tampered with")
        return array

    return load_state_arrays(state, source)


class ModelRegistry:
    """Publishes and reads immutable per-model artifacts for one run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.directory = Path(run_dir) / "models"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _artifact_dir(self, model_id: str) -> Path:
        return self.directory / model_id.replace("/", "_")

    def publish(self, model_id: str, record: dict) -> dict:
        """Write ``models/<model_id>/state.json`` atomically and immutably.

        ``record`` should carry generation, parent_model_id, label_ids,
        recipe, training and updater_state; the registry adds the identity
        fields. Array values (real tensor states) go to
        ``state_arrays.npz`` next to the JSON, which keeps digest
        placeholders. Re-publishing identical content is a no-op; different
        content under the same ID raises. Returns the stored payload (the
        placeholder form the caller binds into commit digests).
        """
        arrays: dict[str, np.ndarray] = {}

        def sink(array: np.ndarray) -> dict:
            key = f"arr{len(arrays)}"
            arrays[key] = np.ascontiguousarray(array)
            return array_placeholder(key, arrays[key])

        payload = {
            "model_id": model_id,
            "format_version": MODEL_ARTIFACT_FORMAT_VERSION,
            "written_unix": time.time(),
            **dump_state_arrays(record, sink),
        }
        # written_unix is wall-clock metadata, not identity: compare content
        # without it for the immutability check.
        directory = self._artifact_dir(model_id)
        path = directory / "state.json"
        if path.exists():
            existing = json.loads(path.read_text())
            comparable = {key: value for key, value in existing.items()
                          if key != "written_unix"}
            incoming = {key: value for key, value in payload.items()
                        if key != "written_unix"}
            if comparable != incoming:
                raise ModelRegistryError(
                    f"model artifact for {model_id!r} already exists with "
                    "different content; artifacts are immutable")
            return existing
        directory.mkdir(parents=True, exist_ok=True)
        if arrays:
            write_arrays_npz(directory / "state_arrays.npz", arrays)
        tmp = directory / "state.json.tmp"
        tmp.write_text(json.dumps(payload, sort_keys=True))
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return payload

    def read(self, model_id: str, *, resolve: bool = False) -> dict | None:
        """Return the artifact record, or None when missing/corrupt.

        The default is the stored placeholder form (what
        :func:`artifact_digest` covers); ``resolve=True`` additionally
        resolves array placeholders against ``state_arrays.npz``, verifying
        each digest.
        """
        artifact_dir = self._artifact_dir(model_id)
        path = artifact_dir / "state.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if resolve:
            for key in ("updater_state",):
                if record.get(key) is not None:
                    record[key] = resolve_artifact_state(record[key],
                                                         artifact_dir)
        return record
