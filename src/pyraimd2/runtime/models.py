"""Immutable model artifacts under ``models/<model_id>/``.

Every published update produces one artifact: parent model, the durable
label IDs it trained on, the recipe, the training cost, and the updater's
continuation state. Artifacts are written once and never edited — a second
publish under the same model ID must carry identical content, otherwise it
is an error (history is never clobbered). The write is atomic: temporary
file, fsync, then rename, so readers never see a half-written artifact.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

MODEL_ARTIFACT_FORMAT_VERSION = 1


class ModelRegistryError(RuntimeError):
    """A model artifact cannot be published or read honestly."""


class ModelRegistry:
    """Publishes and reads immutable per-model artifacts for one run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.directory = Path(run_dir) / "models"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _artifact_dir(self, model_id: str) -> Path:
        return self.directory / model_id.replace("/", "_")

    def publish(self, model_id: str, record: dict) -> Path:
        """Write ``models/<model_id>/state.json`` atomically and immutably.

        ``record`` should carry generation, parent_model_id, label_ids,
        recipe, training and updater_state; the registry adds the identity
        fields. Re-publishing identical content is a no-op; different
        content under the same ID raises.
        """
        payload = {
            "model_id": model_id,
            "format_version": MODEL_ARTIFACT_FORMAT_VERSION,
            "written_unix": time.time(),
            **record,
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
            return path
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "state.json.tmp"
        tmp.write_text(json.dumps(payload, sort_keys=True))
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path

    def read(self, model_id: str) -> dict | None:
        """Return the artifact record, or None when missing/corrupt."""
        path = self._artifact_dir(model_id) / "state.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None
