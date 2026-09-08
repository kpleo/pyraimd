"""Complete-step checkpoints: atomic publish, checksums, previous-generation fallback.

Layout (plan §6): ``checkpoints/<generation>/{state.json, arrays.npz,
manifest.json}`` plus an atomically updated ``latest.json`` pointer.  A
checkpoint is written to a temporary directory, flushed, checksummed into its
manifest, and only then published by replacing the pointer; the previous
usable generation is kept.  Readers walk back from the pointer to the last
*valid* generation, so a truncated or torn checkpoint never blocks recovery.

A checkpoint records a complete integration-step boundary: full-step momenta,
real time, the committed evaluation id and the driving force reusable for the
next step — never the half-step momenta of a mid-step store row.  The Store
trajectory stays what it is: a force-evaluation log, not a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHECKPOINT_SCHEMA_VERSION = 1
KEEP_GENERATIONS = 2


class CheckpointError(RuntimeError):
    """A checkpoint cannot be written, read or validated."""


class ResumeError(RuntimeError):
    """A run cannot be resumed honestly from its persisted state."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass
class Checkpoint:
    """One validated checkpoint: manifest plus payload."""

    generation: int
    directory: Path
    manifest: dict
    state: dict
    arrays: dict[str, np.ndarray]


class CheckpointManager:
    """Writes and reads checkpoint generations for one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.directory = self.run_dir / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _generation_dir(self, generation: int) -> Path:
        return self.directory / str(generation)

    def next_generation(self) -> int:
        generations = [int(p.name) for p in self.directory.iterdir()
                       if p.name.isdigit()]
        return max(generations, default=0) + 1

    def write(self, generation: int, state: dict, arrays: dict[str, np.ndarray],
              manifest_extra: dict) -> Path:
        """Publish one checkpoint generation atomically.

        Payload first (temp directory, fsynced, checksummed into the
        manifest), then the ``latest.json`` pointer is replaced — readers
        never observe a half-written generation.
        """
        tmp = self.directory / f".tmp-{generation}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        try:
            arrays_path = tmp / "arrays.npz"
            with arrays_path.open("wb") as fh:
                np.savez(fh, **arrays)
                fh.flush()
                os.fsync(fh.fileno())
            state_path = tmp / "state.json"
            state_path.write_text(json.dumps(state, sort_keys=True))
            with state_path.open("rb") as fh:
                os.fsync(fh.fileno())
            manifest = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "generation": int(generation),
                "files": {"state.json": _sha256(state_path),
                          "arrays.npz": _sha256(arrays_path)},
                **manifest_extra,
            }
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with manifest_path.open("rb") as fh:
                os.fsync(fh.fileno())
            _fsync_dir(tmp)
            final = self._generation_dir(generation)
            if final.exists():
                shutil.rmtree(final)
            os.replace(tmp, final)
            pointer = self.directory / "latest.json"
            pointer_tmp = self.directory / "latest.json.tmp"
            pointer_tmp.write_text(json.dumps({"generation": int(generation)}))
            os.replace(pointer_tmp, pointer)
            _fsync_dir(self.directory)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._prune()
        return final

    def _prune(self) -> None:
        generations = sorted(int(p.name) for p in self.directory.iterdir()
                              if p.name.isdigit())
        for generation in generations[:-KEEP_GENERATIONS]:
            shutil.rmtree(self._generation_dir(generation), ignore_errors=True)

    def _read_generation(self, generation: int) -> Checkpoint | None:
        """Return the checkpoint if its manifest and checksums validate."""
        directory = self._generation_dir(generation)
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            if int(manifest["generation"]) != generation:
                return None
            files = manifest["files"]
            if _sha256(directory / "state.json") != files["state.json"]:
                return None
            if _sha256(directory / "arrays.npz") != files["arrays.npz"]:
                return None
            state = json.loads((directory / "state.json").read_text())
            with np.load(directory / "arrays.npz") as npz:
                arrays = {key: npz[key] for key in npz.files}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        return Checkpoint(generation, directory, manifest, state, arrays)

    def read_latest_valid(self) -> Checkpoint | None:
        """Last *valid* checkpoint, walking back over corrupt generations."""
        pointer = self.directory / "latest.json"
        if not pointer.exists():
            return None
        try:
            latest = int(json.loads(pointer.read_text())["generation"])
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            raise CheckpointError(f"unreadable checkpoint pointer: {error}") from error
        generations = sorted((int(p.name) for p in self.directory.iterdir()
                              if p.name.isdigit()), reverse=True)
        for generation in generations:
            if generation > latest:
                continue
            checkpoint = self._read_generation(generation)
            if checkpoint is not None:
                return checkpoint
        return None


def rng_state_to_json(bit_state: dict) -> dict:
    """numpy bit-generator state dict -> JSON-safe (numpy ints to ints)."""
    def sanitize(value: object) -> object:
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value
    return sanitize(bit_state)
