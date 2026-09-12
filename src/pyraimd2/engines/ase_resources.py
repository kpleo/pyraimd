"""Declared immutable file resources: content identity and run baselines.

An ASE-adapter backend may declare its calculator's top-level immutable
file parameters (``file_parameters={parameter: role}``).  A declared slot
is identified by its content — the full SHA-256 read from the bytes at
run setup — and the path stays out of the versioned identity payload
(``ase-file-identity-v1``).  Undeclared parameters keep the pre-existing
fingerprint semantics exactly (including path-keyed embedded-file
hashing); the declaration only ever narrows the declared slots.

Run baselines (``file_resources.json``, schema ``file-resource-baseline-v1``)
record, per ``<section>.<role>`` resource: the backend name, the declared
parameter, the same-named factory option, the role, the original path and
the content digest.  The baseline is collected only from the actually
constructed adapters, and each declared parameter is checked against the
configuration's same-named option so a baseline can never record file A
while the calculator reads file B.  All digests come from streamed byte
reads — never from the legacy path/mtime/size stat cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

FILE_IDENTITY_FORMAT = "ase-file-identity-v1"
FILE_RESOURCE_BASELINE_SCHEMA = "file-resource-baseline-v1"
BASELINE_FILENAME = "file_resources.json"


@dataclass(frozen=True)
class DeclaredFileResource:
    """One declared immutable file parameter, content-identified."""

    parameter: str
    role: str
    path: str
    sha256: str


def read_file_sha256(path: str | Path) -> str:
    """Full content SHA-256 from a streamed byte read — never cached by
    path/mtime/size.  Raises OSError when the file is unreadable."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resource_declaration(
        parameters: object,
        file_parameters: object) -> tuple[DeclaredFileResource, ...]:
    """Validate one adapter's declaration at opt-in time; refuse loudly.

    Every declared key must name a top-level parameter whose value is a
    path to an existing, readable, regular file (never a directory, link,
    or device); roles are nonempty and unique.  Returns the immutable
    resource view (with the content digest already read).  Raises
    ValueError on any violation — an opt-in declaration is never silently
    downgraded.
    """
    if not isinstance(parameters, dict):
        raise TypeError(
            "file_parameters requires a calculator with an identifiable "
            "parameter dict; this calculator's effective state cannot be "
            "identified, so the opt-in file identity is refused")
    if not isinstance(file_parameters, dict) or not file_parameters:
        raise ValueError(
            f"file_parameters must be a nonempty dict of "
            f"{{parameter: role}}, got {file_parameters!r}")
    resources = []
    seen_roles: set[str] = set()
    for parameter, role in file_parameters.items():
        if not isinstance(parameter, str) or not parameter:
            raise ValueError(
                f"file_parameters keys must be nonempty parameter names, "
                f"got {parameter!r}")
        if not isinstance(role, str) or not role:
            raise ValueError(
                f"file_parameters[{parameter!r}]: the role must be a "
                f"nonempty string, got {role!r}")
        if role in seen_roles:
            raise ValueError(
                f"file_parameters: role {role!r} is declared twice; roles "
                "must be unique within one backend")
        seen_roles.add(role)
        if parameter not in parameters:
            raise ValueError(
                f"file_parameters[{parameter!r}]: the calculator has no "
                f"such top-level parameter (known: {sorted(parameters)})")
        value = parameters[parameter]
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(
                f"file_parameters[{parameter!r}]: the parameter value must "
                f"be a file path, got {type(value).__name__}")
        path = Path(value)
        if path.is_symlink():
            raise ValueError(
                f"file_parameters[{parameter!r}]: {path} is a symlink; the "
                "first version binds regular files only")
        if not path.is_file():
            raise ValueError(
                f"file_parameters[{parameter!r}]: {path} is not an existing "
                "regular file")
        try:
            sha256 = read_file_sha256(path)
        except OSError as error:
            raise ValueError(
                f"file_parameters[{parameter!r}]: {path} is not readable "
                f"({error})") from error
        resources.append(DeclaredFileResource(
            parameter=parameter, role=role, path=str(path), sha256=sha256))
    return tuple(resources)


def collect_file_resource_baseline(config: object, *, engine: object | None,
                                   surrogate: object | None) -> str | None:
    """Collect and persist the immutable file-resource baseline for a new
    run; returns the baseline bytes' SHA-256, or None when no adapter
    declares resources (old-mode runs keep their structure untouched).

    The baseline is collected only from the actually constructed adapters.
    Each declared parameter must have a same-named backend option pointing
    at the same file — otherwise the run refuses BEFORE any backend
    evaluation: recording file A while the calculator reads file B is
    never written to disk.
    """
    run_dir = Path(config.run.directory)
    resources: dict[str, dict] = {}
    for section, adapter, backend in (("reference", engine,
                                       getattr(config, "reference", None)),
                                      ("surrogate", surrogate,
                                       getattr(config, "surrogate", None))):
        declared = () if adapter is None else getattr(adapter, "file_resources",
                                                      ())
        if not declared:
            continue
        options = (backend.options if backend is not None else {}) or {}
        for resource in declared:
            option = options.get(resource.parameter)
            if option is None:
                raise ValueError(
                    f"{section}.options has no entry {resource.parameter!r} "
                    f"matching the adapter's declared file parameter; the "
                    "baseline would record a file the configuration does not "
                    "name — refusing before any evaluation")
            option_path = Path(option)
            adapter_path = Path(resource.path)
            same = False
            try:
                same = os.path.samefile(option_path, adapter_path)
            except OSError:
                same = option_path == adapter_path
            if not same or not option_path.is_file() or \
                    read_file_sha256(option_path) != resource.sha256:
                raise ValueError(
                    f"{section}.options[{resource.parameter!r}] points at "
                    f"{option_path}, but the constructed calculator's "
                    f"parameter {resource.parameter!r} reads {adapter_path}; "
                    "the declared resource and the configured option must "
                    "name the same file with the same content — refusing "
                    "before any evaluation")
            resources[f"{section}.{resource.role}"] = {
                "backend": getattr(backend, "name", None),
                "parameter": resource.parameter,
                "option": resource.parameter,
                "role": resource.role,
                "original_path": resource.path,
                "sha256": resource.sha256,
                "identity_format": FILE_IDENTITY_FORMAT,
            }
    if not resources:
        return None
    payload = {
        "schema": FILE_RESOURCE_BASELINE_SCHEMA,
        "run_id": config.run.id,
        "resources": resources,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = run_dir / BASELINE_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return hashlib.sha256(text.encode()).hexdigest()


def file_resource_baseline_sha256(run_dir: str | Path) -> str | None:
    """The baseline bytes' SHA-256 for a run directory, or None when the
    run has no resource baseline (old-mode runs).  A corrupted or missing
    baseline reports honestly: missing → None, unreadable → OSError."""
    path = Path(run_dir) / BASELINE_FILENAME
    if not path.is_file():
        return None
    return read_file_sha256(path)


def verify_file_resource_baseline(run_dir: str | Path, *,
                                  expected_sha256: str, run_id: str) -> None:
    """Re-read and check the current baseline against the association
    carried by a valid checkpoint — the resume-time gate.

    The baseline file is re-read from its bytes every time (never the stat
    cache): missing, unreadable, unparsable, wrong schema, wrong run id, or
    a digest that no longer matches the checkpoint's recorded one all
    refuse, with the actual and expected values named.  Nothing is
    reconstructed from a side file and no legacy manifest substitutes.
    Raises ValueError on every violation.
    """
    path = Path(run_dir) / BASELINE_FILENAME
    if not path.is_file():
        raise ValueError(
            f"the run's file-resource baseline {path} is missing; the "
            "checkpoint carries a resource association, so the run cannot "
            "resume without its baseline file — restore the original file")
    try:
        actual_sha256 = read_file_sha256(path)
    except OSError as error:
        raise ValueError(
            f"the run's file-resource baseline {path} is unreadable "
            f"({error}); the checkpoint's recorded association cannot be "
            "verified — restore the original file") from error
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"the file-resource baseline {path} no longer matches the "
            f"checkpoint's recorded digest (current {actual_sha256[:16]}…, "
            f"recorded {expected_sha256[:16]}…); the baseline was modified "
            "or replaced — restore the original file or start a new run")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"the file-resource baseline {path} is corrupt ({error}); its "
            "digest still matches, which is contradictory — the run "
            "directory is inconsistent") from error
    schema = payload.get("schema")
    if schema != FILE_RESOURCE_BASELINE_SCHEMA:
        raise ValueError(
            f"the file-resource baseline {path} declares schema "
            f"{schema!r}, expected {FILE_RESOURCE_BASELINE_SCHEMA!r}")
    recorded_run_id = payload.get("run_id")
    if recorded_run_id != run_id:
        raise ValueError(
            f"the file-resource baseline {path} belongs to run "
            f"{recorded_run_id!r}, not {run_id!r}; the run directory is "
            "inconsistent")

