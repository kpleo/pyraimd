"""Shared workflow setup: validation, backend creation, run-directory layout.

Everything here is used identically by the Python API and the CLI — the CLI
holds no logic of its own beyond argument parsing and exit codes.

Validation is layered:

1. :func:`pyraimd2.config.load_config` — TOML, schema, ranges, cross-field
   rules, path resolution (no I/O beyond reading the file).
2. :func:`validate_setup` — structure loads and is physically usable,
   path-like backend options exist, backends construct through the registry
   (parameter validation at the factory, no SCF/inference), declared
   capabilities are contract-compatible, and the run directory is free.
   ``probe=True`` additionally runs one small backend self-check on the
   structure — the only step that may execute external programs.

Run directories follow the plan §6 layout: ``config.toml`` (verbatim copy),
``resolved_config.json`` (effective parameters, absolute paths, units),
``manifest.json`` (run/input/backend identities), ``trajectory.db``,
``events.jsonl``, ``checkpoints/``, ``models/``, plus derived
``summary.json``/``summary.csv``/``trajectory.extxyz`` outputs.
"""

from __future__ import annotations

import hashlib
import inspect as _inspect
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from pyraimd2 import __version__
from pyraimd2.backends import available_backends, create_backend
from pyraimd2.backends.registry import ENGINE, SURROGATE, backend_capabilities
from pyraimd2.config import BackendConfig, PyramidConfig
from pyraimd2.engines.base import CapabilityMismatchError
from pyraimd2.runtime.inspect import _read_events, inspect_run
from pyraimd2.runtime.inspect import summary_csv as _summary_csv
from pyraimd2.store import Store
from pyraimd2.surrogate.base import assert_compatible_energy_contract
from pyraimd2.workflows.export import (
    append_extxyz,
    frame_from_row,
    frames_for_run,
    write_extxyz,
)


class WorkflowError(RuntimeError):
    """A workflow cannot start or continue; the message states the remedy."""


# ---------------------------------------------------------------------------
# structure


def load_structure(config: PyramidConfig) -> Atoms:
    """Read and sanity-check the configured structure.

    Constraints: the structure may carry FixAtoms and ``[constraints]
    fix_atoms_indices`` may add more; they merge into one fixed set.  Any
    other constraint kind is rejected explicitly (WP07 supports FixAtoms
    only).
    """
    path = config.structure.file
    if not path.is_file():
        raise WorkflowError(
            f"structure.file not found: {path}; write the structure there or "
            "point structure.file at an existing file (paths resolve relative "
            "to the configuration file)")
    try:
        atoms = ase_read(path)
    except Exception as error:
        raise WorkflowError(
            f"structure.file {path} could not be read ({error}); ASE reads "
            "extxyz, CIF, POSCAR, ... — check the file, not the working "
            "directory") from error
    if not len(atoms):
        raise WorkflowError(f"structure.file {path} contains no atoms")
    if not np.isfinite(atoms.positions).all():
        raise WorkflowError(
            f"structure.file {path}: non-finite positions; fix the structure")
    if np.any(atoms.get_masses() <= 0):
        raise WorkflowError(
            f"structure.file {path}: non-positive masses; set explicit masses")
    from pyraimd2.loop.constraints import ConstraintError, validate_constraints

    try:
        projection = validate_constraints(atoms)
    except ConstraintError as error:
        raise WorkflowError(f"structure.file {path}: {error}") from error
    configured = list(config.constraints.fix_atoms_indices)
    if configured:
        existing = [] if projection is None else list(projection.indices)
        merged = sorted(set(existing) | set(configured))
        if max(merged) >= len(atoms):
            raise WorkflowError(
                f"constraints.fix_atoms_indices {configured}: index out of "
                f"range for the {len(atoms)}-atom structure")
        from ase.constraints import FixAtoms

        atoms.set_constraint(FixAtoms(indices=merged))
    if "momenta" in atoms.arrays and not np.isfinite(atoms.get_momenta()).all():
        raise WorkflowError(
            f"structure.file {path}: non-finite momenta; fix or remove the "
            "velocity column")
    return atoms


# ---------------------------------------------------------------------------
# backends


_OPTIONAL_EXTRAS = {"mace": "mace", "pyscf": "pyscf"}


def _factory_run_kwargs(factory: Any, run_dir: Path | None) -> tuple[dict, Any]:
    """Factories that declare ``run_root`` (e.g. QE) receive the run's
    calculations directory — a throwaway root during validation, so
    ``pyramid validate`` never creates user-visible directories."""
    try:
        params = _inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return {}, None
    if "run_root" not in params:
        return {}, None
    if run_dir is not None:
        return {"run_root": run_dir / "calculations"}, None
    tmp = tempfile.TemporaryDirectory(prefix="pyraimd2-validate-")
    return {"run_root": tmp.name}, tmp


def create_configured_backend(section: str, config: BackendConfig, *,
                              run_dir: Path | None = None,
                              event_log: Any = None) -> object:
    """Create one configured backend through the registry.

    ``section`` ("reference"/"surrogate") pins the protocol kind.  Factories
    declaring ``event_log`` receive the run's log.  Optional-dependency
    imports and factory parameter errors surface as :class:`WorkflowError`
    naming the offending option.
    """
    kind = ENGINE if section == "reference" else SURROGATE
    registrations = available_backends()
    if config.name not in registrations:
        known = ", ".join(sorted(registrations))
        raise WorkflowError(
            f"{section}.backend {config.name!r} is not registered; available "
            f"backends: {known} (third-party backends register through the "
            "pyraimd2.backends entry-point group)")
    # Resolve the factory first so its signature decides run_root/event_log
    # passing; create_backend re-checks kind on the created object.
    from pyraimd2.backends import backend_factory

    try:
        factory = backend_factory(config.name)
    except ImportError as error:
        extra = _OPTIONAL_EXTRAS.get(config.name)
        hint = (f"install it with `pip install 'pyraimd2[{extra}]'`"
                if extra else "install the backend's package")
        raise WorkflowError(
            f"{section}.backend {config.name!r} needs an optional dependency "
            f"that is not installed ({error}); {hint}, or choose another "
            "backend") from error
    extra_kwargs, tmp = _factory_run_kwargs(factory, run_dir)
    try:
        if event_log is not None and "event_log" in _inspect.signature(factory).parameters:
            extra_kwargs["event_log"] = event_log
        return create_backend(config.name, kind=kind, **config.options,
                              **extra_kwargs)
    except TypeError as error:
        raise WorkflowError(
            f"{section}: backend {config.name!r} rejected its options "
            f"({error}); check the option names and types in [{section}] "
            "against the backend's documentation") from error
    finally:
        if tmp is not None:
            tmp.cleanup()


def backend_path_problems(section: str, config: BackendConfig) -> list[str]:
    """Missing path-like backend options (already resolved to absolute)."""
    problems = []
    for key, value in config.options.items():
        dotted = f"{section}.{key}"
        if key.endswith("_dir") and isinstance(value, str):
            if not Path(value).is_dir():
                problems.append(
                    f"{dotted}: directory not found: {value}; create it or "
                    f"fix {dotted} (resolved relative to the configuration "
                    "file)")
        elif isinstance(value, str) and (
                key.endswith(("_path", "_file"))
                or (key in ("model", "density_source") and Path(value).is_absolute())):
            if key == "density_source":
                if not Path(value).is_dir():
                    problems.append(
                        f"{dotted}: directory not found: {value}; point it at "
                        "a previous attempt directory holding a density "
                        "manifest, or remove the option for an atomic start")
            elif not Path(value).is_file():
                problems.append(
                    f"{dotted}: file not found: {value}; the backend needs "
                    "this file — download/generate it or fix the path")
        elif key == "pseudos" and isinstance(value, dict):
            for species, filename in value.items():
                if not Path(filename).is_file():
                    problems.append(
                        f"{dotted}.{species}: pseudopotential not found: "
                        f"{filename}; install the pseudopotential set or fix "
                        f"{section}.pseudos / {section}.pseudo_dir")
    return problems


def _resolved_pseudo_problems(engine: object | None, atoms: Atoms) -> list[str]:
    """QE-style engines resolve their full pseudopotential table (config
    defaults included) at construction; check every species the structure
    needs against pseudo_dir — before any SCF is attempted.  Other engines
    expose no ``config.pseudos``/``pseudo_dir`` and are skipped."""
    if engine is None:
        return []
    engine_config = getattr(engine, "config", None)
    pseudos = getattr(engine_config, "pseudos", None)
    pseudo_dir = getattr(engine_config, "pseudo_dir", None)
    if not isinstance(pseudos, dict) or pseudo_dir is None:
        return []
    problems = []
    for species in set(atoms.get_chemical_symbols()):
        filename = pseudos.get(species)
        if filename is None:
            problems.append(
                f"reference.pseudos: no pseudopotential configured for "
                f"species {species!r} (structure needs it); add it to "
                "reference.pseudos")
        elif not (Path(pseudo_dir) / filename).is_file():
            problems.append(
                f"reference.pseudos.{species}: pseudopotential not found: "
                f"{Path(pseudo_dir) / filename}; install the set or fix "
                "reference.pseudos / reference.pseudo_dir")
    return problems


def build_backends(config: PyramidConfig, *, run_dir: Path | None = None,
                   event_log: Any = None) -> tuple[object | None, object | None]:
    """(reference engine, surrogate) per the configured mode."""
    engine = surrogate = None
    if config.reference is not None:
        engine = create_configured_backend("reference", config.reference,
                                           run_dir=run_dir, event_log=event_log)
    if config.surrogate is not None:
        surrogate = create_configured_backend("surrogate", config.surrogate,
                                              run_dir=run_dir)
    return engine, surrogate


# ---------------------------------------------------------------------------
# validation


def _is_controller_materialized_input(config: PyramidConfig) -> bool:
    """Whether ``structure.file`` names the serial-recipe controller's
    materialized input: ``initial.traj`` inside the run directory (the
    exact shape ``run_serial_recipe`` owns and writes at stage start)."""
    try:
        return (config.structure.file.resolve()
                == (config.run.directory / "initial.traj").resolve())
    except OSError:
        return False


def validate_setup(config: PyramidConfig, *, probe: bool = False) -> dict:
    """Full pre-run validation; returns a human-presentable report dict.

    Default mode runs no SCF, no inference and downloads nothing; with
    ``probe=True`` each configured backend additionally evaluates the
    structure once as a self-check.

    A ``structure.file`` that names the serial-recipe controller's
    materialized input (``initial.traj`` in the run directory) is allowed
    to be absent at validate time: it is written by the controller at stage
    start, so the report marks the structure as deferred to run time and
    validation covers schema, backends and capabilities only.  Any other
    missing structure file stays an error, and ``probe=True`` always
    requires a loadable structure.
    """
    if config.task.kind not in ("singlepoint", "relax", "md"):
        raise WorkflowError(
            f"task.kind {config.task.kind!r} is not supported; choose "
            "singlepoint, relax or md")
    report: dict[str, Any] = {"config": config, "probes": {}}

    atoms = None
    try:
        atoms = load_structure(config)
    except WorkflowError as error:
        if "structure.file not found" in str(error) \
                and _is_controller_materialized_input(config):
            report["structure"] = {
                "deferred": (
                    "structure.file is absent and names the serial-recipe "
                    "controller's materialized input (initial.traj in the "
                    "run directory); it is written by the controller at "
                    "stage start and validated there.  Validation here "
                    "covers schema, backends and capabilities only — the "
                    "structure itself, species-pseudopotential coverage and "
                    "any --probe-backends evaluation run at stage start."),
            }
        else:
            raise
    else:
        report["structure"] = {
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            "pbc": [bool(p) for p in atoms.pbc],
            "has_momenta": "momenta" in atoms.arrays,
        }

    problems: list[str] = []
    for section in ("reference", "surrogate"):
        backend = getattr(config, section)
        if backend is not None:
            problems.extend(backend_path_problems(section, backend))
    if problems:
        raise WorkflowError("invalid path options:\n  - " + "\n  - ".join(problems))

    engine, surrogate = build_backends(config)
    if atoms is not None:
        problems.extend(_resolved_pseudo_problems(engine, atoms))
        if problems:
            raise WorkflowError("invalid backend paths:\n  - " + "\n  - ".join(problems))
    capabilities = {}
    for section, backend in (("reference", engine), ("surrogate", surrogate)):
        if backend is None:
            continue
        caps = backend_capabilities(backend)
        capabilities[section] = {
            "energy_kind": caps.energy_kind.value,
            "force_consistent": caps.force_consistent,
            "forces_conservative": caps.forces_conservative,
            "stress_available": caps.stress_available,
        }
        if section == "surrogate":
            capabilities[section]["uncertainty_available"] = getattr(
                caps, "uncertainty_available", False)
    report["capabilities"] = capabilities
    if engine is not None and surrogate is not None:
        try:
            mode = assert_compatible_energy_contract(
                backend_capabilities(engine), backend_capabilities(surrogate))
        except CapabilityMismatchError as error:
            raise WorkflowError(
                f"reference/surrogate energy contract: {error}; the "
                "combination needs each side's reported energy consistent "
                "with its forces, and a cross-kind combination needs both "
                "sides strictly verified") from error
        report["energy_contract"] = {
            "same_kind": "compatible",
            "cross_kind": "compatible_cross_kind",
            "unknown": "undeclared",
        }[mode]

    check_run_directory_available(config)

    if probe and atoms is None:
        raise WorkflowError(
            "--probe-backends evaluates the structure once per backend, so "
            "it needs a loadable structure.file; for a serial-recipe stage "
            "config the structure is materialized by the controller at "
            "stage start — validate without the probe, or probe the "
            "materialized input once the stage has run")
    if probe:
        for section, backend in (("reference", engine), ("surrogate", surrogate)):
            if backend is None:
                continue
            try:
                start = time.perf_counter()
                if section == "reference":
                    result = backend.compute(atoms)
                else:
                    result = backend.predict(atoms)
                elapsed = time.perf_counter() - start
            except ImportError as error:
                name = getattr(config, section).name
                extra = _OPTIONAL_EXTRAS.get(name)
                hint = (f"install it with `pip install 'pyraimd2[{extra}]'`"
                        if extra else "install the backend's package")
                raise WorkflowError(
                    f"--probe-backends: {section} backend {name!r} needs an "
                    f"optional dependency that is not installed ({error}); "
                    f"{hint}, or choose another backend") from error
            except Exception as error:
                raise WorkflowError(
                    f"--probe-backends: {section} self-check failed: {error}; "
                    "the backend could not evaluate the structure — check its "
                    "installation, input files and (for external programs) "
                    "that the executable is on PATH") from error
            if not np.isfinite(result.forces).all() or not np.isfinite(result.energy):
                raise WorkflowError(
                    f"--probe-backends: {section} returned non-finite "
                    "energy/forces on the structure; check the backend "
                    "configuration before running")
            report["probes"][section] = {"energy_eV": float(result.energy),
                                         "elapsed_s": elapsed}
    return report


def check_run_directory_available(config: PyramidConfig) -> None:
    """Refuse to mix runs: an existing event log or trajectory database in
    the run directory means a deliberate resume/fork, never a fresh run."""
    run_dir = config.run.directory
    db_path = run_dir / "trajectory.db"
    if db_path.is_file():
        with Store(db_path) as _store:
            existing = {str(row.key_value_pairs.get("run_id"))
                        for row in _store._db.select()}
        if config.run.id in existing:
            raise WorkflowError(
                f"run.id {config.run.id!r} already exists in {db_path}; a "
                "direct restart would silently lose anchors, check counts "
                "and cost records — continue with "
                f"`pyramid resume {run_dir} --steps N`, or choose a new "
                "run.id for a fresh run")
        raise WorkflowError(
            f"run directory {run_dir} already belongs to run(s) "
            f"{sorted(existing)}; choose a new run.directory (one directory "
            "per run keeps inspect/resume unambiguous)")
    if (run_dir / "events.jsonl").exists():
        raise WorkflowError(
            f"run directory {run_dir} already contains an event log; choose "
            "a new run.directory or resume the existing run")


# ---------------------------------------------------------------------------
# run-directory records and outputs


def _sha256_file(path: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str)
                   + "\n", encoding="utf-8")
    os.replace(tmp, path)


def prepare_run_directory(config: PyramidConfig, *, engine: object | None,
                          surrogate: object | None) -> Path:
    """Create the run directory with config copy, resolved config, manifest."""
    from pyraimd2.runtime.identity import fingerprint_of

    run_dir = config.run.directory
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.source_path is not None and Path(config.source_path).is_file():
        shutil.copy2(config.source_path, run_dir / "config.toml")
    _write_json_atomic(run_dir / "resolved_config.json", config.resolved_dict())
    manifest = {
        "run_id": config.run.id,
        "created_unix": time.time(),
        "pyraimd2_version": __version__,
        "schema_version": config.schema_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "task": {"kind": config.task.kind, "mode": config.task.mode},
        "config_sha256": _sha256_file(config.source_path),
        "structure": {"file": str(config.structure.file),
                      "sha256": _sha256_file(config.structure.file)},
        "reference": (None if engine is None else {
            "backend": config.reference.name,
            "fingerprint": fingerprint_of(engine)}),
        "surrogate": (None if surrogate is None else {
            "backend": config.surrogate.name,
            "fingerprint": fingerprint_of(surrogate)}),
    }
    if config.task.kind == "md":
        # The MD parameters and the effective random-stream identities are
        # part of the run record (M3A-4): NVT derives the velocity/bath (and,
        # in adaptive mode, the check) streams by fixed role
        # (role-derive-v1); NVE keeps the historical raw seeds.  The same
        # values appear in the RUN_START event's streams block.
        from pyraimd2.loop.integrators import STREAM_SCHEME, derive_stream_seed

        dynamics = config.dynamics
        nvt = dynamics.ensemble == "nvt"
        manifest["dynamics"] = {
            "ensemble": dynamics.ensemble,
            "integrator": dynamics.integrator,
            "timestep_fs": dynamics.timestep_fs,
            "temperature_K": dynamics.temperature_K,
            "friction_per_fs": dynamics.friction_per_fs,
            "thermostat_seed": dynamics.thermostat_seed,
            "velocity_seed": dynamics.velocity_seed,
        }
        streams = {
            "scheme": STREAM_SCHEME if nvt else None,
            "velocity_seed": (derive_stream_seed(dynamics.velocity_seed,
                                                 "velocity")
                              if nvt else dynamics.velocity_seed),
        }
        if nvt:
            raw_bath = (dynamics.thermostat_seed
                        if dynamics.thermostat_seed is not None
                        else config.run.seed)
            streams["thermostat_seed"] = derive_stream_seed(raw_bath,
                                                            "thermostat")
        if config.task.mode == "adaptive":
            streams["check_seed"] = (
                derive_stream_seed(config.verification.seed, "verification")
                if nvt else config.verification.seed)
        manifest["streams"] = streams
    _write_json_atomic(run_dir / "manifest.json", manifest)
    return run_dir


class RunOutputs:
    """Derived run-directory outputs, refreshed from the authoritative store.

    ``trajectory.extxyz`` (driving forces, thinned by
    ``output.trajectory_interval_steps``) is appended frame-by-frame during
    the run and fully regenerated at the end; ``summary.json`` and
    ``summary.csv`` are rewritten every ``output.summary_interval_steps``.
    Everything is idempotent, so a resume can regenerate before continuing.
    """

    def __init__(self, run_dir: Path, run_id: str, *,
                 trajectory_interval_steps: int = 1,
                 summary_interval_steps: int = 10) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.trajectory_interval = max(1, int(trajectory_interval_steps))
        self.summary_interval = max(1, int(summary_interval_steps))
        self.trajectory_path = self.run_dir / "trajectory.extxyz"

    def _store(self) -> Store:
        return Store(self.run_dir / "trajectory.db")

    def regenerate_trajectory(self) -> None:
        # Same commit→row selection as the CLI export (R7): orphan rows
        # never enter the user-visible trajectory.
        with self._store() as store:
            frames = frames_for_run(store, self.run_dir, self.run_id,
                                    force_source="driving",
                                    interval_steps=self.trajectory_interval)
        if frames:
            write_extxyz(self.trajectory_path, frames)

    def append_trajectory_step(self, completed_steps: int) -> None:
        """Append the frame for the evaluation completing step
        ``completed_steps`` (1-based) when the interval selects it."""
        evaluation_id = completed_steps
        if evaluation_id % self.trajectory_interval != 0:
            return
        with self._store() as store:
            # The frame comes from the commit-bound row of this evaluation,
            # not from whichever row happens to sit at the step first (R7).
            row = store.committed_row(
                _read_events(self.run_dir / "events.jsonl"), self.run_id,
                evaluation_id)
            append_extxyz(self.trajectory_path,
                          frame_from_row(row, self.run_id,
                                         force_source="driving",
                                         store=store))

    def write_summaries(self) -> None:
        info = inspect_run(self.run_dir, run_id=self.run_id)
        _write_json_atomic(self.run_dir / "summary.json", info)
        with self._store() as store:
            csv_text = _summary_csv(
                store, self.run_id,
                events=_read_events(self.run_dir / "events.jsonl"))
        tmp = self.run_dir / "summary.csv.tmp"
        tmp.write_text(csv_text, encoding="utf-8")
        os.replace(tmp, self.run_dir / "summary.csv")

    def finalize(self) -> None:
        """End-of-run refresh: full trajectory regeneration plus summaries."""
        self.regenerate_trajectory()
        self.write_summaries()
