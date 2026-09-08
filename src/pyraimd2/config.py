"""TOML configuration for Pyramid runs: parsing, validation, path resolution.

One configuration language (TOML, standard-library ``tomllib``), one schema
version, explicit errors.  Units live in the field names (``timestep_fs``,
``force_budget_eV_A``) and follow the package-wide contract: angstrom, eV,
eV/angstrom, fs, K.

Design rules:

- **Unknown fields are errors, never silently ignored.**  Every section and
  key is matched against the schema; a typo fails at load time with the
  dotted field name, a close-match suggestion and a remedy.  This is what
  makes schema migrations safe later.
- **Relative paths resolve against the configuration file's directory**, so
  running from a different working directory writes results to the same
  place.  Directories containing spaces are fine (paths are never shelled
  out).
- **Schema migration is explicit.**  ``schema_version`` is checked; a newer
  version is rejected with a pointer to the migration notes in
  ``docs/configuration.md`` rather than parsed leniently.  When version 2
  arrives, a ``_migrate_1_to_2`` function will translate documents before
  validation — there is no silent keyword swallowing at any version.

Validation here is *configuration-level*: types, ranges, cross-field rules
and path resolution.  Structure loading and backend construction happen one
level up in :mod:`pyraimd2.workflows` so ``load_config`` stays I/O-light.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 1

TASK_KINDS = ("singlepoint", "relax", "md")
TASK_MODES = ("reference", "surrogate", "adaptive")
ENSEMBLES = ("nve",)
POLICY_NAMES = ("energetic",)

# Backend option keys whose string values are filesystem paths, resolved
# against the configuration file's directory and checked at validate time.
_PATH_VALUE_KEYS = ("model", "density_source")
_PSEUDO_DICT_KEY = "pseudos"  # QE species -> filename, resolved under pseudo_dir
_MODEL_FILE_SUFFIXES = (".model", ".pt", ".pth", ".ckpt", ".json")


class ConfigError(ValueError):
    """A configuration cannot be parsed or validated.

    The message always names the dotted field (e.g. ``dynamics.timestep_fs``)
    and states the remedy.
    """


@dataclass(frozen=True)
class RunConfig:
    id: str
    directory: Path
    seed: int


@dataclass(frozen=True)
class TaskConfig:
    kind: str
    mode: str


@dataclass(frozen=True)
class StructureConfig:
    file: Path


@dataclass(frozen=True)
class DynamicsConfig:
    ensemble: str
    timestep_fs: float
    steps: int
    temperature_K: float
    velocity_seed: int


@dataclass(frozen=True)
class BackendConfig:
    """A backend name plus its factory options (``backend = "qe"`` etc.)."""

    name: str
    options: dict[str, Any]


@dataclass(frozen=True)
class PolicyConfig:
    force_budget_eV_A: float
    probe_steps_A: tuple[float, float]
    numerical_floor_eV_A: float
    time_cap_fs: float
    transverse_cap: float


@dataclass(frozen=True)
class VerificationConfig:
    """Independent-check segment settings.  ``probability == 0`` disables
    verification; the remaining fields are then unused (and rejected when
    set explicitly, so a disabled segment carries no pretend parameters)."""

    probability: float
    failure_probability: float
    tilt: float
    seed: int


@dataclass(frozen=True)
class CheckpointConfig:
    interval_steps: int
    keep_generations: int


@dataclass(frozen=True)
class OutputConfig:
    trajectory_interval_steps: int
    summary_interval_steps: int


@dataclass(frozen=True)
class PyramidConfig:
    """Fully parsed and validated run configuration."""

    schema_version: int
    run: RunConfig
    task: TaskConfig
    structure: StructureConfig
    dynamics: DynamicsConfig
    reference: BackendConfig | None
    surrogate: BackendConfig | None
    policy: PolicyConfig | None
    verification: VerificationConfig
    checkpoint: CheckpointConfig
    output: OutputConfig
    source_path: Path | None  # the file this configuration was loaded from

    def resolved_dict(self) -> dict[str, Any]:
        """JSON-safe record of every parameter actually in effect.

        Written to ``resolved_config.json`` at run start; ``resume`` rebuilds
        the configuration from it.  Paths are absolute, units are stated.
        """

        def backend_section(section: BackendConfig | None) -> dict | None:
            if section is None:
                return None
            return {"backend": section.name, "options": dict(section.options)}

        return {
            "schema_version": self.schema_version,
            "units": {"length": "angstrom", "energy": "eV",
                      "force": "eV/angstrom", "time": "fs", "temperature": "K"},
            "run": {"id": self.run.id, "directory": str(self.run.directory),
                    "seed": self.run.seed},
            "task": {"kind": self.task.kind, "mode": self.task.mode},
            "structure": {"file": str(self.structure.file)},
            "dynamics": {"ensemble": self.dynamics.ensemble,
                         "timestep_fs": self.dynamics.timestep_fs,
                         "steps": self.dynamics.steps,
                         "temperature_K": self.dynamics.temperature_K,
                         "velocity_seed": self.dynamics.velocity_seed},
            "reference": backend_section(self.reference),
            "surrogate": backend_section(self.surrogate),
            "policy": (None if self.policy is None else {
                "name": "energetic",
                "force_budget_eV_A": self.policy.force_budget_eV_A,
                "probe_steps_A": list(self.policy.probe_steps_A),
                "numerical_floor_eV_A": self.policy.numerical_floor_eV_A,
                "time_cap_fs": self.policy.time_cap_fs,
                "transverse_cap": self.policy.transverse_cap,
            }),
            "verification": (None if self.task.mode != "adaptive" else {
                "probability": self.verification.probability,
                "failure_probability": self.verification.failure_probability,
                "tilt": self.verification.tilt,
                "seed": self.verification.seed,
            }),
            "checkpoint": {"interval_steps": self.checkpoint.interval_steps,
                           "keep_generations": self.checkpoint.keep_generations},
            "output": {"trajectory_interval_steps": self.output.trajectory_interval_steps,
                       "summary_interval_steps": self.output.summary_interval_steps},
        }


# ---------------------------------------------------------------------------
# document loading


def load_config(path: str | Path) -> PyramidConfig:
    """Load and validate a TOML configuration file.

    Relative input paths (``structure.file``, ``run.directory`` and path-like
    backend options) resolve against the file's directory.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise ConfigError(
            f"configuration file not found: {path}; pass the path written by "
            "`pyramid init` (e.g. my_run/run.toml)")
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error
    return parse_config(document, base_dir=path.resolve().parent,
                        source_path=path.resolve())


def load_resolved_config(run_dir: str | Path) -> PyramidConfig:
    """Rebuild the configuration recorded at run start (``resume`` source)."""
    run_dir = Path(run_dir)
    resolved = run_dir / "resolved_config.json"
    if not resolved.is_file():
        raise ConfigError(
            f"no resolved_config.json in {run_dir}; this is not a run directory "
            "created by `pyramid run` — pass the run directory (the one "
            "containing events.jsonl), not the project directory")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"corrupt {resolved}: {error}") from error
    document = dict(document)
    document.pop("units", None)  # informational block, not schema
    # Sections not in effect for the run's mode are recorded as null
    for section in ("reference", "surrogate", "policy", "verification"):
        if section in document and document[section] is None:
            del document[section]
    # The resolved record nests backend options under "options"; flatten back
    # to the document shape the parser expects.
    for section in ("reference", "surrogate"):
        value = document.get(section)
        if isinstance(value, dict) and "options" in value:
            document[section] = {"backend": value["backend"], **value["options"]}
    return parse_config(document, base_dir=run_dir.resolve(), source_path=resolved)


def parse_config(document: dict[str, Any], *, base_dir: Path,
                 source_path: Path | None = None) -> PyramidConfig:
    """Validate a raw document (from TOML or resolved JSON) into a config."""
    if not isinstance(document, dict):
        raise ConfigError("configuration must be a TOML table")
    version = document.pop("schema_version", CONFIG_SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConfigError(
            f"schema_version must be an integer, got {version!r}; "
            f"this pyraimd2 reads schema_version {CONFIG_SCHEMA_VERSION}")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version {version} is not supported by this pyraimd2 "
            f"(reads {CONFIG_SCHEMA_VERSION}); do not edit the file to bypass "
            "this — follow the migration notes in docs/configuration.md")
    _reject_unknown(document, _SECTIONS, "", "section")

    run = _parse_run(_section(document, "run", required=True), base_dir)
    task = _parse_task(_section(document, "task", required=True))
    structure = _parse_structure(_section(document, "structure", required=True),
                                 base_dir)
    dynamics = _parse_dynamics(_section(document, "dynamics", required=True),
                               run.seed)
    reference = _parse_backend(_section(document, "reference"), "reference",
                               base_dir)
    surrogate = _parse_backend(_section(document, "surrogate"), "surrogate",
                               base_dir)
    policy = _parse_policy(_section(document, "policy"))
    verification_table = _section(document, "verification")
    verification = _parse_verification(verification_table, run.seed)
    checkpoint = _parse_checkpoint(_section(document, "checkpoint"))
    output = _parse_output(_section(document, "output"))
    _check_task_compatibility(task, reference=reference, surrogate=surrogate,
                              policy=policy,
                              verification_present=verification_table is not None)
    return PyramidConfig(
        schema_version=version, run=run, task=task, structure=structure,
        dynamics=dynamics, reference=reference, surrogate=surrogate,
        policy=policy, verification=verification, checkpoint=checkpoint,
        output=output, source_path=source_path)


# ---------------------------------------------------------------------------
# field helpers


_SECTIONS = ("run", "task", "structure", "dynamics", "reference", "surrogate",
             "policy", "verification", "checkpoint", "output")


def _reject_unknown(table: dict, known: tuple[str, ...] | list[str], prefix: str,
                    what: str) -> None:
    for key in table:
        if key not in known:
            where = f"{prefix}.{key}" if prefix else key
            suggestion = get_close_matches(str(key), [str(k) for k in known], n=1)
            hint = ""
            if suggestion:
                full = f"{prefix}.{suggestion[0]}" if prefix else suggestion[0]
                hint = f" did you mean {full!r}?"
            raise ConfigError(
                f"unknown {what} {where!r}:{hint} unknown fields are rejected "
                "rather than silently ignored — remove it or check "
                "docs/configuration.md for the supported schema")


def _section(document: dict, name: str, *, required: bool = False) -> dict | None:
    value = document.get(name)
    if value is None:
        if required:
            raise ConfigError(
                f"missing required section [{name}]; add it to the "
                "configuration (see docs/configuration.md or `pyramid init`)")
        return None
    if not isinstance(value, dict):
        raise ConfigError(
            f"section [{name}] must be a TOML table, got {value!r}; write it "
            f"as [{name}] with key = value lines")
    return dict(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_field(table: dict, key: str, prefix: str, *, default: int | None = None,
               minimum: int | None = None) -> int:
    if key not in table:
        if default is None:
            raise ConfigError(f"missing required field {prefix}.{key}")
        return default
    value = table.pop(key)
    if not _is_int(value):
        raise ConfigError(
            f"{prefix}.{key} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(
            f"{prefix}.{key} must be >= {minimum}, got {value}; choose a "
            f"larger value or remove the field to use the default")
    return int(value)


def _float_field(table: dict, key: str, prefix: str, *,
                 default: float | None = None, minimum: float | None = None,
                 allow_zero: bool = False, maximum: float | None = None) -> float:
    if key not in table:
        if default is None:
            raise ConfigError(f"missing required field {prefix}.{key}")
        return float(default)
    value = table.pop(key)
    if not _is_number(value):
        raise ConfigError(f"{prefix}.{key} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"{prefix}.{key} must be finite, got {value!r}")
    if minimum is not None and (value < minimum or (value == minimum and not allow_zero)):
        bound = f">= {minimum}" if allow_zero else f"> {minimum}"
        raise ConfigError(
            f"{prefix}.{key} must be {bound}, got {value}; fix the value in "
            "the configuration — units are part of the field name")
    if maximum is not None and value > maximum:
        raise ConfigError(
            f"{prefix}.{key} must be <= {maximum}, got {value}")
    return value


def _str_field(table: dict, key: str, prefix: str, *,
               default: str | None = None, choices: tuple[str, ...] | None = None) -> str:
    if key not in table:
        if default is None:
            raise ConfigError(f"missing required field {prefix}.{key}")
        return default
    value = table.pop(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(
            f"{prefix}.{key} must be a nonempty string, got {value!r}")
    if choices is not None and value not in choices:
        raise ConfigError(
            f"{prefix}.{key} must be one of {list(choices)}, got {value!r}; "
            "see docs/configuration.md for what is supported in this version")
    return value


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _path_field(table: dict, key: str, prefix: str, base_dir: Path) -> Path:
    value = table.pop(key, None)
    if not isinstance(value, str) or not value:
        raise ConfigError(
            f"{prefix}.{key} must be a nonempty path string, got {value!r}; "
            "paths are resolved relative to the configuration file")
    return _resolve_path(value, base_dir)


# ---------------------------------------------------------------------------
# sections


def _parse_run(table: dict, base_dir: Path) -> RunConfig:
    _reject_unknown(table, ("id", "directory", "seed"), "run", "field")
    run_id = _str_field(table, "id", "run")
    if "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ConfigError(
            f"run.id {run_id!r} must not contain path separators; it names "
            "the run inside its directory")
    directory = _path_field(table, "directory", "run", base_dir)
    seed = _int_field(table, "seed", "run", default=0)
    return RunConfig(id=run_id, directory=directory, seed=seed)


def _parse_task(table: dict) -> TaskConfig:
    _reject_unknown(table, ("kind", "mode"), "task", "field")
    kind = _str_field(table, "kind", "task", choices=TASK_KINDS)
    mode = _str_field(table, "mode", "task", choices=TASK_MODES)
    return TaskConfig(kind=kind, mode=mode)


def _parse_structure(table: dict, base_dir: Path) -> StructureConfig:
    _reject_unknown(table, ("file",), "structure", "field")
    return StructureConfig(file=_path_field(table, "file", "structure", base_dir))


def _parse_dynamics(table: dict, run_seed: int) -> DynamicsConfig:
    _reject_unknown(table,
                    ("ensemble", "timestep_fs", "steps", "temperature_K",
                     "velocity_seed"),
                    "dynamics", "field")
    ensemble = _str_field(table, "ensemble", "dynamics", default="nve",
                          choices=ENSEMBLES)
    timestep_fs = _float_field(table, "timestep_fs", "dynamics", minimum=0.0)
    steps = _int_field(table, "steps", "dynamics", minimum=1)
    temperature_K = _float_field(table, "temperature_K", "dynamics",
                                 default=300.0, minimum=0.0, allow_zero=True)
    velocity_seed = _int_field(table, "velocity_seed", "dynamics",
                               default=run_seed)
    return DynamicsConfig(ensemble=ensemble, timestep_fs=timestep_fs, steps=steps,
                          temperature_K=temperature_K, velocity_seed=velocity_seed)


def _parse_backend(table: dict | None, prefix: str, base_dir: Path) -> BackendConfig | None:
    if table is None:
        return None
    if "backend" not in table:
        raise ConfigError(
            f"missing required field {prefix}.backend; name a registered "
            "backend (see `pyramid backends` or docs/configuration.md)")
    name = table.pop("backend")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{prefix}.backend must be a nonempty string, got {name!r}")
    options = dict(table)
    # pseudo_dir scopes pseudos resolution, so resolve it before the loop
    # regardless of key order in the file.
    if isinstance(options.get("pseudo_dir"), str):
        options["pseudo_dir"] = str(_resolve_path(options["pseudo_dir"], base_dir))
    for key, value in options.items():
        if key == "pseudo_dir":
            continue
        if not isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            raise ConfigError(
                f"{prefix}.{key}: unsupported option value {value!r}; backend "
                "options must be plain TOML values")
        dotted = f"{prefix}.{key}"
        if isinstance(value, str) and (
                key.endswith(("_path", "_file", "_dir"))
                or (key in _PATH_VALUE_KEYS and _looks_like_path(key, value))):
            options[key] = str(_resolve_path(value, base_dir))
        elif key == _PSEUDO_DICT_KEY and isinstance(value, dict):
            options[key] = _resolve_pseudos(value, table_pseudo_dir=options.get("pseudo_dir"),
                                            base_dir=base_dir, dotted=dotted)
    return BackendConfig(name=name, options=options)


def _looks_like_path(key: str, value: str) -> bool:
    """Backend options that may be names *or* paths (e.g. mace ``model``):
    only resolve the value when it actually looks like a path — a bare model
    name like "small" stays a name."""
    return key != "model" or (
        "/" in value or "\\" in value
        or value.lower().endswith(_MODEL_FILE_SUFFIXES))


def _resolve_pseudos(pseudos: dict, *, table_pseudo_dir: Any, base_dir: Path,
                     dotted: str) -> dict[str, str]:
    pseudo_base = (Path(table_pseudo_dir) if table_pseudo_dir is not None
                   else base_dir)
    resolved = {}
    for species, filename in pseudos.items():
        if not isinstance(filename, str):
            raise ConfigError(
                f"{dotted}.{species} must be a pseudopotential filename, "
                f"got {filename!r}")
        path = Path(filename)
        resolved[str(species)] = str(path if path.is_absolute()
                                     else (pseudo_base / path).resolve())
    return resolved


def _parse_policy(table: dict | None) -> PolicyConfig | None:
    if table is None:
        return None
    _reject_unknown(table,
                    ("name", "force_budget_eV_A", "probe_steps_A",
                     "numerical_floor_eV_A", "time_cap_fs", "transverse_cap"),
                    "policy", "field")
    _str_field(table, "name", "policy", default="energetic",
               choices=POLICY_NAMES)
    force_budget = _float_field(table, "force_budget_eV_A", "policy", minimum=0.0)
    probe_steps = table.pop("probe_steps_A", [0.02, 0.04])
    if (not isinstance(probe_steps, list) or len(probe_steps) != 2
            or not all(_is_number(v) for v in probe_steps)):
        raise ConfigError(
            f"policy.probe_steps_A must be a list of two numbers in "
            f"angstrom, got {probe_steps!r}")
    probe = (float(probe_steps[0]), float(probe_steps[1]))
    if not all(math.isfinite(v) for v in probe) or not 0 < probe[0] < probe[1]:
        raise ConfigError(
            f"policy.probe_steps_A must be two increasing positive distances "
            f"(angstrom), got {list(probe)}")
    numerical_floor = _float_field(table, "numerical_floor_eV_A", "policy",
                                   default=0.0, minimum=0.0, allow_zero=True)
    time_cap = _float_field(table, "time_cap_fs", "policy", default=1.0,
                            minimum=0.0)
    transverse_cap = _float_field(table, "transverse_cap", "policy", default=0.1,
                                  minimum=0.0, allow_zero=True, maximum=1.0)
    return PolicyConfig(force_budget_eV_A=force_budget, probe_steps_A=probe,
                        numerical_floor_eV_A=numerical_floor, time_cap_fs=time_cap,
                        transverse_cap=transverse_cap)


def _parse_verification(table: dict | None, run_seed: int) -> VerificationConfig:
    if table is None:
        return VerificationConfig(probability=0.05, failure_probability=0.05,
                                  tilt=math.log(2.0), seed=run_seed)
    _reject_unknown(table, ("probability", "failure_probability", "tilt", "seed"),
                    "verification", "field")
    explicit = set(table)
    probability = _float_field(table, "probability", "verification", default=0.05,
                               minimum=0.0, allow_zero=True, maximum=1.0)
    if probability == 0.0 and explicit & {"failure_probability", "tilt"}:
        offender = min(explicit & {"failure_probability", "tilt"})
        raise ConfigError(
            f"verification.{offender} is set but verification.probability is "
            "0 (independent checks disabled) — check-segment parameters of a "
            "disabled segment are meaningless; remove failure_probability and "
            "tilt, or set probability > 0 to run checks")
    failure_probability = _float_field(table, "failure_probability", "verification",
                                       default=0.05, minimum=0.0, maximum=1.0)
    tilt = _float_field(table, "tilt", "verification",
                        default=math.log(2.0), minimum=0.0)
    seed = _int_field(table, "seed", "verification", default=run_seed)
    return VerificationConfig(probability=probability,
                              failure_probability=failure_probability,
                              tilt=tilt, seed=seed)


def _parse_checkpoint(table: dict | None) -> CheckpointConfig:
    if table is None:
        return CheckpointConfig(interval_steps=10, keep_generations=2)
    _reject_unknown(table, ("interval_steps", "keep_generations"), "checkpoint",
                    "field")
    interval = _int_field(table, "interval_steps", "checkpoint", default=10,
                          minimum=1)
    keep = _int_field(table, "keep_generations", "checkpoint", default=2,
                      minimum=1)
    if keep != 2:
        raise ConfigError(
            f"checkpoint.keep_generations = {keep} is not supported: the 0.4 "
            "runtime keeps exactly 2 checkpoint generations; remove the "
            "field (configurable retention is planned after 0.4.0)")
    return CheckpointConfig(interval_steps=interval, keep_generations=keep)


def _parse_output(table: dict | None) -> OutputConfig:
    if table is None:
        return OutputConfig(trajectory_interval_steps=1, summary_interval_steps=10)
    _reject_unknown(table, ("trajectory_interval_steps", "summary_interval_steps"),
                    "output", "field")
    trajectory = _int_field(table, "trajectory_interval_steps", "output",
                            default=1, minimum=1)
    summary = _int_field(table, "summary_interval_steps", "output", default=10,
                         minimum=1)
    return OutputConfig(trajectory_interval_steps=trajectory,
                        summary_interval_steps=summary)


def _check_task_compatibility(task: TaskConfig, *, reference: BackendConfig | None,
                              surrogate: BackendConfig | None,
                              policy: PolicyConfig | None,
                              verification_present: bool = False) -> None:
    if task.mode == "adaptive":
        if task.kind != "md":
            raise ConfigError(
                f"task.mode 'adaptive' requires task.kind = 'md', got "
                f"{task.kind!r}; adaptive single-point/relax drivers arrive "
                "with WP07 — use reference or surrogate mode for those")
        missing = [name for name, section in (("reference", reference),
                                              ("surrogate", surrogate))
                   if section is None]
        if missing:
            raise ConfigError(
                f"task.mode 'adaptive' requires [{missing[0]}] with a backend; "
                "adaptive MD drives with the surrogate and checks it against "
                "the reference")
    else:
        sections = {"reference": reference, "surrogate": surrogate}
        needed = task.mode  # "reference" or "surrogate"
        unused = "surrogate" if needed == "reference" else "reference"
        if sections[unused] is not None:
            raise ConfigError(
                f"task.mode {task.mode!r} does not use [{unused}]; remove the "
                f"section or set task.mode = 'adaptive' to use both")
        if sections[needed] is None:
            raise ConfigError(
                f"task.mode {task.mode!r} requires [{needed}] with a backend")
        if policy is not None:
            raise ConfigError(
                f"task.mode {task.mode!r} does not use [policy]; the energetic "
                "policy only applies to adaptive MD — remove the section")
        if verification_present:
            raise ConfigError(
                f"task.mode {task.mode!r} does not use [verification]; "
                "independent checks only exist in adaptive MD — remove the "
                "section")