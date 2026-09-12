"""Use an ASE calculator as a reference engine."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import (
    BaseCalculator,
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from pyraimd2.engines.ase_resources import (
    FILE_IDENTITY_FORMAT,
    DeclaredFileResource,
    validate_resource_declaration,
)
from pyraimd2.engines.base import (
    EnergyKind,
    EngineCapabilities,
    EngineError,
    EngineResult,
)

_FILE_HASH_CACHE: dict[tuple[str, int, int], str | None] = {}


def _file_sha256(path: Path) -> str | None:
    """Content sha256 of a model file, None when unreadable (cached by
    path/mtime/size so per-evaluation fingerprints stay cheap)."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _FILE_HASH_CACHE:
        digest: str | None = None
        try:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        except OSError:
            digest = None
        _FILE_HASH_CACHE[key] = digest
    return _FILE_HASH_CACHE[key]


def _jsonable(value: object) -> object:
    """JSON-normalize a calculator parameter value, or raise TypeError."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    raise TypeError(f"unserializable parameter value of type {type(value).__name__}")


def _embedded_files(value: object) -> list[str]:
    """Parameter values naming existing files (model artifacts to hash)."""
    found: list[str] = []
    if isinstance(value, str):
        try:
            path = Path(value)
            if ("\0" not in value) and path.is_file():
                found.append(str(path))
        except (OSError, ValueError):
            pass
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_embedded_files(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_embedded_files(item))
    return found


def _wrapper_structure(calculator: Calculator) -> tuple[list, list[float]] | None:
    """ASE mixing/wrapper structure: (children, weights), or None.

    ASE's mixing calculators hold their children in ``.mixer.calcs`` (+
    ``.weights``); other wrappers may expose a direct ``.calcs`` list. The
    children of nested wrappers belong to ASE's *other* class hierarchy
    (``BaseCalculator``, not ``Calculator``), so the structure is detected
    independently of the children's types — mistaking a nested wrapper for
    a plain empty-parameter calculator would mint a colliding identity.
    """
    for host in (getattr(calculator, "mixer", None), calculator):
        if host is None:
            continue
        calcs = getattr(host, "calcs", None)
        if isinstance(calcs, (list, tuple)) and len(calcs) > 0:
            weights = getattr(host, "weights", None)
            if weights is None or len(weights) != len(calcs):
                weights = [1.0] * len(calcs)
            return list(calcs), [float(w) for w in weights]
    return None


def calculator_identity(calculator: Calculator,
                        _seen: frozenset[int] = frozenset()) -> dict | None:
    """A serializable identity for an ASE calculator, or None when its
    effective physical state cannot be identified reliably.

    Covers the class path and the declared effective parameters; parameter
    values naming existing files (model artifacts) contribute a content
    hash, so two states of the same path are distinguished. Wrapper
    calculators (e.g. ``SumCalculator``) carry no parameters of their own —
    the identity recurses into the child calculators and their weights.
    Whenever a wrapper structure is visible, its contents decide: children
    that are not calculators, or one unidentifiable child, make the whole
    wrapper unknown — never a fallback to the wrapper's own empty
    parameters. ``calculator.name`` alone is never an identity — LJ
    epsilon 1 → 2 must change this value.
    """
    if id(calculator) in _seen:
        return None  # cyclic wrapper: not identifiable
    wrapper = _wrapper_structure(calculator)
    if wrapper is not None:
        children, weights = wrapper
        if not all(isinstance(child, (Calculator, BaseCalculator))
                   for child in children):
            return None  # wrapper holding non-calculators: not identifiable
        child_ids = []
        for child in children:
            child_id = calculator_identity(child, _seen | {id(calculator)})
            if child_id is None:
                return None
            child_ids.append(child_id)
        return {
            "class": f"{type(calculator).__module__}.{type(calculator).__qualname__}",
            "children": child_ids,
            "weights": weights,
        }
    parameters = getattr(calculator, "parameters", None)
    if parameters is None:
        return None
    try:
        normalized = _jsonable(dict(parameters))
    except (TypeError, ValueError):
        return None
    files = sorted(set(_embedded_files(parameters)))
    return {
        "class": f"{type(calculator).__module__}.{type(calculator).__qualname__}",
        "parameters": normalized,
        "files": {path: _file_sha256(Path(path)) for path in files},
    }


def clear_calculator_results(calculator: Calculator) -> None:
    """Drop any cached results after a failed evaluation, without masking
    the original error: ASE's two class hierarchies do not share ``reset``
    (``Calculator`` has it, ``BaseCalculator``/FileIO calculators do not)."""
    try:
        reset = getattr(calculator, "reset", None)
        if callable(reset):
            reset()
        else:
            results = getattr(calculator, "results", None)
            if isinstance(results, dict):
                results.clear()
    except Exception:  # noqa: BLE001, S110 - cleanup never masks the real error
        pass


class AseEngine:
    """Adapt an externally configured ASE calculator to the Engine protocol.

    Set ``force_consistent=True`` for calculators whose forces differentiate
    a free energy instead of their default reported energy. Stress is opt-in.
    Each adapter owns its calculator; use separate instances for separate runs.

    The adapter always reports the raw physical energy/forces/stress: any
    constraint adjustment (energy terms included) is left to the workflow,
    which applies constraints exactly once. Mixing a constraint-adjusted
    energy with unprojected forces would pair values from different surfaces.

    Capabilities and result metadata follow the flags exactly: with
    ``force_consistent=True`` the reported energy is the free energy whose
    gradient is the forces (``energy_kind="free_energy"``,
    ``force_consistent=True``); otherwise the default ``energy`` is reported
    and whether the forces differentiate it is calculator-dependent, so
    consistency stays unknown rather than claimed.

    The fingerprint hashes the calculator class, its declared effective
    parameters and the content of any parameter-referenced model files —
    never just ``calculator.name``. A calculator whose state cannot be
    identified (no serializable parameters) yields fingerprint ``None``
    (unknown identity: consumers must not cache or compare on it) unless an
    explicit ``identity`` string is supplied.

    ``file_parameters={parameter: role}`` declares the calculator's
    top-level immutable file parameters for the versioned
    ``ase-file-identity-v1`` branch: each declared slot is identified by
    its content digest (streamed byte reads, never the stat cache) and its
    path leaves the identity payload, while every undeclared parameter
    keeps the pre-existing semantics exactly.  The declaration is
    validated at construction and refuses unknown or incomplete
    calculator identities, missing parameters, empty or duplicate roles,
    non-regular or unreadable files, and wrapper/mixing calculator
    subtrees.  An explicit ``identity`` never substitutes for the content
    check.
    """

    def __init__(self, calculator: Calculator, *, force_consistent: bool = False,
                 include_stress: bool = False,
                 identity: str | None = None,
                 file_parameters: dict[str, str] | None = None) -> None:
        self.calculator = calculator
        self.force_consistent = force_consistent
        self.include_stress = include_stress
        self.identity = identity
        self._file_resources: tuple[DeclaredFileResource, ...] = ()
        if file_parameters is not None:
            if _wrapper_structure(calculator) is not None:
                raise ValueError(
                    "file_parameters: wrapper/mixing calculators cannot bind "
                    "file resources in this version; declare resources on a "
                    "plain calculator")
            parameters = getattr(calculator, "parameters", None)
            if not isinstance(parameters, dict) or \
                    calculator_identity(calculator) is None:
                raise ValueError(
                    "file_parameters: the calculator's effective state cannot "
                    "be identified (no serializable parameters); the opt-in "
                    "file identity is refused")
            self._file_resources = validate_resource_declaration(
                parameters, file_parameters)

    @property
    def file_resources(self) -> tuple[DeclaredFileResource, ...]:
        """The immutable declared-resource view (empty when undeclared)."""
        return self._file_resources

    @property
    def name(self) -> str:
        return f"ase-{self.calculator.name}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            energy_kind=(EnergyKind.FREE_ENERGY if self.force_consistent
                         else EnergyKind.ENERGY),
            force_consistent=True if self.force_consistent else None,
            forces_conservative=None,  # property of the wrapped calculator
            stress_available=self.include_stress,
        )

    @property
    def fingerprint(self) -> str | None:
        flags = f"force_consistent={self.force_consistent}:stress={self.include_stress}"
        if self._file_resources:
            return self._file_identity_fingerprint(flags)
        identity = calculator_identity(self.calculator)
        if identity is None:
            if self.identity is None:
                return None  # unknown identity, honestly undeclared
            return f"ase:{self.calculator.name}:explicit:{self.identity}:{flags}"
        payload = dict(identity)
        if self.identity is not None:
            payload["explicit"] = self.identity
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"ase:{self.calculator.name}:{digest}:{flags}"

    def _file_identity_fingerprint(self, flags: str) -> str:
        """The versioned content identity of a declared adapter
        (``ase-file-identity-v1``): the calculator class and name, every
        undeclared physical parameter with the old normalization and
        embedded-file semantics, and per declared slot the parameter name,
        role and full content SHA-256 — never the path."""
        identity = calculator_identity(self.calculator)
        if identity is None:
            # The declaration was validated at construction; this is a
            # defense-in-depth guard, never a silent downgrade.
            raise ValueError(
                "the calculator's identity is unknown; the opt-in file "
                "identity cannot be computed")
        declared = {resource.parameter for resource in self._file_resources}
        parameters = {key: value for key, value in identity["parameters"].items()
                      if key not in declared}
        files = sorted(set(_embedded_files(parameters)))
        payload = {
            "format": FILE_IDENTITY_FORMAT,
            "class": identity["class"],
            "name": self.calculator.name,
            "parameters": parameters,
            # undeclared embedded files keep the legacy path-keyed hashing;
            # declared slots never enter the path-keyed files map again
            "files": {path: _file_sha256(Path(path)) for path in files},
            "resources": [
                {"parameter": resource.parameter, "role": resource.role,
                 "sha256": resource.sha256}
                for resource in sorted(self._file_resources,
                                       key=lambda r: r.parameter)],
        }
        if self.identity is not None:
            payload["explicit"] = self.identity
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"ase:{self.calculator.name}:{digest}:{flags}"

    def compute(self, atoms: Atoms) -> EngineResult:
        work = atoms.copy()
        work.calc = self.calculator
        start = time.perf_counter()
        try:
            # One explicit execution, then read the whole results dict:
            # separate property getters let ASE silently re-run a FileIO
            # calculator when a property (e.g. stress) is missing.
            properties = ["energy", "forces"]
            if self.force_consistent:
                properties.append("free_energy")
            if self.include_stress:
                properties.append("stress")
            self.calculator.calculate(work, properties, all_changes)
            results = self.calculator.results
            energy_key = "free_energy" if self.force_consistent else "energy"
            if energy_key not in results:
                raise PropertyNotImplementedError(
                    f"calculator did not return {energy_key!r}"
                )
            energy = float(results[energy_key])
            forces = np.array(results["forces"], dtype=float, copy=True)
            if self.include_stress:
                if "stress" not in results:
                    raise PropertyNotImplementedError(
                        "calculator did not return 'stress'"
                    )
                stress = np.array(results["stress"], dtype=float, copy=True)
            else:
                stress = None
            if (not np.isfinite(energy) or forces.shape != (len(work), 3)
                    or not np.isfinite(forces).all()):
                raise ValueError("Nonfinite energy or invalid forces")
            if stress is not None and (stress.shape != (6,) or not np.isfinite(stress).all()):
                raise ValueError("Invalid stress")
        except Exception as error:
            clear_calculator_results(self.calculator)
            raise EngineError(f"ASE reference evaluation failed: {error}") from error
        return EngineResult(energy, forces, stress, time.perf_counter() - start,
                            energy_kind=self.capabilities.energy_kind,
                            force_consistent=True if self.force_consistent else None)
