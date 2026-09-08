"""Backend factories and capability registration.

One place to ask for a reference engine or surrogate by name:

- **Builtin names** map to lazy factories shipped with pyraimd2. Registering
  a name never imports the backend's module: the import happens when the
  backend is *selected*, so ``import pyraimd2`` (or listing backends) never
  pulls in torch/pyscf.
- **Entry-point plugins**: third-party packages add backends without touching
  the core by declaring the ``pyraimd2.backends`` entry-point group. An entry
  point that collides with a builtin name or with another entry point is a
  hard error (:class:`BackendRegistryError`) — never a silent winner.
- **Capability rejection**: ``create_backend(..., require=...)`` checks the
  created backend against the WP01 capabilities contract
  (:class:`~pyraimd2.engines.base.EngineCapabilities` /
  :class:`~pyraimd2.surrogate.base.SurrogateCapabilities`). A requirement the
  backend does not *declaredly* satisfy — including fields it leaves unknown —
  raises :class:`~pyraimd2.engines.base.CapabilityMismatchError` at the
  factory, before any computation.

Factories are plain callables taking keyword arguments and returning an
engine (``compute``) or surrogate (``predict``). Config-file wiring
(TOML → factory) is layered on top of this API later; the registry itself
executes no configuration expressions.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass

from pyraimd2.engines.base import (
    CapabilityMismatchError,
    EnergyKind,
    EngineCapabilities,
    engine_capabilities,
)
from pyraimd2.surrogate.base import SurrogateCapabilities, surrogate_capabilities

ENTRY_POINT_GROUP = "pyraimd2.backends"

ENGINE = "engine"
SURROGATE = "surrogate"


class BackendRegistryError(RuntimeError):
    """Backend registration, discovery or selection failed."""


@dataclass(frozen=True)
class BackendRegistration:
    """One named backend factory.

    ``loader`` imports nothing until called; ``kind`` is the declared
    protocol ("engine"/"surrogate", None when a plugin does not declare one —
    the created object is verified against the requested protocol anyway).
    """

    name: str
    loader: Callable[[], Callable[..., object]]
    kind: str | None
    origin: str  # "builtin" or "entry-point:<distribution>"


def _lazy(module_name: str, attr: str) -> Callable[[], Callable[..., object]]:
    def loader() -> Callable[..., object]:
        module = importlib.import_module(module_name)
        return getattr(module, attr)

    return loader


# Builtin factories: name -> (kind, module, attribute). Modules import lazily
# via _lazy; none of these modules import torch/pyscf at module level.
_BUILTINS: dict[str, tuple[str, str, str]] = {
    "qe": (ENGINE, "pyraimd2.engines.qe_engine", "create_qe_engine"),
    "qe-ase": (ENGINE, "pyraimd2.engines.ase_qe", "create_ase_qe_engine"),
    "pyscf": (ENGINE, "pyraimd2.engines.pyscf_engine", "PyscfEngine"),
}

_entry_point_cache: dict[str, BackendRegistration] | None = None


def _entry_point_registrations() -> dict[str, BackendRegistration]:
    """Discover plugin factories; conflicts are errors, never silent wins."""
    global _entry_point_cache
    if _entry_point_cache is not None:
        return _entry_point_cache
    registrations: dict[str, BackendRegistration] = {}
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in _BUILTINS:
            raise BackendRegistryError(
                f"entry point {ep!r} conflicts with the builtin backend "
                f"{ep.name!r}; choose a different name"
            )
        if ep.name in registrations:
            raise BackendRegistryError(
                f"two entry points register the backend name {ep.name!r}; "
                "refusing to pick one silently"
            )
        origin = f"entry-point:{getattr(getattr(ep, 'dist', None), 'name', None) or 'unknown'}"
        registrations[ep.name] = BackendRegistration(
            name=ep.name,
            loader=ep.load,
            kind=None,  # plugins declare kind on the factory (backend_kind) or at creation
            origin=origin,
        )
    _entry_point_cache = registrations
    return registrations


def _registrations() -> dict[str, BackendRegistration]:
    registrations = {
        name: BackendRegistration(name=name, loader=_lazy(module, attr), kind=kind,
                                  origin="builtin")
        for name, (kind, module, attr) in _BUILTINS.items()
    }
    registrations.update(_entry_point_registrations())
    return registrations


def available_backends() -> dict[str, dict[str, str | None]]:
    """All registered backend names with declared kind and origin."""
    return {
        name: {"kind": reg.kind, "origin": reg.origin}
        for name, reg in sorted(_registrations().items())
    }


def backend_capabilities(backend: object) -> EngineCapabilities:
    """Declared capabilities of a created backend, unknown-safe (WP01 rules)."""
    if hasattr(backend, "predict"):
        return surrogate_capabilities(backend)
    return engine_capabilities(backend)


def assert_capabilities_satisfy(caps: EngineCapabilities,
                                require: EngineCapabilities, *,
                                name: str = "backend") -> None:
    """Reject a backend that does not declaredly satisfy a requirement.

    In requirement position, an unknown/None field means "unconstrained"; a
    stated field must be matched exactly by a stated backend field — an
    *unknown* backend capability never satisfies an explicit requirement.
    """
    problems: list[str] = []
    if (require.energy_kind != EnergyKind.UNKNOWN
            and caps.energy_kind != require.energy_kind):
        problems.append(
            f"requires energy_kind={require.energy_kind.value!r} but declares "
            f"{caps.energy_kind.value!r}"
        )
    for field_name in ("force_consistent", "forces_conservative"):
        required = getattr(require, field_name)
        if required is not None and getattr(caps, field_name) is not required:
            problems.append(
                f"requires {field_name}={required} but declares "
                f"{getattr(caps, field_name)}"
            )
    if require.stress_available and not caps.stress_available:
        problems.append("requires stress but does not declare stress_available")
    if (isinstance(require, SurrogateCapabilities) and require.uncertainty_available
            and not getattr(caps, "uncertainty_available", False)):
        problems.append("requires an uncertainty estimate but declares none")
    if problems:
        raise CapabilityMismatchError(f"backend {name!r}: " + "; ".join(problems))


def create_backend(name: str, *, kind: str | None = None,
                   require: EngineCapabilities | None = None, **kwargs) -> object:
    """Create the backend registered under ``name``.

    The backend's module is imported here, at selection time. ``kind``
    ("engine"/"surrogate") is verified against the created object's protocol;
    ``require`` applies the capability contract before the backend is handed
    out. Unknown names and registry conflicts raise
    :class:`BackendRegistryError`; capability mismatches raise
    :class:`CapabilityMismatchError`.
    """
    registrations = _registrations()
    if name not in registrations:
        known = ", ".join(sorted(registrations)) or "<none>"
        raise BackendRegistryError(
            f"unknown backend {name!r}; registered backends: {known}"
        )
    registration = registrations[name]
    if kind is not None and kind not in (ENGINE, SURROGATE):
        raise BackendRegistryError(f"kind must be {ENGINE!r} or {SURROGATE!r}, got {kind!r}")
    factory = registration.loader()
    declared_kind = registration.kind or getattr(factory, "backend_kind", None)
    if kind is not None and declared_kind is not None and declared_kind != kind:
        raise BackendRegistryError(
            f"backend {name!r} is registered as {declared_kind!r}, not {kind!r}"
        )
    backend = factory(**kwargs)
    if kind == ENGINE and not callable(getattr(backend, "compute", None)):
        raise BackendRegistryError(
            f"backend {name!r} does not provide the engine protocol (compute)"
        )
    if kind == SURROGATE and not callable(getattr(backend, "predict", None)):
        raise BackendRegistryError(
            f"backend {name!r} does not provide the surrogate protocol (predict)"
        )
    if require is not None:
        assert_capabilities_satisfy(backend_capabilities(backend), require, name=name)
    return backend


def _reset_entry_point_cache() -> None:
    """Test hook: re-run entry-point discovery on next access."""
    global _entry_point_cache
    _entry_point_cache = None
