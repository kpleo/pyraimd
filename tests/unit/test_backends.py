"""Backend registry: builtin factories, lazy loading, entry-point discovery
and conflicts, protocol/kind verification, and capability rejection at the
factory layer (WP01 contract reuse). No torch, no pyscf, no QE needed."""

from __future__ import annotations

import importlib.metadata
import sys
import types
from pathlib import Path

import pytest

from pyraimd2.backends import (
    BackendRegistryError,
    assert_capabilities_satisfy,
    available_backends,
    backend_capabilities,
    create_backend,
    registry,
)
from pyraimd2.engines.base import (
    CapabilityMismatchError,
    EnergyKind,
    EngineCapabilities,
)
from pyraimd2.surrogate.base import SurrogateCapabilities


@pytest.fixture(autouse=True)
def _fresh_entry_point_cache():
    registry._reset_entry_point_cache()
    yield
    registry._reset_entry_point_cache()


def test_builtins_are_registered_without_importing_heavy_dependencies() -> None:
    backends = available_backends()
    assert backends["qe"] == {"kind": "engine", "origin": "builtin"}
    assert backends["qe-ase"] == {"kind": "engine", "origin": "builtin"}
    assert backends["pyscf"] == {"kind": "engine", "origin": "builtin"}
    # Discovery and selection must stay light: no torch, no pyscf.
    assert "torch" not in sys.modules
    assert "pyscf" not in sys.modules


def test_create_builtin_loads_at_selection_time(tmp_path: Path) -> None:
    from pyraimd2.engines.qe_engine import QeEngine

    engine = create_backend("qe", kind="engine", run_root=tmp_path / "qe",
                            pseudo_dir="/pseudo")
    assert isinstance(engine, QeEngine)
    assert "torch" not in sys.modules and "pyscf" not in sys.modules


def test_unknown_backend_name_lists_what_exists() -> None:
    with pytest.raises(BackendRegistryError, match="unknown backend 'nope'.*qe"):
        create_backend("nope")


def test_kind_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BackendRegistryError, match="registered as 'engine'"):
        create_backend("qe", kind="surrogate", run_root=tmp_path, pseudo_dir="/x")


def test_capability_requirement_is_enforced_at_the_factory(tmp_path: Path) -> None:
    # A non-metallic QE config reports the total energy; requiring the
    # smeared free energy must fail before any computation happens.
    with pytest.raises(CapabilityMismatchError, match="energy_kind"):
        create_backend("qe", run_root=tmp_path, pseudo_dir="/x",
                       require=EngineCapabilities(energy_kind=EnergyKind.FREE_ENERGY))
    engine = create_backend(
        "qe", run_root=tmp_path, pseudo_dir="/x",
        require=EngineCapabilities(energy_kind=EnergyKind.ENERGY,
                                   force_consistent=True,
                                   stress_available=True))
    assert engine.capabilities.stress_available is True


def test_unknown_capabilities_never_satisfy_a_requirement() -> None:
    undeclared = EngineCapabilities()  # all unknown
    with pytest.raises(CapabilityMismatchError, match="force_consistent"):
        assert_capabilities_satisfy(undeclared, EngineCapabilities(force_consistent=True))
    with pytest.raises(CapabilityMismatchError, match="stress"):
        assert_capabilities_satisfy(undeclared, EngineCapabilities(stress_available=True))
    # Unknown requirements are unconstrained: anything passes.
    assert_capabilities_satisfy(undeclared, EngineCapabilities())


def test_surrogate_requirement_checks_uncertainty() -> None:
    caps = SurrogateCapabilities(uncertainty_available=False)
    with pytest.raises(CapabilityMismatchError, match="uncertainty"):
        assert_capabilities_satisfy(caps, SurrogateCapabilities(uncertainty_available=True))


def _fake_entry_point(name: str, module_name: str) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(
        name=name, value=f"{module_name}:factory", group="pyraimd2.backends"
    )


def _install_fake_plugin(monkeypatch, name: str = "toy_backend") -> str:
    """A tiny plugin module living only in sys.modules, exposed through a
    real EntryPoint so discovery/load go through importlib.metadata."""
    module = types.ModuleType(f"{name}_module")

    class ToyEngine:
        name = "toy"

        @property
        def capabilities(self):
            return EngineCapabilities(energy_kind=EnergyKind.ENERGY,
                                      force_consistent=True)

        def compute(self, atoms):  # pragma: no cover - not exercised here
            raise NotImplementedError

    def factory(**kwargs):
        return ToyEngine()

    factory.backend_kind = "engine"
    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ep = _fake_entry_point(name, module.__name__)
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: [ep] if group == "pyraimd2.backends" else [],
    )
    return name


def test_entry_point_plugin_is_discovered_and_created(monkeypatch) -> None:
    name = _install_fake_plugin(monkeypatch)
    backends = available_backends()
    assert backends[name]["origin"].startswith("entry-point:")
    backend = create_backend(name, kind="engine")
    assert backend_capabilities(backend).energy_kind == EnergyKind.ENERGY
    # Plugins enter through the same factory contract as builtins.
    with pytest.raises(CapabilityMismatchError, match="stress"):
        create_backend(name, require=EngineCapabilities(stress_available=True))


def test_entry_point_conflicting_with_a_builtin_is_an_error(monkeypatch) -> None:
    _install_fake_plugin(monkeypatch, name="qe")
    with pytest.raises(BackendRegistryError, match="conflicts with the builtin"):
        available_backends()


def test_two_entry_points_with_the_same_name_are_an_error(monkeypatch) -> None:
    module = types.ModuleType("dup_module")
    module.factory = lambda: object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    eps = [_fake_entry_point("dup", "dup_module"), _fake_entry_point("dup", "dup_module")]
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: eps if group == "pyraimd2.backends" else [],
    )
    with pytest.raises(BackendRegistryError, match="two entry points"):
        available_backends()


def test_factory_returning_the_wrong_protocol_is_rejected(monkeypatch) -> None:
    module = types.ModuleType("wrong_module")

    class NotAnEngine:
        pass

    def factory():
        return NotAnEngine()

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ep = _fake_entry_point("wrong", "wrong_module")
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: [ep] if group == "pyraimd2.backends" else [],
    )
    with pytest.raises(BackendRegistryError, match="engine protocol"):
        create_backend("wrong", kind="engine")
