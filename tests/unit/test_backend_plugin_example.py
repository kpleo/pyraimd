"""The example plugin (examples/backends/pyraimd2_harmonic) works as a real
third-party backend: discovered through the entry-point mechanism, created
without touching the core, usable as reference engine and as surrogate in an
actual energetic evaluation, and rejected when capabilities don't match.

The test loads the plugin package from its example source tree and drives
discovery through real ``importlib.metadata.EntryPoint.load()`` — the same
mechanism an installed distribution uses (a real ``uv pip install`` of the
example is verified in the WP05 report, outside the hermetic suite).
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

from pyraimd2.backends import (
    BackendRegistryError,
    available_backends,
    create_backend,
    registry,
)
from pyraimd2.engines.base import CapabilityMismatchError, EngineCapabilities

PLUGIN_SRC = (
    Path(__file__).parents[2] / "examples" / "backends" / "pyraimd2_harmonic" / "src"
)


@pytest.fixture()
def plugin_entry_points(monkeypatch):
    monkeypatch.syspath_prepend(str(PLUGIN_SRC))
    eps = [
        importlib.metadata.EntryPoint(
            name="harmonic_reference",
            value="pyraimd2_harmonic:reference_factory",
            group="pyraimd2.backends",
        ),
        importlib.metadata.EntryPoint(
            name="harmonic_surrogate",
            value="pyraimd2_harmonic:surrogate_factory",
            group="pyraimd2.backends",
        ),
    ]
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: eps if group == "pyraimd2.backends" else [],
    )
    registry._reset_entry_point_cache()
    yield eps
    registry._reset_entry_point_cache()


def test_example_plugin_is_discovered_without_core_changes(plugin_entry_points) -> None:
    backends = available_backends()
    assert backends["harmonic_reference"]["origin"].startswith("entry-point:")
    assert backends["harmonic_surrogate"]["origin"].startswith("entry-point:")
    # Builtin names are untouched.
    assert backends["qe"]["origin"] == "builtin"


def test_example_plugin_serves_as_reference_and_surrogate(plugin_entry_points, tmp_path) -> None:
    from ase import Atoms

    from pyraimd2.loop import EnergeticRunner
    from pyraimd2.store import Store

    engine = create_backend("harmonic_reference", kind="engine", k=1.0)
    surrogate = create_backend("harmonic_surrogate", kind="surrogate", k=1.0, bias=0.05)
    assert engine.fingerprint.startswith("harmonic-reference:")
    assert surrogate.capabilities.uncertainty_available is False

    atoms = Atoms("H4", positions=[
        [0.00, 0.00, 0.00], [0.92, 0.08, 0.01],
        [0.10, 0.88, 0.21], [0.18, 0.12, 0.94],
    ])
    atoms.set_momenta([
        [0.010, -0.020, 0.015], [-0.012, 0.008, 0.011],
        [0.006, 0.014, -0.009], [-0.004, -0.002, -0.017],
    ])
    store = Store(tmp_path / "run.db")
    runner = EnergeticRunner(atoms, surrogate, engine, store, "plugin-run",
                             force_budget=0.5, timestep_fs=0.1,
                             probe_steps=(0.02, 0.04), check_probability=0.0)
    summary = runner.run(3)
    assert summary.n_evaluations >= 4  # initial evaluation + 3 md steps
    assert engine.capabilities.energy_kind == surrogate.capabilities.energy_kind


def test_example_plugin_capabilities_are_enforced(plugin_entry_points) -> None:
    with pytest.raises(CapabilityMismatchError, match="stress"):
        create_backend("harmonic_reference",
                       require=EngineCapabilities(stress_available=True))
    with pytest.raises(BackendRegistryError, match="registered as 'surrogate'"):
        create_backend("harmonic_surrogate", kind="engine")
