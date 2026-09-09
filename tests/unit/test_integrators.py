"""M1 regression: integrator spec identity and committed/pending step state
round-trips (fake RNG states; the stochastic adapter is exercised in the
NVT suite)."""

from __future__ import annotations

import numpy as np
import pytest

from pyraimd2.loop.integrators import (
    CommittedStepState,
    IntegratorSpec,
    LangevinAdapter,
    PendingStepState,
    VelocityVerletAdapter,
    state_digest,
)


def _spec_nve():
    return IntegratorSpec(algorithm="velocity_verlet", ensemble="nve",
                          timestep_fs=0.5)


def test_spec_identity_binds_algorithm_and_parameters():
    nve = _spec_nve()
    nvt = IntegratorSpec(algorithm="langevin", ensemble="nvt",
                         timestep_fs=0.5, temperature_K=300.0,
                         friction_per_fs=0.01, thermostat_seed=7)
    assert nve.identity() != nvt.identity()
    slower = IntegratorSpec(algorithm="langevin", ensemble="nvt",
                            timestep_fs=0.5, temperature_K=300.0,
                            friction_per_fs=0.02, thermostat_seed=7)
    assert nvt.identity() != slower.identity()
    same = IntegratorSpec(algorithm="langevin", ensemble="nvt",
                          timestep_fs=0.5, temperature_K=300.0,
                          friction_per_fs=0.01, thermostat_seed=7)
    assert nvt.identity() == same.identity()
    assert nvt.as_dict() == same.as_dict()


def test_spec_rejects_bad_combinations():
    with pytest.raises(ValueError, match="thermostat fields"):
        IntegratorSpec(algorithm="velocity_verlet", ensemble="nve",
                       timestep_fs=0.5, friction_per_fs=0.01)
    with pytest.raises(ValueError, match="friction_per_fs"):
        IntegratorSpec(algorithm="langevin", ensemble="nvt",
                       timestep_fs=0.5, temperature_K=300.0)


def test_committed_step_state_roundtrip_and_digest():
    rng_state = np.random.default_rng(123).bit_generator.state
    state = CommittedStepState(
        step=4, physical_time_fs=2.5,
        positions=np.array([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]]),
        momenta=np.array([[0.01, 0.0, 0.0], [-0.02, 0.03, 0.0]]),
        driving_source="reference", model_id="model#g1", spec=_spec_nve(),
        nsteps=4, thermostat_rng=dict(rng_state))
    restored = CommittedStepState.from_dict(state.as_dict())
    np.testing.assert_allclose(restored.positions, state.positions,
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(restored.momenta, state.momenta,
                               rtol=0, atol=1e-15)
    assert restored.thermostat_rng == state.thermostat_rng
    assert restored.spec.as_dict() == state.spec.as_dict()
    assert restored.digest() == state.digest()
    moved = CommittedStepState.from_dict(
        {**state.as_dict(),
         "positions": (state.positions + 1e-12).tolist()})
    assert moved.digest() != state.digest()


def test_pending_step_state_keeps_rng_before_and_after():
    gen = np.random.default_rng(99)
    before = dict(gen.bit_generator.state)
    gen.standard_normal((2, 3))
    after = dict(gen.bit_generator.state)
    pending = PendingStepState(
        evaluation_id=3, positions=np.zeros((1, 3)),
        rng_before=before, rng_after=after,
        model_generation=2, label_id="run-label-3")
    restored = PendingStepState.from_dict(pending.as_dict())
    assert restored.rng_before == before
    assert restored.rng_after == after
    assert restored.rng_before != restored.rng_after
    assert restored.evaluation_id == 3
    assert restored.label_id == "run-label-3"
    # A resume continuing from rng_after reproduces the draw, never re-draws.
    resumed_gen = np.random.default_rng()
    resumed_gen.bit_generator.state = restored.rng_after
    np.testing.assert_allclose(resumed_gen.standard_normal((2, 3)),
                               gen.standard_normal((2, 3)), rtol=0, atol=0.0)


def test_state_digest_covers_content():
    a = np.zeros((2, 2))
    assert state_digest(a) != state_digest(np.zeros(4))
    assert state_digest(np.array([1.0])) != state_digest(
        np.array([4607182418800017408], dtype=np.uint64))
    assert state_digest(a) == state_digest(a.copy())


def test_verlet_adapter_completes_mid_step_momenta():
    from ase import Atoms, units

    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_momenta([[0.1, 0.0, 0.0]])
    adapter = VelocityVerletAdapter(atoms, _spec_nve())
    forces = np.array([[-1.2, 0.0, 0.0]])
    expected = atoms.get_momenta() + 0.5 * 0.5 * units.fs * forces
    np.testing.assert_allclose(
        adapter.complete_momenta(atoms, forces), expected, rtol=0, atol=1e-15)
    assert adapter.thermostat_state() is None
    with pytest.raises(ValueError, match="thermostat state"):
        adapter.load_thermostat_state({"rng": {}})


def test_langevin_adapter_uses_explicit_generator_and_restores_it():
    from ase import Atoms

    spec = IntegratorSpec(algorithm="langevin", ensemble="nvt",
                          timestep_fs=0.5, temperature_K=300.0,
                          friction_per_fs=0.01, thermostat_seed=11)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
    atoms.set_momenta([[0.1, 0, 0], [-0.1, 0, 0]])
    from ase.calculators.emt import EMT

    atoms.calc = EMT()
    adapter = LangevinAdapter(atoms, spec)
    assert adapter.dyn.rng is adapter.rng
    import numpy as _np

    assert adapter.rng is not _np.random  # never the global stream
    first = adapter.thermostat_state()
    adapter.dyn.step()
    second = adapter.thermostat_state()
    assert first != second  # the bath stream advances with the step
    restored = LangevinAdapter(
        Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]]), spec,
        thermostat_rng_state=first)
    assert restored.thermostat_state() == first
    with pytest.raises(ValueError, match="thermostat_seed"):
        LangevinAdapter(atoms,
                        IntegratorSpec(algorithm="langevin", ensemble="nvt",
                                       timestep_fs=0.5, temperature_K=300.0,
                                       friction_per_fs=0.01))
