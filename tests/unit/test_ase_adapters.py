"""Adapter contracts for externally supplied ASE calculators."""
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms

from pyraimd2.engines import AseEngine, EngineError
from pyraimd2.surrogate import AseSurrogate


def test_adapter_preserves_raw_forces_and_input():
    atoms = Atoms('Ar2', positions=[[0, 0, 0], [1.1, 0, 0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    original = atoms.positions.copy()
    expected = atoms.copy()
    expected.calc = LennardJones()
    result = AseEngine(LennardJones()).compute(atoms)
    assert atoms.calc is None
    np.testing.assert_array_equal(atoms.positions, original)
    np.testing.assert_allclose(result.forces, expected.get_forces(apply_constraint=False))
    assert result.energy == pytest.approx(expected.get_potential_energy())
    assert result.stress is None
    prediction = AseSurrogate(LennardJones()).predict(atoms)
    np.testing.assert_allclose(prediction.forces, result.forces)
    assert np.isnan(prediction.uncertainty).all()


def test_constraint_with_energy_term_returns_raw_energy_and_forces():
    """A constraint carrying an energy term must not leak into the backend
    label: energy and forces must come from the same raw physical surface.
    The workflow applies constraints itself, exactly once."""
    from ase.constraints import Hookean

    atoms = Atoms('Ar2', positions=[[0, 0, 0], [1.5, 0, 0]])
    atoms.set_constraint(Hookean(a1=0, a2=1, k=5.0, rt=1.2))
    raw = atoms.copy()
    raw.calc = LennardJones()
    raw_energy = raw.get_potential_energy(apply_constraint=False)
    adjusted_energy = raw.get_potential_energy(apply_constraint=True)
    # The test only means something if this constraint actually has an
    # energy term (FixAtoms alone cannot expose the mismatch).
    assert adjusted_energy != pytest.approx(raw_energy)
    result = AseEngine(LennardJones()).compute(atoms)
    assert result.energy == pytest.approx(raw_energy)
    np.testing.assert_allclose(result.forces, raw.get_forces(apply_constraint=False))
    prediction = AseSurrogate(LennardJones()).predict(atoms)
    assert prediction.energy == pytest.approx(raw_energy)
    np.testing.assert_allclose(prediction.forces, result.forces)


class EnergyChoice(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': 2., 'free_energy': 1.,
                        'forces': np.zeros((len(self.atoms), 3))}


def test_force_consistent_energy_is_explicit():
    atoms = Atoms('He', positions=[[0, 0, 0]])
    assert AseEngine(EnergyChoice()).compute(atoms).energy == 2.
    assert AseEngine(EnergyChoice(), force_consistent=True).compute(atoms).energy == 1.


class InvalidResult(EnergyChoice):
    def calculate(self, *args, **kwargs):
        super().calculate(*args, **kwargs)
        self.results['forces'][0, 0] = np.nan


def test_invalid_calculator_result_raises_and_clears_cache():
    calculator = InvalidResult()
    with pytest.raises(EngineError):
        AseEngine(calculator).compute(Atoms('He'))
    assert calculator.results == {}


# --- fingerprint: physical parameters and model artifacts, never just name ---


def test_fingerprint_tracks_physical_parameters():
    """LJ epsilon 1 -> 2 doubles every energy: the identity must change
    (review R2 — calculator.name used to be the whole fingerprint)."""
    e1 = AseEngine(LennardJones(epsilon=1.0))
    e2 = AseEngine(LennardJones(epsilon=2.0))
    assert e1.fingerprint != e2.fingerprint
    assert AseEngine(LennardJones(epsilon=1.0)).fingerprint == e1.fingerprint
    assert e1.fingerprint.startswith("ase:lennardjones")
    s1 = AseSurrogate(LennardJones(epsilon=1.0))
    s2 = AseSurrogate(LennardJones(epsilon=2.0))
    assert s1.fingerprint != s2.fingerprint
    assert s1.fingerprint == AseSurrogate(LennardJones(epsilon=1.0)).fingerprint


def test_fingerprint_tracks_model_file_content(tmp_path):
    """Same path, different content: the artifact hash changes the identity."""
    from ase.calculators.calculator import Calculator

    class FileBacked(Calculator):
        implemented_properties: ClassVar[list[str]] = ['energy', 'forces']

        def __init__(self, model_path):
            super().__init__()
            self.parameters = {'model': str(model_path)}

        def calculate(self, atoms=None, properties=('energy',), system_changes=None):
            super().calculate(atoms, properties, system_changes or [])
            self.results = {'energy': 0.0, 'forces': np.zeros((len(self.atoms), 3))}

    model = tmp_path / 'model.dat'
    model.write_text('weights-v1')
    first = AseEngine(FileBacked(model)).fingerprint
    model.write_text('weights-v2')
    second = AseEngine(FileBacked(model)).fingerprint
    assert first != second
    assert AseEngine(FileBacked(model)).fingerprint == second  # stable again


def test_fingerprint_none_when_state_cannot_be_identified():
    """A calculator without identifiable parameters yields an unknown
    identity (None), never a name masquerading as one; an explicit identity
    string is honored instead."""
    from ase.calculators.calculator import Calculator

    class Opaque(Calculator):
        implemented_properties: ClassVar[list[str]] = ['energy', 'forces']

        def __init__(self):
            super().__init__()
            self.parameters = {'blob': object()}  # not serializable

        def calculate(self, atoms=None, properties=('energy',), system_changes=None):
            super().calculate(atoms, properties, system_changes or [])

    assert AseEngine(Opaque()).fingerprint is None
    assert AseSurrogate(Opaque()).fingerprint is None
    explicit = AseEngine(Opaque(), identity='lab-benchmark-2026-09')
    assert explicit.fingerprint is not None
    assert 'lab-benchmark-2026-09' in explicit.fingerprint
    assert AseSurrogate(Opaque(), identity='x').fingerprint is not None


class _FailingNoReset(Calculator):
    """Espresso-style calculator: no reset() (BaseCalculator hierarchy) and a
    calculate that fails. Used to prove cleanup never masks the real error."""

    reset = None  # ASE FileIO calculators lack reset entirely
    implemented_properties: ClassVar[list[str]] = ['energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',), system_changes=None):
        super().calculate(atoms, properties, system_changes or [])
        self.results = {'stale': 1}
        raise RuntimeError('the real failure')


def test_failure_without_reset_method_preserves_original_error():
    engine = AseEngine(_FailingNoReset())
    with pytest.raises(EngineError, match='the real failure'):
        engine.compute(Atoms('He'))
    assert engine.calculator.results == {}  # stale results cleared


# --- wrapper calculators: identity must cover children (review B1) ----------


def test_sum_calculator_fingerprint_tracks_children():
    """SumCalculator.parameters is empty — the physics lives in the child.
    epsilon 1 -> 2 inside the wrapper must change the identity (it did not
    when only parameters were hashed)."""
    from ase.calculators.mixing import SumCalculator

    e1 = AseEngine(SumCalculator([LennardJones(epsilon=1.0)]))
    e2 = AseEngine(SumCalculator([LennardJones(epsilon=2.0)]))
    assert e1.fingerprint is not None and e2.fingerprint is not None
    assert e1.fingerprint != e2.fingerprint
    same = AseEngine(SumCalculator([LennardJones(epsilon=1.0)]))
    assert same.fingerprint == e1.fingerprint


def test_mixed_calculator_weights_enter_identity():
    from ase.calculators.mixing import MixedCalculator

    half = AseEngine(MixedCalculator(LennardJones(epsilon=1.0),
                                     LennardJones(epsilon=2.0), 0.5, 0.5))
    quarter = AseEngine(MixedCalculator(LennardJones(epsilon=1.0),
                                        LennardJones(epsilon=2.0), 0.25, 0.75))
    assert half.fingerprint is not None
    assert half.fingerprint != quarter.fingerprint


def test_wrapper_with_unidentifiable_child_is_unknown():
    """One unidentifiable child makes the whole wrapper unknown (None) —
    never a name-level fingerprint pretending to be trustworthy."""
    from ase.calculators.mixing import SumCalculator

    wrapper = SumCalculator([_FailingNoReset.__new__(_FailingNoReset)])
    child = wrapper.mixer.calcs[0]
    Calculator.__init__(child)
    child.parameters = {'blob': object()}
    assert AseEngine(wrapper).fingerprint is None
    explicit = AseEngine(wrapper, identity='documented-mixture-1')
    assert explicit.fingerprint is not None
