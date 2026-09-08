"""Adapter contracts for externally supplied ASE calculators."""
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
