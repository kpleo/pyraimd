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
