"""MaceSurrogate single point on H2O (slow: loads the MACE-MP-0 model).

Marked ``slow`` — deselected by default, run with ``pytest --runslow``.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule

from pyraimd2.surrogate import MaceSurrogate

pytestmark = pytest.mark.slow


def test_mace_single_point_finite_nonzero_forces() -> None:
    atoms = molecule("H2O")
    atoms.positions[1, 0] += 0.10
    surrogate = MaceSurrogate(model="small", device="cpu", default_dtype="float64")

    assert surrogate._calc is None  # lazy: no model loaded before first predict
    prediction = surrogate.predict(atoms)
    first_calc = surrogate._calc

    assert np.isfinite(prediction.energy)
    assert prediction.forces.shape == (len(atoms), 3)
    assert np.isfinite(prediction.forces).all()
    assert np.abs(prediction.forces).max() > 0.01
    assert prediction.stress is None  # non-periodic input
    assert prediction.uncertainty.shape == (len(atoms),)
    assert np.isnan(prediction.uncertainty).all()  # frozen single model: honest NaN

    # Second prediction reuses the same lazily-built calculator.
    surrogate.predict(atoms)
    assert surrogate._calc is first_calc
