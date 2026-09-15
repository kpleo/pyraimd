"""D1 static correction wrappers (surrogate/corrections.py).

Analytic coverage only — no torch, no QE, no RNG.  The reference/base
pair is a two-mode harmonic dimer with unequal masses (H/O) and
off-diagonal x-y coupling; the quadratic correction reproduces the
reference exactly, so energy/force agreement, finite-difference
consistency and mass-weighted mode frequencies are checked against
closed-form values.  Further groups: translation invariance with the
acoustic sum rule enforced on delta_h (projection residuals recorded
before/after), sign/mapping-error sensitivity (a sign-flipped delta_h
or swapped atom blocks must be caught), content-hash fingerprints,
stress non-impersonation, input validation, the periodic image chart
and registry wiring.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.backends import (
    BackendRegistryError,
    available_backends,
    create_backend,
)
from pyraimd2.engines.base import (
    CapabilityMismatchError,
    EnergyKind,
    EngineCapabilities,
)
from pyraimd2.runtime.identity import fingerprint_of, model_id_for
from pyraimd2.surrogate.base import (
    SurrogateCapabilities,
    SurrogatePrediction,
    assert_compatible_energy_contract,
    surrogate_capabilities,
)
from pyraimd2.surrogate.corrections import (
    QuadraticCorrectedSurrogate,
    ScaledSurrogate,
)

# Off-diagonally coupled x-y blocks of rank 2: the dimer's relative motion
# spans exactly two nonzero modes; z and the three translations are free.
K_REFERENCE = np.array([[1.10, 0.35, 0.0],
                        [0.35, 0.80, 0.0],
                        [0.00, 0.00, 0.0]])
K_BASE = np.array([[0.90, 0.15, 0.0],
                   [0.15, 1.05, 0.0],
                   [0.00, 0.00, 0.0]])

# Three-atom system for the mapping-sensitivity test: distinct pair
# couplings make the Hessian translation-invariant (so the acoustic
# projection is a no-op and the correction stays exact) but not symmetric
# under exchanging two atoms' blocks — a 2-atom translation-invariant
# Hessian is always exchange-symmetric and could not serve.
K_PAIR_01 = np.array([[0.50, 0.10, 0.00],
                      [0.10, 0.40, 0.05],
                      [0.00, 0.05, 0.30]])
K_PAIR_02 = np.array([[0.70, -0.05, 0.10],
                      [-0.05, 0.60, 0.00],
                      [0.10, 0.00, 0.20]])
K_PAIR_12 = np.array([[0.30, 0.20, -0.10],
                      [0.20, 0.50, 0.05],
                      [-0.10, 0.05, 0.40]])

Q0_TRIMER = np.array([[0.05, 0.03, -0.02],
                      [0.98, 0.02, 0.03],
                      [0.10, 0.95, 0.04]])
CENTER_TRIMER_REFERENCE = np.array([[0.00, 0.00, 0.00],
                                    [0.95, 0.05, 0.02],
                                    [0.06, 0.90, 0.01]])
CENTER_TRIMER_BASE = np.array([[0.03, -0.02, 0.01],
                               [0.90, 0.08, -0.01],
                               [0.04, 0.93, 0.03]])

CENTER_REFERENCE = np.array([[0.00, 0.00, 0.00], [0.95, 0.05, 0.02]])
CENTER_BASE = np.array([[0.03, -0.02, 0.01], [0.90, 0.08, -0.01]])
Q0 = np.array([[0.05, 0.03, -0.02], [0.98, 0.02, 0.03]])
E0_REFERENCE = -3.25
E0_BASE = -2.75

DISPLACEMENTS = (
    np.array([[0.04, -0.03, 0.02], [-0.02, 0.05, -0.01]]),
    np.array([[-0.06, 0.01, 0.03], [0.03, -0.04, 0.02]]),
    np.array([[0.02, 0.02, -0.04], [-0.05, -0.01, 0.03]]),
)


class QuadraticModel:
    """Analytic quadratic surrogate: U = e0 + 1/2 u·H·u, F = -H·u with
    u = q - center.  Optional stress and constant spread let tests check
    that wrappers forward/declare them honestly."""

    def __init__(self, center, hessian, e0=0.0, *, stress=None, spread=None,
                 fingerprint="quadratic-model"):
        self.center = np.asarray(center, dtype=float)
        self.hessian = np.asarray(hessian, dtype=float)
        self.e0 = float(e0)
        self._stress = None if stress is None else np.asarray(stress, dtype=float)
        self._spread = spread
        self._fingerprint = fingerprint

    @property
    def capabilities(self):
        return SurrogateCapabilities(
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=self._stress is not None,
            uncertainty_available=self._spread is not None,
        )

    @property
    def fingerprint(self):
        return self._fingerprint

    def predict(self, atoms):
        u = (atoms.get_positions() - self.center).reshape(-1)
        uncertainty = (np.full(len(atoms), np.nan) if self._spread is None
                       else np.full(len(atoms), self._spread))
        return SurrogatePrediction(
            energy=self.e0 + 0.5 * float(u @ self.hessian @ u),
            forces=-(self.hessian @ u).reshape(-1, 3),
            stress=self._stress,
            uncertainty=uncertainty,
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
        )


def _dimer(positions, **kwargs) -> Atoms:
    return Atoms("HO", positions=np.asarray(positions, dtype=float), **kwargs)


def _block_hessian(coupling):
    """Translation-invariant dimer Hessian from one 3x3 coupling block."""
    return np.block([[coupling, -coupling], [-coupling, coupling]])


def _pair_hessian():
    """Translation-invariant 3-atom Hessian from distinct pair couplings:
    H_ii = sum_j K_ij, H_ij = -K_ij (row sums vanish per component)."""
    hessian = np.zeros((9, 9))
    for i, j, coupling in (
        (0, 1, K_PAIR_01),
        (0, 2, K_PAIR_02),
        (1, 2, K_PAIR_12),
    ):
        ii = slice(3 * i, 3 * i + 3)
        jj = slice(3 * j, 3 * j + 3)
        hessian[ii, ii] += coupling
        hessian[jj, jj] += coupling
        hessian[ii, jj] -= coupling
        hessian[jj, ii] -= coupling
    return hessian


def _build_correction(hessian_reference=None, hessian_base=None):
    """Reference/base quadratic models plus the exactly corrected wrapper.

    delta_f0 is read off the two models at q0 and delta_h is the analytic
    Hessian difference, so the correction reproduces the reference
    exactly; the energy offset anchors U_c(q0) to the reference zero.
    """
    if hessian_reference is None:
        hessian_reference = _block_hessian(K_REFERENCE)
    if hessian_base is None:
        hessian_base = _block_hessian(K_BASE)
    reference = QuadraticModel(CENTER_REFERENCE, hessian_reference, E0_REFERENCE,
                               fingerprint="reference")
    base = QuadraticModel(CENTER_BASE, hessian_base, E0_BASE, fingerprint="base")
    atoms0 = _dimer(Q0)
    delta_f0 = reference.predict(atoms0).forces - base.predict(atoms0).forces
    delta_h = hessian_reference - hessian_base
    energy_offset = reference.predict(atoms0).energy - base.predict(atoms0).energy
    corrected = QuadraticCorrectedSurrogate(
        base, Q0, delta_f0, delta_h,
        energy_offset=energy_offset, calibration_note="unit-test calibration")
    return reference, base, corrected


def test_corrected_model_reproduces_the_analytic_reference():
    reference, base, corrected = _build_correction()
    # Zero-point convention: U_c(q0) = U_b(q0) + energy_offset = U_r(q0).
    assert corrected.predict(_dimer(Q0)).energy == pytest.approx(
        base.predict(_dimer(Q0)).energy + corrected.energy_offset, abs=1e-12)
    assert corrected.predict(_dimer(Q0)).energy == pytest.approx(
        reference.predict(_dimer(Q0)).energy, abs=1e-12)
    for displacement in DISPLACEMENTS:
        atoms = _dimer(Q0 + displacement)
        got = corrected.predict(atoms)
        want = reference.predict(atoms)
        assert got.energy == pytest.approx(want.energy, abs=1e-10)
        np.testing.assert_allclose(got.forces, want.forces, atol=1e-10)
        assert got.energy_kind == EnergyKind.ENERGY
        assert got.force_consistent is True


def test_corrected_energy_and_forces_are_finite_difference_consistent():
    _, _, corrected = _build_correction()
    eps = 1e-6
    q = Q0 + DISPLACEMENTS[0]
    forces = corrected.predict(_dimer(q)).forces
    gradient = np.zeros(6)
    for j in range(6):
        step = np.zeros(6)
        step[j] = eps
        plus = corrected.predict(_dimer(q + step.reshape(2, 3))).energy
        minus = corrected.predict(_dimer(q - step.reshape(2, 3))).energy
        gradient[j] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(gradient, -forces.reshape(-1), atol=1e-8)


def test_two_mode_frequencies_match_with_unequal_masses():
    _, _, corrected = _build_correction()
    masses = _dimer(Q0).get_masses()
    assert masses[0] != masses[1]  # unequal masses: the acceptance case
    # The corrected Hessian from finite differences of the corrected
    # forces; masses enter only here, in the analysis-side dynamical
    # matrix, never inside delta_h.
    eps = 1e-6
    hessian_fd = np.zeros((6, 6))
    for j in range(6):
        step = np.zeros(6)
        step[j] = eps
        f_plus = corrected.predict(_dimer(Q0 + step.reshape(2, 3))).forces
        f_minus = corrected.predict(_dimer(Q0 - step.reshape(2, 3))).forces
        hessian_fd[:, j] = -(f_plus - f_minus).reshape(-1) / (2 * eps)
    inv_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    dynamical = hessian_fd * np.outer(inv_sqrt_mass, inv_sqrt_mass)
    omega2_corrected = np.sort(np.linalg.eigvalsh(dynamical))
    # Closed form for the block dimer: three free COM translations plus
    # omega^2 = (1/m1 + 1/m2) * eig(K) for the relative coordinate.
    mu = 1.0 / masses[0] + 1.0 / masses[1]
    omega2_analytic = np.sort(
        np.concatenate([np.zeros(3), np.linalg.eigvalsh(K_REFERENCE) * mu]))
    np.testing.assert_allclose(omega2_corrected, omega2_analytic,
                               rtol=1e-6, atol=1e-8)
    assert int((omega2_analytic > 1e-6).sum()) == 2  # exactly two modes


def test_global_translation_leaves_energy_and_correction_force_untouched():
    _, base, corrected = _build_correction()
    # The well-built matrix is already translation-invariant, so the
    # acoustic projection is a no-op and every record shows it.
    assert max(corrected.translation_hessian_residuals_before) < 1e-12
    assert max(corrected.translation_hessian_residuals_after) < 1e-12
    assert corrected.translation_projection_norm < 1e-12
    assert max(corrected.translation_force_residuals) < 1e-12
    shift = np.array([0.31, -0.27, 0.19])
    q = Q0 + DISPLACEMENTS[1]
    plain = corrected.predict(_dimer(q))
    translated = corrected.predict(_dimer(q + shift))
    np.testing.assert_allclose(translated.forces, plain.forces, atol=1e-10)
    assert translated.energy == pytest.approx(plain.energy, abs=1e-10)
    # A rigid translation of the reference center itself (u = t) adds
    # nothing: the correction force stays exactly delta_f0 — delta_h never
    # flips its sign or amplifies it.
    shifted_origin = corrected.predict(_dimer(Q0 + shift))
    expected = base.predict(_dimer(Q0 + shift)).forces + corrected.delta_f0
    np.testing.assert_allclose(shifted_origin.forces, expected, atol=1e-10)


def test_translation_projection_is_applied_and_recorded():
    _, base, corrected = _build_correction()
    delta_h_broken = corrected.delta_h + np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    delta_f0_broken = corrected.delta_f0 + np.array([[0.2, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0]])
    repaired = QuadraticCorrectedSurrogate(base, Q0, delta_f0_broken, delta_h_broken)
    # The pre-projection records expose the broken input ...
    assert max(repaired.translation_hessian_residuals_before) > 1e-3
    assert repaired.translation_projection_norm > 1e-3
    # ... and the enforced matrix satisfies the acoustic sum rule to
    # machine precision.
    assert max(repaired.translation_hessian_residuals_after) < 1e-12
    # delta_f0 is recorded only: the constraint applies to the matrix, so
    # the broken net force is flagged but stays in the physics.
    assert max(repaired.translation_force_residuals) > 1e-3
    net_force = repaired.predict(_dimer(Q0)).forces.sum(axis=0)
    assert net_force[0] == pytest.approx(0.2, abs=1e-12)
    # With the matrix constrained, a rigid translation no longer leaks
    # delta_h into the forces.
    shift = np.array([0.31, -0.27, 0.19])
    q = Q0 + DISPLACEMENTS[1]
    defect = np.linalg.norm(repaired.predict(_dimer(q + shift)).forces
                            - repaired.predict(_dimer(q)).forces)
    assert defect < 1e-10


def test_translation_projection_is_idempotent():
    _, base, corrected = _build_correction()
    delta_h_broken = corrected.delta_h + np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    repaired = QuadraticCorrectedSurrogate(
        base, Q0, corrected.delta_f0, delta_h_broken)
    # Reconstructing from the already-projected matrix is a fixed point:
    # nothing left to project, and the matrix is unchanged.
    twice = QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0,
                                        repaired.delta_h)
    np.testing.assert_allclose(twice.delta_h, repaired.delta_h, atol=1e-12)
    assert twice.translation_projection_norm < 1e-12
    assert max(twice.translation_hessian_residuals_before) < 1e-12
    assert max(twice.translation_hessian_residuals_after) < 1e-12


def test_hessian_symmetrization_is_applied_and_recorded():
    reference, base, corrected = _build_correction()
    assert corrected.symmetrization_residual_before < 1e-15
    assert corrected.symmetrization_residual_after == 0.0
    skew = np.arange(36, dtype=float).reshape(6, 6) * 0.01
    skew = skew - skew.T
    repaired = QuadraticCorrectedSurrogate(
        base, Q0, corrected.delta_f0, corrected.delta_h + skew)
    assert repaired.symmetrization_residual_before > 1e-6
    assert repaired.symmetrization_residual_after == 0.0
    # The wrapper evaluates the symmetrized matrix, not the raw input.
    np.testing.assert_allclose(repaired.delta_h, corrected.delta_h, atol=1e-14)
    atoms = _dimer(Q0 + DISPLACEMENTS[0])
    np.testing.assert_allclose(repaired.predict(atoms).forces,
                               reference.predict(atoms).forces, atol=1e-10)


def test_sign_flip_of_delta_h_is_detectable():
    reference, base, corrected = _build_correction()
    flipped = QuadraticCorrectedSurrogate(
        base, Q0, corrected.delta_f0, -corrected.delta_h,
        energy_offset=corrected.energy_offset)
    atoms = _dimer(Q0 + DISPLACEMENTS[0])
    # Normal assertion first: the correct mapping matches to roundoff.
    np.testing.assert_allclose(corrected.predict(atoms).forces,
                               reference.predict(atoms).forces, atol=1e-10)
    # A sign flip shifts the correction by exactly 2 delta_h·u — far
    # above every tolerance used here, so the test catches it.
    expected_defect = 2.0 * np.linalg.norm(
        corrected.delta_h @ DISPLACEMENTS[0].reshape(-1))
    defect = np.linalg.norm(
        flipped.predict(atoms).forces - reference.predict(atoms).forces)
    assert defect == pytest.approx(expected_defect, rel=1e-8)
    assert defect > 1e-2


def test_swapping_atom_blocks_is_detectable():
    hessian_reference = _pair_hessian()
    hessian_base = 0.5 * hessian_reference
    reference = QuadraticModel(CENTER_TRIMER_REFERENCE, hessian_reference,
                               E0_REFERENCE, fingerprint="reference-3")
    base = QuadraticModel(CENTER_TRIMER_BASE, hessian_base, E0_BASE,
                          fingerprint="base-3")
    atoms0 = Atoms("H3", positions=Q0_TRIMER)
    delta_f0 = reference.predict(atoms0).forces - base.predict(atoms0).forces
    corrected = QuadraticCorrectedSurrogate(
        base, Q0_TRIMER, delta_f0, hessian_reference - hessian_base,
        energy_offset=reference.predict(atoms0).energy
        - base.predict(atoms0).energy)
    # Both matrices are translation-invariant, so the acoustic projection
    # is a no-op and the corrected trimer stays exact.
    assert corrected.translation_projection_norm < 1e-12
    permutation = np.array([3, 4, 5, 0, 1, 2, 6, 7, 8])  # exchange atoms 0/1
    swapped = corrected.delta_h[np.ix_(permutation, permutation)]
    wrong = QuadraticCorrectedSurrogate(
        base, Q0_TRIMER, delta_f0, swapped,
        energy_offset=corrected.energy_offset)
    displacement = np.array([[0.04, -0.03, 0.02],
                             [-0.02, 0.05, -0.01],
                             [0.03, 0.01, -0.04]])
    atoms = Atoms("H3", positions=Q0_TRIMER + displacement)
    # Normal assertion first: the correct block mapping matches.
    np.testing.assert_allclose(corrected.predict(atoms).forces,
                               reference.predict(atoms).forces, atol=1e-10)
    # The swapped matrix is still symmetric and translation-invariant, so
    # only the physics catches it.
    expected_defect = np.linalg.norm(
        (corrected.delta_h - swapped) @ displacement.reshape(-1))
    defect = np.linalg.norm(
        wrong.predict(atoms).forces - reference.predict(atoms).forces)
    assert defect == pytest.approx(expected_defect, rel=1e-8)
    assert defect > 1e-2


def test_correction_content_and_base_identity_enter_the_fingerprint():
    _, base, corrected = _build_correction()
    reference_fp = corrected.fingerprint
    assert reference_fp.startswith("quadratic-corrected:")
    assert "base" in reference_fp
    assert fingerprint_of(corrected) == reference_fp
    assert model_id_for(corrected, 3) == f"{reference_fp}#g3"
    _, _, identical = _build_correction()
    assert identical.fingerprint == reference_fp
    q0_bumped = Q0.copy()
    q0_bumped[0, 0] += 1e-9
    delta_f0_bumped = corrected.delta_f0.copy()
    delta_f0_bumped[1, 2] += 1e-9
    delta_h_bumped = corrected.delta_h.copy()
    delta_h_bumped[0, 0] += 1e-9
    variants = [
        QuadraticCorrectedSurrogate(base, q0_bumped, corrected.delta_f0,
                                    corrected.delta_h,
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, delta_f0_bumped, corrected.delta_h,
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0, delta_h_bumped,
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0,
                                    corrected.delta_h,
                                    energy_offset=corrected.energy_offset + 0.1,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0,
                                    corrected.delta_h,
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="another calibration"),
    ]
    for variant in variants:
        assert variant.fingerprint != reference_fp
    other_base = QuadraticModel(CENTER_BASE, _block_hessian(K_BASE), E0_BASE,
                                fingerprint="base-v2")
    assert QuadraticCorrectedSurrogate(
        other_base, Q0, corrected.delta_f0, corrected.delta_h,
        energy_offset=corrected.energy_offset,
        calibration_note="unit-test calibration").fingerprint != reference_fp


def test_scaled_fingerprint_freezes_scale_note_and_base():
    base = QuadraticModel(CENTER_BASE, _block_hessian(K_BASE), E0_BASE,
                          fingerprint="base")
    scaled = ScaledSurrogate(base, 1.05, calibration_note="fit A")
    assert scaled.fingerprint == ScaledSurrogate(
        base, 1.05, calibration_note="fit A").fingerprint
    assert scaled.fingerprint != ScaledSurrogate(
        base, 1.05 + 1e-12, calibration_note="fit A").fingerprint
    assert scaled.fingerprint != ScaledSurrogate(
        base, 1.05, calibration_note="fit B").fingerprint
    other_base = QuadraticModel(CENTER_BASE, _block_hessian(K_BASE), E0_BASE,
                                fingerprint="base-v2")
    assert scaled.fingerprint != ScaledSurrogate(
        other_base, 1.05, calibration_note="fit A").fingerprint
    assert fingerprint_of(scaled) == scaled.fingerprint


def test_base_stress_is_never_impersonated():
    stress = np.array([0.11, -0.07, 0.05, 0.03, -0.02, 0.01])
    base = QuadraticModel(CENTER_BASE, _block_hessian(K_BASE), E0_BASE,
                          stress=stress, fingerprint="base")
    assert base.capabilities.stress_available is True
    wrappers = [
        ScaledSurrogate(base, 1.1),
        QuadraticCorrectedSurrogate(base, Q0, np.zeros((2, 3)), np.zeros((6, 6))),
    ]
    for wrapper in wrappers:
        assert wrapper.capabilities.stress_available is False
        assert surrogate_capabilities(wrapper).stress_available is False
        assert wrapper.predict(_dimer(Q0)).stress is None
    with pytest.raises(CapabilityMismatchError, match="stress"):
        create_backend(
            "scaled",
            base={"name": "harmonic-surrogate", "kwargs": {}},
            scale=1.0,
            require=SurrogateCapabilities(stress_available=True),
        )


def test_scaled_surrogate_scales_energy_forces_and_spread():
    base = QuadraticModel(CENTER_BASE, _block_hessian(K_BASE), E0_BASE,
                          spread=0.02, fingerprint="base")
    scaled = ScaledSurrogate(base, 1.07, calibration_note="c from unit-test fit")
    atoms = _dimer(Q0 + DISPLACEMENTS[0])
    base_prediction = base.predict(atoms)
    got = scaled.predict(atoms)
    assert got.energy == pytest.approx(1.07 * base_prediction.energy, rel=1e-14)
    np.testing.assert_allclose(got.forces, 1.07 * base_prediction.forces,
                               rtol=1e-14, atol=1e-14)
    # An honest base spread scales with the forces: sigma_c = c sigma_b.
    np.testing.assert_allclose(got.uncertainty, 1.07 * base_prediction.uncertainty,
                               rtol=1e-14, atol=1e-14)
    assert got.energy_kind == EnergyKind.ENERGY
    assert got.force_consistent is True
    caps = scaled.capabilities
    assert caps.force_consistent is True
    assert caps.forces_conservative is True
    assert caps.stress_available is False
    assert caps.uncertainty_available is True


def test_wrappers_pass_the_energy_contract_gate():
    _, _, corrected = _build_correction()
    engine_caps = EngineCapabilities(energy_kind=EnergyKind.ENERGY,
                                     force_consistent=True,
                                     forces_conservative=True)
    assert assert_compatible_energy_contract(
        engine_caps, corrected.capabilities) == "same_kind"
    scaled = ScaledSurrogate(
        QuadraticModel(CENTER_BASE, _block_hessian(K_BASE)), 1.0)
    assert assert_compatible_energy_contract(
        engine_caps, scaled.capabilities) == "same_kind"


def test_correction_inputs_are_strictly_validated():
    base = QuadraticModel(CENTER_BASE, _block_hessian(K_BASE))
    delta_f0 = np.zeros((2, 3))
    delta_h = np.zeros((6, 6))
    with pytest.raises(ValueError, match="q0"):
        QuadraticCorrectedSurrogate(base, np.zeros((2, 2)), delta_f0, delta_h)
    with pytest.raises(ValueError, match="q0"):
        QuadraticCorrectedSurrogate(base, np.zeros((0, 3)), delta_f0, delta_h)
    with pytest.raises(ValueError, match="delta_f0"):
        QuadraticCorrectedSurrogate(base, Q0, np.zeros((3, 3)), delta_h)
    with pytest.raises(ValueError, match="delta_h"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0, np.zeros((5, 5)))
    with pytest.raises(ValueError, match="finite"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0, np.full((6, 6), np.nan))
    with pytest.raises(ValueError, match="real numbers"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0,
                                    np.zeros((6, 6), dtype=complex))
    with pytest.raises(ValueError, match="energy_offset"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0, delta_h,
                                    energy_offset=float("inf"))
    with pytest.raises(TypeError, match="predict"):
        QuadraticCorrectedSurrogate(object(), Q0, delta_f0, delta_h)
    with pytest.raises(TypeError, match="predict"):
        ScaledSurrogate(object(), 1.0)
    for bad_scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale"):
            ScaledSurrogate(base, bad_scale)


def test_atom_count_mismatch_is_rejected():
    _, _, corrected = _build_correction()
    with pytest.raises(ValueError, match="fixed order"):
        corrected.predict(Atoms("H3", positions=np.zeros((3, 3))))


def test_periodic_images_are_fixed_once_not_reselected():
    _, _, corrected = _build_correction()
    cell = [10.0, 10.0, 10.0]
    expected = corrected.predict(_dimer(Q0 + DISPLACEMENTS[0]))
    # First periodic call near q0 freezes every atom's image at zero.
    near = _dimer(Q0 + DISPLACEMENTS[0], cell=cell, pbc=True)
    frozen = corrected.predict(near)
    assert frozen.energy == pytest.approx(expected.energy, abs=1e-12)
    np.testing.assert_allclose(frozen.forces, expected.forces, atol=1e-12)
    # A displacement past half the cell stays continuous with the frozen
    # chart instead of jumping back to the re-selected nearest image.
    far_positions = Q0 + DISPLACEMENTS[0] + np.array([[5.3, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0]])
    far = _dimer(far_positions, cell=cell, pbc=True)
    continuous = corrected.predict(far)
    unwrapped = corrected.predict(_dimer(far_positions))  # same chart, no pbc
    assert continuous.energy == pytest.approx(unwrapped.energy, abs=1e-12)
    np.testing.assert_allclose(continuous.forces, unwrapped.forces, atol=1e-12)
    # Per-step re-selection would wrap the 5.3 A displacement to -4.7 A
    # and land on a measurably different energy.
    rewrapped_positions = Q0 + DISPLACEMENTS[0] + np.array([[-4.7, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0]])
    rewrapped = corrected.predict(_dimer(rewrapped_positions))
    assert abs(continuous.energy - rewrapped.energy) > 1e-3


def test_rigid_cell_shift_is_mapped_back_into_the_chart():
    _, _, corrected = _build_correction()
    cell = [10.0, 10.0, 10.0]
    # The whole configuration one full cell to the left (a trajectory
    # crossing a periodic boundary): the frozen minimum image maps every
    # atom back into q0's chart, and the rigidly translation-invariant
    # base model sees an equivalent geometry.
    shifted_positions = Q0 + DISPLACEMENTS[0] - np.array([[10.0, 0.0, 0.0],
                                                          [10.0, 0.0, 0.0]])
    atoms = _dimer(shifted_positions, cell=cell, pbc=True)
    got = corrected.predict(atoms)
    want = corrected.predict(_dimer(Q0 + DISPLACEMENTS[0]))
    assert got.energy == pytest.approx(want.energy, abs=1e-12)
    np.testing.assert_allclose(got.forces, want.forces, atol=1e-12)


def test_a_changed_cell_after_the_chart_froze_is_rejected():
    _, _, corrected = _build_correction()
    first = _dimer(Q0 + DISPLACEMENTS[0], cell=[10.0, 10.0, 10.0], pbc=True)
    corrected.predict(first)
    second = _dimer(Q0 + DISPLACEMENTS[0], cell=[10.5, 10.0, 10.0], pbc=True)
    with pytest.raises(ValueError, match="fixed cell"):
        corrected.predict(second)


def test_correction_factories_are_registered():
    backends = available_backends()
    assert backends["scaled"] == {"kind": "surrogate", "origin": "builtin"}
    assert backends["quadratic-corrected"] == {"kind": "surrogate",
                                               "origin": "builtin"}


def test_registry_creates_wrappers_from_base_specs():
    scaled = create_backend(
        "scaled",
        base={"name": "harmonic-surrogate",
              "kwargs": {"k": 1.2, "r0": 0.9, "bias": 0.03}},
        scale=1.04,
        calibration_note="unit test",
    )
    assert isinstance(scaled, ScaledSurrogate)
    assert "harmonic-surrogate" in scaled.fingerprint
    quadratic = create_backend(
        "quadratic-corrected",
        base={"name": "harmonic-surrogate", "kwargs": {}},
        q0=Q0,
        delta_f0=np.zeros((2, 3)),
        delta_h=np.zeros((6, 6)),
    )
    assert isinstance(quadratic, QuadraticCorrectedSurrogate)
    base = create_backend("harmonic-surrogate", kind="surrogate")
    atoms = _dimer(Q0 + DISPLACEMENTS[0])
    # Zero correction and default offset: the wrapper returns the base values.
    assert quadratic.predict(atoms).energy == pytest.approx(
        base.predict(atoms).energy, abs=1e-14)
    np.testing.assert_allclose(quadratic.predict(atoms).forces,
                               base.predict(atoms).forces, atol=1e-14)


def test_registry_rejects_unknown_fields_and_bad_specs():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        create_backend("scaled", base={"name": "harmonic-surrogate"}, scale=1.0,
                       bogus=1.0)
    with pytest.raises(TypeError, match="unknown base spec fields"):
        create_backend("scaled", scale=1.0,
                       base={"name": "harmonic-surrogate", "kwargs": {},
                             "bogus": 1})
    with pytest.raises(TypeError, match="requires the backend"):
        create_backend("scaled", base={"kwargs": {}}, scale=1.0)
    with pytest.raises(BackendRegistryError, match="unknown backend"):
        create_backend("scaled", base={"name": "nope"}, scale=1.0)
    with pytest.raises(BackendRegistryError, match="registered as 'surrogate'"):
        create_backend("scaled", kind="engine",
                       base={"name": "harmonic-surrogate"}, scale=1.0)
