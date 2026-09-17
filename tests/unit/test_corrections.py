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
frozen-parameter immutability, species/order identity, stress
non-impersonation, input validation, the periodic chart contract
(stateless minimum image around q0, periodic axes only, cell/pbc change
refusal, resume-identical across fresh instances), registry wiring and
the .npz parameter file path.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    quadratic_corrected_factory,
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
SPECIES_TRIMER = ("H", "H", "H")
CENTER_TRIMER_REFERENCE = np.array([[0.00, 0.00, 0.00],
                                    [0.95, 0.05, 0.02],
                                    [0.06, 0.90, 0.01]])
CENTER_TRIMER_BASE = np.array([[0.03, -0.02, 0.01],
                               [0.90, 0.08, -0.01],
                               [0.04, 0.93, 0.03]])

CENTER_REFERENCE = np.array([[0.00, 0.00, 0.00], [0.95, 0.05, 0.02]])
CENTER_BASE = np.array([[0.03, -0.02, 0.01], [0.90, 0.08, -0.01]])
Q0 = np.array([[0.05, 0.03, -0.02], [0.98, 0.02, 0.03]])
SPECIES_DIMER = ("H", "O")
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
        base, Q0, delta_f0, delta_h, species=SPECIES_DIMER,
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
    repaired = QuadraticCorrectedSurrogate(base, Q0, delta_f0_broken,
                                           delta_h_broken,
                                           species=SPECIES_DIMER)
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
        base, Q0, corrected.delta_f0, delta_h_broken, species=SPECIES_DIMER)
    # Reconstructing from the already-projected matrix is a fixed point:
    # nothing left to project, and the matrix is unchanged.
    twice = QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0,
                                        repaired.delta_h,
                                        species=SPECIES_DIMER)
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
        base, Q0, corrected.delta_f0, corrected.delta_h + skew,
        species=SPECIES_DIMER)
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
        species=SPECIES_DIMER, energy_offset=corrected.energy_offset)
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
        species=SPECIES_TRIMER,
        energy_offset=reference.predict(atoms0).energy
        - base.predict(atoms0).energy)
    # Both matrices are translation-invariant, so the acoustic projection
    # is a no-op and the corrected trimer stays exact.
    assert corrected.translation_projection_norm < 1e-12
    permutation = np.array([3, 4, 5, 0, 1, 2, 6, 7, 8])  # exchange atoms 0/1
    swapped = corrected.delta_h[np.ix_(permutation, permutation)]
    wrong = QuadraticCorrectedSurrogate(
        base, Q0_TRIMER, delta_f0, swapped, species=SPECIES_TRIMER,
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
                                    corrected.delta_h, species=SPECIES_DIMER,
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, delta_f0_bumped, corrected.delta_h,
                                    species=SPECIES_DIMER,
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0, delta_h_bumped,
                                    species=SPECIES_DIMER,
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0,
                                    corrected.delta_h, species=SPECIES_DIMER,
                                    energy_offset=corrected.energy_offset + 0.1,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0,
                                    corrected.delta_h, species=("O", "H"),
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="unit-test calibration"),
        QuadraticCorrectedSurrogate(base, Q0, corrected.delta_f0,
                                    corrected.delta_h, species=SPECIES_DIMER,
                                    energy_offset=corrected.energy_offset,
                                    calibration_note="another calibration"),
    ]
    for variant in variants:
        assert variant.fingerprint != reference_fp
    other_base = QuadraticModel(CENTER_BASE, _block_hessian(K_BASE), E0_BASE,
                                fingerprint="base-v2")
    assert QuadraticCorrectedSurrogate(
        other_base, Q0, corrected.delta_f0, corrected.delta_h,
        species=SPECIES_DIMER, energy_offset=corrected.energy_offset,
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
        QuadraticCorrectedSurrogate(base, Q0, np.zeros((2, 3)),
                                    np.zeros((6, 6)), species=SPECIES_DIMER),
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
        QuadraticCorrectedSurrogate(base, np.zeros((2, 2)), delta_f0, delta_h,
                                    species=SPECIES_DIMER)
    with pytest.raises(ValueError, match="q0"):
        QuadraticCorrectedSurrogate(base, np.zeros((0, 3)), delta_f0, delta_h,
                                    species=SPECIES_DIMER)
    with pytest.raises(ValueError, match="delta_f0"):
        QuadraticCorrectedSurrogate(base, Q0, np.zeros((3, 3)), delta_h,
                                    species=SPECIES_DIMER)
    with pytest.raises(ValueError, match="delta_h"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0, np.zeros((5, 5)),
                                    species=SPECIES_DIMER)
    with pytest.raises(ValueError, match="finite"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0,
                                    np.full((6, 6), np.nan),
                                    species=SPECIES_DIMER)
    with pytest.raises(ValueError, match="real numbers"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0,
                                    np.zeros((6, 6), dtype=complex),
                                    species=SPECIES_DIMER)
    with pytest.raises(ValueError, match="energy_offset"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0, delta_h,
                                    species=SPECIES_DIMER,
                                    energy_offset=float("inf"))
    with pytest.raises(ValueError, match="species"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0, delta_h,
                                    species=("H",))
    with pytest.raises(TypeError, match="species"):
        QuadraticCorrectedSurrogate(base, Q0, delta_f0, delta_h,
                                    species=None)
    with pytest.raises(TypeError, match="predict"):
        QuadraticCorrectedSurrogate(object(), Q0, delta_f0, delta_h,
                                    species=SPECIES_DIMER)
    with pytest.raises(TypeError, match="predict"):
        ScaledSurrogate(object(), 1.0)
    for bad_scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale"):
            ScaledSurrogate(base, bad_scale)


def test_species_and_order_are_validated_at_prediction():
    _, _, corrected = _build_correction()
    with pytest.raises(ValueError, match="fixed order"):
        corrected.predict(Atoms("OH", positions=Q0))  # swapped order
    with pytest.raises(ValueError, match="fixed order"):
        corrected.predict(Atoms("H3", positions=np.zeros((3, 3))))
    with pytest.raises(ValueError, match="fixed order"):
        corrected.predict(Atoms("HOHO", positions=np.zeros((4, 3))))


def test_frozen_parameters_cannot_be_mutated():
    _, _, corrected = _build_correction()
    _, base, _ = _build_correction()
    scaled = ScaledSurrogate(base, 1.05, calibration_note="fit A")
    for wrapper, field, value in (
            (scaled, "scale", 1.07),
            (scaled, "calibration_note", "fit B"),
            (corrected, "energy_offset", 0.5),
            (corrected, "calibration_note", "other"),
            (corrected, "parameters_provenance", {"path": "x"}),
            (corrected, "q0", Q0),
            (corrected, "delta_f0", corrected.delta_f0)):
        with pytest.raises(AttributeError):
            setattr(wrapper, field, value)
    # the frozen arrays themselves are not writable either
    assert not corrected.q0.flags.writeable
    with pytest.raises(ValueError):
        corrected.q0[0, 0] = 1.0
    # and the value itself is the one fixed at construction
    reference = QuadraticModel(CENTER_REFERENCE, _block_hessian(K_REFERENCE),
                               E0_REFERENCE)
    base_model = QuadraticModel(CENTER_BASE, _block_hessian(K_BASE), E0_BASE)
    atoms0 = _dimer(Q0)
    assert corrected.energy_offset == pytest.approx(
        reference.predict(atoms0).energy - base_model.predict(atoms0).energy,
        abs=1e-12)


# ---------------------------------------------------------------------------
# the periodic chart contract (stateless, resume-identical)


def test_periodic_chart_is_a_pure_function_of_positions():
    _, base, corrected = _build_correction()
    cell = [10.0, 10.0, 10.0]
    # the same positions give the same answer regardless of call history
    # — on one instance and across fresh instances (the resume case)
    near = _dimer(Q0 + DISPLACEMENTS[0], cell=cell, pbc=True)
    first = corrected.predict(near)
    assert np.isfinite(first.energy)
    far = _dimer(Q0 + DISPLACEMENTS[0] + np.array([[5.3, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0]]),
                 cell=cell, pbc=True)
    far_after_near = corrected.predict(far)
    _, _, fresh = _build_correction()
    far_fresh = fresh.predict(far)
    assert far_fresh.energy == pytest.approx(far_after_near.energy, abs=1e-12)
    np.testing.assert_allclose(far_fresh.forces, far_after_near.forces,
                               atol=1e-12)
    # the chart's documented boundary: 5.3 A on a 10 A cell wraps onto the
    # nearest branch around q0 (the -4.7 A image) — the correction TERM
    # equals the correction term of the wrapped image (the base model
    # always sees the raw positions, so compare total-minus-base)
    wrapped = _dimer(Q0 + DISPLACEMENTS[0] + np.array([[-4.7, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0]]))
    _, plain_base, plain = _build_correction()
    expected = plain.predict(wrapped)
    correction_far = far_after_near.forces - base.predict(far).forces
    correction_expected = expected.forces - plain_base.predict(wrapped).forces
    np.testing.assert_allclose(correction_far, correction_expected, atol=1e-12)
    correction_energy_far = (far_after_near.energy - base.predict(far).energy
                             - corrected.energy_offset)
    correction_energy_expected = (expected.energy - plain_base.predict(wrapped).energy
                                  - plain.energy_offset)
    assert correction_energy_far == pytest.approx(correction_energy_expected,
                                                  abs=1e-12)


def test_mixed_pbc_wraps_only_the_periodic_axes():
    _, base, corrected = _build_correction()
    cell = np.diag([10.0, 10.0, 10.0])
    pbc = [True, True, False]
    # +5.3 A on x (wraps to -4.7), +6.0 A on z (stays: z is not periodic)
    positions = Q0 + DISPLACEMENTS[0] + np.array([[5.3, 0.0, 6.0],
                                                  [0.0, 0.0, 0.0]])
    got = corrected.predict(_dimer(positions, cell=cell, pbc=pbc))
    expected_positions = Q0 + DISPLACEMENTS[0] + np.array([[-4.7, 0.0, 6.0],
                                                           [0.0, 0.0, 0.0]])
    _, plain_base, plain = _build_correction()
    want = plain.predict(_dimer(expected_positions))
    # compare the correction TERM (total minus base at the same raw
    # positions): the wrapped x image and the untouched z displacement
    correction_got = got.forces - base.predict(
        _dimer(positions, cell=cell, pbc=pbc)).forces
    correction_want = want.forces - plain_base.predict(
        _dimer(expected_positions)).forces
    np.testing.assert_allclose(correction_got, correction_want, atol=1e-12)


def test_rigid_cell_shift_is_mapped_back_into_the_chart():
    _, _, corrected = _build_correction()
    cell = [10.0, 10.0, 10.0]
    # The whole configuration one full cell to the left (a trajectory
    # crossing a periodic boundary): the minimum image maps every atom
    # back into q0's chart, and the rigidly translation-invariant base
    # model sees an equivalent geometry.
    shifted_positions = Q0 + DISPLACEMENTS[0] - np.array([[10.0, 0.0, 0.0],
                                                          [10.0, 0.0, 0.0]])
    atoms = _dimer(shifted_positions, cell=cell, pbc=True)
    got = corrected.predict(atoms)
    _, _, plain = _build_correction()  # a fresh instance never mixing charts
    want = plain.predict(_dimer(Q0 + DISPLACEMENTS[0]))
    assert got.energy == pytest.approx(want.energy, abs=1e-12)
    np.testing.assert_allclose(got.forces, want.forces, atol=1e-12)


def test_a_changed_cell_or_pbc_is_rejected():
    _, _, corrected = _build_correction()
    first = _dimer(Q0 + DISPLACEMENTS[0], cell=[10.0, 10.0, 10.0], pbc=True)
    corrected.predict(first)
    changed_cell = _dimer(Q0 + DISPLACEMENTS[0], cell=[10.5, 10.0, 10.0],
                          pbc=True)
    with pytest.raises(ValueError, match="fixed cell"):
        corrected.predict(changed_cell)
    changed_pbc = _dimer(Q0 + DISPLACEMENTS[0], cell=[10.0, 10.0, 10.0],
                         pbc=[True, True, False])
    with pytest.raises(ValueError, match="pbc"):
        corrected.predict(changed_pbc)
    # and dropping to non-periodic after a periodic chart is a chart change
    with pytest.raises(ValueError, match="non-periodic"):
        corrected.predict(_dimer(Q0 + DISPLACEMENTS[0]))


def test_a_nonperiodic_run_never_records_a_chart():
    _, _, corrected = _build_correction()
    corrected.predict(_dimer(Q0 + DISPLACEMENTS[0]))
    corrected.predict(_dimer(Q0 + DISPLACEMENTS[1]))
    assert corrected._chart_cell is None and corrected._chart_pbc is None


# ---------------------------------------------------------------------------
# registry wiring and the .npz parameter file


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
        scale=1.05,
        calibration_note="registry test",
    )
    assert isinstance(scaled, ScaledSurrogate)
    assert scaled.scale == 1.05
    assert "harmonic-surrogate" in scaled.fingerprint
    corrected = create_backend(
        "quadratic-corrected",
        base={"name": "harmonic-surrogate",
              "kwargs": {"k": 1.2, "r0": 0.9, "bias": 0.03}},
        q0=Q0.tolist(),
        delta_f0=np.zeros((2, 3)).tolist(),
        delta_h=np.zeros((6, 6)).tolist(),
        species=list(SPECIES_DIMER),
    )
    assert isinstance(corrected, QuadraticCorrectedSurrogate)
    assert corrected.species == SPECIES_DIMER
    # predictions actually work through the registry-created wrapper
    prediction = corrected.predict(_dimer(Q0 + DISPLACEMENTS[0]))
    assert prediction.forces.shape == (2, 3)


def test_registry_rejects_unknown_fields_and_bad_specs():
    with pytest.raises(TypeError, match="scale"):
        create_backend("scaled", base="harmonic-surrogate")  # missing scale
    with pytest.raises(TypeError, match="base"):
        create_backend("scaled", base=object(), scale=1.0)
    with pytest.raises(TypeError, match="unknown base spec fields"):
        create_backend(
            "scaled",
            base={"name": "harmonic-surrogate", "kwargs": {}, "extra": 1},
            scale=1.0,
        )
    with pytest.raises(TypeError, match="kwargs"):
        create_backend(
            "scaled",
            base={"name": "harmonic-surrogate", "kwargs": ["not", "a", "dict"]},
            scale=1.0,
        )
    with pytest.raises(BackendRegistryError, match="unknown backend"):
        create_backend(
            "scaled",
            base={"name": "no-such-backend", "kwargs": {}},
            scale=1.0,
        )
    with pytest.raises(ValueError, match="species"):
        create_backend(
            "quadratic-corrected",
            base={"name": "harmonic-surrogate", "kwargs": {}},
            q0=Q0.tolist(), delta_f0=np.zeros((2, 3)).tolist(),
            delta_h=np.zeros((6, 6)).tolist(),
        )  # no species


def test_npz_parameters_match_the_inline_values(tmp_path):
    _, base, corrected = _build_correction()
    npz = tmp_path / "correction.npz"
    np.savez(npz, q0=corrected.q0, delta_f0=corrected.delta_f0,
             delta_h=corrected.delta_h)
    from_file = QuadraticCorrectedSurrogate(
        base, *np.load(npz).values(), species=SPECIES_DIMER,
        energy_offset=corrected.energy_offset,
        calibration_note=corrected.calibration_note)
    atoms = _dimer(Q0 + DISPLACEMENTS[0])
    got = from_file.predict(atoms)
    want = corrected.predict(atoms)
    assert got.energy == pytest.approx(want.energy, abs=1e-14)
    np.testing.assert_allclose(got.forces, want.forces, atol=1e-14)
    # value-based identity: the file content produces the same fingerprint
    assert from_file.fingerprint == corrected.fingerprint
    assert from_file.parameters_provenance is None  # direct load, no factory


def test_npz_and_inline_parameters_are_mutually_exclusive(tmp_path):
    npz = tmp_path / "correction.npz"
    np.savez(npz, q0=Q0, delta_f0=np.zeros((2, 3)), delta_h=np.zeros((6, 6)))
    with pytest.raises(ValueError, match="mutually exclusive"):
        quadratic_corrected_factory(
            base={"name": "harmonic-surrogate", "kwargs": {}},
            q0=Q0.tolist(), delta_f0=np.zeros((2, 3)).tolist(),
            delta_h=np.zeros((6, 6)).tolist(), species=list(SPECIES_DIMER),
            parameters_npz=str(npz))
    with pytest.raises(ValueError, match="needs q0"):
        quadratic_corrected_factory(
            base={"name": "harmonic-surrogate", "kwargs": {}},
            species=list(SPECIES_DIMER))


def test_npz_missing_key_and_unreadable_files_are_rejected(tmp_path):
    npz = tmp_path / "broken.npz"
    np.savez(npz, q0=Q0, delta_f0=np.zeros((2, 3)))  # delta_h missing
    with pytest.raises(ValueError, match="delta_h"):
        quadratic_corrected_factory(
            base={"name": "harmonic-surrogate", "kwargs": {}},
            species=list(SPECIES_DIMER), parameters_npz=str(npz))
    not_npz = tmp_path / "not-an-archive.npz"
    not_npz.write_bytes(b"this is not a zip archive")
    with pytest.raises(ValueError, match="not a readable .npz"):
        quadratic_corrected_factory(
            base={"name": "harmonic-surrogate", "kwargs": {}},
            species=list(SPECIES_DIMER), parameters_npz=str(not_npz))
    with pytest.raises(ValueError, match="cannot be read"):
        quadratic_corrected_factory(
            base={"name": "harmonic-surrogate", "kwargs": {}},
            species=list(SPECIES_DIMER),
            parameters_npz=str(tmp_path / "absent.npz"))
    # wrong array TYPES in a readable archive: complex delta_h is a
    # controlled configuration error, never a silent cast
    bad_type = tmp_path / "complex.npz"
    np.savez(bad_type, q0=Q0, delta_f0=np.zeros((2, 3)),
             delta_h=np.zeros((6, 6), dtype=complex))
    with pytest.raises(ValueError, match="real numbers"):
        quadratic_corrected_factory(
            base={"name": "harmonic-surrogate", "kwargs": {}},
            species=list(SPECIES_DIMER), parameters_npz=str(bad_type))


def test_npz_provenance_records_path_and_sha256_outside_the_fingerprint(
        tmp_path):
    import hashlib

    npz = tmp_path / "correction.npz"
    np.savez(npz, q0=Q0, delta_f0=np.zeros((2, 3)), delta_h=np.zeros((6, 6)))
    wrapper = quadratic_corrected_factory(
        base={"name": "harmonic-surrogate", "kwargs": {}},
        species=list(SPECIES_DIMER), parameters_npz=str(npz))
    provenance = wrapper.parameters_provenance
    assert provenance["path"] == str(npz)
    assert provenance["sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    # the file origin is documentation only, never fingerprinted: the same
    # content built inline through the same factory gives the same
    # fingerprint with no provenance attached
    inline_twin = quadratic_corrected_factory(
        base={"name": "harmonic-surrogate", "kwargs": {}},
        q0=Q0.tolist(), delta_f0=np.zeros((2, 3)).tolist(),
        delta_h=np.zeros((6, 6)).tolist(), species=list(SPECIES_DIMER))
    assert inline_twin.parameters_provenance is None
    assert inline_twin.fingerprint == wrapper.fingerprint


def test_nested_base_spec_paths_resolve_against_the_config_file(tmp_path):
    """A wrapper's nested base backend paths follow the configuration
    file's location, never the process cwd."""
    from pyraimd2.config import load_config

    root = tmp_path / "cfg"
    root.mkdir()
    (root / "structure.extxyz").write_text(
        "1\nH\nH 0.0 0.0 0.0\n")
    (root / "run.toml").write_text(
        "schema_version = 1\n"
        "[run]\nid = \"t\"\ndirectory = \"run\"\nseed = 42\n"
        "[task]\nkind = \"md\"\nmode = \"surrogate\"\n"
        "[structure]\nfile = \"structure.extxyz\"\n"
        "[dynamics]\nensemble = \"nve\"\ntimestep_fs = 0.5\nsteps = 1\n"
        "temperature_K = 300.0\nvelocity_seed = 7\n"
        "[surrogate]\nbackend = \"scaled\"\nscale = 1.05\n"
        "[surrogate.base]\nname = \"harmonic-surrogate\"\n"
        "[surrogate.base.kwargs]\nrel_model_file = \"rel/weights.json\"\n")
    config = load_config(root / "run.toml")
    nested = config.surrogate.options["base"]["kwargs"]["rel_model_file"]
    assert nested == str((root / "rel" / "weights.json").resolve())
    assert Path(nested).is_absolute()


# ---------------------------------------------------------------------------
# workflow integration (plain surrogate MD through the registry and config)


def _wrapper_run_toml(root: Path, *, scale: float = 1.05,
                      quadratic: bool = False, npz: Path | None = None,
                      steps: int = 3) -> Path:
    """A plain surrogate-MD config driving the correction wrapper over the
    harmonic base through the registry."""
    root.mkdir(parents=True, exist_ok=True)
    structure = ("3\nH2O chart smoke; no velocities\n"
                 "O 0.870000 0.910000 0.885000\n"
                 "H 0.945000 0.862000 0.918000\n"
                 "H 0.893000 0.948000 0.955000\n")
    (root / "structure.extxyz").write_text(structure)
    surrogate = ('[surrogate]\nbackend = "scaled"\n'
                 f"scale = {scale}\n"
                 'calibration_note = "integration test"\n'
                 "[surrogate.base]\nname = \"harmonic-surrogate\"\n"
                 "[surrogate.base.kwargs]\nk = 1.0\nr0 = 0.9\nbias = 0.05\n")
    if quadratic:
        if npz is None:
            npz = root / "correction.npz"
            np.savez(npz, q0=np.array([[0.87, 0.91, 0.885],
                                       [0.945, 0.862, 0.918],
                                       [0.893, 0.948, 0.955]]),
                     delta_f0=np.zeros((3, 3)), delta_h=np.zeros((9, 9)))
        surrogate = ('[surrogate]\nbackend = "quadratic-corrected"\n'
                     'species = ["O", "H", "H"]\n'
                     f'parameters_npz = "{npz.name}"\n'
                     "[surrogate.base]\nname = \"harmonic-surrogate\"\n"
                     "[surrogate.base.kwargs]\nk = 1.0\nr0 = 0.9\n"
                     "bias = 0.05\n")
    path = root / "run.toml"
    path.write_text(
        "schema_version = 1\n[run]\nid = \"t\"\ndirectory = \"run\"\n"
        "seed = 42\n[task]\nkind = \"md\"\nmode = \"surrogate\"\n"
        "[structure]\nfile = \"structure.extxyz\"\n"
        "[dynamics]\nensemble = \"nve\"\ntimestep_fs = 0.5\n"
        f"steps = {steps}\ntemperature_K = 300.0\nvelocity_seed = 7\n"
        "[checkpoint]\ninterval_steps = 2\nkeep_generations = 2\n"
        + surrogate)
    return path


def test_wrapped_surrogate_runs_and_resumes_through_the_workflow(tmp_path):
    """A scaled wrapper over the harmonic base drives a plain surrogate MD
    run, and a fresh-process resume reproduces the continuous run exactly
    (the correction identity is part of the run identity)."""
    from pyraimd2.config import load_config
    from pyraimd2.store import Store
    from pyraimd2.workflows import resume_workflow, run_workflow

    continuous = load_config(_wrapper_run_toml(tmp_path / "continuous",
                                               steps=5))
    run_workflow(continuous, verbose=False, handle_sigint=False)
    stopped = load_config(_wrapper_run_toml(tmp_path / "stopped", steps=3))
    run_workflow(stopped, verbose=False, handle_sigint=False)
    result = resume_workflow(stopped.run.directory, 2, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 5

    def rows(run_dir):
        return sorted(Store(run_dir / "trajectory.db")._db.select(run_id="t"),
                      key=lambda row: int(row.key_value_pairs["step"]))

    rows_a, rows_b = rows(continuous.run.directory), rows(stopped.run.directory)
    assert len(rows_a) == len(rows_b) > 0
    for a, b in zip(rows_a, rows_b):
        np.testing.assert_allclose(a.toatoms().positions,
                                   b.toatoms().positions, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(a.data["driving"]["forces"], dtype=float),
            np.asarray(b.data["driving"]["forces"], dtype=float),
            rtol=0, atol=1e-12)


def test_changed_correction_identity_refuses_the_resume(tmp_path):
    """Editing the wrapper's scale inside the run's resolved configuration
    changes the frozen model identity: resume refuses, never silently
    applies a different correction."""
    from pyraimd2.config import load_config, load_resolved_config
    from pyraimd2.workflows import resume_workflow, run_workflow
    from pyraimd2.workflows.setup import WorkflowError

    config = load_config(_wrapper_run_toml(tmp_path / "run", steps=3))
    run_workflow(config, verbose=False, handle_sigint=False)
    resolved_path = config.run.directory / "resolved_config.json"
    document = json.loads(resolved_path.read_text())
    assert document["surrogate"]["options"]["scale"] == 1.05
    document["surrogate"]["options"]["scale"] = 1.07
    resolved_path.write_text(json.dumps(document, indent=2) + "\n")
    assert load_resolved_config(config.run.directory) is not None
    with pytest.raises(WorkflowError, match="surrogate identity"):
        resume_workflow(config.run.directory, 1, verbose=False,
                        handle_sigint=False)


def test_quadratic_wrapper_with_npz_runs_through_the_workflow(tmp_path):
    """The registry + NPZ path wiring end to end: a quadratic-corrected
    wrapper with the parameters in an .npz file (resolved against the
    configuration file's directory) drives a plain surrogate MD run."""
    from pyraimd2.config import load_config
    from pyraimd2.workflows import run_workflow

    config = load_config(_wrapper_run_toml(tmp_path / "quad", quadratic=True))
    assert config.surrogate.options["parameters_npz"].endswith("correction.npz")
    assert Path(config.surrogate.options["parameters_npz"]).is_absolute()
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3
