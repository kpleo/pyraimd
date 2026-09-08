"""Analytic checks of the NumPy energetics core; no external solver needed."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyraimd2.energetics import (
    DegenerateResponseError,
    DirectionalResponse,
    IndependentCheckBound,
    estimate_responses,
    integrate_residual_work,
    residual_work,
)


def _linear_probes():
    directions = np.zeros((2, 2, 3))
    directions[0, 0, 0] = directions[1, 0, 1] = 1
    responses = np.array([[[2, 0, 0], [1, 0, 0]], [[0, -3, 0], [0, 1, 0]]])
    steps = np.array([0.02, 0.04])
    plus_d = directions[:, None] * steps[None, :, None, None]
    plus_r = responses[:, None] * steps[None, :, None, None]
    return directions, steps, plus_d, -plus_d, plus_r, -plus_r


def _response(curvature=2.0):
    return DirectionalResponse(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[curvature, 0.0, 0.0]]),
        eta=0.1,
        transverse_coefficient=4.0,
        remainder_coefficient=3.0,
    )


def test_directional_response_signs_and_participation():
    first, second = estimate_responses(*_linear_probes())
    assert first.curvature == pytest.approx(2)
    assert first.coefficient == pytest.approx(0.5)
    assert first.participation == pytest.approx(1.25)
    assert first.signed_inverse_curvature == pytest.approx(0.4)
    assert second.curvature == pytest.approx(-3)
    assert second.coefficient == pytest.approx(-1 / 3)
    assert second.participation == pytest.approx(10 / 9)
    assert second.signed_inverse_curvature == pytest.approx(-0.3)
    for response in (first, second):
        assert response.coefficient == pytest.approx(
            response.participation * response.signed_inverse_curvature
        )
        assert response.eta == pytest.approx(0)
        assert response.remainder_coefficient == pytest.approx(0)
        assert response.transverse_coefficient == pytest.approx(6)


def test_two_scales_retain_large_probe_and_include_even_and_odd_defects():
    # R(x)=2*x+3*x^2+5*x^3: q(.1)=2.05, q(.2)=2.2.
    u = np.array([[[1.0, 0.0, 0.0]]])
    steps = np.array([0.1, 0.2])
    plus = steps[None, :, None, None] * u[:, None]
    minus = -plus
    plus_r = 2 * plus + 3 * plus**2 + 5 * plus**3
    minus_r = 2 * minus + 3 * minus**2 + 5 * minus**3
    (result,) = estimate_responses(u, steps, plus, minus, plus_r, minus_r)
    assert result.response[0, 0] == pytest.approx(2.2)
    assert result.eta == pytest.approx(0.3)
    assert result.transverse_coefficient == pytest.approx(4.4)
    # The largest defect is at x=-.1: 4*.045/.01 = 18.
    assert result.remainder_coefficient == pytest.approx(18)


def test_zero_curvature_with_nonzero_response_is_valid():
    response = DirectionalResponse(
        np.array([[1.0, 0, 0]]),
        np.array([[0, 2.0, 0]]),
        0,
        4,
        0,
    )
    assert (
        response.curvature
        == response.coefficient
        == response.signed_inverse_curvature
        == 0
    )
    result = response.forecast([[0.1, 0, 0]], 0.1, 1, 0, 1, 1)
    assert result.linear_error == pytest.approx(0.2)
    assert result.predicted_work == 0
    assert result.admitted


def test_zero_retained_response_is_an_explicit_calibration_failure():
    inputs = list(_linear_probes())
    inputs[4][0] = inputs[5][0] = 0
    with pytest.raises(DegenerateResponseError, match="zero"):
        estimate_responses(*inputs)


def test_response_arrays_are_owned_immutable_and_snapshots_detached():
    probes = _linear_probes()
    response = estimate_responses(*probes)[0]
    probes[0][:] = 0
    probes[4][:] = 99
    assert response.direction[0, 0] == 1
    assert response.response[0, 0] == pytest.approx(2)
    for value in (response.direction, response.response):
        with pytest.raises(ValueError):
            value[0, 0] = 4
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        response.eta = 99
    saved = json.loads(json.dumps(response.as_dict(), allow_nan=False))
    saved["direction"][0][0] = 7
    assert response.direction[0, 0] == 1


@pytest.mark.parametrize("curvature", [2.0, -2.0])
def test_forecast_signed_work_and_transverse_dependence(curvature):
    response = _response(curvature)
    parallel = response.forecast([[0.1, 0, 0]], 0.25, 1, 0.005, 1, 1)
    transverse = response.forecast([[0.1, 0.03, 0]], 0.25, 1, 0.005, 1, 1)
    assert parallel.linear_error == pytest.approx(0.2)
    assert parallel.envelope == pytest.approx(0.2 + 0.005 + 0.01 + 0.015)
    assert parallel.predicted_work == pytest.approx(curvature * 0.005)
    assert transverse.linear_error == parallel.linear_error
    assert transverse.predicted_work == parallel.predicted_work
    assert transverse.envelope - parallel.envelope == pytest.approx(
        4 * 0.03 + 1.5 * 0.03**2
    )
    assert transverse.transverse_fraction == pytest.approx(0.03 / math.hypot(0.1, 0.03))


def test_forecast_domain_budget_and_single_point_semantics():
    response = _response()
    origin = response.forecast([[0, 0, 0]], 0, 0.1, 0.005, 1, 0.1)
    assert origin.transverse_fraction == 0
    assert origin.linear_error == origin.predicted_work == 0
    assert origin.envelope == 0.005
    assert origin.admitted
    outside = response.forecast([[0.1, 0.03, 0]], 0.5, 10, 0, 1, 0.1)
    assert not outside.in_domain and not outside.admitted
    late = response.forecast([[0, 0, 0]], 1.1, 10, 0, 1, 1)
    assert not late.in_domain and not late.admitted
    over_budget = response.forecast([[0.1, 0, 0]], 0.5, 0.1, 0, 1, 1)
    assert over_budget.in_domain and not over_budget.admitted
    # The runtime, not this stateless mathematical call, retains prefix refusal.
    assert response.forecast([[0, 0, 0]], 0.75, 0.1, 0, 1, 1).admitted


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1 + 2j])
def test_nonfinite_and_complex_arrays_are_rejected(bad):
    probes = list(_linear_probes())
    probes[4] = probes[4].astype(complex if isinstance(bad, complex) else float)
    probes[4][0, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        estimate_responses(*probes)
    with pytest.raises(ValueError):
        _response().forecast([[bad, 0, 0]], 0, 1, 0, 1, 1)
    with pytest.raises(ValueError):
        integrate_residual_work([[[0, 0, 0]]], [[[bad, 0, 0]]])


@pytest.mark.parametrize("index", range(6))
def test_calibration_rejects_invalid_shapes(index):
    probes = list(_linear_probes())
    probes[index] = probes[index].reshape(-1)
    if index == 1:
        probes[index] = np.array([0.01, 0.02, 0.04])
    with pytest.raises(ValueError):
        estimate_responses(*probes)


def test_probe_geometry_and_order_are_validated():
    for invalid_steps in ([0, 0.04], [0.04, 0.02], [0.02, 0.02]):
        probes = list(_linear_probes())
        probes[1] = invalid_steps
        with pytest.raises(ValueError, match="steps"):
            estimate_responses(*probes)
    probes = list(_linear_probes())
    probes[0] *= 2
    with pytest.raises(ValueError, match="unit"):
        estimate_responses(*probes)
    probes = list(_linear_probes())
    probes[2][0, 0, 0, 1] = 0.001
    with pytest.raises(ValueError, match="displacements"):
        estimate_responses(*probes)


@pytest.mark.parametrize(
    "name,bad",
    [
        ("elapsed_fs", -1),
        ("elapsed_fs", np.nan),
        ("force_budget", 0),
        ("force_budget", np.inf),
        ("numerical_floor", -1),
        ("time_cap_fs", 0),
        ("transverse_cap", -0.1),
        ("transverse_cap", 1.1),
    ],
)
def test_forecast_rejects_invalid_settings(name, bad):
    settings = {
        "elapsed_fs": 0,
        "force_budget": 1,
        "numerical_floor": 0,
        "time_cap_fs": 1,
        "transverse_cap": 1,
    }
    settings[name] = bad
    with pytest.raises(ValueError):
        _response().forecast([[0, 0, 0]], **settings)


def test_signed_energy_work_matches_atomic_quadrature_and_hamiltonian_identity():
    # Two separable quadratic potentials, with opposite curvature mismatches.
    # A nonzero origin makes the correction sign observable.
    origin = np.array([[0.3, 0, 0], [-0.4, 0, 0]])
    displacement = np.array([[0.2, 0, 0], [0.5, 0, 0]])
    kb = np.array([[1.0], [2.0]])
    kr = np.array([[4.0], [1.0]])
    correction = (kb - kr) * origin
    times = np.array([0.0, 0.03, 0.4, 1.0])  # Deliberately uneven sampling.
    positions = origin + times[:, None, None] * displacement
    residuals = (kr - kb) * positions + correction
    atomic = integrate_residual_work(positions, residuals)
    np.testing.assert_allclose(atomic[-1], [0.06, -0.125], atol=1e-14)
    base = 0.5 * np.sum(kb * positions**2, axis=(1, 2))
    reference = 0.5 * np.sum(kr * positions**2, axis=(1, 2))
    kinetic_change = 0.37  # Arbitrary drift; the identity separates it.
    for i, position in enumerate(positions):
        work = residual_work(
            origin, position, base[0], base[i], reference[0], reference[i], correction
        )
        assert work == pytest.approx(float(atomic[i].sum()), abs=1e-14)
        anchor_drift = (
            kinetic_change
            + base[i]
            - base[0]
            - np.sum(correction * (position - origin))
        )
        reference_change = kinetic_change + reference[i] - reference[0]
        assert reference_change == pytest.approx(work + anchor_drift)
    assert atomic[-1].sum() == pytest.approx(-0.065)
    with pytest.raises(ValueError):
        atomic.setflags(write=True)


def test_work_single_frame_and_invalid_inputs():
    np.testing.assert_array_equal(
        integrate_residual_work([[[1, 0, 0]]], [[[3, 0, 0]]]), [[0]]
    )
    with pytest.raises(ValueError):
        integrate_residual_work(np.zeros((0, 1, 3)), np.zeros((0, 1, 3)))
    with pytest.raises(ValueError):
        integrate_residual_work(np.zeros((2, 1, 3)), np.zeros((1, 1, 3)))
    with pytest.raises(ValueError):
        residual_work([[0, 0, 0]], [[0, 0, 0]], 0, np.nan, 0, 1, [[1, 0, 0]])
    with pytest.raises(ValueError):
        residual_work([[0, 0, 0]], [[0, 0, 0], [1, 0, 0]], 0, 1, 0, 1, [[1, 0, 0]])


def test_independent_check_bound_uses_acceptances_and_only_detected_violations():
    record = IndependentCheckBound(0.2)
    assert record.bound is None
    assert record.update(False, False, None) is None
    for i in range(100):
        # Latent violations outside the checked population do not enter D.
        record.update(True, i < 20, i in (0, 1) or i >= 20)
    assert record.accepted_count == 100
    assert record.detected_count == 2
    expected = (math.log(2) * 2 - math.log(0.05)) / (100 * -math.log(0.9))
    assert record.bound == pytest.approx(expected)
    assert (
        json.loads(json.dumps(record.as_dict(), allow_nan=False))["bound"]
        == record.bound
    )
    with pytest.raises(FrozenInstanceError):
        record.probability = 0.5


def test_bound_initial_cap_and_tiny_check_probability():
    assert IndependentCheckBound(0.05).update(True, False, None) == 1
    record = IndependentCheckBound(1e-15, tilt=1e-8)
    assert record.update(True, False, None) == 1


def test_full_checks_report_exact_fraction_and_require_complete_checks():
    record = IndependentCheckBound(1, tilt=1000)
    assert record.update(True, True, True) == 1
    assert record.update(True, True, False) == 0.5
    with pytest.raises(ValueError, match="every accepted"):
        record.update(True, False, None)
    assert record.accepted_count == 2 and record.detected_count == 1


@pytest.mark.parametrize(
    "event",
    [(False, True, False), (True, True, None), (1, False, None), (True, False, "yes")],
)
def test_invalid_check_events_leave_counts_unchanged(event):
    record = IndependentCheckBound(0.1)
    with pytest.raises((ValueError, TypeError)):
        record.update(*event)
    assert record.accepted_count == record.detected_count == 0
    assert record.bound is None


@pytest.mark.parametrize(
    "settings",
    [
        {"probability": 0},
        {"probability": 1.1},
        {"probability": np.nan},
        {"probability": 0.1, "failure_probability": 0},
        {"probability": 0.1, "failure_probability": 1},
        {"probability": 0.1, "tilt": 0},
        {"probability": 0.1, "tilt": np.inf},
    ],
)
def test_invalid_check_settings(settings):
    with pytest.raises(ValueError):
        IndependentCheckBound(**settings)
