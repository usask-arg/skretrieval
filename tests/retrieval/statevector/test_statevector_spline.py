from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from skretrieval.retrieval.statevector import StateVector
from skretrieval.retrieval.statevector.altitude import AltitudeNativeStateVector
from skretrieval.retrieval.statevector.spline import (
    AdditiveSpline,
    MultiplicativeSpline,
    MultiplicativeSplineOne,
)

LOW_NM = 501.0
HIGH_NM = 509.0
NUM_KNOTS = 5
KNOT_WAVELENGTHS = np.linspace(LOW_NM, HIGH_NM, NUM_KNOTS)

# Measurement grid that brackets the spline window without landing exactly on
# its end points, so the in-window mask is unambiguous.
WAVELENGTHS = np.linspace(500.1, 509.7, 25)
INSIDE = (WAVELENGTHS > LOW_NM) & (WAVELENGTHS < HIGH_NM)
OUTSIDE = ~INSIDE


def _radiance_dataset(num_los: int = 2, num_stokes: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    num_wavel = len(WAVELENGTHS)
    return xr.Dataset(
        {
            "radiance": (
                ["wavelength", "los", "stokes"],
                1.0 + rng.random((num_wavel, num_los, num_stokes)),
            ),
            "wf_dummy": (
                ["altitude", "wavelength", "los", "stokes"],
                rng.random((3, num_wavel, num_los, num_stokes)),
            ),
            "tangent_altitude": (["los"], np.linspace(10_000.0, 20_000.0, num_los)),
        },
        coords={
            "wavelength": WAVELENGTHS,
            "stokes": ["I", "Q", "U", "V"][:num_stokes],
        },
    )


def _canonical(wf: xr.DataArray) -> np.ndarray:
    return wf.transpose("x", "wavelength", "los", "stokes").to_numpy()


def _finite_difference_jacobian(element, radiance, x0, step=1e-3):
    """Central differences of the element's own radiance mapping."""
    columns = []
    for k in range(len(x0)):
        x_plus = x0.copy()
        x_plus[k] += step
        x_minus = x0.copy()
        x_minus[k] -= step

        element.update_state(x_plus)
        r_plus = element.modify_input_radiance(radiance.copy(deep=True))["radiance"]
        element.update_state(x_minus)
        r_minus = element.modify_input_radiance(radiance.copy(deep=True))["radiance"]

        columns.append((r_plus - r_minus) / (2 * step))
    element.update_state(x0)
    return _canonical(xr.concat(columns, dim="x"))


def _multiplicative(num_los=2, **kwargs):
    return MultiplicativeSpline(num_los, LOW_NM, HIGH_NM, NUM_KNOTS, 0, **kwargs)


def _multiplicative_one(**kwargs):
    return MultiplicativeSplineOne(LOW_NM, HIGH_NM, NUM_KNOTS, 0, **kwargs)


def _additive(num_los=2, **kwargs):
    kwargs.setdefault("order", 3)
    return AdditiveSpline(num_los, LOW_NM, HIGH_NM, NUM_KNOTS, 0, **kwargs)


@pytest.mark.parametrize(
    ("factory", "state_size", "bounds", "apriori", "inv_cov_scale", "name"),
    [
        (
            _multiplicative,
            2 * NUM_KNOTS,
            (0.1, 3.0),
            1.0,
            1e-10,
            "spline_501.0_509.0",
        ),
        (
            _multiplicative_one,
            NUM_KNOTS,
            (-100.0, 100.0),
            1.0,
            1e-5,
            "spline_501.0_509.0",
        ),
        (
            _additive,
            2 * NUM_KNOTS,
            (-np.inf, np.inf),
            0.0,
            1e-20,
            "add_spline_501.0_509.0",
        ),
    ],
)
def test_spline_defaults(factory, state_size, bounds, apriori, inv_cov_scale, name):
    element = factory()

    np.testing.assert_array_equal(element.state(), np.full(state_size, apriori))
    np.testing.assert_array_equal(element.lower_bound(), np.full(state_size, bounds[0]))
    np.testing.assert_array_equal(element.upper_bound(), np.full(state_size, bounds[1]))
    np.testing.assert_array_equal(element.apriori_state(), np.full(state_size, apriori))
    np.testing.assert_allclose(
        element.inverse_apriori_covariance(), np.eye(state_size) * inv_cov_scale
    )
    assert element.name() == name


@pytest.mark.parametrize("factory", [_multiplicative, _multiplicative_one, _additive])
def test_spline_custom_bounds(factory):
    element = factory(min_value=0.5, max_value=1.5)

    n = len(element.state())
    np.testing.assert_array_equal(element.lower_bound(), np.full(n, 0.5))
    np.testing.assert_array_equal(element.upper_bound(), np.full(n, 1.5))


@pytest.mark.parametrize("factory", [_multiplicative, _multiplicative_one, _additive])
def test_spline_update_state_round_trip(factory):
    element = factory()
    x = np.linspace(0.8, 1.2, len(element.state()))

    element.update_state(x)

    np.testing.assert_array_equal(element.state(), x)
    # Bounds/prior depend only on the state size, which must not change
    assert element.lower_bound().shape == x.shape
    assert element.apriori_state().shape == x.shape


@pytest.mark.parametrize("factory", [_multiplicative, _multiplicative_one, _additive])
def test_spline_update_state_rejects_wrong_size(factory):
    element = factory()

    with pytest.raises(ValueError, match="cannot reshape"):
        element.update_state(np.ones(len(element.state()) + 1))


@pytest.mark.parametrize("factory", [_multiplicative, _multiplicative_one])
def test_multiplicative_spline_prior_precision_factor(factory):
    element = factory()

    factor = element.prior_precision_factor()

    np.testing.assert_allclose(
        factor.T @ factor, element.inverse_apriori_covariance(), atol=1e-25
    )


def test_multiplicative_spline_enabled_toggle():
    element = _multiplicative()
    assert element.enabled

    element.enabled = False
    assert not element.enabled


def test_multiplicative_spline_default_state_is_identity():
    radiance = _radiance_dataset()
    expected = radiance.copy(deep=True)

    result = _multiplicative().modify_input_radiance(radiance)

    np.testing.assert_allclose(result["radiance"], expected["radiance"])
    np.testing.assert_allclose(result["wf_dummy"], expected["wf_dummy"])


def test_multiplicative_spline_maps_knots_per_los():
    element = _multiplicative(num_los=2)
    constant = np.full(NUM_KNOTS, 2.0)
    linear = 1.0 + 0.05 * (KNOT_WAVELENGTHS - LOW_NM)
    # State is ordered los-major: all knots for los 0, then all for los 1
    element.update_state(np.concatenate([constant, linear]))

    radiance = _radiance_dataset(num_los=2)
    base = radiance["radiance"].to_numpy().copy()
    base_wf = radiance["wf_dummy"].to_numpy().copy()

    result = element.modify_input_radiance(radiance)

    expected_factor = np.ones((len(WAVELENGTHS), 2))
    expected_factor[INSIDE, 0] = 2.0
    expected_factor[INSIDE, 1] = 1.0 + 0.05 * (WAVELENGTHS[INSIDE] - LOW_NM)

    np.testing.assert_allclose(
        result["radiance"].transpose("wavelength", "los", "stokes"),
        base * expected_factor[:, :, np.newaxis],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result["wf_dummy"].transpose("altitude", "wavelength", "los", "stokes"),
        base_wf * expected_factor[np.newaxis, :, :, np.newaxis],
        rtol=1e-12,
    )


@pytest.mark.parametrize("num_stokes", [1, 2])
def test_multiplicative_spline_jacobian_matches_finite_difference(num_stokes):
    element = _multiplicative(num_los=2)
    rng = np.random.default_rng(1)
    x0 = 1.0 + 0.1 * rng.standard_normal(len(element.state()))
    element.update_state(x0)
    radiance = _radiance_dataset(num_los=2, num_stokes=num_stokes)

    wf = element.propagate_wf(radiance.copy(deep=True))

    assert wf.sizes["x"] == len(x0)
    expected = _finite_difference_jacobian(element, radiance, x0)
    np.testing.assert_allclose(_canonical(wf), expected, rtol=1e-6, atol=1e-10)

    # Knots of one line of sight must not influence another line of sight
    wf = _canonical(wf)
    assert np.all(wf[:NUM_KNOTS, :, 1, :] == 0)
    assert np.all(wf[NUM_KNOTS:, :, 0, :] == 0)
    # Nor anything outside the spline window
    assert np.all(wf[:, OUTSIDE] == 0)


def test_multiplicative_spline_one_constant_scales_all_wavelength_variables():
    element = _multiplicative_one()
    element.update_state(np.full(NUM_KNOTS, 2.0))
    radiance = _radiance_dataset()
    base = radiance.copy(deep=True)

    result = element.modify_input_radiance(radiance)

    factor = np.where(INSIDE, 2.0, 1.0)
    np.testing.assert_allclose(
        result["radiance"],
        base["radiance"] * xr.DataArray(factor, dims=["wavelength"]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result["wf_dummy"],
        base["wf_dummy"] * xr.DataArray(factor, dims=["wavelength"]),
        rtol=1e-12,
    )
    # Variables without a wavelength dimension are left alone
    np.testing.assert_array_equal(result["tangent_altitude"], base["tangent_altitude"])
    assert result["tangent_altitude"].dims == ("los",)


@pytest.mark.parametrize("order", [1, 3])
def test_multiplicative_spline_one_reproduces_linear_profile(order):
    element = _multiplicative_one(order=order)
    element.update_state(1.0 + 0.05 * (KNOT_WAVELENGTHS - LOW_NM))
    radiance = _radiance_dataset()
    base = radiance["radiance"].copy(deep=True)

    result = element.modify_input_radiance(radiance)

    factor = np.ones_like(WAVELENGTHS)
    factor[INSIDE] = 1.0 + 0.05 * (WAVELENGTHS[INSIDE] - LOW_NM)
    np.testing.assert_allclose(
        result["radiance"],
        base * xr.DataArray(factor, dims=["wavelength"]),
        rtol=1e-12,
    )


@pytest.mark.parametrize("num_stokes", [1, 2])
def test_multiplicative_spline_one_jacobian_matches_finite_difference(num_stokes):
    element = _multiplicative_one()
    rng = np.random.default_rng(2)
    x0 = 1.0 + 0.1 * rng.standard_normal(NUM_KNOTS)
    element.update_state(x0)
    radiance = _radiance_dataset(num_los=3, num_stokes=num_stokes)

    wf = element.propagate_wf(radiance.copy(deep=True))

    assert wf.dims == ("x", "wavelength", "los", "stokes")
    expected = _finite_difference_jacobian(element, radiance, x0)
    np.testing.assert_allclose(_canonical(wf), expected, rtol=1e-6, atol=1e-10)
    assert np.all(_canonical(wf)[:, OUTSIDE] == 0)


def test_spline_elements_through_state_vector():
    one = _multiplicative_one()
    per_los = _multiplicative(num_los=2)
    one.update_state(np.full(NUM_KNOTS, 2.0))
    per_los.update_state(np.full(2 * NUM_KNOTS, 3.0))
    radiance = _radiance_dataset(num_los=2)
    base = radiance["radiance"].copy(deep=True)

    result = StateVector([one, per_los]).update_sasktran_radiance(
        radiance, drop_old_wf=True
    )

    assert "wf_dummy" not in result
    assert result["wf"].sizes["x"] == NUM_KNOTS + 2 * NUM_KNOTS
    factor = xr.DataArray(np.where(INSIDE, 6.0, 1.0), dims=["wavelength"])
    np.testing.assert_allclose(result["radiance"], base * factor, rtol=1e-12)


def test_spline_post_process_in_altitude_state_vector():
    element = _multiplicative_one()
    element.update_state(np.full(NUM_KNOTS, 2.0))
    sv = AltitudeNativeStateVector(np.array([0.0, 1000.0, 2000.0]), spline=element)
    radiance = _radiance_dataset(num_los=2)
    base = radiance["radiance"].copy(deep=True)

    result = sv.post_process_sk2_radiances(radiance)

    assert "wf_dummy" not in result
    assert result["wf"].sizes == {
        "x": NUM_KNOTS,
        "wavelength": len(WAVELENGTHS),
        "los": 2,
        "stokes": 1,
    }
    factor = xr.DataArray(np.where(INSIDE, 2.0, 1.0), dims=["wavelength"])
    np.testing.assert_allclose(result["radiance"], base * factor, rtol=1e-12)


def test_additive_spline_constant_offset_inside_window():
    element = _additive(num_los=2)
    element.update_state(np.full(2 * NUM_KNOTS, 0.25))
    radiance = _radiance_dataset(num_los=2)
    base = radiance["radiance"].to_numpy().copy()

    result = element.modify_input_radiance(radiance)

    np.testing.assert_allclose(
        result["radiance"].to_numpy()[INSIDE], base[INSIDE] + 0.25, rtol=1e-12
    )


def test_additive_spline_is_enabled_by_default():
    assert _additive().enabled


def test_additive_spline_apriori_state_is_identity():
    element = _additive(num_los=2)
    element.update_state(element.apriori_state())
    radiance = _radiance_dataset(num_los=2)
    base = radiance["radiance"].copy(deep=True)

    result = element.modify_input_radiance(radiance)

    np.testing.assert_allclose(result["radiance"], base)


def test_additive_spline_jacobian_matches_finite_difference():
    element = _additive(num_los=2)
    rng = np.random.default_rng(3)
    x0 = 0.1 * rng.standard_normal(len(element.state()))
    element.update_state(x0)
    radiance = _radiance_dataset(num_los=2)

    wf = element.propagate_wf(radiance.copy(deep=True))

    expected = _finite_difference_jacobian(element, radiance, x0)
    np.testing.assert_allclose(_canonical(wf), expected, rtol=1e-6, atol=1e-10)
