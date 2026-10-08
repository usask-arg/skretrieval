from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from skretrieval.retrieval.statevector.altitude import AltitudeNativeStateVector
from skretrieval.retrieval.statevector.shifts import WavenumberShift
from skretrieval.retrieval.statevector.spline import MultiplicativeSplineOne

SPACING_NM = 0.1
WAVELENGTHS = 500.0 + SPACING_NM * np.arange(41)


def _radiance_dataset(radiance: np.ndarray, wf: np.ndarray | None = None):
    data_vars = {"radiance": (["wavelength", "los", "stokes"], radiance)}
    if wf is not None:
        data_vars["wf_dummy"] = (["altitude", "wavelength", "los", "stokes"], wf)
    return xr.Dataset(data_vars, coords={"wavelength": WAVELENGTHS, "stokes": ["I"]})


def _smooth_dataset(num_los: int = 2):
    los = np.arange(num_los)
    rad = 1.0 + 0.5 * np.sin(
        0.7 * (los[np.newaxis, :] + 1) * WAVELENGTHS[:, np.newaxis]
    )
    alt_scale = np.array([1.0, 2.0, 3.0])
    wf = alt_scale[:, np.newaxis, np.newaxis] * np.cos(
        0.3 * WAVELENGTHS[np.newaxis, :, np.newaxis] + los[np.newaxis, np.newaxis, :]
    )
    return _radiance_dataset(rad[:, :, np.newaxis], wf[:, :, :, np.newaxis])


def _linear_dataset(intercepts, slopes):
    intercepts = np.asarray(intercepts)
    slopes = np.asarray(slopes)
    rad = intercepts[np.newaxis, :] + slopes[np.newaxis, :] * (
        WAVELENGTHS[:, np.newaxis] - WAVELENGTHS[0]
    )
    return _radiance_dataset(rad[:, :, np.newaxis])


def test_wavenumber_shift_defaults():
    element = WavenumberShift(3, tikh_factor=1.0)

    np.testing.assert_array_equal(element.state(), np.zeros(3))
    np.testing.assert_array_equal(element.apriori_state(), np.zeros(3))
    np.testing.assert_array_equal(element.lower_bound(), np.full(3, -0.1))
    np.testing.assert_array_equal(element.upper_bound(), np.full(3, 0.1))
    assert element.enabled
    assert element.name() == "wavenumber_shifts"


def test_wavenumber_shift_custom_bounds_and_name():
    element = WavenumberShift(
        2, tikh_factor=1.0, min_shift=-0.5, max_shift=0.25, apply_to_measurement="uv"
    )

    np.testing.assert_array_equal(element.lower_bound(), np.full(2, -0.5))
    np.testing.assert_array_equal(element.upper_bound(), np.full(2, 0.25))
    assert element.name() == "wavenumber_shifts_uv"


def test_wavenumber_shift_state_is_copied():
    element = WavenumberShift(2, tikh_factor=1.0)
    x = np.array([0.01, -0.02])

    element.update_state(x)
    x[0] = 99.0
    returned = element.state()
    returned[1] = 99.0

    np.testing.assert_array_equal(element.state(), [0.01, -0.02])


def test_wavenumber_shift_inverse_apriori_covariance():
    element = WavenumberShift(3, tikh_factor=2.0, prior_factor=0.5)

    # Half-weighted first differences scaled by tikh_factor=2 -> unit differences
    gamma = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    expected = gamma.T @ gamma + 0.5 * np.eye(3)

    np.testing.assert_allclose(element.inverse_apriori_covariance(), expected)


def test_wavenumber_shift_tikhonov_only_leaves_constant_shift_free():
    element = WavenumberShift(4, tikh_factor=10.0)

    inv_cov = element.inverse_apriori_covariance()

    np.testing.assert_allclose(inv_cov, inv_cov.T)
    np.testing.assert_allclose(inv_cov @ np.ones(4), 0.0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(inv_cov) > -1e-10)

    factor = element.prior_precision_factor()
    np.testing.assert_allclose(factor.T @ factor, inv_cov, atol=1e-10)


def test_wavenumber_shift_zero_shift_is_identity():
    radiance = _smooth_dataset()
    expected = radiance.copy(deep=True)

    result = WavenumberShift(2, tikh_factor=1.0).modify_input_radiance(radiance)

    np.testing.assert_allclose(result["radiance"], expected["radiance"], rtol=1e-12)
    np.testing.assert_allclose(result["wf_dummy"], expected["wf_dummy"], rtol=1e-12)


def test_wavenumber_shift_grid_aligned_shift_matches_manual_shift():
    element = WavenumberShift(2, tikh_factor=1.0, min_shift=-1.0, max_shift=1.0)
    # los 0 is shifted by +2 grid points, los 1 by -1 grid point
    element.update_state(np.array([2 * SPACING_NM, -SPACING_NM]))
    radiance = _smooth_dataset()
    base = radiance["radiance"].isel(stokes=0).to_numpy().copy()
    base_wf = radiance["wf_dummy"].isel(stokes=0).to_numpy().copy()

    result = element.modify_input_radiance(radiance)
    rad = result["radiance"].isel(stokes=0).to_numpy()
    wf = result["wf_dummy"].isel(stokes=0).to_numpy()

    # Interior: R'(lambda_k) = R(lambda_k + shift)
    np.testing.assert_allclose(rad[:-2, 0], base[2:, 0], rtol=1e-10)
    np.testing.assert_allclose(rad[1:, 1], base[:-1, 1], rtol=1e-10)
    np.testing.assert_allclose(wf[:, :-2, 0], base_wf[:, 2:, 0], rtol=1e-10)
    np.testing.assert_allclose(wf[:, 1:, 1], base_wf[:, :-1, 1], rtol=1e-10)

    # Edges are linearly extrapolated from the outermost grid segment
    right_slope = base[-1, 0] - base[-2, 0]
    np.testing.assert_allclose(
        rad[-2:, 0], base[-1, 0] + right_slope * np.array([1.0, 2.0]), rtol=1e-10
    )
    left_slope = base[1, 1] - base[0, 1]
    np.testing.assert_allclose(rad[0, 1], base[0, 1] - left_slope, rtol=1e-10)


@pytest.mark.parametrize("shift", [0.037, -0.083])
def test_wavenumber_shift_linear_spectrum_sub_grid_shift(shift):
    intercepts = np.array([1.0, 2.0])
    slopes = np.array([0.5, -0.25])
    element = WavenumberShift(2, tikh_factor=1.0)
    element.update_state(np.array([shift, 0.0]))

    result = element.modify_input_radiance(_linear_dataset(intercepts, slopes))

    rad = result["radiance"].isel(stokes=0).to_numpy()
    offset = WAVELENGTHS - WAVELENGTHS[0]
    np.testing.assert_allclose(rad[:, 0], 1.0 + 0.5 * (offset + shift), rtol=1e-10)
    # The unshifted line of sight is untouched
    np.testing.assert_allclose(rad[:, 1], 2.0 - 0.25 * offset, rtol=1e-12)


def test_wavenumber_shift_jacobian_for_linear_spectrum():
    slopes = np.array([0.5, -0.25, 2.0])
    element = WavenumberShift(3, tikh_factor=1.0)

    wf = element.propagate_wf(_linear_dataset(np.ones(3), slopes))

    assert wf.dims == ("x", "wavelength", "los", "stokes")
    assert wf.shape == (3, len(WAVELENGTHS), 3, 1)
    expected = np.zeros(wf.shape)
    for i, slope in enumerate(slopes):
        expected[i, :, i, 0] = slope
    np.testing.assert_allclose(wf.to_numpy(), expected, rtol=1e-6, atol=1e-9)


def test_wavenumber_shift_jacobian_matches_finite_difference():
    delta = 1e-4
    element = WavenumberShift(2, tikh_factor=1.0, numerical_delta=delta)
    radiance = _smooth_dataset()

    wf = element.propagate_wf(radiance.copy(deep=True))

    base = element.modify_input_radiance(radiance.copy(deep=True))["radiance"]
    for i in range(2):
        x = np.zeros(2)
        x[i] = delta
        element.update_state(x)
        perturbed = element.modify_input_radiance(radiance.copy(deep=True))
        fd = (perturbed["radiance"] - base) / delta
        np.testing.assert_allclose(
            wf.isel(x=i).transpose("wavelength", "los", "stokes"),
            fd.transpose("wavelength", "los", "stokes"),
            rtol=1e-6,
            atol=1e-9,
        )

    # Sanity check against the analytic derivative of the smooth spectrum
    analytic = 0.5 * 0.7 * np.cos(0.7 * WAVELENGTHS)
    np.testing.assert_allclose(
        wf.isel(x=0, los=0, stokes=0), analytic, atol=0.5 * 0.7**2 * SPACING_NM
    )


def test_wavenumber_shift_post_process_with_spline():
    shift = WavenumberShift(2, tikh_factor=1.0, min_shift=-1.0, max_shift=1.0)
    shift.update_state(np.array([SPACING_NM, 0.0]))
    spline = MultiplicativeSplineOne(501.0, 503.0, 4, 0)
    sv = AltitudeNativeStateVector(
        np.array([0.0, 1000.0, 2000.0]), shift=shift, spline=spline
    )
    radiance = _smooth_dataset()
    base = radiance["radiance"].isel(stokes=0).to_numpy().copy()

    result = sv.post_process_sk2_radiances(radiance)

    assert "wf_dummy" not in result
    assert result["wf"].sizes == {
        "x": 2 + 4,
        "wavelength": len(WAVELENGTHS),
        "los": 2,
        "stokes": 1,
    }
    rad = result["radiance"].isel(stokes=0).to_numpy()
    np.testing.assert_allclose(rad[:-1, 0], base[1:, 0], rtol=1e-10)
    np.testing.assert_allclose(rad[:, 1], base[:, 1], rtol=1e-12)
