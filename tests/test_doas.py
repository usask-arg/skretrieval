from __future__ import annotations

import numpy as np

from skretrieval.core.lineshape import Gaussian
from skretrieval.doas import _convolve_template


def _reference_convolution(
    calculation_wavelength_nm: np.ndarray,
    templates: np.ndarray,
    output_wavelength_nm: np.ndarray,
    fwhm_nm: np.ndarray,
) -> np.ndarray:
    weights = np.stack(
        [
            Gaussian(fwhm=float(width_nm)).integration_weights(
                float(center_nm),
                calculation_wavelength_nm,
            )
            for center_nm, width_nm in zip(
                output_wavelength_nm,
                fwhm_nm,
                strict=True,
            )
        ]
    )
    return templates @ weights.T


def test_uniform_doas_convolution_matches_gaussian_lineshape():
    rng = np.random.default_rng(12_345)
    calculation_wavelength_nm = np.arange(340.123, 380.123, 0.001)
    output_wavelength_nm = np.linspace(345.2, 375.4, 117)
    normalized_wavelength = (
        (output_wavelength_nm - np.mean(output_wavelength_nm))
        / (np.ptp(output_wavelength_nm) / 2.0)
    )
    fwhm_nm = 0.83 + 0.12 * normalized_wavelength
    templates = rng.normal(size=(7, calculation_wavelength_nm.size))

    expected = _reference_convolution(
        calculation_wavelength_nm,
        templates,
        output_wavelength_nm,
        fwhm_nm,
    )
    result = _convolve_template(
        calculation_wavelength_nm,
        templates,
        output_wavelength_nm,
        fwhm_nm,
    )

    np.testing.assert_array_equal(result, expected)


def test_nonuniform_doas_convolution_retains_general_path():
    rng = np.random.default_rng(54_321)
    calculation_wavelength_nm = np.linspace(345.0, 365.0, 501) ** 1.001
    output_wavelength_nm = np.linspace(
        calculation_wavelength_nm[50],
        calculation_wavelength_nm[-50],
        21,
    )
    fwhm_nm = np.full(output_wavelength_nm.size, 0.95)
    templates = rng.normal(size=(3, calculation_wavelength_nm.size))

    expected = _reference_convolution(
        calculation_wavelength_nm,
        templates,
        output_wavelength_nm,
        fwhm_nm,
    )
    result = _convolve_template(
        calculation_wavelength_nm,
        templates,
        output_wavelength_nm,
        fwhm_nm,
    )

    np.testing.assert_array_equal(result, expected)
