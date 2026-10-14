from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from skretrieval.core.lineshape import Gaussian
from skretrieval.doas import _convolve_template


def _exact_gaussian_weights(
    calculation_wavelength_nm: np.ndarray,
    output_wavelength_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    cutoff: float = 5.0,
) -> np.ndarray:
    """
    Closed form int G_i(x) hat_j(x) dx, G_i a Gaussian truncated at ``cutoff`` standard
    deviations and hat_j the linear interpolation basis of the calculation grid, rows normalized
    """
    x = calculation_wavelength_nm
    center = output_wavelength_nm[:, np.newaxis]
    sigma = (fwhm_nm / (2 * np.sqrt(2 * np.log(2))))[:, np.newaxis]

    # Each grid interval, clipped to the truncated support
    left = np.clip(x[:-1], center - cutoff * sigma, center + cutoff * sigma)
    right = np.clip(x[1:], center - cutoff * sigma, center + cutoff * sigma)

    def gaussian(t):
        return np.exp(-0.5 * ((t - center) / sigma) ** 2)

    def erf(t):
        return special.erf((t - center) / (np.sqrt(2) * sigma))

    # int G and int (x - center) G over the clipped intervals
    i0 = sigma * np.sqrt(np.pi / 2) * (erf(right) - erf(left))
    i1 = sigma**2 * (gaussian(left) - gaussian(right))

    width = np.diff(x)
    weights = np.zeros((len(output_wavelength_nm), len(x)))
    # Falling half of the hat at the left node, rising half of the hat at the right node
    weights[:, :-1] += ((x[1:] - center) * i0 - i1) / width
    weights[:, 1:] += (i1 + (center - x[:-1]) * i0) / width
    return weights / weights.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("calculation_wavelength_nm", "fwhm_scale"),
    [
        pytest.param(np.arange(340.123, 380.123, 0.001), 0.12, id="uniform"),
        pytest.param(np.linspace(345.0, 365.0, 501) ** 1.001, 0.0, id="nonuniform"),
    ],
)
def test_doas_convolution_matches_exact_gaussian_integral(
    calculation_wavelength_nm, fwhm_scale
):
    rng = np.random.default_rng(12_345)
    output_wavelength_nm = np.linspace(
        calculation_wavelength_nm[0] + 5, calculation_wavelength_nm[-1] - 5, 117
    )
    normalized_wavelength = (output_wavelength_nm - np.mean(output_wavelength_nm)) / (
        np.ptp(output_wavelength_nm) / 2.0
    )
    fwhm_nm = 0.83 + fwhm_scale * normalized_wavelength
    templates = rng.normal(size=(7, calculation_wavelength_nm.size))

    expected = (
        templates
        @ _exact_gaussian_weights(
            calculation_wavelength_nm, output_wavelength_nm, fwhm_nm
        ).T
    )
    result = _convolve_template(
        calculation_wavelength_nm,
        templates,
        output_wavelength_nm,
        fwhm_nm,
    )

    # Limited by cancellation in the erf differences of the reference
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)


def test_doas_convolution_close_to_legacy_gaussian_lineshape():
    # The legacy Gaussian uses a fast erf approximation, accurate to ~2e-5
    calculation_wavelength_nm = np.arange(340.123, 380.123, 0.001)
    output_wavelength_nm = np.linspace(345.2, 375.4, 117)
    fwhm_nm = np.full(output_wavelength_nm.size, 0.83)
    template = np.sin(calculation_wavelength_nm) + 2

    expected = np.array(
        [
            Gaussian(fwhm=float(width)).integration_weights(
                float(center), calculation_wavelength_nm
            )
            @ template
            for center, width in zip(output_wavelength_nm, fwhm_nm, strict=True)
        ]
    )
    result = _convolve_template(
        calculation_wavelength_nm, template, output_wavelength_nm, fwhm_nm
    )

    np.testing.assert_allclose(result, expected, rtol=1e-4)
