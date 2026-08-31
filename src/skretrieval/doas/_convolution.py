from __future__ import annotations

import numpy as np
from numba import njit

from skretrieval.core.lineshape import _gaussian_analytic_linear_weights_helper


@njit(cache=True)
def uniform_gaussian_integration_weights(
    calculation_wavelength: np.ndarray,
    output_wavelength: np.ndarray,
    fwhm: np.ndarray,
) -> np.ndarray:
    """Build unnormalized Gaussian weights for a uniform calculation grid."""
    weights = np.zeros(
        (output_wavelength.size, calculation_wavelength.size),
        dtype=np.float64,
    )
    spacing = np.diff(calculation_wavelength)
    fwhm_to_stdev = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    for row in range(output_wavelength.size):
        center = output_wavelength[row]
        stdev = max(fwhm[row], 1.0e-6) * fwhm_to_stdev
        lower = np.searchsorted(
            calculation_wavelength,
            center - 5.0 * stdev,
            side="right",
        )
        upper = np.searchsorted(
            calculation_wavelength,
            center + 5.0 * stdev,
            side="left",
        )
        for index in range(lower, upper):
            offset = calculation_wavelength[index] - center
            width_left = abs(spacing[max(index - 1, 0)])
            width_right = abs(spacing[min(index, spacing.size - 1)])
            weights[row, index] = abs(
                _gaussian_analytic_linear_weights_helper(
                    width_left,
                    width_right,
                    offset,
                    stdev,
                )
            )

    return weights
