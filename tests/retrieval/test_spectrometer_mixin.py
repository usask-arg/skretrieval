from __future__ import annotations

import numpy as np
import pytest
import rebasis as rb

from skretrieval.core.lineshape import DeltaFunction, Gaussian
from skretrieval.retrieval.forwardmodel import SpectrometerMixin

SAMPLE_WAVELENGTHS = np.array([500.0, 500.5, 600.0])


class _Spectrometer(SpectrometerMixin):
    def _get_required_wavelength(self):
        return {"band": SAMPLE_WAVELENGTHS}


def test_rebasis_lineshape_gives_same_model_wavelengths_as_lineshape():
    legacy = _Spectrometer(lambda _: Gaussian(fwhm=0.4, max_stdev=5))
    rebasis = _Spectrometer(lambda w: rb.Space.gaussians([w], fwhm=0.4, cutoff=5))

    np.testing.assert_allclose(
        rebasis._construct_model_wavelength()["band"],
        legacy._construct_model_wavelength()["band"],
    )


def test_rebasis_lineshape_instrument_model():
    spectrometer = _Spectrometer(lambda w: rb.Space.gaussians([w], fwhm=0.4, cutoff=5))
    model_wavelengths = spectrometer._construct_model_wavelength()["band"]

    sensor = spectrometer._construct_inst_model()["band"]
    w = sensor._construct_interpolators(model_wavelengths)

    expected = rb.measure(
        rb.Space.gaussians(SAMPLE_WAVELENGTHS, fwhm=0.4, cutoff=5),
        rb.Space.hats(model_wavelengths),
        "rowsum",
    )
    np.testing.assert_allclose(w.toarray(), expected.toarray(), atol=1e-15)


def test_rebasis_lineshape_in_wavenumber_covers_kernel_support():
    res_cminv = 0.05
    spectrometer = _Spectrometer(
        lambda w: rb.Space.gaussians([w], fwhm=0.4, cutoff=5).to_wavenumber(
            jacobian=True
        ),
        model_res_cminv=res_cminv,
        spectral_native_coordinate="wavenumber_cminv",
    )

    model_wavenumbers = 1e7 / spectrometer._construct_model_wavelength()["band"]

    for w in SAMPLE_WAVELENGTHS:
        kernel = rb.Space.gaussians([w], fwhm=0.4, cutoff=5).to_wavenumber(
            jacobian=True
        )
        inside = model_wavenumbers[
            (model_wavenumbers >= kernel.lower[0] - res_cminv)
            & (model_wavenumbers <= kernel.upper[0] + res_cminv)
        ]
        # Sampled at least at the model resolution across the whole support
        assert inside.min() <= kernel.lower[0] + res_cminv
        assert inside.max() >= kernel.upper[0] - res_cminv
        assert np.diff(np.sort(inside)).max() <= res_cminv + 1e-6


@pytest.mark.parametrize("coordinate", ["wavelength_nm", "wavenumber_cminv"])
def test_default_delta_lineshape_model_wavelengths(coordinate):
    spectrometer = _Spectrometer(None, spectral_native_coordinate=coordinate)
    explicit = _Spectrometer(
        lambda _: DeltaFunction(), spectral_native_coordinate=coordinate
    )

    model_wavelengths = spectrometer._construct_model_wavelength()["band"]

    np.testing.assert_array_equal(
        model_wavelengths, explicit._construct_model_wavelength()["band"]
    )
    assert len(model_wavelengths) == len(SAMPLE_WAVELENGTHS)
    np.testing.assert_allclose(model_wavelengths, SAMPLE_WAVELENGTHS, rtol=1e-6)
