from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk
import xarray as xr
from sasktran2.optical.base import OpticalProperty, OpticalQuantities

from skretrieval.core.lineshape import Gaussian
from skretrieval.doas import (
    DOASFitter,
    _convolve_template,
    _design_matrix,
    _extract_constituent_profile,
    _extract_species_wf_array,
    _filter_dataarray_wavelength,
    _filter_measurement_input,
    _manual_predictor_to_components,
    _mean_spectrum_from_radiance,
    _measurement_radiance,
    _measurement_wf,
    _normalized_filter_weights,
    _rt_cross_section_components,
    _set_constituent_profile,
    _solve_linear_coefficients,
    _species_log_radiance_derivative_component,
    _temperature_cross_section_components,
    _wf_to_mean_spectrum,
    doas_fit,
)
from skretrieval.doas._convolution import uniform_gaussian_integration_weights

WAVELENGTH_NM = np.round(np.arange(330.0, 340.0 + 1e-9, 0.1), 10)
TANGENT_ALTITUDES_M = np.array([15_000.0, 25_000.0, 35_000.0])
CALC_MARGIN_NM = 2.0
CALC_SPACING_NM = 0.01
TRUE_CALIBRATION = {
    "shift": 0.02,
    "stretch": 2.0e-3,
    "fwhm_zero": 0.4,
    "fwhm_slope": 0.05,
}
TRUE_SCD = np.array([1.0e19, 3.0e19, 6.0e19])


# ---------------------------------------------------------------------------
# Synthetic spectroscopy and offline stand-ins for database-backed inputs
# ---------------------------------------------------------------------------


def _bands(wavelength_nm, centers, widths, depths) -> np.ndarray:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    out = np.zeros_like(wavelength_nm)
    for center, width, depth in zip(centers, widths, depths, strict=True):
        out += depth * np.exp(-0.5 * ((wavelength_nm - center) / width) ** 2)
    return out


def _solar_irradiance(wavelength_nm) -> np.ndarray:
    """Sloped continuum with narrow Fraunhofer-like lines."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    lines = _bands(
        wavelength_nm,
        [331.3, 333.1, 334.4, 336.2, 337.9, 339.0],
        [0.05, 0.08, 0.04, 0.1, 0.06, 0.05],
        [0.5, 0.3, 0.6, 0.4, 0.5, 0.3],
    )
    return 1.0 + 0.01 * (wavelength_nm - 335.0) - lines


def _bro_like_cross_section(wavelength_nm, temperature_k=220.0) -> np.ndarray:
    base = _bands(
        wavelength_nm,
        [330.5, 332.4, 334.0, 335.7, 337.3, 339.2],
        [0.3] * 6,
        [1.0, 0.8, 1.2, 0.9, 1.1, 0.7],
    )
    temperature_shape = _bands(
        wavelength_nm, [331.0, 333.5, 336.5, 338.5], [0.15] * 4, [1.0, -0.6, 0.8, -0.5]
    )
    return 1e-20 * (0.2 + base + 0.01 * (temperature_k - 220.0) * temperature_shape)


def _oclo_like_cross_section(wavelength_nm, temperature_k=220.0) -> np.ndarray:
    base = _bands(wavelength_nm, [331.8, 335.0, 338.1], [0.5] * 3, [1.0, 1.4, 0.9])
    return 1e-20 * (0.1 + base) * (1.0 + 1e-3 * (temperature_k - 220.0))


def _ozone_like_cross_section(wavelength_nm, temperature_k=220.0) -> np.ndarray:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    structure = 1.0 + 0.4 * np.sin(2.0 * np.pi * (wavelength_nm - 330.0) / 2.7)
    decay = np.exp(-(wavelength_nm - 330.0) / 8.0)
    return 1e-23 * structure * decay * (1.0 + 2e-3 * (temperature_k - 220.0))


def _temperature_profile(altitudes_m) -> np.ndarray:
    return 260.0 - 1.5e-3 * np.asarray(altitudes_m, dtype=float)


class _SyntheticCrossSection(OpticalProperty):
    """Absorber whose cross section is an analytic function of (wavelength, T)."""

    def __init__(self, cross_section):
        self._cross_section = cross_section

    def atmosphere_quantities(self, atmo, **kwargs) -> OpticalQuantities:
        wavelength = np.asarray(atmo.wavelengths_nm, dtype=float)
        extinction = np.stack(
            [
                self._cross_section(wavelength, float(temperature))
                for temperature in np.atleast_1d(atmo.temperature_k)
            ]
        )
        return OpticalQuantities(extinction=extinction, ssa=np.zeros_like(extinction))


class _SyntheticAncillary:
    """Provides everything DOASFitter needs from ``anc`` (including ``o3``)."""

    def __init__(self, extra_absorbers=None):
        self._extra_absorbers = dict(extra_absorbers or {})

    def add_to_atmosphere(self, atmo):
        altitudes = np.asarray(atmo.model_geometry.altitudes(), dtype=float)
        atmo.temperature_k = _temperature_profile(altitudes)
        atmo.pressure_pa = 101325.0 * np.exp(-altitudes / 7000.0)
        atmo["rayleigh"] = sk.constituent.Rayleigh()
        atmo["o3"] = sk.constituent.VMRAltitudeAbsorber(
            _SyntheticCrossSection(_ozone_like_cross_section),
            altitudes,
            np.full(altitudes.size, 1e-6),
        )
        for name, (cross_section, vmr) in self._extra_absorbers.items():
            atmo[name] = sk.constituent.VMRAltitudeAbsorber(
                _SyntheticCrossSection(cross_section),
                altitudes,
                np.full(altitudes.size, vmr),
            )


class _SyntheticSolarModel:
    def __init__(self, *args, **kwargs):
        pass

    def irradiance(self, wavelength_nm, **kwargs):
        return _solar_irradiance(wavelength_nm)


@pytest.fixture(autouse=True)
def _offline_solar_model(monkeypatch):
    # DOASFitter loads the HSRS solar spectrum from the sasktran2 database,
    # which would require a download; substitute an analytic spectrum.
    monkeypatch.setattr(sk.solar, "SolarModel", _SyntheticSolarModel)


# ---------------------------------------------------------------------------
# Independent forward model following the DOASFitter conventions
# ---------------------------------------------------------------------------


def _calc_wavelength(wavelength_nm=WAVELENGTH_NM) -> np.ndarray:
    return np.arange(
        wavelength_nm.min() - CALC_MARGIN_NM,
        wavelength_nm.max() + CALC_MARGIN_NM,
        CALC_SPACING_NM,
    )


def _instrument_model(
    templates,
    *,
    shift,
    stretch,
    fwhm_zero,
    fwhm_slope,
    wavelength_nm=WAVELENGTH_NM,
):
    """Shift/stretch the wavelength grid and convolve with a Gaussian ILS."""
    calc = _calc_wavelength(wavelength_nm)
    center = np.mean(wavelength_nm)
    transformed = center + (1.0 + stretch) * (wavelength_nm - center) + shift
    normalized = (transformed - center) / (np.ptp(wavelength_nm) / 2.0)
    fwhm = fwhm_zero + fwhm_slope * normalized
    weights = np.stack(
        [
            Gaussian(fwhm=float(width)).integration_weights(float(mean), calc)
            for mean, width in zip(transformed, fwhm, strict=True)
        ]
    )
    return np.atleast_2d(templates) @ weights.T, transformed, fwhm, normalized


def _convolved_references(cross_sections, calibration=None):
    """Return (convolved cross sections, convolved irradiance, normalized coord)."""
    calibration = calibration or TRUE_CALIBRATION
    calc = _calc_wavelength()
    templates = np.vstack(
        [np.atleast_2d(cross_sections), _solar_irradiance(calc)[np.newaxis, :]]
    )
    convolved, _, _, normalized = _instrument_model(templates, **calibration)
    return convolved[:-1], convolved[-1], normalized


def _closure_terms(num_samples, convolved_irradiance, normalized) -> np.ndarray:
    """Per-sample irradiance scaling and broadband polynomial (log space)."""
    irradiance = convolved_irradiance / np.mean(convolved_irradiance)
    return np.array(
        [
            -1.0
            - 0.1 * idx
            + (0.3 + 0.05 * idx) * irradiance
            - (0.2 - 0.03 * idx) * normalized
            + 0.05 * normalized**2
            - 0.01 * idx * normalized**3
            for idx in range(num_samples)
        ]
    )


def _bro_scene(slant_columns=TRUE_SCD):
    calc = _calc_wavelength()
    convolved_xs, convolved_irradiance, normalized = _convolved_references(
        _bro_like_cross_section(calc)
    )
    log_radiance = -slant_columns[:, np.newaxis] * convolved_xs[0] + _closure_terms(
        len(slant_columns), convolved_irradiance, normalized
    )
    return log_radiance, convolved_xs[0]


def _radiance(log_radiance, dim="tangent_altitude", coord=TANGENT_ALTITUDES_M):
    return xr.DataArray(
        np.exp(log_radiance),
        dims=(dim, "wavelength"),
        coords={dim: coord, "wavelength": WAVELENGTH_NM},
    )


def _make_fitter(radiances, anc=None, **kwargs) -> DOASFitter:
    options = {
        "optical": {"bro": _SyntheticCrossSection(_bro_like_cross_section)},
        "absorber_temperatures": {"bro": 220.0},
        "calc_margin": CALC_MARGIN_NM,
        "calc_spacing": CALC_SPACING_NM,
        "poly_order": 3,
    }
    options.update(kwargs)
    return DOASFitter(radiances, anc or _SyntheticAncillary(), **options)


def _standardize(column) -> np.ndarray:
    column = np.asarray(column, dtype=float)
    return (column - np.mean(column)) / np.std(column)


def _rms(values) -> float:
    return float(np.sqrt(np.nanmean(np.square(values))))


# ---------------------------------------------------------------------------
# Measurement filtering
# ---------------------------------------------------------------------------


def test_normalized_filter_weights_default_and_normalisation():
    np.testing.assert_array_equal(_normalized_filter_weights(None), [1.0])
    np.testing.assert_allclose(
        _normalized_filter_weights([[1.0, 2.0], [1.0, 0.0]]),
        [0.25, 0.5, 0.25, 0.0],
    )


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ([], "at least one value"),
        ([1.0, np.nan], "must be finite"),
        ([1.0, np.inf], "must be finite"),
        ([1.0, -1.0], "must not sum to zero"),
    ],
)
def test_normalized_filter_weights_rejects_invalid_input(weights, match):
    with pytest.raises(ValueError, match=match):
        _normalized_filter_weights(weights)


def test_filter_dataarray_smooths_along_wavelength_and_keeps_layout():
    rng = np.random.default_rng(7)
    values = rng.uniform(1.0, 2.0, size=(9, 4))
    data = xr.DataArray(
        values,
        dims=("wavelength", "los"),
        coords={"wavelength": np.arange(9.0)},
        attrs={"units": "W"},
    )
    weights = np.array([0.25, 0.5, 0.25])

    result = _filter_dataarray_wavelength(data, weights)

    expected = np.stack(
        [np.convolve(values[:, col], weights, mode="same") for col in range(4)],
        axis=1,
    )
    assert result.dims == ("wavelength", "los")
    assert result.attrs == {"units": "W"}
    np.testing.assert_array_equal(result.wavelength, data.wavelength)
    np.testing.assert_allclose(result.to_numpy(), expected)


def test_filter_passthrough_for_identity_weights_or_missing_wavelength():
    data = xr.DataArray(np.ones((2, 3)), dims=("los", "wavelength"))
    no_wavelength = xr.DataArray(np.ones(3), dims=("los",))

    assert _filter_dataarray_wavelength(data, np.array([1.0])) is data
    assert _filter_dataarray_wavelength(no_wavelength, np.array([0.5, 0.5])) is (
        no_wavelength
    )
    assert _filter_measurement_input(data, np.array([1.0])) is data


def test_filter_measurement_input_filters_radiance_and_wf_only():
    rng = np.random.default_rng(11)
    measurement = xr.Dataset(
        {
            "radiance": (("los", "wavelength"), rng.uniform(1, 2, size=(2, 8))),
            "wf": (("los", "wavelength", "x"), rng.normal(size=(2, 8, 3))),
            "flag": (("los", "wavelength"), rng.uniform(size=(2, 8))),
        }
    )
    original = measurement.copy(deep=True)
    weights = np.array([0.25, 0.5, 0.25])

    result = _filter_measurement_input(measurement, weights)

    expected_radiance = np.apply_along_axis(
        np.convolve, -1, original["radiance"].to_numpy(), weights, mode="same"
    )
    expected_wf = np.apply_along_axis(
        np.convolve, 1, original["wf"].to_numpy(), weights, mode="same"
    )
    np.testing.assert_allclose(result["radiance"].to_numpy(), expected_radiance)
    np.testing.assert_allclose(
        result["wf"].transpose("los", "wavelength", "x").to_numpy(), expected_wf
    )
    xr.testing.assert_identical(result["flag"], original["flag"])
    xr.testing.assert_identical(measurement, original)

    data_array = _filter_measurement_input(original["radiance"], weights)
    np.testing.assert_allclose(data_array.to_numpy(), expected_radiance)


@pytest.mark.xfail(
    raises=AssertionError,
    reason=(
        "np.convolve(mode='same') zero-pads, so a normalised smoothing filter "
        "attenuates the first/last samples of the fit window"
    ),
    strict=True,
)
def test_filter_preserves_constant_spectrum_at_band_edges():
    data = xr.DataArray(
        np.full((2, 6), 3.0),
        dims=("los", "wavelength"),
        coords={"wavelength": np.arange(6.0)},
    )
    weights = _normalized_filter_weights([1.0, 1.0, 1.0])

    result = _filter_dataarray_wavelength(data, weights)

    np.testing.assert_allclose(result.to_numpy(), 3.0)


def test_filter_preserves_non_dimension_coordinates():
    data = xr.DataArray(
        np.ones((2, 6)),
        dims=("los", "wavelength"),
        coords={
            "wavelength": np.arange(6.0),
            "tangent_altitude": ("los", [10_000.0, 20_000.0]),
        },
    )

    result = _filter_dataarray_wavelength(data, np.array([0.25, 0.5, 0.25]))

    assert "tangent_altitude" in result.coords


# ---------------------------------------------------------------------------
# Instrument line shape convolution
# ---------------------------------------------------------------------------


def test_uniform_weights_match_unnormalised_lineshape_weights():
    calc = np.arange(340.0, 350.0, 0.01)
    centers = np.array([340.0, 341.234, 345.0, 349.99, 353.0])
    fwhm = np.array([0.3, 0.5, 0.8, 0.4, 0.5])

    weights = uniform_gaussian_integration_weights(calc, centers, fwhm)

    expected = np.stack(
        [
            Gaussian(fwhm=width).integration_weights(center, calc, normalize=False)
            for center, width in zip(centers, fwhm, strict=True)
        ]
    )
    np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=1e-15)
    # The last output is outside the grid by more than five standard deviations.
    assert not np.any(weights[-1])


@pytest.mark.parametrize(
    "calc_wavelength",
    [
        pytest.param(np.arange(340.0, 360.0, 0.002), id="uniform"),
        pytest.param(
            np.concatenate(
                [np.arange(340.0, 350.0, 0.002), np.arange(350.0, 360.0, 0.003)]
            ),
            id="nonuniform",
        ),
    ],
)
def test_convolve_template_matches_analytic_gaussian_broadening(calc_wavelength):
    line_sigma = 0.05
    instrument_fwhm = 0.5
    instrument_sigma = instrument_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    template = np.exp(-0.5 * ((calc_wavelength - 350.0) / line_sigma) ** 2)
    output_wavelength = np.linspace(348.0, 352.0, 81)

    result = _convolve_template(
        calc_wavelength,
        template,
        output_wavelength,
        np.full(output_wavelength.size, instrument_fwhm),
    )

    total_sigma = np.hypot(line_sigma, instrument_sigma)
    expected = (line_sigma / total_sigma) * np.exp(
        -0.5 * ((output_wavelength - 350.0) / total_sigma) ** 2
    )
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_convolve_template_single_template_matches_batched_row():
    calc = np.arange(340.0, 350.0, 0.01)
    rng = np.random.default_rng(5)
    templates = rng.normal(size=(2, calc.size))
    output = np.linspace(342.0, 348.0, 13)
    fwhm = np.full(output.size, 0.4)

    batched = _convolve_template(calc, templates, output, fwhm)
    single = _convolve_template(calc, templates[1], output, fwhm)

    assert single.shape == (output.size,)
    np.testing.assert_allclose(single, batched[1], rtol=1e-12, atol=1e-15)


def test_convolve_template_conserves_level_and_handles_degenerate_rows():
    calc = np.arange(340.0, 360.0, 0.01)
    output = np.array([345.0, calc[1000], 400.0])
    # Zero FWHM is floored rather than producing NaNs; an output far outside
    # the calculation grid gets no contribution at all.
    fwhm = np.array([0.5, 0.0, 0.5])

    result = _convolve_template(calc, np.full(calc.size, 2.5), output, fwhm)

    np.testing.assert_allclose(result[:2], 2.5, rtol=1e-12)
    assert result[2] == 0.0


def test_convolve_template_rejects_template_length_mismatch():
    calc = np.arange(340.0, 341.0, 0.1)
    with pytest.raises(ValueError, match="must match calc_wavel"):
        _convolve_template(calc, np.ones(calc.size + 1), np.array([340.5]), [0.2])


def test_convolve_template_descending_uniform_grid_matches_ascending():
    calc = np.arange(340.0, 360.0, 0.01)
    template = np.sin(calc)
    output = np.linspace(345.0, 355.0, 11)
    fwhm = np.full(output.size, 0.5)

    ascending = _convolve_template(calc, template, output, fwhm)
    descending = _convolve_template(
        calc[::-1].copy(), template[::-1].copy(), output, fwhm
    )

    np.testing.assert_allclose(descending, ascending, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Manual predictors
# ---------------------------------------------------------------------------


def test_manual_predictor_on_calc_grid_is_used_directly():
    calc = np.linspace(330.0, 340.0, 11)
    values = np.arange(11.0)

    result = _manual_predictor_to_components("ring", values, calc)

    np.testing.assert_array_equal(result, values[np.newaxis, :])


@pytest.mark.parametrize("container", ["tuple", "dict", "dataarray"])
def test_manual_predictor_is_sorted_deduplicated_and_interpolated(container):
    calc = np.linspace(330.0, 342.0, 121)
    wavelength = np.array([338.0, 332.0, np.nan, 335.0, 332.0, 340.5])
    values = np.array(
        [
            [3.0, 1.0, 99.0, 2.0, 1.0, 4.0],
            [30.0, 10.0, 99.0, np.nan, 10.0, 40.0],
        ]
    )
    if container == "tuple":
        predictor = (wavelength, values)
    elif container == "dict":
        predictor = {"wavelength": wavelength, "values": values}
    else:
        predictor = xr.DataArray(
            values,
            dims=("component", "wavelength"),
            coords={"wavelength": wavelength},
        )

    result = _manual_predictor_to_components("ring", predictor, calc)

    unique_wavelength = [332.0, 335.0, 338.0, 340.5]
    # Values beyond the predictor range are held constant and NaN values are
    # replaced by zero before interpolation.
    expected = np.vstack(
        [
            np.interp(calc, unique_wavelength, [1.0, 2.0, 3.0, 4.0]),
            np.interp(calc, unique_wavelength, [10.0, 0.0, 30.0, 40.0]),
        ]
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("predictor", "match"),
    [
        ({"wavelength": [330.0, 331.0]}, "must contain a 'values' entry"),
        (np.ones((2, 2, 11)), "must be 1D or 2D"),
        (np.ones(5), "does not provide wavelengths and its length does not match"),
        (([330.0, 331.0, 332.0], [1.0, 2.0]), r"size mismatch \(3 vs 2\)"),
        (([np.nan, np.nan], [1.0, 2.0]), "has no finite wavelengths"),
        (([331.0, 331.0], [1.0, 1.0]), "at least two unique wavelengths"),
    ],
)
def test_manual_predictor_rejects_invalid_input(predictor, match):
    calc = np.linspace(330.0, 340.0, 11)
    with pytest.raises(ValueError, match=match):
        _manual_predictor_to_components("ring", predictor, calc)


# ---------------------------------------------------------------------------
# Design matrix and linear solve
# ---------------------------------------------------------------------------


def test_design_matrix_column_layout_and_standardisation():
    rng = np.random.default_rng(21)
    wavelengths = np.linspace(330.0, 340.0, 51)
    calc = np.arange(328.0, 342.0, CALC_SPACING_NM)
    xs = {
        "a": np.vstack([_bro_like_cross_section(calc), _oclo_like_cross_section(calc)]),
        "b": _ozone_like_cross_section(calc),
    }
    residual_basis = rng.normal(size=(1, wavelengths.size))
    tilt_basis = rng.normal(size=(2, wavelengths.size))

    design, convolved_xs, convolved_irradiance, transformed = _design_matrix(
        wavelengths,
        calc,
        xs,
        _solar_irradiance(calc),
        2,
        shift=0.01,
        stretch=1e-3,
        fwhm_zero=0.4,
        fwhm_slope=0.1,
        nonlinear_orders={"b": (2,)},
        residual_basis=residual_basis,
        tilt_basis=tilt_basis,
    )

    # a(2) + b(1) + b^2(1) + irradiance(1) + poly(3) + residual(1) + tilt(2)
    assert design.shape == (wavelengths.size, 11)
    assert convolved_xs["a"].shape == (2, wavelengths.size)
    assert convolved_xs["b"].shape == (1, wavelengths.size)

    expected_transformed = 335.0 + 1.001 * (wavelengths - 335.0) + 0.01
    np.testing.assert_allclose(transformed, expected_transformed)

    normalized = (expected_transformed - 335.0) / 5.0
    expected_columns = [
        _standardize(convolved_xs["a"][0]),
        _standardize(convolved_xs["a"][1]),
        _standardize(convolved_xs["b"][0]),
        _standardize(convolved_xs["b"][0] ** 2),
        _standardize(convolved_irradiance),
        np.ones_like(normalized),
        normalized,
        normalized**2,
        _standardize(residual_basis[0]),
        _standardize(tilt_basis[0]),
        _standardize(tilt_basis[1]),
    ]
    np.testing.assert_allclose(design, np.column_stack(expected_columns), atol=1e-10)

    reference, _, _, _ = _instrument_model(
        np.vstack([xs["a"], _solar_irradiance(calc)]),
        shift=0.01,
        stretch=1e-3,
        fwhm_zero=0.4,
        fwhm_slope=0.1,
        wavelength_nm=wavelengths,
    )
    np.testing.assert_allclose(convolved_xs["a"], reference[:2], rtol=1e-10)
    np.testing.assert_allclose(convolved_irradiance, reference[2], rtol=1e-10)


def test_design_matrix_manual_offset_shifts_only_the_named_predictor():
    wavelengths = np.linspace(330.0, 340.0, 101)
    calc = np.arange(328.0, 342.0, CALC_SPACING_NM)
    xs = {
        "bro": _bro_like_cross_section(calc),
        "ring": _bands(calc, [331.0, 334.5, 338.2], [0.2] * 3, [1.0, 0.5, 0.8]),
    }
    kwargs = {"shift": 0.0, "stretch": 0.0, "fwhm_zero": 0.3, "fwhm_slope": 0.0}

    _, base, _, _ = _design_matrix(
        wavelengths, calc, xs, _solar_irradiance(calc), 1, **kwargs
    )
    _, shifted, _, _ = _design_matrix(
        wavelengths,
        calc,
        xs,
        _solar_irradiance(calc),
        1,
        manual_predictor_offsets={"ring": 0.1, "bro": 0.0, "not_a_predictor": 0.3},
        **kwargs,
    )

    # A +0.1 nm offset moves the predictor by exactly one output sample.
    np.testing.assert_allclose(shifted["ring"][0, 1:], base["ring"][0, :-1], rtol=1e-9)
    assert shifted["ring"][0, 0] == base["ring"][0, 0]
    np.testing.assert_array_equal(shifted["bro"], base["bro"])


def test_solve_linear_coefficients_respects_mask_and_empty_rows():
    rng = np.random.default_rng(3)
    design = rng.normal(size=(40, 4))
    truth = rng.normal(size=(3, 4))
    observations = truth @ design.T
    mask = np.ones_like(observations, dtype=bool)
    observations[0, :5] = 1e6
    mask[0, :5] = False
    mask[2] = False

    coefficients, fitted = _solve_linear_coefficients(design, observations, mask)

    np.testing.assert_allclose(coefficients[:2], truth[:2], rtol=1e-10)
    # The model is evaluated on the full grid, including masked samples.
    np.testing.assert_allclose(fitted[0], design @ truth[0], rtol=1e-10)
    np.testing.assert_array_equal(coefficients[2], 0.0)
    assert np.all(np.isnan(fitted[2]))


# ---------------------------------------------------------------------------
# Measurement / weighting-function helpers
# ---------------------------------------------------------------------------


def test_measurement_radiance_requires_radiance_variable():
    data = xr.DataArray(np.ones(3), dims=("wavelength",))
    measurement = xr.Dataset({"radiance": data})

    assert _measurement_radiance(data) is data
    xr.testing.assert_identical(
        _measurement_radiance(measurement), data.rename("radiance")
    )
    with pytest.raises(ValueError, match="requires a 'radiance' variable"):
        _measurement_radiance(xr.Dataset({"signal": data}))


def test_measurement_wf_collapses_extra_dims_and_orders_axes():
    rng = np.random.default_rng(13)
    wf = rng.normal(size=(5, 2, 3, 4))
    measurement = xr.Dataset(
        {
            "radiance": (("los", "wavelength"), np.ones((3, 5))),
            "wf": (("wavelength", "stokes", "los", "x"), wf),
        },
        coords={"x": ["a", "b", "c", "d"]},
    )

    values, x_values = _measurement_wf(measurement, ["los"])

    np.testing.assert_array_equal(values, wf[:, 0].transpose(1, 0, 2))
    np.testing.assert_array_equal(x_values, ["a", "b", "c", "d"])

    assert _measurement_wf(measurement["radiance"], ["los"]) is None
    assert _measurement_wf(measurement.drop_vars("wf"), ["los"]) is None
    assert _measurement_wf(measurement, ["time"]) is None
    no_x = measurement.assign(wf=(("los", "wavelength"), np.ones((3, 5))))
    assert _measurement_wf(no_x, ["los"]) is None


def test_mean_spectrum_from_radiance_averages_finite_lines_of_sight():
    rng = np.random.default_rng(17)
    values = rng.uniform(1.0, 2.0, size=(6, 3, 1))
    values[:, 1, 0] = np.nan
    measurement = xr.Dataset({"radiance": (("wavelength", "los", "stokes"), values)})

    spectrum = _mean_spectrum_from_radiance(measurement, 6)

    np.testing.assert_allclose(spectrum, values[:, [0, 2], 0].mean(axis=1))
    np.testing.assert_array_equal(
        _mean_spectrum_from_radiance(xr.DataArray(values[:, 0, 0]), 6),
        values[:, 0, 0],
    )


@pytest.mark.parametrize(
    ("values", "num_wavel", "match"),
    [
        (np.ones((3, 4)), 6, "Unable to identify wavelength axis"),
        (np.full((3, 6), np.nan), 6, "does not contain finite samples"),
        (np.ones(5), 6, "does not match atmosphere grid"),
    ],
)
def test_mean_spectrum_from_radiance_rejects_unusable_output(values, num_wavel, match):
    with pytest.raises(ValueError, match=match):
        _mean_spectrum_from_radiance(xr.DataArray(values), num_wavel)


def test_wf_to_mean_spectrum_moves_wavelength_axis_and_skips_nan_rows():
    values = np.arange(24.0).reshape(6, 4)
    values[:, 2] = np.nan

    spectrum = _wf_to_mean_spectrum(values, 6)

    np.testing.assert_allclose(spectrum, values[:, [0, 1, 3]].mean(axis=1))
    np.testing.assert_array_equal(
        _wf_to_mean_spectrum(np.arange(6.0), 6), np.arange(6.0)
    )


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.ones(5), "wavelength size mismatch"),
        (np.ones((3, 4)), "Unable to identify wavelength axis"),
        (np.full((2, 6), np.nan), "does not contain finite samples"),
    ],
)
def test_wf_to_mean_spectrum_rejects_unusable_input(values, match):
    with pytest.raises(ValueError, match=match):
        _wf_to_mean_spectrum(values, 6)


def test_species_wf_lookup_prefers_named_variables_then_x_coordinate():
    rng = np.random.default_rng(19)
    bro_vmr = rng.normal(size=(2, 6))
    bro_density = rng.normal(size=(2, 6))
    o3_vmr = rng.normal(size=(2, 6))
    stacked = rng.normal(size=(3, 6))
    measurement = xr.Dataset(
        {
            "radiance": (("wavelength",), np.ones(6)),
            "wf_bro_vmr": (("altitude", "wavelength"), bro_vmr),
            "wf_bro_number_density": (("altitude", "wavelength"), bro_density),
            "wf_o3_vmr": (("altitude", "wavelength"), o3_vmr),
            "wf": (("x", "wavelength"), stacked),
        },
        coords={"x": ["no2_scd", "aerosol", "NO2_shift"]},
    )

    np.testing.assert_allclose(
        _extract_species_wf_array(measurement, "BrO"), (bro_vmr + bro_density) / 2
    )
    np.testing.assert_array_equal(_extract_species_wf_array(measurement, "o3"), o3_vmr)
    np.testing.assert_array_equal(
        _extract_species_wf_array(measurement, "no2"), stacked[[0, 2]]
    )
    assert _extract_species_wf_array(measurement, "so2") is None
    assert _extract_species_wf_array(measurement["radiance"], "bro") is None
    assert _extract_species_wf_array(measurement[["radiance"]], "bro") is None
    no_x = xr.Dataset({"wf": (("wavelength",), np.ones(6))})
    assert _extract_species_wf_array(no_x, "bro") is None


def test_species_log_radiance_derivative_is_mean_wf_over_mean_radiance():
    radiance = np.array([[1.0, 2.0, 4.0], [3.0, 2.0, 0.0]])
    wf = np.array([[0.5, -1.0, 2.0], [1.5, -3.0, 0.0]])
    measurement = xr.Dataset(
        {
            "radiance": (("los", "wavelength"), radiance),
            "wf_bro_vmr": (("los", "wavelength"), wf),
        }
    )

    derivative = _species_log_radiance_derivative_component(measurement, "bro", 3)

    np.testing.assert_allclose(derivative, wf.mean(axis=0) / radiance.mean(axis=0))
    assert _species_log_radiance_derivative_component(measurement, "no2", 3) is None


# ---------------------------------------------------------------------------
# Cross-section bases from optical properties / radiative transfer
# ---------------------------------------------------------------------------


class _FakeConstituent:
    def __init__(self, **profiles):
        for name, values in profiles.items():
            setattr(self, name, np.asarray(values, dtype=float))


class _FakeAtmosphere:
    def __init__(self, wavelengths_nm, temperature_k, constituents=None):
        self.wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
        self.num_wavel = self.wavelengths_nm.size
        self.temperature_k = np.asarray(temperature_k, dtype=float)
        self._constituents = dict(constituents or {})

    def __contains__(self, name):
        return name in self._constituents

    def __getitem__(self, name):
        return self._constituents[name]


class _WavelengthMajorCrossSection:
    """Returns extinction as (wavelength, altitude) with an invalid altitude."""

    def __init__(self, extinction_override=None):
        self._override = extinction_override

    def atmosphere_quantities(self, atmo):
        if self._override is not None:
            return OpticalQuantities(extinction=self._override)
        extinction = np.stack(
            [
                _bro_like_cross_section(atmo.wavelengths_nm, temperature)
                for temperature in atmo.temperature_k
            ],
            axis=1,
        )
        extinction[:, 0] = np.nan
        return OpticalQuantities(extinction=extinction)


def test_temperature_components_evaluate_each_temperature_and_restore_state():
    wavelengths = np.linspace(330.0, 340.0, 50)
    original_temperature = np.array([250.0, 240.0, 230.0, 220.0])
    atmo = _FakeAtmosphere(wavelengths, original_temperature)

    components, variance_ratio = _temperature_cross_section_components(
        _WavelengthMajorCrossSection(), atmo, [210.0, 230.0]
    )

    np.testing.assert_allclose(
        components,
        np.vstack(
            [
                _bro_like_cross_section(wavelengths, 210.0),
                _bro_like_cross_section(wavelengths, 230.0),
            ]
        ),
    )
    np.testing.assert_array_equal(variance_ratio, [0.5, 0.5])
    np.testing.assert_array_equal(atmo.temperature_k, original_temperature)


@pytest.mark.parametrize(
    ("extinction", "temperatures", "match"),
    [
        (None, [], "At least one temperature"),
        (np.ones((4, 7)), [220.0], "Unable to identify wavelength axis"),
        (np.full((4, 50), np.nan), [220.0], "does not contain finite samples"),
        (np.ones(7), [220.0], "does not match atmosphere grid"),
    ],
)
def test_temperature_components_reject_unusable_extinction(
    extinction, temperatures, match
):
    original_temperature = np.full(4, 250.0)
    atmo = _FakeAtmosphere(np.linspace(330.0, 340.0, 50), original_temperature)

    with pytest.raises(ValueError, match=match):
        _temperature_cross_section_components(
            _WavelengthMajorCrossSection(extinction), atmo, temperatures
        )
    np.testing.assert_array_equal(atmo.temperature_k, original_temperature)


def test_constituent_profile_helpers_support_vmr_and_number_density():
    atmo = _FakeAtmosphere(
        [330.0, 331.0],
        [220.0],
        {
            "bro": _FakeConstituent(vmr=[1.0, 2.0]),
            "no2": _FakeConstituent(number_density=[3.0, 4.0]),
            "aerosol": _FakeConstituent(extinction=[1.0]),
        },
    )

    kind, profile = _extract_constituent_profile(atmo, "no2")
    assert kind == "number_density"
    profile[:] = 0.0  # A copy is returned; the atmosphere is unaffected.
    np.testing.assert_array_equal(atmo["no2"].number_density, [3.0, 4.0])

    _set_constituent_profile(atmo, "no2", "number_density", np.array([5.0, 6.0]))
    _set_constituent_profile(atmo, "bro", "vmr", np.array([7.0, 8.0]))
    np.testing.assert_array_equal(atmo["no2"].number_density, [5.0, 6.0])
    np.testing.assert_array_equal(atmo["bro"].vmr, [7.0, 8.0])

    with pytest.raises(ValueError, match="'so2' is not present in atmosphere"):
        _extract_constituent_profile(atmo, "so2")
    with pytest.raises(ValueError, match="does not expose vmr or number_density"):
        _extract_constituent_profile(atmo, "aerosol")
    with pytest.raises(ValueError, match="Unsupported constituent profile kind"):
        _set_constituent_profile(atmo, "bro", "mass", np.zeros(2))


class _BeerLambertEngine:
    """Single line of sight with optical depth = column * sigma(lambda, T)."""

    def __init__(self, path_length, fail_on_call=None):
        self._path_length = path_length
        self._fail_on_call = fail_on_call
        self.calls = 0

    def calculate_radiance(self, atmo):
        self.calls += 1
        if self.calls == self._fail_on_call:
            msg = "engine failure"
            raise RuntimeError(msg)
        column = np.sum(atmo["bro"].vmr) * self._path_length
        optical_depth = column * _bro_like_cross_section(
            atmo.wavelengths_nm, float(atmo.temperature_k[0])
        )
        radiance = _solar_irradiance(atmo.wavelengths_nm) * np.exp(-optical_depth)
        return xr.Dataset({"radiance": (("los", "wavelength"), radiance[np.newaxis])})


def test_rt_components_are_slant_optical_depths_and_state_is_restored():
    wavelengths = np.linspace(330.0, 340.0, 50)
    vmr = np.array([1e-12, 2e-12, 1e-12])
    original_temperature = np.array([250.0, 230.0, 210.0])
    atmo = _FakeAtmosphere(
        wavelengths, original_temperature, {"bro": _FakeConstituent(vmr=vmr)}
    )
    path_length = 2e30

    components, variance_ratio = _rt_cross_section_components(
        "bro", atmo, _BeerLambertEngine(path_length), [205.0, 235.0]
    )

    expected = (
        np.sum(vmr)
        * path_length
        * np.vstack(
            [
                _bro_like_cross_section(wavelengths, 205.0),
                _bro_like_cross_section(wavelengths, 235.0),
            ]
        )
    )
    np.testing.assert_allclose(components, expected, rtol=1e-10)
    np.testing.assert_array_equal(variance_ratio, [0.5, 0.5])
    np.testing.assert_array_equal(atmo["bro"].vmr, vmr)
    np.testing.assert_array_equal(atmo.temperature_k, original_temperature)


def test_rt_components_restore_state_when_engine_fails():
    vmr = np.array([1e-12, 2e-12])
    original_temperature = np.array([250.0, 230.0])
    atmo = _FakeAtmosphere(
        np.linspace(330.0, 340.0, 10),
        original_temperature,
        {"bro": _FakeConstituent(vmr=vmr)},
    )

    with pytest.raises(RuntimeError, match="engine failure"):
        _rt_cross_section_components(
            "bro", atmo, _BeerLambertEngine(1.0, fail_on_call=2), [220.0]
        )
    with pytest.raises(ValueError, match="At least one temperature"):
        _rt_cross_section_components("bro", atmo, _BeerLambertEngine(1.0), [])

    np.testing.assert_array_equal(atmo["bro"].vmr, vmr)
    np.testing.assert_array_equal(atmo.temperature_k, original_temperature)


# ---------------------------------------------------------------------------
# Tilt / cos(SZA) helpers (no radiative transfer needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def bare_fitter():
    fitter = object.__new__(DOASFitter)
    fitter._poly_order = 2
    fitter._cos_sza_input = None
    return fitter


def test_prepare_cos_sza_broadcasts_and_validates(bare_fitter):
    radiance = xr.DataArray(np.ones((3, 5)), dims=("los", "wavelength"))

    np.testing.assert_array_equal(bare_fitter._prepare_cos_sza(radiance), np.ones(3))

    bare_fitter._cos_sza_input = 0.5
    np.testing.assert_array_equal(
        bare_fitter._prepare_cos_sza(radiance), np.full(3, 0.5)
    )

    bare_fitter._cos_sza_input = xr.DataArray([0.2, 0.4, 0.6], dims=("los",))
    np.testing.assert_array_equal(
        bare_fitter._prepare_cos_sza(radiance), [0.2, 0.4, 0.6]
    )

    bare_fitter._cos_sza_input = np.array([np.nan, np.nan])
    np.testing.assert_array_equal(bare_fitter._prepare_cos_sza(radiance), np.ones(3))

    bare_fitter._cos_sza_input = [0.1, 0.2]
    with pytest.raises(
        ValueError, match=r"match the number of non-wavelength samples \(3\); got 2"
    ):
        bare_fitter._prepare_cos_sza(radiance)


def test_tilt_spectrum_is_detrended_airmass_scaled_log_ratio(bare_fitter):
    wavelengths = np.linspace(330.0, 340.0, 41)
    x = (wavelengths - 335.0) / 5.0
    rng = np.random.default_rng(23)
    structure = 1e-3 * rng.normal(size=wavelengths.size)
    log_modelled = np.array(
        [
            0.5 - 0.2 * x + 0.1 * x**2 + 3.0 * structure,
            0.1 + 0.3 * x - 0.2 * x**2 + 2.0 * structure,
            -0.4 + 0.1 * x + 0.05 * x**2 + structure,
        ]
    )
    cos_sza = np.array([0.8, 0.5, 0.25])

    polynomial, spectrum = bare_fitter._calculate_tilt_spectrum(
        np.exp(log_modelled), wavelengths, cos_sza
    )

    # The second-to-last line of sight is the reference.
    log_ratio = (log_modelled - log_modelled[1]) * (0.5 / cos_sza)[:, np.newaxis]
    np.testing.assert_allclose(polynomial + spectrum, log_ratio, atol=1e-12)
    np.testing.assert_allclose(spectrum[1], 0.0, atol=1e-12)
    # The spectrum is orthogonal to polynomials up to poly_order.
    vander = np.vander(x, 3)
    np.testing.assert_allclose(vander.T @ spectrum.T, 0.0, atol=1e-12)
    # Only the non-polynomial structure survives, scaled by the airmass ratio.
    detrended = structure - np.polyval(np.polyfit(x, structure, 2), x)
    np.testing.assert_allclose(spectrum[0], (0.5 / 0.8) * detrended, atol=1e-12)
    np.testing.assert_allclose(spectrum[2], -(0.5 / 0.25) * detrended, atol=1e-12)


def test_tilt_spectrum_handles_mismatched_cos_sza_and_invalid_input(bare_fitter):
    wavelengths = np.linspace(330.0, 340.0, 5)
    modelled = np.exp(np.outer([1.0, 2.0, 3.0], np.linspace(0.0, 1.0, 5)) ** 2)
    modelled[0, :4] = -1.0  # Only a single valid sample remains on this line.

    polynomial, spectrum = bare_fitter._calculate_tilt_spectrum(
        modelled, wavelengths, np.array([0.3, 0.9])
    )

    # Mismatched cos(SZA) falls back to the median, i.e. no airmass scaling.
    log_modelled = np.log(modelled[1:])
    np.testing.assert_allclose(
        polynomial[2] + spectrum[2], log_modelled[1] - log_modelled[0], atol=1e-12
    )
    assert np.all(np.isnan(spectrum[0]))

    empty_polynomial, empty_spectrum = bare_fitter._calculate_tilt_spectrum(
        np.ones((3, 1)), np.array([330.0])
    )
    assert np.all(np.isnan(empty_polynomial))
    assert np.all(np.isnan(empty_spectrum))


def test_tilt_pca_ignores_sparse_columns_and_fills_gaps(bare_fitter):
    rng = np.random.default_rng(29)
    pattern = rng.normal(size=12)
    spectrum = np.outer([1.0, -2.0, 0.5, 3.0], pattern)
    spectrum[:, 4] = np.nan
    spectrum[2, 7] = np.nan

    basis, variance_ratio = bare_fitter._tilt_pca_from_spectrum(spectrum, 2)

    assert basis.shape == (2, 12)
    np.testing.assert_array_equal(basis[:, 4], 0.0)
    np.testing.assert_allclose(np.linalg.norm(basis, axis=1), 1.0)
    assert variance_ratio[0] > 0.99

    assert bare_fitter._tilt_pca_from_spectrum(spectrum, 0)[0] is None
    sparse_spectrum = np.full((4, 3), np.nan)
    sparse_spectrum[:, 0] = 1.0
    assert bare_fitter._tilt_pca_from_spectrum(sparse_spectrum, 1)[0] is None


# ---------------------------------------------------------------------------
# End-to-end DOASFitter behaviour on synthetic scenes
# ---------------------------------------------------------------------------


def test_fit_recovers_wavelength_calibration_and_slant_columns():
    log_radiance, convolved_xs = _bro_scene()
    radiance = _radiance(log_radiance)

    result = _make_fitter(radiance).fit(radiance)

    np.testing.assert_allclose(
        [float(result[name]) for name in TRUE_CALIBRATION],
        list(TRUE_CALIBRATION.values()),
        atol=1e-6,
    )
    assert float(result["cost"]) < 1e-15
    assert list(result.basis.values) == [
        "bro_pca_0",
        "irrad",
        "poly_0",
        "poly_1",
        "poly_2",
        "poly_3",
    ]

    # The absorber column is standardised, so the slant column is the
    # coefficient divided by the spread of the convolved cross section.
    slant_columns = -result["bro_coefficient"].to_numpy() / np.std(convolved_xs)
    np.testing.assert_allclose(slant_columns, TRUE_SCD, rtol=1e-6)

    _, transformed, fwhm, _ = _instrument_model(
        np.zeros(_calc_wavelength().size), **TRUE_CALIBRATION
    )
    np.testing.assert_allclose(result["fit_wavelength"], transformed, atol=1e-6)
    np.testing.assert_allclose(result["fitted_fwhm"], fwhm, atol=1e-6)
    np.testing.assert_allclose(result["convolved_xs_bro"][0], convolved_xs, rtol=1e-6)
    assert np.nanmax(np.abs(result["residual"])) < 1e-8
    np.testing.assert_allclose(result["fitted_log_radiance"], log_radiance, atol=1e-8)


def test_fit_reports_bro_contribution_and_placeholder_oclo():
    log_radiance, convolved_xs = _bro_scene()
    radiance = _radiance(log_radiance)

    result = _make_fitter(radiance).fit(radiance)

    expected_contribution = -TRUE_SCD[:, np.newaxis] * (
        convolved_xs - np.mean(convolved_xs)
    )
    np.testing.assert_allclose(
        result["bro_contribution"], expected_contribution, atol=1e-8
    )
    np.testing.assert_allclose(
        result["residual_with_bro_added"],
        result["residual"] + result["bro_contribution"],
        atol=1e-14,
    )
    np.testing.assert_array_equal(result["oclo_coefficient"], 0.0)
    assert "bro_coefficient_wf" not in result
    assert "manual_predictor" not in result.coords


def test_fit_single_spectrum_matches_row_of_batch():
    log_radiance, _ = _bro_scene()
    radiance = _radiance(log_radiance)
    fitter = _make_fitter(radiance)

    batch = fitter.fit(radiance)
    single = fitter.fit(radiance.isel(tangent_altitude=1, drop=True))

    assert single["coefficients"].dims == ("sample", "basis")
    np.testing.assert_array_equal(single["sample"], [0])
    np.testing.assert_allclose(
        single["coefficients"][0], batch["coefficients"][1], rtol=1e-10
    )
    # Modelled-radiance diagnostics are only attached when the LOS count matches.
    assert "convolved_modelled_radiance" in batch
    assert "convolved_modelled_radiance" not in single


def test_fit_uses_generic_sample_dimension_names():
    log_radiance, _ = _bro_scene()
    fitter = _make_fitter(_radiance(log_radiance))
    radiance = _radiance(log_radiance).rename(tangent_altitude="time").drop_vars("time")

    result = fitter.fit(radiance)

    np.testing.assert_array_equal(result["time"], [0, 1, 2])
    assert result["bro_coefficient"].dims == ("time",)


def test_fit_accepts_sample_dimension_named_sample():
    log_radiance, _ = _bro_scene()
    fitter = _make_fitter(_radiance(log_radiance))
    radiance = _radiance(log_radiance, dim="sample", coord=np.arange(3))

    result = fitter.fit(radiance)

    assert result.sizes["sample"] == 3


def test_bro_coefficient_sigma_matches_monte_carlo_scatter():
    log_radiance, convolved_xs = _bro_scene()
    fitter = _make_fitter(_radiance(log_radiance))
    rng = np.random.default_rng(2024)
    num_spectra = 400
    noisy = np.repeat(log_radiance[1:2], num_spectra, axis=0) + rng.normal(
        scale=2e-3, size=(num_spectra, WAVELENGTH_NM.size)
    )

    result = fitter.fit(_radiance(noisy, dim="spectrum", coord=np.arange(num_spectra)))

    coefficients = result["bro_coefficient"].to_numpy()
    sigma = result["bro_coefficient_sigma"].to_numpy()
    truth = -TRUE_SCD[1] * np.std(convolved_xs)
    standard_error = np.std(coefficients, ddof=1) / np.sqrt(num_spectra)
    assert abs(np.mean(coefficients) - truth) < 4.0 * standard_error
    assert 0.85 < np.std(coefficients, ddof=1) / np.mean(sigma) < 1.15


def test_coefficient_weighting_functions_follow_linearised_fit():
    calc = _calc_wavelength()
    convolved_xs, convolved_irradiance, normalized = _convolved_references(
        np.vstack([_bro_like_cross_section(calc), _oclo_like_cross_section(calc)])
    )
    oclo_scd = 0.5 * TRUE_SCD
    log_radiance = (
        -TRUE_SCD[:, np.newaxis] * convolved_xs[0]
        - oclo_scd[:, np.newaxis] * convolved_xs[1]
        + _closure_terms(3, convolved_irradiance, normalized)
    )
    intensity = np.exp(log_radiance)
    # dI/dx for: BrO SCD, OClO SCD and a broadband (polynomial) term.
    wf = np.stack(
        [
            -intensity * convolved_xs[0],
            -intensity * convolved_xs[1],
            intensity * normalized**2,
        ],
        axis=-1,
    )
    measurement = xr.Dataset(
        {
            "radiance": _radiance(log_radiance),
            "wf": (("tangent_altitude", "wavelength", "x"), wf),
        },
        coords={"x": ["bro_scd", "oclo_scd", "smooth"]},
    )
    fitter = _make_fitter(
        measurement,
        optical={
            "bro": _SyntheticCrossSection(_bro_like_cross_section),
            "oclo": _SyntheticCrossSection(_oclo_like_cross_section),
        },
        absorber_temperatures={"bro": 220.0, "oclo": 220.0},
    )

    result = fitter.fit(measurement)

    bro_std = np.std(convolved_xs[0])
    oclo_std = np.std(convolved_xs[1])
    np.testing.assert_allclose(
        result["bro_coefficient"], -TRUE_SCD * bro_std, rtol=1e-6
    )
    np.testing.assert_allclose(
        result["oclo_coefficient"], -oclo_scd * oclo_std, rtol=1e-6
    )
    np.testing.assert_array_equal(result["x"], ["bro_scd", "oclo_scd", "smooth"])
    # Each absorber coefficient responds only to its own slant column; the
    # broadband perturbation is absorbed by the closure polynomial.
    bro_wf = result["bro_coefficient_wf"].to_numpy()
    oclo_wf = result["oclo_coefficient_wf"].to_numpy()
    np.testing.assert_allclose(bro_wf[:, 0], -bro_std, rtol=1e-6)
    np.testing.assert_allclose(bro_wf[:, 1], 0.0, atol=1e-6 * bro_std)
    np.testing.assert_allclose(bro_wf[:, 2], 0.0, atol=1e-9)
    np.testing.assert_allclose(oclo_wf[:, 0], 0.0, atol=1e-6 * oclo_std)
    np.testing.assert_allclose(oclo_wf[:, 1], -oclo_std, rtol=1e-6)
    np.testing.assert_allclose(oclo_wf[:, 2], 0.0, atol=1e-9)


def test_fit_masks_invalid_radiances_and_empty_samples():
    log_radiance, convolved_xs = _bro_scene()
    fitter = _make_fitter(_radiance(log_radiance))
    values = np.exp(log_radiance)
    values[0, [3, 50, 77]] = np.nan
    values[1, [10, 11]] = -1.0
    values[1, 60] = 0.0
    values[2, :] = np.inf
    radiance = xr.DataArray(
        values,
        dims=("tangent_altitude", "wavelength"),
        coords={"tangent_altitude": TANGENT_ALTITUDES_M, "wavelength": WAVELENGTH_NM},
    )

    result = fitter.fit(radiance)

    invalid = ~(np.isfinite(values) & (values > 0))
    np.testing.assert_array_equal(np.isnan(result["log_radiance"]), invalid)
    np.testing.assert_array_equal(np.isnan(result["residual"]), invalid)
    np.testing.assert_allclose(
        -result["bro_coefficient"][:2] / np.std(convolved_xs), TRUE_SCD[:2], rtol=1e-6
    )
    np.testing.assert_array_equal(result["coefficients"][2], 0.0)
    assert np.all(np.isnan(result["fitted_log_radiance"][2]))


def test_fit_rejects_unusable_measurements():
    log_radiance, _ = _bro_scene()
    radiance = _radiance(log_radiance)
    fitter = _make_fitter(radiance)

    with pytest.raises(ValueError, match="No finite positive radiances"):
        fitter.fit(radiance * np.nan)
    with pytest.raises(ValueError, match="wavelengths to match the initialization"):
        fitter.fit(radiance.isel(wavelength=slice(1, None)))
    with pytest.raises(ValueError, match="Not enough tangent altitudes"):
        _make_fitter(radiance.isel(tangent_altitude=[0]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"absorber_nonlinear_orders": {"no2": [2]}}, "Unknown absorber 'no2'"),
        ({"absorber_nonlinear_orders": {"bro": [1]}}, "must be >= 2"),
        (
            {"absorber_basis_method": "lookup"},
            "Unsupported absorber_basis_method 'lookup'",
        ),
        ({"absorber_temperatures": {"bro": "surface"}}, "string mode 'tangent' only"),
        ({"cos_sza": [0.5, 0.6]}, "cos_sza must be scalar or match"),
        ({"radiance_filter": [1.0, -1.0]}, "must not sum to zero"),
        (
            {"manual_predictors": {"bro": np.ones(3)}},
            "'bro' conflicts with an existing absorber name",
        ),
        (
            {"manual_predictors": {"ring": np.ones(3)}},
            "'ring' does not provide wavelengths",
        ),
        (
            {"manual_predictors": {"ring": np.ones(3)}, "initial_params": np.zeros(3)},
            "initial_params must contain either 4 base values or 5 values",
        ),
        (
            {
                "manual_predictors": {"ring": np.ones(3)},
                "bounds": (np.zeros(3), np.ones(3)),
            },
            "bounds must contain either 4 base values or 5 values",
        ),
    ],
)
def test_fitter_rejects_invalid_configuration(kwargs, match):
    log_radiance, _ = _bro_scene()
    with pytest.raises(ValueError, match=match):
        _make_fitter(_radiance(log_radiance), **kwargs)


def test_doas_fit_requires_ancillary_or_fitter_and_reuses_fitter():
    log_radiance, _ = _bro_scene()
    radiance = _radiance(log_radiance)
    fitter = _make_fitter(radiance)

    with pytest.raises(ValueError, match="requires either anc or fitter"):
        doas_fit(radiance)

    reused = doas_fit(radiance, fitter=fitter)
    direct = doas_fit(
        radiance,
        _SyntheticAncillary(),
        optical={"bro": _SyntheticCrossSection(_bro_like_cross_section)},
        absorber_temperatures={"bro": 220.0},
        calc_margin=CALC_MARGIN_NM,
        calc_spacing=CALC_SPACING_NM,
    )

    xr.testing.assert_allclose(reused, fitter.fit(radiance))
    np.testing.assert_allclose(
        direct["coefficients"], reused["coefficients"], rtol=1e-6
    )


def test_tangent_temperature_mode_fits_sample_specific_cross_sections():
    calc = _calc_wavelength()
    tangent_temperatures = _temperature_profile(TANGENT_ALTITUDES_M)
    rows = []
    convolved_rows = []
    for idx, temperature in enumerate(tangent_temperatures):
        convolved_xs, convolved_irradiance, normalized = _convolved_references(
            _bro_like_cross_section(calc, temperature)
        )
        closure = _closure_terms(3, convolved_irradiance, normalized)[idx]
        rows.append(-TRUE_SCD[idx] * convolved_xs[0] + closure)
        convolved_rows.append(convolved_xs[0])
    radiance = _radiance(np.array(rows))
    convolved_rows = np.array(convolved_rows)

    fitter = _make_fitter(radiance, absorber_temperatures={"bro": "Tangent"})
    result = fitter.fit(radiance)
    fixed_temperature = _make_fitter(radiance).fit(radiance)

    assert result["convolved_xs_bro"].dims == (
        "tangent_altitude",
        "bro_component",
        "wavelength",
    )
    np.testing.assert_allclose(
        result["convolved_xs_bro"][:, 0], convolved_rows, rtol=1e-6
    )
    np.testing.assert_allclose(
        -result["bro_coefficient"] / np.std(convolved_rows, axis=1), TRUE_SCD, rtol=1e-6
    )
    assert _rms(result["residual"]) < 1e-9
    # A single 220 K cross section cannot reproduce the temperature structure.
    assert _rms(fixed_temperature["residual"]) > 1e3 * _rms(result["residual"])

    with pytest.raises(ValueError, match="requires 3 samples, got 1"):
        fitter.fit(radiance.isel(tangent_altitude=[0]))


def test_multiple_temperature_components_are_named_and_weighted():
    log_radiance, _ = _bro_scene()
    radiance = _radiance(log_radiance)

    result = _make_fitter(
        radiance, absorber_temperatures={"bro": [203.0, 223.0, 243.0]}
    ).fit(radiance)

    assert list(result.basis.values[:4]) == [
        "bro_pca_0",
        "bro_pca_1",
        "bro_pca_2",
        "irrad",
    ]
    np.testing.assert_allclose(result["xs_pca_variance_ratio_bro"], np.full(3, 1 / 3))
    calc = _calc_wavelength()
    np.testing.assert_allclose(
        result["xs_pca_bro"],
        np.vstack([_bro_like_cross_section(calc, t) for t in (203.0, 223.0, 243.0)]),
    )
    # The 220 K truth lies in the span of the temperature components.
    assert _rms(result["residual"]) < 1e-9


def test_nonlinear_absorber_orders_capture_saturation():
    calc = _calc_wavelength()
    convolved_xs, convolved_irradiance, normalized = _convolved_references(
        _bro_like_cross_section(calc)
    )
    optical_depth = TRUE_SCD[:, np.newaxis] * convolved_xs[0]
    log_radiance = (
        -optical_depth
        + 0.05 * optical_depth**2
        + _closure_terms(3, convolved_irradiance, normalized)
    )
    radiance = _radiance(log_radiance)

    linear = _make_fitter(radiance).fit(radiance)
    result = _make_fitter(radiance, absorber_nonlinear_orders={"bro": [3, 2, 2]}).fit(
        radiance
    )

    assert list(result.basis.values[:4]) == [
        "bro_pca_0",
        "bro_nl_pca_0_pow_2",
        "bro_nl_pca_0_pow_3",
        "irrad",
    ]
    assert result.sizes["basis"] == linear.sizes["basis"] + 2
    np.testing.assert_allclose(
        [float(result[name]) for name in TRUE_CALIBRATION],
        list(TRUE_CALIBRATION.values()),
        atol=1e-6,
    )
    assert _rms(result["residual"]) < 1e-9
    assert _rms(linear["residual"]) > 1e-5


def test_residual_pca_absorbs_common_unmodelled_structure():
    log_radiance, _ = _bro_scene()
    # Etalon-like high-frequency structure shared by all spectra with a
    # sample-dependent amplitude; it is not representable by the DOAS basis.
    etalon = np.sin(2.0 * np.pi * np.arange(WAVELENGTH_NM.size) / 3.3)
    unmodelled = 5e-3 * np.outer([1.0, 2.0, 3.5], etalon)
    radiance = _radiance(log_radiance + unmodelled)

    baseline = _make_fitter(radiance).fit(radiance)
    result = _make_fitter(radiance, residual_pca_components=1).fit(radiance)

    assert result.basis.values[-1] == "residual_pca_0"
    assert result["residual_pca_basis"].dims == ("residual_component", "wavelength")
    np.testing.assert_allclose(np.linalg.norm(result["residual_pca_basis"][0]), 1.0)
    assert 0.99 < float(result["residual_pca_variance_ratio"][0]) <= 1.0
    assert "residual_pca_basis" not in baseline
    assert _rms(result["residual"]) < 2e-2 * _rms(baseline["residual"])
    np.testing.assert_allclose(
        float(result["shift"]), TRUE_CALIBRATION["shift"], atol=5e-4
    )


def test_manual_predictor_spectral_offset_is_recovered():
    calc = _calc_wavelength()
    true_offset = 0.04
    ring_wavelength = np.arange(326.0, 344.0, 0.005)
    ring = _bands(
        ring_wavelength,
        [330.9, 332.7, 334.9, 336.8, 338.6],
        [0.15, 0.2, 0.12, 0.18, 0.15],
        [1.0, 0.6, 0.9, 0.7, 0.8],
    )
    convolved, convolved_irradiance, normalized = _convolved_references(
        np.vstack(
            [
                _bro_like_cross_section(calc),
                np.interp(calc - true_offset, ring_wavelength, ring),
            ]
        )
    )
    log_radiance = (
        -TRUE_SCD[:, np.newaxis] * convolved[0]
        + np.array([0.02, 0.03, 0.05])[:, np.newaxis] * convolved[1]
        + _closure_terms(3, convolved_irradiance, normalized)
    )
    radiance = _radiance(log_radiance)

    result = _make_fitter(
        radiance, manual_predictors={"ring": (ring_wavelength, ring)}
    ).fit(radiance)
    bounded = _make_fitter(
        radiance,
        manual_predictors={"ring": {"wavelength": ring_wavelength, "values": ring}},
        initial_params=np.array([0.0, 0.0, 0.2, 0.0, 0.0]),
        bounds=(
            np.array([-0.5, -0.05, 1e-4, -1.0, -0.01]),
            np.array([0.5, 0.05, 3.0, 1.0, 0.01]),
        ),
    ).fit(radiance)

    np.testing.assert_array_equal(result["manual_predictor"], ["ring"])
    assert "ring_pca_0" in result.basis.values
    np.testing.assert_allclose(
        result["manual_predictor_spectral_offset_nm"], [true_offset], atol=2e-3
    )
    np.testing.assert_allclose(
        float(result["shift"]), TRUE_CALIBRATION["shift"], atol=1e-3
    )
    # Explicit bounds on the offset parameter are honoured.
    assert abs(float(bounded["manual_predictor_spectral_offset_nm"][0])) <= 0.01 + 1e-12


def test_tilt_diagnostics_and_tilt_pca_basis():
    log_radiance, convolved_xs = _bro_scene()
    radiance = _radiance(log_radiance)
    cos_sza = np.array([0.9, 0.6, 0.3])

    result = _make_fitter(radiance, cos_sza=cos_sza, tilt_pca_components=1).fit(
        radiance
    )

    modelled = result["convolved_modelled_radiance"].to_numpy()
    tilt_total = (result["tilt_polynomial"] + result["tilt_spectrum"]).to_numpy()
    assert modelled.shape == log_radiance.shape
    # The two lower lines of sight (including the reference, index -2) see
    # the atmosphere; non-positive modelled radiance yields NaN tilt.
    valid = modelled > 0
    assert valid[:2].all()
    log_ratio = (np.log(modelled[:2]) - np.log(modelled[1])) * (
        0.6 / cos_sza[:2, np.newaxis]
    )
    np.testing.assert_allclose(tilt_total[:2], log_ratio, atol=1e-12)
    np.testing.assert_allclose(result["tilt_spectrum"][1], 0.0, atol=1e-12)
    assert np.all(np.isnan(result["tilt_spectrum"].to_numpy()[~valid]))

    assert result.basis.values[-1] == "tilt_pca_0"
    assert result["tilt_pca_basis"].dims == ("tilt_component", "wavelength")
    np.testing.assert_allclose(np.linalg.norm(result["tilt_pca_basis"][0]), 1.0)
    # Adding the tilt basis after calibration keeps the noise-free fit exact.
    np.testing.assert_allclose(
        -result["bro_coefficient"] / np.std(convolved_xs), TRUE_SCD, rtol=1e-6
    )


@pytest.mark.parametrize("keyword", ["radiance_filter", "filter"])
def test_radiance_filter_is_applied_before_taking_the_log(keyword):
    log_radiance, _ = _bro_scene()
    radiance = _radiance(log_radiance)
    weights = np.array([1.0, 2.0, 1.0])

    result = _make_fitter(radiance, **{keyword: weights}).fit(radiance)

    smoothed = np.array(
        [np.convolve(row, weights / 4.0, mode="same") for row in np.exp(log_radiance)]
    )
    np.testing.assert_allclose(
        result["log_radiance"][:, 1:-1], np.log(smoothed[:, 1:-1]), rtol=1e-12
    )


def test_extract_constituent_profile_terminates_for_sasktran2_atmosphere():
    altitudes = np.array([0.0, 10_000.0])
    atmo = sk.Atmosphere(
        sk.Geometry1D(0.45, 0.0, 6371000, altitudes),
        sk.Config(),
        np.array([330.0, 331.0]),
        calculate_derivatives=False,
    )
    atmo["bro"] = sk.constituent.VMRAltitudeAbsorber(
        _SyntheticCrossSection(_bro_like_cross_section),
        altitudes,
        np.array([1e-12, 2e-12]),
    )

    kind, profile = _extract_constituent_profile(atmo, "bro")

    assert kind == "vmr"
    np.testing.assert_allclose(profile, [1e-12, 2e-12])


@pytest.mark.parametrize("method", ["rt", {"bro": "Radiative-Transfer"}])
def test_radiative_transfer_basis_is_cross_section_times_smooth_airmass(method):
    log_radiance, _ = _bro_scene()
    radiance = _radiance(log_radiance)
    anc = _SyntheticAncillary(extra_absorbers={"bro": (_bro_like_cross_section, 1e-12)})

    rt_basis = (
        _make_fitter(radiance, anc=anc, absorber_basis_method=method)
        .fit(radiance)["xs_pca_bro"]
        .to_numpy()[0]
    )

    calc = _calc_wavelength()
    cross_section = _bro_like_cross_section(calc)
    assert rt_basis.shape == calc.shape
    assert np.all(rt_basis > 0)
    # For an optically thin absorber, log(I_without / I_with) is the slant
    # optical depth: the cross section times a slowly varying air-mass factor.
    airmass = rt_basis / cross_section
    assert np.ptp(airmass) / np.mean(airmass) < 0.2
    x = (calc - calc.mean()) / np.ptp(calc)
    smooth_scaled = np.column_stack([cross_section * x**order for order in range(3)])
    coefficients, _, _, _ = np.linalg.lstsq(smooth_scaled, rt_basis, rcond=None)
    unexplained = rt_basis - smooth_scaled @ coefficients
    assert np.sum(unexplained**2) < 1e-5 * np.sum((rt_basis - rt_basis.mean()) ** 2)


def test_radiative_transfer_basis_requires_absorber_in_atmosphere():
    log_radiance, _ = _bro_scene()
    with pytest.raises(ValueError, match="'bro' is not present in atmosphere"):
        _make_fitter(_radiance(log_radiance), absorber_basis_method="rt")


def test_derivative_predictor_follows_absorber_cross_section():
    log_radiance, _ = _bro_scene()
    radiance = _radiance(log_radiance)

    result = _make_fitter(
        radiance,
        optical={"o3": _SyntheticCrossSection(_ozone_like_cross_section)},
        absorber_cross_section_predictors={"o3": False},
        absorber_derivative_predictors={"o3": True},
    ).fit(radiance)

    assert list(result.basis.values[:2]) == ["o3_dlogI_0", "irrad"]
    np.testing.assert_array_equal(result["xs_pca_variance_ratio_o3"], [0.0])
    derivative = result["xs_pca_o3"].to_numpy()[0]
    # d(log I)/d(vmr) is a negative multiple of the absorption cross section.
    assert np.all(derivative < 0)
    calc = _calc_wavelength()
    assert np.corrcoef(derivative, _ozone_like_cross_section(calc))[0, 1] < -0.99
