from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import skretrieval.core.radianceformat as radianceformat
from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import DeltaFunction
from skretrieval.core.radianceformat import evaluate_vjp_contributions
from skretrieval.core.sasktranformat import (
    LinearizedSASKTRANRadiance,
    SASKTRANRadiance,
)
from skretrieval.core.sensor.spectrograph import (
    Spectrograph,
    SpectrographOnlySpectral,
)


def _radiance_pair(num_stokes: int):
    rng = np.random.default_rng(1234 + num_stokes)
    wavelengths = np.array([500.0, 510.0, 520.0])
    stokes = ["I", "Q", "U", "V"][:num_stokes]
    n_los = 2
    n_state = 4
    radiance = rng.normal(size=(len(wavelengths), n_los, num_stokes))
    jacobian = rng.normal(size=(n_state, len(wavelengths), n_los, num_stokes))
    look_vectors = np.array([[0.0, 0.0, -1.0], [0.01, 0.0, -0.99995]])

    dataset = xr.Dataset(
        {
            "radiance": (
                ["wavelength", "los", "stokes"],
                radiance,
            ),
            "wf_state": (
                ["state", "wavelength", "los", "stokes"],
                jacobian,
            ),
            "look_vectors": (["los", "xyz"], look_vectors),
        },
        coords={
            "wavelength": wavelengths,
            "los": np.arange(n_los),
            "stokes": stokes,
            "state": np.arange(n_state),
            "xyz": ["x", "y", "z"],
        },
    )

    def jvp(direction: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            np.einsum("s,swlk->wlk", direction, jacobian),
            dims=dataset["radiance"].dims,
            coords=dataset["radiance"].coords,
        )

    def vjp(cotangent: xr.DataArray) -> np.ndarray:
        return np.einsum("swlk,wlk->s", jacobian, np.asarray(cotangent))

    materialized = SASKTRANRadiance.from_sasktran2(dataset.copy(deep=True))
    linearized = LinearizedSASKTRANRadiance.from_sasktran2_linearization(
        dataset["radiance"], jvp, vjp, n_state
    )
    linearized.data["look_vectors"] = materialized.data["look_vectors"]
    return materialized, linearized


def _assert_products_match(materialized, linearized):
    np.testing.assert_allclose(
        linearized.data["radiance"], materialized.data["radiance"]
    )
    jacobian = (
        materialized.data["wf_state"].to_numpy().reshape((-1, linearized.n_state))
    )
    direction = np.linspace(-0.4, 0.7, linearized.n_state)
    cotangent = np.linspace(0.2, 1.1, jacobian.shape[0])

    product = linearized.jvp(direction).to_numpy().reshape(-1)
    adjoint = linearized.vjp(
        xr.DataArray(
            cotangent.reshape(linearized.data["radiance"].shape),
            dims=linearized.data["radiance"].dims,
            coords=linearized.data["radiance"].coords,
        )
    )
    np.testing.assert_allclose(product, jacobian @ direction)
    np.testing.assert_allclose(adjoint, jacobian.T @ cotangent)
    np.testing.assert_allclose(cotangent @ product, direction @ adjoint)


@pytest.mark.parametrize("num_stokes", [1, 3])
def test_spectral_spectrograph_products_match_materialized(num_stokes: int):
    materialized, linearized = _radiance_pair(num_stokes)
    if num_stokes == 1:
        sensitivities = None
    else:
        sensitivities = {
            "vertical": np.array([0.5, 0.5, 0.0, 0.0]),
            "horizontal": np.array([0.5, -0.5, 0.0, 0.0]),
        }

    sensor = SpectrographOnlySpectral(
        np.array([500.0, 515.0]),
        [DeltaFunction(), DeltaFunction()],
        stokes_sensitivity=sensitivities,
    )
    materialized_result = sensor.model_radiance(materialized, None)
    linearized_result = sensor.model_radiance(linearized, None)

    assert materialized_result.keys() == linearized_result.keys()
    for key in materialized_result:
        _assert_products_match(materialized_result[key], linearized_result[key])


def test_spectral_spectrograph_fuses_shared_radiance_vjp():
    materialized, linearized = _radiance_pair(3)
    sensor = SpectrographOnlySpectral(
        np.array([500.0, 515.0]),
        [DeltaFunction(), DeltaFunction()],
        stokes_sensitivity={
            "vertical": np.array([0.5, 0.5, 0.0, 0.0]),
            "horizontal": np.array([0.5, -0.5, 0.0, 0.0]),
        },
    )
    materialized_result = sensor.model_radiance(materialized, None)
    linearized_result = sensor.model_radiance(linearized, None)
    source_rmatvec = linearized._rmatvec
    calls = 0

    def counting_rmatvec(cotangent):
        nonlocal calls
        calls += 1
        return source_rmatvec(cotangent)

    linearized._rmatvec = counting_rmatvec
    contributions = []
    expected = np.zeros(linearized.n_state)
    for scale, key in enumerate(linearized_result, start=1):
        cotangent = xr.ones_like(linearized_result[key].data["radiance"]) * scale
        contributions.extend(linearized_result[key].vjp_contributions(cotangent))
        jacobian = (
            materialized_result[key]
            .data["wf_state"]
            .to_numpy()
            .reshape((-1, linearized.n_state))
        )
        expected += jacobian.T @ cotangent.to_numpy().reshape(-1)

    np.testing.assert_allclose(evaluate_vjp_contributions(contributions), expected)
    assert calls == 1


def test_spectral_spectrograph_fuses_before_instrument_pullback():
    _, linearized = _radiance_pair(1)
    sensor = SpectrographOnlySpectral(
        np.array([500.0, 510.0, 520.0]),
        [DeltaFunction(), DeltaFunction(), DeltaFunction()],
    )
    result = sensor.model_radiance(linearized, None)["I"]
    instrument_pullback = result._pullback
    calls = 0

    def counting_pullback(cotangent):
        nonlocal calls
        calls += 1
        return instrument_pullback(cotangent)

    result._pullback = counting_pullback
    contributions = []
    for wavelength in result.data["wavelength"].to_numpy():
        selected = result.select(wavelength=wavelength)
        contributions.extend(
            selected.vjp_contributions(np.ones(selected.data["radiance"].shape))
        )

    evaluate_vjp_contributions(contributions)
    assert calls == 1


def test_spectral_spectrograph_reuses_selection_plans(monkeypatch):
    _, linearized = _radiance_pair(1)
    sensor = SpectrographOnlySpectral(
        np.array([500.0, 510.0]),
        [DeltaFunction(), DeltaFunction()],
    )
    compile_selection_plan = radianceformat._compile_selection_plan
    calls = 0

    def counting_compiler(*args, **kwargs):
        nonlocal calls
        calls += 1
        return compile_selection_plan(*args, **kwargs)

    monkeypatch.setattr(radianceformat, "_compile_selection_plan", counting_compiler)
    for _ in range(2):
        result = sensor.model_radiance(linearized, None)["I"]
        result.select(wavelength=500.0)

    assert calls == 1


def test_spatial_spectrograph_products_match_materialized():
    materialized, linearized = _radiance_pair(1)
    sensor = Spectrograph(
        np.array([500.0, 520.0]),
        [DeltaFunction(), DeltaFunction()],
        DeltaFunction(),
        DeltaFunction(),
    )
    wavelength_weights = np.array([[0.8, 0.2, 0.0], [0.0, 0.25, 0.75]])
    los_weights = np.array([[0.35], [0.65]])
    sensor._construct_interpolators = (
        lambda _orientation, _los_vectors, _spectral_grid: (
            wavelength_weights,
            los_weights,
        )
    )
    orientation = OpticalGeometry(
        observer=np.array([0.0, 0.0, 7_000_000.0]),
        look_vector=np.array([0.0, 0.0, -1.0]),
        local_up=np.array([0.0, 0.0, 1.0]),
        mjd=55_000.0,
    )

    materialized_result = sensor.model_radiance(materialized, orientation)
    linearized_result = sensor.model_radiance(linearized, orientation)
    _assert_products_match(materialized_result, linearized_result)


def test_spatial_spectrograph_rejects_polarized_radiance():
    materialized, _ = _radiance_pair(3)
    sensor = Spectrograph(
        np.array([500.0]),
        [DeltaFunction()],
        DeltaFunction(),
        DeltaFunction(),
    )
    sensor._construct_interpolators = (
        lambda _orientation, _los_vectors, _spectral_grid: (
            np.ones((1, 3)) / 3,
            np.ones((2, 1)) / 2,
        )
    )
    orientation = OpticalGeometry(
        observer=np.ones(3),
        look_vector=np.array([0.0, 0.0, -1.0]),
        local_up=np.array([0.0, 0.0, 1.0]),
        mjd=0.0,
    )

    with pytest.raises(ValueError, match="only supports scalar"):
        sensor.model_radiance(materialized, orientation)
