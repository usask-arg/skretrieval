from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy import sparse

import skretrieval.retrieval.measvec as mv
from skretrieval.core.radianceformat import LinearizedRadianceGridded, RadianceGridded
from skretrieval.retrieval.erroranalysis import (
    estimate_error,
    estimate_error_from_operator,
)


def _l1_pair():
    wavelengths = np.array([300.0, 310.0, 320.0, 330.0, 340.0])
    tangent_altitude = np.array([10_000.0, 20_000.0, 30_000.0])
    y = np.arange(1.0, 16.0).reshape(5, 3)
    K = np.arange(1.0, 76.0).reshape(5, 3, 5) / 100
    noise = np.ones_like(y) * 0.2
    ds = xr.Dataset(
        {
            "radiance": (["wavelength", "los"], y),
            "wf": (["wavelength", "los", "x"], K),
            "radiance_noise": (["wavelength", "los"], noise),
        },
        coords={
            "wavelength": wavelengths,
            "los": np.arange(3),
            "x": np.arange(5),
            "tangent_altitude": (["los"], tangent_altitude),
        },
    ).set_xindex("tangent_altitude")

    flat_K = K.reshape((-1, K.shape[-1]))

    def matvec(x: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            (flat_K @ x).reshape(y.shape),
            dims=["wavelength", "los"],
            coords=ds["radiance"].coords,
        )

    def rmatvec(cotangent: xr.DataArray) -> np.ndarray:
        return flat_K.T @ np.asarray(cotangent).reshape(-1)

    return (
        {"measurement": RadianceGridded(ds.copy(deep=True))},
        {
            "measurement": LinearizedRadianceGridded(
                ds.drop_vars("wf").copy(deep=True),
                matvec,
                rmatvec,
                flat_K.shape[1],
            )
        },
    )


def _assert_operator_matches(materialized, matrix_free):
    np.testing.assert_allclose(matrix_free.y, materialized.y)
    materialized_Sy = (
        materialized.Sy.toarray()
        if sparse.issparse(materialized.Sy)
        else materialized.Sy
    )
    matrix_free_Sy = (
        matrix_free.Sy.toarray() if sparse.issparse(matrix_free.Sy) else matrix_free.Sy
    )
    np.testing.assert_allclose(matrix_free_Sy, materialized_Sy)

    direction = np.linspace(-0.3, 0.7, materialized.K.shape[1])
    cotangent = np.linspace(0.2, 1.1, len(materialized.y))
    np.testing.assert_allclose(matrix_free.jvp(direction), materialized.K @ direction)
    np.testing.assert_allclose(matrix_free.vjp(cotangent), materialized.K.T @ cotangent)
    np.testing.assert_allclose(
        cotangent @ matrix_free.jvp(direction),
        direction @ matrix_free.vjp(cotangent),
    )


def test_matrix_free_measurement_primitives_match_materialized_jacobian():
    materialized_l1, matrix_free_l1 = _l1_pair()

    materialized = mv.select(
        materialized_l1,
        wavelength=slice(310.0, 330.0),
        tangent_altitude=slice(10_000.0, 20_000.0),
    )
    matrix_free = mv.select(
        matrix_free_l1,
        wavelength=slice(310.0, 330.0),
        tangent_altitude=slice(10_000.0, 20_000.0),
    )
    _assert_operator_matches(materialized, matrix_free)

    for transform in (mv.log, mv.mean, lambda m: mv.multiply(m, -2.5)):
        _assert_operator_matches(transform(materialized), transform(matrix_free))

    left_materialized = mv.select(materialized_l1, wavelength=slice(300.0, 310.0))
    right_materialized = mv.select(materialized_l1, wavelength=slice(320.0, 330.0))
    left_matrix_free = mv.select(matrix_free_l1, wavelength=slice(300.0, 310.0))
    right_matrix_free = mv.select(matrix_free_l1, wavelength=slice(320.0, 330.0))

    _assert_operator_matches(
        mv.add(left_materialized, right_materialized),
        mv.add(left_matrix_free, right_matrix_free),
    )
    _assert_operator_matches(
        mv.subtract(left_materialized, right_materialized),
        mv.subtract(left_matrix_free, right_matrix_free),
    )
    _assert_operator_matches(
        mv.concat([left_materialized, right_materialized]),
        mv.concat([left_matrix_free, right_matrix_free]),
    )

    # Triplet normalization broadcasts a one-element mean over an altitude profile.
    _assert_operator_matches(
        mv.subtract(materialized, mv.mean(right_materialized)),
        mv.subtract(matrix_free, mv.mean(right_matrix_free)),
    )


def test_matrix_free_selector_compositions_match_materialized_jacobian():
    materialized_l1, matrix_free_l1 = _l1_pair()

    _assert_operator_matches(
        mv.select(mv.nearest_selector(materialized_l1, wavelength=314.0)),
        mv.select(mv.nearest_selector(matrix_free_l1, wavelength=314.0)),
    )
    _assert_operator_matches(
        mv.wavelength_mean(materialized_l1, wavelength=slice(300.0, 320.0)),
        mv.wavelength_mean(matrix_free_l1, wavelength=slice(300.0, 320.0)),
    )


def test_matrix_free_measurement_can_skip_modeled_covariance():
    _, matrix_free_l1 = _l1_pair()
    measurement = mv.select(
        mv.pre_process(matrix_free_l1, include_error=False),
        wavelength=slice(310.0, 330.0),
    )

    assert measurement.Sy is None
    assert "y_error" not in mv.post_process(measurement)


def test_triplet_reuses_each_wavelength_selection(monkeypatch):
    materialized_l1, _ = _l1_pair()
    materialized_l1 = mv.pre_process(materialized_l1)
    nearest_wavelength = mv._nearest_wavelength
    calls = 0

    def counting_selector(*args, **kwargs):
        nonlocal calls
        calls += 1
        return nearest_wavelength(*args, **kwargs)

    monkeypatch.setattr(mv, "_nearest_wavelength", counting_selector)
    triplet = mv.Triplet(
        wavelength=[300.0, 320.0, 340.0],
        weights=[0.5, -1.0, 0.5],
        altitude_range=[10_000.0, 20_000.0],
        normalization_range=[30_000.0, 30_000.0],
    )

    triplet.apply(materialized_l1)
    triplet.apply(materialized_l1)
    assert calls == 3


def test_triplet_skips_normalization_measurement_when_disabled(monkeypatch):
    materialized_l1, _ = _l1_pair()
    materialized_l1 = mv.pre_process(materialized_l1)
    measurement_from_selection = mv._measurement_from_selection
    calls = 0

    def counting_selection(*args, **kwargs):
        nonlocal calls
        calls += 1
        return measurement_from_selection(*args, **kwargs)

    monkeypatch.setattr(mv, "_measurement_from_selection", counting_selection)
    triplet = mv.Triplet(
        wavelength=[320.0],
        weights=[1.0],
        altitude_range=[10_000.0, 20_000.0],
        normalization_range=[30_000.0, 30_000.0],
        normalize=False,
    )

    triplet.apply(materialized_l1)
    assert calls == 1


def test_concat_fuses_triplet_cotangents_before_radiance_vjp():
    materialized_l1, matrix_free_l1 = _l1_pair()
    source = matrix_free_l1["measurement"]
    source_rmatvec = source._rmatvec
    calls = 0

    def counting_rmatvec(cotangent):
        nonlocal calls
        calls += 1
        return source_rmatvec(cotangent)

    source._rmatvec = counting_rmatvec
    materialized_l1 = mv.pre_process(materialized_l1)
    matrix_free_l1 = mv.pre_process(matrix_free_l1)
    triplets = [
        mv.Triplet(
            wavelength=[wavelength],
            weights=[1.0],
            altitude_range=[10_000.0, 20_000.0],
            normalization_range=[30_000.0, 30_000.0],
        )
        for wavelength in materialized_l1["measurement"].data["wavelength"].to_numpy()
    ]

    materialized = mv.concat([triplet.apply(materialized_l1) for triplet in triplets])
    matrix_free = mv.concat([triplet.apply(matrix_free_l1) for triplet in triplets])
    cotangent = np.linspace(0.2, 1.1, len(matrix_free.y))

    np.testing.assert_allclose(
        matrix_free.vjp(cotangent), materialized.K.T @ cotangent, atol=1e-15
    )
    assert calls == 1


def test_matrix_free_select_supports_auxiliary_xindex_without_dimension_coord():
    materialized_l1, matrix_free_l1 = _l1_pair()
    materialized_l1["measurement"].data = materialized_l1["measurement"].data.drop_vars(
        "los"
    )
    matrix_free_l1["measurement"].data = matrix_free_l1["measurement"].data.drop_vars(
        "los"
    )

    _assert_operator_matches(
        mv.select(
            materialized_l1,
            wavelength=310.0,
            tangent_altitude=slice(10_000.0, 20_000.0),
        ),
        mv.select(
            matrix_free_l1,
            wavelength=310.0,
            tangent_altitude=slice(10_000.0, 20_000.0),
        ),
    )


@pytest.mark.parametrize("correlated", [False, True])
def test_operator_error_analysis_matches_materialized_error_analysis(
    correlated: bool,
):
    rng = np.random.default_rng(1234)
    K = rng.normal(size=(7, 4))
    if correlated:
        covariance_factor = rng.normal(size=(K.shape[0], K.shape[0]))
        Sy = covariance_factor @ covariance_factor.T + np.eye(K.shape[0])
        inv_Sy = np.linalg.inv(Sy)
    else:
        Sy = sparse.diags(np.linspace(0.8, 1.4, K.shape[0]), format="csc")
        inv_Sy = sparse.diags(1 / Sy.diagonal(), format="csc")
    inv_Sa = np.eye(K.shape[1]) * 0.3

    class Operator:
        shape = K.shape

        def __init__(self):
            self.matvec_calls = 0
            self.rmatvec_calls = 0

        def matvec(self, x):
            self.matvec_calls += 1
            return K @ x

        def rmatvec(self, y):
            self.rmatvec_calls += 1
            return K.T @ y

    operator = Operator()
    materialized = estimate_error(K, Sy, inv_Sy, inv_Sa)
    matrix_free = estimate_error_from_operator(operator, inv_Sy, inv_Sa)

    for key in (
        "averaging_kernel",
        "error_covariance_from_noise",
        "solution_covariance",
    ):
        np.testing.assert_allclose(matrix_free[key], materialized[key])

    # Seven measurement-space VJPs are cheaper here than four JVP/VJP pairs.
    assert operator.matvec_calls == 0
    assert operator.rmatvec_calls == K.shape[0]
