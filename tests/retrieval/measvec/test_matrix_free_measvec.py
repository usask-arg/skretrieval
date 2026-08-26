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
    estimate_fisher_diagonal_error_from_operator,
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


def _orbital_l1_pair():
    wavelengths = np.array([300.0, 320.0])
    tangent_altitude = np.tile([10_000.0, 20_000.0, 30_000.0], 2)
    image = np.repeat([4, 5], 3)
    y = np.arange(1.0, 13.0).reshape(2, 6)
    K = np.random.default_rng(42).normal(size=(2, 6, 5)) / 100
    noise = np.full_like(y, 0.2)
    ds = xr.Dataset(
        {
            "radiance": (("wavelength", "los"), y),
            "wf": (("wavelength", "los", "x"), K),
            "radiance_noise": (("wavelength", "los"), noise),
        },
        coords={
            "wavelength": wavelengths,
            "los": np.arange(6),
            "x": np.arange(5),
            "tangent_altitude": ("los", tangent_altitude),
            "image": ("los", image),
        },
    ).set_xindex("tangent_altitude")
    flat_K = K.reshape((-1, K.shape[-1]))

    def matvec(x: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            (flat_K @ x).reshape(y.shape),
            dims=("wavelength", "los"),
            coords=ds.radiance.coords,
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

    _assert_operator_matches(
        mv.multiply_elementwise(
            materialized,
            np.array([0.25, 2.0, -1.0, 0.5, 1.25, -0.75]),
        ),
        mv.multiply_elementwise(
            matrix_free,
            np.array([0.25, 2.0, -1.0, 0.5, 1.25, -0.75]),
        ),
    )


def test_matrix_free_selector_compositions_match_materialized_jacobian():
    materialized_l1, matrix_free_l1 = _l1_pair()

    _assert_operator_matches(
        mv.select(mv.nearest_selector(materialized_l1, wavelength=314.0)),
        mv.select(mv.nearest_selector(matrix_free_l1, wavelength=314.0)),
    )


def test_triplet_groups_repeated_orbital_altitudes_by_image():
    materialized_l1, matrix_free_l1 = _orbital_l1_pair()
    materialized_l1 = mv.pre_process(materialized_l1)
    matrix_free_l1 = mv.pre_process(matrix_free_l1)
    triplet = mv.Triplet(
        wavelength=[300.0, 320.0],
        weights=[1.0, -1.0],
        altitude_range=[10_000.0, 20_000.0],
        normalization_range=[30_000.0, 30_000.0],
        group_by="image",
    )

    materialized = triplet.apply(materialized_l1)
    matrix_free = triplet.apply(matrix_free_l1)

    _assert_operator_matches(materialized, matrix_free)
    expected = []
    for image_slice in (slice(0, 3), slice(3, 6)):
        for los in range(image_slice.start, image_slice.stop - 1):
            expected.append(
                (
                    np.log(materialized_l1["measurement"].data.radiance[0, los])
                    - np.log(
                        materialized_l1["measurement"].data.radiance[
                            0, image_slice.stop - 1
                        ]
                    )
                )
                - (
                    np.log(materialized_l1["measurement"].data.radiance[1, los])
                    - np.log(
                        materialized_l1["measurement"].data.radiance[
                            1, image_slice.stop - 1
                        ]
                    )
                )
            )
    np.testing.assert_allclose(materialized.y, expected)
    _assert_operator_matches(
        mv.wavelength_mean(materialized_l1, wavelength=slice(300.0, 320.0)),
        mv.wavelength_mean(matrix_free_l1, wavelength=slice(300.0, 320.0)),
    )


def test_altitude_weighted_triplet_sum_matches_legacy_combination():
    materialized_l1, matrix_free_l1 = _orbital_l1_pair()
    materialized_l1 = mv.pre_process(materialized_l1)
    matrix_free_l1 = mv.pre_process(matrix_free_l1)
    triplet_sum = mv.AltitudeWeightedTripletSum(
        wavelength=[[300.0, 320.0], [300.0]],
        weights=[[-1.0, 1.0], [0.25]],
        normalization_range=[[30_000.0, 30_000.0], [30_000.0, 30_000.0]],
        altitude_weight_grid=[
            [0.0, 10_000.0, 20_000.0, 100_000.0],
            [0.0, 100_000.0],
        ],
        altitude_weight_values=[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0]],
        altitude_range=[10_000.0, 20_000.0],
        group_by="image",
        legacy_linear_covariance=True,
    )

    materialized = triplet_sum.apply(materialized_l1)
    matrix_free = triplet_sum.apply(matrix_free_l1)
    _assert_operator_matches(materialized, matrix_free)

    radiance = np.asarray(materialized_l1["measurement"].data.radiance)
    noise = np.asarray(materialized_l1["measurement"].data.radiance_noise)
    expected_y = []
    expected_variance = []
    for image_slice in (slice(0, 3), slice(3, 6)):
        normalization_index = image_slice.stop - 1
        for local_index, los in enumerate(
            range(image_slice.start, image_slice.stop - 1)
        ):
            first_triplet = -np.log(
                radiance[0, los] / radiance[0, normalization_index]
            ) + np.log(radiance[1, los] / radiance[1, normalization_index])
            second_triplet = 0.25 * np.log(
                radiance[0, los] / radiance[0, normalization_index]
            )
            first_altitude_weight = float(local_index)
            expected_y.append(
                first_altitude_weight * first_triplet + 2 * second_triplet
            )

            first_variance = (noise[0, los] / radiance[0, los]) ** 2 + (
                noise[1, los] / radiance[1, los]
            ) ** 2
            second_variance = 0.25 * (noise[0, los] / radiance[0, los]) ** 2
            expected_variance.append(
                first_altitude_weight * first_variance + 2 * second_variance
            )

    np.testing.assert_allclose(materialized.y, expected_y)
    np.testing.assert_allclose(materialized.Sy.diagonal(), expected_variance)


def test_compiled_altitude_weighted_triplet_ignores_inactive_nan_triplet():
    materialized_l1, _ = _orbital_l1_pair()
    materialized_l1 = mv.pre_process(materialized_l1)
    materialized_l1["measurement"].data["radiance"][1, 2] = -1.0
    configuration = {
        "wavelength": [[300.0, 320.0], [300.0]],
        "weights": [[-1.0, 1.0], [0.25]],
        "normalization_range": [
            [30_000.0, 30_000.0],
            [30_000.0, 30_000.0],
        ],
        "altitude_weight_grid": [
            [0.0, 10_000.0, 20_000.0, 100_000.0],
            [0.0, 100_000.0],
        ],
        "altitude_weight_values": [[0.0, 0.0, 1.0, 1.0], [2.0, 2.0]],
        "altitude_range": [10_000.0, 20_000.0],
        "group_by": "image",
        "legacy_linear_covariance": True,
    }
    compiled = mv.AltitudeWeightedTripletSum(**configuration).apply(materialized_l1)

    # The invalid 320-nm normalization affects the 20-km row where that
    # triplet has nonzero altitude weight, but must not invalidate the 10-km
    # row where the triplet is exactly inactive.
    assert np.isfinite(compiled.y[0])
    assert np.isnan(compiled.y[1])
    assert np.all(np.isfinite(compiled.y[2:]))


def test_altitude_weighted_triplet_sum_validates_elementwise_factors():
    materialized_l1, _ = _l1_pair()
    measurement = mv.select(materialized_l1, wavelength=300.0)

    with pytest.raises(ValueError, match="must match the measurement shape"):
        mv.multiply_elementwise(measurement, np.ones(2))


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


def test_fisher_diagonal_error_analysis_matches_diagonal_problem():
    K = np.diag([2.0, 3.0, 4.0])
    inv_Sy = sparse.diags([4.0, 1.0, 0.25], format="csc")
    inv_Sa = sparse.diags([1.0, 2.0, 3.0], format="csc")

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
    result = estimate_fisher_diagonal_error_from_operator(
        operator,
        inv_Sy,
        inv_Sa,
        fisher_probe_count=1,
        posterior_probe_count=4096,
        posterior_probe_batch_size=256,
        random_seed=123,
    )

    expected_fisher = np.diag(K.T @ inv_Sy @ K)
    expected_posterior = 1 / (expected_fisher + inv_Sa.diagonal())
    expected_row_sum = expected_fisher * expected_posterior
    np.testing.assert_allclose(
        result["measurement_information_diagonal"], expected_fisher
    )
    np.testing.assert_allclose(
        result["solution_covariance_diagonal"], expected_posterior, rtol=0.02
    )
    np.testing.assert_allclose(
        result["approximate_averaging_kernel_row_sum"], expected_row_sum
    )
    assert result["averaging_kernel_row_sum_group_count"] == 1
    assert np.all(np.isnan(result["measurement_information_diagonal_standard_error"]))
    assert np.all(np.isfinite(result["solution_covariance_diagonal_standard_error"]))
    assert operator.matvec_calls == 0
    assert operator.rmatvec_calls == 1


def test_approximate_row_sum_is_reported_in_output_state_coordinates():
    K = np.diag([2.0, 3.0])
    prior_precision = np.array([[2.0, -1.0], [-1.0, 2.0]])
    output_mapping = np.array([2.0, 0.5])

    class Operator:
        shape = K.shape

        @staticmethod
        def rmatvec(value):
            return K.T @ value

    result = estimate_fisher_diagonal_error_from_operator(
        Operator(),
        np.eye(2),
        prior_precision,
        prior_precision_factor=np.linalg.cholesky(prior_precision).T,
        output_state_derivative_by_retrieval_state=output_mapping,
        fisher_probe_count=1,
        posterior_probe_count=1,
        random_seed=2,
    )

    fisher = np.diag(K.T @ K)
    expected = output_mapping * np.linalg.solve(
        prior_precision + np.diag(fisher),
        fisher / output_mapping,
    )
    np.testing.assert_allclose(result["approximate_averaging_kernel_row_sum"], expected)


def test_matrix_free_row_sum_matches_full_averaging_kernel():
    K = np.array([[1.0, 0.2, -0.1], [0.4, 1.3, 0.5], [-0.2, 0.7, 1.1]])
    inv_Sy = np.diag([2.0, 0.5, 1.5])
    prior_precision = np.array([[1.5, -0.2, 0.0], [-0.2, 1.2, -0.1], [0.0, -0.1, 0.8]])
    output_mapping = np.array([0.5, 2.0, 1.5])

    class Operator:
        shape = K.shape

        @staticmethod
        def matvec(value):
            return K @ value

        @staticmethod
        def rmatvec(value):
            return K.T @ value

    result = estimate_fisher_diagonal_error_from_operator(
        Operator(),
        inv_Sy,
        prior_precision,
        prior_precision_factor=np.linalg.cholesky(prior_precision).T,
        output_state_derivative_by_retrieval_state=output_mapping,
        fisher_probe_count=8,
        posterior_probe_count=1,
        averaging_kernel_row_sum_mode="matrix_free",
        averaging_kernel_row_sum_rtol=1.0e-12,
        averaging_kernel_row_sum_maxiter=10,
        random_seed=3,
    )

    measurement_information = K.T @ inv_Sy @ K
    expected = output_mapping * np.linalg.solve(
        measurement_information + prior_precision,
        measurement_information @ (1 / output_mapping),
    )
    np.testing.assert_allclose(result["averaging_kernel_row_sum"], expected)
    assert result["averaging_kernel_row_sum_krylov_info"] == 0
    assert result["averaging_kernel_row_sum_krylov_iterations"] <= 3
    assert result["averaging_kernel_row_sum_jvp_calls"] > 0
    assert result["averaging_kernel_row_sum_vjp_calls"] > 0


def test_matrix_free_row_sum_stays_within_physical_state_groups():
    K = np.array([[1.0, 0.2, -0.1], [0.4, 1.3, 0.5], [-0.2, 0.7, 1.1]])
    inv_Sy = np.diag([2.0, 0.5, 1.5])
    prior_precision = np.diag([1.5, 1.2, 0.8])
    output_mapping = np.array([0.5, 2.0, 1.5])
    groups = np.array([0, 0, 1])

    class Operator:
        shape = K.shape

        @staticmethod
        def matvec(value):
            return K @ value

        @staticmethod
        def rmatvec(value):
            return K.T @ value

    result = estimate_fisher_diagonal_error_from_operator(
        Operator(),
        inv_Sy,
        prior_precision,
        prior_precision_factor=np.linalg.cholesky(prior_precision).T,
        output_state_derivative_by_retrieval_state=output_mapping,
        averaging_kernel_row_sum_groups=groups,
        fisher_probe_count=8,
        posterior_probe_count=1,
        averaging_kernel_row_sum_mode="matrix_free",
        averaging_kernel_row_sum_rtol=1.0e-12,
        averaging_kernel_row_sum_maxiter=10,
        random_seed=3,
    )

    measurement_information = K.T @ inv_Sy @ K
    transform = np.diag(output_mapping)
    averaging_kernel = (
        transform
        @ np.linalg.solve(
            measurement_information + prior_precision,
            measurement_information,
        )
        @ np.linalg.inv(transform)
    )
    expected = np.concatenate(
        (
            averaging_kernel[:2, :2].sum(axis=1),
            averaging_kernel[2:, 2:].sum(axis=1),
        )
    )
    np.testing.assert_allclose(result["averaging_kernel_row_sum"], expected)
    assert result["averaging_kernel_row_sum_group_count"] == 2
    assert result["averaging_kernel_row_sum_krylov_info"] == 0


def test_averaging_kernel_resolution_matches_full_matrix_moments():
    K = np.diag([2.0, 2.0, 2.0])
    inv_Sy = np.eye(3)
    difference = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    prior_precision = difference.T @ difference
    vertical_m = np.array([0.0, 1_000.0, 2_000.0])
    horizontal_m = np.array([0.0, 2_000.0, 4_000.0])

    class Operator:
        shape = K.shape

        @staticmethod
        def matvec(value):
            return K @ value

        @staticmethod
        def rmatvec(value):
            return K.T @ value

    result = estimate_fisher_diagonal_error_from_operator(
        Operator(),
        inv_Sy,
        prior_precision,
        prior_precision_factor=difference,
        averaging_kernel_resolution_coordinates={
            "vertical_resolution_m": vertical_m,
            "horizontal_resolution_m": horizontal_m,
        },
        fisher_probe_count=1,
        posterior_probe_count=1,
        averaging_kernel_row_sum_mode="matrix_free",
        averaging_kernel_resolution_mode="matrix_free",
        averaging_kernel_row_sum_rtol=1.0e-12,
        averaging_kernel_row_sum_maxiter=10,
    )

    measurement_information = K.T @ K
    averaging_kernel = np.linalg.solve(
        measurement_information + prior_precision,
        measurement_information,
    )
    row_sum = averaging_kernel.sum(axis=1)

    def expected_resolution(coordinate):
        mean = averaging_kernel @ coordinate / row_sum
        variance = averaging_kernel @ coordinate**2 / row_sum - mean**2
        return 2 * np.sqrt(2 * np.log(2)) * np.sqrt(variance)

    for name, coordinate in (
        ("vertical_resolution_m", vertical_m),
        ("horizontal_resolution_m", horizontal_m),
    ):
        expected = expected_resolution(coordinate)
        np.testing.assert_allclose(
            result[f"approximate_averaging_kernel_{name}"],
            expected,
        )
        np.testing.assert_allclose(
            result[f"averaging_kernel_{name}"],
            expected,
        )
    assert result["averaging_kernel_resolution_krylov_info"] == 0
    assert result["averaging_kernel_resolution_jvp_calls"] > 0
    assert (
        result["averaging_kernel_resolution_definition"]
        == "signed_moment_gaussian_equivalent_fwhm"
    )
