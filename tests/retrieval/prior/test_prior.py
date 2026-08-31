from __future__ import annotations

import numpy as np
import sasktran2 as sk
from scipy import sparse

import skretrieval.retrieval.prior as prior
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent
from skretrieval.retrieval.tikhonov import two_dim_horizontal_second_deriv


def _sv_ele():
    # dummy constituent for testing
    const = sk.climatology.mipas.constituent("o3", sk.optical.O3DBM())

    return StateVectorElementConstituent(const, "o3", property_names=["vmr"])


def test_vert_tikh():
    alt = np.arange(0, 50000, 500)
    for order in range(1, 3):
        _ = prior.VerticalTikhonov(alt, order)


def test_multiplicative_prior():
    one = 4 * prior.VerticalTikhonov(1)

    two = prior.VerticalTikhonov(1)

    sv = _sv_ele()

    one.init(sv)
    two.init(sv)

    np.testing.assert_allclose(one.inverse_covariance, two.inverse_covariance * 4)


def test_additive_prior():
    one = 4 * prior.VerticalTikhonov(1)

    two = prior.VerticalTikhonov(1)

    p = one + two

    p.init(_sv_ele())

    _ = p.state
    _ = p.inverse_covariance


def test_two_dimensional_tikhonov_accepts_gridded_factors():
    shape = (2, 3)
    vertical = np.array([1.0, 2.0, 3.0])
    horizontal = np.array([[4.0], [5.0]])
    diagonal = np.arange(1.0, 7.0).reshape(shape)
    gridded = prior.TwoDimensionalTikhonov(
        shape,
        vertical_factor=vertical,
        horizontal_factor=horizontal,
        diagonal_factor=diagonal,
    )
    state = np.arange(6.0)
    gridded.init(type("State", (), {"state": lambda _self: state})())

    assert sparse.issparse(gridded.precision_factor)
    np.testing.assert_allclose(
        gridded.precision_factor[-state.size :].toarray(),
        np.diag(np.sqrt(diagonal.reshape(-1))),
    )
    np.testing.assert_allclose(
        (gridded.precision_factor.T @ gridded.precision_factor).toarray(),
        gridded.inverse_covariance.toarray(),
    )


def test_two_dimensional_tikhonov_targets_vertically_coherent_oscillations():
    shape = (7, 9)
    coherent = prior.TwoDimensionalTikhonov(
        shape,
        coherent_horizontal_factor=1.0,
        coherent_horizontal_order=2,
        coherent_vertical_sigma=2.0,
    )
    state = np.zeros(np.prod(shape))
    coherent.init(type("State", (), {"state": lambda _self: state})())

    horizontal_oscillation = (-1.0) ** np.arange(shape[0])
    vertically_coherent = np.outer(horizontal_oscillation, np.ones(shape[1]))
    vertically_localized = np.zeros(shape)
    vertically_localized[:, shape[1] // 2] = horizontal_oscillation * np.sqrt(shape[1])

    coherent_norm = np.linalg.norm(
        coherent.precision_factor @ vertically_coherent.reshape(-1)
    )
    localized_norm = np.linalg.norm(
        coherent.precision_factor @ vertically_localized.reshape(-1)
    )
    assert coherent_norm > 2.0 * localized_norm
    np.testing.assert_allclose(
        (coherent.precision_factor.T @ coherent.precision_factor).toarray(),
        coherent.inverse_covariance.toarray(),
    )


def test_two_dimensional_tikhonov_requires_coherence_scale_when_enabled():
    with np.testing.assert_raises_regex(
        ValueError,
        "coherent_vertical_sigma must be positive",
    ):
        prior.TwoDimensionalTikhonov(
            (3, 4),
            coherent_horizontal_factor=1.0,
        )


def test_two_dimensional_tikhonov_band_limits_coherent_curvature():
    shape = (129, 5)
    coherent = prior.TwoDimensionalTikhonov(
        shape,
        coherent_horizontal_factor=1.0,
        coherent_horizontal_order=2,
        coherent_horizontal_smoothing_sigma=2.0,
        coherent_vertical_sigma=1.0,
    )
    state = np.zeros(np.prod(shape))
    coherent.init(type("State", (), {"state": lambda _self: state})())

    def residual_norm(wavelength_cells: float) -> float:
        angle = 2.0 * np.pi * np.arange(shape[0]) / wavelength_cells
        field = np.outer(np.sin(angle), np.ones(shape[1]))
        return np.linalg.norm(coherent.precision_factor @ field.reshape(-1))

    target = residual_norm(9.0)
    assert target > 2.0 * residual_norm(3.0)
    assert target > 1.5 * residual_norm(24.0)


def test_two_dimensional_tikhonov_coherent_cost_is_grid_spacing_aware():
    def residual_norm(spacing: float) -> float:
        horizontal = np.arange(0.0, 120.0 + 0.5 * spacing, spacing)
        shape = (horizontal.size, 3)
        coherent = prior.TwoDimensionalTikhonov(
            shape,
            coherent_horizontal_factor=1.0,
            coherent_horizontal_order=2,
            coherent_horizontal_smoothing_sigma=2.0 / spacing,
            coherent_horizontal_spacing=spacing,
            coherent_vertical_sigma=1.0,
        )
        state = np.zeros(np.prod(shape))
        coherent.init(type("State", (), {"state": lambda _self: state})())
        field = np.outer(
            np.sin(2.0 * np.pi * horizontal / 9.0),
            np.ones(shape[1]),
        )
        return np.linalg.norm(coherent.precision_factor @ field.reshape(-1))

    np.testing.assert_allclose(residual_norm(1.0), residual_norm(0.5), rtol=0.08)


def test_two_dimensional_tikhonov_validates_horizontal_coherence_scales():
    with np.testing.assert_raises_regex(
        ValueError,
        "coherent_horizontal_smoothing_sigma must be finite and non-negative",
    ):
        prior.TwoDimensionalTikhonov(
            (3, 4),
            coherent_horizontal_smoothing_sigma=-1.0,
        )
    with np.testing.assert_raises_regex(
        ValueError,
        "coherent_horizontal_spacing must be finite and positive",
    ):
        prior.TwoDimensionalTikhonov(
            (3, 4),
            coherent_horizontal_spacing=0.0,
        )


def test_integrated_column_tikhonov_matches_log_column_linearization():
    shape = (9, 4)
    dz = np.array([500.0, 1_000.0, 1_000.0, 500.0])
    reference_extinction = np.geomspace(1.0e-6, 8.0e-5, np.prod(shape)).reshape(shape)
    column_scale = 1.0e-2
    reference_state = np.log(reference_extinction)
    # A unit log-state displacement at every altitude produces the reference
    # column under the local linearization, so this affine target makes the
    # residual equal the curvature of the absolute reference column.
    affine_target = reference_state - 1.0
    column = prior.TwoDimensionalIntegratedColumnTikhonov(
        shape,
        reference_extinction * dz[np.newaxis, :] / column_scale,
        horizontal_factor=1.0,
        horizontal_order=2,
        prior_state=affine_target,
    )
    column.init(
        type(
            "State",
            (),
            {"state": lambda _self: reference_state.reshape(-1)},
        )()
    )

    expected_column = np.sum(reference_extinction * dz[np.newaxis, :], axis=1)
    expected_curvature = two_dim_horizontal_second_deriv(
        shape[0],
        1,
        sparse=True,
    ) @ (expected_column / column_scale)
    actual = column.precision_factor @ (
        reference_state.reshape(-1) - affine_target.reshape(-1)
    )
    np.testing.assert_allclose(actual, expected_curvature)
    np.testing.assert_allclose(
        (column.precision_factor.T @ column.precision_factor).toarray(),
        column.inverse_covariance.toarray(),
    )


def test_integrated_column_tikhonov_weights_absolute_not_relative_changes():
    shape = (7, 2)
    column_weights = np.ones(shape)
    column_weights[:, 1] = 100.0
    column = prior.TwoDimensionalIntegratedColumnTikhonov(
        shape,
        column_weights,
        horizontal_factor=1.0,
    )
    state = np.zeros(np.prod(shape))
    column.init(type("State", (), {"state": lambda _self: state})())

    horizontal_pattern = np.sin(2.0 * np.pi * np.arange(shape[0]) / 4.0)
    low_weight_change = np.zeros(shape)
    low_weight_change[:, 0] = horizontal_pattern
    high_weight_change = np.zeros(shape)
    high_weight_change[:, 1] = horizontal_pattern
    low_norm = np.linalg.norm(column.precision_factor @ low_weight_change.reshape(-1))
    high_norm = np.linalg.norm(column.precision_factor @ high_weight_change.reshape(-1))
    np.testing.assert_allclose(high_norm, 100.0 * low_norm)


def test_integrated_column_tikhonov_validates_inputs():
    with np.testing.assert_raises_regex(
        ValueError,
        "column_weights must have shape",
    ):
        prior.TwoDimensionalIntegratedColumnTikhonov(
            (3, 4),
            np.ones((4, 3)),
            horizontal_factor=1.0,
        )
    with np.testing.assert_raises_regex(
        ValueError,
        "horizontal_factor must be finite and non-negative",
    ):
        prior.TwoDimensionalIntegratedColumnTikhonov(
            (3, 4),
            np.ones((3, 4)),
            horizontal_factor=-1.0,
        )
