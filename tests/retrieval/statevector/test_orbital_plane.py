from __future__ import annotations

import numpy as np
import sasktran2 as sk2
import xarray as xr
from scipy import sparse

from skretrieval.retrieval.prior import TwoDimensionalTikhonov
from skretrieval.retrieval.statevector.orbital_plane import (
    OrbitalPlaneStateVector,
    OrbitalPlaneStateVectorElement,
)


def _geometry() -> sk2.OrbitalPlaneGeometry:
    angles = np.array([-0.1, 0.0, 0.1])
    radius = 6_372_000.0
    track = radius * np.column_stack(
        (np.sin(angles), np.zeros_like(angles), np.cos(angles))
    )
    return sk2.OrbitalPlaneGeometry(
        radius,
        np.array([0.0, 10_000.0, 20_000.0, 30_000.0]),
        track,
    )


class _Aerosol:
    def __init__(self):
        self.extinction_per_m = np.arange(1.0, 13.0).reshape(3, 4) * 1.0e-6
        self.median_radius = np.arange(100.0, 112.0).reshape(3, 4)


class _Surface:
    def __init__(self):
        self.albedo = np.full((3, 2), 0.2)


def test_orbital_element_separates_non_spatial_row_sum_groups():
    geometry = _geometry()
    element = OrbitalPlaneStateVectorElement(
        _Surface(),
        "surface",
        ["albedo"],
        geometry=geometry,
        property_dimensions={"albedo": ("orbital_position", "albedo_wavelength")},
    )

    np.testing.assert_array_equal(
        element.averaging_kernel_row_sum_groups(),
        np.tile([0, 1], 3),
    )
    coordinates = element.averaging_kernel_resolution_coordinates()
    assert np.all(np.isnan(coordinates["vertical_resolution_m"]))
    assert np.all(np.isfinite(coordinates["horizontal_resolution_m"]))


def test_two_dimensional_tikhonov_is_sparse_and_altitude_fastest():
    geometry = _geometry()
    aerosol = _Aerosol()
    mask = np.broadcast_to(np.array([False, True, True, False]), geometry.shape).copy()
    element = OrbitalPlaneStateVectorElement(
        aerosol,
        "aerosol",
        ["extinction_per_m"],
        geometry=geometry,
        retrieval_masks={"extinction_per_m": mask},
        prior={
            "extinction_per_m": TwoDimensionalTikhonov(
                (3, 2),
                vertical_factor=2.0,
                horizontal_factor=3.0,
                diagonal_factor=0.5,
            )
        },
        log_space=True,
    )

    assert element.state().shape == (6,)
    assert sparse.issparse(element.inverse_apriori_covariance())
    assert element.inverse_apriori_covariance().shape == (6, 6)
    factor = element.prior_precision_factor()
    assert sparse.issparse(factor)
    np.testing.assert_allclose(
        (factor.T @ factor).toarray(),
        element.inverse_apriori_covariance().toarray(),
    )
    np.testing.assert_allclose(
        element.apriori_state(), np.log(aerosol.extinction_per_m[mask])
    )


def test_masked_orbital_element_maps_jvp_vjp_and_describes_grid():
    geometry = _geometry()
    aerosol = _Aerosol()
    initial = aerosol.extinction_per_m.copy()
    mask = np.broadcast_to(np.array([False, True, True, False]), geometry.shape).copy()
    element = OrbitalPlaneStateVectorElement(
        aerosol,
        "aerosol",
        ["extinction_per_m"],
        geometry=geometry,
        retrieval_masks={"extinction_per_m": mask},
        prior={"extinction_per_m": TwoDimensionalTikhonov((3, 2), diagonal_factor=1.0)},
        log_space=True,
    )
    template = xr.Dataset(
        {
            "aerosol_extinction": xr.DataArray(
                np.zeros(geometry.shape),
                dims=("orbital_position", "altitude"),
            )
        }
    )

    tangent = {}
    direction = np.linspace(0.2, 0.7, 6)
    element.add_to_linearization_tangent(tangent, direction, template)
    expected = np.zeros(geometry.shape)
    expected[mask] = initial[mask] * direction
    np.testing.assert_allclose(tangent["aerosol_extinction"], expected)

    gradient = xr.Dataset(
        {
            "aerosol_extinction": xr.DataArray(
                np.arange(12.0).reshape(geometry.shape),
                dims=("orbital_position", "altitude"),
            )
        }
    )
    np.testing.assert_allclose(
        element.linearization_gradient(gradient, template),
        gradient["aerosol_extinction"].values[mask] * initial[mask],
    )

    materialized = xr.Dataset(
        {
            "radiance": xr.DataArray(
                np.zeros((1, 2, 1)), dims=("wavelength", "los", "stokes")
            ),
            "wf_aerosol_extinction": xr.DataArray(
                np.arange(24.0).reshape((*geometry.shape, 1, 2, 1)),
                dims=(
                    "orbital_position",
                    "altitude",
                    "wavelength",
                    "los",
                    "stokes",
                ),
            ),
        }
    )
    mapped = element.propagate_wf(materialized)
    expected_wf = materialized["wf_aerosol_extinction"].values[mask]
    expected_wf *= initial[mask, None, None, None]
    assert mapped.dims == ("x", "wavelength", "los", "stokes")
    np.testing.assert_allclose(mapped, expected_wf)

    element.update_state(np.log(np.full(6, 9.0e-6)))
    np.testing.assert_allclose(aerosol.extinction_per_m[mask], 9.0e-6)
    np.testing.assert_allclose(aerosol.extinction_per_m[~mask], initial[~mask])

    state = OrbitalPlaneStateVector(geometry, aerosol=element).describe(
        {
            "solution_covariance_diagonal": np.full(6, 0.04),
            "solution_covariance_diagonal_standard_error": np.full(6, 0.004),
            "measurement_information_diagonal": np.full(6, 2.0),
            "measurement_information_diagonal_standard_error": np.full(6, 0.5),
            "approximate_averaging_kernel_row_sum": np.full(6, 0.8),
            "averaging_kernel_row_sum": np.full(6, 0.9),
            "approximate_averaging_kernel_vertical_resolution_m": np.full(6, 2_000.0),
            "approximate_averaging_kernel_horizontal_resolution_m": np.full(
                6, 300_000.0
            ),
        }
    )
    assert state["aerosol_extinction_per_m"].dims == (
        "orbital_position",
        "aerosol_altitude",
    )
    np.testing.assert_array_equal(state.aerosol_altitude, [10_000.0, 20_000.0])
    assert "along_track_angle_rad" in state.coords
    np.testing.assert_allclose(
        state["aerosol_extinction_per_m_posterior_1sigma"],
        state["aerosol_extinction_per_m"] * 0.2,
    )
    np.testing.assert_allclose(
        state["aerosol_extinction_per_m_posterior_variance_relative_mc_error"],
        0.1,
    )
    np.testing.assert_allclose(
        state["aerosol_extinction_per_m_measurement_information_relative_mc_error"],
        0.25,
    )
    np.testing.assert_allclose(
        state["aerosol_extinction_per_m_approximate_averaging_kernel_row_sum"],
        0.8,
    )
    np.testing.assert_allclose(
        state["aerosol_extinction_per_m_averaging_kernel_row_sum"],
        0.9,
    )
    np.testing.assert_allclose(
        state[
            "aerosol_extinction_per_m_approximate_averaging_kernel_vertical_resolution_km"
        ],
        10.0,
    )
    np.testing.assert_allclose(
        state[
            "aerosol_extinction_per_m_approximate_averaging_kernel_horizontal_resolution_km"
        ],
        637.2,
    )
    assert (
        state[
            "aerosol_extinction_per_m_approximate_averaging_kernel_vertical_resolution_km"
        ].attrs["units"]
        == "km"
    )


def test_orbital_element_supports_mixed_property_parameterizations():
    geometry = _geometry()
    aerosol = _Aerosol()
    initial_extinction = aerosol.extinction_per_m.copy()
    initial_radius = aerosol.median_radius.copy()
    mask = np.broadcast_to(np.array([False, True, True, False]), geometry.shape).copy()
    shape = (3, 2)
    element = OrbitalPlaneStateVectorElement(
        aerosol,
        "aerosol",
        ["extinction_per_m", "median_radius"],
        geometry=geometry,
        retrieval_masks={
            "extinction_per_m": mask,
            "median_radius": mask,
        },
        min_value={"extinction_per_m": 1.0e-15, "median_radius": 40.0},
        max_value={"extinction_per_m": 1.0e-2, "median_radius": 500.0},
        prior={
            "extinction_per_m": TwoDimensionalTikhonov(
                shape,
                diagonal_factor=1.0,
                prior_state=initial_extinction[mask] * 1.0e6,
            ),
            "median_radius": TwoDimensionalTikhonov(
                shape,
                diagonal_factor=1.0,
                prior_state=np.log(initial_radius[mask]),
            ),
        },
        log_space={"extinction_per_m": False, "median_radius": True},
        scale_factor={"extinction_per_m": 1.0e6, "median_radius": 1.0},
    )

    expected_state = np.concatenate(
        (initial_extinction[mask] * 1.0e6, np.log(initial_radius[mask]))
    )
    np.testing.assert_allclose(element.state(), expected_state)
    np.testing.assert_array_equal(
        element.averaging_kernel_row_sum_groups(),
        np.repeat([0, 1], 6),
    )
    resolution_coordinates = element.averaging_kernel_resolution_coordinates()
    np.testing.assert_allclose(
        resolution_coordinates["vertical_resolution_m"],
        np.tile([10_000.0, 20_000.0], 6),
    )
    assert np.all(np.isfinite(resolution_coordinates["horizontal_resolution_m"]))
    np.testing.assert_allclose(element.lower_bound()[:6], 1.0e-9)
    np.testing.assert_allclose(element.lower_bound()[6:], np.log(40.0))
    np.testing.assert_allclose(element.upper_bound()[:6], 1.0e4)
    np.testing.assert_allclose(element.upper_bound()[6:], np.log(500.0))

    template = xr.Dataset(
        {
            "aerosol_extinction": xr.DataArray(
                np.zeros(geometry.shape),
                dims=("orbital_position", "altitude"),
            ),
            "aerosol_median_radius": xr.DataArray(
                np.zeros(geometry.shape),
                dims=("orbital_position", "altitude"),
            ),
        }
    )
    direction = np.arange(1.0, 13.0)
    tangent = {}
    element.add_to_linearization_tangent(tangent, direction, template)
    expected_extinction_tangent = np.zeros(geometry.shape)
    expected_extinction_tangent[mask] = direction[:6] / 1.0e6
    expected_radius_tangent = np.zeros(geometry.shape)
    expected_radius_tangent[mask] = initial_radius[mask] * direction[6:]
    np.testing.assert_allclose(
        tangent["aerosol_extinction"], expected_extinction_tangent
    )
    np.testing.assert_allclose(
        tangent["aerosol_median_radius"], expected_radius_tangent
    )

    gradient_values = np.arange(1.0, 13.0).reshape(geometry.shape)
    gradient = xr.Dataset(
        {
            "aerosol_extinction": xr.DataArray(
                gradient_values,
                dims=("orbital_position", "altitude"),
            ),
            "aerosol_median_radius": xr.DataArray(
                2 * gradient_values,
                dims=("orbital_position", "altitude"),
            ),
        }
    )
    expected_gradient = np.concatenate(
        (
            gradient_values[mask] / 1.0e6,
            2 * gradient_values[mask] * initial_radius[mask],
        )
    )
    np.testing.assert_allclose(
        element.linearization_gradient(gradient, template), expected_gradient
    )

    updated_state = np.concatenate((np.full(6, 2.0), np.log(np.full(6, 200.0))))
    element.update_state(updated_state)
    np.testing.assert_allclose(aerosol.extinction_per_m[mask], 2.0e-6)
    np.testing.assert_allclose(aerosol.median_radius[mask], 200.0)
    np.testing.assert_allclose(
        aerosol.extinction_per_m[~mask], initial_extinction[~mask]
    )
    np.testing.assert_allclose(aerosol.median_radius[~mask], initial_radius[~mask])
