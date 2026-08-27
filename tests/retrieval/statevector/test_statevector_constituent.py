from __future__ import annotations

import numpy as np
import sasktran2 as sk2
import xarray as xr

from skretrieval.retrieval.prior import ManualPrior
from skretrieval.retrieval.processing import Retrieval, aerosol_extinction_profile
from skretrieval.retrieval.statevector import StateVector
from skretrieval.retrieval.statevector.altitude import AltitudeNativeStateVector
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent


def test_statevector_constituent_vmr():
    const = sk2.climatology.mipas.constituent("O3", sk2.optical.O3DBM())

    sv = StateVectorElementConstituent(const, "O3", ["vmr"])

    assert sv.state().shape == (50,)


def test_statevector_constituent_multiple_params():
    alt_grid = np.arange(0, 100001, 1000.0)

    const = sk2.test_util.scenarios.test_aerosol_constituent(alt_grid)

    sv = StateVectorElementConstituent(
        const, "Aerosol", ["extinction_per_m", "lognormal_median_radius"]
    )

    assert sv.state().shape == (202,)


def test_statevector_constituent_describe_scalar_array():
    class ScalarArrayConstituent:
        scalar_property = np.array([2.0])

    sv = StateVectorElementConstituent(
        ScalarArrayConstituent(),
        "scalar",
        ["scalar_property"],
        prior={"scalar_property": ManualPrior(np.array([1.0]), np.eye(1))},
    )

    result = sv.describe(
        covariance=np.array([[0.25]]), averaging_kernel=np.array([[0.8]])
    )

    assert result["scalar_scalar_property"].item() == 2.0
    assert result["scalar_scalar_property_prior"].item() == 1.0
    assert result["scalar_scalar_property_1sigma_error"].item() == 0.5
    assert result["scalar_scalar_property_averaging_kernel"].item() == 0.8


def test_statevector_describe_only_slices_diagnostics_for_enabled_elements():
    class ScalarArrayConstituent:
        def __init__(self, value):
            self.scalar_property = np.array([value])

    disabled = StateVectorElementConstituent(
        ScalarArrayConstituent(1.0),
        "disabled",
        ["scalar_property"],
        prior={"scalar_property": ManualPrior(np.array([1.0]), np.eye(1))},
        enabled=False,
    )
    enabled = StateVectorElementConstituent(
        ScalarArrayConstituent(2.0),
        "enabled",
        ["scalar_property"],
        prior={"scalar_property": ManualPrior(np.array([2.0]), np.eye(1))},
    )

    result = StateVector([disabled, enabled]).describe(
        {
            "error_covariance_from_noise": np.array([[0.25]]),
            "averaging_kernel": np.array([[0.8]]),
        }
    )

    assert "disabled_scalar_property" in result
    assert "disabled_scalar_property_1sigma_error" not in result
    assert result["enabled_scalar_property_1sigma_error"].item() == 0.5
    assert result["enabled_scalar_property_averaging_kernel"].item() == 0.8


def test_statevector_constituent_linearization_product_scaling():
    class FakeConstituent:
        vmr = np.array([1.0, 2.0, 4.0])

    template = xr.Dataset(
        {
            "gas_vmr": xr.DataArray(
                np.zeros(3),
                dims=["altitude"],
                coords={"altitude": [0.0, 10_000.0, 20_000.0]},
            )
        }
    )

    sv = StateVectorElementConstituent(
        FakeConstituent(),
        "gas",
        ["vmr"],
        scale_factor=2.0,
    )

    tangent = {}
    sv.add_to_linearization_tangent(tangent, np.array([2.0, 4.0, 6.0]), template)
    np.testing.assert_allclose(tangent["gas_vmr"], [1.0, 2.0, 3.0])

    gradient = xr.Dataset(
        {"gas_vmr": xr.DataArray([10.0, 20.0, 30.0], dims=["altitude"])}
    )
    np.testing.assert_allclose(
        sv.linearization_gradient(gradient, template), [5.0, 10.0, 15.0]
    )

    log_sv = StateVectorElementConstituent(
        FakeConstituent(),
        "gas",
        ["vmr"],
        log_space=True,
    )
    tangent = {}
    log_sv.add_to_linearization_tangent(tangent, np.ones(3), template)
    np.testing.assert_allclose(tangent["gas_vmr"], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        log_sv.linearization_gradient(gradient, template), [10.0, 40.0, 120.0]
    )


def test_statevector_constituent_linearization_extinction_alias():
    class FakeConstituent:
        extinction_per_m = np.array([1.0, 2.0])

    template = xr.Dataset(
        {
            "aerosol_extinction": xr.DataArray(
                np.zeros(2),
                dims=["altitude"],
            )
        }
    )

    sv = StateVectorElementConstituent(
        FakeConstituent(),
        "aerosol",
        ["extinction_per_m"],
    )

    assert sv.linearization_parameter_names(template) == ("aerosol_extinction",)


def test_statevector_constituent_describe_scales_linear_profile_error():
    class ProfileConstituent:
        altitudes_m = np.array([0.0, 1000.0, 2000.0])
        vmr = np.array([1.0e-6, 2.0e-6, 3.0e-6])

    scale_factor = 1000.0
    sv = StateVectorElementConstituent(
        ProfileConstituent(),
        "o3",
        ["vmr"],
        prior={
            "vmr": ManualPrior(
                np.array([1.0e-3, 2.0e-3, 3.0e-3]),
                np.eye(3),
            )
        },
        scale_factor=scale_factor,
    )

    result = sv.describe(
        covariance=np.diag([1.0, 4.0, 9.0]),
        averaging_kernel=np.eye(3),
    )

    np.testing.assert_allclose(
        result["o3_vmr_1sigma_error"],
        np.array([1.0, 2.0, 3.0]) / scale_factor,
    )


def _absorber_config(**kwargs):
    cfg = {
        "prior_influence": 5e0,
        "tikh_factor": 1e-2,
        "log_space": False,
        "min_value": 0,
        "max_value": 1,
    }
    cfg.update(kwargs)
    return cfg


def test_aerosol_prior_state_is_copied_and_all_zero_profile_is_supported():
    altitude_grid = np.arange(0.0, 4000.0, 1000.0)
    reference = sk2.test_util.scenarios.test_aerosol_constituent(altitude_grid)
    processor = Retrieval.__new__(Retrieval)
    processor._optical_property = lambda _name: reference._optical_property
    supplied_prior = np.zeros_like(altitude_grid)
    config = {
        "prior_state": supplied_prior,
        "prior": {
            "extinction_per_m": {"value": 0.0},
            "lognormal_median_radius": {"value": 105.0},
        },
        "retrieved_quantities": {
            "extinction_per_m": {
                "min_value": 0.0,
                "max_value": 1.0,
                "tikh_factor": 1.0,
                "prior_influence": 1.0,
            }
        },
    }

    element = aerosol_extinction_profile(
        processor,
        "aerosol",
        altitude_grid,
        config,
    )

    np.testing.assert_array_equal(supplied_prior, np.zeros_like(altitude_grid))
    np.testing.assert_array_equal(element.state(), np.full(4, 1.0e-15))


def _rodgers_output(num_state: int):
    return {
        "error_covariance_from_noise": np.eye(num_state) * 0.25,
        "averaging_kernel": np.eye(num_state),
    }


def test_default_absorber_can_use_separate_retrieval_altitude_grid():
    model_grid = np.arange(0, 70001, 5000.0)
    retrieval_grid = np.arange(0, 70001, 10000.0)
    processor = Retrieval.__new__(Retrieval)

    absorber = Retrieval._default_state_absorber(
        processor,
        "o3",
        model_grid,
        _absorber_config(altitude_grid=retrieval_grid),
    )
    state_vector = AltitudeNativeStateVector(model_grid, o3=absorber)
    result = state_vector.describe(_rodgers_output(len(absorber.state())))

    assert absorber.state().shape == retrieval_grid.shape
    assert result["o3_vmr"].dims == ("o3_altitude",)
    assert result["o3_vmr_prior"].dims == ("o3_altitude",)
    assert result["o3_vmr_1sigma_error"].dims == ("o3_altitude",)
    assert result["o3_vmr_averaging_kernel"].dims == (
        "o3_altitude",
        "o3_altitude_2",
    )
    np.testing.assert_allclose(result["o3_altitude"], retrieval_grid)
    np.testing.assert_allclose(result["o3_altitude_2"], retrieval_grid)
    np.testing.assert_allclose(result["altitude"], model_grid)


def test_default_absorber_keeps_legacy_altitude_dim_on_model_grid():
    model_grid = np.arange(0, 70001, 5000.0)
    processor = Retrieval.__new__(Retrieval)

    absorber = Retrieval._default_state_absorber(
        processor,
        "o3",
        model_grid,
        _absorber_config(),
    )
    state_vector = AltitudeNativeStateVector(model_grid, o3=absorber)
    result = state_vector.describe(_rodgers_output(len(absorber.state())))

    assert absorber.state().shape == model_grid.shape
    assert result["o3_vmr"].dims == ("altitude",)
    assert result["o3_vmr_prior"].dims == ("altitude",)
    assert result["o3_vmr_1sigma_error"].dims == ("altitude",)
    assert result["o3_vmr_averaging_kernel"].dims == ("altitude", "altitude_2")
    assert "o3_altitude" not in result.dims
    np.testing.assert_allclose(result["altitude"], model_grid)
