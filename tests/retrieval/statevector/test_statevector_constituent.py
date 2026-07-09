from __future__ import annotations

import numpy as np
import sasktran2 as sk2

from skretrieval.retrieval.prior import ManualPrior
from skretrieval.retrieval.processing import Retrieval
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
