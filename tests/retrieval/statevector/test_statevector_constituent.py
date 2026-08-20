from __future__ import annotations

import numpy as np
import sasktran2 as sk2
import xarray as xr

from skretrieval.retrieval.prior import ManualPrior
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
