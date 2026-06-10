from __future__ import annotations

import numpy as np
import sasktran2 as sk2

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
