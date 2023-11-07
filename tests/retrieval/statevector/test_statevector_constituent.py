from __future__ import annotations

import numpy as np
import sasktran2 as sk2

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
