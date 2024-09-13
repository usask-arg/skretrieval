from __future__ import annotations

import numpy as np
import sasktran2 as sk

import skretrieval.retrieval.prior as prior
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent


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
