from __future__ import annotations

import numpy as np

import skretrieval.retrieval.usarm.prior as prior


def test_vert_tikh():
    alt = np.arange(0, 50000, 500)
    for order in range(1, 3):
        _ = prior.VerticalTikhonov(alt, order)


def test_multiplicative_prior():
    alt = np.arange(0, 50000, 500)

    one = 4 * prior.VerticalTikhonov(alt, 1)

    two = prior.VerticalTikhonov(alt, 1, tikhonov=np.ones(len(alt)) * 2)

    np.testing.assert_allclose(one.inverse_covariance, two.inverse_covariance)


def test_additive_prior():
    alt = np.arange(0, 50000, 500)

    one = 4 * prior.VerticalTikhonov(alt, 1)

    two = prior.VerticalTikhonov(alt, 1, tikhonov=np.ones(len(alt)) * 2)

    p = one + two

    _ = p.state
    _ = p.inverse_covariance
