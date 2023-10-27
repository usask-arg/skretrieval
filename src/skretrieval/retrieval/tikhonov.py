from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix


def two_dim_vertical_second_deriv(numangle, numalt, factor=1, sparse=False):
    """
    Calculates the second derivatvie Tikhonov regularization matrix for a two dimensional uniform grid.
    The matrix is calculated assuming that the measurement vector is constructed with altitude being the leading
    dimension

    Parameters
    ----------
    numangle : scalar
        The number of angular grid points
    numalt : scalar
        The number of altitude grid points
    factor : scalar or length numalt, optional
        If scalar, the resulting matrix is multiplied by this value.  If a vector of length numalt, then each altitude
        level is multiplied by its corresponding factor.  Default is 1
    """

    n = numalt * numangle
    gamma = lil_matrix((n, n)) if sparse else np.zeros((n, n))
    for idangle in range(numangle):
        for idalt in range(1, numalt - 1):
            mfactor = factor[idalt] if np.shape(factor) != () else factor
            idx = idangle * numalt + idalt
            gamma[idx, idx - 1] = -1 / 4 * mfactor
            gamma[idx, idx] = 1 / 2 * mfactor
            gamma[idx, idx + 1] = -1 / 4 * mfactor

    if sparse:
        gamma = gamma.asformat("csr")

    return gamma


def two_dim_vertical_first_deriv(numangle, numalt, factor=1, sparse=False):
    """
    Calculates the first derivatvie Tikhonov regularization matrix for a two dimensional uniform grid.
    The matrix is calculated assuming that the measurement vector is constructed with altitude being the leading
    dimension

    Parameters
    ----------
    numangle : scalar
        The number of angular grid points
    numalt : scalar
        The number of altitude grid points
    factor : scalar or length numalt, optional
        If scalar, the resulting matrix is multiplied by this value.  If a vector of length numalt, then each altitude
        level is multiplied by its corresponding factor.  Default is 1
    """

    n = numalt * numangle
    gamma = lil_matrix((n, n)) if sparse else np.zeros((n, n))
    for idangle in range(numangle):
        for idalt in range(numalt - 1):
            mfactor = factor[idalt] if np.shape(factor) != () else factor
            idx = idangle * numalt + idalt
            gamma[idx, idx] = -1 / 2 * mfactor
            gamma[idx, idx + 1] = 1 / 2 * mfactor

    if sparse:
        gamma = gamma.asformat("csr")

    return gamma


def two_dim_horizontal_second_deriv(numangle, numalt, factor=1, sparse=False):
    """
    Calculates the second derivatvie Tikhonov regularization matrix for a two dimensional uniform grid.
    The matrix is calculated assuming that the measurement vector is constructed with altitude being the leading
    dimension

    Parameters
    ----------
    numangle : scalar
        The number of angular grid points
    numalt : scalar
        The number of altitude grid points
    factor : scalar or vector length numangle, optional
        If scalar, the resulting matrix is multiplied by this value.  If a vector of length numangle, then each angular
        level is multiplied by the corresponding value. Default is 1
    """

    n = numalt * numangle
    gamma = lil_matrix((n, n)) if sparse else np.zeros((n, n))
    for idx in range(numalt, n - numalt):
        if np.shape(factor) != ():
            horiz = int(idx / numalt)
            mfactor = factor[horiz]
        else:
            mfactor = factor
        gamma[idx, idx - numalt] = -1 / 4 * mfactor
        gamma[idx, idx] = 1 / 2 * mfactor
        gamma[idx, idx + numalt] = -1 / 4 * mfactor

    if sparse:
        # Convert gamma to CSR format for faster matrix operations
        gamma = gamma.asformat("csr")
    return gamma
