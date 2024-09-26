from __future__ import annotations

import numpy as np
from scipy import sparse


def estimate_error(
    K: np.ndarray,
    Sy: np.ndarray,
    inv_Sy: np.ndarray,
    inv_Sa: np.ndarray,
    left_side_eqn: np.ndarray | None = None,
) -> dict:
    """
    Estimates the error and averaging kernel for the retrieval process

    Parameters
    ----------
    K : np.ndarray
        Jacobian matrix
    Sy : np.ndarray
        Instrument error covariance matrix
    inv_Sy : np.ndarray
        Invers of the instrument error covariance matrix
    inv_Sa : np.ndarray
        Inverse of the a priori error covariance matrix
    left_side_eqn : np.ndarray | None, optional
        Left side of the retrieval equation, (K.T @ inv_Sy @ K + inv_Sa), by default None.
        If set to None it is calculated by this function.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - gain_matrix: Gain matrix
        - averaging_kernel: Averaging kernel
        - error_covariance_from_noise: Error covariance from noise
        - solution_covariance: Solution covariance
    """
    output_dict = {}

    if left_side_eqn is not None:
        A_without_lm = left_side_eqn
    else:
        A_without_lm = K.T @ inv_Sy @ K + inv_Sa

    # Calculate the solution covariance and averaging kernels
    try:
        if sparse.issparse(A_without_lm):
            S = np.linalg.inv(A_without_lm.toarray())
        else:
            S = np.linalg.inv(A_without_lm)
    except np.linalg.LinAlgError:
        if sparse.issparse(A_without_lm):
            S = np.linalg.pinv(A_without_lm.toarray())
        else:
            S = np.linalg.pinv(A_without_lm)

    G = S @ K.T @ inv_Sy
    A = G @ K
    meas_error_covar = G @ (Sy.dot(G.T))

    output_dict["gain_matrix"] = G
    output_dict["averaging_kernel"] = A

    output_dict["error_covariance_from_noise"] = meas_error_covar
    output_dict["solution_covariance"] = S

    return output_dict
