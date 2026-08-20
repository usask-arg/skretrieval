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
    inv_Sy : np.ndarray or scipy.sparse.spmatrix
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


def _matvec(operator, x: np.ndarray) -> np.ndarray:
    if hasattr(operator, "matvec"):
        return operator.matvec(x)
    return operator @ x


def _rmatvec(operator, y: np.ndarray) -> np.ndarray:
    if hasattr(operator, "rmatvec"):
        return operator.rmatvec(y)
    return operator.T @ y


def information_sqrt(
    information: np.ndarray | sparse.spmatrix,
    matrix_name: str = "Measurement inverse covariance",
) -> np.ndarray | sparse.spmatrix:
    """Return ``W`` such that ``W.T @ W`` equals an information matrix."""
    if len(information.shape) != 2 or information.shape[0] != information.shape[1]:
        msg = f"{matrix_name} must be square"
        raise ValueError(msg)
    if sparse.issparse(information):
        diagonal = information.diagonal()
        off_diagonal = information - sparse.diags(diagonal, format="csc")
        off_diagonal.eliminate_zeros()
        if off_diagonal.nnz == 0:
            if np.any(~np.isfinite(diagonal)) or np.any(diagonal < 0):
                msg = f"{matrix_name} must be positive semidefinite"
                raise ValueError(msg)
            return sparse.diags(np.sqrt(diagonal), format="csc")
        information = information.toarray()

    information = np.asarray(information, dtype=float)
    if np.any(~np.isfinite(information)):
        msg = f"{matrix_name} must contain only finite values"
        raise ValueError(msg)
    if not np.allclose(information, information.T):
        msg = f"{matrix_name} must be symmetric"
        raise ValueError(msg)
    information = (information + information.T) / 2
    try:
        return np.linalg.cholesky(information).T
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        tolerance = (
            np.finfo(float).eps
            * max(information.shape)
            * max(1.0, np.max(np.abs(eigenvalues)))
        )
        if np.any(eigenvalues < -tolerance):
            msg = f"{matrix_name} must be positive semidefinite"
            raise ValueError(msg) from None
        eigenvalues[eigenvalues < 0] = 0
        return np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def estimate_error_from_operator(
    operator,
    inv_Sy: np.ndarray | sparse.spmatrix,
    inv_Sa: np.ndarray,
) -> dict:
    """
    Estimate retrieval diagnostics without constructing the measurement Jacobian.

    The posterior information matrix is formed one column or row at a time using
    Jacobian products. The gain matrix is intentionally omitted because storing it
    would defeat the purpose of matrix-free diagnostics.

    Parameters
    ----------
    operator
        Jacobian-like object with ``shape``, ``matvec``, and ``rmatvec``.
    inv_Sy : np.ndarray
        Inverse measurement covariance.
    inv_Sa : np.ndarray
        Inverse a priori covariance.
    """
    n_state = operator.shape[1]
    n_measurement = operator.shape[0]

    hessian_from_measurement = np.zeros((n_state, n_state))
    if n_measurement < 2 * n_state:
        # A row requires one VJP; a column requires one JVP and one VJP.
        sqrt_inv_Sy = information_sqrt(inv_Sy)
        for idx in range(n_measurement):
            if sparse.issparse(sqrt_inv_Sy):
                cotangent = sqrt_inv_Sy.getrow(idx).toarray().reshape(-1)
            else:
                cotangent = np.asarray(sqrt_inv_Sy[idx]).reshape(-1)
            weighted_row = np.asarray(_rmatvec(operator, cotangent)).reshape(-1)
            hessian_from_measurement += np.outer(weighted_row, weighted_row)
    else:
        for idx in range(n_state):
            basis = np.zeros(n_state)
            basis[idx] = 1
            weighted = inv_Sy @ _matvec(operator, basis)
            hessian_from_measurement[:, idx] = _rmatvec(operator, weighted)

    hessian_from_measurement = (
        hessian_from_measurement + hessian_from_measurement.T
    ) / 2

    left_side = hessian_from_measurement + inv_Sa

    try:
        if sparse.issparse(left_side):
            solution_covariance = np.linalg.inv(left_side.toarray())
        else:
            solution_covariance = np.linalg.inv(left_side)
    except np.linalg.LinAlgError:
        if sparse.issparse(left_side):
            solution_covariance = np.linalg.pinv(left_side.toarray())
        else:
            solution_covariance = np.linalg.pinv(left_side)

    solution_covariance = (solution_covariance + solution_covariance.T) / 2
    averaging_kernel = solution_covariance @ hessian_from_measurement
    measurement_error_covariance = (
        solution_covariance @ hessian_from_measurement @ solution_covariance.T
    )

    return {
        "averaging_kernel": averaging_kernel,
        "error_covariance_from_noise": measurement_error_covariance,
        "solution_covariance": solution_covariance,
    }
