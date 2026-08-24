from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu


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


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool | np.bool_) or int(value) != value or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


def _rademacher(
    rng: np.random.Generator,
    shape: int | tuple[int, ...],
) -> np.ndarray:
    return 2.0 * rng.integers(0, 2, size=shape) - 1.0


def _gaussian_equivalent_fwhm(
    row_sum: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    coordinate_scale: float,
) -> np.ndarray:
    """Return a signed-moment Gaussian-equivalent FWHM."""
    result = np.full(row_sum.shape, np.nan)
    valid = np.isfinite(row_sum) & (np.abs(row_sum) > np.finfo(float).eps)
    mean = np.full(row_sum.shape, np.nan)
    mean_square = np.full(row_sum.shape, np.nan)
    mean[valid] = first_moment[valid] / row_sum[valid]
    mean_square[valid] = second_moment[valid] / row_sum[valid]
    variance = mean_square - mean**2
    roundoff_tolerance = 100 * np.finfo(float).eps * (1 + np.abs(mean_square) + mean**2)
    nearly_nonnegative = valid & (variance >= -roundoff_tolerance)
    variance[nearly_nonnegative] = np.maximum(variance[nearly_nonnegative], 0.0)
    result[nearly_nonnegative] = (
        2
        * np.sqrt(2 * np.log(2))
        * coordinate_scale
        * np.sqrt(variance[nearly_nonnegative])
    )
    return result


def estimate_fisher_diagonal_error_from_operator(
    operator,
    inv_Sy: np.ndarray | sparse.spmatrix,
    inv_Sa: np.ndarray | sparse.spmatrix,
    *,
    prior_precision_factor: np.ndarray | sparse.spmatrix | None = None,
    output_state_derivative_by_retrieval_state: np.ndarray | None = None,
    averaging_kernel_row_sum_groups: np.ndarray | None = None,
    averaging_kernel_resolution_coordinates: dict[str, np.ndarray] | None = None,
    fisher_probe_count: int = 32,
    posterior_probe_count: int = 1024,
    posterior_probe_batch_size: int = 32,
    random_seed: int = 0,
    averaging_kernel_row_sum_mode: str = "approximate",
    averaging_kernel_resolution_mode: str = "approximate",
    averaging_kernel_row_sum_rtol: float = 1.0e-3,
    averaging_kernel_row_sum_maxiter: int = 30,
) -> dict:
    """Estimate posterior variances using a diagonal measurement Hessian.

    Random measurement-space probes estimate
    ``diag(K.T @ inv_Sy @ K)`` using VJP products only. The complete sparse
    prior precision is retained, producing the approximate posterior
    information matrix ``inv_Sa + diag(fisher_diagonal)``. Its inverse
    diagonal is then estimated with inexpensive positive covariance samples.
    Random right-hand sides with covariance equal to the approximate
    information matrix are solved through one reusable sparse factorization;
    the mean squared solution is its inverse diagonal.

    This is deliberately an approximation: measurement-driven correlations
    between different state elements are omitted, while correlations from the
    prior/Tikhonov precision are retained exactly.
    """
    fisher_probe_count = _validate_positive_integer(
        fisher_probe_count,
        "fisher_probe_count",
    )
    posterior_probe_count = _validate_positive_integer(
        posterior_probe_count,
        "posterior_probe_count",
    )
    posterior_probe_batch_size = _validate_positive_integer(
        posterior_probe_batch_size,
        "posterior_probe_batch_size",
    )
    if isinstance(random_seed, bool | np.bool_) or int(random_seed) != random_seed:
        msg = "random_seed must be an integer"
        raise ValueError(msg)
    random_seed = int(random_seed)
    if averaging_kernel_row_sum_mode not in {
        "approximate",
        "matrix_free",
        "none",
    }:
        msg = (
            "averaging_kernel_row_sum_mode must be 'approximate', "
            "'matrix_free', or 'none'"
        )
        raise ValueError(msg)
    if averaging_kernel_resolution_mode not in {
        "approximate",
        "matrix_free",
        "none",
    }:
        msg = (
            "averaging_kernel_resolution_mode must be 'approximate', "
            "'matrix_free', or 'none'"
        )
        raise ValueError(msg)
    if (
        averaging_kernel_resolution_mode == "matrix_free"
        and averaging_kernel_row_sum_mode != "matrix_free"
    ):
        msg = (
            "matrix-free averaging-kernel resolution requires "
            "averaging_kernel_row_sum_mode='matrix_free'"
        )
        raise ValueError(msg)
    if (
        not np.isfinite(averaging_kernel_row_sum_rtol)
        or averaging_kernel_row_sum_rtol <= 0
    ):
        msg = "averaging_kernel_row_sum_rtol must be finite and positive"
        raise ValueError(msg)
    averaging_kernel_row_sum_maxiter = _validate_positive_integer(
        averaging_kernel_row_sum_maxiter,
        "averaging_kernel_row_sum_maxiter",
    )

    n_measurement, n_state = operator.shape
    sqrt_inv_Sy = information_sqrt(inv_Sy)
    if sqrt_inv_Sy.shape[1] != n_measurement:
        msg = "Measurement information factor does not match the operator"
        raise ValueError(msg)

    fisher_rng = np.random.default_rng(random_seed)
    fisher_sum = np.zeros(n_state)
    fisher_sum_squares = np.zeros(n_state)
    fisher_vjp_runtime_s = 0.0
    for _ in range(fisher_probe_count):
        residual_probe = _rademacher(fisher_rng, sqrt_inv_Sy.shape[0])
        cotangent = np.asarray(sqrt_inv_Sy.T @ residual_probe).reshape(-1)
        vjp_start = perf_counter()
        gradient = np.asarray(_rmatvec(operator, cotangent)).reshape(-1)
        fisher_vjp_runtime_s += perf_counter() - vjp_start
        sample = gradient**2
        fisher_sum += sample
        fisher_sum_squares += sample**2

    fisher_diagonal = fisher_sum / fisher_probe_count
    if fisher_probe_count == 1:
        fisher_standard_error = np.full(n_state, np.nan)
    else:
        fisher_sample_variance = np.maximum(
            (fisher_sum_squares - fisher_probe_count * fisher_diagonal**2)
            / (fisher_probe_count - 1),
            0.0,
        )
        fisher_standard_error = np.sqrt(fisher_sample_variance / fisher_probe_count)

    prior_precision = (
        inv_Sa.astype(float).tocsc()
        if sparse.issparse(inv_Sa)
        else sparse.csc_matrix(np.asarray(inv_Sa, dtype=float))
    )
    if prior_precision.shape != (n_state, n_state):
        msg = "A priori inverse covariance does not match the operator"
        raise ValueError(msg)
    if prior_precision_factor is None:
        prior_factor = information_sqrt(
            prior_precision,
            "A priori inverse covariance",
        )
    elif sparse.issparse(prior_precision_factor):
        prior_factor = prior_precision_factor.astype(float).tocsr()
    else:
        prior_factor = np.asarray(prior_precision_factor, dtype=float)
    if prior_factor.ndim != 2 or prior_factor.shape[1] != n_state:
        msg = "A priori precision factor does not match the operator"
        raise ValueError(msg)
    if output_state_derivative_by_retrieval_state is None:
        output_mapping = np.ones(n_state)
    else:
        output_mapping = np.asarray(
            output_state_derivative_by_retrieval_state,
            dtype=float,
        ).reshape(-1)
    if (
        output_mapping.shape != (n_state,)
        or np.any(~np.isfinite(output_mapping))
        or np.any(output_mapping == 0)
    ):
        msg = "Output-state derivative must be finite, nonzero, and match the operator"
        raise ValueError(msg)
    if averaging_kernel_row_sum_groups is None:
        row_sum_groups = np.zeros(n_state, dtype=int)
    else:
        row_sum_groups = np.asarray(averaging_kernel_row_sum_groups).reshape(-1)
    if row_sum_groups.shape != (n_state,):
        msg = "Averaging-kernel row-sum groups must match the operator"
        raise ValueError(msg)
    _, first_group_indices = np.unique(row_sum_groups, return_index=True)
    group_values = row_sum_groups[np.sort(first_group_indices)]
    group_masks = [row_sum_groups == value for value in group_values]
    resolution_coordinates = {}
    for name, values in (averaging_kernel_resolution_coordinates or {}).items():
        coordinate = np.asarray(values, dtype=float).reshape(-1)
        if coordinate.shape != (n_state,):
            msg = (
                f"Averaging-kernel resolution coordinate {name} must match the operator"
            )
            raise ValueError(msg)
        if np.any(np.isinf(coordinate)):
            msg = (
                f"Averaging-kernel resolution coordinate {name} cannot contain infinity"
            )
            raise ValueError(msg)
        resolution_coordinates[name] = coordinate
    if averaging_kernel_resolution_mode == "none":
        resolution_coordinates = {}
    approximate_information = prior_precision + sparse.diags(
        fisher_diagonal,
        format="csc",
    )

    factorization_start = perf_counter()
    try:
        factor = splu(approximate_information)
    except RuntimeError:
        factorization_runtime_s = perf_counter() - factorization_start
        row_sum = (
            np.zeros(n_state)
            if np.all(fisher_diagonal == 0)
            else np.full(n_state, np.nan)
        )
        singular_results = {}
        if averaging_kernel_row_sum_mode != "none":
            singular_results.update(
                approximate_averaging_kernel_row_sum=row_sum,
                averaging_kernel_row_sum_method="singular_information",
            )
        return {
            "measurement_information_diagonal": fisher_diagonal,
            "measurement_information_diagonal_standard_error": (fisher_standard_error),
            "solution_covariance_diagonal": np.full(n_state, np.inf),
            "solution_covariance_diagonal_standard_error": np.full(n_state, np.nan),
            "fisher_diagonal_probe_count": fisher_probe_count,
            "posterior_diagonal_probe_count": posterior_probe_count,
            "posterior_diagonal_completed_probe_count": 0,
            "fisher_diagonal_vjp_calls": fisher_probe_count,
            "fisher_diagonal_vjp_runtime_s": fisher_vjp_runtime_s,
            "fisher_diagonal_factorization_runtime_s": factorization_runtime_s,
            "fisher_diagonal_solve_runtime_s": 0.0,
            "fisher_diagonal_information_singular": True,
            **singular_results,
        }
    factorization_runtime_s = perf_counter() - factorization_start

    output_inverse_mapping = sparse.diags(1 / output_mapping, format="csc")
    output_prior_precision = (
        output_inverse_mapping @ prior_precision @ output_inverse_mapping
    ).tocsc()
    output_fisher_diagonal = fisher_diagonal / output_mapping**2
    row_sum_factorization_start = perf_counter()
    output_approximate_information = output_prior_precision + sparse.diags(
        output_fisher_diagonal,
        format="csc",
    )
    output_factor = splu(output_approximate_information)
    row_sum_factorization_runtime_s = perf_counter() - row_sum_factorization_start
    approximate_averaging_kernel_row_sum = np.empty(n_state)
    approximate_group_solutions = []
    for group_mask in group_masks:
        group_rhs = output_fisher_diagonal * group_mask
        group_solution = output_factor.solve(group_rhs)
        approximate_group_solutions.append(group_solution)
        approximate_averaging_kernel_row_sum[group_mask] = group_solution[group_mask]

    resolution_tasks = []
    for group_index, group_mask in enumerate(group_masks):
        for name, coordinate in resolution_coordinates.items():
            selected_coordinate = coordinate[group_mask]
            finite = np.isfinite(selected_coordinate)
            if not np.any(finite):
                continue
            if not np.all(finite):
                msg = (
                    f"Averaging-kernel resolution coordinate {name} must be "
                    "finite for either all or none of each row-sum group"
                )
                raise ValueError(msg)
            coordinate_center = 0.5 * (
                np.min(selected_coordinate) + np.max(selected_coordinate)
            )
            coordinate_scale = float(
                np.max(np.abs(selected_coordinate - coordinate_center))
            )
            first_direction = np.zeros(n_state)
            if coordinate_scale == 0:
                normalized_coordinate = np.zeros(selected_coordinate.shape)
            else:
                normalized_coordinate = (
                    selected_coordinate - coordinate_center
                ) / coordinate_scale
            first_direction[group_mask] = normalized_coordinate
            second_direction = first_direction**2
            resolution_tasks.append(
                (
                    group_index,
                    name,
                    group_mask,
                    coordinate_scale,
                    first_direction,
                    second_direction,
                )
            )

    approximate_resolution_results = {
        name: np.full(n_state, np.nan) for name in resolution_coordinates
    }
    approximate_resolution_solutions = {}
    resolution_solve_start = perf_counter()
    for (
        group_index,
        name,
        group_mask,
        coordinate_scale,
        first_direction,
        second_direction,
    ) in resolution_tasks:
        first_solution = output_factor.solve(output_fisher_diagonal * first_direction)
        second_solution = output_factor.solve(output_fisher_diagonal * second_direction)
        approximate_resolution_solutions[(group_index, name)] = (
            first_solution,
            second_solution,
        )
        approximate_resolution_results[name][group_mask] = _gaussian_equivalent_fwhm(
            approximate_averaging_kernel_row_sum[group_mask],
            first_solution[group_mask],
            second_solution[group_mask],
            coordinate_scale,
        )
    resolution_solve_runtime_s = perf_counter() - resolution_solve_start

    row_sum_results = {}
    if averaging_kernel_row_sum_mode != "none":
        row_sum_results["approximate_averaging_kernel_row_sum"] = (
            approximate_averaging_kernel_row_sum
        )
        row_sum_results["averaging_kernel_row_sum_method"] = "diagonal_fisher"
        row_sum_results["averaging_kernel_row_sum_group_count"] = len(group_masks)
        row_sum_results["averaging_kernel_row_sum_factorization_runtime_s"] = (
            row_sum_factorization_runtime_s
        )

    resolution_results = {}
    if averaging_kernel_resolution_mode != "none":
        for name, values in approximate_resolution_results.items():
            resolution_results[f"approximate_averaging_kernel_{name}"] = values
        resolution_results.update(
            averaging_kernel_resolution_method=(
                "diagonal_fisher_gaussian_equivalent_fwhm"
            ),
            averaging_kernel_resolution_definition=(
                "signed_moment_gaussian_equivalent_fwhm"
            ),
            averaging_kernel_resolution_sparse_solve_runtime_s=float(
                resolution_solve_runtime_s
            ),
        )

    if averaging_kernel_row_sum_mode == "matrix_free":
        row_sum_jvp_calls = 0
        row_sum_vjp_calls = 0
        row_sum_jvp_runtime_s = 0.0
        row_sum_vjp_runtime_s = 0.0

        def measurement_hessian_product(direction: np.ndarray) -> np.ndarray:
            nonlocal row_sum_jvp_calls
            nonlocal row_sum_vjp_calls
            nonlocal row_sum_jvp_runtime_s
            nonlocal row_sum_vjp_runtime_s
            jvp_start = perf_counter()
            measurement_direction = np.asarray(
                _matvec(operator, direction / output_mapping)
            ).reshape(-1)
            row_sum_jvp_runtime_s += perf_counter() - jvp_start
            row_sum_jvp_calls += 1
            weighted_direction = inv_Sy @ measurement_direction
            vjp_start = perf_counter()
            result = (
                np.asarray(_rmatvec(operator, weighted_direction)).reshape(-1)
                / output_mapping
            )
            row_sum_vjp_runtime_s += perf_counter() - vjp_start
            row_sum_vjp_calls += 1
            return result

        full_information = LinearOperator(
            (n_state, n_state),
            matvec=lambda direction: (
                measurement_hessian_product(np.asarray(direction).reshape(-1))
                + output_prior_precision @ np.asarray(direction).reshape(-1)
            ),
            dtype=float,
        )
        preconditioner = LinearOperator(
            (n_state, n_state),
            matvec=lambda value: output_factor.solve(np.asarray(value).reshape(-1)),
            dtype=float,
        )
        row_sum_output = np.empty(n_state)
        row_sum_iterations = 0
        row_sum_info_by_group = []
        row_sum_runtime_s = 0.0
        for group_mask, approximate_group_solution in zip(
            group_masks,
            approximate_group_solutions,
            strict=True,
        ):
            group_direction = group_mask.astype(float)
            row_sum_rhs = measurement_hessian_product(group_direction)
            group_iterations = 0

            def count_iteration(_value):
                nonlocal group_iterations
                group_iterations += 1

            row_sum_start = perf_counter()
            group_output, group_info = cg(
                full_information,
                row_sum_rhs,
                x0=approximate_group_solution,
                rtol=averaging_kernel_row_sum_rtol,
                atol=0.0,
                maxiter=averaging_kernel_row_sum_maxiter,
                M=preconditioner,
                callback=count_iteration,
            )
            row_sum_runtime_s += perf_counter() - row_sum_start
            row_sum_output[group_mask] = group_output[group_mask]
            row_sum_iterations += group_iterations
            row_sum_info_by_group.append(int(group_info))
        row_sum_info = max(row_sum_info_by_group, key=abs)
        row_sum_results.update(
            averaging_kernel_row_sum=row_sum_output,
            averaging_kernel_row_sum_method="matrix_free_cg",
            averaging_kernel_row_sum_krylov_info=int(row_sum_info),
            averaging_kernel_row_sum_krylov_iterations=int(row_sum_iterations),
            averaging_kernel_row_sum_jvp_calls=int(row_sum_jvp_calls),
            averaging_kernel_row_sum_vjp_calls=int(row_sum_vjp_calls),
            averaging_kernel_row_sum_jvp_runtime_s=float(row_sum_jvp_runtime_s),
            averaging_kernel_row_sum_vjp_runtime_s=float(row_sum_vjp_runtime_s),
            averaging_kernel_row_sum_runtime_s=float(row_sum_runtime_s),
        )

        if averaging_kernel_resolution_mode == "matrix_free":
            resolution_jvp_calls_start = row_sum_jvp_calls
            resolution_vjp_calls_start = row_sum_vjp_calls
            resolution_jvp_runtime_start = row_sum_jvp_runtime_s
            resolution_vjp_runtime_start = row_sum_vjp_runtime_s
            exact_resolution_results = {
                name: np.full(n_state, np.nan) for name in resolution_coordinates
            }
            resolution_iterations = 0
            resolution_info = []
            resolution_runtime_s = 0.0
            for (
                group_index,
                name,
                group_mask,
                coordinate_scale,
                first_direction,
                second_direction,
            ) in resolution_tasks:
                approximate_first, approximate_second = (
                    approximate_resolution_solutions[(group_index, name)]
                )
                exact_moments = []
                for direction, initial_moment in (
                    (first_direction, approximate_first),
                    (second_direction, approximate_second),
                ):
                    moment_rhs = measurement_hessian_product(direction)
                    moment_iterations = 0

                    def count_moment_iteration(_value):
                        nonlocal moment_iterations
                        moment_iterations += 1

                    moment_start = perf_counter()
                    moment_output, moment_info = cg(
                        full_information,
                        moment_rhs,
                        x0=initial_moment,
                        rtol=averaging_kernel_row_sum_rtol,
                        atol=0.0,
                        maxiter=averaging_kernel_row_sum_maxiter,
                        M=preconditioner,
                        callback=count_moment_iteration,
                    )
                    resolution_runtime_s += perf_counter() - moment_start
                    resolution_iterations += moment_iterations
                    resolution_info.append(int(moment_info))
                    exact_moments.append(moment_output)
                exact_resolution_results[name][group_mask] = _gaussian_equivalent_fwhm(
                    row_sum_output[group_mask],
                    exact_moments[0][group_mask],
                    exact_moments[1][group_mask],
                    coordinate_scale,
                )
            for name, values in exact_resolution_results.items():
                resolution_results[f"averaging_kernel_{name}"] = values
            resolution_results.update(
                averaging_kernel_resolution_method=(
                    "matrix_free_cg_gaussian_equivalent_fwhm"
                ),
                averaging_kernel_resolution_krylov_info=max(
                    resolution_info,
                    key=abs,
                    default=0,
                ),
                averaging_kernel_resolution_krylov_iterations=int(
                    resolution_iterations
                ),
                averaging_kernel_resolution_jvp_calls=int(
                    row_sum_jvp_calls - resolution_jvp_calls_start
                ),
                averaging_kernel_resolution_vjp_calls=int(
                    row_sum_vjp_calls - resolution_vjp_calls_start
                ),
                averaging_kernel_resolution_jvp_runtime_s=float(
                    row_sum_jvp_runtime_s - resolution_jvp_runtime_start
                ),
                averaging_kernel_resolution_vjp_runtime_s=float(
                    row_sum_vjp_runtime_s - resolution_vjp_runtime_start
                ),
                averaging_kernel_resolution_runtime_s=float(resolution_runtime_s),
            )

    posterior_rng = np.random.default_rng(random_seed + 1)
    posterior_sum = np.zeros(n_state)
    posterior_sum_squares = np.zeros(n_state)
    posterior_solve_runtime_s = 0.0
    completed = 0
    while completed < posterior_probe_count:
        batch_size = min(
            posterior_probe_batch_size,
            posterior_probe_count - completed,
        )
        data_probe = _rademacher(posterior_rng, (n_state, batch_size))
        prior_probe = _rademacher(
            posterior_rng,
            (prior_factor.shape[0], batch_size),
        )
        information_sample = (
            np.sqrt(fisher_diagonal)[:, np.newaxis] * data_probe
            + prior_factor.T @ prior_probe
        )
        solve_start = perf_counter()
        covariance_sample = factor.solve(information_sample)
        posterior_solve_runtime_s += perf_counter() - solve_start
        sample = covariance_sample**2
        posterior_sum += np.sum(sample, axis=1)
        posterior_sum_squares += np.sum(sample**2, axis=1)
        completed += batch_size

    posterior_diagonal = posterior_sum / posterior_probe_count
    if posterior_probe_count == 1:
        posterior_standard_error = np.full(n_state, np.nan)
    else:
        posterior_sample_variance = np.maximum(
            (posterior_sum_squares - posterior_probe_count * posterior_diagonal**2)
            / (posterior_probe_count - 1),
            0.0,
        )
        posterior_standard_error = np.sqrt(
            posterior_sample_variance / posterior_probe_count
        )

    return {
        "measurement_information_diagonal": fisher_diagonal,
        "measurement_information_diagonal_standard_error": (fisher_standard_error),
        "solution_covariance_diagonal": posterior_diagonal,
        "solution_covariance_diagonal_standard_error": (posterior_standard_error),
        "fisher_diagonal_probe_count": fisher_probe_count,
        "posterior_diagonal_probe_count": posterior_probe_count,
        "posterior_diagonal_completed_probe_count": posterior_probe_count,
        "fisher_diagonal_vjp_calls": fisher_probe_count,
        "fisher_diagonal_vjp_runtime_s": fisher_vjp_runtime_s,
        "fisher_diagonal_factorization_runtime_s": factorization_runtime_s,
        "fisher_diagonal_solve_runtime_s": posterior_solve_runtime_s,
        **row_sum_results,
        **resolution_results,
    }
