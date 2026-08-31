from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, least_squares, minimize
from scipy.optimize._optimize import MemoizeJac
from scipy.sparse.linalg import LinearOperator

from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval import ForwardModel, Minimizer, RetrievalTarget
from skretrieval.retrieval.erroranalysis import (
    estimate_error,
    estimate_error_from_operator,
    estimate_fisher_diagonal_error_from_operator,
)


class MatrixFreeUnsupportedError(NotImplementedError):
    """Raised when a retrieval cannot provide matrix-free Jacobian products."""


def _operator_matvec(operator, x: np.ndarray) -> np.ndarray:
    if hasattr(operator, "jvp"):
        return np.asarray(operator.jvp(x)).reshape(-1)
    if hasattr(operator, "matvec"):
        return np.asarray(operator.matvec(x)).reshape(-1)
    msg = "Matrix-free Jacobian operator does not provide jvp() or matvec()"
    raise MatrixFreeUnsupportedError(msg)


def _operator_rmatvec(operator, y: np.ndarray) -> np.ndarray:
    if hasattr(operator, "vjp"):
        return np.asarray(operator.vjp(y)).reshape(-1)
    if hasattr(operator, "rmatvec"):
        return np.asarray(operator.rmatvec(y)).reshape(-1)
    msg = "Matrix-free Jacobian operator does not provide vjp() or rmatvec()"
    raise MatrixFreeUnsupportedError(msg)


class _GoodMeasurementOperator:
    """Restrict a Jacobian operator to finite rows of the measured vector."""

    def __init__(self, operator, good_meas: np.ndarray):
        self._operator = operator
        self._good_meas = good_meas
        n_state = getattr(operator, "n_state", None)
        if n_state is None and hasattr(operator, "shape"):
            n_state = operator.shape[1]
        if n_state is None:
            msg = "Matrix-free Jacobian operator does not declare its state size"
            raise MatrixFreeUnsupportedError(msg)
        self.n_state = int(n_state)
        self.shape = (int(np.count_nonzero(good_meas)), self.n_state)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        return _operator_matvec(self._operator, x)[self._good_meas]

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        full = np.zeros(len(self._good_meas))
        full[self._good_meas] = y
        return _operator_rmatvec(self._operator, full)


@dataclass(frozen=True)
class _MeasurementWeighting:
    """Residual whitener ``W`` with ``W.T @ W`` equal to the information matrix."""

    whitener: np.ndarray | sparse.spmatrix
    inverse_covariance: np.ndarray | sparse.spmatrix

    def apply(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(self.whitener @ value).reshape(-1)

    def adjoint(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(self.whitener.T @ value).reshape(-1)

    def downweight(self, scale: np.ndarray) -> _MeasurementWeighting:
        inverse_scale = 1 / np.asarray(scale).reshape(-1)
        # Scaling acts in measurement space, before the residual is whitened.
        if sparse.issparse(self.whitener):
            whitener = self.whitener @ sparse.diags(inverse_scale)
        else:
            whitener = np.asarray(self.whitener) * inverse_scale[np.newaxis, :]
        return type(self)(whitener, whitener.T @ whitener)


@dataclass
class _MatrixFreeProblem:
    apriori_state: np.ndarray
    initial_state: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    inverse_apriori_covariance: np.ndarray | sparse.spmatrix
    prior_whitener: np.ndarray | sparse.spmatrix | None
    measurement: np.ndarray
    weighting: _MeasurementWeighting
    state_scale: np.ndarray
    output_state_mapping: np.ndarray
    averaging_kernel_row_sum_groups: np.ndarray
    averaging_kernel_resolution_coordinates: dict[str, np.ndarray]
    cache: _LinearizedMeasurementCache

    @property
    def solver_apriori_state(self) -> np.ndarray:
        return self.apriori_state / self.state_scale

    @property
    def solver_initial_state(self) -> np.ndarray:
        return self.initial_state / self.state_scale

    @property
    def solver_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        first = self.lower_bound / self.state_scale
        second = self.upper_bound / self.state_scale
        return np.minimum(first, second), np.maximum(first, second)

    @property
    def n_prior_residual(self) -> int:
        return 0 if self.prior_whitener is None else int(self.prior_whitener.shape[0])


class _LinearizedMeasurementCache:
    """Keep one SASKTRAN2 linearization alive for each solver state."""

    def __init__(
        self,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
        good_measurement: np.ndarray,
        state_scale: np.ndarray,
    ) -> None:
        self._forward_model = forward_model
        self._retrieval_target = retrieval_target
        self._good_measurement = good_measurement
        self._state_scale = state_scale
        self._x: np.ndarray | None = None
        self._value: dict | None = None

    def evaluate(self, x: np.ndarray) -> dict:
        x = np.asarray(x).reshape(-1)
        if self._x is not None and np.array_equal(self._x, x):
            return self._value

        # A native SASKTRAN2 linearization can own many gigabytes of orbital-
        # plane derivative workspace.  Drop the previous state's operator
        # before constructing the next one so two complete linearizations are
        # never resident at the same time.  Some measurement-vector operators
        # contain callback cycles, hence the explicit cyclic collection.
        self._x = None
        self._value = None
        gc.collect()

        self._retrieval_target.update_state(self._state_scale * x)

        try:
            linearized_l1 = self._forward_model.calculate_linearized_radiance()
            measurement = self._retrieval_target.matrix_free_measurement_vector(
                linearized_l1
            )
        except (AttributeError, KeyError, NotImplementedError, TypeError) as err:
            msg = f"Could not construct matrix-free measurement products: {err}"
            raise MatrixFreeUnsupportedError(msg) from err

        if "jacobian_operator" not in measurement:
            msg = "Measurement vector did not return a matrix-free Jacobian operator"
            raise MatrixFreeUnsupportedError(msg)

        y = np.asarray(measurement["y"]).reshape(-1)
        if len(y) != len(self._good_measurement):
            msg = "Measurement vector size changed during matrix-free optimization"
            raise MatrixFreeUnsupportedError(msg)

        operator = _GoodMeasurementOperator(
            measurement["jacobian_operator"], self._good_measurement
        )
        if operator.n_state != len(self._state_scale):
            msg = (
                "Matrix-free Jacobian state size does not match the retrieval state: "
                f"{operator.n_state} != {len(self._state_scale)}"
            )
            raise MatrixFreeUnsupportedError(msg)

        value = {"y": y[self._good_measurement], "operator": operator}
        self._x = np.array(x, copy=True)
        self._value = value
        return value


def _measurement_weighting(
    y_error, good_measurement: np.ndarray
) -> _MeasurementWeighting:
    """Construct residual whitening from diagonal, sparse, or dense covariance."""
    n_measurement = int(np.count_nonzero(good_measurement))
    if y_error is None:
        identity = sparse.eye(n_measurement, format="csc")
        return _MeasurementWeighting(identity, identity)

    if np.ndim(y_error) == 1:
        variance = np.asarray(y_error)[good_measurement]
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            msg = "Measurement variances must be finite and positive"
            raise ValueError(msg)
        inverse_variance = 1 / variance
        return _MeasurementWeighting(
            sparse.diags(np.sqrt(inverse_variance), format="csc"),
            sparse.diags(inverse_variance, format="csc"),
        )

    if sparse.issparse(y_error):
        covariance = y_error.tocsc()[good_measurement][:, good_measurement]
        if np.any(~np.isfinite(covariance.data)):
            msg = "Measurement covariance must contain only finite values"
            raise ValueError(msg)
        asymmetry = covariance - covariance.T
        asymmetry.eliminate_zeros()
        if asymmetry.nnz and not np.allclose(asymmetry.data, 0.0):
            msg = "Measurement covariance must be symmetric"
            raise ValueError(msg)
        diagonal = covariance.diagonal()
        off_diagonal = covariance - sparse.diags(diagonal, format="csc")
        off_diagonal.eliminate_zeros()
        if off_diagonal.nnz == 0:
            if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
                msg = "Measurement variances must be finite and positive"
                raise ValueError(msg)
            inverse_variance = 1 / diagonal
            return _MeasurementWeighting(
                sparse.diags(np.sqrt(inverse_variance), format="csc"),
                sparse.diags(inverse_variance, format="csc"),
            )
        component_count, component_label = sparse.csgraph.connected_components(
            covariance,
            directed=False,
        )
        if component_count > 1:
            rows = []
            columns = []
            values = []
            for component in range(component_count):
                indices = np.flatnonzero(component_label == component)
                block = covariance[indices][:, indices].toarray()
                block_weighting = _measurement_weighting(
                    block,
                    np.ones(len(indices), dtype=bool),
                )
                block_whitener = np.asarray(block_weighting.whitener)
                block_rows, block_columns = np.nonzero(block_whitener)
                rows.extend(indices[block_rows])
                columns.extend(indices[block_columns])
                values.extend(block_whitener[block_rows, block_columns])
            whitener = sparse.csc_matrix(
                (values, (rows, columns)),
                shape=(n_measurement, n_measurement),
            )
            return _MeasurementWeighting(whitener, whitener.T @ whitener)
        covariance = covariance.toarray()
    else:
        covariance = np.asarray(y_error)[np.ix_(good_measurement, good_measurement)]

    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (n_measurement, n_measurement):
        msg = "Measurement covariance has an invalid shape"
        raise ValueError(msg)
    if np.any(~np.isfinite(covariance)):
        msg = "Measurement covariance must contain only finite values"
        raise ValueError(msg)
    if not np.allclose(covariance, covariance.T):
        msg = "Measurement covariance must be symmetric"
        raise ValueError(msg)

    try:
        covariance_cholesky = np.linalg.cholesky(covariance)
        whitener = np.linalg.solve(covariance_cholesky, np.eye(n_measurement))
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = (
            np.finfo(float).eps
            * max(covariance.shape)
            * max(1.0, np.max(np.abs(eigenvalues)))
        )
        if np.any(eigenvalues < -tolerance) or not np.any(eigenvalues > tolerance):
            msg = "Measurement covariance must be positive semidefinite"
            raise ValueError(msg) from None
        inverse_sqrt = np.zeros_like(eigenvalues)
        inverse_sqrt[eigenvalues > tolerance] = 1 / np.sqrt(
            eigenvalues[eigenvalues > tolerance]
        )
        whitener = np.diag(inverse_sqrt) @ eigenvectors.T

    return _MeasurementWeighting(whitener, whitener.T @ whitener)


class SciPyMinimizer(Minimizer):
    def __init__(
        self,
        method="trf",
        max_nfev=20,
        ftol=1e-3,
        xtol=None,
        x_scale="jac",
        tr_solver="exact",
        apply_state_scaling=False,
        matrix_free_state_scale: float | np.ndarray | None = None,
        include_bounds=False,
        num_passes=1,
        jacobian_mode="materialized",
        matrix_free_fallback="raise",
        tr_options: dict | None = None,
        matrix_free_diagnostics="full",
        fisher_diagonal_probe_count=32,
        posterior_diagonal_probe_count=1024,
        posterior_diagonal_probe_batch_size=32,
        diagonal_error_random_seed=0,
        averaging_kernel_row_sum_mode="approximate",
        averaging_kernel_resolution_mode="approximate",
        averaging_kernel_row_sum_rtol=1.0e-3,
        averaging_kernel_row_sum_maxiter=30,
        matrix_free_solver="lsmr",
        minimize_options: dict | None = None,
        materialized_jacobian_source="calculate_radiance",
        diagnostic_only=False,
        verbose=2,
        **kwargs,
    ) -> None:
        """
        A minimization wrapper around SciPy's ``least_squares`` function.

        Parameters
        ----------
        method : str, optional
            Minimization method, see scipy.least_squares, by default "trf".
            Recommended to only use "lm" or "trf".
        max_nfev : int, optional
            Maximum function evaluations, see ``scipy.optimize.least_squares``,
            by default 20.
        ftol : float | None, optional
            Tolerance on the cost function, by default 1e-3.
        xtol : float | None, optional
            Tolerance on the change in state. The default, ``None``, disables
            this termination condition.
        x_scale : str | float | np.ndarray, optional
            Internal scaling applied by the minimizer, by default "jac". When
            using ``jacobian_mode="matrix_free"``, ``x_scale="jac"`` is replaced
            by ``1.0`` because SciPy does not support Jacobian scaling for
            ``LinearOperator`` Jacobians.
        tr_solver : str, optional
            For the "trf" method, how to solve the trust region problem, by default "exact"
        apply_state_scaling: bool, optional
            If true, then the state vector is scaled relative to the apriori in the solver, useful
            when the state vector elements are of largely varying magnitudes and you have a well
            specified prior, by default False
        matrix_free_state_scale: float or numpy.ndarray, optional
            Explicit positive diagonal change of variables for matrix-free
            solvers. The physical state is ``matrix_free_state_scale * z``
            where ``z`` is the solver coordinate. This changes optimizer
            conditioning without changing the retrieval objective. It is
            mutually exclusive with ``apply_state_scaling``.
        include_bounds : bool, optional
            If true, bounds are included inside the solver, by default False.
        num_passes : int, optional
            Number of passes through the minimizer. Between passes the noise
            covariance is adjusted so measurements with large residuals receive
            less weight, by default 1.
        jacobian_mode : str, optional
            Jacobian strategy. ``"materialized"`` keeps the existing dense/sparse
            Jacobian path, ``"matrix_free"`` uses SASKTRAN2 JVP/VJP products with
            SciPy's LSMR trust-region solver, and ``"auto"`` tries matrix-free
            before falling back according to ``matrix_free_fallback``.
        matrix_free_fallback : str, optional
            Either ``"materialized"`` or ``"raise"``, by default ``"raise"``.
            Controls what happens when matrix-free products are unavailable for
            the current forward model, state vector, or measurement vector.
            ``jacobian_mode="auto"`` always falls back to the materialized path.
        tr_options : dict | None, optional
            Trust-region solver options passed to
            :py:func:`scipy.optimize.least_squares`. For matrix-free LSMR,
            options such as ``{"maxiter": 10}`` can cap inner solver
            iterations and directly reduce JVP/VJP calls.
        matrix_free_diagnostics : str, optional
            Diagnostic strategy for matrix-free retrievals. ``"full"`` forms
            covariance and averaging-kernel diagnostics by repeated products.
            ``"fisher_diagonal"`` estimates the measurement-information
            diagonal with VJP-only randomized probes, retains the complete
            sparse prior precision, and returns an approximate posterior
            covariance diagonal without dense state matrices.
            ``"none"`` skips those post-solve diagnostics and avoids the extra
            product calls.
        fisher_diagonal_probe_count : int, optional
            Number of measurement-space VJP probes used by the fast diagonal
            diagnostic, by default 32.
        posterior_diagonal_probe_count : int, optional
            Number of inexpensive sparse inverse probes used after estimating
            the Fisher diagonal, by default 1024.
        posterior_diagonal_probe_batch_size : int, optional
            Number of sparse inverse right-hand sides solved together, by
            default 32.
        diagonal_error_random_seed : int, optional
            Deterministic seed for both randomized diagonal estimators, by
            default 0.
        averaging_kernel_row_sum_mode : str, optional
            ``"approximate"`` returns within-state-element row sums from the
            diagonal-Fisher information matrix at negligible extra cost.
            ``"matrix_free"`` solves for full-Hessian row sums with
            preconditioned CG, and ``"none"`` disables the diagnostic. The
            default is ``"approximate"``.
        averaging_kernel_resolution_mode : str, optional
            ``"approximate"`` computes Gaussian-equivalent FWHM resolution
            from sparse diagonal-Fisher averaging-kernel moments.
            ``"matrix_free"`` evaluates the same moments with full-Hessian
            JVP/VJP products and requires matrix-free row sums. ``"none"``
            disables resolution diagnostics. The default is ``"approximate"``.
        averaging_kernel_row_sum_rtol : float, optional
            Relative tolerance for the optional matrix-free row-sum solve, by
            default 1e-3.
        averaging_kernel_row_sum_maxiter : int, optional
            Maximum preconditioned CG iterations for matrix-free row sums, by
            default 30.
        matrix_free_solver : str, optional
            Matrix-free optimizer to use. ``"lsmr"`` uses
            :py:func:`scipy.optimize.least_squares` with a
            :class:`scipy.sparse.linalg.LinearOperator`. ``"lbfgsb"`` uses
            gradient-only L-BFGS-B with VJP products.
        minimize_options : dict | None, optional
            Options passed to :py:func:`scipy.optimize.minimize` when
            ``matrix_free_solver="lbfgsb"``.
        materialized_jacobian_source : str, optional
            Forward-model source used by the dense/sparse Jacobian path.
            ``"calculate_radiance"`` keeps the legacy SASKTRAN2 weighting
            function path, while ``"linearization"`` calls
            ``Engine.linearize(...).jacobian`` and materializes the Jacobian
            through SASKTRAN2's linearization object.
        diagnostic_only : bool, optional
            Evaluate matrix-free diagnostics at the supplied initial state
            without taking an optimization step, by default False.
        verbose : int, optional
            SciPy optimizer reporting level: 0 is silent, 1 reports
            termination, and 2 reports every iteration. The default is 2.
        """
        self._method = method
        self._ftol = ftol
        self._xtol = xtol
        self._max_nfev = max_nfev
        self._x_scale = x_scale
        self._tr_solver = tr_solver
        self._include_bounds = include_bounds
        self._num_passes = num_passes
        self._jacobian_mode = jacobian_mode
        self._matrix_free_fallback = matrix_free_fallback
        self._tr_options = dict(tr_options) if tr_options is not None else {}
        self._matrix_free_diagnostics = matrix_free_diagnostics
        self._fisher_diagonal_probe_count = fisher_diagonal_probe_count
        self._posterior_diagonal_probe_count = posterior_diagonal_probe_count
        self._posterior_diagonal_probe_batch_size = posterior_diagonal_probe_batch_size
        self._diagonal_error_random_seed = diagonal_error_random_seed
        self._averaging_kernel_row_sum_mode = averaging_kernel_row_sum_mode
        self._averaging_kernel_resolution_mode = averaging_kernel_resolution_mode
        self._averaging_kernel_row_sum_rtol = averaging_kernel_row_sum_rtol
        self._averaging_kernel_row_sum_maxiter = averaging_kernel_row_sum_maxiter
        self._matrix_free_solver = matrix_free_solver
        self._minimize_options = (
            dict(minimize_options) if minimize_options is not None else {}
        )
        self._materialized_jacobian_source = materialized_jacobian_source
        self._diagnostic_only = bool(diagnostic_only)
        self._verbose = verbose

        self._apply_state_scaling = apply_state_scaling
        self._matrix_free_state_scale = matrix_free_state_scale

        self._kwargs = kwargs

        self._validate_options()

    def _validate_options(self) -> None:
        if self._apply_state_scaling and self._matrix_free_state_scale is not None:
            msg = (
                "apply_state_scaling and matrix_free_state_scale are mutually exclusive"
            )
            raise ValueError(msg)
        if self._jacobian_mode not in {"materialized", "auto", "matrix_free"}:
            msg = "jacobian_mode must be 'materialized', 'auto', or 'matrix_free'"
            raise ValueError(msg)
        if self._matrix_free_fallback not in {"materialized", "raise"}:
            msg = "matrix_free_fallback must be 'materialized' or 'raise'"
            raise ValueError(msg)
        if self._matrix_free_diagnostics not in {
            "full",
            "fisher_diagonal",
            "none",
        }:
            msg = "matrix_free_diagnostics must be 'full', 'fisher_diagonal', or 'none'"
            raise ValueError(msg)
        diagonal_counts = {
            "fisher_diagonal_probe_count": self._fisher_diagonal_probe_count,
            "posterior_diagonal_probe_count": (self._posterior_diagonal_probe_count),
            "posterior_diagonal_probe_batch_size": (
                self._posterior_diagonal_probe_batch_size
            ),
        }
        for name, value in diagonal_counts.items():
            if isinstance(value, bool | np.bool_) or int(value) != value or value < 1:
                msg = f"{name} must be a positive integer"
                raise ValueError(msg)
        if (
            isinstance(self._diagonal_error_random_seed, bool | np.bool_)
            or int(self._diagonal_error_random_seed) != self._diagonal_error_random_seed
        ):
            msg = "diagonal_error_random_seed must be an integer"
            raise ValueError(msg)
        if self._averaging_kernel_row_sum_mode not in {
            "approximate",
            "matrix_free",
            "none",
        }:
            msg = (
                "averaging_kernel_row_sum_mode must be 'approximate', "
                "'matrix_free', or 'none'"
            )
            raise ValueError(msg)
        if self._averaging_kernel_resolution_mode not in {
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
            self._averaging_kernel_resolution_mode == "matrix_free"
            and self._averaging_kernel_row_sum_mode != "matrix_free"
        ):
            msg = (
                "matrix-free averaging-kernel resolution requires "
                "averaging_kernel_row_sum_mode='matrix_free'"
            )
            raise ValueError(msg)
        if (
            not np.isfinite(self._averaging_kernel_row_sum_rtol)
            or self._averaging_kernel_row_sum_rtol <= 0
        ):
            msg = "averaging_kernel_row_sum_rtol must be finite and positive"
            raise ValueError(msg)
        if (
            isinstance(self._averaging_kernel_row_sum_maxiter, bool | np.bool_)
            or int(self._averaging_kernel_row_sum_maxiter)
            != self._averaging_kernel_row_sum_maxiter
            or self._averaging_kernel_row_sum_maxiter < 1
        ):
            msg = "averaging_kernel_row_sum_maxiter must be a positive integer"
            raise ValueError(msg)
        if self._matrix_free_solver not in {"lsmr", "lbfgsb"}:
            msg = "matrix_free_solver must be 'lsmr' or 'lbfgsb'"
            raise ValueError(msg)
        if self._materialized_jacobian_source not in {
            "calculate_radiance",
            "linearization",
        }:
            msg = (
                "materialized_jacobian_source must be 'calculate_radiance' "
                "or 'linearization'"
            )
            raise ValueError(msg)
        if self._verbose not in {0, 1, 2}:
            msg = "verbose must be 0, 1, or 2"
            raise ValueError(msg)
        if self._num_passes < 1:
            msg = "num_passes must be at least 1"
            raise ValueError(msg)

    def _least_squares_tr_options(self) -> dict:
        tr_options = dict(self._tr_options)
        if self._method == "trf":
            tr_options.setdefault("regularize", False)
        else:
            tr_options.pop("regularize", None)
        return tr_options

    def _calculate_materialized_radiance(self, forward_model: ForwardModel):
        if self._materialized_jacobian_source == "linearization":
            if not hasattr(forward_model, "calculate_materialized_linearized_radiance"):
                msg = (
                    "Forward model does not provide "
                    "calculate_materialized_linearized_radiance()"
                )
                raise NotImplementedError(msg)
            return forward_model.calculate_materialized_linearized_radiance()
        return forward_model.calculate_radiance()

    def retrieve(
        self,
        measurement_l1: RadianceBase,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
    ):
        if self._jacobian_mode == "materialized":
            return self._retrieve_materialized(
                measurement_l1, forward_model, retrieval_target
            )

        initial_state = np.array(retrieval_target.state_vector(), copy=True)
        try:
            return self._retrieve_matrix_free(
                measurement_l1, forward_model, retrieval_target
            )
        except MatrixFreeUnsupportedError as err:
            retrieval_target.update_state(initial_state)
            if (
                self._jacobian_mode == "auto"
                or self._matrix_free_fallback == "materialized"
            ):
                warnings.warn(
                    "Matrix-free retrieval is unavailable; falling back to the "
                    f"materialized Jacobian path. {err}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._retrieve_materialized(
                    measurement_l1, forward_model, retrieval_target
                )
            raise

    def _retrieve_materialized(
        self,
        measurement_l1: RadianceBase,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
    ):
        ### Get the prior values
        x_a = retrieval_target.apriori_state()
        initial_guess = retrieval_target.state_vector()
        lb = retrieval_target.lower_bound()
        ub = retrieval_target.upper_bound()

        inv_Sa = retrieval_target.inverse_apriori_covariance()

        if inv_Sa is None:
            # No apriori covariance/regularization
            # Use initial guess to make the matrices as x_a might be None as well
            inv_Sa = np.zeros((len(initial_guess), len(initial_guess)))

        if x_a is None:
            x_a = np.zeros_like(initial_guess)

        ### Get the measurement values
        y_meas_dict = retrieval_target.measurement_vector(measurement_l1)

        y_meas = y_meas_dict["y"]
        good_meas = ~np.isnan(y_meas)
        y_meas = y_meas[good_meas]

        if "y_error" in y_meas_dict:
            # Have measurement error
            if len(np.shape(y_meas_dict["y_error"])) == 1:
                # Only supplied is the diagonal of the error elements
                Sy = sparse.csc_matrix(
                    sparse.diags(y_meas_dict["y_error"][good_meas], 0)
                )
                inv_Sy = sparse.csc_matrix(
                    sparse.diags(1 / y_meas_dict["y_error"][good_meas], 0)
                )
                y_scaler_inv = np.linalg.cholesky(inv_Sy)
            elif sparse.issparse(y_meas_dict["y_error"]):
                Sy = y_meas_dict["y_error"][np.ix_(good_meas, good_meas)]

                rows = Sy.indices  # Row indices of non-zero elements
                cols = np.repeat(
                    np.arange(Sy.shape[1]), np.diff(Sy.indptr)
                )  # Column indices of non-zero elements

                # Check if all non-zero elements are on the diagonal
                is_diagonal = np.all(rows == cols)

                if is_diagonal:
                    # Ensure there are no zeros on the diagonal
                    if np.any(Sy.data == 0):
                        msg = "Cannot invert a matrix with zeros on the diagonal."
                        raise ZeroDivisionError(msg)

                    # Compute the inverse of the diagonal elements
                    inv_data = 1.0 / Sy.data

                    # Create the inverse matrix using the same indices and indptr
                    inv_Sy = sparse.csc_matrix(
                        (inv_data, Sy.indices, Sy.indptr), shape=Sy.shape
                    )

                    y_scaler_inv = sparse.csc_matrix(
                        (np.sqrt(inv_data), Sy.indices, Sy.indptr), shape=Sy.shape
                    )
            else:
                Sy = y_meas_dict["y_error"][np.ix_(good_meas, good_meas)]
                inv_Sy = np.linalg.inv(Sy)
                y_scaler_inv = np.linalg.cholesky(inv_Sy)
        else:
            # No user supplied error, use identity matrix
            Sy = sparse.csc_matrix(sparse.eye(len(y_meas), len(y_meas)))
            inv_Sy = sparse.csc_matrix(sparse.eye(len(y_meas), len(y_meas)))

            y_scaler_inv = np.linalg.cholesky(inv_Sy)

        prior_whitener = retrieval_target.prior_precision_factor()
        if sparse.issparse(prior_whitener):
            prior_whitener = prior_whitener.astype(float).tocsr()
        else:
            prior_whitener = np.asarray(prior_whitener, dtype=float)
        if prior_whitener.ndim != 2 or prior_whitener.shape[1] != len(initial_guess):
            msg = (
                "A priori precision factor must be two-dimensional with "
                "one column per state element"
            )
            raise ValueError(msg)

        if self._apply_state_scaling:
            x_scaler_inv = np.diag(1 / x_a)
            x_scaler = np.diag(x_a)
        else:
            x_scaler_inv = np.eye(len(x_a))
            x_scaler = np.eye(len(x_a))

        x_a = x_scaler_inv @ x_a

        def residual_fun(x):
            retrieval_target.update_state(x_scaler @ x)

            y_ret_dict = retrieval_target.measurement_vector(
                self._calculate_materialized_radiance(forward_model)
            )

            K = y_ret_dict["jacobian"]

            K = y_scaler_inv @ K[good_meas, :] @ x_scaler
            y_ret = y_scaler_inv @ y_ret_dict["y"][good_meas]

            # First part of residuals is from y, y_meas - y_ret, and jacobian K
            res = y_ret - y_scaler_inv @ y_meas
            # Second part of residuals is x-x_a, with identity jacobian in scaled space
            res_x = prior_whitener @ x_scaler @ (x - x_a)
            K_x = prior_whitener @ x_scaler
            if sparse.issparse(K_x):
                K_x = K_x.toarray()

            # To match the cost of the standard "Rodgers" minimizer we have to scale by the number of measurements,
            # and also multiply by 2 since the scipy least squares does 0.5 * res.T @ res
            n = len(res) / 2

            return np.concatenate((res, res_x)) / np.sqrt(n), np.vstack(
                [K, K_x]
            ) / np.sqrt(n)

        fun = MemoizeJac(residual_fun)
        jac = fun.derivative

        bounds = (
            (x_scaler_inv @ lb, x_scaler_inv @ ub)
            if self._include_bounds
            else (np.ones_like(lb) * (-np.inf), np.ones_like(ub) * np.inf)
        )

        results = {}

        for _ in range(self._num_passes):
            results["minimizer"] = least_squares(
                fun,
                x0=x_scaler_inv @ initial_guess,
                jac=jac,
                x_scale=self._x_scale,
                verbose=self._verbose,
                tr_solver=self._tr_solver,
                max_nfev=self._max_nfev,
                tr_options=self._least_squares_tr_options(),
                method=self._method,
                xtol=self._xtol,
                ftol=self._ftol,
                bounds=bounds,
                **self._kwargs,
            )

            y_ret_dict = retrieval_target.measurement_vector(
                self._calculate_materialized_radiance(forward_model)
            )

            meas_resid = y_meas - y_ret_dict["y"][good_meas]

            median_resid = np.median(np.abs(meas_resid))
            # Adjust the scaler based on the residual fractions
            scaler = (
                np.abs(meas_resid) / median_resid
                if median_resid > 0
                else np.ones_like(meas_resid)
            )
            scaler[scaler < 1] = 1

            # Rescale the measurement errrors
            y_scaler_inv = sparse.diags(np.sqrt(inv_Sy.diagonal() / scaler**2))

            initial_guess = retrieval_target.state_vector()

        K = y_ret_dict["jacobian"][good_meas, :]

        results.update(estimate_error(K, Sy, inv_Sy, inv_Sa))

        return retrieval_target.state_vector_error_output(results)

    def _retrieve_matrix_free(
        self,
        measurement_l1: RadianceBase,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
    ):
        if not hasattr(forward_model, "calculate_linearized_radiance"):
            msg = "Forward model does not provide calculate_linearized_radiance()"
            raise MatrixFreeUnsupportedError(msg)
        solver_start = perf_counter()
        if self._diagnostic_only:
            if self._matrix_free_diagnostics == "none":
                msg = "diagnostic_only requires matrix_free_diagnostics"
                raise ValueError(msg)
            problem = self._matrix_free_problem(
                measurement_l1,
                forward_model,
                retrieval_target,
                construct_prior_whitener=True,
            )
            results = self._matrix_free_diagnostic_results(
                problem,
                problem.weighting,
                problem.solver_initial_state,
            )
            results.update(
                minimizer=OptimizeResult(
                    x=problem.solver_initial_state,
                    fun=np.nan,
                    cost=np.nan,
                    jac=np.full_like(problem.solver_initial_state, np.nan),
                    optimality=np.nan,
                    nit=0,
                    nfev=0,
                    njev=0,
                    success=True,
                    message="Diagnostic-only evaluation at the supplied state",
                ),
                jvp_calls=0,
                vjp_calls=0,
                jvp_runtime_s=0.0,
                vjp_runtime_s=0.0,
                diagnostic_only=True,
            )
            results["solver_wall_time_s"] = perf_counter() - solver_start
            return retrieval_target.state_vector_error_output(results)
        if self._matrix_free_solver == "lbfgsb":
            results = self._retrieve_matrix_free_lbfgsb(
                measurement_l1, forward_model, retrieval_target
            )
        else:
            results = self._retrieve_matrix_free_lsmr(
                measurement_l1, forward_model, retrieval_target
            )
        results["solver_wall_time_s"] = perf_counter() - solver_start
        return retrieval_target.state_vector_error_output(results)

    def _matrix_free_problem(
        self,
        measurement_l1: RadianceBase,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
        *,
        construct_prior_whitener: bool = True,
    ) -> _MatrixFreeProblem:
        initial_state = np.asarray(
            retrieval_target.state_vector(), dtype=float
        ).reshape(-1)
        apriori_state = retrieval_target.apriori_state()
        if apriori_state is None:
            apriori_state = np.zeros_like(initial_state)
        else:
            apriori_state = np.asarray(apriori_state, dtype=float).reshape(-1)

        inverse_apriori_covariance = retrieval_target.inverse_apriori_covariance()
        if inverse_apriori_covariance is None:
            inverse_apriori_covariance = np.zeros(
                (len(initial_state), len(initial_state))
            )
        elif sparse.issparse(inverse_apriori_covariance):
            inverse_apriori_covariance = inverse_apriori_covariance.astype(
                float
            ).tocsr()
        else:
            inverse_apriori_covariance = np.asarray(
                inverse_apriori_covariance, dtype=float
            )

        if apriori_state.shape != initial_state.shape:
            msg = "A priori state size does not match the retrieval state"
            raise ValueError(msg)
        if inverse_apriori_covariance.shape != (
            len(initial_state),
            len(initial_state),
        ):
            msg = "A priori inverse covariance size does not match the retrieval state"
            raise ValueError(msg)

        measurement = retrieval_target.observed_measurement_vector(measurement_l1)
        y_meas = np.asarray(measurement["y"], dtype=float).reshape(-1)
        good_measurement = np.isfinite(y_meas)
        if not np.any(good_measurement):
            msg = "Retrieval measurement vector contains no finite measurements"
            raise ValueError(msg)
        y_meas = y_meas[good_measurement]
        weighting = _measurement_weighting(measurement.get("y_error"), good_measurement)

        state_scale = np.ones_like(apriori_state)
        if self._matrix_free_state_scale is not None:
            explicit_scale = np.asarray(self._matrix_free_state_scale, dtype=float)
            if explicit_scale.ndim == 0:
                state_scale.fill(float(explicit_scale))
            elif explicit_scale.shape == initial_state.shape:
                state_scale = np.array(explicit_scale, copy=True)
            else:
                msg = (
                    "matrix_free_state_scale must be scalar or match the state "
                    f"shape {initial_state.shape}"
                )
                raise ValueError(msg)
            if np.any(~np.isfinite(state_scale)) or np.any(state_scale <= 0):
                msg = "matrix_free_state_scale must contain finite positive values"
                raise ValueError(msg)
        elif self._apply_state_scaling:
            if np.any(~np.isfinite(apriori_state)) or np.any(apriori_state == 0):
                msg = (
                    "apply_state_scaling requires finite, nonzero a priori state values"
                )
                raise ValueError(msg)
            state_scale = np.array(apriori_state, copy=True)

        lower_bound = np.asarray(retrieval_target.lower_bound(), dtype=float).reshape(
            -1
        )
        upper_bound = np.asarray(retrieval_target.upper_bound(), dtype=float).reshape(
            -1
        )
        if (
            lower_bound.shape != initial_state.shape
            or upper_bound.shape != initial_state.shape
        ):
            msg = "Retrieval bounds size does not match the retrieval state"
            raise ValueError(msg)

        prior_whitener = None
        if construct_prior_whitener:
            prior_whitener = retrieval_target.prior_precision_factor()
            if sparse.issparse(prior_whitener):
                prior_whitener = prior_whitener.astype(float).tocsr()
            else:
                prior_whitener = np.asarray(prior_whitener, dtype=float)
            if prior_whitener.ndim != 2 or prior_whitener.shape[1] != len(
                initial_state
            ):
                msg = (
                    "A priori precision factor must be two-dimensional with "
                    "one column per state element"
                )
                raise ValueError(msg)

        output_state_mapping = np.asarray(
            retrieval_target.output_state_derivative_by_retrieval_state(),
            dtype=float,
        ).reshape(-1)
        if output_state_mapping.shape != initial_state.shape:
            msg = "Output-state derivative size does not match the retrieval state"
            raise ValueError(msg)
        averaging_kernel_row_sum_groups = np.asarray(
            retrieval_target.averaging_kernel_row_sum_groups(),
        ).reshape(-1)
        if averaging_kernel_row_sum_groups.shape != initial_state.shape:
            msg = "Averaging-kernel row-sum groups must match the retrieval state"
            raise ValueError(msg)
        averaging_kernel_resolution_coordinates = {
            name: np.asarray(values, dtype=float).reshape(-1)
            for name, values in (
                retrieval_target.averaging_kernel_resolution_coordinates().items()
            )
        }
        for name, values in averaging_kernel_resolution_coordinates.items():
            if values.shape != initial_state.shape:
                msg = (
                    f"Averaging-kernel resolution coordinate {name} must match "
                    "the retrieval state"
                )
                raise ValueError(msg)

        cache = _LinearizedMeasurementCache(
            forward_model,
            retrieval_target,
            good_measurement,
            state_scale,
        )
        return _MatrixFreeProblem(
            apriori_state=apriori_state,
            initial_state=initial_state,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            inverse_apriori_covariance=inverse_apriori_covariance,
            prior_whitener=prior_whitener,
            measurement=y_meas,
            weighting=weighting,
            state_scale=state_scale,
            output_state_mapping=output_state_mapping,
            averaging_kernel_row_sum_groups=averaging_kernel_row_sum_groups,
            averaging_kernel_resolution_coordinates=(
                averaging_kernel_resolution_coordinates
            ),
            cache=cache,
        )

    @staticmethod
    def _outlier_scale(measurement_residual: np.ndarray) -> np.ndarray:
        median_residual = np.median(np.abs(measurement_residual))
        if median_residual <= 0:
            return np.ones_like(measurement_residual)
        scale = np.abs(measurement_residual) / median_residual
        scale[scale < 1] = 1
        return scale

    def _matrix_free_diagnostic_results(
        self,
        problem: _MatrixFreeProblem,
        weighting: _MeasurementWeighting,
        solver_state: np.ndarray,
    ) -> dict:
        evaluated = problem.cache.evaluate(solver_state)
        if self._matrix_free_diagnostics == "fisher_diagonal":
            return estimate_fisher_diagonal_error_from_operator(
                evaluated["operator"],
                weighting.inverse_covariance,
                problem.inverse_apriori_covariance,
                prior_precision_factor=problem.prior_whitener,
                output_state_derivative_by_retrieval_state=(
                    problem.output_state_mapping
                ),
                averaging_kernel_row_sum_groups=(
                    problem.averaging_kernel_row_sum_groups
                ),
                averaging_kernel_resolution_coordinates=(
                    problem.averaging_kernel_resolution_coordinates
                ),
                fisher_probe_count=self._fisher_diagonal_probe_count,
                posterior_probe_count=self._posterior_diagonal_probe_count,
                posterior_probe_batch_size=(self._posterior_diagonal_probe_batch_size),
                random_seed=self._diagonal_error_random_seed,
                averaging_kernel_row_sum_mode=(self._averaging_kernel_row_sum_mode),
                averaging_kernel_resolution_mode=(
                    self._averaging_kernel_resolution_mode
                ),
                averaging_kernel_row_sum_rtol=(self._averaging_kernel_row_sum_rtol),
                averaging_kernel_row_sum_maxiter=(
                    self._averaging_kernel_row_sum_maxiter
                ),
            )
        return estimate_error_from_operator(
            evaluated["operator"],
            weighting.inverse_covariance,
            problem.inverse_apriori_covariance,
        )

    def _retrieve_matrix_free_lsmr(
        self,
        measurement_l1: RadianceBase,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
    ):
        if self._method not in {"trf", "dogbox"}:
            msg = "Matrix-free SciPy retrieval requires method='trf' or method='dogbox'"
            raise MatrixFreeUnsupportedError(msg)
        problem = self._matrix_free_problem(
            measurement_l1,
            forward_model,
            retrieval_target,
            construct_prior_whitener=True,
        )
        cache = problem.cache
        weighting = problem.weighting
        # Match the normalization used by the existing materialized residual.
        normalization = np.sqrt(len(problem.measurement) / 2)
        solver_apriori = problem.solver_apriori_state
        objective_history = []
        measurement_objective_history = []
        prior_objective_history = []
        evaluation_runtime_history_s = []
        product_statistics = {
            "jvp_calls": 0,
            "vjp_calls": 0,
            "jvp_runtime_s": 0.0,
            "vjp_runtime_s": 0.0,
        }

        def residual_fun(x):
            evaluation_start = perf_counter()
            evaluated = cache.evaluate(x)
            measurement_residual = weighting.apply(evaluated["y"] - problem.measurement)
            prior_residual = problem.prior_whitener @ (
                problem.state_scale * (x - solver_apriori)
            )
            residual = (
                np.concatenate((measurement_residual, prior_residual)) / normalization
            )
            objective_history.append(float(0.5 * (residual @ residual)))
            measurement_objective_history.append(
                float(
                    0.5
                    * (measurement_residual @ measurement_residual)
                    / normalization**2
                )
            )
            prior_objective_history.append(
                float(0.5 * (prior_residual @ prior_residual) / normalization**2)
            )
            evaluation_runtime_history_s.append(perf_counter() - evaluation_start)
            return residual

        def jacobian_fun(x):
            # The LinearOperator must stay tied to the state where SciPy requested it.
            anchor_x = np.array(x, copy=True)

            def operator():
                return cache.evaluate(anchor_x)["operator"]

            def matvec(dx):
                product_start = perf_counter()
                dx = np.asarray(dx).reshape(-1)
                target_direction = problem.state_scale * dx
                measurement_part = weighting.apply(operator().matvec(target_direction))
                prior_part = problem.prior_whitener @ target_direction
                result = np.concatenate((measurement_part, prior_part)) / normalization
                product_statistics["jvp_calls"] += 1
                product_statistics["jvp_runtime_s"] += perf_counter() - product_start
                return result

            def rmatvec(cotangent):
                product_start = perf_counter()
                cotangent = np.asarray(cotangent).reshape(-1)
                measurement_cotangent = cotangent[: len(problem.measurement)]
                prior_cotangent = cotangent[len(problem.measurement) :]
                measurement_part = operator().rmatvec(
                    weighting.adjoint(measurement_cotangent)
                )
                prior_part = problem.prior_whitener.T @ prior_cotangent
                result = (
                    problem.state_scale * (measurement_part + prior_part)
                ) / normalization
                product_statistics["vjp_calls"] += 1
                product_statistics["vjp_runtime_s"] += perf_counter() - product_start
                return result

            return LinearOperator(
                (
                    len(problem.measurement) + problem.n_prior_residual,
                    len(problem.apriori_state),
                ),
                matvec=matvec,
                rmatvec=rmatvec,
                dtype=float,
            )

        results = {}
        effective_x_scale = (
            1.0
            if isinstance(self._x_scale, str) and self._x_scale == "jac"
            else self._x_scale
        )
        solver_initial = problem.solver_initial_state
        if self._include_bounds:
            bounds = problem.solver_bounds
        else:
            bounds = (
                np.full_like(solver_initial, -np.inf),
                np.full_like(solver_initial, np.inf),
            )

        for pass_index in range(self._num_passes):
            results["minimizer"] = least_squares(
                residual_fun,
                x0=solver_initial,
                jac=jacobian_fun,
                x_scale=effective_x_scale,
                verbose=self._verbose,
                tr_solver="lsmr",
                max_nfev=self._max_nfev,
                tr_options=self._least_squares_tr_options(),
                method=self._method,
                xtol=self._xtol,
                ftol=self._ftol,
                bounds=bounds,
                **self._kwargs,
            )

            evaluated = cache.evaluate(results["minimizer"].x)
            if pass_index + 1 < self._num_passes:
                weighting = weighting.downweight(
                    self._outlier_scale(problem.measurement - evaluated["y"])
                )
                solver_initial = np.array(results["minimizer"].x, copy=True)

        if self._matrix_free_diagnostics != "none":
            results.update(
                self._matrix_free_diagnostic_results(
                    problem, weighting, results["minimizer"].x
                )
            )

        results["objective_history"] = np.asarray(objective_history)
        results["measurement_objective_history"] = np.asarray(
            measurement_objective_history
        )
        results["prior_objective_history"] = np.asarray(prior_objective_history)
        results["evaluation_runtime_history_s"] = np.asarray(
            evaluation_runtime_history_s
        )
        results.update(product_statistics)

        return results

    def _retrieve_matrix_free_lbfgsb(
        self,
        measurement_l1: RadianceBase,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
    ):
        unsupported_kwargs = set(self._kwargs) - {"callback", "tol"}
        if unsupported_kwargs:
            names = ", ".join(sorted(unsupported_kwargs))
            msg = (
                "L-BFGS-B does not accept least_squares keyword argument(s): "
                f"{names}. Use minimize_options for optimizer options."
            )
            raise ValueError(msg)

        problem = self._matrix_free_problem(
            measurement_l1,
            forward_model,
            retrieval_target,
            construct_prior_whitener=(
                self._matrix_free_diagnostics == "fisher_diagonal"
            ),
        )
        cache = problem.cache
        weighting = problem.weighting
        # L-BFGS-B receives the scalar cost, so this is the squared equivalent
        # of the least-squares residual normalization above.
        normalization = len(problem.measurement) / 2
        objective_history = []
        measurement_objective_history = []
        prior_objective_history = []
        gradient_inf_norm_history = []
        evaluation_runtime_history_s = []
        vjp_calls = 0
        vjp_runtime_s = 0.0

        def objective_and_gradient(x):
            nonlocal vjp_calls, vjp_runtime_s
            evaluation_start = perf_counter()
            evaluated = cache.evaluate(x)
            operator = evaluated["operator"]

            measurement_residual = weighting.apply(evaluated["y"] - problem.measurement)
            prior_cost_raw, prior_gradient = retrieval_target.prior_cost_and_gradient()
            measurement_cost = (
                0.5 * (measurement_residual @ measurement_residual) / normalization
            )
            prior_cost = prior_cost_raw / normalization
            cost = measurement_cost + prior_cost

            vjp_start = perf_counter()
            measurement_part = operator.rmatvec(weighting.adjoint(measurement_residual))
            vjp_runtime_s += perf_counter() - vjp_start
            vjp_calls += 1
            grad = (
                problem.state_scale * (measurement_part + prior_gradient)
            ) / normalization

            objective_history.append(float(cost))
            measurement_objective_history.append(float(measurement_cost))
            prior_objective_history.append(float(prior_cost))
            gradient_inf_norm_history.append(float(np.linalg.norm(grad, ord=np.inf)))
            evaluation_runtime_history_s.append(perf_counter() - evaluation_start)

            return cost, grad

        bounds = list(zip(*problem.solver_bounds)) if self._include_bounds else None

        results = {}

        minimize_options = {
            "maxiter": self._max_nfev,
            "maxfun": self._max_nfev,
        }
        if self._ftol is not None:
            minimize_options["ftol"] = self._ftol
        minimize_options.update(self._minimize_options)

        minimize_kwargs = {}
        for key in ("callback", "tol"):
            if key in self._kwargs:
                minimize_kwargs[key] = self._kwargs[key]

        solver_initial = problem.solver_initial_state
        for pass_index in range(self._num_passes):
            results["minimizer"] = minimize(
                objective_and_gradient,
                x0=solver_initial,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options=minimize_options,
                **minimize_kwargs,
            )

            # Expose the fields downstream code expects from least_squares.
            results["minimizer"].cost = results["minimizer"].fun
            results["minimizer"].optimality = np.linalg.norm(
                results["minimizer"].jac, ord=np.inf
            )
            if "njev" not in results["minimizer"]:
                results["minimizer"].njev = results["minimizer"].nfev

            evaluated = cache.evaluate(results["minimizer"].x)
            if pass_index + 1 < self._num_passes:
                weighting = weighting.downweight(
                    self._outlier_scale(problem.measurement - evaluated["y"])
                )
                solver_initial = np.array(results["minimizer"].x, copy=True)

        if self._matrix_free_diagnostics != "none":
            results.update(
                self._matrix_free_diagnostic_results(
                    problem, weighting, results["minimizer"].x
                )
            )

        results["objective_history"] = np.asarray(objective_history)
        results["measurement_objective_history"] = np.asarray(
            measurement_objective_history
        )
        results["prior_objective_history"] = np.asarray(prior_objective_history)
        results["gradient_inf_norm_history"] = np.asarray(gradient_inf_norm_history)
        results["evaluation_runtime_history_s"] = np.asarray(
            evaluation_runtime_history_s
        )
        results["jvp_calls"] = 0
        results["vjp_calls"] = vjp_calls
        results["jvp_runtime_s"] = 0.0
        results["vjp_runtime_s"] = vjp_runtime_s

        return results


class SciPyMinimizerGrad(Minimizer):
    def __init__(self) -> None:
        super().__init__()

    def retrieve(
        self,
        measurement_l1: RadianceBase,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
    ):
        ### Get the prior values
        x_a = retrieval_target.apriori_state()
        initial_guess = retrieval_target.state_vector()

        inv_Sa = retrieval_target.inverse_apriori_covariance()

        if inv_Sa is None:
            # No apriori covariance/regularization
            # Use initial guess to make the matrices as x_a might be None as well
            inv_Sa = np.zeros((len(initial_guess), len(initial_guess)))

        if x_a is None:
            x_a = np.zeros_like(initial_guess)

        ### Get the measurement values
        y_meas_dict = retrieval_target.measurement_vector(measurement_l1)

        y_meas = y_meas_dict["y"]
        good_meas = ~np.isnan(y_meas)
        y_meas = y_meas[good_meas]

        if "y_error" in y_meas_dict:
            # Have measurement error
            if len(np.shape(y_meas_dict["y_error"])) == 1:
                # Only supplied is the diagonal of the error elements
                Sy = sparse.csc_matrix(
                    sparse.diags(y_meas_dict["y_error"][good_meas], 0)
                )
                inv_Sy = sparse.csc_matrix(
                    sparse.diags(1 / y_meas_dict["y_error"][good_meas], 0)
                )
            else:
                Sy = y_meas_dict["y_error"][np.ix_(good_meas, good_meas)]
                inv_Sy = np.linalg.inv(Sy)
        else:
            # No user supplied error, use identity matrix
            Sy = sparse.csc_matrix(sparse.eye(len(y_meas), len(y_meas)))
            inv_Sy = sparse.csc_matrix(sparse.eye(len(y_meas), len(y_meas)))

        y_scaler_inv = sparse.diags(np.sqrt(inv_Sy.diagonal()))

        x_scaler_inv = np.linalg.cholesky(inv_Sa)
        x_scaler = np.linalg.inv(x_scaler_inv)

        y_meas = y_scaler_inv @ y_meas

        x_a = x_scaler_inv @ x_a

        def residual_fun(x):
            retrieval_target.update_state(x_scaler @ x)

            y_ret_dict = retrieval_target.measurement_vector(
                forward_model.calculate_radiance()
            )

            K = y_scaler_inv @ y_ret_dict["jacobian"][good_meas, :] @ x_scaler
            y_ret = y_scaler_inv @ y_ret_dict["y"][good_meas]

            cost = (y_ret - y_meas).T @ (y_ret - y_meas).T + (x - x_a).T @ (x - x_a)

            grad = K.T @ (y_ret - y_meas) + (x - x_a)
            K_x = np.eye(len(x))

            full_K = np.vstack([K, K_x])

            return (cost, 2 * grad), 2 * full_K.T @ full_K

        fun = MemoizeJac(residual_fun)
        hess = fun.derivative

        return minimize(
            fun,
            x0=x_scaler_inv @ initial_guess,
            jac=True,
            hess=hess,
            options={"disp": True, "maxiter": 30},
            method="trust-exact",
        )
