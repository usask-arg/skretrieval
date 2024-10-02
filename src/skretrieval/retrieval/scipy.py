from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, minimize
from scipy.optimize._optimize import MemoizeJac

from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval import ForwardModel, Minimizer, RetrievalTarget
from skretrieval.retrieval.erroranalysis import estimate_error


class SciPyMinimizer(Minimizer):
    def __init__(
        self,
        method="trf",
        max_nfev=20,
        ftol=1e-3,
        xtol=1e-36,
        x_scale="jac",
        tr_solver="exact",
        apply_state_scaling=False,
        include_bounds=False,
        num_passes=1,
        **kwargs,
    ) -> None:
        """
        A minimization wrapper around Scipy's least_squares function

        Parameters
        ----------
        method : str, optional
            Minimization method, see scipy.least_squares, by default "trf".
            Recommended to only use "lm" or "trf".
        max_nfev : int, optional
            Maximum function evalations, see scipy.least_squares, by default 20
        ftol : _type_, optional
            Tolerance on the cost function, see sci, by default 1e-3
        xtol : _type_, optional
            Tolerance on the change in state, by default 1e-36
        x_scale : str, optional
            Internal scaling applyed by the minimizer, by default "jac"
        tr_solver : str, optional
            For the "trf" method, how to solve the trust region problem, by default "exact"
        apply_state_scaling: bool, optional
            If true, then the state vector is scaled relative to the apriori in the solver, useful
            when the state vector elements are of largely varying magnitudes and you have a well
            specified prior, by default False
        include_bounds : bool, optional
            If true, then bounds are included inside the solver. Only has an effect
            weth method is "trf", by default False
        num_passes : int, optional
            Number of passes to do through the minimizer. After each pass the noise covariance is adjusted
            where measurements with large residuals are given less weight on the next pass, by default 1
        """
        self._method = method
        self._ftol = ftol
        self._xtol = xtol
        self._max_nfev = max_nfev
        self._x_scale = x_scale
        self._tr_solver = tr_solver
        self._include_bounds = include_bounds
        self._num_passes = num_passes

        self._apply_state_scaling = apply_state_scaling

        self._kwargs = kwargs

    def retrieve(
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
            else:
                Sy = y_meas_dict["y_error"][np.ix_(good_meas, good_meas)]
                inv_Sy = np.linalg.inv(Sy)
        else:
            # No user supplied error, use identity matrix
            Sy = sparse.csc_matrix(sparse.eye(len(y_meas), len(y_meas)))
            inv_Sy = sparse.csc_matrix(sparse.eye(len(y_meas), len(y_meas)))

        y_scaler_inv = np.linalg.cholesky(inv_Sy)

        try:
            chol_inv_Sa = np.linalg.cholesky(inv_Sa)
        except np.linalg.LinAlgError:
            # If the inverse covariance is not positive definite, then we can't use the cholesky
            # decomposition, but we can use an eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(inv_Sa)
            eigvals[eigvals < 0] = 0
            chol_inv_Sa = np.diag(np.sqrt(eigvals)) @ eigvecs.T

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
                forward_model.calculate_radiance()
            )

            K = y_ret_dict["jacobian"]

            K = y_scaler_inv @ K[good_meas, :] @ x_scaler
            y_ret = y_scaler_inv @ y_ret_dict["y"][good_meas]

            # First part of residuals is from y, y_meas - y_ret, and jacobian K
            res = y_ret - y_scaler_inv @ y_meas
            # Second part of residuals is x-x_a, with identity jacobian in scaled space
            res_x = chol_inv_Sa @ x_scaler @ (x - x_a)
            K_x = chol_inv_Sa @ x_scaler

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
                verbose=2,
                tr_solver=self._tr_solver,
                max_nfev=self._max_nfev,
                tr_options={"regularize": False},
                method=self._method,
                xtol=self._xtol,
                ftol=self._ftol,
                bounds=bounds,
                **self._kwargs,
            )

            y_ret_dict = retrieval_target.measurement_vector(
                forward_model.calculate_radiance()
            )

            meas_resid = y_meas - y_ret_dict["y"][good_meas]

            median_resid = np.median(np.abs(meas_resid))
            # Adjust the scaler based on the residual fractions
            scaler = np.abs(meas_resid) / median_resid
            scaler[scaler < 1] = 1

            # Rescale the measurement errrors
            y_scaler_inv = sparse.diags(np.sqrt(inv_Sy.diagonal() / scaler**2))

            initial_guess = retrieval_target.state_vector()

        K = y_ret_dict["jacobian"][good_meas, :]

        results.update(estimate_error(K, Sy, inv_Sy, inv_Sa))

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
