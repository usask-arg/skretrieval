from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, minimize
from scipy.optimize._optimize import MemoizeJac

from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval import ForwardModel, Minimizer, RetrievalTarget


class SciPyMinimizer(Minimizer):
    def __init__(
        self,
        method="trf",
        max_nfev=20,
        ftol=1e-3,
        xtol=1e-36,
        x_scale="jac",
        tr_solver="exact",
        include_bounds=False,
        **kwargs,
    ) -> None:
        """
        A minimization wrapper around Scipy's least_squares function

        Parameters
        ----------
        method : str, optional
            Minimization method, see scipy.least_squares, by default "trf"
        max_nfev : int, optional
            Maximum function evalations, see scipy.least_squares, by default 20
        ftol : _type_, optional
            Tolerance on the cost function, see sci, by default 1e-3
        xtol : _type_, optional
            _description_, by default 1e-36
        x_scale : str, optional
            _description_, by default "jac"
        tr_solver : str, optional
            _description_, by default "exact"
        include_bounds : bool, optional
            _description_, by default False
        """
        self._method = method
        self._ftol = ftol
        self._xtol = xtol
        self._max_nfev = max_nfev
        self._x_scale = x_scale
        self._tr_solver = tr_solver
        self._include_bounds = include_bounds

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

            K = y_ret_dict["jacobian"]

            K = y_scaler_inv @ K[good_meas, :] @ x_scaler
            y_ret = y_scaler_inv @ y_ret_dict["y"][good_meas]

            # First part of residuals is from y, y_meas - y_ret, and jacobian K
            res = y_ret - y_meas
            # Second part of residuals is x-x_a, with identity jacobian in scaled space
            res_x = x - x_a
            K_x = np.eye(len(x))

            return np.concatenate((res, res_x)), np.vstack([K, K_x])

        fun = MemoizeJac(residual_fun)
        jac = fun.derivative

        bounds = (
            (lb, ub)
            if self._include_bounds
            else (np.ones_like(lb) * (-np.inf), np.ones_like(ub) * np.inf)
        )

        return least_squares(
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
