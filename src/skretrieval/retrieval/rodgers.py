from __future__ import annotations

import logging
from copy import copy

import numpy as np
from scipy import sparse

from skretrieval.retrieval import ForwardModel, Minimizer, RetrievalTarget
from skretrieval.retrieval.erroranalysis import estimate_error


class Rodgers(Minimizer):
    def __init__(
        self,
        max_iter: int = 10,
        lm_damping: float = 0,
        iterative_update_lm: bool = False,
        retreat_lm: bool = False,
        lm_change_factor: float = 1.5,
        convergence_factor: float = 1,
        convergence_check_method="linear",
        lm_damping_method="fletcher",
        apply_cholesky_scaling=False,
    ):
        """
        Implements the standard inverse problem method described in "Inverse Methods for Atmospheric Sounding"
        by Rodgers (2000).

        Parameters
        ----------
        max_iter: int, optional
            The maximum number of iterations to perform when calling retrieve. Default: 10
        lm_damping: float, optional
            The Levenberg-Marquardt damping parameter.  A value of 0 would indicate no damping.  Default: 0
        iterative_update_lm: bool, optional
            If True, the LM damping factor is modified each iteration based on how well the problem is converging.
            Default: False
        retreat_lm: bool, optional
            If True, if the chi sq is worse for one iteration the state vector is retreated back to the previous
            iteration's value. Default: False
        lm_change_factor: float, optional
            The multiplicative factor applied to the LM damping parameter each iteration if iterative_update_lm is
            set to True.  Default: 1.5
        convergence_factor: float, optional
            See convergence_check_method.  Reasonable values are 1.01 if convergence_check_method is linear, and 1
            if convergence_check_method is dcost.  Default: 1
        convergence_check_method: str, optional
            Sets the method of checking for convergence.  If 'linear' then (expected chi_sq) / (chi_sq) is checked if
            it is less than convergence_factor. If 'dcost', then the derivative of the cost function is checked against
            convergence_factor.
        lm_damping_method: str, optional
            One of 'fletcher', 'prior', or 'identity'.  If 'fletcher', the LM damping term will be
            lm_damping * diag(K^T inv_Sy K).  If 'prior', the lm damping is lm_damping * inv_Sa.
            If 'identity' the damping is lm_damping * identity. Default is 'fletcher'
        apply_cholesky_scaling: bool, optional
            If True, then attempts will be made to rescale the state vector and measurement vectors to a more
            numerically suitable space
        """
        self._max_iter = max_iter
        self._lm_damping = lm_damping
        self._iterative_update_lm = iterative_update_lm
        self._retreat_lm = retreat_lm
        self._lm_change_factor = lm_change_factor
        self._convergence_factor = convergence_factor
        self._convergence_check_method = convergence_check_method
        self._lm_damping_method = lm_damping_method
        self._apply_cholesky_scaling = apply_cholesky_scaling

    @staticmethod
    def _measurement_parameters(retrieval_target: RetrievalTarget, measurement_l1):
        """
        Calculates parameters necessary for the iteration that depend on the measurement data.

        Parameters
        ----------
        retrieval_target: RetrievalTarget
            Target that is being retrieved
        measurement_l1:
            Measurement l1 data

        Returns
        -------
        y_meas_dict: dict
            Dictionary containing various parameters of the measurement vector
        y_meas: np.array
            The measurement vector, length (m,)
        Sy: matrix like
            Matrix of size (m,m) that contains the error covariance of the measurements
        inv_Sy: matrix like
            Matrix of size (m,m), that is the inverse of inv_Sy
        good_meas: np.array
            Flag of which measurements are used in y_meas.  y_meas = y_meas_dict['y'][good_meas]
        """
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

        return y_meas_dict, y_meas, Sy, inv_Sy, good_meas

    @staticmethod
    def _apriori_parameters(retrieval_target: RetrievalTarget):
        """
        Calculates several parameters related to the apriori using the retrieval target.

        Parameters
        ----------
        retrieval_target: RetrievalTarget

        Returns
        -------
        x_a: np.array
            Array of length (n,) that is the apriori state
        inv_Sa: matrix_like
            Matrix of size (n,n) that is the apriori covariance
        initial_guess: np.array
            Array of length (n,) that is the current state
        """
        x_a = retrieval_target.apriori_state()
        initial_guess = retrieval_target.state_vector()

        inv_Sa = retrieval_target.inverse_apriori_covariance()

        if inv_Sa is None:
            # No apriori covariance/regularization
            # Use initial guess to make the matrices as x_a might be None as well
            inv_Sa = np.zeros((len(initial_guess), len(initial_guess)))

        if x_a is None:
            x_a = np.zeros_like(initial_guess)

        return x_a, inv_Sa, initial_guess

    def retrieve(
        self,
        measurement_l1,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
    ):
        retrieval_target.initialize(forward_model, measurement_l1)

        output_dict = {}

        y_meas_dict, y_meas, Sy, inv_Sy, good_meas = self._measurement_parameters(
            retrieval_target, measurement_l1
        )
        x_a, inv_Sa, initial_guess = self._apriori_parameters(retrieval_target)

        xs = []
        xs.append(initial_guess)
        ys = []
        chi_sq_prev = None
        best_x = retrieval_target.state_vector()
        best_chi_sq = 1e99

        if self._apply_cholesky_scaling:
            x_scaler_inv = sparse.diags(np.sqrt(inv_Sa.diagonal()))
            y_scaler = sparse.diags(1 / np.sqrt(inv_Sy.diagonal()))
            y_scaler_inv = sparse.diags(np.sqrt(inv_Sy.diagonal()))
            x_scaler = sparse.diags(1 / np.sqrt(inv_Sa.diagonal()))
        else:
            x_scaler_inv = sparse.eye(len(x_a))
            x_scaler = x_scaler_inv
            y_scaler = sparse.eye(len(y_meas))
            y_scaler_inv = sparse.eye(len(y_meas))

        y_meas = y_scaler_inv @ y_meas

        inv_Sa = x_scaler @ inv_Sa @ x_scaler
        inv_Sy = y_scaler @ inv_Sy @ y_scaler
        x_a = x_scaler_inv @ x_a

        for iter_idx in range(self._max_iter):
            x = x_scaler_inv @ retrieval_target.state_vector()
            forward_l1 = forward_model.calculate_radiance()

            y_ret_dict = retrieval_target.measurement_vector(forward_l1)

            K = y_ret_dict["jacobian"]

            K = y_scaler_inv @ K[good_meas, :] @ x_scaler
            y_ret = y_scaler_inv @ y_ret_dict["y"][good_meas]

            ys.append(y_ret_dict["y"])

            approx_hessian = K.T @ inv_Sy @ K

            # Left side of rodgers equation
            if sparse.issparse(K):
                if self._lm_damping_method.lower() == "fletcher":
                    A = (
                        approx_hessian
                        + self._lm_damping * sparse.diags((approx_hessian).diagonal())
                        + inv_Sa
                    )
                elif self._lm_damping_method.lower() == "prior":
                    A = approx_hessian + (self._lm_damping + 1) * inv_Sa
                elif self._lm_damping_method == "identity":
                    A = (
                        approx_hessian
                        + inv_Sa
                        + self._lm_damping * np.eye(inv_Sa.shape)
                    )
                else:
                    msg = "lm_damping_method should be one of fletcher, prior, or identity"
                    raise ValueError(msg)
            else:
                if self._lm_damping_method.lower() == "fletcher":
                    A = (
                        approx_hessian
                        + self._lm_damping * np.diag(np.diag(approx_hessian))
                        + inv_Sa
                    )
                elif self._lm_damping_method.lower() == "prior":
                    A = approx_hessian + (self._lm_damping + 1) * inv_Sa
                elif self._lm_damping_method == "identity":
                    A = (
                        approx_hessian
                        + inv_Sa
                        + self._lm_damping * np.eye(inv_Sa.shape[0])
                    )
                else:
                    msg = "lm_damping_method should be one of fletcher, prior, or identity"
                    raise ValueError(msg)

            A_without_lm = approx_hessian + inv_Sa

            # Right side of rodgers equation
            B = K.T @ inv_Sy @ (y_meas - y_ret) - inv_Sa @ (x - x_a)

            if sparse.issparse(A):
                try:
                    delta_x = np.linalg.solve(A.toarray(), B)
                except np.linalg.LinAlgError:
                    delta_x = np.linalg.lstsq(A.toarray(), B)[0]
            else:
                try:
                    delta_x = np.linalg.solve(A, B)
                except np.linalg.LinAlgError:
                    delta_x = np.linalg.lstsq(A, B)[0]

            if sparse.issparse(A_without_lm):
                try:
                    delta_x_without_lm = np.linalg.solve(A_without_lm.toarray(), B)
                except np.linalg.LinAlgError:
                    delta_x_without_lm = np.linalg.lstsq(A_without_lm.toarray(), B)[0]
            else:
                try:
                    delta_x_without_lm = np.linalg.solve(A_without_lm, B)
                except np.linalg.LinAlgError:
                    delta_x_without_lm = np.linalg.lstsq(A_without_lm, B)[0]

            x_new = x_scaler @ (x + delta_x)

            chi_sq_only_meas = (y_meas - y_ret).T @ inv_Sy @ (y_meas - y_ret)
            chi_sq = chi_sq_only_meas + (x_a - x).T @ inv_Sa @ (x_a - x)

            chi_sq_only_meas_linear = (
                (y_meas - y_ret - K @ delta_x_without_lm).T
                @ inv_Sy
                @ (y_meas - y_ret - K @ delta_x_without_lm)
            )

            chi_sq_linear = (y_meas - y_ret - K @ delta_x_without_lm).T @ inv_Sy @ (
                y_meas - y_ret - K @ delta_x_without_lm
            ) + (x_a - x - delta_x_without_lm).T @ inv_Sa @ (
                x_a - x - delta_x_without_lm
            )

            chi_sq_only_meas /= len(y_meas)
            chi_sq /= len(y_meas)

            if np.isnan(chi_sq) or np.isinf(chi_sq):
                msg = "chi_sq is infinite or nan"
                raise ValueError(msg)

            chi_sq_only_meas_linear /= len(y_meas)
            chi_sq_linear /= len(y_meas)

            logging.info("Iteration %i of %i", iter_idx + 1, self._max_iter)
            logging.info(
                "    chi_sq: %f, chi_sq_only_meas: %f, expected_chi_sq: %f, chi_sq_only_meas_linear: %f",
                chi_sq,
                chi_sq_only_meas,
                chi_sq_linear,
                chi_sq_only_meas_linear,
            )

            retrieval_target.update_state(x_new)

            dcost = (
                delta_x_without_lm
                @ (K.T @ inv_Sy @ (y_meas - y_ret) + inv_Sa @ (x_a - x))
                / len(x)
            )
            logging.info("    dcost: %f", dcost)

            if chi_sq_prev is not None and chi_sq_prev < chi_sq:
                # Iteration was worse
                if self._iterative_update_lm:
                    self._lm_damping *= self._lm_change_factor**2
                if self._retreat_lm:
                    retrieval_target.update_state(x_scaler @ best_x)
                else:
                    chi_sq_prev = chi_sq
                logging.info(
                    "    Iteration was worse increasing LM factor to: %f",
                    self._lm_damping,
                )
            elif chi_sq_prev is None or chi_sq < chi_sq_prev:
                best_x = copy(x)
                best_chi_sq = chi_sq
                chi_sq_prev = best_chi_sq
                if self._iterative_update_lm and iter_idx > 0:
                    self._lm_damping /= self._lm_change_factor
                    logging.info(
                        "    Iteration was better, decreasing LM factor to: %f",
                        self._lm_damping,
                    )

            xs.append(retrieval_target.state_vector())

            if self._convergence_check_method.lower() == "linear":
                if (
                    chi_sq / chi_sq_linear < self._convergence_factor
                    and chi_sq / chi_sq_linear > 1
                ):
                    logging.info(
                        "    Stopping due to early convergence, converage_ratio: %f",
                        chi_sq / chi_sq_linear,
                    )
                    break
            elif (self._convergence_check_method.lower() == "dcost") and (
                dcost < self._convergence_factor
            ):
                logging.info(
                    "    Stopping due to early convergence. dcost is less than convergence_factor"
                )
                break

            if iter_idx != self._max_iter - 1:
                predicted_delta_y = y_meas - y_ret - K @ delta_x_without_lm
                retrieval_target.adjust_parameters(
                    forward_model,
                    y_ret_dict,
                    chi_sq,
                    chi_sq_linear,
                    iter_idx,
                    predicted_delta_y,
                )

                if retrieval_target.state_vector_allowed_to_change():
                    x_a, inv_Sa, initial_guess = self._apriori_parameters(
                        retrieval_target
                    )

                    if self._apply_cholesky_scaling:
                        x_scaler_inv = sparse.diags(np.sqrt(inv_Sa.diagonal()))
                        x_scaler = sparse.diags(1 / np.sqrt(inv_Sa.diagonal()))
                    else:
                        x_scaler_inv = sparse.eye(len(x_a))
                        x_scaler = x_scaler_inv

                    x_a = x_scaler_inv @ x_a

                    inv_Sa = x_scaler @ inv_Sa @ x_scaler
                if retrieval_target.measurement_vector_allowed_to_change():
                    (
                        y_meas_dict,
                        y_meas,
                        Sy,
                        inv_Sy,
                        good_meas,
                    ) = self._measurement_parameters(retrieval_target, measurement_l1)
                    if self._apply_cholesky_scaling:
                        y_scaler = sparse.diags(1 / np.sqrt(inv_Sy.diagonal()))
                        y_scaler_inv = sparse.diags(np.sqrt(inv_Sy.diagonal()))
                    else:
                        y_scaler = sparse.eye(len(y_meas))
                        y_scaler_inv = sparse.eye(len(y_meas))

                    y_meas = y_scaler_inv @ y_meas
                    inv_Sy = y_scaler @ inv_Sy @ y_scaler

        if self._max_iter > 0:
            output_dict.update(estimate_error(K, Sy, inv_Sy, inv_Sa, A_without_lm))

        output_dict["xs"] = xs
        output_dict["ys"] = ys
        output_dict["y_meas"] = y_meas_dict["y"]
        output_dict["forward_l1"] = forward_l1

        if self._max_iter > 0:
            output_dict["chi_sq_meas"] = chi_sq_only_meas
            output_dict["chi_sq_meas_linear"] = chi_sq_only_meas_linear

            output_dict["chi_sq"] = chi_sq
            output_dict["chi_sq_linear"] = chi_sq_linear
        else:
            output_dict["chi_sq_meas"] = None
            output_dict["chi_sq_meas_linear"] = None

            output_dict["chi_sq"] = None
            output_dict["chi_sq_linear"] = None

        return output_dict
