from __future__ import annotations

import abc

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy.special import expit

from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval import RetrievalTarget
from skretrieval.retrieval.statevector import StateVector


def _operator_matvec(operator, value: np.ndarray) -> np.ndarray:
    if hasattr(operator, "jvp"):
        return np.asarray(operator.jvp(value)).reshape(-1)
    return np.asarray(operator.matvec(value)).reshape(-1)


def _operator_rmatvec(operator, value: np.ndarray) -> np.ndarray:
    if hasattr(operator, "vjp"):
        return np.asarray(operator.vjp(value)).reshape(-1)
    return np.asarray(operator.rmatvec(value)).reshape(-1)


class _MappedJacobianOperator:
    """Apply a diagonal state-coordinate mapping around a Jacobian operator."""

    def __init__(self, operator, mapping: np.ndarray) -> None:
        self._operator = operator
        self._mapping = np.asarray(mapping).reshape(-1)
        self.n_state = len(self._mapping)
        operator_n_state = getattr(operator, "n_state", None)
        if operator_n_state is None and hasattr(operator, "shape"):
            operator_n_state = operator.shape[1]
        if operator_n_state is not None and operator_n_state != self.n_state:
            msg = "Jacobian operator and state-coordinate mapping sizes do not match"
            raise ValueError(msg)
        if hasattr(operator, "shape"):
            n_measurement = operator.shape[0]
        else:
            n_measurement = len(operator.y)
        self.shape = (n_measurement, self.n_state)

    def matvec(self, value: np.ndarray) -> np.ndarray:
        return _operator_matvec(self._operator, self._mapping * value)

    def rmatvec(self, value: np.ndarray) -> np.ndarray:
        return self._mapping * _operator_rmatvec(self._operator, value)

    jvp = matvec
    vjp = rmatvec


class LogisticBoundingMixin:
    def _state_space_transform_mode(self) -> str:
        mode = getattr(self, "_state_space_transform", None)
        if mode is not None:
            return mode
        return "logistic" if self._rescale_state_elements else "none"

    def _mapping_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "_native_lower_bound"):
            return self._native_lower_bound(), self._native_upper_bound()
        return self.lower_bound(), self.upper_bound()

    def _affine_reference_and_scale(
        self, lb: np.ndarray, ub: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        reference = getattr(self, "_affine_state_reference", lb)
        scale = getattr(self, "_affine_state_scale", ub - lb)
        return reference, scale

    def _map_bounded_to_internal(self, x: np.array) -> np.array:
        """
        Maps the bounded (user) state vector to the internal object

        Parameters
        ----------
        x : np.array
            Bounded state vector

        Returns
        -------
        np.array
            Internal state vector
        """
        mode = self._state_space_transform_mode()
        if mode == "none":
            return x
        lb, ub = self._mapping_bounds()
        internal_x = np.array(x, copy=True)

        both_bounds = (lb != -np.inf) & (ub != np.inf)
        if mode == "affine":
            reference, scale = self._affine_reference_and_scale(lb, ub)
            internal_x[both_bounds] = (x[both_bounds] - reference[both_bounds]) / scale[
                both_bounds
            ]
            return internal_x

        width = ub[both_bounds] - lb[both_bounds]
        if np.any(width <= 0):
            msg = "logistic state scaling requires ordered finite bounds"
            raise ValueError(msg)
        fraction = (x[both_bounds] - lb[both_bounds]) / width
        eps = np.finfo(internal_x.dtype).eps
        fraction = np.clip(fraction, eps, 1.0 - eps)
        internal_x[both_bounds] = np.log(fraction / (1.0 - fraction))

        return internal_x

    def _map_internal_to_bounded(self, x: np.array) -> np.array:
        """
        Maps the internal state vector to the bounded state vector

        Parameters
        ----------
        x : np.array
            Internal state vector

        Returns
        -------
        np.array
            Bounded state vector
        """
        mode = self._state_space_transform_mode()
        if mode == "none":
            return x
        lb, ub = self._mapping_bounds()
        bounded_x = np.array(x, copy=True)

        both_bounds = (lb != -np.inf) & (ub != np.inf)
        if mode == "affine":
            reference, scale = self._affine_reference_and_scale(lb, ub)
            bounded_x[both_bounds] = (
                reference[both_bounds] + scale[both_bounds] * x[both_bounds]
            )
            return bounded_x

        bounded_x[both_bounds] = lb[both_bounds] + (
            ub[both_bounds] - lb[both_bounds]
        ) * expit(x[both_bounds])

        at_lower = both_bounds & (bounded_x <= lb)
        at_upper = both_bounds & (bounded_x >= ub)
        bounded_x[at_lower] = np.nextafter(lb[at_lower], ub[at_lower])
        bounded_x[at_upper] = np.nextafter(ub[at_upper], lb[at_upper])

        return bounded_x

    def map_K(self, K: np.ndarray) -> np.ndarray:
        """
        Maps the bounded K (from SASKTRAN basically) to K for the internal variables

        Parameters
        ----------
        K : np.ndarray
            Jacobian matrix for the bounded state vector

        Returns
        -------
        np.ndarray
            Jacobian matrix for the internal state vector
        """
        if self._state_space_transform_mode() == "none":
            return K
        return K @ np.diag(self._bounded_state_derivative_by_internal())

    def map_jacobian_operator(self, operator):
        if self._state_space_transform_mode() == "none":
            return operator
        return _MappedJacobianOperator(
            operator, self._bounded_state_derivative_by_internal()
        )

    def _bounded_state_derivative_by_internal(self) -> np.ndarray:
        mode = self._state_space_transform_mode()
        if mode == "none":
            return np.ones_like(self.state_vector())

        x = self._map_internal_to_bounded(self.state_vector())
        lb, ub = self._mapping_bounds()
        mapping = np.ones_like(x)

        both_bounds = (lb != -np.inf) & (ub != np.inf)
        if mode == "affine":
            _, scale = self._affine_reference_and_scale(lb, ub)
            mapping[both_bounds] = scale[both_bounds]
        else:
            mapping[both_bounds] = (
                (x[both_bounds] - lb[both_bounds])
                * (ub[both_bounds] - x[both_bounds])
                / (ub[both_bounds] - lb[both_bounds])
            )

        return mapping

    def _map_inv_Sa_by_dinternal(self, x: np.array, inv_Sa: np.ndarray) -> np.ndarray:
        """
        Maps the inverse covariance for the bounded variables to the inverse covariance on the
        internal retrieval variables

        Parameters
        ----------
        x : np.array
            INTERNAL state vector
        inv_Sa : np.ndarray
            Bounded inverse covariance

        Returns
        -------
        np.ndarray
            Internal inverse covariance
        """
        mode = self._state_space_transform_mode()
        if mode == "none":
            return inv_Sa
        lb, ub = self._mapping_bounds()
        if mode == "affine":
            derivative = np.ones_like(x)
            both_bounds = (lb != -np.inf) & (ub != np.inf)
            _, scale = self._affine_reference_and_scale(lb, ub)
            derivative[both_bounds] = scale[both_bounds]
            transform = sparse.diags(derivative, format="csr")
            if sparse.issparse(inv_Sa):
                return transform @ inv_Sa @ transform
            transform = transform.toarray()
            return transform @ inv_Sa @ transform

        xb = self._map_internal_to_bounded(x)
        mapping = np.zeros_like(x)
        no_map = (lb == -np.inf) & (ub == np.inf)

        both_bounds = (lb != -np.inf) & (ub != np.inf)

        mapping[no_map] = 1

        mapping[both_bounds] = (ub[both_bounds] - lb[both_bounds]) / (
            (xb[both_bounds] - lb[both_bounds]) * (ub[both_bounds] - xb[both_bounds])
        )

        if sparse.issparse(inv_Sa):
            transform = sparse.diags(1 / mapping, format="csr")
            return transform @ inv_Sa @ transform
        return np.diag(1 / mapping) @ inv_Sa @ np.diag(1 / mapping)


class CosineBoundingMixin:
    def _map_bounded_to_internal(self, x: np.array) -> np.array:
        """
        Maps the bounded (user) state vector to the internal object

        Parameters
        ----------
        x : np.array
            Bounded state vector

        Returns
        -------
        np.array
            Internal state vector
        """
        if not self._rescale_state_elements:
            return x
        lb = self.lower_bound()
        ub = self.upper_bound()
        internal_x = np.zeros_like(x)

        no_map = (lb == -np.inf) & (ub == np.inf)

        both_bounds = (lb != -np.inf) & (ub != np.inf)

        internal_x[no_map] = x[no_map]

        internal_x[both_bounds] = np.arcsin(
            2 * (x[both_bounds] - lb[both_bounds]) / (ub[both_bounds] - lb[both_bounds])
            - 1
        )

        return internal_x

    def _map_internal_to_bounded(self, x: np.array) -> np.array:
        """
        Maps the internal state vector to the bounded state vector

        Parameters
        ----------
        x : np.array
            Internal state vector

        Returns
        -------
        np.array
            Bounded state vector
        """
        if not self._rescale_state_elements:
            return x
        lb = self.lower_bound()
        ub = self.upper_bound()
        bounded_x = np.zeros_like(x)

        no_map = (lb == -np.inf) & (ub == np.inf)

        both_bounds = (lb != -np.inf) & (ub != np.inf)

        bounded_x[no_map] = x[no_map]

        bounded_x[both_bounds] = (
            lb[both_bounds]
            + (np.sin(x[both_bounds]) + 1) * (ub[both_bounds] - lb[both_bounds]) / 2
        )

        return bounded_x

    def map_K(self, K: np.ndarray) -> np.ndarray:
        """
        Maps the bounded K (from SASKTRAN basically) to K for the internal variables

        Parameters
        ----------
        K : np.ndarray
            Jacobian matrix for the bounded state vector

        Returns
        -------
        np.ndarray
            Jacobian matrix for the internal state vector
        """
        if not self._rescale_state_elements:
            return K
        x = self.state_vector()
        lb = self.lower_bound()
        ub = self.upper_bound()
        mapping = np.zeros_like(x)

        no_map = (lb == -np.inf) & (ub == np.inf)

        both_bounds = (lb != -np.inf) & (ub != np.inf)

        mapping[no_map] = 1
        mapping[both_bounds] = (
            np.cos(x[both_bounds]) * (ub[both_bounds] - lb[both_bounds]) / 2
        )

        return K @ np.diag(mapping)

    def _map_inv_Sa_by_dinternal(self, x: np.array, inv_Sa: np.ndarray) -> np.ndarray:
        """
        Maps the inverse covariance for the bounded variables to the inverse covariance on the
        internal retrieval variables

        Parameters
        ----------
        x : np.array
            INTERNAL state vector
        inv_Sa : np.ndarray
            Bounded inverse covariance

        Returns
        -------
        np.ndarray
            Internal inverse covariance
        """
        if not self._rescale_state_elements:
            return inv_Sa
        lb = self.lower_bound()
        ub = self.upper_bound()
        xb = self._map_internal_to_bounded(x)
        mapping = np.zeros_like(x)
        no_map = (lb == -np.inf) & (ub == np.inf)

        both_bounds = (lb != -np.inf) & (ub != np.inf)

        mapping[no_map] = 1

        mapping[both_bounds] = 2 / (
            (ub[both_bounds] - lb[both_bounds])
            * np.sqrt(
                1
                - (
                    1
                    - (
                        2
                        * (xb[both_bounds] - lb[both_bounds])
                        / (ub[both_bounds] - lb[both_bounds])
                    )
                )
                ** 2
            )
        )

        return np.diag(1 / mapping) @ inv_Sa @ np.diag(1 / mapping)


class GenericTarget(RetrievalTarget, LogisticBoundingMixin):
    def measurement_vector(self, l1_data: RadianceBase):
        result = self._internal_measurement_vector(l1_data)

        return self._map_measurement_result(result)

    def _map_measurement_result(self, result: dict) -> dict:
        if "jacobian" in result:
            result["jacobian"] = self.map_K(result["jacobian"])
        if "jacobian_operator" in result:
            result["jacobian_operator"] = self.map_jacobian_operator(
                result["jacobian_operator"]
            )
        return result

    @abc.abstractmethod
    def _internal_measurement_vector(self, l1_data: RadianceBase):
        pass

    def state_vector(self):
        return self._map_bounded_to_internal(self._native_state_vector())

    def _native_state_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                state_element.state()
                for state_element in self._state_vector.state_elements
                if state_element.enabled
            ]
        )

    def _native_lower_bound(self):
        vec = []
        for state_element in self._state_vector.state_elements:
            if state_element.enabled:
                vec.append(state_element.lower_bound())

        return np.concatenate(vec)

    def _native_upper_bound(self):
        vec = []
        for state_element in self._state_vector.state_elements:
            if state_element.enabled:
                vec.append(state_element.upper_bound())

        return np.concatenate(vec)

    def lower_bound(self):
        lower = self._native_lower_bound()
        if self._state_space_transform_mode() != "affine":
            return lower
        return self._map_bounded_to_internal(lower)

    def upper_bound(self):
        upper = self._native_upper_bound()
        if self._state_space_transform_mode() != "affine":
            return upper
        return self._map_bounded_to_internal(upper)

    def update_state(self, x: np.ndarray):
        rescaled_x = self._map_internal_to_bounded(x)

        for state_element, state_slice in zip(
            self._state_vector.state_elements, self._state_slices
        ):
            if state_element.enabled:
                state_element.update_state(rescaled_x[state_slice])

    def apriori_state(self) -> np.array:
        return self._map_bounded_to_internal(self._native_apriori_state())

    def _native_apriori_state(self) -> np.ndarray:
        return np.concatenate(
            [
                state_element.apriori_state()
                for state_element in self._state_vector.state_elements
                if state_element.enabled
            ]
        )

    def inverse_apriori_covariance(self):
        combined = self._native_inverse_apriori_covariance()
        return self._map_inv_Sa_by_dinternal(self.state_vector(), combined)

    def _native_inverse_apriori_covariance(self):
        inv_covar = [
            state_element.inverse_apriori_covariance()
            for state_element in self._state_vector.state_elements
            if state_element.enabled
        ]
        if any(sparse.issparse(matrix) for matrix in inv_covar):
            return sparse.block_diag(inv_covar, format="csr")
        return block_diag(*inv_covar)

    def prior_cost_and_gradient(self) -> tuple[float, np.ndarray]:
        """Evaluate the exact native-coordinate prior under transformed bounds."""
        state = self._native_state_vector()
        delta = state - self._native_apriori_state()
        information = self._native_inverse_apriori_covariance()
        native_gradient = np.asarray(information @ delta).reshape(-1)
        gradient = native_gradient * self._bounded_state_derivative_by_internal()
        return 0.5 * float(delta @ native_gradient), gradient

    def prior_precision_factor(self):
        """Return a prior residual factor in the target's solver coordinates."""
        factor = self._state_vector.prior_precision_factor()
        if not self._rescale_state_elements:
            return factor
        mapping = sparse.diags(
            self._bounded_state_derivative_by_internal(), format="csr"
        )
        return factor @ mapping

    def output_state_derivative_by_retrieval_state(self) -> np.ndarray:
        """Map locally from transformed retrieval to native state coordinates."""
        return self._bounded_state_derivative_by_internal()

    def averaging_kernel_row_sum_groups(self) -> np.ndarray:
        """Keep row sums within each enabled physical state quantity."""
        return self._state_vector.averaging_kernel_row_sum_groups()

    def averaging_kernel_resolution_coordinates(self) -> dict[str, np.ndarray]:
        """Return state-vector coordinates for averaging-kernel moments."""
        return self._state_vector.averaging_kernel_resolution_coordinates()

    def state_vector_error_output(self, output_dict: dict) -> dict:
        if not self._rescale_state_elements:
            return output_dict

        mapping = self._bounded_state_derivative_by_internal()
        transform = np.diag(mapping)
        inv_transform = np.diag(1 / mapping)

        result = output_dict.copy()

        for key in ("error_covariance_from_noise", "solution_covariance"):
            if key in result:
                result[key] = transform @ result[key] @ transform

        for key in (
            "solution_covariance_diagonal",
            "solution_covariance_diagonal_standard_error",
        ):
            if key in result:
                result[key] = mapping**2 * result[key]

        for key in (
            "measurement_information_diagonal",
            "measurement_information_diagonal_standard_error",
        ):
            if key in result:
                result[key] = result[key] / mapping**2

        if "gain_matrix" in result:
            result["gain_matrix"] = transform @ result["gain_matrix"]

        if "averaging_kernel" in result:
            result["averaging_kernel"] = (
                transform @ result["averaging_kernel"] @ inv_transform
            )

        return result

    def update_state_slices(self):
        # Construct slices that map the full state vector to each individual state vector element
        self._state_slices = []
        cur_idx = 0
        for state_element in self._state_vector.state_elements:
            if state_element.enabled:
                n = len(state_element.state())
                self._state_slices.append(slice(cur_idx, cur_idx + n))
                cur_idx += n
            else:
                self._state_slices.append(None)

    def __init__(
        self,
        state_vector: StateVector,
        rescale_state_elements: bool | str = False,
    ):
        """
        Implements a generic abstract base target class that is composed of a StateVector.  Derived classes of this
        type are responsible for implementing _internal_measurement_vector which computes the measurement vector.
        All of the other functionality, updating the state, apriori parameters, are handled by the state vector elements
        and this class.

        Parameters
        ----------
        state_vector: StateVector
            The state vector
        rescale_state_elements: bool or str
            True or ``"logistic"`` applies the legacy unbounded logistic
            transform. ``"affine"`` applies a fixed local scale about the
            initial state and is intended for optimizers that enforce box
            constraints. False leaves state coordinates unchanged.
        """
        self._state_vector = state_vector
        if isinstance(rescale_state_elements, str):
            transform = rescale_state_elements.lower()
            if transform not in {"logistic", "affine", "none"}:
                msg = (
                    "rescale_state_elements must be False, True, "
                    "'logistic', or 'affine'"
                )
                raise ValueError(msg)
        else:
            transform = "logistic" if rescale_state_elements else "none"
        self._state_space_transform = transform
        self._rescale_state_elements = transform != "none"
        if transform == "affine":
            native_state = np.concatenate(
                [
                    element.state()
                    for element in self._state_vector.state_elements
                    if element.enabled
                ]
            )
            lower, upper = self._mapping_bounds()
            both_bounds = np.isfinite(lower) & np.isfinite(upper)
            if np.any(upper[both_bounds] <= lower[both_bounds]):
                msg = "affine state scaling requires ordered finite bounds"
                raise ValueError(msg)
            width = np.ones_like(native_state)
            width[both_bounds] = upper[both_bounds] - lower[both_bounds]
            fraction = np.zeros_like(native_state)
            fraction[both_bounds] = (
                native_state[both_bounds] - lower[both_bounds]
            ) / width[both_bounds]
            local_logistic_scale = np.ones_like(native_state)
            local_logistic_scale[both_bounds] = (
                width[both_bounds] * fraction[both_bounds] * (1 - fraction[both_bounds])
            )
            minimum_scale = 0.01 * width
            scale = np.ones_like(native_state)
            scale[both_bounds] = np.maximum(
                local_logistic_scale[both_bounds], minimum_scale[both_bounds]
            )
            self._affine_state_reference = np.array(native_state, copy=True)
            self._affine_state_scale = scale

        self.update_state_slices()
