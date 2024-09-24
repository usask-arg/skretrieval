from __future__ import annotations

import abc

import numpy as np
from scipy.linalg import block_diag
from scipy.special import expit

from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval import RetrievalTarget
from skretrieval.retrieval.statevector import StateVector


class LogisticBoundingMixin:
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

        internal_x[both_bounds] = np.log(
            (x[both_bounds] - lb[both_bounds]) / (ub[both_bounds] - x[both_bounds])
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

        bounded_x[both_bounds] = lb[both_bounds] + (
            ub[both_bounds] - lb[both_bounds]
        ) * expit(x[both_bounds])

        eps = np.finfo(bounded_x[0]).eps

        bounded_x[bounded_x == lb] = lb[bounded_x == lb] + eps
        bounded_x[bounded_x == ub] = ub[bounded_x == ub] - eps

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
        x = self._map_internal_to_bounded(self.state_vector())
        lb = self.lower_bound()
        ub = self.upper_bound()
        mapping = np.zeros_like(x)

        no_map = (lb == -np.inf) & (ub == np.inf)

        both_bounds = (lb != -np.inf) & (ub != np.inf)

        mapping[no_map] = 1
        mapping[both_bounds] = (
            (x[both_bounds] - lb[both_bounds])
            * (ub[both_bounds] - x[both_bounds])
            / (ub[both_bounds] - lb[both_bounds])
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

        mapping[both_bounds] = (ub[both_bounds] - lb[both_bounds]) / (
            (xb[both_bounds] - lb[both_bounds]) * (ub[both_bounds] - xb[both_bounds])
        )

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

        if "jacobian" in result:
            result["jacobian"] = self.map_K(result["jacobian"])
        return result

    @abc.abstractmethod
    def _internal_measurement_vector(self, l1_data: RadianceBase):
        pass

    def state_vector(self):
        vec = []
        for state_element in self._state_vector.state_elements:
            if state_element.enabled:
                vec.append(state_element.state())

        return self._map_bounded_to_internal(np.concatenate(vec))

    def lower_bound(self):
        vec = []
        for state_element in self._state_vector.state_elements:
            if state_element.enabled:
                vec.append(state_element.lower_bound())

        return np.concatenate(vec)

    def upper_bound(self):
        vec = []
        for state_element in self._state_vector.state_elements:
            if state_element.enabled:
                vec.append(state_element.upper_bound())

        return np.concatenate(vec)

    def update_state(self, x: np.ndarray):
        rescaled_x = self._map_internal_to_bounded(x)

        for state_element, state_slice in zip(
            self._state_vector.state_elements, self._state_slices
        ):
            if state_element.enabled:
                state_element.update_state(rescaled_x[state_slice])

    def apriori_state(self) -> np.array:
        vec = []
        for state_element in self._state_vector.state_elements:
            if state_element.enabled:
                vec.append(state_element.apriori_state())
        return self._map_bounded_to_internal(np.concatenate(vec))

    def inverse_apriori_covariance(self):
        inv_covar = []
        for state_element in self._state_vector.state_elements:
            if state_element.enabled:
                inv_covar.append(state_element.inverse_apriori_covariance())

        return self._map_inv_Sa_by_dinternal(
            self.state_vector(), block_diag(*inv_covar)
        )

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

    def __init__(self, state_vector: StateVector, rescale_state_elements: bool = False):
        """
        Implements a generic abstract base target class that is composed of a StateVector.  Derived classes of this
        type are responsible for implementing _internal_measurement_vector which computes the measurement vector.
        All of the other functionality, updating the state, apriori parameters, are handled by the state vector elements
        and this class.

        Parameters
        ----------
        state_vector: StateVector
            The state vector
        rescale_state_elements: bool
            If true, then the state vector respace is internally scaled so that the lower_bound and upper_bound of each
            state vector element is enforced.  This is useful if the minimization method does not allow for bounds,
            such as Rodgers or scipy LM
        """
        self._state_vector = state_vector
        self._rescale_state_elements = rescale_state_elements

        self.update_state_slices()
