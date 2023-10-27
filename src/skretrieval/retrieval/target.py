from __future__ import annotations

import abc

import numpy as np
from scipy.linalg import block_diag

from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval import RetrievalTarget
from skretrieval.retrieval.statevector import StateVector


class GenericTarget(RetrievalTarget):
    def measurement_vector(self, l1_data: RadianceBase):
        return self._internal_measurement_vector(l1_data)

    @abc.abstractmethod
    def _internal_measurement_vector(self, l1_data: RadianceBase):
        pass

    def state_vector(self):
        vec = []
        for state_element in self._state_vector.state_elements:
            vec.append(state_element.state())

        return np.concatenate(vec)

    def update_state(self, x: np.ndarray):
        for state_element, state_slice in zip(
            self._state_vector.state_elements, self._state_slices
        ):
            state_element.update_state(x[state_slice])

    def apriori_state(self) -> np.array:
        vec = []
        for state_element in self._state_vector.state_elements:
            vec.append(state_element.apriori_state())
        return np.concatenate(vec)

    def inverse_apriori_covariance(self):
        inv_covar = []
        for state_element in self._state_vector.state_elements:
            inv_covar.append(state_element.inverse_apriori_covariance())

        return block_diag(*inv_covar)

    def __init__(self, state_vector: StateVector):
        """
        Implements a generic abstract base target class that is composed of a StateVector.  Derived classes of this
        type are responsible for implementing _internal_measurement_vector which computes the measurement vector.
        All of the other functionality, updating the state, apriori parameters, are handled by the state vector elements
        and this class.

        Parameters
        ----------
        state_vector: StateVector
        """
        self._state_vector = state_vector

        # Construct slices that map the full state vector to each individual state vector element
        self._state_slices = []
        cur_idx = 0
        for state_element in self._state_vector.state_elements:
            n = len(state_element.state())
            self._state_slices.append(slice(cur_idx, cur_idx + n))
            cur_idx += n
