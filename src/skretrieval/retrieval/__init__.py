from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval.erroranalysis import information_sqrt

from . import observation


class ForwardModel(ABC):
    """
    A ForwardModel is an object which is capable of calculating a radiance.  This serves as the primary interface
    to the retrieval, along with the RetrievalTarget.
    """

    @abstractmethod
    def calculate_radiance(self):
        pass


class RetrievalTarget(ABC):
    """
    The retrieval target defines the parameter that is to be retrieved, and also what measurements are going to be
    used to retrieve it.  Notation is similar to that of Rodgers.
    """

    @abstractmethod
    def state_vector(self):
        """
        Returns
        -------
        np.array
            The state vector, x
        """

    @abstractmethod
    def measurement_vector(self, l1_data: RadianceBase):
        """
        Parameters
        ----------
        l1_data: RadianceBase
            Radiance data.  Usually this is an instrument specific instance of RadianceBase, and the RetrievalTarget
            only works with specific formats.

        Returns
        -------
        dict
            Keys 'y' for the measurement vector, 'jacobian' for the jacobian of the measurement vector (if weighting
            functions are in l1_data, 'y_error' the covariance of 'y' (if error information is provided in l1_data)
        """

    def matrix_free_measurement_vector(self, l1_data: RadianceBase):
        """Evaluate modeled measurements used inside matrix-free optimization.

        Targets may override this when modeled error covariance is unnecessary.
        """
        return self.measurement_vector(l1_data)

    def observed_measurement_vector(self, l1_data: RadianceBase):
        """Evaluate observed measurements without requiring a Jacobian.

        Targets may override this to avoid allocating dummy derivative columns
        when a matrix-free solver only needs the observed values and errors.
        """
        return self.measurement_vector(l1_data)

    @abstractmethod
    def update_state(self, x: np.ndarray):
        """
        Updates the state for the new state vector.  Note that this change has to propagate backwards to the ForwardModel
        somehow.  Typically this is done by passing a climatology into the RetrievalTarget at initiliazation which is
        used in the ForwardModel.

        Parameters
        ----------
        x: np.array
            New state vector
        """

    @abstractmethod
    def apriori_state(self) -> np.array:
        """
        Returns
        -------
        np.array
            Apriori state vector, x_a.  If no apriori is used return None
        """

    @abstractmethod
    def inverse_apriori_covariance(self):
        """
        Returns
        -------
        np.array
            Inverse of the apriori covariance matrix.  If no apriori is used return None.
        """

    def prior_precision_factor(self):
        """Return ``R`` such that ``R.T @ R`` is the prior precision."""
        information = self.inverse_apriori_covariance()
        if information is None:
            return np.zeros((0, len(self.state_vector())))
        return information_sqrt(information, "A priori inverse covariance")

    def output_state_derivative_by_retrieval_state(self) -> np.ndarray:
        """Return the local diagonal map from retrieval to reported state."""
        return np.ones_like(np.asarray(self.state_vector(), dtype=float))

    def averaging_kernel_row_sum_groups(self) -> np.ndarray:
        """Label state entries whose averaging-kernel columns may be summed.

        The default treats the state as one physical quantity. Targets with a
        heterogeneous state should return a distinct integer label for each
        quantity so row sums do not mix variables with incompatible units.
        """
        return np.zeros_like(np.asarray(self.state_vector()), dtype=int)

    def averaging_kernel_resolution_coordinates(self) -> dict[str, np.ndarray]:
        """Return physical coordinates used for averaging-kernel moments."""
        return {}

    def prior_cost_and_gradient(self) -> tuple[float, np.ndarray]:
        """Return the quadratic prior cost and gradient at the current state.

        The gradient is expressed in the same coordinates as :meth:`state_vector`.
        Targets with nonlinear state-coordinate transforms should override this
        method so the prior remains defined in its native physical coordinates.
        """
        state = np.asarray(self.state_vector(), dtype=float).reshape(-1)
        apriori = self.apriori_state()
        if apriori is None:
            return 0.0, np.zeros_like(state)
        apriori = np.asarray(apriori, dtype=float).reshape(-1)
        information = self.inverse_apriori_covariance()
        if information is None:
            return 0.0, np.zeros_like(state)
        delta = state - apriori
        gradient = np.asarray(information @ delta).reshape(-1)
        return 0.5 * float(delta @ gradient), gradient

    def initialize(  # noqa: B027
        self, forward_model: ForwardModel, meas_l1: RadianceBase
    ):
        """
        Called at the beginning of the retrieval and can be used to initialize parameters

        Parameters
        ----------
        forward_model
        meas_l1

        """

    @staticmethod
    def state_vector_allowed_to_change():
        """
        Returns
        -------
        bool
            True if the state vector/apriori may change shape between iterations, False otherwise.
        """
        return False

    @staticmethod
    def measurement_vector_allowed_to_change():
        """
        Returns
        -------
        bool
            True if the measurement_vector may change shape between iterations, False otherwise.
        """
        return False

    def adjust_parameters(
        self,
        forward_model,  # noqa: ARG002
        y_dict,  # noqa: ARG002
        chi_sq,  # noqa: ARG002
        chi_sq_linear,  # noqa: ARG002
        iter_idx,  # noqa: ARG002
        predicted_delta_y,  # noqa: ARG002
    ):
        return None

    @staticmethod
    def state_vector_error_output(output_dict: dict) -> dict:
        return output_dict


class Minimizer(ABC):
    """
    A class which performs minimization between some aspect of measurement level1 data and the forward model simulations.
    """

    @abstractmethod
    def retrieve(
        self,
        measurement_l1: RadianceBase,
        forward_model: ForwardModel,
        retrieval_target: RetrievalTarget,
    ):
        """

        Parameters
        ----------
        measurement_l1: RadianceBase
            The data we are trying to match, either from a real instrument or simulations.
        forward_model: ForwardModel
            A model for the data in measurement_l1
        retrieval_target: RetrievalTarget
            What we are trying to retrieve

        Returns
        -------
        dict
            Various parameters specific to the minimizer
        """
