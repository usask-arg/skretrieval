from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from skretrieval.core.radianceformat import RadianceBase


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
