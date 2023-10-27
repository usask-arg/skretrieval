from __future__ import annotations

import abc
from typing import Iterable

import numpy as np
import xarray as xr


class StateVectorElement(abc.ABC):
    """
    A state vector element is a component of the full state vector used in the retrieval. Each state vector element
    has a state, and a prior state/covariance associated with it.  The state vector element must also be able
    to update itself, calculate the jacobian matrix for itself.
    """

    @abc.abstractmethod
    def state(self) -> np.array:
        pass

    def inverse_apriori_covariance(self) -> np.ndarray:
        n = len(self.state())
        return np.zeros((n, n))

    def apriori_state(self) -> np.array:
        return np.zeros_like(self.state())

    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def propagate_wf(self, radiance) -> np.ndarray:
        pass

    @abc.abstractmethod
    def update_state(self, x: np.array):
        pass

    def modify_input_radiance(self, radiance: xr.Dataset):
        return radiance


class StateVector:
    def __init__(self, elements: Iterable[StateVectorElement]):
        """
        A full state vector made up of a collection of state vector elements.

        Parameters
        ----------
        elements: Iterable[StateVectorElement]
            A collection of state vector elements
        """
        self._elements = elements

    @property
    def state_elements(self):
        return self._elements

    def update_sasktran_radiance(self, radiance: xr.Dataset, drop_old_wf: bool = False):
        """
        Modifies radiances output from sasktran based on the state vector elements if applicable, e.g., if a state
        vector element is a wavelength shift this will apply it.

        Propagates weighting functions from the sasktran radiance raw output to weighting functions for each
        state vector element.

        If drop_old_wf is set to true then the old weighting functions are removed from the radiance.

        Parameters
        ----------
        radiance: xr.Dataset
            Output from sk.Engine.calculate_radiance(output_format='xarray')
        drop_old_wf: bool, Optional
            If true then the old weighting functions are removed after being propagated to the state vector. Default
            False

        Returns
        -------
        radiance: xr.Dataset
            Modified radiance with a new key 'wf' that is the jacobian with respect to the full state vector.
        """
        all_jacobian = []
        for state_element in self._elements:
            all_jacobian.append(state_element.propagate_wf(radiance))
            radiance = state_element.modify_input_radiance(radiance)

        new_wf = xr.concat(all_jacobian, dim="x")
        radiance["wf"] = new_wf

        if drop_old_wf:
            wf_names = [key for key in radiance if key.startswith("wf_")]
            radiance = radiance.drop(wf_names)
        return radiance
