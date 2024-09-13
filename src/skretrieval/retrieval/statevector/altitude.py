from __future__ import annotations

import numpy as np
import sasktran2 as sk2
import xarray as xr

from skretrieval.retrieval.statevector import StateVector
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent


class AltitudeNativeStateVector(StateVector):
    def __init__(self, altitude_grid: np.array, **kwargs):
        """
        Class representing the state vector where state vector elements are natively
        specified on altitude levels

        Parameters
        ----------
        altitude_grid : np.array
            Native altitude grid that state vector elements should be specified on. Often
            matched to the SASKTRAN2 model grid
        """
        self._sv_eles = kwargs
        self._altitude_grid = altitude_grid

        elements = []

        for _, val in self._sv_eles.items():
            elements.append(val)

        super().__init__(elements)

    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        """
        Add`s the state vector elements to the atmosphere object

        Parameters
        ----------
        atmo : sk2.Atmosphere
        """
        for key, val in self._sv_eles.items():
            if isinstance(val, StateVectorElementConstituent):
                atmo[key] = val

    @property
    def sv(self):
        """
        The state vector elements

        Returns
        -------
        dict[StateVectorElement]
        """
        return self._sv_eles

    @property
    def altitude_grid(self):
        """
        The altitude grid that the state vector elements are defined on in [m]

        Returns
        -------
        np.array
        """
        return self._altitude_grid

    def post_process_sk2_radiances(self, radiance: xr.Dataset):
        """
        Called after the SASKTRAN2 radiance calculation to modify the radiance if the state vector elements
        require it.

        Parameters
        ----------
        radiance : xr.Dataset
        """
        wfs = []
        for val in self.state_elements:
            if val.enabled:
                wfs.append(val.propagate_wf(radiance))

        for val in self.state_elements:
            if val.enabled:
                radiance = val.modify_input_radiance(radiance)

        wfs = xr.concat(wfs, dim="x")

        wf_keys = []
        for key in list(radiance):
            if key.startswith("wf"):
                wf_keys.append(key)

        radiance = radiance.drop_vars(wf_keys)
        radiance["wf"] = wfs

        return radiance

    def describe(self, rodgers_output: dict) -> xr.Dataset:
        """
        Returns a human readable dataset containing the state vector elements

        Parameters
        ----------
        rodgers_output : dict

        Returns
        -------
        xr.Dataset
        """
        result = super().describe(rodgers_output)

        result.coords["altitude"] = self._altitude_grid

        return result
