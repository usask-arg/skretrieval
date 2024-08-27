from __future__ import annotations

import sasktran2 as sk2
import xarray as xr

from skretrieval.retrieval.statevector import StateVector
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent


class USARMStateVector(StateVector):
    def __init__(self, **kwargs):
        self._sv_eles = kwargs

        elements = []

        for _key, val in self._sv_eles.items():
            elements.append(val)

        super().__init__(elements)

    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        for key, val in self._sv_eles.items():
            if isinstance(val, StateVectorElementConstituent):
                atmo[key] = val

    @property
    def sv(self):
        return self._sv_eles

    def post_process_sk2_radiances(self, radiance: xr.Dataset):
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

        radiance = radiance.drop(wf_keys)
        radiance["wf"] = wfs

        return radiance
