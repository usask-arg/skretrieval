from __future__ import annotations

import numpy as np
import sasktran as sk

from skretrieval.core.radianceformat import RadianceBase, RadianceGridded
from skretrieval.retrieval import ForwardModel, RetrievalTarget


class OzoneRetrieval(RetrievalTarget):
    def __init__(self, ozone_species: sk.Species):
        """
        A test species used just for testing, not intended for real use
        """
        self._ozone_species = ozone_species

        self._retrieval_altitudes = np.arange(10500, 60500, 1000)
        self._atmosphere_altitudes = ozone_species.climatology.altitudes

    def initialize(self, forward_model: ForwardModel, meas_l1: RadianceBase):
        return super().initialize(forward_model, meas_l1)

    def measurement_vector(self, l1_data: RadianceBase):
        if not isinstance(l1_data, RadianceGridded):
            msg = "Class OzoneRetrieval only supports data in the form RadianceGridded"
            raise ValueError(msg)

        triplet_values = l1_data.data.sel(wavelength=600, method="nearest")

        result = {}

        result["y"] = triplet_values["radiance"].to_numpy()
        if "wf" in triplet_values:
            result["jacobian"] = triplet_values["wf"].to_numpy()
        return result

    def state_vector(self):
        return self._ozone_species.climatology.get_parameter(
            self._ozone_species.species, 0, 0, self._retrieval_altitudes, 54372
        )

    def update_state(self, x: np.ndarray):
        current_x = self.state_vector()

        mult_factors = x / current_x

        mult_factors = np.interp(
            self._atmosphere_altitudes,
            self._retrieval_altitudes,
            mult_factors,
            left=mult_factors[0],
            right=mult_factors[-1],
        )

        mult_factors[mult_factors < 0.2] = 0.2
        mult_factors[mult_factors > 5] = 5

        self._ozone_species.climatology[self._ozone_species.species] *= mult_factors

    def apriori_state(self):
        return None

    def inverse_apriori_covariance(self):
        return None
