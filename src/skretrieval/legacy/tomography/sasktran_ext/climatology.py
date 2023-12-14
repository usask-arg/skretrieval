from __future__ import annotations

import numpy as np
import sasktran as sk
import sasktranif.sasktranif as skif
from sasktran.climatology import ClimatologyBase

from skretrieval.tomography.grids import OrbitalPlaneGrid


class OrbitalPlaneClimatology(ClimatologyBase):
    def __init__(self, grid: OrbitalPlaneGrid, values: dict):
        """
        2D Climatology constructed out of a grid representing an orbital plane

        Parameters
        ----------
        grid : OrbitalPlaneGrid
            The grid object defining the orbital plane
        values : dict
            Dictionary with keys being the species names, and the values being 2D arrays of (numhoriz, numalt) where
            numhoriz is the number of horizontal steps in grid, and numalt is the number of altitude steps in grid
        """
        self._grid = grid
        self._values = values

    def supported_species(self):
        return list(self._values.keys())

    def skif_object(self, **kwargs) -> skif.ISKClimatology:
        reference_point = kwargs["engine"].model_parameters["referencepoint"]

        geo = sk.Geodetic()
        geo.from_lat_lon_alt(reference_point[0], reference_point[1], reference_point[2])

        local_angles, angleidx, normalandreference = self._grid.get_local_plane(
            geo.location
        )

        local_values = {}
        for key, item in self._values.items():
            local_values[key] = item[angleidx, :]

        user_clim = sk.ClimatologyUserDefined2D(
            np.rad2deg(local_angles),
            self._grid.altitudes,
            local_values,
            normalandreference[3:],
            normalandreference[:3],
        )

        return user_clim.skif_object()

    def __getitem__(self, item):
        return self._values[item]

    def __setitem__(self, item, value):
        self._values[item] = np.asarray(value)
