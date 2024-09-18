from __future__ import annotations

import numpy as np
import sasktran as sk
import sasktranif.sasktranif as skif
from sasktran.climatology import ClimatologyBase

from skretrieval.legacy.tomography.grids import OrbitalPlaneGrid


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

        for _, item in local_values.items():
            if len(local_angles) * len(self._grid.altitudes) != len(item.flatten()):
                raise ValueError()

        return user_clim.skif_object()

    def __getitem__(self, item):
        return self._values[item]

    def __setitem__(self, item, value):
        self._values[item] = np.asarray(value)


class OrbitalPlaneAlbedo(sk.BRDF):
    def __init__(self, grid: OrbitalPlaneGrid):
        self._values = np.zeros(grid._numretprof)
        self._grid = grid

    def skif_object(self, **kwargs):
        """
        Returns the internel SasktranIF object
        """
        reference_point = kwargs["engine"].model_parameters["referencepoint"]

        geo = sk.Geodetic()
        geo.from_lat_lon_alt(reference_point[0], reference_point[1], reference_point[2])

        local_angles, angleidx, normalandreference = self._grid.get_local_plane(
            geo.location
        )

        user_clim = sk.ClimatologyUserDefined2D(
            np.rad2deg(local_angles),
            np.array([0.0, 100000.0]),
            {
                "SKCLIMATOLOGY_ALBEDO": np.tile(
                    self._values[angleidx], (2, 1)
                ).transpose()
            },
            normalandreference[3:],
            normalandreference[:3],
        )

        self._brdf = sk.BRDF("PLANE")
        self._brdf.skif_object().SetProperty("clim", user_clim.skif_object())
        return self._brdf.skif_object()

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values
