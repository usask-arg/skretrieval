from __future__ import annotations

import math
from datetime import datetime

import numpy as np

import skretrieval.time.eci

from .satellitebase import SatelliteBase


# -----------------------------------------------------------------------------
#           SatelliteSimpleGeostationary
# -----------------------------------------------------------------------------
class SatelliteSimpleGeostationary(SatelliteBase):
    """
    Implements a simple geostationary satellite that simply stays above a fixed location on the equator.
    """

    def __init__(self, longitude_degrees: float):
        """
        Create a geostationary orbit that remains fixed over the the given longitude.

        Parameters
        ----------
        longitude_degrees:: float
            The geographic longitude of the geostationary orbit. expressed in degrees, +ve east.
        """

        super().__init__()
        radius = 42164000.0  # radius of orbit in meters
        theta = math.radians(longitude_degrees)
        omega = math.pi * 2.0 / self.period()  # angular velocaity radiancs.sec
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        vx = -radius * omega * math.sin(theta)
        vy = radius * omega * math.cos(theta)
        self._geoposition = np.array(
            [x, y, 0.0]
        )  # The geocentric, geographic location of the satellite
        self._geovelocity = np.array(
            [vx, vy, 0.0]
        )  # The geocentric, geographic velocity of the satellite

    # -----------------------------------------------------------------------------
    #               def update_eci_position(self, platform_utc:datetime):
    # -----------------------------------------------------------------------------
    def update_eci_position(self, utc: datetime):
        """
        Not usually used by users. Updates the ECI location of the satellite.

        Parameters
        ----------
        utc : datetime
            Updates the ECI position and ECI velocity of the satellite to this Coordinated Universal Time
        """
        ecipos = skretrieval.time.eci.itrf_to_eciteme(self._geoposition, utc)
        ecivel = skretrieval.time.eci.itrf_to_eciteme(self._geovelocity, utc)
        self._set_current_state(ecipos, ecivel, utc)

    # ------------------------------------------------------------------------------
    #           period
    # ------------------------------------------------------------------------------
    def period(self) -> float:
        """
        Returns the period of the geostationary orbit to be exactly one sidereal day.

        Returns
        -------
        float
            The period of the satellite in seconds
        """
        return 86164.091

    # ------------------------------------------------------------------------------
    #           eccentricity
    # ------------------------------------------------------------------------------
    def eccentricity(self) -> float:
        """
        Returns the eccentricity of the orbit. It will always be zero.
        """
        return 0.0
