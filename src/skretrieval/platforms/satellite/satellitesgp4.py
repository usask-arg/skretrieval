from __future__ import annotations

from datetime import datetime, timedelta
from math import pi

import numpy as np
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

import skretrieval.time

from .satellitebase import SatelliteBase
from .satellitekepler import SatelliteKepler


# -----------------------------------------------------------------------------
#           class SatelliteKepler
# -----------------------------------------------------------------------------
class SatelliteSGP4(SatelliteBase):
    """
    Implements the SGP4 satellite propagator. This is just a shallow wrapper for the `SGP4 python package <https://pypi.org/project/sgp4/>`_
    from PyPi
    """

    def __init__(self, twolines: tuple[str, str] | None = None):
        """
        Creates and initializes the sgp4 orbit predictor using two line elements provided by the user. The lines must be in the
        promper two line format. It is possible to leave the lines as None and initialize the predictor after creation.

        Parameters
        ----------
        twolines: Tuple( str, str )
            Two strings that contain the two line elements used to initialize the sgp4 orbit predictor. This paremeter
            may be None, in which case, the object can be configured by calling :meth:`~.from_kepler_orbit` or :meth:`~.from_twoline_elements`

        """
        super().__init__()
        self.m_sgp4 = None
        if twolines is not None:
            self.from_twoline_elements(twolines[0], twolines[1])

    # -----------------------------------------------------------------------------
    #           from_kepler_orbit
    # -----------------------------------------------------------------------------

    def from_kepler_orbit(self, kepler: SatelliteKepler, bstar=0.0):
        """
        Initializes the sgp4 orbit predictor with two line elements extracted from a kepler orbit.
        A simple algorithm that extracts two line elements from a kepler orbit and feeds them
        into the SGP4 model. Note that the kepler orbit does not correct coordinates to the true equator mean equinox (TEME)
        coordinates used by sgp4 so there will be minor issues for people seeking precision. On the other hand it is a
        simple way to fire up the SGP4 from a Kepler orbit, which is quite useful when modelling simple classes of orbit for
        study work.

        Parameters
        ----------
        kepler : SatelliteKepler
            The Kepler elements for the satellite. The two line elements are generated for the current time and eciposition
            stored in the kep[ler satellite.

        bstar:float
            The B* drag term used by SGP4. This is not part of the Kepler elements and must be manually included by
            the user. We recommend you leave this as default 0.0 unless you know what you are doing as the drag term
            used by SGP4 is tricky to get right (it can change by one or two orders of magnitude for example)

        """
        twolines = kepler.make_two_line_elements(bstar=bstar)
        self.from_twoline_elements(twolines[0], twolines[1])

    # -----------------------------------------------------------------------------
    #           from_twoline_elements
    # -----------------------------------------------------------------------------

    def from_twoline_elements(self, line1: str, line2: str):
        """
        Initializes the sgp4 orbit predictor with classic two line elements.

        Parameters
        ----------
        line1:str
            The first line of the two line elements

        line2:str
            The second line of the two line elements

        """
        self.m_sgp4 = twoline2rv(line1, line2, wgs84)

        orbit_number = int(line2[63 : 63 + 5])  # Get the orbit number of this orbit.
        self.set_orbit_number_from_last_equator_crossing(
            orbit_number, self.m_sgp4.epoch
        )
        self.update_eci_position(self.m_sgp4.epoch)

    # ---------------------------------------------------------------------------
    #             SatelliteKepler::orbital_period
    # ---------------------------------------------------------------------------

    def period(self) -> timedelta:
        """
        Return the orbital period of this orbit using the mean motion

        Returns
        -------
        float
            The orbital period in seconds
        """
        mean_motion = (
            self.m_sgp4.no / 60.0
        )  # Convert sgp4 mean motion from radians/minute to radians per second
        return timedelta(seconds=2.0 * pi / mean_motion)

    # -----------------------------------------------------------------------------
    #           eccentricity
    # -----------------------------------------------------------------------------

    def eccentricity(self) -> float:
        return self.m_sgp4.ecco

    # --------------------------------------------------------------------------
    #            SatelliteKepler::update_eci_position
    # --------------------------------------------------------------------------

    def update_eci_position(self, autc: datetime):
        utc = skretrieval.time.ut_to_datetime(autc)
        if (self._m_time is None) or (
            utc != self._m_time
        ):  # Do we need to update eciposition or is it already the cached value
            r, v = self.m_sgp4.propagate(
                year=utc.year,
                month=utc.month,
                day=utc.day,
                hour=utc.hour,
                minute=utc.minute,
                second=(utc.second + utc.microsecond / 1.0e6),
            )
            pos = np.array(r) * 1000.0
            vel = np.array(v) * 1000.0
            self._set_current_state(pos, vel, utc)
