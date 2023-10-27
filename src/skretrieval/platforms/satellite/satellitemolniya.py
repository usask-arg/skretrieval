from __future__ import annotations

from datetime import datetime
from math import radians

import numpy as np

import skretrieval.time

from .satellitebase import SatelliteBase
from .satellitekepler import SatelliteKepler
from .satellitesgp4 import SatelliteSGP4


# ------------------------------------------------------------------------------
#           class SatelliteMolniya
# ------------------------------------------------------------------------------
class SatelliteMolniya(SatelliteBase):
    """
    Implements a Molniya orbit. This model can internally utilize either a Keplerian or SGP4 orbit propagator.
    The SGP4 propagator will capture gravitational perturbations on the orbit but the Kepler will not. The user will
    typically initialize the molniya orbit in the constructor but this simply calls
    :meth:`~.set_molniya_from_elements`. A full desccription of the optional keyword arguments
    is given in that function. A sensible default Molniya orbit is provided in the defaults
    """

    def __init__(
        self,
        utc: datetime | (np.datetime64 | (float | str)),
        orbittype="kepler",
        period_from_seconds: float = 718.0 * 60.0,
        inclination_user_defined: float = radians(63.4),
        argument_of_perigee: float = radians(270.0),
        eccentricity: float = 0.74,
        sgp4_bstar_drag: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        ut = skretrieval.time.ut_to_datetime(utc)
        self.satellite = None
        self.set_molniya_from_elements(
            ut,
            orbittype=orbittype,
            period_from_seconds=period_from_seconds,
            inclination_user_defined=inclination_user_defined,
            argument_of_perigee=argument_of_perigee,
            eccentricity=eccentricity,
            sgp4_bstar_drag=sgp4_bstar_drag,
            **kwargs,
        )

    # -----------------------------------------------------------------------------
    #           update_eci_position
    # -----------------------------------------------------------------------------

    def update_eci_position(self, tnow: datetime):
        """
        Updates the eciposition and ecivelocity of the satellite at the given instant in time.

        Parameters
        ----------
        tnow : datetime
            Updates the ECI eciposition and ECI ecivelocity of the stellite to this time
        """
        self.satellite.update_eci_position(tnow)
        self._set_current_state(
            self.satellite.eciposition(), self.satellite.ecivelocity(), tnow
        )

    # ------------------------------------------------------------------------------
    #           period
    # ------------------------------------------------------------------------------

    def period(self) -> float:
        """
        Return the period of the satellite.  For low earth orbit spacecraft
        this is accurate to about 0.1 seconds.

        Returns
        -------
        datetime.timedelta
            The period of the satellite
        """
        return self.satellite.period()

    # ------------------------------------------------------------------------------
    #           eccentricity
    # ------------------------------------------------------------------------------

    def eccentricity(self) -> float:
        return self.satellite.eccentricity()

    # -----------------------------------------------------------------------------
    #           set_sun_sync_from_elements
    # -----------------------------------------------------------------------------

    def set_molniya_from_elements(
        self,
        utc: datetime,
        orbittype="kepler",
        period_from_seconds: float = 718.0 * 60.0,
        inclination_user_defined: float = radians(63.4),
        argument_of_perigee: float = radians(270.0),
        eccentricity: float = 0.74,
        sgp4_bstar_drag: float = 0.0,
        **kwargs,
    ):
        """
        Defines the Molniya orbit using Keplerian orbital elements. A default, standard Molniya orbit is provided via the default
        although the user must supply the right ascension of ascending node using one of the standard :class:`~.SatelliteKepler`
        options (i) **localtime_of_ascending_node_hours**, (ii) **longitude_of_ascending_node_degrees** or (iii) **right_ascension_ascending_node**

        The Molniya orbit is always kick-started using Keplerian orbital elements. It can continue to use the keplerian
        orbit to propagate the position and velocity or it can switch to an SGP4 model orbit propagator.

        Parameters
        ----------
        utc : datetime.datetime
            The universal time of the orbital elements.
        orbittype: str
            Specifies the type of orbit predictor to be used to propagate the orbit. It can be either 'kepler' or 'SGP4'. Default is 'kepler'
        sgp4_bstar_drag: float
            Optional argument that is only used if the `sgp4` propagator is selected. This value is used as the bstar drag
            term in the SGP4 orbit predictor. The default value is 0.0 implying drag is ignored.
        kwargs: extra key word arguments.
            These keywords are from the keyword options in :meth:`.SatelliteKepler.from_elements`.
        """
        kwargs["period_from_seconds"] = period_from_seconds
        kwargs["inclination_radians"] = inclination_user_defined
        kwargs["inclination_is_sun_sync"] = False
        kwargs["argument_of_perigee"] = argument_of_perigee
        kwargs["eccentricity"] = eccentricity
        kepler = SatelliteKepler(utc, **kwargs)
        if orbittype.lower() == "kepler":
            self.satellite = kepler
        elif orbittype.lower() == "sgp4":
            sgp4 = SatelliteSGP4()
            sgp4.from_kepler_orbit(kepler, bstar=sgp4_bstar_drag)
            self.satellite = sgp4
        else:
            msg = f"the requested type of orbit predictor {orbittype} is not supported"
            raise Exception(msg)
