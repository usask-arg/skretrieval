from __future__ import annotations

from datetime import datetime

import numpy as np

import skretrieval.time

from .satellitebase import SatelliteBase
from .satellitekepler import SatelliteKepler
from .satellitesgp4 import SatelliteSGP4


# ------------------------------------------------------------------------------
#           class SatelliteKepler
# ------------------------------------------------------------------------------
class SatelliteSunSync(SatelliteBase):
    """
    Implements a sun-synchronous orbit. This model starts the chosen  orbit propagator with
    appropriate initial conditions given by the user. The orbit inclination is chosen by the code
    so the orbit with the given period (or altitude) is sun-synchronous. The user can utilize
    a Keplerian or SGP4 orbit propagator. The kepler propagator is useful for short, one orbit, studies but will not
    actually precess the orbit. The SGP4 propagator is slower but will capture the sun-synchronous precession over
    a period of days and months.
    """

    def __init__(
        self,
        utc: np.datetime64 | (datetime | (float | str)),
        orbittype="sgp4",
        sgp4_bstar_drag=0.0,
        **kwargs,
    ):
        """

        Parameters
        -----------
        utc: datetime.datetime
            The UTC time of the elements. This is set as the time of the ascending node of the selected orbit.

        period_from_seconds : float
            Optional, specifies the orbital period in seconds. An alternative to specify the period is with the optional
            parameter `period_from_altitude`. One, but only one, of the two optional methods must be used.

        period_from_altitude : float
             Optional, specifies the orbital period using the altitude of the satellite in meters. The altitude is nominal as we do not
             account for oblateness of the Earth etc. The radius of the Earth is internally assumed to be 6378000.0 meters.
             An alternative to specify the period is with the optional parameter `period_from_seconds`.  One, but only one, of the
             two optional methods must be used.

        orbittype: str
            (default='sgp4'). User can choose the orbit propagator. Currently accepted values are 'kepler' and 'sgp4'.

        sgp4_bstar_drag : float
            (default 0.0).  Allows the user to add a drag term. This value is only used by the SGP4 predictor and must
            be compatible with the SGP4 predictor.

        localtime_of_ascending_node_hours: float
            Specifies the nominal local time of the ascending node in hours (0-24).  If set then its value
            represents the "hour" at which you want the ascending node for example a floating point value of 18.25 will
            set the node to 18:15 LT. The current implementation may not be exact but will be close.

        longitude_of_ascending_node_degrees: float
            Specifies the geographic longitude of the ascending node (-180 to 360)at time `platform_utc`. This is an alternative method
            to using parameters `localtime_of_ascending_node_hours` and `right_ascension_ascending_node` to specify the right ascension of the ascending node.
            One, but only one, of the 3 optional methods may be used, if neither option is used the RAAN is set to 0.0.

        right_ascension_ascending_node: float
            Specifies the Right Ascension of the ascending node in radians (0 to :math:`2\\pi`). This is an alternative method
            to parameter `longitude_of_ascending_node_degrees` to specify the right ascension of the ascending node.
            One, but only one, of the two optional methods must be used, if neither option is used the RAAN is set to 0.0.

        argument_of_perigee: float
             Default 0. The argument of perigree in radians (0 to :math:`2\\pi`).

        eccentricity: float
            Default 0. The eccentricity of the orbit (0 to 1)

        orbitnumber: int
            Default 0. The orbit number at the epoch given by `platform_utc`


        """
        super().__init__()
        ut = skretrieval.time.ut_to_datetime(utc)
        self.satellite = None
        self.set_sun_sync_from_elements(
            ut, orbittype=orbittype, sgp4_bstar_drag=sgp4_bstar_drag, **kwargs
        )

    # -----------------------------------------------------------------------------
    #           update_eci_position
    # -----------------------------------------------------------------------------

    def update_eci_position(self, tnow: datetime):
        """
        Updates the eciposition and ecivelocity of the satellite at the given instant in time.  The user does not
        usually call this function directly. It is usually called by the base class :class:`~.SatelliteBase`

        Parameters
        ----------
        tnow : datetime
            Updates the ECI eci-position and ECI eci-velocity of the satellite to this time
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

    # -----------------------------------------------------------------------------
    #           eccentricity
    # -----------------------------------------------------------------------------

    def eccentricity(self) -> float:
        return self.satellite.eccentricity()

    # -----------------------------------------------------------------------------
    #           set_sun_sync_from_elements
    # -----------------------------------------------------------------------------

    def set_sun_sync_from_elements(
        self,
        utc: datetime,
        orbittype: str = "sgp4",
        sgp4_bstar_drag: float = 0.0,
        **kwargs,
    ):
        """
        Defines the sun-synchronous orbit using Keplerian orbital elements. The sun-synchronous orbit is always
        kick-started using a Kepler based orbit and it can continue to use that kepler orbit to propagate the position
        and velocity or it can use the SGP4 model. This function is called by the class instance constructor and the
        user does not normally need to call this function directly.

        Parameters
        ----------
        utc : datetime.datetime
            The universal time of the orbital elements.


        orbittype: str
            Specifies the type of orbit predictor to be used to propagate the orbit. It can be either 'kepler' or 'sgp4'. Default is 'sgp4'

        sgp4_bstar_drag: float
            Optional argument that is only used if the `sgp4` propagator is selected. This value is used as the bstar drag
            term in the SGP4 orbit predictor. The default value is 0.0 implying drag is ignored.

        kwargs: extra key word arguments.
            These keywords are from the keyword options in :meth:`.SatelliteKepler.from_elements`.
            Note that parameter `localtime_of_ascending_node` will override, if set, the `longitude_of_ascending_node_degrees`
            setting in the key words options.


        """

        kepler = SatelliteKepler(utc, inclination_is_sun_sync=True, **kwargs)
        if orbittype.lower() == "kepler":
            self.satellite = kepler
        elif orbittype.lower() == "sgp4":
            sgp4 = SatelliteSGP4()
            sgp4.from_kepler_orbit(kepler, bstar=sgp4_bstar_drag)
            self.satellite = sgp4
        else:
            msg = f"the requested type of orbit predictor {orbittype} is not supported"
            raise Exception(msg)
