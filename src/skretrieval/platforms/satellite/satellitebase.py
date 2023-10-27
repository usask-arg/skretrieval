from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timedelta

import numpy as np
import scipy.optimize

import skretrieval.time.eci

from ..platform.platformlocator import PlatformLocation

# ----------------------------------------------------------------------------
#          class SatelliteBase                                  2004-11-23*/
#   SatelliteBase is a base class for artificial satellites orbiting the Earth.
#   The class depends upon derived classes to provide the calculation of eciposition and ecivelocity
#   at any instant in time. The class uses Earth Centred Intertial coordinates using the true equator and true
#   equinox.  Individual satellite predictors should ensure they are consistent
#   with true equator and mean equinox in all calculations.
# --------------------------------------------------------------------------*/


class SatelliteBase(PlatformLocation):
    """
    The SatelliteBase is a base class for artificial satellites orbiting the Earth. The class derives from
    :class:`~.PlatformLocation` which allows the SatelliteBase class and its derivatives to be  used by the *skretrieval*
    package to calculate the location of instruments mounted on satellites. The ECI-TEME coordinate system is used for
    storage of satellite position and velocity. There is some looseness in the definition of Earth Centered Inertial.
    We use the definition of ECI employed by the SGP4 orbital propagator code which is a Celestial Reference system using
    the true equator and mean equinox, TEME,  i.e. precession is accounted for but nutation is not. This system is not the same as
    the GCRS system commonly used in astronomy applications today so care must be taken when transforming coordinate systems. The
    conversion to and from ECI-TEME is provided by the `eci` functions in package `sktime` which have been checked to centimeter
    precision.

    The SatelliteBase class overrides and implements the following three functions of parent class :class:`~.PlatformLocation`

    #.  :attr:`~PlatformLocation.position`
    #.  :attr:`~PlatformLocation.velocity`
    #.  :meth:`~PlatformLocation.update_position`

    Note that the above three functions return answers in the geographic geocentric system and not the eci system.

    Child classes that implement specific satellite orbit propagator methods and derive from *SatelliteBase*  must
    implement the following abstract methods:

    #.  :meth:`~SatelliteBase.update_eci_position`
    #.  :attr:`~SatelliteBase.period`

    """

    def __init__(self):
        super().__init__()
        self._m_time: datetime = (
            None  # The time used in the last call to self.update_eci_position
        )
        self._temptime: datetime = (
            None  # Local variable used for internal calculations, eg _zcomponent
        )
        self._m_orbit_number_at_start_time: int = (
            None  # Orbit number at time *_m_start_time_of_orbit*
        )
        self._m_start_time_of_orbit: datetime = None  # The Start time of orbitnumber.
        self._m_ecivelocity: np.ndarray = np.zeros(
            [3]
        )  # The ECI velocity in m/s generated in the last call to self.update_eci_position
        self._m_ecilocation: np.ndarray = np.zeros(
            [3]
        )  # The ECI position in meters generated in the last call to self.update_eci_position

    # -----------------------------------------------------------------------------
    #           time
    # -----------------------------------------------------------------------------
    @property
    def time(self) -> datetime:
        """
        Returns the UTC time of the current state vector

        Returns
        -------
        datetime.datetime
            The time of the current state vector
        """
        return self._m_time

    # -----------------------------------------------------------------------------
    #           geoposition
    # -----------------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """
        Returns the position of the satellite in meters in the geocentric geographic ECEF
        coordinate system.

        Returns
        -------
        np.ndarray[3]
            The three element array specifying (X,Y,Z) in meters
        """
        return skretrieval.time.eci.eciteme_to_itrf(self._m_ecilocation, self._m_time)

    # -----------------------------------------------------------------------------
    #           geovelocity
    # -----------------------------------------------------------------------------
    @property
    def velocity(self) -> np.ndarray:
        """ECEFITRF/GEO coordinate system.

        ..  note::

            The velocity is the same vector as calculated for the ECI/TEME frame. It does not include the component
            due to the rotation of the Earth.

        Returns
        -------
        np.ndarray[3]
            The three element array specifying (X,Y,Z) in meters
        """
        return skretrieval.time.eci.eciteme_to_itrf(self._m_ecivelocity, self._m_time)

    # -----------------------------------------------------------------------------
    #           update_position
    # -----------------------------------------------------------------------------

    def update_position(self, utc: datetime | (np.datetime64 | float)) -> np.ndarray:
        """
        Overloads the methods from :class:`~.PlatformLocation`. Updates the position of the satellite to the requested time.
        The geocentric ECEF/ITRF position of the satellite can be retrieved with attribute :attr:`~position`. It is
        strongly recommended that time is always increasing in reasonable step sizes. Various orbit propagators will not
        step backwards in time  and may fail if the time step is unreasonably large.

        Parameters
        -----------
        utc : datetime
            The time at which to calculate the new position.

        """
        ut = skretrieval.time.ut_to_datetime(
            utc
        )  # The satellite code is based upon datetime, so ensure we havethe right time units
        self.update_eci_position(ut)
        return self.position

    # ------------------------------------------------------------------------------
    #           update_velocity
    # ------------------------------------------------------------------------------

    def update_velocity(
        self, utc: datetime | (np.datetime64 | float)
    ) -> np.ndarray | None:
        """
        Overloads the method from :class:`~.PlatformLocation`. Updates the velocity of the satellite to the requested time.
        The geocentric ECEF/ITRF velocity of the satellite can be retrieved with attribute :attr:`~velocity`. It is
        strongly recommended that time is always increasing in reasonable step sizes. Various orbit propagators will not
        step backwards in time  and may fail if the time step is unreasonably large.

        Parameters
        -----------
        utc : datetime
            The time at which to calculate the new velocity.
        """

        ut = skretrieval.time.ut_to_datetime(utc)
        self.update_eci_position(ut)
        return self.velocity

    # ------------------------------------------------------------------------------
    #           update_orientation
    # ------------------------------------------------------------------------------

    def update_orientation(
        self, utc: datetime | np.datetime64  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Returns a platform orientation that has no modification.
        """
        return (
            np.array((1.0, 0.0, 0.0)),
            np.array((0.0, 1.0, 0.0)),
            np.array((0.0, 0.0, 1.0)),
        )

    # -----------------------------------------------------------------------------
    #           update_eci_position
    # -----------------------------------------------------------------------------
    @abstractmethod
    def update_eci_position(self, utc: datetime):
        """
        All satellite classes must implement update_eci_position. This purely abstract method  will  update the internal attributes
        self._m_ecilocation, self._m_ecivelocity and self._m_time

        Parameters
        ----------
        utc : datetime
            Updates the ECI eciposition and ECI ecivelocity of the satellite to this Coordinated Universal Time
        """

        msg = "SatelliteBase::update_eci_position, Do not call base class method"
        raise Exception(msg)

    # ------------------------------------------------------------------------------
    #           period
    # ------------------------------------------------------------------------------
    @abstractmethod
    def period(self) -> timedelta:
        """
        Purely abstract method that returns the orbital period of the satellite in seconds.  For low earth orbit spacecraft
        this only needs to be accurate to ~0.1 seconds.

        Returns
        -------
        timedelta
            The period of the satellite in seconds
        """
        msg = "SatelliteBase::period, Do not call base class method"
        raise Exception(msg)

    @abstractmethod
    def eccentricity(self) -> float:
        """
        Purely abstract method that returns the eccentricty of the satellite orbit. This is used in the base class
        when locating the next crossing of teh ascending node and only requires moderate precision.

        Returns
        -------
        float
            The eccentricty of the satellite orbit in seconds
        """
        msg = "SatelliteBase::period, Do not call base class method"
        raise Exception(msg)

    # -----------------------------------------------------------------------------
    #           eciposition
    # -----------------------------------------------------------------------------
    def eciposition(self) -> np.ndarray:
        """
        The ECI cartesian position of the satellite in meters after the last call to :meth:`~.update_eci_position`.

        Returns
        -------
        np.ndarray[3]
            The three element array specifying ECI location as (X,Y,Z) meters
        """
        return self._m_ecilocation.copy()

    # ------------------------------------------------------------------------------
    #           ecivelocity
    # ------------------------------------------------------------------------------
    def ecivelocity(self) -> np.ndarray:
        """
        The ECI cartesian velocity of the satellite in meters per second

        Returns
        -------
        np.ndarray[3]
            The three element array specifying ECI velocity as (X,Y,Z) meters per second
        """
        return self._m_ecivelocity.copy()

    # -----------------------------------------------------------------------------
    #           _set_current_state
    # -----------------------------------------------------------------------------
    def _set_current_state(self, ecipos: np.ndarray, ecivel: np.ndarray, t: datetime):
        """
        Sets the ECI position, ECI velocity and current time of the satellite. This is typically
        called by child satellite implementation classes at the end of a call to method *update_eci_position*

        Parameters
        ----------
        ecipos : np.ndarray[3]
            The three element (X,Y,Z) array specifying ECI position in meters .
        ecivel : np.ndarray[3]
            The three element (X,Y,Z) array specifying ECI velocity in meters/second
        t  : datetime.datetime
            The current UTC time of the eciposition and ecivelocity.

        """
        self._m_ecilocation[
            :
        ] = ecipos  # Copy the values to ensure we have 3 element arrays
        self._m_ecivelocity[
            :
        ] = ecivel  # Copy the values to ensure we have 3 element arrays
        self._m_time = t

    # ------------------------------------------------------------------------------
    #      nxSatelliteBase::ZComponent
    # ------------------------------------------------------------------------------
    def _zcomponent(self, deltamjd: float) -> float:
        """
        Return the Z component of the satellite at a time deltamjd from
        the current epoch.  This is often used to find the ascending node crossing
        of the satellite.

        Parameters
        -----------
        deltamjd:float
            The delta time in seconds

        Returns
        -------
        float
            The Z component of the satellite ECI position. Negative means it is below the equator. Positive means it is above the equator.
        """

        Tnow = self._temptime + timedelta(
            seconds=deltamjd
        )  # Set up a new self.m_time to get satellite eciposition.
        self.update_eci_position(
            Tnow
        )  # Get the satellite eciposition atthis self.m_time.
        return self._m_ecilocation[2]  # and return the Z component.

    # ----------------------------------------------------------------------------
    #     nxSatelliteBase::equator_crossing
    # --------------------------------------------------------------------------
    def equator_crossing(self, utc: datetime) -> datetime:
        """
        Determines the last equator crossing before or equal to Tnow.

        Parameters
        ----------
        Tnow: datetime.datetime
            Find the equator crossing before (or at) this time

        Returns
        -------
        datetime.datetime
            The time at which this satellite object crosses the equator
        """

        self._temptime = utc
        ecc = self.eccentricity()
        frac = (
            0.2 if ecc < 0.1 else 0.01
        )  # small eccentricty we can do big steps. Large eccentricity do small steps
        deltat = (frac * self.period()).total_seconds()
        positiveT = (utc - self._temptime).total_seconds()
        z = -1
        while (
            z < 0.0
        ):  # ----  Find when Z component is positive before or equal to now.
            z = self._zcomponent(positiveT)
            if z < 0.0:
                positiveT = positiveT - deltat

        negativeT = (
            positiveT - deltat
        )  # ---- Find when Z component is negative before positiveT.
        z = 1.0
        while z >= 0.0:
            z = self._zcomponent(negativeT)
            if z >= 0.0:
                negativeT = negativeT - deltat

        crosstime = scipy.optimize.brentq(
            self._zcomponent, negativeT, positiveT, xtol=1.0e-12
        )
        positiveT = self._temptime + timedelta(seconds=crosstime)
        self.update_eci_position(positiveT)
        return positiveT

    # -----------------------------------------------------------------------------
    #           set_orbit_number_from_last_equator_crossing
    # -----------------------------------------------------------------------------
    def set_orbit_number_from_last_equator_crossing(
        self, orbitnumber: int, utc: datetime
    ):
        """
        Sets the orbit number and calculates the start time/ascending node of the orbit.

        Parameters
        ----------
        orbitnumber: int
            The orbit number at time Tnow
        Tnow: datetime.datetime
            The current time.
        """

        self._m_orbit_number_at_start_time = orbitnumber
        self._m_start_time_of_orbit = self.equator_crossing(utc)

    # ----------------------------------------------------------------------------
    #        nxSatelliteBase::orbit_number      2004-11-23*/
    # --------------------------------------------------------------------------

    def orbit_number(self, utc: datetime) -> tuple[int, datetime]:
        """
         Calculates the orbit number at time Tnow and returns the start time
         of that orbit.

        Parameters
        ----------
        utc : datetime
            The Coordinated Universal Time at which to calculate the orbit number

        Returns
        -------
        (int,datetime)
            Returns a two element tuple specifiying the orbit number and the time of the last equator crossing
        """
        Start = self.equator_crossing(utc)
        DeltaT = Start - self._m_start_time_of_orbit
        norbits = int(DeltaT / self.period() + 0.5)
        return (norbits + self._m_orbit_number_at_start_time, Start)
