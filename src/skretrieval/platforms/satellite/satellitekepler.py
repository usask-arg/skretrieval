from __future__ import annotations

import logging
from datetime import datetime, timedelta
from math import (
    acos,
    atan2,
    cos,
    degrees,
    fabs,
    floor,
    log10,
    pi,
    pow,
    radians,
    sin,
    sqrt,
)

import numpy as np

import skretrieval.time
from skretrieval.time.eci import gmst

from .satellitebase import SatelliteBase


# -----------------------------------------------------------------------------
#           degrees360
# -----------------------------------------------------------------------------
def degrees360(rad_angle: float) -> float:
    degangle = degrees(rad_angle)
    if degangle < 0.0:
        degangle += 360.0
    return degangle


# -----------------------------------------------------------------------------
#           UnitVector
# -----------------------------------------------------------------------------
def UnitVector(v):
    return v / np.linalg.norm(v)


# -----------------------------------------------------------------------------
#           Magnitude
# -----------------------------------------------------------------------------
def Magnitude(v):
    return np.linalg.norm(v)


# -----------------------------------------------------------------------------
#           power_exponent
# -----------------------------------------------------------------------------
def value_exponent(x):
    if x == 0.0:
        expon = 0
        value = 0.0
        signstr = " "
    else:
        expon = int(floor(log10(fabs(x)) + 1.0))
        value = x / pow(10.0, expon)
        signstr = " " if value >= 0.0 else "-"
    return (value, expon, signstr)


# -----------------------------------------------------------------------------
#           class SatelliteKepler
# -----------------------------------------------------------------------------
class SatelliteKepler(SatelliteBase):
    """
    Implements a simple Kepler orbit orbiting the Earth. THe user will normally set all of the required
    orbital parameters in the instance constructor.
    """

    def __init__(
        self,
        ut: np.datetime64 | (datetime | (float | str)),
        period_from_seconds: float | None = None,
        period_from_altitude: float | None = None,
        inclination_radians: float | None = None,
        inclination_is_sun_sync: bool = False,
        mean_anomaly: float = 0.0,
        argument_of_perigee: float = 0.0,
        localtime_of_ascending_node_hours: float | None = None,
        longitude_of_ascending_node_degrees: float | None = None,
        right_ascension_ascending_node: float | None = None,
        eccentricity: float = 0.0000001,
        orbitnumber: int = 0,
    ):
        """
        Initializes the Kepler orbit. Multiple options are provided to describe the required orbit. A complete description
        of the various parameters is given in :meth:`~.from_elements`

        :param utc:  option identical to that described in :meth:`~.from_elements`
        :param period_from_seconds:  keyword option identical to that described in  :meth:`~.from_elements`
        :param period_from_altitude: keyword option identical to that described in  :meth:`~.from_elements`
        :param inclination_radians: keyword option identical to that described in  :meth:`~.from_elements`
        :param inclination_is_sun_sync: keyword option identical to that described in  :meth:`~.from_elements`
        :param mean_anomaly: keyword option identical to that described in  :meth:`~.from_elements`
        :param argument_of_perigee: keyword option identical to that described in  :meth:`~.from_elements`
        :param localtime_of_ascending_node_hours: keyword option identical to that described in  :meth:`~.from_elements`
        :param longitude_of_ascending_node_degrees: keyword option identical to that described in  :meth:`~.from_elements`
        :param right_ascension_ascending_node: keyword option identical to that described in  :meth:`~.from_elements`
        :param eccentricity: keyword option identical to that described in  :meth:`~.from_elements`
        :param orbitnumber: keyword option identical to that described in  :meth:`~.from_elements`
        """

        super().__init__()
        utc = skretrieval.time.ut_to_datetime(ut)
        self.m_epoch: datetime = None  # The time when the kepler elements are defined.
        self.m_mu: float = 3.98601210e14  # The gravitational parameter for Earth
        self.m_i: float = None  # inclination in radians
        self.m_raan: float = None  # Right Ascension of ascending node in radians
        self.m_argument_of_perigee: float = None  # Argument of perigee in radians
        self.m_N0: float = 0.0  # Mean motion in radians per second
        self.m_M0: float = (
            0.0  # Mean Anomaly in radians per second at the specified epoch
        )
        self.m_e: float = 0.0  # eccentricity
        self.m_a: float = 0.0  # semi-major axis in meters
        self.m_b: float = 0.0  # semi minor axis in meters
        self.m_h: float = 0.0  # angular momentum of the orbit. This is conserved for a pure kepler orbit
        self.m_xunit: np.ndarray = np.zeros(
            [3]
        )  # Get the unit vector pointing to the perigree (i.e. same direction as eccentricy vector)
        self.m_yunit: np.ndarray = np.zeros(
            [3]
        )  # Get the unit vector pointing along minor axis of ellips, (circa 90 degrees mean anomaly).
        self.m_zunit: np.ndarray = np.zeros(
            [3]
        )  # Get the unit vector perpendicular to the plane (ie in the direction of the angular momentum.

        self.from_elements(
            utc,
            period_from_seconds=period_from_seconds,
            period_from_altitude=period_from_altitude,
            inclination_radians=inclination_radians,
            inclination_is_sun_sync=inclination_is_sun_sync,
            mean_anomaly=mean_anomaly,
            argument_of_perigee=argument_of_perigee,
            localtime_of_ascending_node_hours=localtime_of_ascending_node_hours,
            longitude_of_ascending_node_degrees=longitude_of_ascending_node_degrees,
            right_ascension_ascending_node=right_ascension_ascending_node,
            eccentricity=eccentricity,
            orbitnumber=orbitnumber,
        )

    # --------------------------------------------------------------------------
    #        SatelliteKepler::orbital_period
    # --------------------------------------------------------------------------

    def period(self) -> timedelta:
        """
        Return the orbital period of this orbit. Uses keplers third law.

        Returns
        -------
        datetime.timedelta
            The orbital period.
        """
        return timedelta(
            seconds=2.0 * pi * sqrt(self.m_a * self.m_a * self.m_a / self.m_mu)
        )

    # -----------------------------------------------------------------------------
    #           eccentricity
    # -----------------------------------------------------------------------------

    def eccentricity(self) -> float:
        return self.m_e

    # --------------------------------------------------------------------------
    #             SatelliteKepler::self.eccentric_anomaly
    # --------------------------------------------------------------------------

    def _eccentric_anomaly(self, M: float, ecc: float, eps: float) -> float:
        """
        Calculate the eccentric anomaly by solving Kepler's equation using a
        Newton-Raphsom method.  Iterate to a precision specified by eps which
        is typically 1.0E-07 to 1.0E-10

        Parameters
        ----------
        M: float
            Mean Anomaly
        ecc: float
            Eccentricity
        eps: float
            Precision, typically 1.0E-07 to 1.0E-10

        Returns
        -------
        float
            The eccentric anomaly
        """

        E = M  # first guess (which is exact for a circle)
        delta = 1.0e20
        while fabs(delta) >= eps:  # and see if we have converged
            delta = E - ecc * sin(E) - M  # estimate the correction
            E -= delta / (1 - ecc * cos(E))  # get the corrected guess
        return E

    # -----------------------------------------------------------------------------
    #           sun_synchronous_inclination
    # -----------------------------------------------------------------------------

    def sun_synchronous_inclination(self, semi_major_axis_meters: float) -> float:
        """
        Method that returns the inclination of a sun synchronous orbit given the semi-major axis.
        For reference see `Sun-Synchronous orbit <https://en.wikipedia.org/wiki/Sun-synchronous_orbit>`_

        Parameters
        ----------
        semi_major_axis_meters: float

        Returns
        -------
        The required inclination of the orbit in radians
        """

        cosi = -pow(
            semi_major_axis_meters / 12352000.0, 3.5
        )  # see https://en.wikipedia.org/wiki/Sun-synchronous_orbit
        if cosi < -1:
            msg = (
                "Orbit semi major axis must be less than 12352 km for sun sync to work"
            )
            raise Exception(msg)
        return acos(cosi)

    # --------------------------------------------------------------------------
    #                   SatelliteKepler::update_eci_position
    # --------------------------------------------------------------------------

    def update_eci_position(self, autc: np.datetime64 | datetime):
        """
        Updates the ECI position
        """

        utc = skretrieval.time.ut_to_datetime(autc)
        if (self._m_time is None) or (
            utc != self._m_time
        ):  # Do we need to update eciposition or is it already the cached value
            dt = (utc - self.m_epoch).total_seconds()  # Get the difference in seconds
            M = self.m_M0 + self.m_N0 * dt  # Get the new mean Anomaly
            E = self._eccentric_anomaly(
                M, self.m_e, 1.0e-10
            )  # Solve Keplers equation to a precision of 1.0E-10 radians.

            x = self.m_a * (cos(E) - self.m_e)  # Get the x coordinate of the location
            y = self.m_b * sin(E)  # Get the y coordinate of the location
            r = sqrt(x * x + y * y)  # Get the radial distance
            k = self.m_mu / self.m_h
            vx = -k * y / r
            vy = k * (x / r + self.m_e)
            loc = self.m_xunit * x + self.m_yunit * y  # get the eci-location.
            vel = self.m_xunit * vx + self.m_yunit * vy  # get the eci-velocity
            self._set_current_state(loc, vel, utc)  # set the current state

    # -----------------------------------------------------------------------------
    #           make_two_line_elements
    # -----------------------------------------------------------------------------

    def make_two_line_elements(
        self, first_derivative=0.0, second_derivative=0.0, bstar=0.0, satnum=1
    ) -> tuple[str, str]:
        """
        Returns the current orbital elements as two line elements. This is handy if you want to initialize an SGP4
        predictor from a Kepler "starter orbit" using two line elements. The kepler orbit does not directly support
        drag terms, these have to be provided via the `bstar` parameter.

        Parameters
        ----------
        first_derivative: float
            The first derivative term of the SGP4 two line elements. This is usually not used and can be left as the default value of 0.0

        second_derivative: float
            The second derivative term of the SGP4 two line elements. This is usually not used and can be left as the default value of 0.0

        bstar: float
            The drag term used by SGP4. It is usually entered from empirical measurements. A value of 0.0 should disable drag terms in the SGP4 propagator

        satnum: int
            The satellite number. This is provide for convenience

        Returns
        --------
        tuple( str,str)
            Returns the two lines of the elements as a two element tuple of two strings
        """
        t = self.time
        t0 = datetime(t.year, 1, 1)
        dt = t - t0
        year = (t.year) % 100
        dayofyear = dt.total_seconds() / 86400.0 + 1.0  # January 1 is day 1, not day 0
        revsperday = (self.m_N0 * 86400.0) / (2 * pi)
        orbitnum = self.orbit_number(t)[0]
        secondderiv_value, secondderivexponent, secsign = value_exponent(
            second_derivative
        )
        bstar_value, bstarexponent, bstarsign = value_exponent(bstar)

        line1 = "1 {:5d}U 00001A   {:2d}{:12.8f}{:11.8f} {}{:0>5d}{:2d} {}{:0>5d}{:2d} 0    0X".format(
            satnum,
            year,
            dayofyear,
            first_derivative,
            secsign,
            int(fabs(secondderiv_value) * 1.0e5 + 0.5),
            secondderivexponent,
            bstarsign,
            int(fabs(bstar_value) * 1.0e5 + 0.5),
            bstarexponent,
        )
        line2 = "2 {:5d} {:8.4f} {:8.4f} {:07d} {:8.4f} {:8.4f} {:11.8f}{:5d}X".format(
            satnum,
            degrees(self.m_i),
            degrees360(self.m_raan),
            int(self.m_e * 1.0e7 + 0.5),
            degrees360(self.m_argument_of_perigee),
            degrees360(self.m_M0),
            revsperday,
            orbitnum,
        )
        #       Note that the checksum field has not been filled out in either line
        #        print(line1)
        #        print(line2)
        return (line1, line2)

    # --------------------------------------------------------------------------
    #                       SatelliteKepler::from_elements
    # --------------------------------------------------------------------------

    def from_elements(
        self,
        autc: datetime,
        period_from_seconds: float | None = None,
        period_from_altitude: float | None = None,
        inclination_radians: float | None = None,
        inclination_is_sun_sync: bool = False,
        mean_anomaly: float = 0.0,
        argument_of_perigee: float = 0.0,
        localtime_of_ascending_node_hours: float | None = None,
        longitude_of_ascending_node_degrees: float | None = None,
        right_ascension_ascending_node: float | None = None,
        eccentricity: float = 0.0000001,  # Eccentricity thats not quite zero avoids divide by zero problems in WXTRACK
        orbitnumber: int = 0,
    ):
        """
        Define the Kepler orbit from the classic 6 `Keplerian elements <https://en.wikipedia.org/wiki/Orbital_elements>`_.

        #. period
        #. inclination
        #. eccentricity
        #. mean anomaly
        #. right ascension of ascending node
        #. argument of perigee

        Several of the orbital elements can be specified in multiple ways. For example the orbital period can be
        defined directly with seconds or it can be defined using the altitude of the satellite. You must use only one
        of the options to specify a parameter otherwise exceptions will/should be raised.

        The caller must ensure that they explictly set the following three orbital elements,

        #. period or altitude must be set.
        #. inclination must be set.
        #. right ascension of the ascending node must be set.

        Leaving the remaining three elements, (i) mean anomaly, (ii) argument of perigee and (iii) eccentricity as default values
        produces a circular orbit with the satellite initially placed at the location of the perigee (which for a
        circular orbit is somewhat undefined and arbitrary!).

        Parameters
        ----------
        platform_utc: datetime.datetime
            The UTC time of the elements

        period_from_seconds : float
            specifies the orbital period in seconds. An alternative to specify the period is with the optional
            parameter `period_from_altitude`. One, but only one, of the two optional methods must be used.

        period_from_altitude : float
             specifies the orbital period using the altitude of the satellite in meters. The altitude is nominal as we do not
             account for oblateness of the Earth etc. The radius of the Earth is internally assumed to be 6378000.0 meters.
             An alternative to specify the period is with the optional parameter `period_from_seconds`.  One, but only one, of the
             two optional methods must be used.

        inclination_radians : float
            Specifes the inclination of the orbit in radians. An alternative method to specify the inclination is with the optional
            parameter `inclination_is_sun_sync`. One, but only one, of the two optional methods must be used.

        inclination_is_sun_sync : bool
             Default False. If True then specify the inclination of the orbit so it is sun-synchronous. This only
             works for orbits below an altitude of ~5974 km,.  An alternative is to specify the inclination with
             the optional parameter `inclination_radians`.  One, but only one, of the two optional methods may be used.
             Note this only sets the inclination of the orbit as a Kepler orbit propagator does not have the oblate Earth
             model (J2) necessary to model sun synchronicity.

        localtime_of_ascending_node_hours: float
            Default: None. Specifies the nominal local time of the ascending node in hours (0-24).  If set then its value
            represents the "hour" at which you want the ascending node for example a floating point value of 18.25 will
            set the node to 18:15 LT. This value is implemented by overwriting the "longitude_of_ascending_node_degrees" keyword
            used to initialize the kepler orbit. It also forces the satellite to be close to the ascending node at time
            `platform_utc` by changing the mean_anomaly value. Default is None.

        longitude_of_ascending_node_degrees: float
            Default None. Specifies the geographic longitude of the ascending node (-180 to 360)at time `platform_utc`. This is an alternative method
            to using parameters `localtime_of_ascending_node_hours` and `right_ascension_ascending_node` to specify the right ascension of the ascending node.
            One, but only one, of the 3 optional methods may be used, if neither option is used the RAAN is set to 0.0.
            If this option is used then the satellite is forced to be close to the ascending node at time `platform_utc` by changing the mean_anomaly value.
            Note that longitude is in degrees not radians.

        right_ascension_ascending_node: float
            Default None. Specifies the Right Ascension of the ascending node in radians (0 to 2.pi). This is an alternative method
            to parameter `longitude_of_ascending_node_degrees` to specify the right ascension of the ascending node.
            One, but only one, of the two optional methods must be used, if neither option is used the RAAN is set to 0.0.

        mean_anomaly: float
            The mean anomaly in radians at time mjd (0-2.pi). This value will be overidden if you use options
            `localtime_of_ascending_node_hours` or `longitude_of_ascending_node_degrees`.

        argument_of_perigee: float
             Default 0. The argument of perigree in radians (0 to 2.pi).

        eccentricity: float
            Default 0. The eccentricity of the orbit (0 to 1)

        orbitnumber: int
            Default 0. The orbit number at the epoch givern by `mjd`

        """
        # ---- determine the period from user supplied period or user supplied altitude

        utc = skretrieval.time.ut_to_datetime(autc)
        if period_from_seconds is not None:
            assert (
                period_from_altitude is None
            ), "You cannot set both period_from_altitude and period_from_seconds"
            N0 = 2 * pi / period_from_seconds
        elif period_from_altitude is not None:
            a = 6378135.0 + period_from_altitude
            N0 = sqrt(self.m_mu / (a * a * a))
        else:
            msg = "You must set one of (i) period_from_altitude or (ii) period_from_seconds"
            raise ValueError(msg)
        self.m_N0 = N0  # Get mean motion in radians per second
        self.m_a = pow(self.m_mu / (self.m_N0 * self.m_N0), 1.0 / 3.0)  # Get the semi

        # ---- determine the inclination of the orbit from user supplied inclination or request for sun-synchronous orbit

        if inclination_radians is not None:
            assert (
                inclination_is_sun_sync is False
            ), "You cannot set both inclination_radians and inclination_is_sun_sync"
            I0 = inclination_radians
        elif inclination_is_sun_sync:
            I0 = self.sun_synchronous_inclination(self.m_a)
        else:
            msg = "You must set one of (i) period_from_altitude or (ii) period_from_seconds"
            raise ValueError(msg)

        if localtime_of_ascending_node_hours is not None:
            assert (
                longitude_of_ascending_node_degrees is None
            ), "You cannot set both (i) localtime_of_ascending_node_hours and (ii) longitude_of_ascending_node_degrees "
            ut_ha = (
                float(utc.hour)
                + utc.minute / 60.0
                + (utc.second + utc.microsecond * 1.0e-06) / 3600.0
            )  # hour angle of Greenwich meridian
            deltaha = (
                localtime_of_ascending_node_hours - ut_ha
            )  # get the diffrence between desired hour angle of ascending node and greenwich
            longitude_of_ascending_node_degrees = deltaha * 15.0

        if longitude_of_ascending_node_degrees is not None:
            assert (
                right_ascension_ascending_node is None
            ), "You cannot set both (i) longitude_of_ascending_node_degrees (or localtime_of_ascending_node_hours)  and  (ii) right_ascension_ascending_node"
            st = gmst(utc)
            ra = (
                st + longitude_of_ascending_node_degrees / 15.0
            )  # Get RA at the longitude in hours (0-24)
            if ra >= 24.0:
                ra -= 24.0  # put it in the range 0-24
            if ra < 0.0:
                ra += 24.0
            RAAN = radians(ra * 15.0)  # and then convert to radians
            mean_anomaly = (
                -argument_of_perigee
            )  # Set the mean_anomaly so the satellite is close to the ascending node at the specified time
        elif right_ascension_ascending_node is not None:
            RAAN = right_ascension_ascending_node  # use the provided right ascension
        else:
            RAAN = 0.0  # Use 0 if no option is provided

        self.m_i = I0
        M0 = mean_anomaly
        if M0 < 0.0:
            M0 += 2.0 * pi
        W0 = argument_of_perigee
        E0 = eccentricity

        self.m_argument_of_perigee = argument_of_perigee
        self.m_raan = RAAN
        self.m_M0 = M0
        self.m_e = E0

        eciz = np.array([0.0, 0.0, 1.0])
        ecix = np.array([cos(RAAN), sin(RAAN), 0.0])  # Get vector to ascending node
        ytemp = np.array(
            [cos(RAAN + pi / 2.0), sin(RAAN + pi / 2.0), 0.0]
        )  # Get vector perpendicular to ascending node i the equatorial plane
        eciy = ytemp * cos(I0) + eciz * sin(
            I0
        )  # Get vector perpendicular to ascending node but in the orbital plane
        self.m_zunit = UnitVector(
            np.cross(ecix, eciy)
        )  # Get vector perpendicular to the orbital plane.
        self.m_xunit = UnitVector(
            ecix * cos(W0) + eciy * sin(W0)
        )  # Get the unit vector pointing at the perigee (Which is the semi major axis)
        self.m_yunit = np.cross(self.m_zunit, self.m_xunit)

        e2 = 1.0 - self.m_e * self.m_e
        self.m_b = self.m_a * sqrt(e2)
        self.m_h = sqrt(self.m_mu * self.m_a * e2)

        self._m_time = None
        self.m_epoch = utc
        self.set_orbit_number_from_last_equator_crossing(orbitnumber, self.m_epoch)
        self.update_eci_position(self.m_epoch)

    # --------------------------------------------------------------------------
    #                       from_state_vector
    # --------------------------------------------------------------------------

    def from_state_vector(
        self, autc: datetime, orbitnumber: int, r: np.ndarray, v: np.ndarray
    ):
        """
        Update the object so it calculates orbits derived from the specified state vector

        Parameters
        ----------
        platform_utc: datetime
            The coordinated universal time of the state vector.
        orbitnumber: int
            The orbit number at the epoch
        r:np.ndarray [3]
            ECI coordinates at epoch (in metres) eciposition of the satellite
        v:np.ndarray[3]
           ECI ecivelocity at epoch in m/s
        """

        logging.warning("Must update a few extra orbit parameters inside this code")
        utc = skretrieval.time.ut_to_datetime(autc)
        self.m_epoch = utc
        h = np.cross(
            r, v
        )  # Get the angular momentum vector (which is constant for the orbit)
        runit = UnitVector(r)  # Get the radial unit vector
        self.m_h = Magnitude(h)  # Get the magnitude of the angular ecivelocity
        self.m_zunit = UnitVector(
            h
        )  # Get the angular momementum unit vector (aka z axis of the orbital plane)
        ev = np.cross(v, h) / self.m_mu - runit  # Get the eccentricity vector
        self.m_e = Magnitude(ev)  # Get the eccentricity of the ellipse.
        if (
            self.m_e <= 0.0
        ):  # if the eccentricity is zero (or even less due to round off
            self.m_e = 0.0  # set the eccentricity to 0
            ev = runit  # and let the perigree be at this point
        self.m_xunit = UnitVector(
            ev
        )  # Get the eccentricity unitvector which points to the perigree (ie 0 degrees Mean Anomaly)
        self.m_yunit = np.cross(
            self.m_zunit, self.m_xunit
        )  # Get the y axis of the orbital plane from z cross x
        e2 = 1.0 - self.m_e * self.m_e  # 1 - e**2
        self.m_a = np.dot(h, h) / (self.m_mu * e2)  # Get the semi major axis
        self.m_b = self.m_a * sqrt(e2)  # Get the semi-minor axis
        self.m_N0 = sqrt(
            self.m_mu / (self.m_a * self.m_a * self.m_a)
        )  # Get the mean motion  (radians per second)

        x = np.dot(
            r, self.m_xunit
        )  # Get the x component parallel to the semi-major axis
        y = np.dot(
            r, self.m_yunit
        )  # Get the y component parallel to the semi-minor axis
        sinE = y / self.m_b
        cosE = x / self.m_a + self.m_e
        E = atan2(sinE, cosE)  # use y = b*sin(E) and x = a*(cos(E)-e) to get E
        self.m_M0 = E - self.m_e * sinE  # now get the mean anomaly at the epoch.

        self._m_time = None
        self.m_epoch = utc
        self.set_orbit_number_from_last_equator_crossing(orbitnumber, self.m_epoch)
        self.update_eci_position(self.m_epoch)
