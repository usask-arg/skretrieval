from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation


# ------------------------------------------------------------------------------
#           Class PlatformLocation
# ------------------------------------------------------------------------------
class PlatformLocation(ABC):
    """
    The base class used to specify the location and velocity, if appropriate, of platforms that carry instruments.
    This covers spacecraft, aircraft, balloons and ground sites. The class exposes two abstract functions
    that all classes derived form PlatformLocation must override,

        #. :meth:`~.PlatformLocation.update_position`.
        #. :attr:`~.PlatformLocation.position`

    In addition, platforms that move can override,

        #. :attr:`~.PlatformLocation.velocity`. Optional, defaults to [0.0, 0.0, 0.0]

    This class also provides the platform position as an astropy EarthLocation which allows latitude, longitude and height to be determined.

        #. :attr:`~.PlatformLocation.earth_location`
        #. :attr:`~.PlatformLocation.lat_lon_height`.
    """

    # -----------------------------------------------------------------------------
    #           __init__
    # -----------------------------------------------------------------------------

    def __init__(self):  # noqa: B027
        pass

    # ------------------------------------------------------------------------------
    #           update_position
    # ------------------------------------------------------------------------------

    @abstractmethod
    def update_position(self, utc: datetime | np.datetime64) -> np.ndarray | None:
        """
        Updates the geographic geocentric location of the platform at the given coordinated universal time and returns the
        new position. This is an abstract method and must be implemented in derived child classes.

        Parameters
        ----------
        utc : datetime
            The time at which to update the location of the platform

        Returns
        -------
        np.ndarray, Array[3]
            The three element X,Y,Z geographic geocentric location of the platform
        """
        logging.warning(
            "PlatformLocation.update_position, you are calling the base method which does nothing. You probably want to call the method in a derived class."
        )
        return None
        # raise Exception("Do not call base class method")

    # ------------------------------------------------------------------------------
    #           update_velocity
    # ------------------------------------------------------------------------------

    @abstractmethod
    def update_velocity(self, utc: datetime | np.datetime64) -> np.ndarray | None:
        """
        Updates the geographic geocentric location of the platform at the given coordinated universal time and returns the
        new position. This is an abstract method and must be implemented in derived child classes.

        Parameters
        ----------
        utc : datetime
            The time at which to update the location of the platform

        Returns
        -------
        np.ndarray, Array[3]
            The three element X,Y,Z geographic geocentric location of the platform
        """
        tnow = np.datetime64(utc)
        dt = np.timedelta64(1.0, "s")
        t1 = tnow + dt
        p0 = self.update_position(utc)
        p1 = self.update_position(t1)
        return p1 - p0

    # ------------------------------------------------------------------------------
    #           update_orientation
    # ------------------------------------------------------------------------------

    @abstractmethod
    def update_orientation(
        self, utc: datetime | np.datetime64
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Updates the orientation of the  at the given time and retursn an x,y,z ITRF unit vector.

        Parameters
        ----------
        utc : datetime
            The time at which to update the location of the platform

        Returns
        -------
        Tuple[ x:np.ndarray(3,),  y:np.ndarray(3,), z:np.ndarray(3,) ]
            A three element storing X,Y,Z unit vectors of the platform orientation. Returns None if there is an issue
            fetching the pointing data at the given time.
        """
        logging.warning(
            "PlatformLocation.update_orientation, you are calling the base method which does nothing. You probably want to call the method in a derived class."
        )
        return None

    # ------------------------------------------------------------------------------
    #           position
    # ------------------------------------------------------------------------------
    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        """
        Returns the geographic, ITRF, geocentric location calculated in the last call to :meth:`~.PlatformLocation.update_position`.
        This is an abstract method and must be implemented in a derived classes.

        Returns
        -------
        np.ndarray(3,)
            A three element array storing the X,Y,Z geocentric location (ITRF ecef, wgs84) of the platform. All numbers are in meters.
        """
        msg = "Do not call base class method"
        raise Exception(msg)

    # -----------------------------------------------------------------------------
    #           earth_location
    # -----------------------------------------------------------------------------
    @property
    def earth_location(self) -> EarthLocation:
        """
        Returns the current position of the platform as an Astropy EarthLocation. This is a convenience function
        wrapped around :attr:`~.position`

        Returns
        -------
        EarthLocation : Location of the platform as an astropy EarthLocation object.

        """
        geo = self.position
        return EarthLocation.from_geocentric(geo[0], geo[1], geo[2], unit=units.meter)

    # ------------------------------------------------------------------------------
    #           lat_lon_height
    # ------------------------------------------------------------------------------
    @property
    def lat_lon_height(self) -> np.ndarray:
        """
        Returns the current platform location as a 3 element array of  geodetic coordinates,
        latitude in degrees, longitude in degrees and height in meters. This is a convenience function
        wrapped around :attr:`~.position`

        Returns
        --------
        numpy.ndarray, Array[3,]
            A 3 element array of [ latitude, longitude, height in meters]

        """
        geoid = self.earth_location
        return np.array([geoid.lat.value, geoid.lon.value, geoid.height.value])

        # ------------------------------------------------------------------------------
        #           velocity
        # ------------------------------------------------------------------------------

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        """
        Returns the geocentric velocity calculated in the last call to :meth:`~.PlatformLocation.update_position`.
        This may be implemented in derived child classes to override the default method which returns an array of zeros,

        Returns
        -------
        np.ndarray, Array[3,]
            The three element X,Y,Z geographic geocentric velocity of the platform. All numbers are in meters/second.
        """
        return np.zeros([3])
