from __future__ import annotations

from datetime import datetime

import astropy.coordinates
import astropy.units as units
import numpy as np
from astropy.time import Time

astropy.coordinates.solar_system_ephemeris.set(
    "jpl"
)  # Ensure the JPL ephemerides are used by default


# -----------------------------------------------------------------------------
#           planetary_body_itrf
# -----------------------------------------------------------------------------
def solsys_body_vector_itrf(utc: datetime, body: str) -> np.ndarray:
    """
    Fetches the ITRF geocentric vector from the center of the Earth to the solar system body at the requested time.

    Parameters
    ----------
    utc : datetime, np.datetime64, str
        The UTC time at which the position of the body is required
    body: str
        The name of the solar system body, e.g. "moon", "sun"

    Returns
    -------
    np.ndarray(3): The geocentric positon of the requested body as a 3 element column vector. Expressed in meters.
    """

    t = Time(utc)
    coord = astropy.coordinates.get_body(body, t, ephemeris="jpl")
    itrs = astropy.coordinates.ITRS(representation_type="cartesian")
    itrscoord = coord.transform_to(itrs)
    return np.array(
        (
            itrscoord.x.to(units.meter).value,
            itrscoord.y.to(units.meter).value,
            itrscoord.z.to(units.meter).value,
        )
    )


def star_unitvector_itrf(utc: datetime, body: str) -> np.ndarray:
    """
    Fetches the ITRF geocentric unit vector from the center of the Earth to the desired star at the requested time.
    This currently does not account for **proper motion** of the star.

    Parameters
    ----------
    utc : datetime, np.datetime64, str
        The UTC time at which the position of the star is required
    body: str
        The name of the star, e.g. "Betelgeuse"

    Returns
    -------
    np.ndarray(3): The geocentric unit vector toward the requested star as a 3 element column vector. Dimensionless Unit vector.
    """

    t = Time(utc)
    star = astropy.coordinates.SkyCoord.from_name(
        body
    )  # **** DOES NOT SEEM TO INCLUDE proper motion corrections
    itrs = astropy.coordinates.ITRS(representation_type="cartesian", obstime=t)
    itrscoord = star.transform_to(itrs)
    return np.array((itrscoord.x.value, itrscoord.y.value, itrscoord.z.value))
