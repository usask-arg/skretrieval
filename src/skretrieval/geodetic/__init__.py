from __future__ import annotations

import numpy as np
import sasktran as sk


def geodetic() -> sk.Geodetic:
    return sk.Geodetic("wgs84")


def target_lat_lon_alt(
    los_vector: np.array, obs_position: np.array
) -> tuple[float, float, float]:
    """
    Parameters
    ----------
    los_vector : np.array
        3 Element look vector in ECI
    obs_position : np.array
        3 element observer position in ECI

    Returns
    -------
    Tuple[float, float, float]
        Tangent latitude/longitude/altitude if limb looking. Ground latitude/longitude if nadir looking
    """

    los = sk.LineOfSight(mjd=0, observer=obs_position, look_vector=los_vector)

    location = los.tangent_location()

    if location is None:
        location = los.ground_intersection(0)

    return location.latitude, location.longitude, location.altitude
