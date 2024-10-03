from __future__ import annotations

import numpy as np
from sasktran2.geodetic import WGS84


def geodetic() -> WGS84:
    return WGS84()


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
    location = geodetic()

    location.from_tangent_point(obs_position, los_vector)

    if location.altitude > 0:
        return location.latitude, location.longitude, location.altitude

    location.from_xyz(location.altitude_intercepts(0, obs_position, los_vector)[0])

    return location.latitude, location.longitude, location.altitude
