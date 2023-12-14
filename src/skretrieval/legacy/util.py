from __future__ import annotations

import numpy as np
import sasktran as sk
import sasktran2 as sk2

from skretrieval.geodetic import geodetic


def _limb_viewing_los(
    los: sk.LineOfSight, forced_sun: np.array = None
) -> sk2.TangentAltitudeSolar:
    tangent_location = los.tangent_location()

    # TODO: Get this from astropy
    sun = None if forced_sun is None else forced_sun

    cos_sza = np.dot(tangent_location.local_up, sun)

    los_projected = los.look_vector - tangent_location.local_up * (
        los.look_vector.dot(tangent_location.local_up)
    )
    los_projected /= np.linalg.norm(los_projected)

    sun_projected = sun - tangent_location.local_up * (
        sun.dot(tangent_location.local_up)
    )
    sun_projected /= np.linalg.norm(sun_projected)

    y_axis = np.cross(tangent_location.local_up, sun_projected)

    saa = np.arctan2(y_axis.dot(los_projected), sun_projected.dot(los_projected))

    obs_geo = geodetic()
    obs_geo.from_xyz(los.observer)

    return sk2.TangentAltitudeSolar(
        tangent_altitude_m=tangent_location.altitude,
        relative_azimuth=saa,
        observer_altitude_m=obs_geo.altitude,
        cos_sza=cos_sza,
    )


def _ground_viewing_los(
    los: sk.LineOfSight, forced_sun: np.array = None
) -> sk2.GroundViewingSolar:
    ground_location = los.ground_intersection()

    # TODO: Get this from astropy
    sun = None if forced_sun is None else forced_sun

    cos_sza = np.dot(ground_location.local_up, sun)

    los_projected = los.look_vector - ground_location.local_up * (
        los.look_vector.dot(ground_location.local_up)
    )
    los_projected /= np.linalg.norm(los_projected)

    sun_projected = sun - ground_location.local_up * (sun.dot(ground_location.local_up))
    sun_projected /= np.linalg.norm(sun_projected)

    y_axis = np.cross(ground_location.local_up, sun_projected)

    saa = np.arctan2(y_axis.dot(los_projected), sun_projected.dot(los_projected))

    obs_geo = geodetic()
    obs_geo.from_xyz(los.observer)

    cos_zenith = np.abs(np.dot(los.look_vector, ground_location.local_up))

    return sk2.GroundViewingSolar(
        cos_sza,
        saa,
        cos_zenith,
        np.linalg.norm(obs_geo.location) - np.linalg.norm(ground_location.location),
    )


def convert_sasktran_legacy_geometry(
    legacy_geometry: sk.Geometry,
) -> sk2.ViewingGeometry:
    """

    Parameters
    ----------
    legacy_geometry : sk.Geometry
        _description_

    Returns
    -------
    sk2.ViewingGeometry
        _description_
    """
    sk2_geometry = sk2.ViewingGeometry()

    for los in legacy_geometry.lines_of_sight:
        if los.tangent_location() is not None:
            ray = _limb_viewing_los(los, legacy_geometry.sun)
        else:
            ray = _ground_viewing_los(los, legacy_geometry.sun)

        sk2_geometry.add_ray(ray)

    return sk2_geometry
