from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..orientation_techniques import PointingAlgorithms
    from ..platform import Platform

import numpy as np
import scipy.optimize

from skretrieval.geodetic import geodetic
from skretrieval.time import datetime64_to_datetime

from ..platform.rotationmatrix import UnitVectors


# ------------------------------------------------------------------------------
#           ObserverBearingFunction
# ------------------------------------------------------------------------------
class ObserverBearingFunction:
    """
    An internal callable object that is used when calculating bearings of tangent points from the observer
    """

    def __init__(self, tanlat, tanlng, tanheight, observerheight, observer_bearing):
        self.target_observer_bearing = observer_bearing
        self._geo = geodetic()
        self._geo.from_lat_lon_alt(tanlat, tanlng, tanheight)
        self.tpnorth = -self._geo.local_south
        self.tpeast = -self._geo.local_west
        self.tplocation = self._geo.location
        self.obslocation = None
        self.observerheight = observerheight

        if self.target_observer_bearing > 180.0:
            self.target_observer_bearing -= 360.0

    def __call__(self, angledeg):
        angle = math.radians(
            angledeg
        )  # start with a nominal bearing at tangent point from observer to tangent point
        h = (
            math.cos(angle) * self.tpnorth + math.sin(angle) * self.tpeast
        )  # get the horizontal unit vector towards the observer from tan point
        points = self._geo.altitude_intercepts(self.observerheight, self.tplocation, h)
        self.obslocation = points[0]  # get the new observer location
        self._geo.from_xyz(self.obslocation)  # and set the location
        obsnorth = (
            -self._geo.local_south
        )  # get the local horizontal vectors at the observer location
        obseast = -self._geo.local_west
        xval = obsnorth.dot(h)  # get the northward component
        yval = obseast.dot(h)  # get the eastward componet
        bearing = math.degrees(
            math.atan2(yval, xval)
        )  # get the bearing of the tangent point from the observer
        offset = bearing - self.target_observer_bearing
        if offset < -180:
            offset += 360.0
        if offset > 180:
            offset -= 360.0
        # print('Angle deg {}, bearing at observer {}, offset {} ({},{}) at {}'.format( angledeg, bearing,  offset, self._geo.latitude, self._geo.longitude, self._geo.altitude))
        return offset


# ------------------------------------------------------------------------------
#           _technique_set_instrument_internal_orientation
# ------------------------------------------------------------------------------
def _technique_set_instrument_internal_orientation(
    platform: Platform, ut: np.datetime64, turntable_definition: np.ndarray
):
    platform.platform_pointing.rotate_instrument_in_icf(
        turntable_definition[0], turntable_definition[1], turntable_definition[2]
    )
    return True


# ------------------------------------------------------------------------------
#           _technique_set_platform_position_from_xyz
# ------------------------------------------------------------------------------
def _technique_set_platform_position_from_xyz(
    platform: Platform, ut: np.datetime64, position_definition: np.ndarray
):
    platform.platform_pointing.set_platform_location(
        xyzt=(
            position_definition[0],
            position_definition[1],
            position_definition[2],
            ut,
        )
    )
    return True


# ------------------------------------------------------------------------------
#           _technique_set_platform_position_from_llh
# ------------------------------------------------------------------------------
def _technique_set_platform_position_from_llh(
    platform: Platform, ut: np.datetime64, position_definition: np.ndarray
) -> bool:
    platform.platform_pointing.set_platform_location(
        latlonheightandt=(
            position_definition[0],
            position_definition[1],
            position_definition[2],
            ut,
        )
    )
    return True


# ------------------------------------------------------------------------------
#           _technique_set_platform_position_from_llh
# ------------------------------------------------------------------------------
def _technique_set_platform_position_from_observer_looking_at_llh(
    platform: Platform, ut: np.datetime64, position_definition: np.ndarray
) -> bool:
    latitude = position_definition[0]
    longitude = position_definition[1]
    height = position_definition[2]
    observer_bearing = position_definition[3]
    if observer_bearing > 180.0:
        observer_bearing -= 360.0
    observer_height = position_definition[4]
    ok = (
        (latitude > -91.0)
        and (latitude < 91.0)
        and (longitude > -361)
        and (longitude < 721)
        and (observer_bearing > -361)
        and (observer_bearing < 721)
        and (observer_height >= height)
    )
    if not ok:
        msg = "_technique_set_platform_position_from_observer_looking_at_llh parameters that were out of acceptable range. You might want to check you havethe correct parameter order"
        raise ValueError(msg)

    bearingfunc = ObserverBearingFunction(
        latitude, longitude, height, observer_height, observer_bearing
    )  # Initialize the observer bearing calculator function
    minval = 9999.0
    minangle = 9999.0
    for angle in range(
        -180, 190, 10
    ):  # Coarse step through azimuths at the tangent point
        val = math.fabs(bearingfunc(angle))  # to find the azimuths at the tangent point
        if (
            val < minval
        ):  # that are generate the proper bearing required at the observer
            minval = val  # the calculation is usually within 10 centimeters.
            minangle = angle

    roots = scipy.optimize.brentq(
        bearingfunc,
        minangle - 10,
        minangle + 10,
        xtol=1.0e-04,
        rtol=1.0e-06,
        full_output=True,
    )
    angle = roots[0]
    converged = roots[1].converged
    if converged:
        obs = bearingfunc.obslocation
        bearingfunc._geo.from_xyz(obs)
        latitude = (
            bearingfunc._geo.latitude
        )  # Get the latitude of the observer location
        longitude = (
            bearingfunc._geo.longitude
        )  # Get the longitude of the observer position
        platform.platform_pointing.set_platform_location(
            latlonheightandt=(latitude, longitude, observer_height, ut)
        )  # But get the height from the user defined value so it is bang on.
    if not converged:
        logging.warning(
            "technique_set_platform_position_from_observer_looking_at_llh, cannot find observer position that matches the bearing requiremenets"
        )
    return converged


# ------------------------------------------------------------------------------
#           _technique_set_platform_position_from_platform
# ------------------------------------------------------------------------------
def _technique_set_platform_position_from_platform(
    platform: Platform, ut: np.datetime64, position_definition: np.ndarray
) -> bool:
    utc = datetime64_to_datetime(ut)
    locator = platform.platform_locator
    ok = locator is not None
    if not ok:
        logging.warning(
            "You must add an instance of PlatformLocator to the Platform class if you wish to get positions data using positioning technique 'from_platform'"
        )
    else:
        position = platform.platform_locator.update_position(utc)
        ok = position is not None
        if ok:
            platform.platform_pointing.set_platform_location(
                xyzt=(position[0], position[1], position[2], ut)
            )
    return ok


# ------------------------------------------------------------------------------
#           from_tangent_altitudes
# ------------------------------------------------------------------------------
def _technique_set_look_vectors_from_tangent_altitude(
    pointing_algorithm: PointingAlgorithms,
    look_vector_definition: np.ndarray,
    roll_control: str,
) -> bool:
    platform = pointing_algorithm.platform
    observer = platform.platform_pointing.location()
    tangent_altitude = look_vector_definition[0]
    geographic_bearing_degrees = look_vector_definition[1]
    roll_angle = look_vector_definition[2] if look_vector_definition.size == 3 else 0.0
    return pointing_algorithm.set_limb_boresight_to_look_at_tangent_altitude(
        observer, tangent_altitude, geographic_bearing_degrees, roll_control, roll_angle
    )


# ------------------------------------------------------------------------------
#           set_limb_look_vectors_from_xyz
# ------------------------------------------------------------------------------
def _technique_set_limb_look_vectors_from_unit_xyz(
    pointing_algorithm: PointingAlgorithms,
    look_vector_definition: np.ndarray,
    roll_control: str,
) -> bool:
    observer = pointing_algorithm.platform.platform_pointing.location()
    look = look_vector_definition[0:3]
    roll_angle = look_vector_definition[3] if look_vector_definition.size == 4 else 0.0
    return pointing_algorithm.set_limb_boresight_from_lookvector(
        observer, look, roll_control, roll_angle
    )


# ------------------------------------------------------------------------------
#           set_nadir_look_vectors_from_xyz
# ------------------------------------------------------------------------------
def _technique_set_boresight_look_at_location_xyz(
    pointing_algorithm: PointingAlgorithms,
    look_vector_definition: np.ndarray,
    roll_control: str,
) -> bool:
    platform = pointing_algorithm.platform
    observer = platform.platform_pointing.location()
    target = look_vector_definition[0:3]
    roll_angle = look_vector_definition[3] if look_vector_definition.size == 4 else 0.0
    return pointing_algorithm.set_boresight_to_look_at_geocentric_location(
        observer, target, roll_control, roll_angle
    )


# ------------------------------------------------------------------------------
#           set_nadir_look_vectors_from_xyz
# ------------------------------------------------------------------------------
def _technique_set_boresight_look_at_location_llh(
    pointing_algorithm: PointingAlgorithms,
    look_vector_definition: np.ndarray,
    roll_control: str,
) -> bool:
    platform = pointing_algorithm.platform
    observer = platform.platform_pointing.location()
    latitude = look_vector_definition[0]
    longitude = look_vector_definition[1]
    height = look_vector_definition[2]
    roll_angle = look_vector_definition[3] if look_vector_definition.size == 4 else 0.0
    pointing_algorithm.geo.from_lat_lon_alt(latitude, longitude, height)
    target = pointing_algorithm.geo.location
    return pointing_algorithm.set_boresight_to_look_at_geocentric_location(
        observer, target, roll_control, roll_angle
    )


# ------------------------------------------------------------------------------
#           set_nadir_look_vectors_from_xyz
# ------------------------------------------------------------------------------
def _technique_set_boresight_pointing_from_unitvectors(
    pointing_algorithm: PointingAlgorithms,
    look_vector_definition: np.ndarray,
    roll_control: str,
) -> bool:
    xunit = look_vector_definition[0:3]
    zunit = look_vector_definition[3:6]
    yunit = np.cross(zunit, xunit)
    G = UnitVectors(vectors=(xunit, yunit, zunit))
    pointing_algorithm.set_boresight_pointing_from_unitvectors(G)
    return True


# ------------------------------------------------------------------------------
#           set_nadir_look_vectors_from_xyz
# ------------------------------------------------------------------------------
def _technique_set_observer_to_look_in_azi_elev(
    pointing_algorithm: PointingAlgorithms,
    look_vector_definition: np.ndarray,
    roll_control: str,
) -> bool:
    platform = pointing_algorithm.platform
    observer = platform.platform_pointing.location()
    azimuth = look_vector_definition[0]
    elevation = look_vector_definition[1]
    roll_angle = look_vector_definition[2] if look_vector_definition.size == 3 else 0.0
    return pointing_algorithm.set_observer_to_look_in_azi_elev(
        observer, elevation, azimuth, roll_control, roll_angle
    )


# ------------------------------------------------------------------------------
#           _technique_set_platform_pointing_from_platform
# ------------------------------------------------------------------------------
def _technique_set_platform_pointing_from_platform(
    pointing_algorithm: PointingAlgorithms,
    look_vector_definition: np.ndarray,
    roll_control: str,
) -> bool:
    platform = pointing_algorithm.platform
    utc = platform.platform_pointing.utc()
    locator = platform.platform_locator
    ok = locator is not None
    if not ok:
        logging.warning(
            "You must add an instance of PlatformLocator to the Platform class if you wish to get look vector data using pointing technique 'from_platform'"
        )
    else:
        vectors = platform.platform_locator.update_orientation(utc)
        ok = vectors is not None
        if ok:
            G = UnitVectors(vectors=vectors)
            pointing_algorithm.set_boresight_pointing_from_unitvectors(G)
    return ok


# ------------------------------------------------------------------------------
#           set_nadir_look_vectors_from_xyz
# ------------------------------------------------------------------------------
def _technique_set_boresight_look_at_location_orbitangle(
    pointing_algorithm: PointingAlgorithms,
    look_vector_definition: np.ndarray,
    roll_control: str,
) -> bool:
    platform = pointing_algorithm.platform
    observer = platform.platform_locator.position
    velocity = platform.platform_locator.velocity
    v = np.linalg.norm(velocity)
    ok = v > 0.0
    if ok:
        vunit = velocity / v
        pointing_algorithm.geo.from_xyz(observer)
        north = -pointing_algorithm.geo.local_south
        east = -pointing_algorithm.geo.local_west
        x = np.dot(north, vunit)
        y = np.dot(
            east, vunit
        )  # get the eastward component of the velocity at the observers location
        theta = math.degrees(math.atan2(y, x))
        tangent_altitude = look_vector_definition[0]
        orbitplane_bearing_degrees = look_vector_definition[1]
        geographic_bearing_degrees = theta + orbitplane_bearing_degrees
        roll_angle = (
            look_vector_definition[2] if look_vector_definition.size == 3 else 0.0
        )
        ok = pointing_algorithm.set_limb_boresight_to_look_at_tangent_altitude(
            observer,
            tangent_altitude,
            geographic_bearing_degrees,
            roll_control,
            roll_angle,
        )
    return ok


# ------------------------------------------------------------------------------
#           _technique_set_platform_position_from_platform
# ------------------------------------------------------------------------------
def _technique_set_icf_orientation_from_azi_elev(
    platform: Platform, azielevdata: np.ndarray
) -> bool:
    azimuth = azielevdata[0]
    elevation = azielevdata[1]
    roll = azielevdata[2] if azielevdata.size == 3 else 0
    platform.platform_pointing.rotate_instrument_in_icf(azimuth, elevation, roll)
    return True


# ------------------------------------------------------------------------------
#           _technique_set_platform_position_from_platform
# ------------------------------------------------------------------------------
def _technique_set_icf_look_from_xyz(platform: Platform, xyz_data: np.ndarray) -> bool:
    raise AssertionError()  # platform.platform_pointing.set_platform_location(xyzt=(position[0], position[1], position[2], ut))
    return True


# ------------------------------------------------------------------------------
#           _technique_set_platform_position_from_platform
# ------------------------------------------------------------------------------
def _technique_set_icf_orientation_no_operation(
    platform: Platform, xyz_data: np.ndarray
) -> bool:
    platform.platform_pointing.reset_icf_rotation_matrices()
    return True
