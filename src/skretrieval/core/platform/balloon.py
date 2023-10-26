import numpy as np
from skretrieval.util import rotation_matrix
import sasktran as sk
from skretrieval.core import OpticalGeometry


class Balloon(object):
    def __init__(self, location, target_altitudes, scan_rate_rad_s, mjd, bore_sight_plane):
        self._location = location
        self._target_altitudes = target_altitudes
        self._scan_rate_rad_s = scan_rate_rad_s
        self._mjd = mjd
        self._bore_sight_plane = bore_sight_plane

    def optical_axis(self, sample_rate_s):
        los_geo = sk.Geodetic()

        low_alt_los = los_geo.from_tangent_altitude(self._target_altitudes[0], self._location, self._bore_sight_plane)
        high_alt_los = los_geo.from_tangent_altitude(self._target_altitudes[1], self._location, self._bore_sight_plane)

        rot_axis = np.cross(low_alt_los, high_alt_los)
        rot_axis /= np.linalg.norm(rot_axis)

        current_los = low_alt_los

        theta = self._scan_rate_rad_s * sample_rate_s

        optical_ax = []

        while True:
            # Do motion of balloon
            location = self._location
            local_up = location / np.linalg.norm(location)
            mjd = self._mjd

            optical_ax.append(OpticalGeometry(location, current_los, local_up, mjd))

            los_geo.from_tangent_point(location, current_los)

            if los_geo.altitude > self._target_altitudes[1]:
                break
            else:
                current_los = rotation_matrix(rot_axis, theta) @ current_los

        return optical_ax


if __name__ == "__main__":
    geo = sk.Geodetic()

    geo.from_lat_lon_alt(40, 40, 35000)

    location = geo.location
    bore = geo.local_west
    mjd = 54372

    balloon = Balloon(location, [20000, 30000], 0.01, mjd, bore)

    optical = balloon.optical_axis(0.1)

    print(optical)
