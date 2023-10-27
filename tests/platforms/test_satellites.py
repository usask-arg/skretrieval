from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from math import radians

import numpy as np

from skretrieval.platforms.satellite import (
    SatelliteBase,
    SatelliteKepler,
    SatelliteMolniya,
    SatelliteSGP4,
    SatelliteSimpleGeostationary,
    SatelliteSunSync,
)


# -----------------------------------------------------------------------------
#           SkimpySatelliteTests
# -----------------------------------------------------------------------------
class SkretrievalSatelliteTests(unittest.TestCase):
    def setUp(self):
        self._numtimes = 100
        self._locations = np.zeros((3, self._numtimes))
        self._times = np.zeros([self._numtimes], dtype="datetime64[us]")
        self._utc = datetime(2019, 7, 26, hour=20, minute=15, second=00)
        dt = timedelta(minutes=1.0)
        for i in range(self._numtimes):  # for each time step
            tnow = self._utc + i * dt
            self._times[i] = tnow  # save the time of the step

    # -----------------------------------------------------------------------------
    #           _propagate_satellite
    # -----------------------------------------------------------------------------

    def _propagate_satellite(self, sat: SatelliteBase) -> np.ndarray:
        for i in range(self._numtimes):
            _ = sat.update_position(
                self._times[i]
            )  # get the geographic geocentrix position of each satellite
            self._locations[:, i] = sat.lat_lon_height
            # print('{:3d} {:18.13f} {:18.13f} {:23.13f}'.format(i,self._locations[0,i], self._locations[1,i],self._locations[2,i] ))
        return self._locations

    # -----------------------------------------------------------------------------
    #           test_SGP4
    # -----------------------------------------------------------------------------

    def test_SGP4(self):  # create our different satellites::
        line1 = "1 26702U 01007A   19206.65122582  .00000348  00000-0  26798-4 0  9995"  # two line elements that will be used for the SGP4 satellite
        line2 = "2 26702  97.5720 223.8268 0009640 316.2599  43.7864 15.07871693  7426"
        sgp4 = SatelliteSGP4(twolines=[line1, line2])
        positions = self._propagate_satellite(sgp4)
        self.assertAlmostEqual(positions[0, 99], 2.3180298075593, 8)  # Verify Latitude
        self.assertAlmostEqual(
            positions[1, 99], -48.040070205433835, 8
        )  # Verify Longitude
        self.assertAlmostEqual(
            positions[2, 99], 541249.5916230629664, 4
        )  # Verify Altitude
        # print("Done SGP4")

    # -----------------------------------------------------------------------------
    #           test_Kepler
    # -----------------------------------------------------------------------------

    def test_Kepler(self):
        kepler = SatelliteKepler(
            self._utc,
            period_from_altitude=600000.0,
            inclination_radians=radians(97.0),
            longitude_of_ascending_node_degrees=82.0,
            eccentricity=0.05,
        )
        positions = self._propagate_satellite(kepler)
        self.assertAlmostEqual(
            positions[0, 99], 9.514609171885569, 8
        )  # Verify Latitude
        self.assertAlmostEqual(
            positions[1, 99], -127.05193163355204, 8
        )  # Verify Longitude
        self.assertAlmostEqual(
            positions[2, 99], 256026.9476713663025, 6
        )  # Verify Altitude
        # print("Done Kepler")

    def test_Molniya_with_kepler(self):
        molniya = SatelliteMolniya(
            self._utc, orbittype="kepler", longitude_of_ascending_node_degrees=-124.0
        )
        positions = self._propagate_satellite(molniya)
        self.assertAlmostEqual(positions[0, 99], 61.9918640393, 8)  # Verify Latitude
        self.assertAlmostEqual(
            positions[1, 99], 98.26748942190007, 8
        )  # Verify Longitude
        self.assertAlmostEqual(
            positions[2, 99], 38227979.3644920811, 4
        )  # Verify Altitude
        # print("Done molniya with kepler")

    def test_Molniya_with_sgp4(self):
        molniya = SatelliteMolniya(
            self._utc, orbittype="sgp4", longitude_of_ascending_node_degrees=-124.0
        )
        positions = self._propagate_satellite(molniya)
        self.assertAlmostEqual(
            positions[0, 99], 61.997018199205556, 8
        )  # Verify Latitude
        self.assertAlmostEqual(
            positions[1, 99], 98.21136004139319, 8
        )  # Verify Longitude
        self.assertAlmostEqual(
            positions[2, 99], 38189494.5561303272843, 4
        )  # Verify Altitude
        # print("Done molniya with sgp4")

    def test_sunsync_with_kepler(self):
        sunsync = SatelliteSunSync(
            self._utc,
            orbittype="kepler",
            period_from_altitude=600000.0,
            localtime_of_ascending_node_hours=18.25,
        )
        positions = self._propagate_satellite(sunsync)
        self.assertAlmostEqual(positions[0, 99], 8.5835838875757, 8)  # Verify Latitude
        self.assertAlmostEqual(
            positions[1, 99], 120.9438695139474, 8
        )  # Verify Longitude
        self.assertAlmostEqual(
            positions[2, 99], 600470.0526328303386, 4
        )  # Verify Altitude
        # print("Done sun sync with kepler")

    def test_sunsync_with_sgp4(self):
        sunsync = SatelliteSunSync(
            self._utc,
            orbittype="sgp4",
            period_from_altitude=600000.0,
            localtime_of_ascending_node_hours=18.25,
        )
        positions = self._propagate_satellite(sunsync)
        self.assertAlmostEqual(positions[0, 99], 8.2393662574276, 8)  # Verify Latitude
        self.assertAlmostEqual(
            positions[1, 99], 121.05861297575522, 8
        )  # Verify Longitude
        self.assertAlmostEqual(
            positions[2, 99], 602331.4414312287699, 4
        )  # Verify Altitude
        # print("Done sun sync with sgp4")

    def test_simple_geostationary(self):
        geostat = SatelliteSimpleGeostationary(-80.0)
        positions = self._propagate_satellite(geostat)
        self.assertAlmostEqual(positions[0, 99], 0.0000000000000, 8)  # Verify Latitude
        self.assertAlmostEqual(
            positions[1, 99], -80.0000000000000, 8
        )  # Verify Longitude
        self.assertAlmostEqual(
            positions[2, 99], 35785863.0000000000000, 4
        )  # Verify Altitude
