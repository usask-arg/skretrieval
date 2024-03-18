import unittest
from datetime import datetime, timedelta
from math import radians
import numpy as np
from skretrieval.platforms.satellite import (
    SatelliteBase,
    SatelliteKepler,
    SatelliteSGP4,
    SatelliteSunSync,
    SatelliteMolniya,
    SatelliteSimpleGeostationary,
)
from skretrieval.platforms.satellite.gibbs import gibbs
from skretrieval.platforms.satellite.coe import coe_from_state_vector


# -----------------------------------------------------------------------------
#           SkimpySatelliteTests
# -----------------------------------------------------------------------------
class SkretrievalSatelliteTests(unittest.TestCase):
    def setUp(self):
        self._numtimes = 200
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
            position = sat.update_position(
                self._times[i]
            )  # get the geographic geocentrix position of each satellite
            self._locations[:, i] = sat.lat_lon_height
            # print('{:3d} {:18.13f} {:18.13f} {:23.13f}'.format(i,self._locations[0,i], self._locations[1,i],self._locations[2,i] ))
        return self._locations

    def test_sunsync_twolinelements(self):
        utc = datetime.fromisoformat("2023-06-23 16:00:00.000000")
        t = datetime.fromisoformat("2023-06-23 16:10:00.000000")
        kepler = SatelliteKepler()
        sunsync = SatelliteSunSync(
            utc,
            orbittype="sgp4",
            period_from_altitude=450000.0,
            localtime_of_ascending_node_hours=13.5,
        )  # get a sun sync satellite, this is only accurate to a couple of seconds of time as it uses Kepler two line elements to initialize an SGP8 orbit.
        teq = sunsync.equator_crossing(t)  # get the time of the equator crossing
        satpos = sunsync.update_position(teq)  # get its position at this time
        ecipos = sunsync.eciposition()
        ecivel = sunsync.ecivelocity()
        kepler.from_state_vector(
            teq, ecipos, ecivel, 1000
        )  # get a kepler orbit that matches the SGP4
        kpos = kepler.update_position(teq)
        kecipos = kepler.eciposition()
        kecivel = kepler.ecivelocity()
        lines = kepler.make_two_line_elements()
        sgp4 = SatelliteSGP4(lines)
        sgp4pos = sgp4.update_position(teq)
        spos = sgp4.eciposition()
        svel = sgp4.ecivelocity()
        ds = sgp4pos - satpos
        s = np.linalg.norm(ds)
        self.assertLessEqual(s, 15500.0)

    # ------------------------------------------------------------------------------
    #           test_gibbs
    # ------------------------------------------------------------------------------
    def test_gibbs(self):
        r1 = np.array([-294320.0, 4265100.0, 5986700.0])
        r2 = np.array([-1365400.0, 3637600.0, 6346800.0])
        r3 = np.array([-2940300.0, 2473700.0, 6555800.0])
        v2 = gibbs(
            r1, r2, r3
        )  # Use Gibbs to calculate satellite velocity at the middle point of 3 points
        self.assertAlmostEqual(v2[0], -6217.6009496, 3)
        self.assertAlmostEqual(v2[1], -4012.3770592, 3)
        self.assertAlmostEqual(v2[2], 1599.15006807, 3)

        coe = coe_from_state_vector(
            r2, v2
        )  # use the satellite position and velocity to calculate the classical orbital elements
        self.assertAlmostEqual(coe.H * 1.0e-6, 56193.04806713033, 3)
        self.assertAlmostEqual(coe.RA, 0.6981716759913223, 8)
        self.assertAlmostEqual(coe.TA, 0.8707351512007719, 8)
        self.assertAlmostEqual(coe.W, 0.5255060496425946, 8)
        self.assertAlmostEqual(coe.a, 8002140.855890051, 2)
        self.assertAlmostEqual(coe.e, 0.1001592330355506, 9)
        self.assertAlmostEqual(coe.i, 1.0472156172639657, 8)

        kepler = SatelliteKepler()
        kepler.from_state_vector(self._utc, r2, v2)
        p = kepler.period()
        self.assertAlmostEqual(p.total_seconds(), 7123.940246, 3)

    # ------------------------------------------------------------------------------
    #           test_three_points
    # ------------------------------------------------------------------------------
    def test_three_points(self):
        r1 = np.array([-294320.0, 4265100.0, 5986700.0])
        r2 = np.array([-1365400.0, 3637600.0, 6346800.0])
        r3 = np.array([-2940300.0, 2473700.0, 6555800.0])

        kepler = SatelliteKepler()
        kepler.from_three_positions(self._utc, r1, r2, r3)
        p = kepler.period()
        self.assertAlmostEqual(p.total_seconds(), 7123.940246, 3)

    # -----------------------------------------------------------------------------
    #           test_SGP4
    # -----------------------------------------------------------------------------

    def test_SGP4(self):  # create our different satellites::
        line1 = "1 26702U 01007A   19206.65122582  .00000348  00000-0  26798-4 0  9995"  # two line elements that will be used for the SGP4 satellite
        line2 = "2 26702  97.5720 223.8268 0009640 316.2599  43.7864 15.07871693  7426"
        sgp4 = SatelliteSGP4(twolines=[line1, line2])

        for nloop in range(
            4
        ):  # Do 4 loops to ensure we can step backwards in time without issue
            positions = self._propagate_satellite(sgp4)
            self.assertAlmostEqual(
                positions[0, 99], 2.3180298075593, 2
            )  # Verify Latitude
            self.assertAlmostEqual(
                positions[1, 99], -48.0400704025521, 2
            )  # Verify Longitude
            self.assertAlmostEqual(
                positions[2, 99], 541249.5916230629664, 2
            )  # Verify Altitude
            self.assertAlmostEqual(
                positions[0, 10], 26.960762147970545, 2
            )  # Verify Latitude
            self.assertAlmostEqual(
                positions[1, 10], -29.33926117654872, 2
            )  # Verify Longitude
            self.assertAlmostEqual(
                positions[2, 10], 544393.0853459131, 2
            )  # Verify Altitude

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
        for nloop in range(4):
            positions = self._propagate_satellite(kepler)
            self.assertAlmostEqual(
                positions[0, 99], 9.514217229999177, 2
            )  # Verify Latitude
            self.assertAlmostEqual(
                positions[1, 99], 56.010676386422446, 2
            )  # Verify Longitude
            self.assertAlmostEqual(
                positions[2, 99], 256026.54192426076, 2
            )  # Verify Altitude
        # print("Done Kepler")

    def test_Molniya_with_kepler(self):
        molniya = SatelliteMolniya(
            self._utc, orbittype="kepler", longitude_of_ascending_node_degrees=-124.0
        )
        for nloop in range(4):
            positions = self._propagate_satellite(molniya)
            self.assertAlmostEqual(
                positions[0, 99], 61.99186405390857, 2
            )  # Verify Latitude
            self.assertAlmostEqual(
                positions[1, 99], -78.6699517266697, 2
            )  # Verify Longitude
            self.assertAlmostEqual(
                positions[2, 99], 38227950.71962145, 2
            )  # Verify Altitude
        # print("Done molniya with kepler")

    def test_Molniya_with_sgp4(self):
        molniya = SatelliteMolniya(
            self._utc, orbittype="sgp4", longitude_of_ascending_node_degrees=-124.0
        )
        for nloop in range(4):
            positions = self._propagate_satellite(molniya)
            self.assertAlmostEqual(
                positions[0, 99], 61.99170529136819, 2
            )  # Verify Latitude
            self.assertAlmostEqual(
                positions[1, 99], -78.72355534491894, 2
            )  # Verify Longitude
            self.assertAlmostEqual(
                positions[2, 99], 38206706.45955145, 2
            )  # Verify Altitude
        # print("Done molniya with sgp4")

    def test_sunsync_with_kepler(self):
        sunsync = SatelliteSunSync(
            self._utc,
            orbittype="kepler",
            period_from_altitude=600000.0,
            localtime_of_ascending_node_hours=18.25,
        )
        for nloop in range(4):
            positions = self._propagate_satellite(sunsync)
            self.assertAlmostEqual(
                positions[0, 99], 8.583229967118568, 2
            )  # Verify Latitude
            self.assertAlmostEqual(
                positions[1, 99], -55.993522413900365, 2
            )  # Verify Longitude
            self.assertAlmostEqual(
                positions[2, 99], 600470.0139338467, 2
            )  # Verify Altitude
        # print("Done sun sync with kepler")

    def test_sunsync_with_sgp4(self):
        sunsync = SatelliteSunSync(
            self._utc,
            orbittype="sgp4",
            period_from_altitude=600000.0,
            localtime_of_ascending_node_hours=18.25,
        )
        for nloop in range(4):
            positions = self._propagate_satellite(sunsync)
            self.assertAlmostEqual(
                positions[0, 99], 8.23901273452463, 2
            )  # Verify Latitude
            self.assertAlmostEqual(
                positions[1, 99], -55.87883826556556, 2
            )  # Verify Longitude
            self.assertAlmostEqual(
                positions[2, 99], 602335.9354597726, 2
            )  # Verify Altitude
        # print("Done sun sync with sgp4")

    def test_simple_geostationary(self):
        geostat = SatelliteSimpleGeostationary(-80.0)
        for nloop in range(4):
            positions = self._propagate_satellite(geostat)
            self.assertAlmostEqual(
                positions[0, 99], 0.0000000000000, 2
            )  # Verify Latitude
            self.assertAlmostEqual(
                positions[1, 99], -80.0000000000000, 2
            )  # Verify Longitude
            self.assertAlmostEqual(
                positions[2, 99], 35785863.0000000000000, 2
            )  # Verify Altitude
