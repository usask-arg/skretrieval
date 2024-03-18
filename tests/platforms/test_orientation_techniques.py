import unittest
import numpy as np
import math
import skretrieval.time as sktime
import sasktran as sk
from skretrieval.platforms.platform import Platform
from skretrieval.platforms.satellite import SatelliteSunSync


class SkretrievalGeometryStatesTests(unittest.TestCase):
    # ------------------------------------------------------------------------------
    #           test_example1
    # ------------------------------------------------------------------------------

    def test_example1(self):
        platform = Platform()
        utc = ["2020-07-15T15:00:00.0000000"]
        observer = [52.0, -107.0, 600000.0]
        tanpoint = [35000.0, 45.0]
        platform.add_measurement_set(
            utc, ("llh", observer), ("tangent_altitude", "limb", tanpoint)
        )
        opticalmeasurements = platform.make_optical_geometry()

        geo = sk.Geodetic()
        for i in range(len(opticalmeasurements)):
            entry = opticalmeasurements[i]
            geo.from_xyz(entry.observer)
            obslat = geo.latitude
            obslng = geo.longitude
            obshgt = geo.altitude / 1000.0
            geo.from_tangent_point(entry.observer, entry.look_vector)
            tanlat = geo.latitude
            tanlng = geo.longitude
            tanhgt = geo.altitude / 1000.0

        self.assertAlmostEqual(obslat, 52.0, 5)
        self.assertAlmostEqual(obslng, 253.0, 5)
        self.assertAlmostEqual(obshgt, 600.0, 5)
        self.assertAlmostEqual(tanlat, 63.64674494609161, 3)
        self.assertAlmostEqual(tanlng, 291.7842681940222, 3)
        self.assertAlmostEqual(tanhgt, 35.000064782094675, 3)

    # ------------------------------------------------------------------------------
    #           test_example2
    # ------------------------------------------------------------------------------

    def test_example2(self):
        platform = Platform()
        utc = ["2020-07-15T15:00:00.0000000"]
        observer = [52.0, -107.0, 35000.0, 45.0, 600000.0]
        tanpoint = [35000.0, 45.0]
        platform.add_measurement_set(
            utc, ("looking_at_llh", observer), ("tangent_altitude", "limb", tanpoint)
        )
        opticalmeasurements = platform.make_optical_geometry()

        geo = sk.Geodetic()
        for i in range(len(opticalmeasurements)):
            entry = opticalmeasurements[i]
            geo.from_xyz(entry.observer)
            obslat = geo.latitude
            obslng = geo.longitude
            obshgt = geo.altitude / 1000.0
            geo.from_tangent_point(entry.observer, entry.look_vector)
            tanlat = geo.latitude
            tanlng = geo.longitude
            tanhgt = geo.altitude / 1000.0
            # print('Satellite location = ({:5.2f}N,{:6.2f}E) at a height of {:6.2f} km'.format(obslat, obslng, obshgt))
            # print('Tangent location = ({:5.2f}N,{:6.2f}E) at a height of {:6.2f} km'.format(tanlat, tanlng, tanhgt))
        self.assertAlmostEqual(obslat, 38.23016297193056, 5)
        self.assertAlmostEqual(obslng, 226.128606362275, 5)
        self.assertAlmostEqual(obshgt, 600.0, 5)
        self.assertAlmostEqual(tanlat, 52.018228238070634, 2)
        self.assertAlmostEqual(tanlng, 252.9849577769396, 2)
        self.assertAlmostEqual(tanhgt, 35.00005703700089, 2)

    # ------------------------------------------------------------------------------
    #           test_position_from_platform
    # ------------------------------------------------------------------------------
    def test_position_from_platform(self):
        mjd0 = sktime.ut_to_mjd(
            "2020-07-15T15:00:00.000000"
        )  # Define the time of the ascending node
        satellite = SatelliteSunSync(
            mjd0,  # Create a sun-synchronous satellite
            orbittype="sgp4",
            period_from_altitude=600000.0,
            localtime_of_ascending_node_hours=2.75,
        )
        platform = Platform(
            platform_locator=satellite
        )  # Create a platform which can use the sun-synchronous satellite for its position

        utc = (
            mjd0 + np.arange(0, 100) / 1440.0
        )  # Get measurements every minute along the orbit starting at the ascending node.
        looktanalt = [
            (35000.0, 45)
        ]  # Look at a tangent altitude of 35 km, at a geographic bearing of 45 degrees at the satellite location. Note, the parameter set will be expanded to N measurements
        platform.add_measurement_set(
            utc, ("from_platform",), ("tangent_altitude", "limb", looktanalt)
        )  # Add the measurement set
        opticalmeasurements = platform.make_optical_geometry()

        geo = sk.Geodetic()
        obslat = np.zeros([100])
        obslng = np.zeros([100])
        obshgt = np.zeros([100])
        tanlat = np.zeros([100])
        tanlng = np.zeros([100])
        tanhgt = np.zeros([100])
        for i in range(len(opticalmeasurements)):
            entry = opticalmeasurements[i]
            geo.from_xyz(entry.observer)
            obslat[i] = geo.latitude
            obslng[i] = geo.longitude
            obshgt[i] = geo.altitude / 1000.0
            geo.from_tangent_point(entry.observer, entry.look_vector)
            tanlat[i] = geo.latitude
            tanlng[i] = geo.longitude
            tanhgt[i] = geo.altitude / 1000.0
            # print('{:3d} Satellite location = ({:5.2f}N,{:6.2f}E) at a height of {:6.2f} km'.format(i, obslat[i], obslng[i], obshgt[i]))
            # print('{:3d} Tangent location   = ({:5.2f}N,{:6.2f}E) at a height of {:6.2f} km'.format(i, tanlat[i], tanlng[i], tanhgt[i]))

        self.assertAlmostEqual(obslat[22], 78.80683736020426, 2)
        self.assertAlmostEqual(obslng[22], 127.3185951847006, 2)
        self.assertAlmostEqual(obshgt[22], 613.2565386016553, 2)
        self.assertAlmostEqual(tanlat[22], 72.68372646991565, 2)
        self.assertAlmostEqual(tanlng[22], 236.5054513231322, 2)
        self.assertAlmostEqual(tanhgt[22], 35.000029306320094, 2)
        self.assertAlmostEqual(obslat[67], -68.11695918550532, 2)
        self.assertAlmostEqual(obslng[67], 319.7019529964772, 2)
        self.assertAlmostEqual(obshgt[67], 625.6432617439019, 2)
        self.assertAlmostEqual(tanlat[67], -48.066050277130465, 2)
        self.assertAlmostEqual(tanlng[67], 344.85005649622474, 2)
        self.assertAlmostEqual(tanhgt[67], 34.99995200953633, 2)

    # ------------------------------------------------------------------------------
    #           test_height_profile
    # ------------------------------------------------------------------------------

    def test_height_profile(self):
        platform = Platform()
        utc = "2020-09-24T18:00:00.0000000"  # Use just a string to define a single UTC
        observer = [
            52.0,
            -107.0,
            10000.0,
            45.0,
            600000.0,
        ]  # define a single observer position
        tanpoint = np.tile(
            [0.0, 45.0], [100, 1]
        )  # define 100 tangent heights, note we use the same geographic bearing as the observer parameters
        tanpoint[:, 0] = np.arange(0, 100) * 1000.0  # fill in the height field
        platform.add_measurement_set(
            utc, ("looking_at_llh", observer), ("tangent_altitude", "limb", tanpoint)
        )  # find the observer position and look vectors etc for the 100 measurements
        opticalmeasurements = platform.make_optical_geometry()

        geo = sk.Geodetic()
        obslat = np.zeros([100])
        obslng = np.zeros([100])
        obshgt = np.zeros([100])
        tanlat = np.zeros([100])
        tanlng = np.zeros([100])
        tanhgt = np.zeros([100])
        for i in range(len(opticalmeasurements)):
            entry = opticalmeasurements[i]
            geo.from_xyz(entry.observer)
            obslat[i] = geo.latitude
            obslng[i] = geo.longitude
            obshgt[i] = geo.altitude / 1000.0
            geo.from_tangent_point(entry.observer, entry.look_vector)
            tanlat[i] = geo.latitude
            tanlng[i] = geo.longitude
            tanhgt[i] = geo.altitude / 1000.0
            # print('self.assertAlmostEqual(obslat[{}], {})'.format(i, obslat[i]))
            # print('self.assertAlmostEqual(obslng[{}], {})'.format(i, obslng[i]))
            # print('self.assertAlmostEqual(obshgt[{}], {})'.format(i, obshgt[i]))
            # print('self.assertAlmostEqual(tanlat[{}], {})'.format(i, tanlat[i]))
            # print('self.assertAlmostEqual(tanlng[{}], {})'.format(i, tanlng[i]))
            # print('self.assertAlmostEqual(tanhgt[{}], {})'.format(i, tanhgt[i]))

        self.assertAlmostEqual(obslat[22], 37.980130977495996, 2)
        self.assertAlmostEqual(obslng[22], 225.5195159438611, 2)
        self.assertAlmostEqual(obshgt[22], 600.0000000000007, 2)
        self.assertAlmostEqual(tanlat[22], 51.91297836680919, 2)
        self.assertAlmostEqual(tanlng[22], 252.62476172260668, 2)
        self.assertAlmostEqual(obslat[67], 37.980130977495996, 2)
        self.assertAlmostEqual(obslng[67], 225.5195159438611, 2)
        self.assertAlmostEqual(obshgt[67], 600.0000000000007, 2)
        self.assertAlmostEqual(tanlat[67], 51.49473602100524, 2)
        self.assertAlmostEqual(tanlng[67], 251.25662163977978, 2)
        self.assertAlmostEqual(tanhgt[67], 67.00005620048381, 2)

    # ------------------------------------------------------------------------------
    #           test_elev_azi
    # ------------------------------------------------------------------------------
    def test_elev_azi(self):
        platform = Platform()
        utc = [
            "2020-09-24T18:00:00.0000000"
        ]  # Use just a string to define a single UTC
        observer = [52.0, -107.0, 10000.0]  # define a single observer position
        yaw_pitch = np.tile([0.0, 0.0], [180, 1])  # define 180 yaw and pitch fields
        yaw_pitch[:, 1] = np.arange(-90, 90)  # fill in the pitch field
        platform.add_measurement_set(
            utc, ("llh", observer), ("yaw_pitch_roll", "standard", yaw_pitch)
        )  # find the observer position and look vectors etc for the 100 measurements
        opticalmeasurements = platform.make_optical_geometry()

        geo = sk.Geodetic()
        obslat = np.zeros([180])
        obslng = np.zeros([180])
        obshgt = np.zeros([180])
        for i in range(len(opticalmeasurements)):
            entry = opticalmeasurements[i]
            geo.from_xyz(entry.observer)
            obslat[i] = geo.latitude
            obslng[i] = geo.longitude
            obshgt[i] = geo.altitude / 1000.0
            up = geo.local_up
            elevation = math.degrees(math.asin(np.dot(up, entry.look_vector)))
            self.assertAlmostEqual(obslat[i], 52.0)
            self.assertAlmostEqual(obslng[i], 360.0 - 107.0)
            self.assertAlmostEqual(obshgt[i], 10.0)
            self.assertAlmostEqual(yaw_pitch[i, 1], elevation)

    # ------------------------------------------------------------------------------
    #           test_instrument_internal_rotation
    # ------------------------------------------------------------------------------
    def test_instrument_internal_rotation(self):
        platform = Platform()
        utc = [
            "2020-09-24T18:00:00.0000000"
        ]  # Use just a string to define a single UTC
        observer = [52.0, -107.0, 10000.0]  # define a single observer position
        azi_elev = np.tile([0.0, 0.0], [180, 1])  # define 180 yaw and pitch fields
        azi_elev[:, 0] = np.arange(0, 180) * 0.1
        platform.add_measurement_set(
            utc,
            ("llh", observer),
            ("azi_elev", "standard", azi_elev),
            icf_orientation=("azi_elev", [30.0, 60.0]),
        )  # find the observer position and look vectors etc for the 100 measurements
        opticalmeasurements = platform.make_optical_geometry()

        geo = sk.Geodetic()
        obslat = np.zeros([180])
        obslng = np.zeros([180])
        obshgt = np.zeros([180])
        for i in range(len(opticalmeasurements)):
            entry = opticalmeasurements[i]
            geo.from_xyz(entry.observer)
            obslat[i] = geo.latitude
            obslng[i] = geo.longitude
            obshgt[i] = geo.altitude / 1000.0
            up = geo.local_up
            north = -geo.local_south
            elevation = math.degrees(math.asin(np.dot(up, entry.look_vector)))
            horiz = entry.look_vector - math.sin(math.radians(elevation)) * up
            horiz /= np.linalg.norm(horiz)
            azi = math.degrees(math.acos(np.dot(north, horiz)))
            self.assertAlmostEqual(obslat[i], 52.0)
            self.assertAlmostEqual(obslng[i], 360.0 - 107.0)
            self.assertAlmostEqual(obshgt[i], 10.0)
            self.assertAlmostEqual(azi, azi_elev[i, 0] + 30.0)
            self.assertAlmostEqual(elevation, 60.0)

    # ------------------------------------------------------------------------------
    #           test_nadir
    # ------------------------------------------------------------------------------
    def test_nadir(self):
        platform = Platform()
        utc = "2020-09-24T18:00:00.0000000"  # Use just a string to define a single UTC
        observer = [52.0, -107.0, 600000.0]  # define a single observer position
        ground = np.tile(
            [57.0, -110.0, 0], [100, 1]
        )  # define 100 tangent heights, note we use the same geographic bearing as the observer parameters
        ground[:, 1] = -107.0 + np.arange(0, 100) / 100.0  # fill in the height field
        platform.add_measurement_set(
            utc, ("llh", observer), ("location_llh", "nadir", ground)
        )  # find the observer position and look vectors etc for the 100 measurements
        opticalmeasurements = platform.make_optical_geometry()

        geo = sk.Geodetic()
        obslat = np.zeros([100])
        obslng = np.zeros([100])
        obshgt = np.zeros([100])
        gndlat = np.zeros([100])
        gndlng = np.zeros([100])
        gndhgt = np.zeros([100])
        for i in range(len(opticalmeasurements)):
            entry = opticalmeasurements[i]
            geo.from_xyz(entry.observer)
            obslat[i] = geo.latitude
            obslng[i] = geo.longitude
            obshgt[i] = geo.altitude / 1000.0
            (l1, l2) = geo.altitude_intercepts(0.0, entry.observer, entry.look_vector)
            geo.from_xyz(l1)
            gndlat[i] = geo.latitude
            gndlng[i] = geo.longitude
            gndhgt[i] = geo.altitude / 1000.0
            self.assertAlmostEqual(obslat[i], 52.0)
            self.assertAlmostEqual(obslng[i], 360.0 - 107.0)
            self.assertAlmostEqual(obshgt[i], 600.0)
            self.assertAlmostEqual(gndlat[i], ground[i, 0], places=4)
            self.assertAlmostEqual(gndlng[i], 360.0 + ground[i, 1], places=4)
            self.assertAlmostEqual(gndhgt[i], ground[i, 2] / 1000.0, places=4)

    # ------------------------------------------------------------------------------
    #           test_tangentalt_geometry
    # ------------------------------------------------------------------------------
    def test_tangent_from_orbitplane_geometry(self):
        mjd0 = sktime.ut_to_mjd("2020-09-24T12:15:36.123456")
        platform = Platform(
            platform_locator=SatelliteSunSync(
                mjd0,
                orbittype="sgp4",
                period_from_altitude=600000.0,
                localtime_of_ascending_node_hours=5.20,
            )
        )
        mjd = mjd0 + np.arange(0, 100) / 1440.0
        looktanalt = [(35000.0, 5.0)]
        platform.add_measurement_set(
            mjd, ("from_platform"), ("tangent_from_orbitplane", "limb", looktanalt)
        )
        opgeom = platform.make_optical_geometry()
        # print("Done")

    # ------------------------------------------------------------------------------
    #           test_tangentalt_geometry
    # ------------------------------------------------------------------------------
    def test_tangentalt_geometry(self):
        platform = Platform()
        utc0 = "2020-09-24T12:15:36.123456"
        platform.platform_locator = SatelliteSunSync(
            utc0,
            orbittype="kepler",
            period_from_altitude=600000.0,
            localtime_of_ascending_node_hours=5.20,
        )

        utc = [utc0, utc0]
        llhobserver = [(52, -107, 600000), (53, -107, 600000)]
        looktanalt = [(35000.0, 45), (35000.0, 45)]
        azielev = [(15, 45.0), (23, 90.0)]

        geodetic = sk.Geodetic()
        geodetic.from_lat_lon_alt(52, -107, 600000.0)
        observer = geodetic.location
        up = geodetic.local_up
        north = -geodetic.local_south
        east = -geodetic.local_west
        plane = north * math.cos(math.radians(45.0)) + east * math.sin(math.radians(45))
        look = geodetic.from_tangent_altitude(35000.0, observer, plane)
        target = geodetic.location

        observer = [observer, observer]
        sixvector = [look[0], look[1], look[2], up[0], up[1], up[2]]
        sixvector = [sixvector, sixvector]
        targetxyz = [target, target]
        lookxyz = [look, look]
        targetllh = [geodetic.latitude, geodetic.longitude, geodetic.altitude]
        targetllh = [targetllh, targetllh]

        platform.add_measurement_set(
            utc, ("llh", llhobserver), ("tangent_altitude", "limb", looktanalt)
        )
        platform.add_measurement_set(
            utc, ("llh", llhobserver), ("tangent_xyz_look", "limb", lookxyz)
        )
        platform.add_measurement_set(
            utc, ("llh", llhobserver), ("location_xyz", "nadir", targetxyz)
        )
        platform.add_measurement_set(
            utc, ("llh", llhobserver), ("location_llh", "nadir", targetllh)
        )
        platform.add_measurement_set(
            utc, ("llh", llhobserver), ("unit_vectors", "none", sixvector)
        )
        platform.add_measurement_set(
            utc, ("llh", llhobserver), ("azi_elev", "standard", azielev)
        )
        platform.add_measurement_set(
            utc, ("llh", llhobserver), ("yaw_pitch_roll", "standard", azielev)
        )
        platform.add_measurement_set(utc, ("llh", llhobserver), ("from_platform"))
        platform.add_measurement_set(
            utc, ("xyz", observer), ("tangent_altitude", "limb", looktanalt)
        )
        platform.add_measurement_set(
            utc, ("from_platform"), ("tangent_altitude", "limb", looktanalt)
        )
        platform.add_measurement_set(
            utc, ("from_platform",), ("tangent_altitude", "limb", looktanalt)
        )
        platform.add_measurement_set(
            utc, ("from_platform", observer), ("tangent_altitude", "limb", looktanalt)
        )
        platform.add_measurement_set(
            utc, ("from_platform", observer), ("from_platform", "limb", looktanalt)
        )
        platform.add_measurement_set(
            utc, ("from_platform", observer), ("from_platform",)
        )
        platform.make_position_and_orientation_array()

    # ------------------------------------------------------------------------------
    #           test_sunsync_satellite
    # ------------------------------------------------------------------------------
    def test_sunsync_satellite(self):
        mjdequator = sktime.ut_to_mjd(
            "2020-07-15 14:50:00.000000"
        )  # Time when satellite crosses equator
        mjd0 = sktime.ut_to_mjd(
            "2020-07-15 15:00:00.000000"
        )  # Time for the start of measurements

        satellite = SatelliteSunSync(
            mjdequator,  # Create the sun-sync satellite
            period_from_altitude=450000.0,
            localtime_of_ascending_node_hours=13.5,
        )

        platform = Platform(
            platform_locator=satellite
        )  # Create the platform using the sun-sync satellite

        twosecs = 2.0 / 86400.0  # Get 2 seconds in MJD days
        tanheights = np.arange(
            8000.0, 40000.0, 250.0
        )  # Get the tangent height of each measurement, 8 km to 40 km in steps of 250 m
        num_meas = tanheights.size

        tanpoint = np.zeros(
            [num_meas, 2]
        )  # specify the parameters for the platform pointing technique
        tanpoint[
            :, 0
        ] = tanheights  # First element is the tangent height the platform will point towards
        tanpoint[
            :, 1
        ] = 0.0  # Second element is the geographic bearing from North. Not quite the same as bearing from orbit plane

        utc = (
            np.arange(num_meas) * twosecs + mjd0
        )  # Get the time of each measurement, 2 seconds apart

        platform.add_measurement_set(
            utc, ("from_platform"), ("tangent_from_orbitplane", "limb", tanpoint)
        )  # Create this measurement set
        platform.make_position_and_orientation_array()
        lookv = platform.icf_to_ecef(np.array([[1], [0], [0]]))
        pos = platform.platform_ecef_positions
        tp = np.zeros([3, num_meas])
        geodetic = sk.Geodetic()
        for i in range(num_meas):
            geodetic.from_tangent_point(pos[:, i], lookv[:, 0, i])
            tp[0, i] = geodetic.latitude
            tp[1, i] = geodetic.longitude
            tp[2, i] = geodetic.altitude
            dh = np.abs(geodetic.altitude - tanheights[i])
            self.assertLess(dh, 0.5)

    # ------------------------------------------------------------------------------
    #           test_backward_looking_system
    # ------------------------------------------------------------------------------
    def test_backward_looking_system(self):
        platform = Platform()
        utc = ["2020-07-15 14:00:00.0000000"]
        observer = [40.0, 0.0, 50000.0, 180.0, 691000.0]
        tanpoint = [50000.0, 45.0]
        platform.add_measurement_set(
            utc, ("looking_at_llh", observer), ("tangent_altitude", "limb", tanpoint)
        )
        obspolicy = platform.make_position_and_orientation_array()
        return platform


if __name__ == "__main__":
    tests = SkretrievalGeometryStatesTests()
    unittest.main()
