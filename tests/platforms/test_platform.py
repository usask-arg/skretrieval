import unittest
import numpy as np
from skretrieval.platforms.platform import Platform


class SkretrievalPlatformTests(unittest.TestCase):
    def test_instrument_mounting(self):
        """
        Tests the instrument mounting in the platform control frame
        """
        platform = Platform()
        vicf = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))).transpose()
        utc = np.datetime64("2020-10-14T15:00:00.000000")
        platform.platform_pointing.set_platform_location(
            latlonheightandt=(0, 0, 0, utc)
        )

        platform.platform_pointing.mount_instrument_on_platform(60, 30.0, 60.0)
        vgeo = platform.platform_pointing.convert_icf_to_gcf(vicf)
        vecef = platform.platform_pointing.convert_icf_to_ecef(vicf)
        # print('rotation = (60,30,0)')
        # print('Geographic Unit vectors')
        # print(vgeo)
        # print('ECEF Unit vectors')
        # print(vecef)
        self.assertAlmostEqual(vgeo[0, 0], 0.75, 8)
        self.assertAlmostEqual(vgeo[0, 1], -0.625, 8)
        self.assertAlmostEqual(vgeo[0, 2], 0.21650635, 8)
        self.assertAlmostEqual(vgeo[1, 0], 0.4330127, 8)
        self.assertAlmostEqual(vgeo[1, 1], 0.21650635, 8)
        self.assertAlmostEqual(vgeo[1, 2], -0.875, 8)
        self.assertAlmostEqual(vgeo[2, 0], 0.5, 8)
        self.assertAlmostEqual(vgeo[2, 1], 0.75, 8)
        self.assertAlmostEqual(vgeo[2, 2], 0.4330127, 8)

        self.assertAlmostEqual(vgeo[0, 0], vecef[1, 0], 8)
        self.assertAlmostEqual(vgeo[1, 0], vecef[2, 0], 8)
        self.assertAlmostEqual(vgeo[2, 0], vecef[0, 0], 8)
        self.assertAlmostEqual(vgeo[0, 1], vecef[1, 1], 8)
        self.assertAlmostEqual(vgeo[1, 1], vecef[2, 1], 8)
        self.assertAlmostEqual(vgeo[2, 1], vecef[0, 1], 8)
        self.assertAlmostEqual(vgeo[0, 2], vecef[1, 2], 8)
        self.assertAlmostEqual(vgeo[1, 2], vecef[2, 2], 8)
        self.assertAlmostEqual(vgeo[2, 2], vecef[0, 2], 8)

    # ------------------------------------------------------------------------------
    #               def test_instrument_rotation_matrix(self):
    # ------------------------------------------------------------------------------
    def test_instrument_rotation_matrix(self):
        """
        Tests the rotation of the instrument in the instrument control frame.
        """
        platform = Platform()
        vicf = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))).transpose()
        utc = np.datetime64("2020-10-14T15:00:00.000000")
        platform.platform_pointing.set_platform_location(
            latlonheightandt=(0, 0, 0, utc)
        )

        platform.platform_pointing.rotate_instrument_in_icf(30, 60.0, 0)
        vgeo = platform.platform_pointing.convert_icf_to_gcf(vicf)
        vecef = platform.platform_pointing.convert_icf_to_ecef(vicf)
        # print('Geographic Unit vectors')
        # print(vgeo)
        # print('ECEF Unit vectors')
        # print(vecef)
        # print('Done')
        self.assertAlmostEqual(vgeo[0, 0], 0.25, 8)
        self.assertAlmostEqual(vgeo[0, 1], -0.8660254, 8)
        self.assertAlmostEqual(vgeo[0, 2], -0.4330127, 8)
        self.assertAlmostEqual(vgeo[1, 0], 0.4330127, 8)
        self.assertAlmostEqual(vgeo[1, 1], 0.5, 8)
        self.assertAlmostEqual(vgeo[1, 2], -0.75, 8)
        self.assertAlmostEqual(vgeo[2, 0], 0.8660254, 8)
        self.assertAlmostEqual(vgeo[2, 1], 0.0, 8)
        self.assertAlmostEqual(vgeo[2, 2], 0.5, 8)

        self.assertAlmostEqual(vgeo[0, 0], vecef[1, 0], 8)
        self.assertAlmostEqual(vgeo[1, 0], vecef[2, 0], 8)
        self.assertAlmostEqual(vgeo[2, 0], vecef[0, 0], 8)
        self.assertAlmostEqual(vgeo[0, 1], vecef[1, 1], 8)
        self.assertAlmostEqual(vgeo[1, 1], vecef[2, 1], 8)
        self.assertAlmostEqual(vgeo[2, 1], vecef[0, 1], 8)
        self.assertAlmostEqual(vgeo[0, 2], vecef[1, 2], 8)
        self.assertAlmostEqual(vgeo[1, 2], vecef[2, 2], 8)
        self.assertAlmostEqual(vgeo[2, 2], vecef[0, 2], 8)


if __name__ == "__main__":
    tests = SkretrievalPlatformTests()
    unittest.main()
