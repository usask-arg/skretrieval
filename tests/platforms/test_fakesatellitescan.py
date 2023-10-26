import unittest as unittest
from skretrieval.core.platform.satellite import fake_satellite_scan
from skretrieval.core.sensor.spectrograph import Spectrograph
from skretrieval.core.lineshape import Gaussian
import numpy as np


class TestFakeSatelliteScan(unittest.TestCase):
    def test_basic(self):
        axis = fake_satellite_scan()

        pass