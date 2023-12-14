from __future__ import annotations

import unittest

import numpy as np
import sasktran as sk

from skretrieval.core.lineshape import Gaussian, Rectangle
from skretrieval.core.platform.satellite import fake_satellite_scan
from skretrieval.legacy.core.sensor.spectrograph import Spectrograph
from skretrieval.legacy.core.simulator import MeasurementSimulator


class TestMeasSimulator(unittest.TestCase):
    def test_basic(self):
        axis, sun = fake_satellite_scan(altitudes=[20, 25, 30, 35, 40])

        atmo = sk.Atmosphere()
        atmo["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())
        atmo["ozone"] = sk.Species(sk.O3DBM(), sk.Labow())

        sensor = Spectrograph(
            np.arange(300, 330, 1),
            pixel_shape=Gaussian(fwhm=1, mode="linear"),
            vert_fov=Gaussian(fwhm=0.001),
            horiz_fov=Rectangle(width=1, mode="constant"),
        )

        simulator = MeasurementSimulator(
            sensor, axis, atmo, options={"numordersofscatter": 1}
        )

        simulator.engine.geometry.sun = sun

        simulator.calculate_radiance()
