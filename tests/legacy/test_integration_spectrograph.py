from __future__ import annotations

import unittest

import numpy as np
import sasktran as sk

from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import DeltaFunction, Gaussian
from skretrieval.legacy.core.sensor.spectrograph import Spectrograph


class TestSpectrographIntegration(unittest.TestCase):
    def setUp(self):
        self.optical_geometry = OpticalGeometry(
            observer=np.array([0, 6392 * 1000, 6392 * 1000]),
            look_vector=np.array([0, -1, 0]),
            local_up=np.array([0, 0, 1]),
            mjd=54372,
        )

        self.wavel_nm = np.arange(300, 330, 1)

        line_shape = Gaussian(fwhm=1, mode="linear")
        self.spectrograph = Spectrograph(
            self.wavel_nm, line_shape, DeltaFunction(), DeltaFunction()
        )

    def test_radiance(self):
        geo = sk.Geometry()
        geo.lines_of_sight = self.spectrograph.measurement_geometry(
            self.optical_geometry
        )

        atmo = sk.Atmosphere()
        atmo["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())
        atmo["ozone"] = sk.Species(sk.O3DBM(), sk.Labow())

        engine = sk.EngineHR(atmosphere=atmo, geometry=geo)
        engine.options["numordersofscatter"] = 1

        engine.wavelengths = self.spectrograph.required_wavelengths(0.1)

        rad = engine.calculate_radiance()

        _ = self.spectrograph.model_radiance(
            self.optical_geometry, engine.wavelengths, engine.geometry, rad
        )

    def test_weighting_function(self):
        geo = sk.Geometry()
        geo.lines_of_sight = self.spectrograph.measurement_geometry(
            self.optical_geometry
        )

        atmo = sk.Atmosphere()
        atmo["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())
        atmo["ozone"] = sk.Species(sk.O3DBM(), sk.Labow())

        atmo.wf_species = "ozone"

        engine = sk.EngineHR(atmosphere=atmo, geometry=geo)
        engine.options["numordersofscatter"] = 1

        engine.wavelengths = self.spectrograph.required_wavelengths(0.1)

        rad, wf = engine.calculate_radiance()

        _ = self.spectrograph.model_radiance(
            self.optical_geometry, engine.wavelengths, engine.geometry, rad, wf
        )
