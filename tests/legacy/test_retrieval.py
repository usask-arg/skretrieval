from __future__ import annotations

import unittest

import numpy as np
import sasktran as sk

from skretrieval.core.lineshape import Gaussian, Rectangle
from skretrieval.core.platform.satellite import fake_satellite_scan
from skretrieval.legacy.core.sensor.spectrograph import Spectrograph
from skretrieval.legacy.core.simulator import MeasurementSimulator
from skretrieval.legacy.retrieval.ozone import OzoneRetrieval
from skretrieval.retrieval.rodgers import Rodgers


class TestRetrieval(unittest.TestCase):
    def test_retrieval_basic(self):
        alts = np.arange(10, 60.5, 1)
        axis, sun = fake_satellite_scan(altitudes=alts)

        atmo = sk.Atmosphere()
        atmo["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())
        atmo["ozone"] = sk.Species(sk.O3DBM(), sk.Labow())

        sensor = Spectrograph(
            np.arange(590, 610, 1),
            pixel_shape=Gaussian(fwhm=1, mode="linear"),
            vert_fov=Gaussian(fwhm=0.0001),
            horiz_fov=Rectangle(width=1, mode="constant"),
        )

        options = {}
        options["numordersofscatter"] = 1

        simulator = MeasurementSimulator(sensor, axis, atmo, options)

        simulator.engine.geometry.sun = sun

        measurement_l1 = simulator.calculate_radiance()

        rodgers = Rodgers(max_iter=1)
        labow = sk.Labow()

        altitudes = np.linspace(500, 99500, 100)
        ozone = labow.get_parameter(
            "SKCLIMATOLOGY_O3_CM3",
            latitude=0,
            longitude=0,
            mjd=54372,
            altitudes=altitudes,
        )

        atmo_sim = sk.Atmosphere()
        atmo_sim["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())
        atmo_sim["ozone"] = sk.Species(
            sk.O3DBM(), sk.ClimatologyUserDefined(altitudes, {"ozone": ozone * 0.5})
        )

        atmo_sim.wf_species = "ozone"

        sensor = Spectrograph(
            np.arange(590, 610, 1),
            pixel_shape=Gaussian(fwhm=1, mode="linear"),
            vert_fov=Gaussian(fwhm=0.0001),
            horiz_fov=Rectangle(width=1, mode="constant"),
        )
        ozone_ret = OzoneRetrieval(atmo_sim["ozone"])

        options = {}
        options["numordersofscatter"] = 1
        options["calcwf"] = 2
        options["wfheights"] = ozone_ret._retrieval_altitudes
        options["wfwidths"] = np.ones_like(ozone_ret._retrieval_altitudes) * 1000

        forward_model = MeasurementSimulator(sensor, axis, atmo_sim, options)

        forward_model.engine.geometry.sun = sun

        rodgers.retrieve(measurement_l1, forward_model, ozone_ret)
