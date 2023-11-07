from __future__ import annotations

import unittest

import numpy as np
from sasktran import Geometry, LineOfSight

from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import Gaussian
from skretrieval.legacy.core.sensor.spectrograph import Spectrograph


class TestSpectrograph(unittest.TestCase):
    def test_initialization(self):
        wavel_nm = np.linspace(200, 300, 101)

        line_shape = Gaussian(fwhm=1)

        _ = Spectrograph(
            wavel_nm, line_shape, vert_fov=Gaussian(fwhm=1), horiz_fov=Gaussian(fwhm=1)
        )

    def test_line_shape(self):
        wavel_nm = np.linspace(200, 300, 101)

        line_shape = Gaussian(fwhm=1)

        optical_geo = OpticalGeometry(
            observer=np.array([0, 0, 1]),
            look_vector=np.array([0, 1, 0]),
            local_up=np.array([0, 0, 1]),
            mjd=54372,
        )

        spectrograph = Spectrograph(
            wavel_nm, line_shape, Gaussian(fwhm=1), Gaussian(fwhm=1)
        )

        hires_wavel = spectrograph.required_wavelengths(0.01)

        model_geometry = Geometry()

        angles = np.linspace(-0.1, 0.1, 100)

        for a in angles:
            model_geometry.lines_of_sight.append(
                LineOfSight(
                    mjd=54372, observer=[0, 0, 1], look_vector=[0, np.cos(a), np.sin(a)]
                )
            )

        rad = np.zeros((len(hires_wavel), len(angles)))

        rad[300, 50] = 1

        _ = spectrograph.model_radiance(optical_geo, hires_wavel, model_geometry, rad)
