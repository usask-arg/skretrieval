from __future__ import annotations

import unittest

import numpy as np
from sasktran import Geometry, LineOfSight

from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import Gaussian
from skretrieval.legacy.core.sensor.pixel import Pixel


class TestPixel(unittest.TestCase):
    def test_initialization(self):
        wavel_nm = 350

        line_shape = Gaussian(fwhm=1)

        _ = Pixel(wavel_nm, line_shape, Gaussian(fwhm=1), Gaussian(fwhm=1))

    def test_line_shape(self):
        wavel_nm = 350

        line_shape = Gaussian(fwhm=1)

        optical_geo = OpticalGeometry(
            observer=[0, 0, 1], look_vector=[0, 1, 0], local_up=[0, 0, 1], mjd=54372
        )

        pixel = Pixel(wavel_nm, line_shape, Gaussian(fwhm=1), Gaussian(fwhm=1))

        hires_wavel = pixel.required_wavelengths(0.01)

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

        _ = pixel.model_radiance(optical_geo, hires_wavel, model_geometry, rad)
