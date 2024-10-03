from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk2

sk = pytest.importorskip("sasktran")

from skretrieval.core import OpticalGeometry  # noqa: E402
from skretrieval.core.lineshape import Gaussian, Rectangle  # noqa: E402
from skretrieval.core.sasktranformat import SASKTRANRadiance  # noqa: E402
from skretrieval.core.sensor.spectrograph import Spectrograph  # noqa: E402
from skretrieval.legacy.util import convert_sasktran_legacy_geometry  # noqa: E402


def test_spectrograph():
    alt_grid = np.arange(0, 65001, 1000.0)

    tanalts_km = np.arange(10, 60, 2.0)

    geo = sk.VerticalImage()

    geo.from_sza_saa(
        sza=60,
        saa=60,
        lat=0,
        lon=0,
        tanalts_km=tanalts_km,
        mjd=54372,
        locallook=0,
        satalt_km=600,
        refalt_km=20,
    )

    new_geo = convert_sasktran_legacy_geometry(geo)

    model_geometry = sk2.Geometry1D(
        np.cos(np.deg2rad(60)),
        0,
        6372000,
        alt_grid,
        sk2.InterpolationMethod.LinearInterpolation,
        sk2.GeometryType.Spherical,
    )

    config = sk2.Config()

    engine = sk2.Engine(config, model_geometry, new_geo)

    atmosphere = sk2.Atmosphere(
        model_geometry, config, wavelengths_nm=np.arange(280, 800, 0.1)
    )

    sk2.climatology.us76.add_us76_standard_atmosphere(atmosphere)

    atmosphere["rayleigh"] = sk2.constituent.Rayleigh()
    atmosphere["o3"] = sk2.climatology.mipas.constituent("O3", sk2.optical.O3DBM())

    radiance = SASKTRANRadiance.from_sasktran2(
        engine.calculate_radiance(atmosphere), collapse_scalar_stokes=True
    )
    radiance.data["look_vectors"] = (
        ["los", "xyz"],
        np.vstack([los.look_vector for los in geo.lines_of_sight]),
    )

    wavel_grid = np.arange(280, 800, 1.0)

    pixel_shape = [Gaussian(fwhm=1) for _ in wavel_grid]

    spectrograph = Spectrograph(
        wavel_grid, pixel_shape, Gaussian(fwhm=1), Rectangle(1, mode="constant")
    )

    optical_geo = OpticalGeometry(
        observer=geo.lines_of_sight[10].observer,
        look_vector=geo.lines_of_sight[10].look_vector,
        mjd=geo.lines_of_sight[10].mjd,
        local_up=geo.lines_of_sight[10].observer
        / np.linalg.norm(geo.lines_of_sight[10].observer),
    )

    _ = spectrograph.model_radiance(radiance=radiance, orientation=optical_geo)
