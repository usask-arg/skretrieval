from __future__ import annotations

import numpy as np
import sasktran2 as sk

import skretrieval.retrieval.measvec as mv
from skretrieval.core.lineshape import DeltaFunction
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.core.sensor.spectrograph import SpectrographOnlySpectral


def _construct_l1():
    config = sk.Config()

    model_geometry = sk.Geometry1D(
        cos_sza=0.6,
        solar_azimuth=0,
        earth_radius_m=6372000,
        altitude_grid_m=np.arange(0, 65001, 1000),
        interpolation_method=sk.InterpolationMethod.LinearInterpolation,
        geometry_type=sk.GeometryType.Spherical,
    )

    viewing_geo = sk.ViewingGeometry()

    t_alts = np.arange(500, 60000, 1000.0)

    for alt in t_alts:
        ray = sk.TangentAltitudeSolar(
            tangent_altitude_m=alt,
            relative_azimuth=0,
            observer_altitude_m=200000,
            cos_sza=0.6,
        )
        viewing_geo.add_ray(ray)

    wavel = np.arange(280.0, 800.0, 1)
    atmosphere = sk.Atmosphere(
        model_geometry,
        config,
        wavelengths_nm=wavel,
        pressure_derivative=False,
        temperature_derivative=False,
        specific_humidity_derivative=False,
    )

    sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)

    atmosphere["rayleigh"] = sk.constituent.Rayleigh()
    atmosphere["ozone"] = sk.climatology.mipas.constituent("O3", sk.optical.O3DBM())
    atmosphere["no2"] = sk.climatology.mipas.constituent(
        "NO2", sk.optical.NO2Vandaele()
    )

    engine = sk.Engine(config, model_geometry, viewing_geo)
    output = engine.calculate_radiance(atmosphere)

    inst_model = SpectrographOnlySpectral(wavel, [DeltaFunction() for _ in wavel])

    l1 = inst_model.model_radiance(SASKTRANRadiance.from_sasktran2(output), None)

    l1.data = l1.data.drop_vars(["wf_no2_vmr"])
    l1.data = l1.data.rename_vars({"wf_ozone_vmr": "wf"})
    l1.data = l1.data.rename_dims({"ozone_altitude": "x"})

    l1.data.coords["tangent_altitude"] = (["los"], t_alts)
    l1.data = l1.data.set_xindex("tangent_altitude")

    return l1


def test_selector():
    l1 = _construct_l1()

    l1 = mv.pre_process({"measurement": l1})

    mv.select(l1, wavelength=500, tangent_altitude=slice(10000, 40000))
