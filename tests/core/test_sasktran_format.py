from __future__ import annotations

import numpy as np
import sasktran as sk

from skretrieval.core.sasktranformat import SASKTRANRadiance


def test_sasktran_legacy_format_vector():
    tanalts_km = np.arange(10, 50, 1)

    # First recreate our geometry and atmosphere classes
    geometry = sk.VerticalImage()
    geometry.from_sza_saa(
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

    atmosphere = sk.Atmosphere()

    atmosphere["ozone"] = sk.Species(sk.O3OSIRISRes(), sk.Labow())
    atmosphere["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())

    # And now make the engine
    engine = sk.EngineHR(geometry=geometry, atmosphere=atmosphere)

    # Choose some wavelengths to do the calculation at
    engine.wavelengths = [340, 600]
    engine.polarization = "vector"

    atmosphere.wf_species = "ozone"

    # And do the calculation
    radiance = engine.calculate_radiance(
        full_stokes_vector=True, output_format="xarray"
    )

    _ = SASKTRANRadiance.from_sasktran_legacy_xr(radiance)


def test_sasktran_legacy_format_scalar():
    tanalts_km = np.arange(10, 50, 1)

    # First recreate our geometry and atmosphere classes
    geometry = sk.VerticalImage()
    geometry.from_sza_saa(
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

    atmosphere = sk.Atmosphere()

    atmosphere["ozone"] = sk.Species(sk.O3OSIRISRes(), sk.Labow())
    atmosphere["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())

    # And now make the engine
    engine = sk.EngineHR(geometry=geometry, atmosphere=atmosphere)

    # Choose some wavelengths to do the calculation at
    engine.wavelengths = [340, 600]
    # engine.polarization = 'vector'

    atmosphere.wf_species = "ozone"

    # And do the calculation
    radiance = engine.calculate_radiance(
        full_stokes_vector=False, output_format="xarray"
    )

    _ = SASKTRANRadiance.from_sasktran_legacy_xr(radiance)


def test_sasktran2_format():
    pass
