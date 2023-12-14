from __future__ import annotations

import numpy as np
import sasktran as sk

from skretrieval.legacy.util import convert_sasktran_legacy_geometry


def test_convert_sasktran_legacy_geometry():
    tanalts_km = np.arange(10, 60, 1.0)

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

    _ = convert_sasktran_legacy_geometry(geo)
