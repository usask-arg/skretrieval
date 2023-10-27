from __future__ import annotations

import numpy as np
from sasktran import geometry

from skretrieval.core import OpticalGeometry


class ScanningSatellite:
    def __init__(
        self,
        instrument,
        engine,
    ):
        self._engine = engine
        self._instrument = instrument


def fake_satellite_scan(altitudes=None):
    geo = geometry.VerticalImage()

    if altitudes is None:
        altitudes = np.arange(25, 40, 0.1)

    geo.from_sza_saa(60, 60, 0, 0, altitudes, 54372, 0)

    optical_axis = []

    for los in geo.lines_of_sight:
        local_up = los.observer / np.linalg.norm(los.observer)

        optical_axis.append(
            OpticalGeometry(
                look_vector=los.look_vector,
                observer=los.observer,
                mjd=los.mjd,
                local_up=local_up,
            )
        )

    return optical_axis, geo.sun
