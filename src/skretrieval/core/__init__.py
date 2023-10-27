from __future__ import annotations

from collections import namedtuple

# Quantities that are necessary to uniquely define the measurement geometry for a specified instrument
OpticalGeometry = namedtuple(
    "OpticalGeometry", ["observer", "look_vector", "local_up", "mjd"]
)
