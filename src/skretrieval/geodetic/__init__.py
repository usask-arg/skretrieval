from __future__ import annotations

import sasktran as sk


def geodetic() -> sk.Geodetic:
    return sk.Geodetic("wgs84")
