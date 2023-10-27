from __future__ import annotations

import unittest

from skretrieval.core.platform.satellite import fake_satellite_scan


class TestFakeSatelliteScan(unittest.TestCase):
    def test_basic(self):
        _ = fake_satellite_scan()
