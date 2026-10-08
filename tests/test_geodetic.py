from __future__ import annotations

import numpy as np
import pytest
from sasktran2.geodetic import WGS84

from skretrieval.geodetic import geodetic, target_lat_lon_alt

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def _ellipsoid_ground_intercept(obs, look):
    """
    Independent reference: first intersection of obs + t * look with the WGS84 ellipsoid and its
    geodetic latitude/longitude in degrees.
    """
    scale = np.array([1 / WGS84_A, 1 / WGS84_A, 1 / WGS84_B])
    o, d = obs * scale, look * scale
    a, b, c = d @ d, 2 * o @ d, o @ o - 1
    t = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    point = obs + t * look

    lat = np.rad2deg(
        np.arctan2(point[2], (1 - WGS84_E2) * np.hypot(point[0], point[1]))
    )
    lon = np.rad2deg(np.arctan2(point[1], point[0])) % 360
    return lat, lon, t


def _location(lat, lon, alt):
    g = geodetic()
    g.from_lat_lon_alt(lat, lon, alt)
    return g


def test_geodetic_returns_wgs84():
    assert isinstance(geodetic(), WGS84)


@pytest.mark.parametrize(
    ("lat", "lon", "alt", "expected_xyz"),
    [
        (0.0, 0.0, 0.0, [WGS84_A, 0.0, 0.0]),
        (0.0, 90.0, 1000.0, [0.0, WGS84_A + 1000.0, 0.0]),
        (0.0, 180.0, 0.0, [-WGS84_A, 0.0, 0.0]),
        (90.0, 0.0, 0.0, [0.0, 0.0, WGS84_B]),
        (-90.0, 0.0, 500.0, [0.0, 0.0, -WGS84_B - 500.0]),
    ],
)
def test_geodetic_reference_points(lat, lon, alt, expected_xyz):
    np.testing.assert_allclose(
        _location(lat, lon, alt).location, expected_xyz, atol=1e-6
    )


def test_geodetic_round_trip():
    g = _location(-37.5, 123.25, 12345.0)

    other = geodetic()
    other.from_xyz(g.location)

    np.testing.assert_allclose(
        (other.latitude, other.longitude, other.altitude),
        (-37.5, 123.25, 12345.0),
        atol=1e-6,
    )


def test_geodetic_returns_independent_instances():
    first = _location(10.0, 20.0, 0.0)
    second = _location(-45.0, 200.0, 5000.0)

    assert first is not second
    np.testing.assert_allclose(first.latitude, 10.0)
    np.testing.assert_allclose(first.longitude, 20.0)


def test_target_lat_lon_alt_limb_equator_hand_computed():
    tangent_alt = 25000.0
    # Looking north, horizontally, past the point above lat=0, lon=0
    look = np.array([0.0, 0.0, 1.0])
    obs = np.array([WGS84_A + tangent_alt, 0.0, -2.5e6])

    lat, lon, alt = target_lat_lon_alt(look, obs)

    np.testing.assert_allclose(lat, 0.0, atol=1e-9)
    np.testing.assert_allclose(lon % 360, 0.0, atol=1e-9)
    np.testing.assert_allclose(alt, tangent_alt, atol=1e-6)


@pytest.mark.parametrize(
    ("lat", "lon", "tangent_alt"),
    [(45.0, 30.0, 25000.0), (-60.0, 200.0, 10000.0), (-5.0, 340.0, 50000.0)],
)
@pytest.mark.parametrize("direction", ["local_south", "local_west"])
def test_target_lat_lon_alt_limb_returns_tangent_point(
    lat, lon, tangent_alt, direction
):
    tangent = _location(lat, lon, tangent_alt)
    # Horizontal look direction at the tangent point, observer 2500 km back along the LOS
    look = getattr(tangent, direction)
    obs = tangent.location - 2.5e6 * look

    result = target_lat_lon_alt(look, obs)

    np.testing.assert_allclose(result, (lat, lon, tangent_alt), rtol=0, atol=1e-5)
    assert all(isinstance(v, float) for v in result)


def test_target_lat_lon_alt_returns_longitude_in_0_360():
    tangent = _location(10.0, -20.0, 30000.0)
    look = tangent.local_west

    _, lon, _ = target_lat_lon_alt(look, tangent.location - 2e6 * look)

    np.testing.assert_allclose(lon, 340.0)


@pytest.mark.parametrize(
    ("obs", "expected_lat_lon"),
    [
        (np.array([WGS84_A + 600e3, 0.0, 0.0]), (0.0, 0.0)),
        (np.array([0.0, WGS84_A + 600e3, 0.0]), (0.0, 90.0)),
        (np.array([0.0, -(WGS84_A + 600e3), 0.0]), (0.0, 270.0)),
    ],
)
def test_target_lat_lon_alt_nadir_hand_computed(obs, expected_lat_lon):
    look = -obs / np.linalg.norm(obs)

    lat, lon, alt = target_lat_lon_alt(look, obs)

    np.testing.assert_allclose((lat, lon), expected_lat_lon, atol=1e-9)
    np.testing.assert_allclose(alt, 0.0, atol=1e-3)


@pytest.mark.parametrize(("lat", "lon"), [(10.0, 50.0), (-75.0, 300.0)])
def test_target_lat_lon_alt_nadir_returns_sub_observer_point(lat, lon):
    observer = _location(lat, lon, 600e3)

    result = target_lat_lon_alt(-observer.local_up, observer.location)

    np.testing.assert_allclose(result, (lat, lon, 0.0), atol=1e-3)


@pytest.mark.parametrize(
    ("obs_lat", "obs_lon", "nadir_angle", "toward"),
    [
        (0.0, 0.0, 30.0, "local_south"),
        (52.1, 253.4, 45.0, "local_west"),
        (-20.0, 100.0, 10.0, "local_south"),
    ],
)
def test_target_lat_lon_alt_oblique_returns_near_ground_intercept(
    obs_lat, obs_lon, nadir_angle, toward
):
    observer = _location(obs_lat, obs_lon, 600e3)
    obs = observer.location
    theta = np.deg2rad(nadir_angle)
    look = -np.cos(theta) * observer.local_up - np.sin(theta) * getattr(
        observer, toward
    )

    lat, lon, alt = target_lat_lon_alt(look, obs)

    expected_lat, expected_lon, distance = _ellipsoid_ground_intercept(obs, look)
    # sasktran2's intercept solver is accurate to a few tens of cm
    np.testing.assert_allclose(lat, expected_lat, atol=1e-6)
    np.testing.assert_allclose(lon, expected_lon, atol=1e-6)
    np.testing.assert_allclose(alt, 0.0, atol=1.0)
    # The near intercept is no more than a few hundred km beyond the 600 km altitude
    assert 600e3 < distance < 1.2e6


def test_target_lat_lon_alt_tangent_below_ground_returns_ground_intercept():
    # Horizontal LOS whose tangent point would be 20 km below the surface
    tangent = _location(30.0, 60.0, -20000.0)
    look = tangent.local_south
    obs = tangent.location - 2.5e6 * look

    lat, lon, alt = target_lat_lon_alt(look, obs)

    expected_lat, expected_lon, distance = _ellipsoid_ground_intercept(obs, look)
    np.testing.assert_allclose(alt, 0.0, atol=1.0)
    np.testing.assert_allclose((lat, lon), (expected_lat, expected_lon), atol=1e-6)
    # Ground is hit before reaching the (underground) tangent point, so north of it
    assert distance < 2.5e6
    assert lat > 30.0
