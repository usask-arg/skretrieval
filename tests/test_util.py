from __future__ import annotations

import contextlib
import io
import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from skretrieval import util
from skretrieval.util import (
    Timer,
    configure_log,
    linear_interpolating_matrix,
    rotation_matrix,
)

# ---------------------------------------------------------------------------------------
# rotation_matrix
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "angle"),
    [
        ([0.0, 0.0, 1.0], np.pi / 2),
        ([1.0, 0.0, 0.0], -0.3),
        ([0.0, 2.0, 0.0], 2.5),  # non-unit axis
        ([1.0, -2.0, 0.5], 1.234),
        ([0.3, 0.1, -0.7], 0.0),
        ([0.3, 0.1, -0.7], np.pi),
    ],
)
def test_rotation_matrix_matches_scipy(axis, angle):
    axis = np.asarray(axis)

    expected = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()

    np.testing.assert_allclose(rotation_matrix(axis, angle), expected, atol=1e-15)


def test_rotation_matrix_quarter_turn_about_z():
    R = rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2)

    np.testing.assert_allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(R @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(R @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=1e-15)


def test_rotation_matrix_is_proper_rotation_about_axis():
    axis = np.array([1.0, -2.0, 0.5])
    angle = 0.8

    R = rotation_matrix(axis, angle)

    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(R), 1.0)
    np.testing.assert_allclose(R @ axis, axis, atol=1e-15)
    np.testing.assert_allclose(R @ rotation_matrix(axis, -angle), np.eye(3), atol=1e-15)

    # Angle between a perpendicular vector and its rotation is the rotation angle
    perp = np.cross(axis, [0.0, 0.0, 1.0])
    rotated = R @ perp
    cos_angle = perp @ rotated / (np.linalg.norm(perp) * np.linalg.norm(rotated))
    np.testing.assert_allclose(cos_angle, np.cos(angle))


# ---------------------------------------------------------------------------------------
# linear_interpolating_matrix
# ---------------------------------------------------------------------------------------


def test_linear_interpolating_matrix_hand_computed():
    from_grid = np.array([0.0, 1.0, 2.0])
    to_grid = np.array([-1.0, 0.0, 0.25, 1.5, 2.0, 3.0])

    M = linear_interpolating_matrix(from_grid, to_grid)

    np.testing.assert_allclose(
        M,
        [
            [1.0, 0.0, 0.0],  # below the grid, clamped to the first value
            [1.0, 0.0, 0.0],
            [0.75, 0.25, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],  # above the grid, clamped to the last value
        ],
    )


def test_linear_interpolating_matrix_matches_np_interp():
    rng = np.random.default_rng(3)
    from_grid = np.sort(rng.uniform(0, 10, 12))
    # Points inside, outside and exactly on the from grid
    to_grid = np.concatenate([rng.uniform(-2, 12, 40), from_grid])
    values = rng.normal(size=from_grid.shape)

    M = linear_interpolating_matrix(from_grid, to_grid)

    assert M.shape == (len(to_grid), len(from_grid))
    np.testing.assert_allclose(M @ values, np.interp(to_grid, from_grid, values))
    np.testing.assert_allclose(M.sum(axis=1), 1.0)
    assert np.all(M >= 0)
    assert np.all(np.count_nonzero(M, axis=1) <= 2)


def test_linear_interpolating_matrix_identity_on_same_grid():
    grid = np.array([0.0, 0.5, 2.0, 2.1, 7.0])

    np.testing.assert_array_equal(linear_interpolating_matrix(grid, grid), np.eye(5))


def test_linear_interpolating_matrix_reproduces_linear_functions():
    from_grid = np.array([0.0, 0.3, 1.0, 4.0, 4.5])
    to_grid = np.linspace(0, 4.5, 23)

    M = linear_interpolating_matrix(from_grid, to_grid)

    np.testing.assert_allclose(M @ (3.0 * from_grid - 2.0), 3.0 * to_grid - 2.0)


def test_linear_interpolating_matrix_single_point_grid():
    M = linear_interpolating_matrix(np.array([1.0]), np.array([0.0, 1.0, 2.0]))

    np.testing.assert_array_equal(M, np.ones((3, 1)))


# ---------------------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------------------


def _fake_clock(monkeypatch, start, stop):
    times = iter([start])
    monkeypatch.setattr(util.time, "time", lambda: next(times, stop))


def test_timer_logs_name_and_elapsed(monkeypatch, caplog):
    _fake_clock(monkeypatch, 100.0, 102.5)

    with caplog.at_level(logging.INFO), Timer("my block"):
        pass

    assert caplog.messages == ["my block", "Elapsed: 2.5s"]


def test_timer_without_name_logs_only_elapsed(monkeypatch, caplog):
    _fake_clock(monkeypatch, 10.0, 10.25)

    with caplog.at_level(logging.INFO), Timer():
        pass

    assert caplog.messages == ["Elapsed: 0.25s"]


def _timed_failure():
    with Timer("failing"):
        msg = "boom"
        raise RuntimeError(msg)


def test_timer_logs_even_when_block_raises(caplog):
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="boom"):
        _timed_failure()

    assert caplog.messages[0] == "failing"
    assert caplog.messages[1].startswith("Elapsed: ")


# ---------------------------------------------------------------------------------------
# configure_log
# ---------------------------------------------------------------------------------------


@contextlib.contextmanager
def _isolated_root_logger(handlers=()):
    """
    Temporarily replace the root logger handlers/level, restoring them afterwards.  Used inside
    the test body since pytest attaches its own capture handlers to the root logger for the
    duration of each test.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    for h in saved_handlers:
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    try:
        yield root
    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)


def test_configure_log_installs_info_stream_handler():
    with _isolated_root_logger() as root:
        root.setLevel(logging.WARNING)

        configure_log()

        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)

        stream = io.StringIO()
        handler.setStream(stream)
        logger = logging.getLogger("skretrieval.tests.util")
        logger.info("plain message")
        logger.info("with extras", extra={"retrieval": "ozone", "iteration": 3})
        logger.debug("below the configured level")

    assert stream.getvalue().splitlines() == [
        "plain message",
        "with extras - {'retrieval': 'ozone', 'iteration': 3}",
    ]


def test_configure_log_formats_its_own_handler():
    existing_formatter = logging.Formatter("existing: %(message)s")
    existing = logging.StreamHandler(io.StringIO())
    existing.setFormatter(existing_formatter)

    with _isolated_root_logger(handlers=[existing]) as root:
        configure_log()

        added = [h for h in root.handlers if h is not existing]
        assert existing.formatter is existing_formatter
        assert len(added) == 1
        assert type(added[0].formatter).__name__ == "ExFormatter"
