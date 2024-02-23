from __future__ import annotations

import logging
import time

import numpy as np


class Timer:
    """
    Simple wrapper to time things easier.
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            logging.info(self.name)
        msg = "Elapsed: " + str(time.time() - self.tstart) + "s"
        logging.info(msg)


def rotation_matrix(axis, angle):
    """
    Create a 3D rotation matrix using the Euler-Rodrigues formula

    :param axis: axis of rotation
    :param angle: angle to rotate around the axis (radians)
    :return: 3x3 rotation matrix
    """

    unit_axis = axis / np.linalg.norm(axis)

    half_angle_rad = angle / 2.0
    sin_half_angle_rad = np.sin(half_angle_rad)
    cos_half_angle_rad = np.cos(half_angle_rad)
    a = cos_half_angle_rad
    b = unit_axis[0] * sin_half_angle_rad
    c = unit_axis[1] * sin_half_angle_rad
    d = unit_axis[2] * sin_half_angle_rad

    R = np.zeros((3, 3))
    R[0, 0] = a * a + b * b - c * c - d * d
    R[0, 1] = 2 * (b * c - a * d)
    R[0, 2] = 2 * (b * d + a * c)
    R[1, 0] = 2 * (b * c + a * d)
    R[1, 1] = a * a + c * c - b * b - d * d
    R[1, 2] = 2 * (c * d - a * b)
    R[2, 0] = 2 * (b * d - a * c)
    R[2, 1] = 2 * (c * d + a * b)
    R[2, 2] = a * a + d * d - b * b - c * c

    return R


def linear_interpolating_matrix(from_grid: np.array, to_grid: np.array):
    M = np.zeros((len(to_grid), len(from_grid)))

    for idx, ele in enumerate(to_grid):
        try:
            idx_above = np.nonzero(from_grid > ele)[0][0]
            interp_ele = ele
            if idx_above == 0:
                idx_above = 1
                interp_ele = from_grid[0]
        except Exception as _:
            idx_above = len(from_grid)
            interp_ele = from_grid[0]

        if idx_above == len(from_grid):
            M[idx, idx_above - 1] = 1
        else:
            w = (from_grid[idx_above] - interp_ele) / (
                from_grid[idx_above] - from_grid[idx_above - 1]
            )
            M[idx, idx_above] = 1 - w
            M[idx, idx_above - 1] = w

    return M


def configure_log():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    class ExFormatter(logging.Formatter):
        def_keys = [  # noqa: RUF012
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "message",
        ]

        def format(self, record):
            string = super().format(record)
            extra = {k: v for k, v in record.__dict__.items() if k not in self.def_keys}
            if len(extra) > 0:
                string += " - " + str(extra)
            return string

    logger.addHandler(logging.StreamHandler())
    logger.handlers[0].setFormatter(ExFormatter())
