from __future__ import annotations

import numpy as np
from numpy.linalg import norm


def gibbs(pos1, pos2, pos3):
    """
    This function uses the Gibbs method of orbit determination from three position vectors. It computes the velocity
    corresponding to the second of three supplied position vectors. It is taken from section 5.2 and Appendix D.10 of
    Orbital Mechanics for Engineering Students, Howard Curtis, 2005, ISBN 0 7506 6169 0.

    Parameters
    ----------
    pos1: np.ndarray(3)
        The ECI position of the first point

    pos2: np.ndarray(3)
        The ECI position of the second point

    pos3: np.ndarray(3)
        The ECI position of the third point.

    Returns
    -------
    np.ndarray(3)
         the velocity corresponding to pos2 in km/s.
    """

    # mu =  398600.0
    mu = 3.986004418e14
    tol = 1e-4

    r1 = norm(pos1)  # Magnitudes of R1, R2 and R3:
    r2 = norm(pos2)
    r3 = norm(pos3)

    c12 = np.cross(pos1, pos2)  # Cross products among pos1, pos2 and pos3:
    c23 = np.cross(pos2, pos3)
    c31 = np.cross(pos3, pos1)

    if (
        np.abs(np.dot(pos1, c23) / r1 / norm(c23)) > tol
    ):  # Check that pos1, pos2 and pos3 are coplanar; if not set error flag:
        V2 = None
    else:
        N = r1 * c23 + r2 * c31 + r3 * c12  # Equation 5.13:
        D = c12 + c23 + c31  # Equation 5.14:
        S = pos1 * (r2 - r3) + pos2 * (r3 - r1) + pos3 * (r1 - r2)  # Equation 5.21:
        V2 = np.sqrt(mu / norm(N) / norm(D)) * (
            np.cross(D, pos2) / r2 + S
        )  # Equation 5.22:

    return V2
