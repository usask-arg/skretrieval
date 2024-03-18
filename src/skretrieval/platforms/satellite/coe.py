from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
from numpy.linalg import norm

# ------------------------------------------------------------------------------
#           ClassicalOrbitalElements
# ------------------------------------------------------------------------------
ClassicalOrbitalElements = namedtuple(
    "ClassicalOrbitalElements", ("H", "e", "RA", "i", "W", "TA", "a")
)
ClassicalOrbitalElements.__doc__ = """
    The 6 elements of classical orbital elements

    Parameters
    ----------
    H : float
        the angular momemtum (m^2/s)
    e : float
        eccentricity
    RA : float
        Right ascension of the ascending node in radians
    i : float
        inclination of the orbit
    W : float
        argument of perigee in radians
    TA : float
        True anomaly in radians
    a : float
        semimajor axis
    """


# ------------------------------------------------------------------------------
#           coe_from_state_vector
# ------------------------------------------------------------------------------
def coe_from_state_vector(R: np.ndarray, V: np.ndarray) -> ClassicalOrbitalElements:
    """
    Calculates classical orbital elements (ie Kepler) given a position and velocity. This is taken from
    Section 4.1 and Appendix D.8 Algorithm 4.1: of Orbital Mechanics for Engineering Students, Howard Curtis, 2005,
    ISBN 0 7506 6169 0.

    Parameters
    ----------
    R : np.ndarray(3)
        The position vector of the satellite in meters. This is typically expressed in ECI coordinates for Earth based
        satellites
    V : np.ndarray()
    mu - gravitational parameter (km^3/s^2)
    R - position vector in the geocentric equatorial frame (m)
    V - velocity vector in the geocentric equatorial frame (m/s)
    r, v - the magnitudes of R and V

    a - semimajor axis (km)
    pi - 3.1415926...
    coe - vector of orbital elements [h e RA incl w TA a]
    """

    mu = 3.986004418e14
    eps = 1.0e-10
    r = norm(R)
    v = norm(V)
    vr = np.dot(R, V) / r  # radial velocity component (m/s)
    H = np.cross(R, V)  # the angular momentum vector (m^2/s)
    h = norm(H)  # the magnitude of H (m^2/s)
    incl = math.acos(H[2] / h)  # Equation 4.7: inclination of the orbit (rad)
    N = np.cross([0, 0, 1], H)  # Equation 4.8: the node line vector (km^2/s)
    n = norm(N)  # the magnitude of N

    if n != 0:  # Equation 4.9: right ascension of the ascending node (rad)
        RA = math.acos(N[0] / n)
        if N[1] < 0:
            RA = 2 * math.pi - RA
    else:
        RA = 0

    E = (
        1.0 / mu * ((v * v - mu / r) * R - r * vr * V)
    )  # Equation 4.10: eccentricity vector
    e = norm(E)  # Calculate eccentricity

    if n != 0:  # Equation 4.12 (incorporating the case e = 0)
        if (
            e > eps
        ):  # eps - a small number below which the eccentricity is considered to be zero
            w = math.acos(np.dot(N, E) / n / e)  # w is argument of perigee (rad)
            if E[2] < 0:
                w = 2 * math.pi - w
        else:
            w = 0
    else:
        w = 0

    if e > eps:  # Equation 4.13a (incorporating the case e = 0):
        TA = math.acos(np.dot(E, R) / e / r)  # TA is true anomaly (rad)
        if vr < 0:
            TA = 2 * math.pi - TA
    else:
        cp = np.cross(N, R)
        if cp[2] >= 0:
            TA = math.acos(np.dot(N, R) / n / r)
        else:
            TA = 2 * math.pi - math.acos(np.dot(N, R) / n / r)

    a = (
        h * h / mu / (1.0 - e * e)
    )  # Equation 2.61, semimajor axis (m),  (a < 0 for a hyperbola)
    return ClassicalOrbitalElements(H=h, e=e, RA=RA, i=incl, W=w, TA=TA, a=a)
