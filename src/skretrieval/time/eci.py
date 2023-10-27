from __future__ import annotations

import datetime
from collections.abc import Sequence

import astropy.time
import numpy as np
from astropy.utils import iers

from .mjd import utc_to_astropy


# ------------------------------------------------------------------------------
#           gmst
# Calculates the Greenwich Mean Sidereal time.
# ------------------------------------------------------------------------------
def gmst(
    utc: np.datetime64 | (datetime.datetime | (float | np.ndarray)), **kwargs
) -> float | np.ndarray:
    """
    Calculates the Greenwich Mean Sidereal Time for the given Universal time(s). The Universal time is UTC. We use the
    `astropy.time.Time.sidereal_time` function to calculate sidereal time; this function uses IERS corrections for timescales and
    SOFA algorithms. It is is very accurate. The algorithm has given the same results as the novas.sidereal_time function
    to 10 or 11 decimal places.

    Parameters
    ----------
    utc: float, list or array of times.
        A scalar,array or list of coordinated universal times. These Sideral time will be calculated for each of these elements. The
        UTC can be given in any of the formats supported by `juliandate.ut_to_jd` but are typically datetime.datetime
        or numpy.datetime.

    ** kwargs : Optional keyword arguments
        User can pass in optional keyword arguments for method `astropy.time.Time.sideral_time`. Typically used for applications
        that need to specify the IAU algorithm, eg `IAU1982`.

    Returns
    -------
    float or numpy.ndarray
        The greenwich mean sidereal time. This is expressed as a angle in radians between 0 of :math:`2pi`.

    """
    t = utc_to_astropy(utc)
    st = t.sidereal_time("mean", "greenwich", **kwargs)
    return st.radian


# ------------------------------------------------------------------------------
#           rotate_about_zaxis
# ------------------------------------------------------------------------------
def _rotate_about_zaxis(u: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
    v = np.empty_like(u)
    costheta = np.cos(theta)  # do the rotation around the Z axis
    sintheta = np.sin(theta)
    v[0, ...] = costheta * u[0, ...] + sintheta * u[1, ...]
    v[1, ...] = -sintheta * u[0, ...] + costheta * u[1, ...]
    v[2, ...] = u[2, ...]
    return v


# ------------------------------------------------------------------------------
#           polar_motion
# ------------------------------------------------------------------------------
def polar_motion_rotation_matrix(utc, transpose=False) -> tuple[float, float]:
    """
    Returns the polar motion rotation matrix for the given times. The rotation matrix is written as the matrix product
    of :math:`ROT1(y_p)*ROT2(x_p)` where ROT1 is rotation around the global x axis and ROT2 is rotation around the global
    y axis.

    ..  math::

        \\begin{equation}
        \\mathbf{ROT1} =
        \\left( \\begin{array}{ccc}
        \\cos x & 0  & -\\sin x  \\\\
        0      & 1  & 0        \\\\
        \\sin x & 0  & \\cos x
        \\end{array} \\right)
        \\end{equation}

    ..  math::

        \\begin{equation}
        \\mathbf{ROT2} =
        \\left( \\begin{array}{ccc}
        1  & 0       & 0       \\\\
        0  & \\cos y  & \\sin y  \\\\
        0  & -\\sin y & \\cos y
        \\end{array} \\right)
        \\end{equation}

    Parameters
    ----------
    utc : scalar or array
        A scalar or array [N] of coordinated universal times. The number of elements must match the number, `N`, of eci vectors
        The values must be convertible to `astropy.time.Time` values which includes, strings, datetime, numpy.datetime64,
        astropy.time.Time and floats. Float and strings should represent Coordinated Universal Time and floating point values must
        be expressed as a modified julian date.

    transpose : bool
        If true then generate the transpose of the rotation matrix. Typically True is used when rotating
        from Psuedo Earth Fixed to ITRF. False will map ITRF to Psuedo Fixed Earth.

    Returns
    -------
    np.ndarray
        Returns an array of stacked rotation matrices. It will be of shape [3,3] if parameter `utc` is scalar. It will be of shape
        [N,3,3] if parameter `utc` is an array of `N` values.
    """

    t = utc_to_astropy(utc)
    iers_b = iers.IERS_B.open()
    pxarcsec, pyarcsec, status = iers_b.pm_xy(t, return_status=True)

    px = pxarcsec.value * np.pi / 648000.0  # convert arc seconds to radians
    py = pyarcsec.value * np.pi / 648000.0  # convert arc seconds to radians
    cosxp = np.cos(px)
    sinxp = np.sin(px)
    cosyp = np.cos(py)
    sinyp = np.sin(py)

    s = (
        (t.size, 3, 3)
        if (
            (type(utc) is np.ndarray)
            or (type(utc) is astropy.time.Time)
            or (isinstance(utc, Sequence))
        )
        else (3, 3)
    )
    pm = np.zeros(s)
    if transpose:
        pm[..., 0, 0] = cosxp
        pm[..., 0, 1] = sinxp * sinyp
        pm[..., 0, 2] = sinxp * cosyp
        pm[..., 1, 0] = 0
        pm[..., 1, 1] = cosyp
        pm[..., 1, 2] = -sinyp
        pm[..., 2, 0] = -sinxp
        pm[..., 2, 1] = cosxp * sinyp
        pm[..., 2, 2] = cosxp * cosyp
    else:
        pm[..., 0, 0] = cosxp
        pm[..., 0, 1] = 0.0
        pm[..., 0, 2] = -sinxp
        pm[..., 1, 0] = sinxp * sinyp
        pm[..., 1, 1] = cosyp
        pm[..., 1, 2] = cosxp * sinyp
        pm[..., 2, 0] = sinxp * cosyp
        pm[..., 2, 1] = -sinyp
        pm[..., 2, 2] = cosxp * cosyp
    return pm


# ------------------------------------------------------------------------------
#           eci_to_geo
# ------------------------------------------------------------------------------
def eciteme_to_itrf(
    eciv: np.ndarray,
    utc: np.datetime64 | (datetime.datetime | float),
    do_polar_motion: bool = False,
) -> np.ndarray:
    """
    Converts ECI TEME vectors to ITRF/GEO geocentric vectors. Calculates greenwich mean sidereal time using IAU1982 and
    rotates the vector around the Earth's Z axis to get Psuedo Earth Fixed (PEF). This is usually good enough for most
    application and by default does not account for Polar Motion, which is generally quite small, around a few meters.
    Applications that require higher precision can request that polar motion be included but it does slow the code down.

    This code has been tested against the skyfield package using method skyfield.sgp4lib.TEME_to_ITRF.  Agreement is generally
    very good, typically around the centimeter level or better for Low earth satellites. Much of the diffrence seems to be due
    to slight differences in the implementation details of GMST IAU1982. I think the differences are primarily numerical roundoff differences
    rather than algorithm differences. The polar motion has also been checked against skyfield and gives the same answer.


    Parameters
    ----------
    eciv :  np.ndarray
        A 1-D array of size[3,] or a 2-D array of size [3, N] where `N` is the number of vectors. The first column must
        be size 3 and stores the X,Y,Z components of each **ECI** vector.

    utc : scalar or array
        A scalar or array of coordinated universal times. The number of elements must match the number, `N`, of eci vectors
        The values must be convertible to `astropy.time.Time` values which includes, strings, datetime, numpy.datetime64,
        astropy.time.Time and floats. Float and strings should represent Coordinated Universal Time and floating point values must
        be expressed as a modified julian date.

    do_polar_motion : bool
        If true then apply a polar motion correction

    Returns
    -------
    np.ndarray
        Returns an array of geocentric vectors. The array is the same size, shape and units as parameter `eciv`

    References
    ----------
    See `Revisiting Spacetrack Report #3: Rev 1 <https://www.celestrak.com/publications/AIAA/2006-6753/AIAA-2006-6753-Rev1.pdf>`_, D. Vallado,  August 2006, AIAA 2006-6753 Appendix C.
    Also see  function  `TEME_to_ITRF <https://github.com/skyfielders/python-skyfield/blob/master/skyfield/sgp4lib.py>`_
    in package `skyfield.sgp4lib.py` and the matlab function `teme2ecef <https://www.mathworks.com/matlabcentral/fileexchange/62013-sgp4?focused=6160a950-1145-4fa4-b3e7-1f7afc493bee&tab=function>`_
    on Matlab Central. The latter was written in 2007 by David Vallado, who seems to be the civilian expert on all matters relating to SGP4 and TEME ECI.

    """

    if type(eciv) is not np.ndarray:
        eciv = np.array(eciv)
    s = eciv.shape
    assert (
        s[0] == 3
    ), "eciteme_to_itrf, requires the first dimension of the data to be of size 3"

    theta = gmst(
        utc, model="IAU1982"
    )  # Get the Greenwish Mean Sideral time with IAU1982. Return answer in radians.               This will return a scalar or an array N
    geov = _rotate_about_zaxis(
        eciv, theta
    )  # Rotate around the Earths spin axis to get to Psuedo Earth Fixed (PEF), no polar motion.   Returns an array [3,] or an array[3,N]

    if (
        do_polar_motion
    ):  # For highest precision we want to include polar motion. This normally not require
        s = geov.shape  # Get the shape of the vector, save it for later
        if len(s) == 1:  # if we have 1-d column vector {3,}
            geopef = geov  # then use as is.
        else:  # if we have 2 D array [3,N] then
            geopef = (
                geov.transpose()
            )  # transpose it to [N,3] for upcoming matrix multiplication
            geopef = geopef.reshape(
                (s[1], s[0], 1)
            )  # and reshape it to [N,3,1] for upcoming matrix multiplication
        pm = polar_motion_rotation_matrix(
            utc, transpose=True
        )  # Get the polar motion rotation axis, Array[3,3] or [N,3,3]
        geov = (
            pm @ geopef
        )  # do the matrix multiplication (3,3)*(3,)  or stacked multiplication (N,3,3)*(N,3,1) to give (3,) or (N,3)
        geov = np.squeeze(
            geov
        )  # We now have (3,) or (N,3,1) so remove the trailing 1 dimensions (plus any others)
        geov = geov.transpose().reshape(
            s
        )  # transpose back to (3,N) and reshape back to original form and we are done
    return geov  # return the vector


# ------------------------------------------------------------------------------
#           eci_to_geo
# ------------------------------------------------------------------------------
def itrf_to_eciteme(
    geov: np.ndarray,
    utc: np.datetime64 | (datetime.datetime | float),
    do_polar_motion: bool = False,
) -> np.ndarray:
    """
    Converts ITRF/GEO geocentric vectors to ECI TEME vectors. See the sister function :func:`eciteme_to_itrf` for specific details

    Parameters
    ----------
    geov :  np.ndarray
        A 1-D array of size[3,] or a 2-D array of size [3, N] where `N` is the number of vectors. The first column must
        be size 3 and stores the X,Y,Z components of each **ITRF** vector.

    utc : scalar or array
        A scalar or array of coordinated universal times. The number of elements must match the number, `N`, of itrf vectors
        The values must be convertible to `astropy.time.Time` values which includes, strings, datetime, numpy.datetime64,
        astropy.time.Time and floats. Float and strings should represent Coordinated Universal Time and floating point values must
        be expressed as a modified julian date.

    do_polar_motion : bool
        If true then apply a polar motion correction

    Returns
    -------
    np.ndarray
        Returns an array of geocentric vectors. The array is the same size, shape and units as parameter `geov`.

    """
    if type(geov) is not np.ndarray:
        geov = np.array(geov)
    s = geov.shape
    assert (
        s[0] == 3
    ), "itrf_to_eciteme, requires the first dimension of the data to be of size 3"

    if (
        do_polar_motion
    ):  # For highest precision we want to include polar motion. This normally not require
        s = geov.shape  # Get the shape of the vector, save it for later
        if len(s) == 1:  # if we have 1-d column vector {3,}
            geopef = geov  # then use as is.
        else:  # if we have 2 D array [3,N] then
            geopef = (
                geov.transpose()
            )  # transpose it to [N,3] for upcoming matrix multiplication
            geopef = geopef.reshape(
                (s[1], s[0], 1)
            )  # and reshape it to [N,3,1] for upcoming matrix multiplication
        pm = polar_motion_rotation_matrix(
            utc, transpose=False
        )  # Get the polar motion rotation axis, Array[3,3] or [N,3,3]
        geov = (
            pm @ geopef
        )  # do the matrix multiplication (3,3)*(3,)  or stacked multiplication (N,3,3)*(N,3,1) to give (3,) or (N,3)
        geov = np.squeeze(
            geov
        )  # We now have (3,) or (N,3,1) so remove the trailing 1 dimensions (plus any others)
        geov = geov.transpose().reshape(
            s
        )  # transpose back to (3,N) and reshape back to original form and we are done

    theta = -gmst(utc, model="IAU1982")
    return _rotate_about_zaxis(geov, theta)
