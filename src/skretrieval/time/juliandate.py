from __future__ import annotations

import math
import numbers
from datetime import datetime
from typing import Any, Sequence

import jdcal
import numpy as np

from .datetime64 import datetime64_to_datetime


# ------------------------------------------------------------------------------
#           datetime64_to_jd
# ------------------------------------------------------------------------------
def datetime_to_jd(t: datetime) -> tuple[float, float]:
    """
    Converts a scalar datetime.datetime object to julian date which is returned as
    a two element floating point tuple. No attempt has been made to manage timezones
    inside the datetime object. We recommend that you don't use objects with time-zones in this code.

    Parameters
    ----------
    ut1 : datetime.datetime
        The universal time to be converted to julian date. The Universal time is assumed to follow the
        Gregorian calendar

    Returns
    -------
    Tuple( float, float)
        Returns a two element floating point tuple. The first element contains the high order (julian days) part of
        the julian date. The second element contains the low order fraction of a day.
    """

    jd1, jd2 = jdcal.gcal2jd(t.year, t.month, t.day)
    jd1 = jd1 + jd2 - 0.5
    frac = (
        t.hour * 3600.0 + t.minute * 60.0 + t.second + t.microsecond / 1.0e6
    ) / 86400.0 + 0.5
    if frac >= 1.0:
        jd1 += 1.0
        frac -= 1.0
    return jd1, frac


# ------------------------------------------------------------------------------
#           datetime64_to_jd
# ------------------------------------------------------------------------------
def datetime64_to_jd(ut1: np.datetime64) -> tuple[float, float]:
    """
    Converts a scalar numpy datetime64 object to julian date which is returned as
    a two element floating point tuple. No attempt has been made to manage timezones
    inside the datetime64 object. We recommend you don't use objects with time-zones

    Parameters
    ----------
    ut1 : numpy.datetime64
        The universal time (strictly UT1) to be converted to julian date.

    Returns
    -------
    Tuple( float, float)
        Returns a two element floating point tuple. The first element contains the high order (julian days) part of
        the julian date. The second element contains the low order fraction of a day.
    """
    t = datetime64_to_datetime(ut1)
    return datetime_to_jd(t)


# ------------------------------------------------------------------------------
#           isoformat_to_jd
# ------------------------------------------------------------------------------
def isoformat_to_jd(utc: str) -> tuple[float, float]:
    """
    Converts a string representing a UT instant encoded in isoformat to a julian date which is returned as
    a two element floating point tuple. No attempt has been made to manage timezones
    inside the datetime object. We recommend that you don't use objects with time-zones in this code.

    Parameters
    ----------
    ut1 : str
        The universal time (strictly UT1) to be converted to julian date. The string is encoded is isoformat,
        eg '2009-11-15T15:36:49.000123'

    Returns
    -------
    Tuple( float, float)
        Returns a two element floating point tuple. The first element contains the high order (julian days) part of
        the julian date. The second element contains the low order fraction of a day.
    """

    t = datetime.fromisoformat(str(utc))  # convert the string to datetime
    return datetime_to_jd(t)  # and from datetime to mjd


# ------------------------------------------------------------------------------
#           number_to_jd
# ------------------------------------------------------------------------------
def number_to_jd(utc: float) -> tuple[float, float]:
    """
    Converts a number to a julian date which is returned as a two element floating point tuple. Numbers less than
    500,000 are assumed to represent modified julian date. Numbers greater than 500,000 are assumed to be julian dates.
    The calculation is useful for converting floating point values to the two element tuples used by the novas
    and sofa software.

    Parameters
    ----------
    ut1 : float
        The universal time to be converted to julian date.

    Returns
    -------
    Tuple( float, float)
        Returns a two element floating point tuple. The first element contains the high order (julian days) part of
        the julian date. The second element contains the low order fraction of a day.
    """

    x = float(utc)
    jd1 = float(math.floor(x))
    jd2 = x - jd1
    if jd1 < 500000.0:
        jd1 += 2400000.0
        jd2 += 0.5
        if jd2 >= 1.0:
            jd1 += 1.0
            jd2 -= 1.0
    return jd1, jd2  # and from datetime to mjd


# ------------------------------------------------------------------------------
#           _iterate_over_utc_array
# ------------------------------------------------------------------------------
def _iterate_over_utc_array(utc, elementfunc):
    s = utc.shape  # get the shape
    jd1 = np.zeros(
        s, dtype=np.float64
    )  # create the output array for the high order julian date components
    jd2 = np.zeros(
        s, dtype=np.float64
    )  # create the output array for the low order julian date component
    with np.nditer(
        [utc, jd1, jd2],
        flags=["refs_ok"],
        op_flags=[["readonly"], ["writeonly"], ["writeonly"]],
    ) as it:  # create the numpy iterator object
        for a, b, c in it:  # and convert each elemnet
            j1, j2 = elementfunc(
                a.item()
            )  # from the numpy array element datetime to jd
            b[...] = j1
            c[...] = j2
        jd1 = it.operands[1]
        jd2 = it.operands[2]
    return jd1, jd2


# ------------------------------------------------------------------------------
#           utc_to_jd
# ------------------------------------------------------------------------------
def ut_to_jd(ut: np.ndarray | (list[Any] | Any)):
    """
    A convenience function that converts various arrays or scalar representations of UT to julian date. Each Julian date is returned
    as two floating point components, `jd1` and `jd2`. If the input is an array of value sthen jd1 and jd2 will be returned as
    floating point arrays with the same shape and size as the input array. If the inputs are scalar values then `jd1` and `jd2` will
    be returned as scalar floating point values.

    Parameters
    ----------
    utc : scalar, list, tuple or array
        The input time which represents a universal time (strictly UT1). It can be represented by
        (i) a string in a supported python datetime.isoformat
        (ii) a number which is assumed to represent a julian date if it is greater than 500,000 or a modified julian date if it less than 500,000,
        (iii) a datetime.datetime object representing UT
        (iv) a numpy.datetime64 object representing UT

        The *ut* object can be a scalar, list, tuple or numpy array. The same representation format must be used for
        all objects in the arrays and sequences.

    Returns
    -------
    scalar or numpy.ndarray of float
        The mjd is returned as a scalar if the input was a scalar or as a numpy array for input sequences and arrays
    """

    if (
        type(ut) is np.ndarray
    ):  # Handle an numpy array of values, datetime64, datetime objects or mjd numbers
        if ut.dtype.kind == "M":  # if we have a numpy array of datetime64 object
            jd1, jd2 = _iterate_over_utc_array(
                ut, datetime64_to_jd
            )  # Then convert all the elements to julian date
        elif (
            (ut.dtype.kind == "O") and (len(ut) > 0) and (type(ut[0]) is datetime)
        ):  # otherwise if we have a numpy array of  datetime objects
            jd1, jd2 = _iterate_over_utc_array(
                ut, datetime_to_jd
            )  # create the output array
        elif ut.dtype.kind == "U":
            jd1, jd2 = _iterate_over_utc_array(
                ut, isoformat_to_jd
            )  # otherwise if the array is strings
        elif (ut.dtype.kind == "f") or (ut.dtype.kind == "i"):
            jd1, jd2 = _iterate_over_utc_array(
                ut, number_to_jd
            )  # otherwise if the array is strings
    elif isinstance(ut, str):  # if we have a single string
        jd1, jd2 = isoformat_to_jd(ut)  # then convert the string to datetime
    elif isinstance(ut, Sequence):  # if we have a list or tuple but not numpy array
        newutc = np.array(ut)  # then convert the array to numpy array
        jd1, jd2 = ut_to_jd(newutc)  # and convert the numpy array to mjd
    else:  # Its not a numpy array, its not a sequence, so assume its a scalar
        if type(ut) is datetime:  # if its a datetime object
            jd1, jd2 = datetime_to_jd(ut)  # to mjd as a floating point
        elif type(ut) is np.datetime64:  # if we have a single numpy.datetime64
            jd1, jd2 = datetime64_to_jd(ut)  # then convert it
        elif isinstance(ut, numbers.Number):  # if we have a number
            jd1, jd2 = number_to_jd(ut)
        else:  # otherwise
            jd1 = math.nan
            jd2 = math.nan
            msg = "utc_to_jd, unsupported data type {}. Cannot convert to julian date".format(
                str(type(ut))
            )
            raise Exception(msg)
    return jd1, jd2
