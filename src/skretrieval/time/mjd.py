"""
The `mjd` module provides time conversion routines to and from Modified Julian Date. It provides convenient
functions which convert scalars, arrays or sequences of various time representations (strings, datetimes and numpy datetime64 into floats) into
corresponding scalars or arrays of modified julian date. The code uses the time functions native to python and makes
no attempt to implements IERS (Earth Rotation) details such as leap seconds etc. The following functions are provided,

 - `utc_to_mjd`. Converts scalars, arrays and sequences of various time representations to floating point MJD
 - `mjd_to_utc`, Converts scalars, arrays and sequences of floating point MJD to numpy.datetime64
 - `mjd_to_datetime64`, converts scalar mjd to numpy.datetime64
 - `mjd_to_datetime`, converts scalar mjd to datetime.datetime
 - `datetime64_to_datetime`, converts numpy.datetime64 to datetime.datetime
"""
from __future__ import annotations

import math
import numbers
import re
import string
import time
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
from astropy.time import Time

_g_mjd0str = "1858-11-17 00:00:00.000000"
_t0_64 = np.datetime64(_g_mjd0str)
_t0 = datetime.strptime(_g_mjd0str, "%Y-%m-%d %H:%M:%S.%f")


# ------------------------------------------------------------------------------
#           datetime64_to_mjd
# ------------------------------------------------------------------------------
def datetime64_to_mjd(utc: np.datetime64) -> float:
    """
    Converts a numpy datetime64 to a floating point MJD. Only scalar values are supported
    by this function.

    Parameters
    ----------
    utc : numpy.datetime64
         A time specified with a numpy datetime64 object. Explicit time zones are
         not currently supported. Only single scalar values are supported in this function.

    Returns
    -------
    float
        The corresponding modified julian date.

    """
    return (
        (utc - _t0_64).astype("timedelta64[us]")
        / np.timedelta64(1, "us")
        / 86400000000.0
    )


# ------------------------------------------------------------------------------
#           mjd_to_datetime64
# ------------------------------------------------------------------------------
def mjd_to_datetime64(mjd: float) -> np.datetime64:
    """
    Converts an modified julian date to a numpy.datetime64. This function only works
    on scalar values of mjd.

    Parameters
    ----------
    mjd : float
        The modified julian date.

    Returns
    -------
    numpy.datetime64
        The numpy.datetime64 corresponding to the modified julian date
    """
    dt = np.timedelta64(int(mjd * 86400000000.0 + 0.5), "us")
    return (_t0_64 + dt).astype("datetime64[us]")


# ------------------------------------------------------------------------------
#           mjd_to_datetime64
# ------------------------------------------------------------------------------
def mjd_to_datetime(mjd: float) -> datetime:
    """
    Converts a modified julian date to a datetime.datetime object. This function only works
    on scalar values of mjd. Time zones are not set.

    Parameters
    ----------
    mjd : float
        The modified julian date.

    Returns
    -------
    datetime.datetime
        The datetime.datetime corresponding to the modified julian date. No time zone is set
    """

    dt = timedelta(days=mjd)
    return _t0 + dt


# ------------------------------------------------------------------------------
#           datetime_to_mjd
# ------------------------------------------------------------------------------
def datetime_to_mjd(t: datetime) -> float:
    """
    Converts a python datetime object to a floating point MJD. Only scalar values are supported
    by this function.

    Parameters
    ----------
    utc : datetime.datetime
         A time specified with a regular python datetime.datetime object. Explicit time zones are
         not currently supported. Only single scalar values are supported in this function.

    Returns
    -------
    float
        The corresponding modified julian date.

    """
    dt = t - _t0  # from datetime
    return (
        float(dt.days)
        + (float(dt.seconds) + float(dt.microseconds) * 1.0e-06) / 86400.0
    )  # to mjd as a floating point


# ------------------------------------------------------------------------------
#           mjd_to_utc
# ------------------------------------------------------------------------------
def mjd_to_ut(mjd):
    """
    A convenience function that converts scalars or arrays of floating point modified julian dates to universal time representations.
    The universal time is always returned as a scalar or array of numpy.datetime64 object(s).

    Parameters
    ----------
    mjd : scalar, array, or sequence of float
        The input can be a scalar, numpy array or regular python list/tuple of floats storing MJD values.

    Returns
    -------
    numpy.datetime64 or numpy.ndarray< numpy.datetime64 >
        Returns either a scalar or array of numpy.datetime64 that matches the input object
    """

    if (type(mjd) is np.ndarray) or (isinstance(mjd, Sequence)):
        if type(mjd) is np.ndarray:
            utc = np.zeros_like(mjd, dtype="datetime64[us]")
            with np.nditer(
                [mjd, utc],
                flags=["buffered"],  # create the nupy iterator object
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # and convert each elemnet
                    b[...] = mjd_to_datetime64(a)  # from datetime to mjd
                utc = it.operands[1]
        else:
            utc = np.zeros([len(mjd)])
            for i in range(len(mjd)):
                utc[i] = mjd_to_datetime64(mjd[i])
    else:
        utc = mjd_to_datetime64(mjd)
    return utc


# ------------------------------------------------------------------------------
#           datetime64_to_mjd
# ------------------------------------------------------------------------------
def ut_to_mjd(utc):
    """
    A convenience function that converts various arrays or scalar representations of UT to modified julian date.
    Scalar input values will return as a scalar float modified julian date while array, list or tuple input values are all returned
    as numpy arrays of mjd in float64.

    Parameters
    ----------

    utc : scalar, list, tuple or array
        The input time which represents a coordinated universal time. It can be represented by
        (i) a string in a supported python datetime.isoformat
        (ii) a number. The number is assumed to represent a modified julian date.
        (iii) a datetime.datetime object.
        (iv) a numpy.datetime64 object.
        The *utc* object can be a scalar, list, tuple or numpy array. The same representation format must be used for
        all objects in the arrays and sequences.

    Returns
    -------
    scalar or numpy.ndarray of float
        The mjd is returned as a scalar if the input was a scalar or as a numpy array for input sequences and arrays
    """

    if type(utc) is np.ndarray:
        # Handle an numpy array of MJD values, datetime64, datetime objects or mjd numbers
        if utc.dtype.kind == "M":  # if we have a numpy array of datetime64 object
            mjd = datetime64_to_mjd(utc)  # then convert it to floating point MJD
        elif (
            (utc.dtype.kind == "O") and (len(utc) > 0) and (type(utc[0]) is datetime)
        ):  # otherwiseif we have a numpy array of  datetime objects
            mjd = np.zeros_like(utc, dtype=np.float64)  # create the output array
            with np.nditer(
                [utc, mjd],
                flags=["refs_ok", "buffered"],  # create the nupy iterator object
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # and convert each elemnet
                    t = datetime_to_mjd(a)  # from datetime to mjd
                    b[...] = t
                mjd = it.operands[1]
        elif utc.dtype.kind == "U":  # otherwise if the array is strings
            mjd = np.zeros_like(
                utc, dtype=np.float64
            )  # then make an array to hold the answer
            with np.nditer(
                [utc, mjd],
                flags=["refs_ok", "buffered"],
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # For each element in the array
                    t = datetime.fromisoformat(str(a))  # convert the string to datetime
                    b[...] = datetime_to_mjd(t)  # and from datetime to mjd
                mjd = it.operands[1]
        else:  # otherwise we have an array of something else, I assume its array of numbers
            mjd = np.array(
                utc, dtype=np.float64
            )  # so try and convert the array of numbers to floating point MJD.

    elif isinstance(utc, str):  # if we have a single string
        t = datetime.fromisoformat(utc)  # then convert the string to datetime
        mjd = datetime_to_mjd(t)  # and from datetime to mjd

    elif isinstance(utc, Sequence):  # if we have a list or tuple but not numpy array
        newutc = np.array(utc)  # then convert the array to numpy array
        mjd = ut_to_mjd(newutc)  # and convert the numpy array to mjd
    else:  # Its not a numpy array, its not a sequence, so assume its a scalar
        if type(utc) is datetime:  # if its a datetime object
            mjd = datetime_to_mjd(utc)  # to mjd as a floating point
        elif type(utc) is np.datetime64:  # if we have a single numpy.datetime64
            mjd = float(datetime64_to_mjd(utc))  # then convert it
        elif isinstance(utc, numbers.Number):  # if we have a number
            mjd = float(utc)  # then assume its already an MJD
        else:  # otherwise
            mjd = math.nan
            msg = "utc_to_mjd, unsupported data type {}. Cannot convert to mjd".format(
                str(type(utc))
            )
            raise Exception(msg)
    return mjd


# ------------------------------------------------------------------------------
#           utc_to_astropy
# ------------------------------------------------------------------------------
def utc_to_astropy(utc):
    """
    A convenience function that converts list,  arrays or scalar representations of UTC to an astropy Time object.
    The function is a shallow wrapper for strings, datetime and datetime64 arrays as numpy arrays of mjd in float64.

    Parameters
    ----------

    utc : scalar, list, tuple or array
        The input time which represents a coordinated universal time. It can be represented by
        (i) a string in a supported python datetime.isoformat
        (ii) a number. The number is assumed to represent a modified julian date.
        (iii) a datetime.datetime object.
        (iv) a numpy.datetime64 object.
        The *utc* object can be a scalar, list, tuple or numpy array. The same representation format must be used for
        all objects in the arrays and sequences.

    Returns
    -------
    scalar or numpy.ndarray of float
        The mjd is returned as a scalar if the input was a scalar or as a numpy array for input sequences and arrays
    """

    formattype = None
    if (type(utc) is np.ndarray) and utc.dtype.kind in ["i", "u", "f"]:
        formattype = "mjd"
    elif isinstance(utc, Sequence):
        if isinstance(utc[0], numbers.Number):
            formattype = "mjd"
    elif isinstance(utc, numbers.Number):
        formattype = "mjd"

    return (
        Time(utc, format=formattype, scale="utc")
        if formattype is not None
        else Time(utc, scale="utc")
    )


# ---------------------------------------------------------------------------
# This is an old class from the early OSIRIS days in 2001. Its needs to be
# updated before it is used
# ----------------------------------------------------------------------------
class MJD:
    def __init__(self, mjd=0.0):
        self.m_mjd = mjd

    def from_date(self, year, month, day):
        self.from_utc(year, month, day, 0, 0, 0)

    def from_utc(self, year, month, day, hour, minutes, seconds):
        tupl = [
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minutes),
            int(seconds),
            0,
            0,
            -1,
        ]
        secs = time.mktime(tupl) - time.timezone
        self.FromUnixSeconds(secs)

    def from_str(self, utcstr):
        pattern = f"[{string.punctuation}{string.whitespace}]+"  # Get all of the puctuation and whitespace charcaters
        s = re.split(
            pattern, utcstr
        )  # split the string with white space and punctuation
        t = [0, 1, 1, 0, 0, 0]  # get default values for fields not suplied by user
        n = min(len(s), 6)  # make sure we dont use more than 6 fields
        for i in range(n):
            t[i] = int(s[i])  # now fill in the values supplied by user
        self.from_utc(t[0], t[1], t[2], t[3], t[4], t[5])  # convert those times to UTC

    def as_unix_seconds(self):
        secs = (self.m_mjd - 40587.0) * 86400.0
        if secs < 0:
            secs = 0
        return secs

    def AsTimeTuple(self):
        return time.gmtime(self.as_unix_seconds())

    def FromUnixSeconds(self, secs):
        self.m_mjd = secs / 86400.0 + 40587.0

    def SetTo(self, mjd):
        self.m_mjd = mjd

    def FromSystem(self):
        self.FromUnixSeconds(time.time())

    def MJD(self):
        return self.m_mjd

    def as_utc_str(self, doticks=0):
        secs = (
            self.as_unix_seconds() + 0.0005
        )  # Round the seconds to the nearest millisecond
        tupl = time.gmtime(secs)  # Now get the time using the system call
        utcstr = time.strftime("%Y-%m-%d %H:%M:%S", tupl)  # Encode as a string
        if doticks:  # add the milliseconds if required
            ticks = int((secs - int(secs)) * 1000.0)
            utcstr = "%s.%03d" % (utcstr, ticks)
        return utcstr

    def AsDateStr(self):
        secs = (
            self.as_unix_seconds() + 0.0005
        )  # Round the seconds to the nearest millisecond
        tupl = time.gmtime(secs)  # Now get the time using the system call
        return time.strftime("%Y-%m-%d", tupl)  # Encode as a string

    def AsLocalTimeStr(self, formatstr="%H:%M"):
        secs = self.as_unix_seconds()
        tupl = time.localtime(secs)
        return time.strftime(formatstr, tupl)

    def __add__(self, other):
        self.m_mjd = self.m_mjd + float(other)
        return self

    def __sub__(self, other):
        self.mjd = self.m_mjd - float(other)
        return self

    def __mul__(self, other):
        self.m_mjd = self.m_mjd * float(other)
        return self

    def __div__(self, other):
        self.m_mjd = self.m_mjd / float(other)
        return self

    def __neg__(self):
        self.m_mjd = -self.m_mjd
        return self

    def __pos__(self):
        return self

    def __abs__(self):
        self.m_mjd = abs(self.m_mjd)
        return self

    def __int__(self):
        return int(self.m_mjd)

    def __long__(self):
        return int(self.m_mjd)

    def __float__(self):
        return float(self.m_mjd)

    def __str__(self):
        return self.as_utc_str(1)
