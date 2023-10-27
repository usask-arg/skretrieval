from __future__ import annotations

import logging
import math
import numbers
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from .mjd import mjd_to_datetime64


# ------------------------------------------------------------------------------
#           def datetime64_to_timestamp_new( utc ):
# ------------------------------------------------------------------------------
def datetime64_to_timestamp(utc):
    """
    Converts a scalar or array of numpy.datetime64 to floating point timestamp (seconds since 1970). The conversion includes
    the fraction of seconds.

    Parameters
    ----------
    utc : numpy.datetime64 or Array[numpy.datetime64]
        A time specified with a numpy datetime64 object. Explicit time zones are
        not currently supported. Only single scalar values are supported in this function.

    Returns
    -------
    float
        The time expressed as a timestamp value. Number of seconds since Jan 1, 1970.
    """

    if type(utc) is np.ndarray:
        if utc.dtype.kind == "M":  # if we have an array of datetime64
            tstamp = np.zeros_like(utc, dtype=np.float64)  # create the output array
            with np.nditer(
                [utc, tstamp],
                flags=["refs_ok", "buffered"],  # create the numpy iterator object
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # and convert each elemnet
                    t = (
                        np.datetime64(a, "us")
                        .astype(datetime)
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                    )  # from datetime64 to a timestamp
                    b[...] = t
                tstamp = it.operands[1]

        # Handle an numpy array of MJD values, datetime64, datetime objects or mjd numbers
        elif (
            (utc.dtype.kind == "O") and (len(utc) > 0) and (type(utc[0]) is datetime)
        ):  # otherwiseif we have a numpy array of  datetime objects
            tstamp = np.zeros_like(utc, dtype=np.float64)  # create the output array
            with np.nditer(
                [utc, tstamp],
                flags=["refs_ok", "buffered"],  # create the nupy iterator object
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # and convert each elemnet
                    t = a.replace(
                        tzinfo=timezone.utc
                    ).timestamp()  # from datetime to mjd
                    b[...] = t
                tstamp = it.operands[1]
        elif utc.dtype.kind == "U":  # otherwise if the array is strings
            tstamp = np.zeros_like(
                utc, dtype=np.float64
            )  # then make an array to hold the answer
            with np.nditer(
                [utc, tstamp],
                flags=["refs_ok", "buffered"],
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # For each element in the array
                    t = datetime.fromisoformat(str(a))  # convert the string to datetime
                    b[...] = t.replace(
                        tzinfo=timezone.utc
                    ).timestamp()  # and from datetime to mjd
                tstamp = it.operands[1]
        else:  # otherwise we have an array of something else, I assume its array of numbers
            tstamp = np.array(
                utc, dtype=np.float64
            )  # so try and convert the array of numbers to floating point MJD.

    elif isinstance(utc, str):  # if we have a single string
        tstamp = datetime.fromisoformat(
            utc
        ).timestamp()  # then convert the string to datetime

    elif isinstance(utc, Sequence):  # if we have a list or tuple but not numpy array
        newutc = np.array(utc)  # then convert the array to numpy array
        tstamp = datetime64_to_timestamp(newutc)  # and convert the numpy array to mjd

    else:  # Its not a numpy array, its not a sequence, so assume its a scalar
        if type(utc) is datetime:  # if its a datetime object
            tstamp = utc.replace(
                tzinfo=timezone.utc
            ).timestamp()  # to mjd as a floating point
        elif type(utc) is np.datetime64:  # if we have a single numpy.datetime64
            tstamp = (
                np.datetime64(utc, "us")
                .astype(datetime)
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )  # then convert it
        else:
            logging.warning(
                "utc_to_timestamp, unsupported data type. Cannot robustly convert to timestamp. I will try and convert it to a float",
                extra={"utc": utc},
            )
            tstamp = float(utc)  # then assume its already an MJD
    return tstamp


# ------------------------------------------------------------------------------
#           datetime64_to_datetime
# ------------------------------------------------------------------------------
def datetime64_to_datetime(usertime: np.datetime64) -> datetime:
    """
    Converts a single numpy.datetime64 to a datetime.datetime

    Parameters
    ----------
    usertime : numpy.datetime64
        A time specified with a numpy datetime64 object. Explicit time zones are
        not currently supported. Only single scalar values are supported in this function.

    Returns
    -------
    datetime.datetime
        The time expressed as a regular python datetime.datetime.
    """
    return np.datetime64(usertime, "us").astype(datetime)


# ------------------------------------------------------------------------------
#           datetime_to_datetime64
# ------------------------------------------------------------------------------
def datetime_to_datetime64(usertime: np.datetime64) -> np.datetime64:
    """
    Converts a single datetime to a np.datetime64

    Parameters
    ----------
    usertime : datetime.datetime
        A time specified with a python datetime object. Explicit time zones are
        not currently supported. Only single scalar values are supported in this function.

    Returns
    -------
    datetime.datetime
        The time expressed as a regular python datetime.datetime.
    """
    return np.datetime64(usertime, "us")


# ------------------------------------------------------------------------------
#           ut_to_datetime64
# ------------------------------------------------------------------------------
def ut_to_datetime64(utc):
    """
    A convenience function that converts various arrays or scalar representations of UT to datetime64
    Scalar input values will return as a scalar datyetime64 while array, list or tuple input values are all returned
    as numpy arrays of utc in datetime64.

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
    scalar or numpy.ndarray of np.datetim64
        The time is returned as a scalar if the input was a scalar or as a numpy array for input sequences and arrays
    """
    d64 = None
    if type(utc) is np.ndarray:
        # Handle an numpy array of MJD values, datetime64, datetime objects or mjd numbers
        if utc.dtype.kind == "M":  # if we have a numpy array of datetime64 object
            d64 = utc  # then do nothing
        elif (
            (utc.dtype.kind == "O") and (len(utc) > 0) and (type(utc[0]) is datetime)
        ):  # otherwiseif we have a numpy array of  datetime objects
            d64 = np.zeros_like(utc, dtype="datetime64[us]")  # create the output array
            with np.nditer(
                [utc, d64], flags=["refs_ok"], op_flags=[["readwrite"], ["writeonly"]]
            ) as it:
                for a, b in it:  # and convert each elemnet
                    t = datetime_to_datetime64(
                        a.tolist()
                    )  # The object comes in as a zero shape object of datetime. But the toplist converts it back to a datetime
                    b[...] = t
                d64 = it.operands[1]
        elif utc.dtype.kind == "U":  # otherwise if the array is strings
            d64 = np.zeros_like(
                utc, dtype="datetime64[us]"
            )  # then make an array to hold the answer
            with np.nditer(
                [utc, d64],
                flags=["refs_ok", "buffered"],
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # For each element in the array
                    b[...] = np.datetime64(str(a), "us")  # and from datetime to mjd
                d64 = it.operands[1]
        else:  # otherwise we have an array of something else, I assume its array of numbers
            d64 = np.zeros_like(
                utc, dtype="datetime64[us]"
            )  # then make an array to hold the answer
            with np.nditer(
                [utc, d64],
                flags=["refs_ok", "buffered"],
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # For each element in the array
                    b[...] = mjd_to_datetime64(
                        float(a)
                    )  # atry and convert the MJD to datetime64
                d64 = it.operands[1]
    elif isinstance(utc, str):  # if we have a single string
        d64 = np.datetime64(utc, "us")  # then convert the string to datetime

    elif isinstance(utc, Sequence):  # if we have a list or tuple but not numpy array
        newutc = np.array(utc)  # then convert the array to numpy array
        d64 = ut_to_datetime64(newutc)  # and convert the numpy array to mjd
    else:  # Its not a numpy array, its not a sequence, so assume its a scalar
        if type(utc) is datetime:  # if its a datetime object
            d64 = datetime_to_datetime64(utc)  # to mjd as a floating point
        elif type(utc) is np.datetime64:  # if we have a single numpy.datetime64
            d64 = utc  # then convert it
        elif isinstance(utc, numbers.Number):  # if we have a number
            d64 = mjd_to_datetime64(float(utc))  # then assume its already an MJD
        else:  # otherwise
            d64 = math.nan
            msg = "unsupported data type {}. Cannot yet convert this to np.datetime64".format(
                str(type(utc))
            )
            raise ValueError(msg)
    return d64


# ------------------------------------------------------------------------------
#           ut_to_datetime64
# ------------------------------------------------------------------------------
def ut_to_datetime(
    utc: np.ndarray | (Sequence | (float | (str | (np.datetime64 | datetime)))),
) -> np.ndarray | (datetime | None):
    """
    A convenience function that converts various arrays or scalar representations of UT to datetime
    Scalar input values will return as a scalar datetime while array, list or tuple input values are all returned
    as numpy arrays of datetime objects

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
    scalar or numpy.ndarray of datetime
        The time is returned as a scalar if the input was a scalar or as a numpy array for input sequences and arrays
    """
    d64 = None
    if type(utc) is np.ndarray:
        # Handle an numpy array of MJD values, datetime64, datetime objects or mjd numbers
        if utc.dtype.kind == "O":  # we have a numpy array of  datetime objects
            d64 = utc  # then do nothing
            if (len(utc) > 0) and (type(utc[0]) is not datetime):
                msg = "ut_to_datetime does support conversion of arrays of python objects other than datetime"
                raise ValueError(msg)
        elif (
            utc.dtype.kind == "M"
        ):  # otherwiseif we have a numpy array of  datetime64 objects
            d64 = np.zeros_like(utc, dtype="O")  # create the output array
            with np.nditer(
                [utc, d64], flags=["refs_ok"], op_flags=[["readwrite"], ["writeonly"]]
            ) as it:
                for a, b in it:  # and convert each elemnet
                    t = datetime64_to_datetime(
                        a.tolist()
                    )  # The object comes in as a zero shape object of datetime. But the toplist converts it back to a datetime
                    b[...] = t
                d64 = it.operands[1]
        elif utc.dtype.kind == "U":  # otherwise if the array is strings
            d64 = np.zeros_like(utc, dtype="O")  # then make an array to hold the answer
            with np.nditer(
                [utc, d64],
                flags=["refs_ok", "buffered"],
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # For each element in the array
                    b[...] = datetime.fromisoformat(str(a))  # and from datetime to mjd
                d64 = it.operands[1]
        else:  # otherwise we have an array of something else, I assume its array of numbers
            d64 = np.zeros_like(utc, dtype="O")  # then make an array to hold the answer
            with np.nditer(
                [utc, d64],
                flags=["refs_ok", "buffered"],
                op_flags=[["readonly"], ["writeonly"]],
            ) as it:
                for a, b in it:  # For each element in the array
                    t = mjd_to_datetime64(float(a))  # Convert the mjd to datetime 64
                    b[...] = datetime64_to_datetime(t)  # and then convert to  datetime
                d64 = it.operands[1]
    elif isinstance(utc, str):  # if we have a single string
        d64 = datetime.fromisoformat(utc)  # then convert the string to datetime
    elif isinstance(utc, Sequence):  # if we have a list or tuple but not numpy array
        newutc = np.array(utc)  # then convert the array to numpy array
        d64 = ut_to_datetime(newutc)  # and convert the numpy array to mjd

    else:  # Its not a numpy array, its not a sequence, so assume its a scalar
        if type(utc) is datetime:  # if its a datetime object
            d64 = utc  # to mjd as a floating point
        elif type(utc) is np.datetime64:  # if we have a single numpy.datetime64
            d64 = datetime64_to_datetime(utc)  # then convert it
        elif isinstance(utc, numbers.Number):  # if we have a number
            t = mjd_to_datetime64(float(utc))  # then assume its already an MJD
            d64 = datetime64_to_datetime(t)
        else:  # otherwise
            msg = "unsupported data type {}. Cannot yet convert this to np.datetime64".format(
                str(type(utc))
            )
            raise ValueError(msg)
    return d64
