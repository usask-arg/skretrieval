from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr
from scipy.linalg import block_diag
from simpleeval import simple_eval

from skretrieval.core.radianceformat import RadianceGridded


def _resolve_value(expr, variables):
    if isinstance(expr, str):
        expr = expr.replace("$", "")
        return simple_eval(expr, names=variables)
    return expr


@dataclass
class Measurement:
    """
    A dataclass representing the core objects of a measurement vector. This is an internal object
    that is passed around when doing measurement vector transformations
    """

    y: np.array
    K: np.array
    Sy: np.array


class MeasurementVector:
    def __init__(self, fn: Callable, apply_to_filter="*"):
        """
        A class that represents a measurement vector. This is a callable object that can be used to
        transform L1 data to a measurement vector.

        Parameters
        ----------
        fn : Callable
            Function which takes in L1 data and returns a Measurement object
        apply_to_filter : str, optional
            Only L1 data matching the apply_to_filter will be affected by this measurement vector, by default "*"
        """
        self._fn = fn
        self._filter = apply_to_filter
        self._enabled = True

    @property
    def fn(self):
        return self._fn

    @property
    def filter(self):
        return self._filter

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def apply(
        self, l1_data: dict[RadianceGridded], ctxt: dict | None = None
    ) -> Measurement:
        """
        Applies the function to the l1 data, returning back a Measurement object

        Parameters
        ----------
        l1_data : dict[RadianceGridded]

        Returns
        -------
        Measurement
        """
        if not self._enabled:
            return None
        apply_vals = {
            k: d for k, d in l1_data.items() if fnmatch.fnmatch(k, self._filter)
        }
        if len(apply_vals) > 0:
            local_ctxt = ctxt if ctxt is not None else {}
            return self._fn(apply_vals, ctxt=local_ctxt, filter=self._filter)
        return None

    def required_sample_wavelengths(
        self, obs_samples: dict[np.array]
    ) -> dict[np.array]:
        """
        Determines which sample wavelengths are required for this measurement vector

        Default is to just return back all of the observation wavelengths

        Parameters
        ----------
        obs_samples : dict[np.array]

        Returns
        -------
        dict[np.array]
        """
        return obs_samples


def pre_process(l1: dict[RadianceGridded], n: int = 1) -> dict[RadianceGridded]:
    """
    Called before the measurement vector is applied. This function will ensure that the L1 data
    always has the necessary fields for the measurement vector to work.

    Parameters
    ----------
    l1 : dict[RadianceGridded]
    n: int, optional
        Number of elements in the state vector, used to create the dummy Jacobian. Default is 1
        which can be used if the number of elements in the state vector is not important.

    Returns
    -------
    dict[RadianceGridded]
    """
    # Copy and modify the data to always include wf and noise values
    new_l1 = {}

    for key, val in l1.items():
        new_val = val.data.copy(deep=True)

        if "wf" not in new_val:
            new_val["wf"] = xr.zeros_like(
                new_val["radiance"].expand_dims({"x": n}, axis=-1)
            )

        if "radiance_noise" not in new_val:
            new_val["radiance_noise"] = new_val["radiance"] * 1

        new_l1[key] = RadianceGridded(new_val)
    return new_l1


def concat(measurements: list[Measurement]) -> Measurement:
    """
    Concatenates a list of measurements into a single measurement

    Parameters
    ----------
    measurements : list[Measurement]

    Returns
    -------
    Measurement
    """
    if len(measurements) == 0:
        return None
    return Measurement(
        y=np.concatenate([m.y for m in measurements]),
        K=np.vstack([m.K for m in measurements]),
        Sy=block_diag(*[m.Sy for m in measurements]),
    )


def post_process(measurement: Measurement) -> dict:
    """
    Called after the measurement vector is applied. This function will convert the measurement
    object back into a dictionary for the retrieval to use.

    Parameters
    ----------
    measurement : Measurement

    Returns
    -------
    dict
    """
    # At this stage we have to remove the jacobian if it was a dummy one in the beginning
    res = {"y": measurement.y, "jacobian": measurement.K, "y_error": measurement.Sy}

    if measurement.K.shape[-1] == 0:
        del res["jacobian"]

    return res


def select(l1: dict[RadianceGridded], filter: str = "*", **kwargs) -> Measurement:
    """
    Selects the L1 data that matches the filter and applies the selector stored in kwargs
    to the underlying xarray datasets


    Parameters
    ----------
    l1 : dict[RadianceGridded]
    filter : str, optional
         by default "*"

    Returns
    -------
    Measurement
    """
    measurements = []

    for key, val in l1.items():
        if fnmatch.fnmatch(key, filter):
            selected = val.data.sel(**kwargs)

            measurements.append(
                Measurement(
                    y=selected["radiance"].to_numpy().flatten(),
                    K=selected["wf"].to_numpy().reshape((-1, len(selected["x"]))),
                    Sy=np.diag(selected["radiance_noise"].to_numpy().flatten() ** 2),
                )
            )

    return concat(measurements)


def nearest_selector(l1: dict[RadianceGridded], filter: str = "*", **kwargs) -> dict:
    """
    A special selector that will select the nearest value to the kwargs in the L1 data.
    Returns back another dictionary with the same keys as the input dictionary but with the
    data modified to only contain the nearest values to the kwargs

    Parameters
    ----------
    l1 : dict[RadianceGridded]
    filter : str, optional
        , by default "*"

    Returns
    -------
    dict
    """
    res = {}
    for key, val in l1.items():
        if fnmatch.fnmatch(key, filter):
            res[key] = RadianceGridded(
                val.data.sel(**kwargs, method="nearest").assign_coords(**kwargs)
            )

    return res


def log(measurement: Measurement) -> Measurement:
    """
    Log transform the measurement

    Parameters
    ----------
    measurement : Measurement

    Returns
    -------
    Measurement
    """
    return Measurement(
        y=np.log(measurement.y),
        K=measurement.K / measurement.y[:, np.newaxis],
        Sy=measurement.Sy / np.outer(measurement.y, measurement.y),
    )


def mean(measurement: Measurement) -> Measurement:
    """
    Take the mean of the measurement

    Parameters
    ----------
    measurement : Measurement

    Returns
    -------
    Measurement
    """
    return Measurement(
        y=np.mean(measurement.y),
        K=np.mean(measurement.K, axis=0),
        Sy=np.mean(np.diag(measurement.Sy)),
    )


def multiply(measurement: Measurement, factor: float) -> Measurement:
    """
    Multiply the measurement by a factor

    Parameters
    ----------
    measurement : Measurement
    factor : float

    Returns
    -------
    Measurement
    """
    return Measurement(
        y=measurement.y * factor,
        K=measurement.K * factor,
        Sy=measurement.Sy * factor**2,
    )


def subtract(measurement: Measurement, other: Measurement) -> Measurement:
    """
    Subtract one measurement from another

    Parameters
    ----------
    measurement : Measurement
    other : Measurement

    Returns
    -------
    Measurement
    """
    return Measurement(
        y=measurement.y - other.y,
        K=measurement.K - other.K,
        Sy=measurement.Sy + other.Sy,
    )


def add(measurement: Measurement, other: Measurement) -> Measurement:
    """
    Add two measurements together

    Parameters
    ----------
    measurement : Measurement
    other : Measurement

    Returns
    -------
    Measurement
    """
    return Measurement(
        y=measurement.y + other.y,
        K=measurement.K + other.K,
        Sy=measurement.Sy + other.Sy,
    )


class Triplet(MeasurementVector):
    def __init__(
        self,
        wavelength: list[int],
        weights: list[float],
        altitude_range: list[float],
        normalization_range: list[float],
        normalize=True,
        **kwargs,
    ):
        """
        A class that represents a measurement vector that is a weighted combination of log radiances, high altitude normalized

        Note that this measurement vector requires the l1 data to contain the "tangent_altitude" field.

        Both altitude_range and normalization_range can be set through the retrieval context by prefixing the value with a '$'


        Parameters
        ----------
        wavelength : list[int]
            Wavelengths to select
        weights : list[float]
            Weights to apply to the wavelengths
        altitude_range : list[float]
            Altidude range to select
        normalization_range : list[float]
            Altitude range to normalize to
        """
        self._wavelength = wavelength

        def y(l1, ctxt, **kwargs):
            res_altitude_range = [_resolve_value(v, ctxt) for v in altitude_range]
            res_norm_range = [_resolve_value(v, ctxt) for v in normalization_range]

            t_vals = []
            for w, weight in zip(wavelength, weights):
                # Get the useful wavelength data
                wavel_data = log(
                    select(
                        nearest_selector(l1, wavelength=w),
                        tangent_altitude=slice(
                            res_altitude_range[0], res_altitude_range[1]
                        ),
                        **kwargs,
                    )
                )

                # And the normalization value
                norm_vals = mean(
                    log(
                        select(
                            nearest_selector(l1, wavelength=w),
                            tangent_altitude=slice(
                                res_norm_range[0], res_norm_range[1]
                            ),
                            **kwargs,
                        )
                    )
                )

                # The triplet value is the difference of the log radiances subtracted by the normalization multiplied by the weight
                if normalize:
                    t_vals.append(multiply(subtract(wavel_data, norm_vals), weight))
                else:
                    t_vals.append(multiply(wavel_data, weight))
            # Add all of the wavelengths together
            res = t_vals[0]
            for i in range(1, len(t_vals)):
                res = add(res, t_vals[i])

            return res

        super().__init__(y, **kwargs)

    def required_sample_wavelengths(
        self, obs_samples: dict[np.array]
    ) -> dict[np.array]:
        """
        Determines which sample wavelengths are required for this measurement vector

        Default is to just return back all of the observation wavelengths

        Parameters
        ----------
        obs_samples : dict[np.array]

        Returns
        -------
        dict[np.array]
        """
        all_wv = {}

        for key, val in obs_samples.items():
            all_wv[key] = []
            if fnmatch.fnmatch(key, self.filter):
                all_wv[key] = np.array(
                    [val[np.abs(val - w).argmin()] for w in self._wavelength]
                )
        return all_wv
