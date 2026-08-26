from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy import sparse
from scipy.linalg import block_diag
from simpleeval import simple_eval

from skretrieval.core.radianceformat import (
    LinearizedRadianceGridded,
    RadianceGridded,
    VJPContribution,
    evaluate_vjp_contributions,
    select_dataset,
)


def _resolve_value(expr, variables):
    if isinstance(expr, str):
        expr = expr.replace("$", "")
        return simple_eval(expr, names=variables)
    return expr


@dataclass
class Measurement:
    """
    Internal result passed between measurement-vector transformations.

    A measurement carries either a materialized Jacobian in ``K`` or matching
    forward and adjoint product callbacks for matrix-free products. ``pullback``
    defers source VJPs so composed measurements can combine their cotangents.
    """

    y: np.array
    K: np.array | None
    Sy: np.array | None
    matvec: Callable[[np.ndarray], np.ndarray] | None = None
    rmatvec: Callable[[np.ndarray], np.ndarray] | None = None
    pullback: Callable[[np.ndarray], list[VJPContribution]] | None = None
    n_state: int | None = None
    vjp_depth: int = 0

    def __post_init__(self) -> None:
        has_forward_product = self.matvec is not None
        has_reverse_product = self.rmatvec is not None or self.pullback is not None
        if has_forward_product != has_reverse_product:
            msg = "Matrix-free measurements require forward and adjoint products"
            raise ValueError(msg)
        if self.has_operator and self.n_state is None:
            msg = "Matrix-free measurements must declare n_state"
            raise ValueError(msg)
        if not self.has_operator and self.K is None:
            msg = "Measurement must contain a Jacobian or matrix-free products"
            raise ValueError(msg)

    @property
    def has_operator(self) -> bool:
        return self.matvec is not None and (
            self.rmatvec is not None or self.pullback is not None
        )

    @property
    def shape(self) -> tuple[int, int]:
        if self.has_operator:
            return len(self.y), self.n_state
        if self.K is None:
            msg = "Measurement does not contain a Jacobian"
            raise ValueError(msg)
        return self.K.shape

    def jvp(self, x: np.ndarray) -> np.ndarray:
        if self.matvec is not None:
            return self.matvec(x)
        if self.K is None:
            msg = "Measurement does not contain a Jacobian"
            raise ValueError(msg)
        return self.K @ x

    def vjp(self, y: np.ndarray) -> np.ndarray:
        if self.has_operator:
            return evaluate_vjp_contributions(self.vjp_contributions(y))
        if self.K is None:
            msg = "Measurement does not contain a Jacobian"
            raise ValueError(msg)
        return self.K.T @ y

    def vjp_contributions(self, y: np.ndarray) -> list[VJPContribution]:
        """Return deferred source cotangents for composition with other products."""
        if not self.has_operator:
            msg = "Measurement does not contain matrix-free adjoint products"
            raise ValueError(msg)
        cotangent = np.asarray(y).reshape(self.y.shape)
        return [VJPContribution(self, cotangent, self._continue_vjp)]

    def _continue_vjp(self, y: np.ndarray) -> np.ndarray | list[VJPContribution]:
        if self.pullback is not None:
            return self.pullback(y)
        if self.rmatvec is None:
            msg = "Measurement does not contain matrix-free adjoint products"
            raise ValueError(msg)
        return self.rmatvec(y)


def _sum_covariance(left, right):
    """
    Add covariances while preserving sparse matrices of equal shape.

    Unequal shapes use NumPy broadcasting; Triplet normalization relies on a
    shared one-element covariance broadcasting over the altitude profile.
    """
    if left is None and right is None:
        return None
    if left is None or right is None:
        msg = "Cannot combine measurements with mixed covariance availability"
        raise TypeError(msg)
    if (sparse.issparse(left) or sparse.issparse(right)) and np.shape(left) == np.shape(
        right
    ):
        return sparse.csc_matrix(left) + sparse.csc_matrix(right)
    left = left.toarray() if sparse.issparse(left) else np.asarray(left)
    right = right.toarray() if sparse.issparse(right) else np.asarray(right)
    return left + right


def _sum_to_shape(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Apply the adjoint of NumPy broadcasting to an array."""
    result = np.asarray(value)
    while result.ndim > len(shape):
        result = result.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    return result.reshape(shape)


def _combine_measurements(
    measurement: Measurement, other: Measurement, sign: int
) -> Measurement:
    """Add or subtract measurements, including broadcast-aware adjoint products."""
    result_y = measurement.y + sign * other.y
    result_covariance = _sum_covariance(measurement.Sy, other.Sy)

    if not measurement.has_operator and not other.has_operator:
        return Measurement(
            y=result_y,
            K=measurement.K + sign * other.K,
            Sy=result_covariance,
        )

    if not (measurement.has_operator and other.has_operator):
        msg = "Cannot combine mixed materialized and matrix-free measurements"
        raise TypeError(msg)
    if measurement.n_state != other.n_state:
        msg = "Matrix-free measurements must have the same state size"
        raise ValueError(msg)

    def matvec(x: np.ndarray) -> np.ndarray:
        return measurement.jvp(x) + sign * other.jvp(x)

    def pullback(y: np.ndarray) -> list[VJPContribution]:
        cotangent = np.asarray(y).reshape(result_y.shape)
        left_cotangent = _sum_to_shape(cotangent, measurement.y.shape)
        right_cotangent = _sum_to_shape(cotangent, other.y.shape)
        return measurement.vjp_contributions(left_cotangent) + other.vjp_contributions(
            sign * right_cotangent
        )

    return Measurement(
        y=result_y,
        K=None,
        Sy=result_covariance,
        matvec=matvec,
        pullback=pullback,
        n_state=measurement.n_state,
        vjp_depth=max(measurement.vjp_depth, other.vjp_depth) + 1,
    )


class MeasurementVector:
    def __init__(self, fn: Callable, apply_to_filter="*", sample_fn=None):
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
        self._sample_fn = sample_fn
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
        if self._sample_fn is None:
            return obs_samples
        return self._sample_fn(obs_samples)


def pre_process(
    l1: dict[RadianceGridded], n: int = 1, include_error: bool = True
) -> dict[RadianceGridded]:
    """
    Called before the measurement vector is applied. This function will ensure that the L1 data
    always has the necessary fields for the measurement vector to work.

    Parameters
    ----------
    l1 : dict[RadianceGridded]
    n: int, optional
        Number of elements in the state vector, used to create the dummy Jacobian. Default is 1
        which can be used if the number of elements in the state vector is not important.
    include_error : bool, optional
        If false, skip modeled measurement covariance construction.

    Returns
    -------
    dict[RadianceGridded]
    """
    # Copy and modify the data to always include wf and noise values
    new_l1 = {}

    for key, val in l1.items():
        new_val = val.data.copy(deep=not isinstance(val, LinearizedRadianceGridded))

        if "wf" not in new_val and not isinstance(val, LinearizedRadianceGridded):
            new_val["wf"] = xr.zeros_like(
                new_val["radiance"].expand_dims({"x": n}, axis=-1)
            )

        if include_error and "radiance_noise" not in new_val:
            new_val["radiance_noise"] = new_val["radiance"] * 1
        elif not include_error and "radiance_noise" in new_val:
            new_val = new_val.drop_vars("radiance_noise")

        if isinstance(val, LinearizedRadianceGridded):
            new_l1[key] = val.with_data(new_val)
        else:
            new_l1[key] = RadianceGridded(new_val)
    return new_l1


def _measurement_from_selection(radiance: RadianceGridded, **kwargs) -> Measurement:
    """Select one radiance into a materialized or matrix-free measurement."""
    if not isinstance(radiance, LinearizedRadianceGridded):
        selected = select_dataset(radiance.data, kwargs)
        covariance = None
        if "radiance_noise" in selected:
            covariance = sparse.diags(
                selected["radiance_noise"].to_numpy().flatten() ** 2,
                format="csc",
            )
        num_measurements = selected["radiance"].size
        num_state = len(selected["x"])
        return Measurement(
            y=selected["radiance"].to_numpy().flatten(),
            K=selected["wf"].to_numpy().reshape((num_measurements, num_state)),
            Sy=covariance,
        )

    plan = radiance.selection_plan(**kwargs)

    def matvec(x: np.ndarray) -> np.ndarray:
        return radiance.selection_jvp(plan, x).reshape(-1)

    def pullback(y: np.ndarray) -> list[VJPContribution]:
        return radiance.selection_vjp_contributions(plan, y)

    covariance = None
    if "radiance_noise" in radiance.data:
        covariance = sparse.diags(
            radiance.selection_values(plan, "radiance_noise").reshape(-1) ** 2,
            format="csc",
        )

    return Measurement(
        y=radiance.selection_values(plan).reshape(-1),
        K=None,
        Sy=covariance,
        matvec=matvec,
        pullback=pullback,
        n_state=radiance.n_state,
        vjp_depth=radiance.vjp_depth + 1,
    )


def _nearest_wavelength(radiance: RadianceGridded, wavelength: float) -> float:
    values = radiance.data["wavelength"].to_numpy()
    return float(values[np.abs(values - wavelength).argmin()])


def concat(measurements: list[Measurement]) -> Measurement | None:
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

    combined_y = np.concatenate([m.y for m in measurements])
    if all(m.Sy is None for m in measurements):
        combined_covariance = None
    elif any(m.Sy is None for m in measurements):
        msg = "Cannot concatenate measurements with mixed covariance availability"
        raise TypeError(msg)
    else:
        combined_covariance = sparse.block_diag(
            [sparse.csc_matrix(m.Sy) for m in measurements], format="csc"
        )
    if any(m.has_operator for m in measurements):
        if not all(m.has_operator for m in measurements):
            msg = "Cannot concatenate mixed materialized and matrix-free measurements"
            raise TypeError(msg)

        lengths = [len(m.y) for m in measurements]
        n_state = measurements[0].n_state
        if any(m.n_state != n_state for m in measurements):
            msg = "All matrix-free measurements must have the same state size"
            raise ValueError(msg)

        def matvec(x: np.ndarray) -> np.ndarray:
            return np.concatenate([m.jvp(x) for m in measurements])

        def pullback(y: np.ndarray) -> list[VJPContribution]:
            contributions = []
            start = 0
            for length, measurement in zip(lengths, measurements):
                end = start + length
                contributions.extend(measurement.vjp_contributions(y[start:end]))
                start = end
            return contributions

        return Measurement(
            y=combined_y,
            K=None,
            Sy=combined_covariance,
            matvec=matvec,
            pullback=pullback,
            n_state=n_state,
            vjp_depth=max(m.vjp_depth for m in measurements) + 1,
        )

    return Measurement(
        y=combined_y,
        K=np.vstack([m.K for m in measurements]),
        Sy=combined_covariance,
    )


def post_process(measurement: Measurement | None) -> dict:
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
    if measurement is None:
        msg = "No enabled measurement vectors produced measurements"
        raise ValueError(msg)

    # At this stage we have to remove the jacobian if it was a dummy one in the beginning
    res = {"y": measurement.y}
    if measurement.Sy is not None:
        res["y_error"] = measurement.Sy

    if measurement.has_operator:
        res["jacobian_operator"] = measurement
    elif measurement.K is not None and measurement.K.shape[-1] != 0:
        res["jacobian"] = measurement.K

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
            measurements.append(_measurement_from_selection(val, **kwargs))

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
            if isinstance(val, LinearizedRadianceGridded):
                res[key] = val.select(method="nearest", assign_coords=kwargs, **kwargs)
            else:
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
    result_y = np.log(measurement.y)
    result_covariance = None
    if measurement.Sy is not None:
        result_covariance = measurement.Sy / np.outer(measurement.y, measurement.y)
    if measurement.has_operator:

        def matvec(x: np.ndarray, *, meas=measurement) -> np.ndarray:
            return meas.jvp(x) / meas.y

        def pullback(y: np.ndarray, *, meas=measurement) -> list[VJPContribution]:
            return meas.vjp_contributions(y / meas.y)

        return Measurement(
            y=result_y,
            K=None,
            Sy=result_covariance,
            matvec=matvec,
            pullback=pullback,
            n_state=measurement.n_state,
            vjp_depth=measurement.vjp_depth + 1,
        )

    return Measurement(
        y=result_y,
        K=measurement.K / measurement.y[:, np.newaxis],
        Sy=result_covariance,
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
    result_y = np.atleast_1d(np.mean(measurement.y))
    result_covariance = None
    if measurement.Sy is not None:
        result_covariance = np.atleast_2d(np.mean(measurement.Sy.diagonal()))
    if measurement.has_operator:

        def matvec(x: np.ndarray, *, meas=measurement) -> np.ndarray:
            return np.atleast_1d(np.mean(meas.jvp(x)))

        def pullback(y: np.ndarray, *, meas=measurement) -> list[VJPContribution]:
            cotangent = (
                np.ones_like(meas.y) * np.asarray(y).reshape(-1)[0] / len(meas.y)
            )
            return meas.vjp_contributions(cotangent)

        return Measurement(
            y=result_y,
            K=None,
            Sy=result_covariance,
            matvec=matvec,
            pullback=pullback,
            n_state=measurement.n_state,
            vjp_depth=measurement.vjp_depth + 1,
        )

    return Measurement(
        y=result_y,
        K=np.atleast_2d(np.mean(measurement.K, axis=0)),
        Sy=result_covariance,
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
    result_y = measurement.y * factor
    result_covariance = None if measurement.Sy is None else measurement.Sy * factor**2
    if measurement.has_operator:

        def matvec(x: np.ndarray, *, meas=measurement, scale=factor) -> np.ndarray:
            return meas.jvp(x) * scale

        def pullback(
            y: np.ndarray, *, meas=measurement, scale=factor
        ) -> list[VJPContribution]:
            return meas.vjp_contributions(y * scale)

        return Measurement(
            y=result_y,
            K=None,
            Sy=result_covariance,
            matvec=matvec,
            pullback=pullback,
            n_state=measurement.n_state,
            vjp_depth=measurement.vjp_depth + 1,
        )

    return Measurement(
        y=result_y,
        K=measurement.K * factor,
        Sy=result_covariance,
    )


def multiply_elementwise(
    measurement: Measurement,
    factor: np.ndarray,
) -> Measurement:
    """Multiply each measurement row by its corresponding scale factor."""
    factor = np.asarray(factor, dtype=float).reshape(-1)
    if factor.shape != measurement.y.shape:
        msg = (
            "Elementwise measurement factors must match the measurement shape; "
            f"got {factor.shape} and {measurement.y.shape}"
        )
        raise ValueError(msg)

    result_covariance = None
    if measurement.Sy is not None:
        if sparse.issparse(measurement.Sy):
            scaling = sparse.diags(factor, format="csc")
            result_covariance = scaling @ measurement.Sy @ scaling
        else:
            result_covariance = np.asarray(measurement.Sy) * np.outer(factor, factor)

    if measurement.has_operator:

        def matvec(x: np.ndarray, *, meas=measurement, scale=factor) -> np.ndarray:
            return meas.jvp(x) * scale

        def pullback(
            y: np.ndarray,
            *,
            meas=measurement,
            scale=factor,
        ) -> list[VJPContribution]:
            return meas.vjp_contributions(np.asarray(y) * scale)

        return Measurement(
            y=measurement.y * factor,
            K=None,
            Sy=result_covariance,
            matvec=matvec,
            pullback=pullback,
            n_state=measurement.n_state,
            vjp_depth=measurement.vjp_depth + 1,
        )

    return Measurement(
        y=measurement.y * factor,
        K=measurement.K * factor[:, np.newaxis],
        Sy=result_covariance,
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
    return _combine_measurements(measurement, other, -1)


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
    return _combine_measurements(measurement, other, 1)


def wavelength_mean(
    l1: dict[RadianceGridded], filter: str = "*", **kwargs
) -> Measurement:
    """
    Takes the mean over a wavelength band


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
            if isinstance(val, LinearizedRadianceGridded):
                selected = val.select(**kwargs)
                wavelength_axis = selected.data["radiance"].get_axis_num("wavelength")
                result_y = np.mean(
                    selected.data["radiance"].to_numpy(), axis=wavelength_axis
                ).reshape(-1)

                def matvec(
                    x: np.ndarray,
                    *,
                    selected_value=selected,
                    axis=wavelength_axis,
                ) -> np.ndarray:
                    return np.mean(
                        np.asarray(selected_value.jvp(x)), axis=axis
                    ).reshape(-1)

                def pullback(
                    y: np.ndarray,
                    *,
                    selected_value=selected,
                    axis=wavelength_axis,
                ) -> list[VJPContribution]:
                    output_shape = list(selected_value.data["radiance"].shape)
                    count = output_shape.pop(axis)
                    cotangent = np.asarray(y).reshape(output_shape) / count
                    cotangent = np.expand_dims(cotangent, axis)
                    cotangent = np.broadcast_to(
                        cotangent, selected_value.data["radiance"].shape
                    )
                    return selected_value.vjp_contributions(cotangent)

                covariance = None
                if "radiance_noise" in selected.data:
                    noise = np.mean(
                        selected.data["radiance_noise"].to_numpy(),
                        axis=wavelength_axis,
                    ).reshape(-1)
                    covariance = sparse.diags(noise**2, format="csc")

                measurements.append(
                    Measurement(
                        y=result_y,
                        K=None,
                        Sy=covariance,
                        matvec=matvec,
                        pullback=pullback,
                        n_state=val.n_state,
                        vjp_depth=selected.vjp_depth + 1,
                    )
                )
                continue

            selected = select_dataset(val.data, kwargs).mean(dim="wavelength")
            covariance = None
            if "radiance_noise" in selected:
                covariance = sparse.diags(
                    selected["radiance_noise"].to_numpy().flatten() ** 2,
                    format="csc",
                )
            measurements.append(
                Measurement(
                    y=selected["radiance"].to_numpy().flatten(),
                    K=selected["wf"].to_numpy().reshape((-1, len(selected["x"]))),
                    Sy=covariance,
                )
            )

    return concat(measurements)


class Triplet(MeasurementVector):
    def __init__(
        self,
        wavelength: list[int],
        weights: list[float],
        altitude_range: list[float],
        normalization_range: list[float],
        normalize=True,
        log_space=True,
        group_by: str | None = None,
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
        group_by : str, optional
            One-dimensional coordinate used to normalize independent profiles,
            for example ``"image"`` for a flattened orbital observation.
        """
        self._wavelength = wavelength
        nearest_wavelength_cache = {}

        def y(l1, ctxt, **kwargs):
            res_altitude_range = [_resolve_value(v, ctxt) for v in altitude_range]
            if normalize:
                res_norm_range = [_resolve_value(v, ctxt) for v in normalization_range]

            matched = {
                key: value
                for key, value in l1.items()
                if fnmatch.fnmatch(key, kwargs.get("filter", "*"))
            }
            grouped = []
            if group_by is None:
                grouped.append((matched, {}))
            else:
                for key, value in matched.items():
                    if group_by not in value.data.coords:
                        msg = f"Triplet group coordinate {group_by!r} is missing"
                        raise KeyError(msg)
                    group_coordinate = value.data.coords[group_by]
                    if group_coordinate.ndim != 1:
                        msg = "Triplet group coordinate must be one-dimensional"
                        raise ValueError(msg)
                    group_values = np.asarray(group_coordinate)
                    _, first_index = np.unique(group_values, return_index=True)
                    for group_value in group_values[np.sort(first_index)]:
                        grouped.append(({key: value}, {group_by: group_value}))

            grouped_results = []
            for measurements, group_selector in grouped:
                t_vals = []
                for w, weight in zip(wavelength, weights):
                    for key, value in measurements.items():
                        cache_key = key, w
                        wavelength_grid = value.data["wavelength"].to_numpy()
                        cached = nearest_wavelength_cache.get(cache_key)
                        if cached is None or not np.array_equal(
                            cached[0], wavelength_grid
                        ):
                            nearest_wavelength_cache[cache_key] = (
                                np.array(wavelength_grid, copy=True),
                                _nearest_wavelength(value, w),
                            )

                    def altitude_measurement(
                        altitude_slice,
                        *,
                        selected_l1=measurements,
                        wavelength=w,
                        selector=group_selector,
                    ):
                        selected = [
                            _measurement_from_selection(
                                value,
                                wavelength=nearest_wavelength_cache[(key, wavelength)][
                                    1
                                ],
                                tangent_altitude=altitude_slice,
                                **selector,
                            )
                            for key, value in selected_l1.items()
                        ]
                        return concat(selected)

                    if log_space:
                        wavel_data = log(
                            altitude_measurement(
                                slice(res_altitude_range[0], res_altitude_range[1])
                            )
                        )
                    else:
                        wavel_data = altitude_measurement(
                            slice(res_altitude_range[0], res_altitude_range[1])
                        )

                    if normalize:
                        norm_vals = mean(
                            log(
                                altitude_measurement(
                                    slice(res_norm_range[0], res_norm_range[1])
                                )
                            )
                        )
                        t_vals.append(multiply(subtract(wavel_data, norm_vals), weight))
                    else:
                        t_vals.append(multiply(wavel_data, weight))

                result = t_vals[0]
                for value in t_vals[1:]:
                    result = add(result, value)
                grouped_results.append(result)

            return concat(grouped_results)

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


@dataclass(frozen=True)
class _GroupedAltitudeTripletPlan:
    """Sparse log-radiance transforms shared by every retrieval iteration."""

    transform: sparse.csr_matrix
    legacy_variance_weights: sparse.csr_matrix
    validity_inputs: sparse.csr_matrix


def _grouped_altitude_triplet_plan(
    radiance: RadianceGridded,
    *,
    wavelength: tuple[tuple[float, ...], ...],
    weights: tuple[tuple[float, ...], ...],
    normalization_range: tuple[tuple[float, ...], ...],
    altitude_weight_grid: tuple[np.ndarray, ...],
    altitude_weight_values: tuple[np.ndarray, ...],
    altitude_range: tuple[float, float],
    group_by: str,
    open_altitude_bounds: bool,
) -> _GroupedAltitudeTripletPlan:
    """Compile one grouped triplet sum into sparse forward/adjoint maps."""
    data = radiance.data
    template = data["radiance"]
    if set(template.dims) != {"wavelength", "los"} or template.ndim != 2:
        msg = "Grouped altitude triplets require radiance dimensions wavelength and los"
        raise ValueError(msg)
    for coordinate in ("wavelength", "tangent_altitude", group_by):
        if coordinate not in data.coords:
            msg = f"Triplet coordinate {coordinate!r} is missing"
            raise KeyError(msg)
    if data.coords["tangent_altitude"].dims != ("los",):
        msg = "Triplet tangent altitude must be one-dimensional over los"
        raise ValueError(msg)
    if data.coords[group_by].dims != ("los",):
        msg = "Triplet group coordinate must be one-dimensional over los"
        raise ValueError(msg)

    wavelength_grid = np.asarray(data.coords["wavelength"], dtype=float)
    tangent_altitude = np.asarray(data.coords["tangent_altitude"], dtype=float)
    group_coordinate = np.asarray(data.coords[group_by])
    _, first_group_index = np.unique(group_coordinate, return_index=True)
    group_values = group_coordinate[np.sort(first_group_index)]

    lower_altitude, upper_altitude = altitude_range
    if open_altitude_bounds:
        profile_mask = (tangent_altitude > lower_altitude) & (
            tangent_altitude < upper_altitude
        )
    else:
        profile_mask = (tangent_altitude >= lower_altitude) & (
            tangent_altitude <= upper_altitude
        )
    output_los_parts = [
        np.flatnonzero(profile_mask & (group_coordinate == group_value))
        for group_value in group_values
    ]
    output_los = np.concatenate(output_los_parts)
    output_altitude = tangent_altitude[output_los]
    output_count = output_los.size
    input_shape = template.shape
    wavelength_axis = template.get_axis_num("wavelength")
    los_axis = template.get_axis_num("los")

    def flat_indices(wavelength_index: int, los_indices: np.ndarray) -> np.ndarray:
        coordinates = [None] * 2
        coordinates[wavelength_axis] = np.full(
            len(los_indices), wavelength_index, dtype=int
        )
        coordinates[los_axis] = np.asarray(los_indices, dtype=int)
        return np.ravel_multi_index(tuple(coordinates), input_shape)

    rows = []
    columns = []
    coefficients = []
    variance_rows = []
    variance_columns = []
    variance_coefficients = []
    validity_rows = []
    validity_columns = []
    validity_coefficients = []

    group_output_rows = []
    start = 0
    for local_los in output_los_parts:
        group_output_rows.append(np.arange(start, start + len(local_los), dtype=int))
        start += len(local_los)

    for (
        triplet_wavelengths,
        triplet_weights,
        triplet_normalization,
        weight_grid,
        weight_values,
    ) in zip(
        wavelength,
        weights,
        normalization_range,
        altitude_weight_grid,
        altitude_weight_values,
        strict=True,
    ):
        altitude_weight = np.interp(
            output_altitude,
            weight_grid,
            weight_values,
            left=weight_values[0],
            right=weight_values[-1],
        )
        altitude_weight[output_altitude >= triplet_normalization[0]] = 0.0

        for requested_wavelength, spectral_weight in zip(
            triplet_wavelengths, triplet_weights, strict=True
        ):
            wavelength_index = int(
                np.argmin(np.abs(wavelength_grid - requested_wavelength))
            )
            point_coefficient = altitude_weight * spectral_weight
            nonzero = point_coefficient != 0
            validity_rows.append(np.flatnonzero(nonzero))
            validity_columns.append(flat_indices(wavelength_index, output_los[nonzero]))
            validity_coefficients.append(
                np.ones(np.count_nonzero(nonzero), dtype=np.int8)
            )
            rows.append(np.flatnonzero(nonzero))
            columns.append(flat_indices(wavelength_index, output_los[nonzero]))
            coefficients.append(point_coefficient[nonzero])
            variance_rows.append(np.flatnonzero(nonzero))
            variance_columns.append(flat_indices(wavelength_index, output_los[nonzero]))
            variance_coefficients.append(
                altitude_weight[nonzero] * abs(spectral_weight)
            )

            normalization_mask = (tangent_altitude >= triplet_normalization[0]) & (
                tangent_altitude <= triplet_normalization[1]
            )
            for group_value, output_rows_for_group in zip(
                group_values, group_output_rows, strict=True
            ):
                normalization_los = np.flatnonzero(
                    normalization_mask & (group_coordinate == group_value)
                )
                if normalization_los.size == 0:
                    msg = (
                        "Triplet normalization range contains no samples for "
                        f"{group_by}={group_value!r}"
                    )
                    raise ValueError(msg)
                local_weight = point_coefficient[output_rows_for_group]
                local_nonzero = local_weight != 0
                if not np.any(local_nonzero):
                    continue
                active_rows = output_rows_for_group[local_nonzero]
                normalization_columns = flat_indices(
                    wavelength_index, normalization_los
                )
                validity_rows.append(np.repeat(active_rows, normalization_los.size))
                validity_columns.append(
                    np.tile(normalization_columns, active_rows.size)
                )
                validity_coefficients.append(
                    np.ones(
                        active_rows.size * normalization_los.size,
                        dtype=np.int8,
                    )
                )
                rows.append(np.repeat(active_rows, normalization_los.size))
                columns.append(np.tile(normalization_columns, active_rows.size))
                coefficients.append(
                    np.repeat(
                        -local_weight[local_nonzero] / normalization_los.size,
                        normalization_los.size,
                    )
                )

    input_count = int(np.prod(input_shape))

    def matrix(row_parts, column_parts, value_parts):
        return sparse.coo_matrix(
            (
                np.concatenate(value_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(output_count, input_count),
        ).tocsr()

    return _GroupedAltitudeTripletPlan(
        transform=matrix(rows, columns, coefficients),
        legacy_variance_weights=matrix(
            variance_rows,
            variance_columns,
            variance_coefficients,
        ),
        validity_inputs=matrix(
            validity_rows,
            validity_columns,
            validity_coefficients,
        ),
    )


def _apply_grouped_altitude_triplet_plan(
    radiance: RadianceGridded,
    plan: _GroupedAltitudeTripletPlan,
) -> Measurement:
    """Apply a compiled grouped triplet map to values and derivative products."""
    base = _measurement_from_selection(radiance)
    log_radiance = np.log(base.y)
    result_y = np.asarray(plan.transform @ log_radiance).reshape(-1)
    invalid_input = ~np.isfinite(log_radiance)
    invalid_output = (
        np.asarray(plan.validity_inputs @ invalid_input.astype(np.int8)).reshape(-1)
        != 0
    )
    result_y[invalid_output] = np.nan

    covariance = None
    if base.Sy is not None:
        relative_variance = np.asarray(base.Sy.diagonal()).reshape(-1) / base.y**2
        variance = np.asarray(plan.legacy_variance_weights @ relative_variance).reshape(
            -1
        )
        variance[variance <= 0] = 1.0
        covariance = sparse.diags(variance, format="csc")

    if base.has_operator:

        def matvec(x: np.ndarray, *, measurement=base, transform=plan.transform):
            return np.asarray(transform @ (measurement.jvp(x) / measurement.y)).reshape(
                -1
            )

        def pullback(
            y: np.ndarray, *, measurement=base, transform=plan.transform
        ) -> list[VJPContribution]:
            cotangent = np.asarray(transform.T @ np.asarray(y).reshape(-1)).reshape(-1)
            return measurement.vjp_contributions(cotangent / measurement.y)

        return Measurement(
            y=result_y,
            K=None,
            Sy=covariance,
            matvec=matvec,
            pullback=pullback,
            n_state=base.n_state,
            vjp_depth=base.vjp_depth + 1,
        )

    return Measurement(
        y=result_y,
        K=np.asarray(plan.transform @ (base.K / base.y[:, np.newaxis])),
        Sy=covariance,
    )


class AltitudeWeightedTripletSum(MeasurementVector):
    """Sum normalized spectral triplets on one altitude-dependent output grid.

    Each triplet is normalized independently, multiplied by a piecewise-linear
    altitude weight, and added to the same measurement row. This is useful for
    staged limb-retrieval measurement vectors where one spectral combination
    hands sensitivity to another as tangent altitude changes.

    ``legacy_linear_covariance`` reproduces algorithms that accumulated each
    triplet's point variance with one power of its altitude weight and omitted
    normalization-window noise. The default propagates the available modeled
    covariance through the measurement transformations instead.
    """

    def __init__(
        self,
        wavelength: list[list[float]],
        weights: list[list[float]],
        normalization_range: list[list[float]],
        altitude_weight_grid: list[list[float]],
        altitude_weight_values: list[list[float]],
        altitude_range: list[float],
        *,
        group_by: str | None = None,
        open_altitude_bounds: bool = False,
        legacy_linear_covariance: bool = False,
        **kwargs,
    ):
        triplet_count = len(wavelength)
        if triplet_count == 0:
            msg = "At least one triplet is required"
            raise ValueError(msg)
        if not all(
            len(values) == triplet_count
            for values in (
                weights,
                normalization_range,
                altitude_weight_grid,
                altitude_weight_values,
            )
        ):
            msg = "All triplet configuration lists must have equal length"
            raise ValueError(msg)
        if len(altitude_range) != 2 or altitude_range[0] >= altitude_range[1]:
            msg = "altitude_range must contain two increasing values"
            raise ValueError(msg)

        for index in range(triplet_count):
            if len(wavelength[index]) != len(weights[index]):
                msg = f"Triplet {index} wavelengths and weights must have equal length"
                raise ValueError(msg)
            if (
                len(normalization_range[index]) != 2
                or normalization_range[index][0] > normalization_range[index][1]
            ):
                msg = f"Triplet {index} normalization range is invalid"
                raise ValueError(msg)
            grid = np.asarray(altitude_weight_grid[index], dtype=float)
            values = np.asarray(altitude_weight_values[index], dtype=float)
            if (
                grid.ndim != 1
                or grid.size < 2
                or values.shape != grid.shape
                or np.any(~np.isfinite(grid))
                or np.any(~np.isfinite(values))
                or np.any(np.diff(grid) <= 0)
            ):
                msg = f"Triplet {index} altitude weighting is invalid"
                raise ValueError(msg)

        self._wavelength = tuple(tuple(value) for value in wavelength)
        self._weights = tuple(tuple(value) for value in weights)
        self._normalization_range = tuple(tuple(value) for value in normalization_range)
        self._altitude_weight_grid = tuple(
            np.asarray(value, dtype=float) for value in altitude_weight_grid
        )
        self._altitude_weight_values = tuple(
            np.asarray(value, dtype=float) for value in altitude_weight_values
        )
        self._altitude_range = tuple(float(value) for value in altitude_range)
        self._group_by = group_by
        self._open_altitude_bounds = bool(open_altitude_bounds)
        self._legacy_linear_covariance = bool(legacy_linear_covariance)
        self._compiled_group_plan_enabled = True
        nearest_wavelength_cache = {}
        grouped_plan_cache = {}

        def y(l1, ctxt, **apply_kwargs):
            del ctxt
            matched = {
                key: value
                for key, value in l1.items()
                if fnmatch.fnmatch(key, apply_kwargs.get("filter", "*"))
            }
            if (
                group_by is not None
                and self._legacy_linear_covariance
                and self._compiled_group_plan_enabled
            ):
                results = []
                for key, value in matched.items():
                    wavelength_grid = np.asarray(value.data["wavelength"])
                    tangent_altitude = np.asarray(value.data["tangent_altitude"])
                    group_coordinate = np.asarray(value.data[group_by])
                    cached = grouped_plan_cache.get(key)
                    if cached is None or not (
                        np.array_equal(cached[0], wavelength_grid)
                        and np.array_equal(cached[1], tangent_altitude)
                        and np.array_equal(cached[2], group_coordinate)
                        and cached[3] == value.data["radiance"].dims
                    ):
                        plan = _grouped_altitude_triplet_plan(
                            value,
                            wavelength=self._wavelength,
                            weights=self._weights,
                            normalization_range=self._normalization_range,
                            altitude_weight_grid=self._altitude_weight_grid,
                            altitude_weight_values=self._altitude_weight_values,
                            altitude_range=self._altitude_range,
                            group_by=group_by,
                            open_altitude_bounds=self._open_altitude_bounds,
                        )
                        cached = (
                            np.array(wavelength_grid, copy=True),
                            np.array(tangent_altitude, copy=True),
                            np.array(group_coordinate, copy=True),
                            value.data["radiance"].dims,
                            plan,
                        )
                        grouped_plan_cache[key] = cached
                    results.append(
                        _apply_grouped_altitude_triplet_plan(value, cached[4])
                    )
                return concat(results)

            grouped = []
            if group_by is None:
                grouped.append((matched, {}))
            else:
                for key, value in matched.items():
                    if group_by not in value.data.coords:
                        msg = f"Triplet group coordinate {group_by!r} is missing"
                        raise KeyError(msg)
                    group_coordinate = value.data.coords[group_by]
                    if group_coordinate.ndim != 1:
                        msg = "Triplet group coordinate must be one-dimensional"
                        raise ValueError(msg)
                    group_values = np.asarray(group_coordinate)
                    _, first_index = np.unique(group_values, return_index=True)
                    for group_value in group_values[np.sort(first_index)]:
                        grouped.append(({key: value}, {group_by: group_value}))

            lower_altitude, upper_altitude = self._altitude_range
            if self._open_altitude_bounds:
                lower_altitude = np.nextafter(lower_altitude, np.inf)
                upper_altitude = np.nextafter(upper_altitude, -np.inf)
            profile_altitude_slice = slice(lower_altitude, upper_altitude)

            grouped_results = []
            for measurements, group_selector in grouped:
                for triplet_wavelengths in self._wavelength:
                    for requested in triplet_wavelengths:
                        for key, value in measurements.items():
                            cache_key = key, requested
                            wavelength_grid = value.data["wavelength"].to_numpy()
                            cached = nearest_wavelength_cache.get(cache_key)
                            if cached is None or not np.array_equal(
                                cached[0], wavelength_grid
                            ):
                                nearest_wavelength_cache[cache_key] = (
                                    np.array(wavelength_grid, copy=True),
                                    _nearest_wavelength(value, requested),
                                )

                first_requested = self._wavelength[0][0]
                altitude_parts = []
                for key, value in measurements.items():
                    selected = select_dataset(
                        value.data,
                        {
                            "wavelength": nearest_wavelength_cache[
                                (key, first_requested)
                            ][1],
                            "tangent_altitude": profile_altitude_slice,
                            **group_selector,
                        },
                    )
                    altitude_parts.append(
                        np.asarray(selected["tangent_altitude"], dtype=float).reshape(
                            -1
                        )
                    )
                tangent_altitude = np.concatenate(altitude_parts)

                def altitude_measurement(
                    altitude_slice,
                    requested_wavelength,
                    *,
                    selected_l1=measurements,
                    selector=group_selector,
                ):
                    selected = [
                        _measurement_from_selection(
                            value,
                            wavelength=nearest_wavelength_cache[
                                (key, requested_wavelength)
                            ][1],
                            tangent_altitude=altitude_slice,
                            **selector,
                        )
                        for key, value in selected_l1.items()
                    ]
                    return concat(selected)

                combined = None
                legacy_variance = np.zeros(tangent_altitude.size, dtype=float)
                has_legacy_covariance = False
                for (
                    triplet_wavelengths,
                    triplet_weights,
                    triplet_normalization,
                    weight_grid,
                    weight_values,
                ) in zip(
                    self._wavelength,
                    self._weights,
                    self._normalization_range,
                    self._altitude_weight_grid,
                    self._altitude_weight_values,
                    strict=True,
                ):
                    triplet_measurement = None
                    triplet_point_variance = np.zeros(
                        tangent_altitude.size,
                        dtype=float,
                    )
                    for requested, spectral_weight in zip(
                        triplet_wavelengths,
                        triplet_weights,
                        strict=True,
                    ):
                        point = log(
                            altitude_measurement(
                                profile_altitude_slice,
                                requested,
                            )
                        )
                        normalization = mean(
                            log(
                                altitude_measurement(
                                    slice(*triplet_normalization),
                                    requested,
                                )
                            )
                        )
                        term = multiply(
                            subtract(point, normalization),
                            spectral_weight,
                        )
                        triplet_measurement = (
                            term
                            if triplet_measurement is None
                            else add(triplet_measurement, term)
                        )
                        if point.Sy is not None:
                            has_legacy_covariance = True
                            triplet_point_variance += abs(spectral_weight) * np.asarray(
                                point.Sy.diagonal()
                            ).reshape(-1)

                    altitude_weight = np.interp(
                        tangent_altitude,
                        weight_grid,
                        weight_values,
                        left=weight_values[0],
                        right=weight_values[-1],
                    )
                    altitude_weight[tangent_altitude >= triplet_normalization[0]] = 0
                    weighted = multiply_elementwise(
                        triplet_measurement,
                        altitude_weight,
                    )
                    combined = weighted if combined is None else add(combined, weighted)
                    legacy_variance += altitude_weight * triplet_point_variance

                if self._legacy_linear_covariance:
                    if has_legacy_covariance:
                        legacy_variance[legacy_variance <= 0] = 1.0
                        combined.Sy = sparse.diags(legacy_variance, format="csc")
                    else:
                        combined.Sy = None
                grouped_results.append(combined)

            return concat(grouped_results)

        super().__init__(y, **kwargs)

    def required_sample_wavelengths(
        self,
        obs_samples: dict[np.array],
    ) -> dict[np.array]:
        all_wavelengths = np.unique(np.concatenate(self._wavelength))
        selected = {}
        for key, values in obs_samples.items():
            selected[key] = []
            if fnmatch.fnmatch(key, self.filter):
                selected[key] = np.asarray(
                    [
                        values[np.abs(values - value).argmin()]
                        for value in all_wavelengths
                    ]
                )
        return selected


class IntegratedLine(MeasurementVector):
    def __init__(
        self,
        central_wavelength: float,
        integration_range: float,
        baseline_range: float,
        **kwargs,
    ):
        self._left_boundary = central_wavelength - integration_range - baseline_range
        self._right_boundary = central_wavelength + integration_range + baseline_range

        def y(l1, ctxt, **kwargs):  # noqa: ARG001

            ta_s = slice(70000, 110000)

            integration_vals = wavelength_mean(
                l1,
                wavelength=slice(
                    central_wavelength - integration_range,
                    central_wavelength + integration_range,
                ),
                tangent_altitude=ta_s,
            )
            baseline_left = wavelength_mean(
                l1,
                wavelength=slice(
                    central_wavelength - integration_range - baseline_range,
                    central_wavelength - integration_range,
                ),
                tangent_altitude=ta_s,
            )
            baseline_right = wavelength_mean(
                l1,
                wavelength=slice(
                    central_wavelength + integration_range,
                    central_wavelength + integration_range + baseline_range,
                ),
                tangent_altitude=ta_s,
            )

            baseline = multiply(add(baseline_left, baseline_right), 0.5)
            return subtract(integration_vals, baseline)

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
                all_wv[key] = val[
                    (val > self._left_boundary) & (val < self._right_boundary)
                ]
        return all_wv


class WavelengthAltitude(MeasurementVector):
    def __init__(
        self,
        wavelength_range: list[float],
        altitude_range: list[float],
        **kwargs,
    ):
        """
        A measurement vector that selects all measurements inside wavelength and
        tangent altitude ranges.

        Both ranges can be set through the retrieval context by prefixing values
        with '$'.

        Parameters
        ----------
        wavelength_range : list[float]
            Wavelength range to select as [min, max]
        altitude_range : list[float]
            Tangent altitude range to select as [min, max]
        """
        self._wavelength_range = wavelength_range

        def y(l1, ctxt, **kwargs):
            res_wavelength_range = [_resolve_value(v, ctxt) for v in wavelength_range]
            res_altitude_range = [_resolve_value(v, ctxt) for v in altitude_range]

            return select(
                l1,
                wavelength=slice(res_wavelength_range[0], res_wavelength_range[1]),
                tangent_altitude=slice(res_altitude_range[0], res_altitude_range[1]),
                **kwargs,
            )

        super().__init__(y, **kwargs)

    def required_sample_wavelengths(
        self, obs_samples: dict[np.array]
    ) -> dict[np.array]:
        """
        Determines which sample wavelengths are required for this measurement vector.

        Parameters
        ----------
        obs_samples : dict[np.array]

        Returns
        -------
        dict[np.array]
        """
        all_wv = {}

        # If range values are context-dependent, we cannot resolve static sampling here.
        if any(isinstance(v, str) for v in self._wavelength_range):
            return obs_samples

        left = self._wavelength_range[0]
        right = self._wavelength_range[1]

        for key, val in obs_samples.items():
            all_wv[key] = []
            if fnmatch.fnmatch(key, self.filter):
                all_wv[key] = val[(val >= left) & (val <= right)]
        return all_wv
