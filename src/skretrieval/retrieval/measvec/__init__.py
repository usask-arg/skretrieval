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
        selected = radiance.data.sel(**kwargs)
        covariance = None
        if "radiance_noise" in selected:
            covariance = sparse.diags(
                selected["radiance_noise"].to_numpy().flatten() ** 2,
                format="csc",
            )
        return Measurement(
            y=selected["radiance"].to_numpy().flatten(),
            K=selected["wf"].to_numpy().reshape((-1, len(selected["x"]))),
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

            selected = val.data.sel(**kwargs).mean(dim="wavelength")
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
        nearest_wavelength_cache = {}

        def y(l1, ctxt, **kwargs):
            res_altitude_range = [_resolve_value(v, ctxt) for v in altitude_range]
            if normalize:
                res_norm_range = [_resolve_value(v, ctxt) for v in normalization_range]

            t_vals = []
            for w, weight in zip(wavelength, weights):
                measurements = {}
                for key, value in l1.items():
                    if not fnmatch.fnmatch(key, kwargs.get("filter", "*")):
                        continue
                    cache_key = key, w
                    wavelength_grid = value.data["wavelength"].to_numpy()
                    cached = nearest_wavelength_cache.get(cache_key)
                    if cached is None or not np.array_equal(cached[0], wavelength_grid):
                        nearest_wavelength_cache[cache_key] = (
                            np.array(wavelength_grid, copy=True),
                            _nearest_wavelength(value, w),
                        )
                    measurements[key] = value

                def altitude_measurement(
                    altitude_slice, *, selected_l1=measurements, wavelength=w
                ):
                    selected = [
                        _measurement_from_selection(
                            value,
                            wavelength=nearest_wavelength_cache[(key, wavelength)][1],
                            tangent_altitude=altitude_slice,
                        )
                        for key, value in selected_l1.items()
                    ]
                    return concat(selected)

                # Get the useful wavelength data
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

                # The triplet value is the difference of the log radiances subtracted by the normalization multiplied by the weight
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
