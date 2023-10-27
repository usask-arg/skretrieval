from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.special as special
from numba import vectorize


class LineShape(ABC):
    """
    Base class for implementing line shapes.  A line shape represents integration across a high resolution measurement
    down to a lower resolution measurement.
    """

    @abstractmethod
    def integration_weights(
        self, mean: float, available_samples: np.ndarray, normalize=True
    ):
        """
        Integration weights for the line shape.

        Parameters
        ----------
        mean : float
            Value to integrate to

        available_samples : np.ndarray
            Array of sample values that are available to use in the integration.

        normalize : bool, Optional
            If true, resulting weights are normalized such that np.nansum(weights) = 1

        Returns
        -------
        np.ndarray
            Integration weights, same size as available_samples.
        """

    @abstractmethod
    def bounds(self):
        """
        Boundaries of the line shape.  Values outside this range are 0

        Returns
        -------
        (left, right)
            Left and right boundaries of the line shape
        """


class Gaussian(LineShape):
    def __init__(
        self,
        fwhm: float | None = None,
        stdev: float | None = None,
        max_stdev=5,
        mode="linear",
    ):
        """
        Gaussian line shape.

        Parameters
        ----------
        fwhm : float
            Full width half maximum, only specify one of fwhm or stdev

        stdev : float
            Standard deviation, only specify one of fwhm or stdev

        max_stdev: int, optional
            Values farther than max_stdev*stdev are truncated to 0.  Default=5

        mode : string, one of ['constant', 'linear']
             If constant, then the gaussian is sampled at the integration location.  If linear, then the gaussian
             is integrated with a triangular base function representing linear integration across the sample.  Linear
             is much more accurate at the cost of a small performance hit.  Default 'linear'.
        """

        self._fwhm_to_stdev = 1 / (2 * np.sqrt(2 * np.log(2)))

        if fwhm is not None and stdev is not None:
            msg = "Only one of fwhm or stdev should be specified"
            raise ValueError(msg)

        if fwhm is None and stdev is None:
            msg = "One of fwhm or stdev needs to be specified"
            raise ValueError(msg)

        if fwhm is not None:
            self._stdev = fwhm * self._fwhm_to_stdev
        else:
            self._stdev = stdev

        self.max_stdev = max_stdev

        self._mode = mode

    def integration_weights(
        self, mean: float, available_samples: np.ndarray, normalize=True
    ):
        """
        The lineshape converts a function at high resolution, H, to one at low resolution, L.

        L(mean) = np.dot(H, integration_weights(mean, available_samples))

        Parameters
        ----------
        mean : float
            value (in the range of available_samples) to integrate to
        available_samples : np.ndarray
            Values that the high resolution function is available at
        normalize : bool, optional
            If true, the returned weights will sum to 1

        Returns
        -------
        np.ndarray
            Weights to use in the high resolution integration
        """
        if self._stdev == 0:
            # Special case, just interpolate to the mean
            # TODO: This is nearest neighbor, switch to interpolate
            weights = np.zeros_like(available_samples)

            weights[np.argmin(np.abs(mean - available_samples))] = 1

            return weights

        difference = mean - available_samples

        if self._mode == "constant":
            gaussian = self._analytic_constant_weights(mean, available_samples)
        elif self._mode == "linear":
            gaussian = self._analytic_linear_weights(mean, available_samples)
        else:
            msg = "mode must be one of linear or constant"
            raise ValueError(msg)

        gaussian[np.abs(difference) > self.max_stdev * self._stdev] = 0
        # Sometimes numerical funniness can cause negative values
        # TODO: Check if this should be abs or set to 0
        gaussian = np.abs(gaussian)

        if normalize:
            if np.abs(np.sum(gaussian)) > 0:
                gaussian /= np.sum(gaussian)
            else:
                # Should be all zeros, indicating no contribution to the point which is okay
                # Just to be sure
                gaussian = np.zeros_like(gaussian)

        return gaussian

    def bounds(self):
        """
        If integration_weights is called with mean=0, all values outside the range [lower_bound, upper_bound] are
        guaranteed to be 0.

        Returns
        -------
        [lower_bound, upper_bound]
        """
        return -self.max_stdev * self._stdev, self.max_stdev * self._stdev

    def _analytic_linear_weights(self, mean, available_samples):
        return _gaussian_analytic_linear_weights(
            self.max_stdev, self._stdev, mean, np.ascontiguousarray(available_samples)
        )

    def _analytic_constant_weights(self, mean, available_samples):
        difference = mean - available_samples

        return np.exp(-0.5 * (difference / self._stdev) ** 2)


@vectorize("f8(f8)", nopython=True)
def fasterf1(x):
    """
    Fast error function approximation
    """
    p = 0.47047
    a1 = 0.3480242
    a2 = -0.0958798
    a3 = 0.7478556

    t = 1 / (1 + p * np.abs(x))

    return (
        1 - (a1 * t + a2 * t**2 + a3 * t**3) * np.exp(-np.abs(x) ** 2)
    ) * np.sign(x)


@vectorize("f8(f8)", nopython=True)
def fasterf2(x):
    """
    Fast error function approximation
    """
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1 / (1 + p * np.abs(x))

    return (
        1
        - (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5)
        * np.exp(-np.abs(x) ** 2)
    ) * np.sign(x)


def fasterf3(x):
    """
    Fast error function approximation
    """
    return special.erf(x)


@vectorize("f8(f8, f8, f8, f8)")
def _gaussian_analytic_linear_weights_helper(width_left, width_right, offsets, stdev):
    """
    See theory/line_shape_integrals.nb for explanation
    """
    stdev /= np.sqrt(0.5)

    return (
        0.5
        * stdev
        * (
            stdev
            * (
                np.exp(-((offsets - width_left) ** 2) / stdev**2) / width_left
                + np.exp(-((offsets + width_right) ** 2) / stdev**2) / width_right
                - np.exp(-(offsets**2) / stdev**2)
                * (width_left + width_right)
                / (width_left * width_right)
            )
            + 1
            / (width_left * width_right)
            * np.sqrt(np.pi)
            * (
                -offsets * (width_right + width_left) * fasterf1(offsets / stdev)
                + (offsets - width_left)
                * width_right
                * fasterf1((offsets - width_left) / stdev)
                + width_left
                * (offsets + width_right)
                * fasterf1((offsets + width_right) / stdev)
            )
        )
    )


def _gaussian_analytic_linear_weights(max_stdev, stdev, mean, available_samples):
    """
    Calculates the weights by integrating the gaussian form with a triangle function
    """
    weights = np.zeros_like(available_samples)

    # Difference between center of the gaussian and the sample
    offsets = available_samples - mean
    within_stdev = np.abs(offsets) < max_stdev * stdev

    # Interpolation width on the left side of each sample
    widths = np.diff(available_samples)
    width_left = np.abs(np.hstack((np.array([widths[0]]), widths)))

    # Interpolation width on the right side of each sample
    width_right = np.abs(np.hstack((widths, np.array([widths[-1]]))))

    offsets = offsets[within_stdev]
    width_left = width_left[within_stdev]
    width_right = width_right[within_stdev]

    # Formula derived by integrating a triangle function with a gaussian
    weights[within_stdev] = _gaussian_analytic_linear_weights_helper(
        width_left, width_right, offsets, stdev
    )

    return weights


class DeltaFunction(LineShape):
    def __init__(self):
        """
        DeltaFunction line shape.  The nearest sample is always taken.

        """

    def integration_weights(
        self,
        mean: float,
        available_samples: np.ndarray,
        normalize=True,  # noqa: ARG002
        tolerance=1e-7,
    ):
        # Interpolate to the mean value
        weights = np.zeros_like(available_samples)

        if np.sum(np.abs(mean - available_samples) < tolerance) > 1:
            weights[np.where(np.abs(mean - available_samples) < tolerance)[0]] = 1
        else:
            weights[np.argmin(np.abs(mean - available_samples))] = 1

        return weights

    def bounds(self):
        return 0, 0


class Rectangle(LineShape):
    def __init__(self, width, mode="linear"):
        """
        Rectangular line shape

        Parameters
        ----------
        width : float
            Full width of the line shape.
        """
        self._width = width
        self._mode = mode

    def integration_weights(
        self, mean: float, available_samples: np.ndarray, normalize=True
    ):
        if self._mode == "linear":
            weights = self._analytic_linear_weights(mean, available_samples)
        else:
            weights = np.zeros_like(available_samples)

            offsets = mean - available_samples

            weights[np.abs(offsets) < self._width / 2] = 1

        if normalize:
            if np.nansum(weights) == 0:
                raise ValueError()
            weights /= np.nansum(weights)

        return weights

    def _analytic_linear_weights(self, mean, available_samples):
        """
        See theory/line_shape_integrals.nb for more information.  Linear weights are calculated by integrating
        the rectangle function with a triangle function.
        """
        weights = np.zeros_like(available_samples)

        # Difference between center of the gaussian and the sample
        offsets = mean - available_samples
        # within_width = np.abs(offsets) < self._width

        # Interpolation width on the left side of each sample
        widths = np.diff(available_samples)

        width_left = np.abs(np.hstack((np.array([widths[0]]), widths)))

        # Interpolation width on the right side of each sample
        width_right = np.abs(np.hstack((widths, np.array([widths[-1]]))))

        # offsets = offsets[within_width]
        # width_left = width_left[within_width]
        # width_right = width_right[within_width]

        weights += _rectangle_analytic_linear_weights_helper_left(
            width_left, offsets, self._width
        )
        weights += _rectangle_analytic_linear_weights_helper_right(
            width_right, offsets, self._width
        )

        return weights

    def bounds(self):
        return [-self._width / 2, self._width / 2]


@vectorize("f8(f8, f8, f8)", nopython=True)
def _rectangle_analytic_linear_weights_helper_left(width_left, offset, rect_width):
    if offset == 0 and rect_width < 2 * width_left:
        return -rect_width * (rect_width - 4 * width_left) / (8 * width_left)
    elif (  # noqa: RET505
        offset > 0
        and 2 * offset + rect_width < 2 * width_left
        and 2 * offset < rect_width
    ) or (
        offset < 0
        and 2 * offset + rect_width > 0
        and 2 * offset + rect_width < 2 * width_left
    ):
        return (
            -(2 * offset + rect_width)
            * (2 * offset + rect_width - 4 * width_left)
            / (8 * width_left)
        )
    elif (
        offset > 0
        and 2 * offset + rect_width <= 2 * width_left
        and (2 * offset == rect_width or (rect_width > 0 and 2 * offset >= rect_width))
    ):
        return rect_width - offset * rect_width / width_left
    elif width_left > 0 and (
        (offset == 0 and rect_width > 2 * width_left)
        or (2 * offset + rect_width >= 2 * width_left and offset < 0)
        or (
            offset > 0
            and 2 * offset + rect_width > 2 * width_left
            and 2 * offset < rect_width
        )
    ):
        return width_left / 2
    elif (
        offset > 0
        and 2 * offset + rect_width > 2 * width_left
        and (
            (2 * offset == rect_width and width_left < 0)
            or (2 * offset > rect_width and 2 * offset < rect_width + 2 * width_left)
        )
    ):
        return (-2 * offset + rect_width + 2 * width_left) ** 2 / (8 * width_left)
    else:
        return 0


@vectorize("f8(f8, f8, f8)", nopython=True)
def _rectangle_analytic_linear_weights_helper_right(width_right, offset, rect_width):
    if width_right > 0 and (
        (offset == 0 and rect_width > 0 and rect_width > 2 * width_right)
        or (
            2 * (offset + width_right) < rect_width
            and (
                (2 * offset + rect_width > 0 and offset < 0)
                or (offset > 0 and 2 * offset < rect_width)
            )
        )
    ):
        return width_right / 2
    elif offset < 0 and (  # noqa: RET505
        (2 * offset + rect_width == 0 and 2 * (offset + width_right) >= rect_width)
        or (
            rect_width > 0
            and 2 * (offset + width_right) > rect_width
            and 2 * offset + rect_width < 0
        )
    ):
        return rect_width * (offset + width_right) / width_right
    elif offset == 0 and rect_width > 0 and rect_width <= 2 * width_right:
        return -rect_width * (rect_width - 4 * width_right) / (8 * width_right)
    elif 2 * (offset + width_right) >= rect_width and (
        (2 * offset + rect_width > 0 and offset < 0)
        or (offset > 0 and 2 * offset < rect_width)
    ):
        return (
            -(2 * offset - rect_width)
            * (2 * offset - rect_width + 4 * width_right)
            / (8 * width_right)
        )
    elif (
        2 * offset + rect_width + 2 * width_right > 0
        and offset < 0
        and (
            (2 * offset + rect_width == 0 and 2 * (offset + width_right) < rect_width)
            or (
                rect_width > 0
                and 2 * offset + rect_width < 0
                and 2 * (offset + width_right) <= rect_width
            )
        )
    ):
        return (2 * offset + rect_width + 2 * width_right) ** 2 / (8 * width_right)
    else:
        return 0


class UserLineShape(LineShape):
    def __init__(
        self,
        x_values: np.array,
        line_values: np.array,
        zero_centered: bool,
        mode="simple",
    ):
        """
        Line shape created from a user specified function

        Parameters
        ----------
        x_values: np.array
            x values for the lineshape, could be wavelength, could be angle, etc.
        line_values: np.array
            Values for the line shape.  Same size as x_values.  Any values outside the range of x_values are assumed
            to be 0.
        zero_centered: bool
            True if the line shape values are centered at 0, false if the line shape is not centered.
        mode: str
            If set to 'simple', the line shape is interpolated to the sample values.  If mode is set to 'integrate'
            then the line shape is analytically integrated assuming linear interpolation over the sample values.
            If the mode is 'integrate' then the line shape samples must be evenly spaced.

        """
        self._x_values = x_values
        self._line_values = line_values
        self._zero_centered = zero_centered

        self._mode = mode
        if self._mode == "integrate":
            self._widths_left = np.zeros_like(self._x_values, dtype=np.float64)
            self._widths_right = np.zeros_like(self._x_values, dtype=np.float64)

            self._widths_left[1:] = self._x_values[1:] - self._x_values[:-1]
            self._widths_left[0] = 1e99
            self._widths_right[:-1] = self._x_values[1:] - self._x_values[:-1]
            self._widths_right[-1] = 1e99

    def integration_weights(
        self, mean: float, available_samples: np.ndarray, normalize=True
    ):
        # interpolate the line shape to the available samples

        x_interp = self._x_values + mean if self._zero_centered else self._x_values

        if self._mode == "simple":
            line_shape_interp = np.interp(
                available_samples, x_interp, self._line_values, left=0, right=0
            )
        elif self._mode == "integrate":
            line_shape_interp = self._linear_weights(mean, available_samples, x_interp)
        else:
            msg = "UserLineShape mode must be one of simple or integrate"
            raise ValueError(msg)

        if not normalize:
            msg = "UserLineShape currently only supports normalized line shapes"
            raise ValueError(msg)

        line_shape_interp /= np.sum(line_shape_interp)

        return line_shape_interp

    def bounds(self):
        return np.min(self._x_values), np.max(self._x_values)

    def _linear_weights(self, mean, available_samples, xs):
        offsets = mean - available_samples if self._zero_centered else available_samples

        # Interpolation width on the left side of each sample
        widths = np.diff(available_samples)

        width_left = np.abs(np.hstack((np.array([widths[0]]), widths)))

        # Interpolation width on the right side of each sample
        width_right = np.abs(np.hstack((widths, np.array([widths[-1]]))))

        # Have to sum over all internal x values
        weights = np.zeros_like(available_samples)
        temp = np.zeros_like(available_samples)
        for x, ls, wl, wr in zip(
            xs, self._line_values, self._widths_left, self._widths_right
        ):
            weights += (
                _triangle_analytic_linear_weights_helper2(
                    wr, -1 * (offsets - x), width_right
                )
                * ls
            )
            # Integral should be symmetric around x->-x and offset->-offset so use same helper for both cases
            weights += (
                _triangle_analytic_linear_weights_helper2(wl, offsets - x, width_left)
                * ls
            )

            temp += (
                _triangle_analytic_linear_weights_helper2(
                    wr, -1 * (offsets - x), width_right
                )
                * ls
            )
            temp += (
                _triangle_analytic_linear_weights_helper2(wl, offsets - x, width_left)
                * ls
            )

        return weights


@vectorize("f8(f8, f8, f8)", nopython=True)
def _triangle_analytic_linear_weights_helper(width_right, offset, width):
    if offset < 0 and (offset + width_right) > width:
        return (
            -1.0
            * (width * (-3.0 * offset + width - 3.0 * width_right))
            / (6.0 * width_right)
        )
    elif (  # noqa: RET505
        offset < 0 and (offset + width_right) > 0 and (offset + width_right) <= width
    ):
        return (
            -1.0
            * (offset + width_right) ** 2
            * (offset - 3.0 * width + width_right)
            / (6.0 * width * width_right)
        )
    elif (offset + width_right) > width and (
        offset == 0 or ((width > offset) and offset > 0)
    ):
        return (
            (offset - width) ** 2
            * (offset - width + 3.0 * width_right)
            / (6.0 * width * width_right)
        )
    elif (offset + width_right) <= width and offset >= 0:
        return (
            -1.0
            * width_right
            * (3.0 * offset - 3.0 * width + width_right)
            / (6.0 * width)
        )
    else:
        return 0.0


@vectorize("f8(f8, f8, f8)", nopython=True)
def _triangle_analytic_linear_weights_helper2(width_right, offset, width):
    if (
        offset < 0
        and offset + width_right <= width
        and (
            (offset + width == 0 and offset + width_right < 0)
            or (
                offset + width < 0
                and (
                    offset + width_right == 0
                    and (offset + width + width_right > 0 and offset + width_right <= 0)
                )
            )
        )
    ):
        return (offset + width + width_right) ** 3 / (6 * width * width_right)
    elif offset + width_right < 0 and offset + width > 0:  # noqa: RET505
        return (width_right * (3 * (offset + width) + width_right)) / (6 * width)
    elif offset > 0 and offset < width and offset + width_right > width:
        return ((offset - width) ** 2 * (offset - width + 3 * width_right)) / (
            6 * width * width_right
        )
    elif offset + width_right > width and offset + width <= 0:
        return (width * (offset + width_right)) / width_right
    elif offset == 0 and width < width_right:
        return -1 * (width * (width - 3 * width_right)) / (6 * width_right)
    elif offset + width == 0 and offset + width_right == 0:
        return width_right / 6
    elif offset + width_right == 0 and offset + width > 0:
        return ((3 * width - 2 * width_right) * width_right) / (6 * width)
    elif offset < 0 and offset + width > 0 and offset + width_right > width:
        return (
            -1
            * (
                offset**3
                + width**2 * (width - 3 * width_right)
                - 3 * offset * width * (width - 2 * width_right)
                + 3 * offset**2 * (width + width_right)
            )
            / (6 * width * width_right)
        )
    elif offset == 0 and width >= width_right:
        return ((3 * width - width_right) * width_right) / (6 * width)
    elif offset > 0 and offset + width_right <= width:
        return -1 * (width_right * (3 * offset - 3 * width + width_right)) / (6 * width)
    elif (
        offset < 0
        and offset + width_right > 0
        and offset + width > 0
        and offset + width_right <= width
    ):
        return (
            -1
            * (
                2 * offset**3
                + 6 * offset**2 * width_right
                + 3 * (offset - width) * width_right**2
                + width_right**3
            )
            / (6 * width * width_right)
        )
    elif (
        offset + width <= 0
        and offset + width_right > 0
        and offset + width_right <= width
    ):
        return (
            -(offset**3)
            + width**3
            + 3 * offset**2 * (width - width_right)
            + 3 * width**2 * width_right
            + 3 * width * width_right**2
            - width_right**3
            + 3 * offset * (width**2 + 2 * width * width_right - width_right**2)
        ) / (6 * width * width_right)
    else:
        return 0.0


def _triangle_analytic_linear_weights_helper3(width_right, offset, width):
    if (
        offset < 0
        and offset + width_right <= width
        and (
            (offset + width == 0 and offset + width_right < 0)
            or (
                offset + width < 0
                and (
                    offset + width_right == 0
                    and (offset + width + width_right > 0 and offset + width_right <= 0)
                )
            )
        )
    ):
        return (offset + width + width_right) ** 3 / (6 * width * width_right)
    elif offset + width_right < 0 and offset + width > 0:  # noqa: RET505
        return (width_right * (3 * (offset + width) + width_right)) / (6 * width)
    elif offset > 0 and offset < width and offset + width_right > width:
        return ((offset - width) ** 2 * (offset - width + 3 * width_right)) / (
            6 * width * width_right
        )
    elif offset + width_right > width and offset + width <= 0:
        return (width * (offset + width_right)) / width_right
    elif offset == 0 and width < width_right:
        return -1 * (width * (width - 3 * width_right)) / (6 * width_right)
    elif offset + width == 0 and offset + width_right == 0:
        return width_right / 6
    elif offset + width_right == 0 and offset + width > 0:
        return ((3 * width - 2 * width_right) * width_right) / (6 * width)
    elif offset < 0 and offset + width > 0 and offset + width_right > width:
        return (
            -1
            * (
                offset**3
                + width**2 * (width - 3 * width_right)
                - 3 * offset * width * (width - 2 * width_right)
                + 3 * offset**2 * (width + width_right)
            )
            / (6 * width * width_right)
        )
    elif offset == 0 and width >= width_right:
        return ((3 * width - width_right) * width_right) / (6 * width)
    elif offset > 0 and offset + width_right <= width:
        return -1 * (width_right * (3 * offset - 3 * width + width_right)) / (6 * width)
    elif (
        offset < 0
        and offset + width_right > 0
        and offset + width > 0
        and offset + width_right <= width
    ):
        return (
            -1
            * (
                2 * offset**3
                + 6 * offset**2 * width_right
                + 3 * (offset - width) * width_right**2
                + width_right**3
            )
            / (6 * width * width_right)
        )
    elif (
        offset + width <= 0
        and offset + width_right > 0
        and offset + width_right <= width
    ):
        return (
            -(offset**3)
            + width**3
            + 3 * offset**2 * (width - width_right)
            + 3 * width**2 * width_right
            + 3 * width * width_right**2
            - width_right**3
            + 3 * offset * (width**2 + 2 * width * width_right - width_right**2)
        ) / (6 * width * width_right)
    else:
        return 0.0
