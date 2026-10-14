"""
Line shapes for integrating a high resolution measurement down to a lower resolution one.

The instrument models (:py:class:`skretrieval.core.sensor.spectrograph.Spectrograph`,
:py:class:`skretrieval.core.sensor.spectrograph.SpectrographOnlySpectral`, the forward model
``lineshape_fn``) apply line shapes through :py:func:`measurement_matrix`, which uses ``rebasis``
(the ``pyrebasis`` package).  They accept ``rebasis.Space`` kernels directly, which is preferred for
new code: rebasis offers many more line shapes (Voigt, FTS, super-Gaussian, ...), exact integrals
and derivatives with respect to the line shape parameters.

The :py:class:`LineShape` classes here are kept for backwards compatibility.  The built-in ones
convert themselves to rebasis kernels with :py:meth:`LineShape.to_space`; subclasses that only
implement :py:meth:`LineShape.integration_weights` still work, one channel at a time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import rebasis as rb
import scipy.sparse as sp
import scipy.special as special
from numba import vectorize


class LineShape(ABC):
    """
    Base class for implementing line shapes.  A line shape represents integration across a high resolution measurement
    down to a lower resolution measurement.

    New code can use ``rebasis.Space`` kernels instead, see :py:func:`measurement_matrix`.
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
    def bounds(self, center=0):
        """
        Boundaries of the line shape.  Values outside this range are 0

        Parameters
        ----------
        center : float, optional
            Center of the line shape.  Default is 0

        Returns
        -------
        (left, right)
            Left and right boundaries of the line shape
        """

    def zero_centered(self):
        """
        True if the lineshape is centered on 0 rather than the nominal sample

        Returns
        -------
        bool
        """
        return True

    def to_space(self, centers) -> rb.Space:
        """
        The line shape as rebasis kernels, one centred at each of ``centers``

        Parameters
        ----------
        centers : array_like
            Centers of the kernels.  Ignored by line shapes that are not zero centered.

        Returns
        -------
        rebasis.Space
        """
        msg = f"{type(self).__name__} has no rebasis representation"
        raise NotImplementedError(msg)

    def _rebasis_basis(self) -> str | None:
        """
        How :py:func:`measurement_matrix` represents the high resolution samples for this line
        shape: ``"hats"`` (linear interpolation between the samples, the line shape is integrated),
        ``"deltas"`` (the line shape is sampled at the samples), ``"nearest"`` (the nearest sample
        is taken), or None to use :py:meth:`integration_weights`.
        """
        return None


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

    def bounds(self, center=0):
        """
        If integration_weights is called with mean=0, all values outside the range [lower_bound, upper_bound] are
        guaranteed to be 0.

        Parameters
        ----------
        center : float, optional
            Center of the line shape.  Default is 0

        Returns
        -------
        [lower_bound, upper_bound]
        """
        return (
            -self.max_stdev * self._stdev + center,
            self.max_stdev * self._stdev + center,
        )

    def to_space(self, centers) -> rb.Space:
        if self._stdev == 0:
            return rb.Space.deltas(centers)
        return rb.Space.gaussians(centers, sigma=self._stdev, cutoff=self.max_stdev)

    def _rebasis_basis(self) -> str | None:
        if self._stdev == 0:
            return "nearest"
        if self._mode == "linear":
            return "hats"
        if self._mode == "constant":
            return "deltas"
        msg = "mode must be one of linear or constant"
        raise ValueError(msg)

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

    return (1 - (a1 * t + a2 * t**2 + a3 * t**3) * np.exp(-np.abs(x) ** 2)) * np.sign(x)


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

    def bounds(self, center=0):
        return center, center

    def to_space(self, centers) -> rb.Space:
        return rb.Space.deltas(centers)

    def _rebasis_basis(self) -> str | None:
        return "nearest"


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

        # Difference between the sample and the center of the rectangle
        offsets = available_samples - mean
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

    def bounds(self, center=0):
        return [-self._width / 2 + center, self._width / 2 + center]

    def to_space(self, centers) -> rb.Space:
        return rb.Space.rectangles(centers, width=self._width)

    def _rebasis_basis(self) -> str | None:
        # Constant mode excludes samples exactly on either edge, rebasis boxes are half open, so
        # constant mode keeps using integration_weights
        return "hats" if self._mode == "linear" else None


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
        integration_fraction: float = 0.999,
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
        integration_fraction: float
            Fraction of the line shape to integrate.  Default is 1, which means the entire line shape is integrated.
        mode: str
            If set to 'simple', the line shape is interpolated to the sample values.  If mode is set to 'integrate'
            then the line shape is analytically integrated assuming linear interpolation over the sample values.
            If the mode is 'integrate' then the line shape samples must be evenly spaced.

        """
        self._x_values = x_values
        self._line_values = line_values
        self._zero_centered = zero_centered

        center = np.argmax(np.abs(self._line_values))
        right_integral = np.cumsum(np.abs(self._line_values[center:]))
        left_integral = np.cumsum(np.abs(self._line_values)[:center][::-1])

        try:
            self._right_cutoff = (
                np.nonzero(right_integral / right_integral[-1] >= integration_fraction)[
                    0
                ][0]
                + center
            )
        except IndexError:
            self._right_cutoff = len(self._line_values) - 1
        try:
            self._left_cutoff = (
                center
                - 1
                - np.nonzero(left_integral / left_integral[-1] >= integration_fraction)[
                    0
                ][0]
            )
        except IndexError:
            self._left_cutoff = 0

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
            line_shape_interp = self._linear_weights(available_samples, x_interp)
        else:
            msg = "UserLineShape mode must be one of simple or integrate"
            raise ValueError(msg)

        if not normalize:
            msg = "UserLineShape currently only supports normalized line shapes"
            raise ValueError(msg)

        line_shape_interp /= np.sum(line_shape_interp)

        return line_shape_interp

    def bounds(self, center=0):
        return (
            self._x_values[self._left_cutoff] + center,
            self._x_values[self._right_cutoff] + center,
        )
        return np.min(self._x_values) + center, np.max(self._x_values) + center

    def zero_centered(self):
        return self._zero_centered

    def to_space(self, centers) -> rb.Space:
        centers = np.atleast_1d(np.asarray(centers, dtype=float))
        if not self._zero_centered:
            centers = np.zeros_like(centers)
        return rb.Space.shifted(centers, self._x_values, self._line_values)

    def _rebasis_basis(self) -> str | None:
        if self._mode == "simple":
            return "deltas"
        if self._mode == "integrate":
            return "hats"
        msg = "UserLineShape mode must be one of simple or integrate"
        raise ValueError(msg)

    def _linear_weights(self, available_samples, xs):
        offsets = available_samples

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


def _defining_class(cls: type, name: str) -> type:
    return next(c for c in cls.__mro__ if name in vars(c))


def _rebasis_basis(lineshape: LineShape) -> str | None:
    cls = type(lineshape)
    # A subclass that overrides integration_weights but not the rebasis conversion, e.g. a
    # modified Gaussian, must keep using its own integration_weights
    if not issubclass(
        _defining_class(cls, "_rebasis_basis"),
        _defining_class(cls, "integration_weights"),
    ):
        return None
    return lineshape._rebasis_basis()


def measurement_matrix(
    lineshapes: rb.Space | list[LineShape | rb.Space],
    centers,
    samples,
) -> sp.csr_array:
    """
    The matrix W that integrates a high resolution function down to instrument channels,
    ``low_resolution = W @ high_resolution``.  Each row sums to 1, except for channels that do not
    overlap the samples, which are 0.

    Parameters
    ----------
    lineshapes : rebasis.Space | list[LineShape | rebasis.Space]
        A rebasis Space with one kernel per channel, or one line shape per channel, where each
        rebasis Space has a single kernel.  rebasis kernels are integrated against the linear
        interpolant of the high resolution samples, the function is 0 outside of the samples.
    centers : array_like | None
        Center of each channel, used for the :py:class:`LineShape` objects.  rebasis kernels carry
        their own centers.
    samples : array_like
        Coordinates of the high resolution samples, in any order.  Must be distinct unless all line
        shapes are sampled rather than integrated.

    Returns
    -------
    scipy.sparse.csr_array
        Shape (number of channels, len(samples))
    """
    samples = np.asarray(samples, dtype=float)
    order = np.argsort(samples, kind="stable")
    sorted_samples = samples[order]

    # Channels grouped by how the samples are represented
    kernels: dict[str, tuple[list[int], list[rb.Space]]] = {
        "hats": ([], []),
        "deltas": ([], []),
    }
    nearest: list[int] = []
    fallback: list[int] = []

    if isinstance(lineshapes, rb.Space):
        n_channels = len(lineshapes)
        kernels["hats"] = (list(range(n_channels)), [lineshapes])
    else:
        n_channels = len(lineshapes)
        if centers is None:
            centers = np.full(n_channels, np.nan)
        centers = np.broadcast_to(np.asarray(centers, dtype=float), (n_channels,))
        for i, (lineshape, center) in enumerate(zip(lineshapes, centers, strict=True)):
            if isinstance(lineshape, rb.Space):
                if len(lineshape) != 1:
                    msg = f"channel {i}: a rebasis Space per channel must have one kernel, got {len(lineshape)}"
                    raise ValueError(msg)
                basis = "hats"
            else:
                basis = _rebasis_basis(lineshape)

            if basis in kernels:
                kernel = (
                    lineshape
                    if isinstance(lineshape, rb.Space)
                    else lineshape.to_space([center])
                )
                kernels[basis][0].append(i)
                kernels[basis][1].append(kernel)
            elif basis == "nearest":
                nearest.append(i)
            else:
                fallback.append(i)

    rows, cols, values = [], [], []

    for basis, (channels, spaces) in kernels.items():
        if not channels:
            continue
        if basis == "hats":
            if np.any(np.diff(sorted_samples) <= 0):
                msg = "samples must be distinct to integrate line shapes across them"
                raise ValueError(msg)
            basis_space = rb.Space.hats(sorted_samples)
        else:
            basis_space = rb.Space.deltas(sorted_samples)
        space = spaces[0] if len(spaces) == 1 else rb.Space.concatenate(spaces)
        w = rb.measure(space, basis_space, "rowsum").tocoo()
        rows.append(np.asarray(channels)[w.row])
        cols.append(order[w.col])
        values.append(w.data)

    if nearest:
        c = centers[nearest]
        right = np.searchsorted(sorted_samples, c).clip(0, len(sorted_samples) - 1)
        left = (right - 1).clip(0)
        # Ties go to the lower sample
        closest = np.where(
            np.abs(c - sorted_samples[left]) <= np.abs(sorted_samples[right] - c),
            left,
            right,
        )
        rows.append(np.asarray(nearest))
        cols.append(order[closest])
        values.append(np.ones(len(nearest)))

    for i in fallback:
        weights = lineshapes[i].integration_weights(centers[i], samples)
        weights = weights / weights.sum()
        (nonzero,) = np.nonzero(weights)
        rows.append(np.full(len(nonzero), i))
        cols.append(nonzero)
        values.append(weights[nonzero])

    if not rows:
        return sp.csr_array((n_channels, len(samples)))
    return sp.csr_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_channels, len(samples)),
    )
