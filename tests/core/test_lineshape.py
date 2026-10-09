from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, special

from skretrieval.core.lineshape import (
    DeltaFunction,
    Gaussian,
    Rectangle,
    UserLineShape,
    fasterf1,
    fasterf2,
    fasterf3,
)

FWHM_PER_STDEV = 2 * np.sqrt(2 * np.log(2))


def _nonuniform_grid(n=80, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.cumsum(rng.uniform(0.05, 0.3, n))
    return grid - grid.mean()


def _hat(grid, i, x):
    """Linear-interpolation basis function of sample ``i`` of ``grid`` evaluated at ``x``."""
    unit = np.zeros(len(grid))
    unit[i] = 1.0
    return np.interp(x, grid, unit, left=0.0, right=0.0)


def _quad_weights(f, grid):
    """Reference weights int f(x) * hat_i(x) dx computed with adaptive quadrature."""
    weights = np.zeros(len(grid))
    for i in range(1, len(grid) - 1):
        for a, b in ((grid[i - 1], grid[i]), (grid[i], grid[i + 1])):
            weights[i] += integrate.quad(lambda x, i=i: f(x) * _hat(grid, i, x), a, b)[
                0
            ]
    return weights


def _exact_rectangle_weights(grid, lower, upper):
    """
    Exact int_lower^upper hat_i(x) dx.  hat_i is linear between the points used, so the trapezoid
    rule is exact.
    """
    pts = np.union1d(grid[(grid > lower) & (grid < upper)], [lower, upper])
    return np.array(
        [integrate.trapezoid(_hat(grid, i, pts), pts) for i in range(len(grid))]
    )


def _exact_piecewise_linear_weights(f, grid, breakpoints):
    """
    Exact int f(x) * hat_i(x) dx for a continuous f that is linear between ``breakpoints``.
    The integrand is quadratic between consecutive knots so Simpson's rule is exact there.
    """
    knots = np.union1d(grid, breakpoints)
    knots = knots[(knots >= grid[0]) & (knots <= grid[-1])]
    a, b = knots[:-1], knots[1:]
    m = 0.5 * (a + b)
    weights = []
    for i in range(len(grid)):

        def g(x, i=i):
            return f(x) * _hat(grid, i, x)

        weights.append(np.sum((b - a) / 6 * (g(a) + 4 * g(m) + g(b))))
    return np.array(weights)


# ---------------------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fwhm": 1.0, "stdev": 1.0}, "Only one of fwhm or stdev"),
        ({}, "One of fwhm or stdev needs to be specified"),
    ],
)
def test_gaussian_requires_exactly_one_width(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Gaussian(**kwargs)


@pytest.mark.parametrize("mode", ["linear", "constant"])
def test_gaussian_fwhm_and_stdev_are_equivalent(mode):
    samples = np.arange(-5, 5.001, 0.1)
    fwhm = 1.3

    from_fwhm = Gaussian(fwhm=fwhm, mode=mode).integration_weights(0.07, samples)
    from_stdev = Gaussian(stdev=fwhm / FWHM_PER_STDEV, mode=mode).integration_weights(
        0.07, samples
    )

    np.testing.assert_allclose(from_fwhm, from_stdev, rtol=1e-12, atol=1e-15)


def test_gaussian_constant_mode_is_half_maximum_at_half_fwhm():
    lineshape = Gaussian(fwhm=2.0, mode="constant")

    weights = lineshape.integration_weights(
        0.0, np.array([-1.0, 0.0, 1.0]), normalize=False
    )

    np.testing.assert_allclose(weights, [0.5, 1.0, 0.5])


def test_gaussian_constant_mode_samples_gaussian():
    stdev = 0.8
    samples = np.linspace(-3, 3, 41)

    weights = Gaussian(stdev=stdev, mode="constant").integration_weights(
        0.25, samples, normalize=False
    )

    np.testing.assert_allclose(weights, np.exp(-0.5 * ((samples - 0.25) / stdev) ** 2))


@pytest.mark.parametrize("mode", ["linear", "constant"])
@pytest.mark.parametrize("mean", [0.0, 0.123, -1.3, 6.5])
def test_gaussian_weights_are_normalized(mode, mean):
    # mean=6.5 puts part of the line beyond the end of the grid
    samples = _nonuniform_grid()

    weights = Gaussian(stdev=0.7, mode=mode).integration_weights(mean, samples)

    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=1e-12)
    assert np.all(weights >= 0)


def test_gaussian_linear_unnormalized_weights_integrate_gaussian():
    stdev = 0.7

    weights = Gaussian(stdev=stdev).integration_weights(
        0.123, _nonuniform_grid(), normalize=False
    )

    # Hat functions form a partition of unity, so the weights sum to the Gaussian's integral
    np.testing.assert_allclose(np.sum(weights), stdev * np.sqrt(2 * np.pi), rtol=1e-6)


@pytest.mark.parametrize("mean", [0.0, 0.123, -1.3])
def test_gaussian_linear_weights_match_quadrature(mean):
    stdev = 0.7
    samples = _nonuniform_grid()

    weights = Gaussian(stdev=stdev).integration_weights(mean, samples, normalize=False)
    expected = _quad_weights(
        lambda x: np.exp(-0.5 * ((x - mean) / stdev) ** 2), samples
    )

    # Tolerance is set by the fast erf approximation used internally
    np.testing.assert_allclose(weights, expected, atol=2e-4)


def test_gaussian_linear_mode_reproduces_linear_signal():
    # A linear signal L(x) = x integrated against the lineshape must return the line centre.
    # Linear mode does this (up to the erf approximation) even when the grid spacing is
    # larger than the line width, constant mode does not
    coarse = np.arange(-6, 6.001, 1.0)
    mean = 0.3

    linear = Gaussian(stdev=0.5).integration_weights(mean, coarse)
    constant = Gaussian(stdev=0.5, mode="constant").integration_weights(mean, coarse)

    np.testing.assert_allclose(linear @ coarse, mean, atol=1e-3)
    assert abs(constant @ coarse - mean) > 1e-2

    samples = _nonuniform_grid()
    for m in [0.0, 0.123, -1.3]:
        weights = Gaussian(stdev=0.7).integration_weights(m, samples)
        np.testing.assert_allclose(weights @ samples, m, atol=1e-5)


@pytest.mark.parametrize("mode", ["linear", "constant"])
def test_gaussian_weights_are_symmetric_about_centre(mode):
    mean = 2.5
    samples = mean + np.arange(-40, 41) * 0.125

    weights = Gaussian(fwhm=1.7, mode=mode).integration_weights(mean, samples)

    np.testing.assert_allclose(weights, weights[::-1], rtol=1e-12, atol=1e-15)
    assert np.argmax(weights) == 40


def test_gaussian_linear_converges_to_constant_on_fine_grid():
    stdev = 0.7
    samples = np.arange(-5, 5.0001, stdev / 20)

    linear = Gaussian(stdev=stdev).integration_weights(0.01, samples)
    constant = Gaussian(stdev=stdev, mode="constant").integration_weights(0.01, samples)

    np.testing.assert_allclose(linear, constant, atol=1e-3 * constant.max())


@pytest.mark.parametrize("mode", ["linear", "constant"])
def test_gaussian_truncated_beyond_max_stdev(mode):
    samples = np.arange(-10, 10.001, 0.25)
    lineshape = Gaussian(stdev=1.0, max_stdev=3, mode=mode)

    weights = lineshape.integration_weights(0.0, samples, normalize=False)

    assert np.all(weights[np.abs(samples) > 3.0] == 0)
    assert np.all(weights[np.abs(samples) < 3.0] > 0)


@pytest.mark.parametrize("mode", ["linear", "constant"])
def test_gaussian_line_entirely_outside_grid_gives_zero_weights(mode):
    samples = np.linspace(0, 10, 101)

    weights = Gaussian(stdev=0.5, mode=mode).integration_weights(50.0, samples)

    np.testing.assert_array_equal(weights, np.zeros_like(samples))


@pytest.mark.parametrize("mean", [1.3, 1.7, -0.4, 2.6])
def test_gaussian_zero_width_selects_nearest_sample(mean):
    samples = np.array([0.0, 1.0, 2.0])

    weights = Gaussian(stdev=0).integration_weights(mean, samples)

    np.testing.assert_array_equal(
        weights, DeltaFunction().integration_weights(mean, samples)
    )
    assert weights.sum() == 1


def test_gaussian_invalid_mode_raises():
    lineshape = Gaussian(stdev=1.0, mode="cubic")

    with pytest.raises(ValueError, match="mode must be one of linear or constant"):
        lineshape.integration_weights(0.0, np.linspace(-5, 5, 11))


def test_gaussian_bounds():
    lineshape = Gaussian(stdev=2.0, max_stdev=4)
    samples = np.arange(-20, 20.001, 0.5)

    lower, upper = lineshape.bounds()
    weights = lineshape.integration_weights(0.0, samples)

    np.testing.assert_allclose((lower, upper), (-8.0, 8.0))
    assert np.all(weights[(samples < lower) | (samples > upper)] == 0)
    assert lineshape.zero_centered()


def test_gaussian_bounds_follow_center():
    lineshape = Gaussian(stdev=2.0, max_stdev=4)

    np.testing.assert_allclose(lineshape.bounds(center=10.0), (2.0, 18.0))


def test_gaussian_linear_weights_independent_of_uniform_grid_order():
    samples = np.arange(-5, 5.001, 0.2)
    lineshape = Gaussian(stdev=0.6)

    forward = lineshape.integration_weights(0.37, samples)
    backward = lineshape.integration_weights(0.37, samples[::-1])

    # Tail weights differ at the 1e-15 level across platforms (x86 vs arm64)
    np.testing.assert_allclose(backward, forward[::-1], rtol=1e-12, atol=1e-12)


@pytest.mark.xfail(
    reason="Linear-mode left/right interpolation widths are taken in array order, so a "
    "descending non-uniform grid gets mirrored hat functions (lineshape.py:289-293)",
    strict=True,
)
def test_gaussian_linear_weights_independent_of_nonuniform_grid_order():
    samples = _nonuniform_grid()
    lineshape = Gaussian(stdev=0.7)

    forward = lineshape.integration_weights(0.123, samples)
    backward = lineshape.integration_weights(0.123, samples[::-1])

    np.testing.assert_allclose(backward, forward[::-1], atol=1e-10)


@pytest.mark.parametrize(
    ("fn", "atol"),
    [
        (fasterf1, 2.5e-5),  # Abramowitz & Stegun 7.1.25
        (fasterf2, 1.5e-7),  # Abramowitz & Stegun 7.1.26
        (fasterf3, 1e-15),
    ],
)
def test_fast_erf_approximations(fn, atol):
    x = np.linspace(-6, 6, 2001)

    np.testing.assert_allclose(fn(x), special.erf(x), atol=atol)
    np.testing.assert_allclose(fn(-x), -fn(x), atol=1e-15)


# ---------------------------------------------------------------------------------------
# DeltaFunction
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mean", "expected_index"),
    [(0.0, 0), (0.45, 1), (0.95, 2), (3.0, 3), (-7.0, 0), (100.0, 3)],
)
def test_delta_function_selects_nearest_sample(mean, expected_index):
    samples = np.array([0.0, 0.7, 1.0, 2.0])
    expected = np.zeros(4)
    expected[expected_index] = 1.0

    weights = DeltaFunction().integration_weights(mean, samples)

    np.testing.assert_array_equal(weights, expected)


def test_delta_function_unsorted_samples():
    samples = np.array([5.0, 1.0, 3.0, 2.0])

    weights = DeltaFunction().integration_weights(2.9, samples)

    np.testing.assert_array_equal(weights, [0, 0, 1, 0])


def test_delta_function_weights_all_coincident_samples():
    samples = np.array([0.0, 1.0, 1.0 + 5e-8, 2.0])

    weights = DeltaFunction().integration_weights(1.0, samples)

    assert np.all(weights[[0, 3]] == 0)
    assert weights[1] > 0
    assert weights[1] == weights[2]


@pytest.mark.parametrize("center", [0.0, -3.5, 12.0])
def test_delta_function_bounds(center):
    lineshape = DeltaFunction()

    assert lineshape.bounds(center) == (center, center)
    assert lineshape.zero_centered()


# ---------------------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------------------


def test_rectangle_constant_mode_weights_samples_inside_equally():
    samples = np.arange(-8, 9) * 0.125

    weights = Rectangle(0.9, mode="constant").integration_weights(0.0, samples)

    inside = np.abs(samples) < 0.45
    assert inside.sum() == 7
    np.testing.assert_allclose(weights[inside], 1 / 7)
    assert np.all(weights[~inside] == 0)


@pytest.mark.parametrize("width", [0.05, 0.13, 0.25, 0.5, 1.7])
@pytest.mark.parametrize("mean", [0.0, 0.0371, -0.81, 0.3])
def test_rectangle_linear_matches_exact_integral_uniform_grid(width, mean):
    # Includes rectangles narrower and wider than the grid spacing.  None of these put a
    # rectangle edge exactly on a sample, see the xfail test below for that case
    samples = np.linspace(-5, 5, 101)

    weights = Rectangle(width).integration_weights(mean, samples, normalize=False)
    expected = _exact_rectangle_weights(samples, mean - width / 2, mean + width / 2)

    np.testing.assert_allclose(weights, expected, atol=1e-12)
    np.testing.assert_allclose(weights.sum(), width, rtol=1e-12)
    np.testing.assert_allclose(weights @ samples / weights.sum(), mean, atol=1e-12)


def test_rectangle_linear_weights_are_normalized():
    samples = np.linspace(-5, 5, 101)

    weights = Rectangle(0.73).integration_weights(0.21, samples)

    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12)
    assert np.all(weights >= 0)


def test_rectangle_linear_matches_exact_integral_nonuniform_grid():
    samples = np.array([0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0])

    weights = Rectangle(1.0).integration_weights(2.25, samples, normalize=False)

    # Hand computed: int_{1.75}^{2.75} hat_i(x) dx
    np.testing.assert_allclose(weights, [0, 0.03125, 0.46875, 0.4375, 0.0625, 0, 0])


@pytest.mark.xfail(
    reason="Rectangle linear-mode piecewise helpers have gaps at exact equalities "
    "(2*offset == rect_width, rect_width == 2*width) and return 0 when a rectangle edge "
    "lands exactly on a sample (lineshape.py:404-447)",
    strict=True,
)
def test_rectangle_linear_edges_on_samples():
    # 1 nm boxcar centred on a sample of a 0.5 nm grid: edges fall exactly on samples
    samples = np.arange(498, 502.01, 0.5)

    weights = Rectangle(1.0).integration_weights(500.0, samples, normalize=False)

    # Hand computed: int_{499.5}^{500.5} hat_i(x) dx
    np.testing.assert_allclose(weights, [0, 0, 0, 0.25, 0.5, 0.25, 0, 0, 0])


@pytest.mark.parametrize(
    ("mode", "width", "mean"),
    [("constant", 0.05, 0.25), ("linear", 1.0, 100.0)],
)
def test_rectangle_raises_when_no_samples_contribute(mode, width, mean):
    samples = np.arange(0, 10, 0.5)

    # Source raises a bare ValueError without a message
    with pytest.raises(ValueError):  # noqa: PT011
        Rectangle(width, mode=mode).integration_weights(mean, samples)


@pytest.mark.parametrize("center", [0.0, -3.5, 12.0])
def test_rectangle_bounds(center):
    lineshape = Rectangle(2.0)

    np.testing.assert_allclose(lineshape.bounds(center), (center - 1.0, center + 1.0))
    assert lineshape.zero_centered()


# ---------------------------------------------------------------------------------------
# UserLineShape
# ---------------------------------------------------------------------------------------


def _asymmetric_lineshape():
    x = np.arange(-8, 9) * 0.125
    values = np.interp(x, [-1.0, -0.25, 1.0], [0.0, 1.0, 0.0])
    return x, values


@pytest.mark.parametrize("mean", [0.0, 0.3, -1.1])
def test_user_lineshape_simple_mode_interpolates_shifted_lineshape(mean):
    x, values = _asymmetric_lineshape()
    samples = np.linspace(-3, 3, 97)

    weights = UserLineShape(x, values, zero_centered=True).integration_weights(
        mean, samples
    )

    expected = np.interp(samples - mean, x, values, left=0, right=0)
    np.testing.assert_allclose(weights, expected / expected.sum(), atol=1e-14)


def test_user_lineshape_simple_mode_matches_gaussian_constant_mode():
    stdev = 0.5
    x = np.arange(-400, 401) * 0.01
    samples = 2.3 + np.arange(-200, 201) * 0.05

    user = UserLineShape(x, np.exp(-0.5 * (x / stdev) ** 2), zero_centered=True)
    gaussian = Gaussian(stdev=stdev, mode="constant", max_stdev=8)

    np.testing.assert_allclose(
        user.integration_weights(2.3, samples),
        gaussian.integration_weights(2.3, samples),
        atol=1e-12,
    )


def test_user_lineshape_not_zero_centered_ignores_mean():
    x, values = _asymmetric_lineshape()
    x = x + 4.0
    samples = np.linspace(2, 6, 81)
    lineshape = UserLineShape(x, values, zero_centered=False)

    first = lineshape.integration_weights(0.0, samples)
    second = lineshape.integration_weights(123.0, samples)

    np.testing.assert_array_equal(first, second)
    expected = np.interp(samples, x, values, left=0, right=0)
    np.testing.assert_allclose(first, expected / expected.sum(), atol=1e-14)


@pytest.mark.parametrize("zero_centered", [True, False])
def test_user_lineshape_zero_centered_flag(zero_centered):
    x, values = _asymmetric_lineshape()

    lineshape = UserLineShape(x, values, zero_centered=zero_centered)

    assert lineshape.zero_centered() is zero_centered


def test_user_lineshape_requires_normalize():
    x, values = _asymmetric_lineshape()
    lineshape = UserLineShape(x, values, zero_centered=True)

    with pytest.raises(ValueError, match="only supports normalized"):
        lineshape.integration_weights(0.0, np.linspace(-2, 2, 11), normalize=False)


def test_user_lineshape_invalid_mode_raises():
    x, values = _asymmetric_lineshape()
    lineshape = UserLineShape(x, values, zero_centered=True, mode="cubic")

    with pytest.raises(ValueError, match="must be one of simple or integrate"):
        lineshape.integration_weights(0.0, np.linspace(-2, 2, 11))


def test_user_lineshape_bounds():
    x = np.linspace(-5, 5, 201)
    values = np.exp(-0.5 * x**2)

    wide = UserLineShape(x, values, zero_centered=True, integration_fraction=0.999)
    narrow = UserLineShape(x, values, zero_centered=True, integration_fraction=0.5)

    lower, upper = wide.bounds()
    assert x[0] < lower < -2.5
    assert 2.5 < upper < x[-1]
    np.testing.assert_allclose(wide.bounds(center=7.5), (lower + 7.5, upper + 7.5))

    narrow_lower, narrow_upper = narrow.bounds()
    assert lower < narrow_lower <= 0 <= narrow_upper < upper


def test_user_lineshape_bounds_one_sided():
    # Peak at the first sample, e.g. an exponential response
    x = np.linspace(0, 10, 101)

    lower, upper = UserLineShape(x, np.exp(-x), zero_centered=True).bounds()

    assert lower == x[0]
    assert 5 < upper < x[-1]


def test_user_lineshape_full_integration_fraction_bounds_cover_lineshape():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    values = np.array([1.0, 2.0, 4.0, 2.0, 1.0])

    lineshape = UserLineShape(x, values, zero_centered=True, integration_fraction=1.0)

    np.testing.assert_allclose(lineshape.bounds(), (-2.0, 2.0))


def test_user_lineshape_integrate_mode_exact_when_grids_align():
    x, values = _asymmetric_lineshape()
    x = x + 0.25
    samples = np.arange(-24, 25) * 0.125

    weights = UserLineShape(
        x, values, zero_centered=False, mode="integrate"
    ).integration_weights(0.0, samples)

    expected = _exact_piecewise_linear_weights(
        lambda s: np.interp(s, x, values, left=0, right=0), samples, x
    )
    np.testing.assert_allclose(weights, expected / expected.sum(), atol=1e-12)


def test_user_lineshape_integrate_mode_zero_centered_follows_mean():
    x, values = _asymmetric_lineshape()
    samples = np.arange(-24, 25) * 0.125
    mean = 0.5

    centered = UserLineShape(x, values, zero_centered=True, mode="integrate")
    absolute = UserLineShape(x + mean, values, zero_centered=False, mode="integrate")

    np.testing.assert_allclose(
        centered.integration_weights(mean, samples),
        absolute.integration_weights(mean, samples),
        atol=1e-12,
    )


@pytest.mark.xfail(
    reason="Integrate mode is only exact when sample and lineshape knots coincide; "
    "otherwise _triangle_analytic_linear_weights_helper2 gives errors of a few percent",
    strict=True,
)
def test_user_lineshape_integrate_mode_exact_when_grids_do_not_align():
    x = np.arange(-8, 9) * 0.125
    values = 1 - np.abs(x)
    samples = np.arange(-48, 49) * 0.0625 + 0.015625

    weights = UserLineShape(
        x, values, zero_centered=False, mode="integrate"
    ).integration_weights(0.0, samples)

    expected = _exact_piecewise_linear_weights(
        lambda s: np.interp(s, x, values, left=0, right=0), samples, x
    )
    np.testing.assert_allclose(weights, expected / expected.sum(), atol=1e-10)
    # Symmetric lineshape: first moment must be at the centre
    np.testing.assert_allclose(weights @ samples, 0.0, atol=1e-10)
