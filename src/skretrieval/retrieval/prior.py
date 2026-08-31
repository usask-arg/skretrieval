from __future__ import annotations

import abc
import dataclasses
from copy import copy

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from skretrieval.retrieval.erroranalysis import information_sqrt
from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.tikhonov import (
    two_dim_horizontal_first_deriv,
    two_dim_horizontal_second_deriv,
    two_dim_vertical_first_deriv,
    two_dim_vertical_second_deriv,
)


@dataclasses.dataclass
class Prior:
    """
    A dataceass to hold a prior state of size (n,) and an inverse covariance of size (n, n)
    """

    state: np.array
    inverse_covariance: np.ndarray


class BasePrior:
    """
    Base class to handle prior states
    """

    @property
    @abc.abstractmethod
    def state(self) -> np.array:
        """
        The prior state $x_a$ of size (n,)
        """

    @property
    @abc.abstractmethod
    def inverse_covariance(self):
        """The inverse covariance of the prior state.

        The returned matrix has shape ``(n, n)``.
        """

    @property
    def precision_factor(self):
        """Return ``R`` such that ``R.T @ R`` is the prior precision.

        Structured priors override this to retain their sparse residual form.
        The fallback is intended for small or dense custom priors.
        """
        return information_sqrt(self.inverse_covariance, "A priori inverse covariance")

    def __mul__(self, other):
        return MultipliedPrior(self, other)

    __rmul__ = __mul__

    def __add__(self, other):
        return AdditivePrior(self, other)

    def init(self, sv: StateVectorElement, sv_slice: slice | None = None):
        pass


class MultipliedPrior(BasePrior):
    def __init__(self, prior: BasePrior, multiplier: float):
        """
        A prior where the inverse covariance is multiplied by a scalar, the prior state remains unchanged

        Parameters
        ----------
        prior : BasePrior
        multiplier : float
        """
        self._prior = prior
        self._multiplier = multiplier

    @property
    def state(self):
        return self._prior.state

    @property
    def inverse_covariance(self):
        return self._prior.inverse_covariance * self._multiplier

    @property
    def precision_factor(self):
        if self._multiplier < 0:
            msg = "Prior precision multiplier must be non-negative"
            raise ValueError(msg)
        return self._prior.precision_factor * np.sqrt(self._multiplier)

    def init(self, sv: StateVectorElement, sv_slice: slice | None = None):
        self._prior.init(sv, sv_slice)


class AdditivePrior(BasePrior):
    def __init__(self, prior1: BasePrior, prior2: BasePrior):
        """
        A prior where two priors are added together.  This results in a sum of the inverse covariance,
        and then a new prior state x_a

        Parameters
        ----------
        prior1 : BasePrior
        prior2 : BasePrior
        """
        self._prior1 = prior1
        self._prior2 = prior2

    @property
    def state(self):
        # Have to solve the system to get the equivalent prior state

        inv_S_a_1 = self._prior1.inverse_covariance
        inv_S_a_2 = self._prior2.inverse_covariance
        x_a_1 = self._prior1.state
        x_a_2 = self._prior2.state

        full_inv_S_a = inv_S_a_1 + inv_S_a_2

        rhs = inv_S_a_1 @ x_a_1 + inv_S_a_2 @ x_a_2

        # For some priors the inverse covariance will be singular
        try:
            if sparse.issparse(full_inv_S_a):
                return spsolve(full_inv_S_a.tocsc(), rhs)
            return np.linalg.solve(full_inv_S_a, rhs)
        except (np.linalg.LinAlgError, RuntimeError):
            # If the inverse covariance is singular, we can't solve the system
            # TODO: Is this actually right? It seems okay in most cases, but in general
            # i'm not so sure
            return 0.5 * (x_a_1 + x_a_2)

    @property
    def inverse_covariance(self):
        return self._prior1.inverse_covariance + self._prior2.inverse_covariance

    @property
    def precision_factor(self):
        first = self._prior1.precision_factor
        second = self._prior2.precision_factor
        if sparse.issparse(first) or sparse.issparse(second):
            return sparse.vstack((first, second), format="csr")
        return np.vstack((first, second))

    def init(self, sv: StateVectorElement, sv_slice: slice | None = None):
        self._prior1.init(sv, sv_slice)
        self._prior2.init(sv, sv_slice)


class VerticalPrior(BasePrior):
    def __init__(self, altitudes: np.array):
        self._altitudes = altitudes


class VerticalTikhonov(VerticalPrior):
    def __init__(
        self,
        order: int,
        prior_state: np.array = None,
        tikhonov: np.array = None,
    ):
        """
        A prior that is constructed as a Tikhonov constraint.

        Parameters
        ----------
        order : int
            Order of the Tikhonov constraint, only 1 and 2 are supported
        prior_state : np.array, optional
            Prior state. If set to None a zero prior is used, by default None
        tikhonov : np.array, optional
            Array of factors to multiply the constraint by, by default None
        """
        self._tikhonov = tikhonov
        self._prior_state = prior_state
        self._order = order

    def init(self, sv: StateVectorElement, sv_slice: slice | None = None):
        n = len(sv.state()[sv_slice])

        if self._order == 1:
            self._gamma = two_dim_vertical_first_deriv(1, n, factor=1)
        elif self._order == 2:
            self._gamma = two_dim_vertical_second_deriv(1, n, factor=1)
        else:
            msg = f"Order {self._order} not supported."
            raise ValueError(msg)

        if self._tikhonov is not None:
            # Scale by the weights
            self._gamma *= self._tikhonov[np.newaxis, :]

        self._prior = Prior(
            inverse_covariance=self._gamma.T @ self._gamma,
            state=(np.zeros(n) if self._prior_state is None else self._prior_state),
        )

    @property
    def state(self):
        return self._prior.state

    @property
    def inverse_covariance(self):
        return self._prior.inverse_covariance

    @property
    def precision_factor(self):
        return self._gamma


class ManualPrior(BasePrior):
    def __init__(self, state: np.array, inverse_covariance: np.array):
        """
        A prior that is manually specified, both the prior state and it's covariance

        Parameters
        ----------
        state : np.array
        inverse_covariance : np.array
        """
        self._state = state
        self._inverse_covariance = inverse_covariance

    @property
    def state(self):
        return self._state

    @property
    def inverse_covariance(self):
        return self._inverse_covariance

    def init(self, sv: StateVectorElement, sv_slice: slice | None = None):
        pass


class ConstantDiagonalPrior(BasePrior):
    def __init__(self, value: float = 1.0):
        """
        A prior that is constant along the diagonal. The initial state is pulled from
        the StateVectorElement upon initialization.

        Parameters
        ----------
        value : float, optional
            _description_, by default 1.0
        """
        self._value = value

    def init(self, sv: StateVectorElement, sv_slice: slice | None = None):
        n = len(sv.state()[sv_slice])
        self._prior = Prior(
            inverse_covariance=np.eye(n) * self._value,
            state=copy(sv.state()[sv_slice]),
        )

    @property
    def state(self):
        return self._prior.state

    @property
    def inverse_covariance(self):
        return self._prior.inverse_covariance

    @property
    def precision_factor(self):
        if self._value < 0:
            msg = "Diagonal prior precision must be non-negative"
            raise ValueError(msg)
        return sparse.eye(len(self._prior.state), format="csr") * np.sqrt(self._value)


class TwoDimensionalTikhonov(BasePrior):
    """Sparse smoothness and diagonal prior for a structured 2D state.

    The state is flattened in C order from ``(horizontal, altitude)`` so that
    altitude varies fastest. This is the native SASKTRAN2 ``Geometry2D`` and
    ``OrbitalPlaneGeometry`` ordering.

    Parameters
    ----------
    shape : tuple[int, int]
        Number of horizontal and altitude grid points.
    vertical_factor : float or numpy.ndarray, optional
        Precision multiplier for vertical differences. Arrays must broadcast to
        ``shape`` and weight each difference row on the two-dimensional grid.
    horizontal_factor : float or numpy.ndarray, optional
        Precision multiplier for horizontal differences. Arrays must broadcast
        to ``shape`` and weight each difference row on the two-dimensional grid.
    diagonal_factor : float or numpy.ndarray, optional
        Diagonal precision anchoring the solution to the prior state. Arrays
        must broadcast to ``shape``.
    vertical_order : int, optional
        Vertical difference order, either 1 or 2.
    horizontal_order : int, optional
        Horizontal difference order, either 1 or 2.
    coherent_horizontal_factor : float or numpy.ndarray, optional
        Precision multiplier for an additional horizontal-difference penalty
        applied after Gaussian smoothing in altitude. This preferentially
        penalizes horizontally oscillatory structures that persist through
        several altitude levels. Arrays must broadcast to ``shape``.
    coherent_horizontal_order : int, optional
        Horizontal difference order for the vertically coherent penalty.
    coherent_horizontal_smoothing_sigma : float, optional
        Standard deviation of a horizontal Gaussian smoother in grid cells.
        A positive value band-limits the coherent derivative so it targets a
        finite range of horizontal wavelengths instead of increasing
        monotonically towards the grid scale. Zero disables horizontal
        smoothing and retains the native derivative.
    coherent_horizontal_spacing : float, optional
        Physical spacing between horizontal grid points. The coherent
        derivative is scaled as a quadrature-weighted physical derivative, so
        the same factor has comparable strength on grids with different
        spacing. The units are chosen by the caller; the physical Gaussian
        scale is this spacing multiplied by
        ``coherent_horizontal_smoothing_sigma``.
    coherent_vertical_sigma : float, optional
        Standard deviation of the vertical Gaussian smoother in grid cells.
        Must be positive when ``coherent_horizontal_factor`` is non-zero.
    prior_state : numpy.ndarray, optional
        Prior in retrieval-state coordinates. When omitted, the state at
        initialization is used.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        vertical_factor: float | np.ndarray = 0.0,
        horizontal_factor: float | np.ndarray = 0.0,
        diagonal_factor: float | np.ndarray = 0.0,
        vertical_order: int = 1,
        horizontal_order: int = 1,
        coherent_horizontal_factor: float | np.ndarray = 0.0,
        coherent_horizontal_order: int = 2,
        coherent_horizontal_smoothing_sigma: float = 0.0,
        coherent_horizontal_spacing: float = 1.0,
        coherent_vertical_sigma: float = 0.0,
        prior_state: np.ndarray | None = None,
    ) -> None:
        if len(shape) != 2 or any(int(size) < 1 for size in shape):
            msg = "shape must contain positive (horizontal, altitude) sizes"
            raise ValueError(msg)
        if (
            vertical_order not in (1, 2)
            or horizontal_order not in (1, 2)
            or coherent_horizontal_order not in (1, 2)
        ):
            msg = "all vertical and horizontal difference orders must be 1 or 2"
            raise ValueError(msg)
        self._shape = tuple(int(size) for size in shape)
        self._vertical_factor = self._validated_factor(
            vertical_factor,
            "vertical_factor",
        )
        self._horizontal_factor = self._validated_factor(
            horizontal_factor,
            "horizontal_factor",
        )
        self._diagonal_factor = self._validated_factor(
            diagonal_factor,
            "diagonal_factor",
        )
        self._coherent_horizontal_factor = self._validated_factor(
            coherent_horizontal_factor,
            "coherent_horizontal_factor",
        )
        if not np.isfinite(coherent_vertical_sigma) or coherent_vertical_sigma < 0:
            msg = "coherent_vertical_sigma must be finite and non-negative"
            raise ValueError(msg)
        if (
            not np.isfinite(coherent_horizontal_smoothing_sigma)
            or coherent_horizontal_smoothing_sigma < 0
        ):
            msg = "coherent_horizontal_smoothing_sigma must be finite and non-negative"
            raise ValueError(msg)
        if (
            not np.isfinite(coherent_horizontal_spacing)
            or coherent_horizontal_spacing <= 0
        ):
            msg = "coherent_horizontal_spacing must be finite and positive"
            raise ValueError(msg)
        if (
            np.any(np.asarray(self._coherent_horizontal_factor) > 0)
            and coherent_vertical_sigma <= 0
        ):
            msg = (
                "coherent_vertical_sigma must be positive when the coherent "
                "horizontal penalty is enabled"
            )
            raise ValueError(msg)
        self._vertical_order = vertical_order
        self._horizontal_order = horizontal_order
        self._coherent_horizontal_order = coherent_horizontal_order
        self._coherent_horizontal_smoothing_sigma = float(
            coherent_horizontal_smoothing_sigma
        )
        self._coherent_horizontal_spacing = float(coherent_horizontal_spacing)
        self._coherent_vertical_sigma = float(coherent_vertical_sigma)
        self._prior_state = (
            None
            if prior_state is None
            else np.asarray(prior_state, dtype=float).reshape(-1)
        )

    def _validated_factor(
        self,
        factor: float | np.ndarray,
        name: str,
    ) -> float | np.ndarray:
        values = np.asarray(factor, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            msg = f"{name} must be finite and non-negative"
            raise ValueError(msg)
        if values.ndim == 0:
            return float(values)
        try:
            return np.broadcast_to(values, self._shape).copy()
        except ValueError as error:
            msg = f"{name} must be scalar or broadcast to shape {self._shape}"
            raise ValueError(msg) from error

    def _scaled_factor_rows(
        self,
        operator: sparse.spmatrix,
        factor: float | np.ndarray,
    ) -> sparse.spmatrix | None:
        if not np.any(np.asarray(factor) > 0):
            return None
        if np.ndim(factor) == 0:
            return np.sqrt(float(factor)) * operator
        row_scale = sparse.diags(
            np.sqrt(np.asarray(factor, dtype=float).reshape(-1)),
            format="csr",
        )
        return row_scale @ operator

    @staticmethod
    def _one_dimensional_gaussian_smoother(
        size: int,
        sigma: float,
    ) -> sparse.csr_matrix:
        radius = max(1, int(np.ceil(3.0 * sigma)))
        row = []
        column = []
        data = []
        for grid_index in range(size):
            start = max(0, grid_index - radius)
            stop = min(size, grid_index + radius + 1)
            local_column = np.arange(start, stop)
            local_weight = np.exp(-0.5 * ((local_column - grid_index) / sigma) ** 2)
            local_weight /= np.sum(local_weight)
            row.extend([grid_index] * local_column.size)
            column.extend(local_column.tolist())
            data.extend(local_weight.tolist())
        return sparse.csr_matrix(
            (data, (row, column)),
            shape=(size, size),
        )

    @classmethod
    def _vertical_gaussian_smoother(
        cls,
        num_horizontal: int,
        num_altitude: int,
        sigma: float,
    ) -> sparse.csr_matrix:
        vertical = cls._one_dimensional_gaussian_smoother(num_altitude, sigma)
        return sparse.kron(
            sparse.eye(num_horizontal, format="csr"),
            vertical,
            format="csr",
        )

    @classmethod
    def _horizontal_gaussian_smoother(
        cls,
        num_horizontal: int,
        num_altitude: int,
        sigma: float,
    ) -> sparse.csr_matrix:
        horizontal = cls._one_dimensional_gaussian_smoother(
            num_horizontal,
            sigma,
        )
        return sparse.kron(
            horizontal,
            sparse.eye(num_altitude, format="csr"),
            format="csr",
        )

    def init(self, sv: StateVectorElement, sv_slice: slice | None = None):
        state = np.asarray(sv.state()[sv_slice], dtype=float).reshape(-1)
        expected_size = int(np.prod(self._shape))
        if state.size != expected_size:
            msg = (
                "TwoDimensionalTikhonov shape does not match the state slice: "
                f"{self._shape} contains {expected_size} values, got {state.size}"
            )
            raise ValueError(msg)
        if self._prior_state is not None and self._prior_state.size != expected_size:
            msg = "prior_state size does not match the two-dimensional grid"
            raise ValueError(msg)

        num_horizontal, num_altitude = self._shape
        vertical_fn = (
            two_dim_vertical_first_deriv
            if self._vertical_order == 1
            else two_dim_vertical_second_deriv
        )
        horizontal_fn = (
            two_dim_horizontal_first_deriv
            if self._horizontal_order == 1
            else two_dim_horizontal_second_deriv
        )
        vertical = vertical_fn(num_horizontal, num_altitude, factor=1, sparse=True)
        horizontal = horizontal_fn(num_horizontal, num_altitude, factor=1, sparse=True)
        residual_factors = []
        scaled_vertical = self._scaled_factor_rows(vertical, self._vertical_factor)
        if scaled_vertical is not None:
            residual_factors.append(scaled_vertical)
        scaled_horizontal = self._scaled_factor_rows(
            horizontal,
            self._horizontal_factor,
        )
        if scaled_horizontal is not None:
            residual_factors.append(scaled_horizontal)
        if np.any(np.asarray(self._coherent_horizontal_factor) > 0):
            coherent_horizontal_fn = (
                two_dim_horizontal_first_deriv
                if self._coherent_horizontal_order == 1
                else two_dim_horizontal_second_deriv
            )
            coherent_horizontal = coherent_horizontal_fn(
                num_horizontal,
                num_altitude,
                factor=1,
                sparse=True,
            )
            # sqrt(spacing) supplies the quadrature weight while
            # spacing**-order converts the native finite difference to a
            # physical derivative. This keeps the integrated prior cost
            # comparable when the horizontal grid resolution changes.
            coherent_horizontal *= self._coherent_horizontal_spacing ** (
                0.5 - self._coherent_horizontal_order
            )
            if self._coherent_horizontal_smoothing_sigma > 0:
                horizontal_smoother = self._horizontal_gaussian_smoother(
                    num_horizontal,
                    num_altitude,
                    self._coherent_horizontal_smoothing_sigma,
                )
                coherent_horizontal = coherent_horizontal @ horizontal_smoother
            vertical_smoother = self._vertical_gaussian_smoother(
                num_horizontal,
                num_altitude,
                self._coherent_vertical_sigma,
            )
            scaled_coherent_horizontal = self._scaled_factor_rows(
                coherent_horizontal @ vertical_smoother,
                self._coherent_horizontal_factor,
            )
            if scaled_coherent_horizontal is not None:
                residual_factors.append(scaled_coherent_horizontal)
        scaled_diagonal = self._scaled_factor_rows(
            sparse.eye(expected_size, format="csr"),
            self._diagonal_factor,
        )
        if scaled_diagonal is not None:
            residual_factors.append(scaled_diagonal)
        self._precision_factor = (
            sparse.vstack(residual_factors, format="csr")
            if residual_factors
            else sparse.csr_matrix((0, expected_size))
        )
        precision = self._precision_factor.T @ self._precision_factor
        self._prior = Prior(
            inverse_covariance=precision.tocsr(),
            state=(state.copy() if self._prior_state is None else self._prior_state),
        )

    @property
    def state(self):
        return self._prior.state

    @property
    def inverse_covariance(self):
        return self._prior.inverse_covariance

    @property
    def precision_factor(self):
        return self._precision_factor


class TwoDimensionalIntegratedColumnTikhonov(BasePrior):
    """Horizontal smoothness prior on a linearized integrated 2D column.

    The state is flattened in C order from ``(horizontal, altitude)``.  The
    supplied ``column_weights`` define the local linear map from state changes
    to one integrated-column value per horizontal grid point.  For example,
    for a log extinction state linearized about extinction ``k_ref``, weights
    proportional to ``k_ref * dz`` make the prior act on absolute column
    extinction instead of weighting every log-extinction cell equally.

    Parameters
    ----------
    shape : tuple[int, int]
        Number of horizontal and altitude grid points.
    column_weights : numpy.ndarray
        Derivative of the normalized column quantity with respect to the state,
        with shape ``shape``.
    horizontal_factor : float or numpy.ndarray
        Precision multiplier for horizontal differences of the integrated
        column.  Arrays must broadcast to the horizontal grid size.
    horizontal_order : int, optional
        Horizontal difference order, either 1 or 2.
    horizontal_smoothing_sigma : float, optional
        Standard deviation of a Gaussian smoother in horizontal grid cells,
        applied before the difference operator.  A positive value targets a
        finite band of horizontal scales.
    horizontal_spacing : float, optional
        Physical spacing between horizontal grid points.  The difference
        operator is quadrature-scaled so a factor has comparable meaning on
        grids with different spacing.
    prior_state : numpy.ndarray, optional
        Affine target in retrieval-state coordinates.  When omitted, the state
        at initialization is used.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        column_weights: np.ndarray,
        *,
        horizontal_factor: float | np.ndarray,
        horizontal_order: int = 2,
        horizontal_smoothing_sigma: float = 0.0,
        horizontal_spacing: float = 1.0,
        prior_state: np.ndarray | None = None,
    ) -> None:
        if len(shape) != 2 or any(int(size) < 1 for size in shape):
            msg = "shape must contain positive (horizontal, altitude) sizes"
            raise ValueError(msg)
        if horizontal_order not in (1, 2):
            msg = "horizontal_order must be 1 or 2"
            raise ValueError(msg)
        self._shape = tuple(int(size) for size in shape)
        self._column_weights = np.asarray(column_weights, dtype=float)
        if self._column_weights.shape != self._shape:
            msg = f"column_weights must have shape {self._shape}"
            raise ValueError(msg)
        if np.any(~np.isfinite(self._column_weights)):
            msg = "column_weights must be finite"
            raise ValueError(msg)
        factor = np.asarray(horizontal_factor, dtype=float)
        if np.any(~np.isfinite(factor)) or np.any(factor < 0):
            msg = "horizontal_factor must be finite and non-negative"
            raise ValueError(msg)
        if factor.ndim == 0:
            self._horizontal_factor = float(factor)
        else:
            try:
                self._horizontal_factor = np.broadcast_to(
                    factor,
                    (self._shape[0],),
                ).copy()
            except ValueError as error:
                msg = (
                    "horizontal_factor must be scalar or broadcast to the "
                    f"horizontal size {self._shape[0]}"
                )
                raise ValueError(msg) from error
        if (
            not np.isfinite(horizontal_smoothing_sigma)
            or horizontal_smoothing_sigma < 0
        ):
            msg = "horizontal_smoothing_sigma must be finite and non-negative"
            raise ValueError(msg)
        if not np.isfinite(horizontal_spacing) or horizontal_spacing <= 0:
            msg = "horizontal_spacing must be finite and positive"
            raise ValueError(msg)
        self._horizontal_order = horizontal_order
        self._horizontal_smoothing_sigma = float(horizontal_smoothing_sigma)
        self._horizontal_spacing = float(horizontal_spacing)
        self._prior_state = (
            None
            if prior_state is None
            else np.asarray(prior_state, dtype=float).reshape(-1)
        )

    def init(self, sv: StateVectorElement, sv_slice: slice | None = None):
        state = np.asarray(sv.state()[sv_slice], dtype=float).reshape(-1)
        expected_size = int(np.prod(self._shape))
        if state.size != expected_size:
            msg = (
                "TwoDimensionalIntegratedColumnTikhonov shape does not match "
                f"the state slice: {self._shape} contains {expected_size} "
                f"values, got {state.size}"
            )
            raise ValueError(msg)
        if self._prior_state is not None and self._prior_state.size != expected_size:
            msg = "prior_state size does not match the two-dimensional grid"
            raise ValueError(msg)

        num_horizontal, num_altitude = self._shape
        row = np.repeat(np.arange(num_horizontal), num_altitude)
        column = np.arange(expected_size)
        column_map = sparse.csr_matrix(
            (self._column_weights.reshape(-1), (row, column)),
            shape=(num_horizontal, expected_size),
        )
        horizontal_fn = (
            two_dim_horizontal_first_deriv
            if self._horizontal_order == 1
            else two_dim_horizontal_second_deriv
        )
        horizontal = horizontal_fn(
            num_horizontal,
            1,
            factor=1,
            sparse=True,
        )
        horizontal *= self._horizontal_spacing ** (0.5 - self._horizontal_order)
        if self._horizontal_smoothing_sigma > 0:
            smoother = TwoDimensionalTikhonov._one_dimensional_gaussian_smoother(
                num_horizontal,
                self._horizontal_smoothing_sigma,
            )
            horizontal = horizontal @ smoother
        operator = horizontal @ column_map
        if np.ndim(self._horizontal_factor) == 0:
            self._precision_factor = (
                np.sqrt(float(self._horizontal_factor)) * operator
            ).tocsr()
        else:
            row_scale = sparse.diags(
                np.sqrt(np.asarray(self._horizontal_factor, dtype=float)),
                format="csr",
            )
            self._precision_factor = (row_scale @ operator).tocsr()
        self._prior = Prior(
            inverse_covariance=(
                self._precision_factor.T @ self._precision_factor
            ).tocsr(),
            state=(state.copy() if self._prior_state is None else self._prior_state),
        )

    @property
    def state(self):
        return self._prior.state

    @property
    def inverse_covariance(self):
        return self._prior.inverse_covariance

    @property
    def precision_factor(self):
        return self._precision_factor
