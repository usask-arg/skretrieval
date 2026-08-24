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
        return information_sqrt(
            self.inverse_covariance, "A priori inverse covariance"
        )

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
        return sparse.eye(len(self._prior.state), format="csr") * np.sqrt(
            self._value
        )


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
        prior_state: np.ndarray | None = None,
    ) -> None:
        if len(shape) != 2 or any(int(size) < 1 for size in shape):
            msg = "shape must contain positive (horizontal, altitude) sizes"
            raise ValueError(msg)
        if vertical_order not in (1, 2) or horizontal_order not in (1, 2):
            msg = "vertical_order and horizontal_order must be 1 or 2"
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
        self._vertical_order = vertical_order
        self._horizontal_order = horizontal_order
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
