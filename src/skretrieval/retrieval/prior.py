from __future__ import annotations

import abc
import dataclasses
from copy import copy

import numpy as np

from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.tikhonov import (
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
        """
        The inverse covariance of the prior state $S_a^{-1}$ of size (n, n)
        """

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
            return np.linalg.solve(full_inv_S_a, rhs)
        except np.linalg.LinAlgError:
            # If the inverse covariance is singular, we can't solve the system
            # TODO: Is this actually right? It seems okay in most cases, but in general
            # i'm not so sure
            return 0.5 * (x_a_1 + x_a_2)

    @property
    def inverse_covariance(self):
        return self._prior1.inverse_covariance + self._prior2.inverse_covariance

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
