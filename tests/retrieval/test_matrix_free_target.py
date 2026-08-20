from __future__ import annotations

import numpy as np

from skretrieval.retrieval.target import LogisticBoundingMixin


def test_logistic_bound_mapping_wraps_operator_without_mutating_it():
    jacobian = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]])

    class Operator:
        shape = jacobian.shape
        n_state = jacobian.shape[1]

        def matvec(self, value):
            return jacobian @ value

        def rmatvec(self, value):
            return jacobian.T @ value

    class Target(LogisticBoundingMixin):
        _rescale_state_elements = True

        def state_vector(self):
            return np.array([0.0, np.log(3.0)])

        def lower_bound(self):
            return np.array([-1.0, 2.0])

        def upper_bound(self):
            return np.array([1.0, 6.0])

    operator = Operator()
    original_matvec = operator.matvec
    original_rmatvec = operator.rmatvec
    mapped = Target().map_jacobian_operator(operator)

    bounded_state = np.array([0.0, 5.0])
    lower = Target().lower_bound()
    upper = Target().upper_bound()
    mapping = (bounded_state - lower) * (upper - bounded_state) / (upper - lower)
    direction = np.array([0.3, -0.8])
    cotangent = np.array([0.2, -0.4, 0.7])

    np.testing.assert_allclose(
        mapped.matvec(direction), jacobian @ (mapping * direction)
    )
    np.testing.assert_allclose(
        mapped.rmatvec(cotangent), mapping * (jacobian.T @ cotangent)
    )
    np.testing.assert_allclose(
        cotangent @ mapped.matvec(direction), direction @ mapped.rmatvec(cotangent)
    )
    assert mapped is not operator
    assert operator.matvec == original_matvec
    assert operator.rmatvec == original_rmatvec
