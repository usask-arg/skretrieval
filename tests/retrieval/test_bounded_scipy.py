from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from skretrieval.retrieval.scipy import SciPyMinimizer
from skretrieval.retrieval.statevector import StateVector, StateVectorElement
from skretrieval.retrieval.target import GenericTarget


class _BoundedElement(StateVectorElement):
    def __init__(self):
        super().__init__()
        self.x = np.array([0.1, 0.2])

    def state(self):
        return self.x

    def update_state(self, x):
        self.x = np.array(x, copy=True)

    def apriori_state(self):
        return np.array([0.1, 0.2])

    def lower_bound(self):
        return np.zeros(2)

    def upper_bound(self):
        return np.ones(2)

    def prior_precision_factor(self):
        return sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])

    def inverse_apriori_covariance(self):
        factor = self.prior_precision_factor()
        return factor.T @ factor

    def name(self):
        return "bounded"

    def propagate_wf(self, radiance):
        return radiance


class _IdentityOperator:
    n_state = 2
    shape = (2, 2)

    def matvec(self, value):
        return np.asarray(value)

    def rmatvec(self, value):
        return np.asarray(value)


class _BoundedTarget(GenericTarget):
    def __init__(self, transform=True):
        super().__init__(StateVector([_BoundedElement()]), transform)

    def _internal_measurement_vector(self, l1_data):
        if l1_data == "measurement":
            return {"y": np.array([0.9, 0.7]), "y_error": sparse.eye(2, format="csc")}
        result = {"y": self._native_state_vector().copy()}
        if l1_data == "linearized":
            result["jacobian_operator"] = _IdentityOperator()
        else:
            result["jacobian"] = np.eye(2)
        return result


class _ForwardModel:
    def calculate_linearized_radiance(self):
        return "linearized"

    def calculate_radiance(self):
        return "materialized"


@pytest.mark.parametrize("solver", ["materialized", "lsmr", "lbfgsb"])
@pytest.mark.parametrize("transform", [True, "affine", False])
@pytest.mark.parametrize("start_at_midpoint", [False, True])
def test_bounded_solvers_recover_native_map_and_covariance(
    solver, transform, start_at_midpoint
):
    target = _BoundedTarget(transform)
    if start_at_midpoint:
        target._state_vector.state_elements[0].update_state(
            np.full(2, np.nextafter(0.5, 1.0))
        )
    prior = target._native_inverse_apriori_covariance().toarray()
    expected_covariance = np.linalg.inv(np.eye(2) + prior)
    expected_state = expected_covariance @ (
        np.array([0.9, 0.7]) + prior @ target._native_apriori_state()
    )
    options = (
        {"minimize_options": {"gtol": 1e-10}} if solver == "lbfgsb" else {"gtol": 1e-10}
    )
    if solver != "materialized":
        options["matrix_free_state_scale"] = np.array([0.5, 2.0])
    result = SciPyMinimizer(
        jacobian_mode="materialized" if solver == "materialized" else "matrix_free",
        matrix_free_solver="lbfgsb" if solver == "lbfgsb" else "lsmr",
        max_nfev=100,
        ftol=1e-12,
        xtol=1e-12,
        verbose=0,
        **options,
    ).retrieve("measurement", _ForwardModel(), target)

    assert result["minimizer"].success
    np.testing.assert_allclose(target._native_state_vector(), expected_state, atol=1e-7)
    np.testing.assert_allclose(
        result["solution_covariance"], expected_covariance, atol=1e-7
    )
    np.testing.assert_allclose(
        result["averaging_kernel"], expected_covariance, atol=1e-7
    )
    np.testing.assert_allclose(
        result["error_covariance_from_noise"],
        expected_covariance @ expected_covariance,
        atol=1e-7,
    )


@pytest.mark.parametrize("solver", ["lsmr", "lbfgsb"])
def test_diagonal_diagnostics_use_final_bounded_state(solver):
    target = _BoundedTarget()
    result = SciPyMinimizer(
        jacobian_mode="matrix_free",
        matrix_free_solver=solver,
        matrix_free_diagnostics="fisher_diagonal",
        posterior_diagonal_probe_count=16384,
        posterior_diagonal_probe_batch_size=512,
        max_nfev=100,
        ftol=1e-12,
        xtol=1e-12,
        verbose=0,
    ).retrieve("measurement", _ForwardModel(), target)

    np.testing.assert_allclose(result["measurement_information_diagonal"], [1.0, 1.0])
    np.testing.assert_allclose(
        result["approximate_averaging_kernel_row_sum"], [0.5, 0.5], atol=1e-10
    )
    np.testing.assert_allclose(
        result["solution_covariance_diagonal"], [0.375, 0.375], rtol=0.03
    )


def test_bounded_prior_residual_derivative_matches_finite_difference():
    target = _BoundedTarget()
    state = np.array([0.7, -0.4])
    direction = np.array([0.3, -0.8])
    step = 1e-6
    target.update_state(state + step * direction)
    plus = target.prior_residual()
    target.update_state(state - step * direction)
    minus = target.prior_residual()
    target.update_state(state)
    residual = target.prior_residual()
    jacobian = target.prior_precision_factor()
    cost, gradient = target.prior_cost_and_gradient()

    np.testing.assert_allclose(
        jacobian @ direction, (plus - minus) / (2 * step), rtol=1e-8
    )
    np.testing.assert_allclose(0.5 * residual @ residual, cost)
    np.testing.assert_allclose(jacobian.T @ residual, gradient)


@pytest.mark.parametrize("diagnostics", [False, True])
def test_bounded_output_does_not_allocate_dense_diagonal_matrices(
    monkeypatch, diagnostics
):
    target = _BoundedTarget()
    output = {"minimizer": object()}
    if diagnostics:
        output["solution_covariance_diagonal"] = np.ones(2)

    def reject_dense_diagonal(*_args, **_kwargs):
        pytest.fail("Matrix-free output must not allocate dense diagonal matrices")

    monkeypatch.setattr("skretrieval.retrieval.target.np.diag", reject_dense_diagonal)
    result = target.state_vector_error_output(output)

    assert result["minimizer"] is output["minimizer"]
    if diagnostics:
        np.testing.assert_allclose(
            result["solution_covariance_diagonal"], np.array([0.09, 0.16]) ** 2
        )
