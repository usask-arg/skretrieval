from __future__ import annotations

import numpy as np

from skretrieval.retrieval import ForwardModel, RetrievalTarget
from skretrieval.retrieval.rodgers import Rodgers
from skretrieval.retrieval.statevector import StateVector, StateVectorElement
from skretrieval.retrieval.target.mvtarget import MeasVecTarget


class LinearForwardModel(ForwardModel):
    def calculate_radiance(self):
        return "model"


class LinearTarget(RetrievalTarget):
    def __init__(self):
        self._x = np.array([0.1, -0.2])
        self._xa = np.array([0.0, 0.0])
        self._inv_sa = np.diag([0.5, 2.0])
        self._y_error = np.array([0.25, 4.0, 9.0])
        self._k = np.array(
            [
                [1.0, 0.2],
                [0.3, 2.0],
                [1.5, -0.5],
            ]
        )
        self._y_meas = np.array([1.0, -0.5, 0.25])

    def state_vector(self):
        return self._x

    def measurement_vector(self, l1_data):
        y = self._y_meas if l1_data == "measurement" else self._k @ self._x
        return {"y": y, "jacobian": self._k, "y_error": self._y_error}

    def update_state(self, x: np.ndarray):
        self._x = x

    def apriori_state(self) -> np.ndarray:
        return self._xa

    def inverse_apriori_covariance(self):
        return self._inv_sa


class BoundedElement(StateVectorElement):
    def __init__(self):
        super().__init__()
        self._state = np.array([0.2, 1.4])

    def state(self) -> np.ndarray:
        return self._state

    def lower_bound(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    def upper_bound(self) -> np.ndarray:
        return np.array([1.0, 2.0])

    def name(self) -> str:
        return "bounded"

    def propagate_wf(self, radiance):
        return radiance

    def update_state(self, x: np.ndarray):
        self._state = x


class PriorBoundedElement(BoundedElement):
    def __init__(self, initial_state: np.ndarray):
        super().__init__()
        self._state = np.asarray(initial_state, dtype=float)

    def apriori_state(self) -> np.ndarray:
        return np.array([0.25, 0.8])

    def inverse_apriori_covariance(self) -> np.ndarray:
        return np.array([[3.0, -0.4], [-0.4, 2.0]])


def test_rodgers_error_estimates_are_independent_of_cholesky_scaling():
    unscaled = Rodgers(
        max_iter=1,
        lm_damping=0,
        iterative_update_lm=False,
        apply_cholesky_scaling=False,
    ).retrieve("measurement", LinearForwardModel(), LinearTarget())

    scaled = Rodgers(
        max_iter=1,
        lm_damping=0,
        iterative_update_lm=False,
        apply_cholesky_scaling=True,
    ).retrieve("measurement", LinearForwardModel(), LinearTarget())

    for key in (
        "error_covariance_from_noise",
        "solution_covariance",
        "averaging_kernel",
    ):
        np.testing.assert_allclose(scaled[key], unscaled[key])


def test_bounded_target_error_output_maps_to_state_vector_coordinates():
    target = MeasVecTarget(
        StateVector([BoundedElement()]),
        measurement_vectors={},
        context={},
        rescale_state_space=True,
    )
    averaging_kernel = np.array([[1.0, 2.0], [3.0, 4.0]])
    gain_matrix = np.ones((2, 3))
    output = {
        "error_covariance_from_noise": np.eye(2),
        "solution_covariance": np.eye(2) * 2,
        "averaging_kernel": averaging_kernel,
        "gain_matrix": gain_matrix,
        "solution_covariance_diagonal": np.array([2.0, 3.0]),
        "solution_covariance_diagonal_standard_error": np.array([0.2, 0.3]),
        "measurement_information_diagonal": np.array([4.0, 5.0]),
        "measurement_information_diagonal_standard_error": np.array([0.4, 0.5]),
    }

    result = target.state_vector_error_output(output)

    mapping = np.array([0.2 * 0.8, 1.4 * 0.6 / 2])
    transform = np.diag(mapping)
    inv_transform = np.diag(1 / mapping)

    np.testing.assert_allclose(
        result["error_covariance_from_noise"], transform @ transform
    )
    np.testing.assert_allclose(
        result["solution_covariance"],
        transform @ output["solution_covariance"] @ transform,
    )
    np.testing.assert_allclose(result["gain_matrix"], transform @ gain_matrix)
    np.testing.assert_allclose(
        result["averaging_kernel"], transform @ averaging_kernel @ inv_transform
    )
    np.testing.assert_allclose(
        result["solution_covariance_diagonal"],
        mapping**2 * output["solution_covariance_diagonal"],
    )
    np.testing.assert_allclose(
        result["solution_covariance_diagonal_standard_error"],
        mapping**2 * output["solution_covariance_diagonal_standard_error"],
    )
    np.testing.assert_allclose(
        result["measurement_information_diagonal"],
        output["measurement_information_diagonal"] / mapping**2,
    )
    np.testing.assert_allclose(
        result["measurement_information_diagonal_standard_error"],
        output["measurement_information_diagonal_standard_error"] / mapping**2,
    )


def test_affine_bounded_target_uses_fixed_local_scale_and_exact_mapping():
    element = BoundedElement()
    target = MeasVecTarget(
        StateVector([element]),
        measurement_vectors={},
        context={},
        rescale_state_space="affine",
    )

    scale = np.array([0.16, 0.42])
    np.testing.assert_allclose(target.state_vector(), [0.0, 0.0])
    np.testing.assert_allclose(target.lower_bound(), [-0.2 / 0.16, -1.4 / 0.42])
    np.testing.assert_allclose(target.upper_bound(), [0.8 / 0.16, 0.6 / 0.42])
    np.testing.assert_allclose(target._bounded_state_derivative_by_internal(), scale)

    target.update_state(np.array([0.4, 0.25]))
    np.testing.assert_allclose(
        element.state(), np.array([0.2, 1.4]) + scale * [0.4, 0.25]
    )

    native_precision = np.array([[2.0, 0.5], [0.5, 3.0]])
    mapped_precision = target._map_inv_Sa_by_dinternal(
        target.state_vector(), native_precision
    )
    transform = np.diag(scale)
    np.testing.assert_allclose(
        mapped_precision, transform @ native_precision @ transform
    )


def test_generic_target_groups_averaging_kernel_rows_by_state_element():
    target = MeasVecTarget(
        StateVector([BoundedElement(), BoundedElement()]),
        measurement_vectors={},
        context={},
        rescale_state_space="affine",
    )

    np.testing.assert_array_equal(
        target.averaging_kernel_row_sum_groups(),
        np.array([0, 0, 1, 1]),
    )


def test_logistic_target_prior_is_independent_of_phase_initial_state():
    evaluation_state = np.array([0.7, 1.1])
    targets = [
        MeasVecTarget(
            StateVector([PriorBoundedElement(initial)]),
            measurement_vectors={},
            context={},
            rescale_state_space=True,
        )
        for initial in (np.array([0.2, 0.5]), np.array([0.85, 1.7]))
    ]

    costs = []
    gradients = []
    for target in targets:
        target.update_state(target._map_bounded_to_internal(evaluation_state))
        cost, gradient = target.prior_cost_and_gradient()
        costs.append(cost)
        gradients.append(gradient)

    prior = np.array([0.25, 0.8])
    information = np.array([[3.0, -0.4], [-0.4, 2.0]])
    delta = evaluation_state - prior
    native_gradient = information @ delta
    derivative = np.array(
        [
            evaluation_state[0] * (1.0 - evaluation_state[0]),
            evaluation_state[1] * (2.0 - evaluation_state[1]) / 2.0,
        ]
    )
    expected_cost = 0.5 * delta @ native_gradient
    expected_gradient = derivative * native_gradient

    np.testing.assert_allclose(costs, expected_cost)
    for gradient in gradients:
        np.testing.assert_allclose(gradient, expected_gradient)

    target = targets[0]
    internal_state = target.state_vector()
    finite_difference = np.empty_like(internal_state)
    step = 1.0e-6
    for index in range(internal_state.size):
        direction = np.zeros_like(internal_state)
        direction[index] = step
        target.update_state(internal_state + direction)
        upper_cost, _ = target.prior_cost_and_gradient()
        target.update_state(internal_state - direction)
        lower_cost, _ = target.prior_cost_and_gradient()
        finite_difference[index] = (upper_cost - lower_cost) / (2 * step)
    target.update_state(internal_state)
    np.testing.assert_allclose(finite_difference, expected_gradient, rtol=1.0e-6)
