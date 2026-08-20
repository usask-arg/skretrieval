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
