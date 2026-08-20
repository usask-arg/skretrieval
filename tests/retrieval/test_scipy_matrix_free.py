from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from skretrieval.retrieval import ForwardModel, RetrievalTarget
from skretrieval.retrieval.forwardmodel import StandardForwardModel
from skretrieval.retrieval.scipy import (
    MatrixFreeUnsupportedError,
    SciPyMinimizer,
)


class _Target(RetrievalTarget):
    def __init__(self):
        self.x = np.array([2.0])

    def state_vector(self):
        return self.x

    def update_state(self, x: np.ndarray):
        self.x = np.asarray(x)

    def apriori_state(self):
        return np.array([0.0])

    def inverse_apriori_covariance(self):
        return np.array([[0.0]])

    def lower_bound(self):
        return np.array([-np.inf])

    def upper_bound(self):
        return np.array([np.inf])

    def measurement_vector(self, l1_data):
        if l1_data == "measurement":
            return {"y": np.array([1.0]), "y_error": np.eye(1)}
        if l1_data == "linearized_operator":
            return {
                "y": np.asarray([self.x[0]]),
                "jacobian_operator": _IdentityOperator(),
                "y_error": np.eye(1),
            }
        return {
            "y": np.asarray([self.x[0]]),
            "jacobian": np.ones((1, 1)),
            "y_error": np.eye(1),
        }


class _IdentityOperator:
    n_state = 1
    shape = (1, 1)

    def jvp(self, x: np.ndarray):
        return np.asarray(x).reshape(-1)

    def vjp(self, y: np.ndarray):
        return np.asarray(y).reshape(-1)


class _MaterializedOnlyForwardModel(ForwardModel):
    def __init__(self):
        self.materialized_calls = 0

    def calculate_radiance(self):
        self.materialized_calls += 1
        return "model"


class _UnsupportedLinearizedForwardModel(_MaterializedOnlyForwardModel):
    def calculate_linearized_radiance(self):
        return "linearized"


class _OperatorForwardModel(_MaterializedOnlyForwardModel):
    def calculate_linearized_radiance(self):
        return "linearized_operator"


class _LinearizationMaterializedForwardModel(_MaterializedOnlyForwardModel):
    def __init__(self):
        super().__init__()
        self.linearization_materialized_calls = 0

    def calculate_materialized_linearized_radiance(self):
        self.linearization_materialized_calls += 1
        return "model"


class _DenseLinearOperator:
    def __init__(self, jacobian: np.ndarray):
        self._jacobian = jacobian
        self.n_state = jacobian.shape[1]
        self.shape = jacobian.shape

    def jvp(self, value: np.ndarray):
        return self._jacobian @ value

    def vjp(self, value: np.ndarray):
        return self._jacobian.T @ value


class _LinearTarget(RetrievalTarget):
    def __init__(
        self,
        jacobian: np.ndarray,
        measurement: np.ndarray,
        covariance: np.ndarray,
    ):
        self.jacobian = jacobian
        self.measurement = measurement
        self.covariance = covariance
        self.x = np.zeros(jacobian.shape[1])

    def state_vector(self):
        return self.x

    def update_state(self, x: np.ndarray):
        self.x = np.asarray(x)

    def apriori_state(self):
        return np.zeros_like(self.x)

    def inverse_apriori_covariance(self):
        return np.zeros((len(self.x), len(self.x)))

    def lower_bound(self):
        return np.full_like(self.x, -np.inf)

    def upper_bound(self):
        return np.full_like(self.x, np.inf)

    def measurement_vector(self, l1_data):
        if l1_data == "measurement":
            return {"y": self.measurement, "y_error": self.covariance}
        return {
            "y": self.jacobian @ self.x,
            "jacobian_operator": _DenseLinearOperator(self.jacobian),
            "y_error": self.covariance,
        }


def test_scipy_matrix_free_strict_raises_when_unavailable():
    minimizer = SciPyMinimizer(
        jacobian_mode="matrix_free",
        matrix_free_fallback="raise",
        max_nfev=1,
    )

    with pytest.raises(MatrixFreeUnsupportedError):
        minimizer.retrieve("measurement", _MaterializedOnlyForwardModel(), _Target())


def test_scipy_auto_falls_back_to_materialized_when_unavailable():
    forward_model = _UnsupportedLinearizedForwardModel()
    minimizer = SciPyMinimizer(jacobian_mode="auto", max_nfev=1)

    with pytest.warns(RuntimeWarning, match="falling back"):
        result = minimizer.retrieve("measurement", forward_model, _Target())

    assert forward_model.materialized_calls > 0
    assert "gain_matrix" in result


def test_scipy_matrix_free_can_skip_operator_diagnostics_and_accept_tr_options():
    minimizer = SciPyMinimizer(
        jacobian_mode="matrix_free",
        matrix_free_diagnostics="none",
        tr_options={"maxiter": 1},
        max_nfev=1,
    )

    result = minimizer.retrieve("measurement", _OperatorForwardModel(), _Target())

    assert "minimizer" in result
    assert "averaging_kernel" not in result
    assert "solution_covariance" not in result


def test_scipy_materialized_can_use_linearization_jacobian_source():
    forward_model = _LinearizationMaterializedForwardModel()
    minimizer = SciPyMinimizer(
        materialized_jacobian_source="linearization",
        max_nfev=1,
    )

    result = minimizer.retrieve("measurement", forward_model, _Target())

    assert forward_model.materialized_calls == 0
    assert forward_model.linearization_materialized_calls > 0
    assert "gain_matrix" in result


def test_scipy_matrix_free_lbfgsb_uses_operator_gradient():
    target = _Target()
    minimizer = SciPyMinimizer(
        jacobian_mode="matrix_free",
        matrix_free_solver="lbfgsb",
        matrix_free_diagnostics="none",
        max_nfev=20,
        minimize_options={"gtol": 1e-10},
    )

    result = minimizer.retrieve("measurement", _OperatorForwardModel(), target)

    np.testing.assert_allclose(target.state_vector(), np.array([1.0]), atol=1e-8)
    assert result["minimizer"].success
    assert result["minimizer"].cost < 1e-12
    assert "averaging_kernel" not in result


def test_scipy_matrix_free_supports_correlated_measurement_covariance():
    jacobian = np.array([[1.0, 0.2], [0.5, -1.0], [1.5, 0.7]])
    measurement = np.array([1.2, -0.3, 2.0])
    covariance = np.array([[1.0, 0.2, 0.1], [0.2, 0.8, -0.05], [0.1, -0.05, 1.4]])
    target = _LinearTarget(jacobian, measurement, covariance)
    expected = np.linalg.solve(
        jacobian.T @ np.linalg.solve(covariance, jacobian),
        jacobian.T @ np.linalg.solve(covariance, measurement),
    )

    minimizer = SciPyMinimizer(
        jacobian_mode="matrix_free",
        matrix_free_diagnostics="none",
        max_nfev=20,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    minimizer.retrieve("measurement", _OperatorForwardModel(), target)

    np.testing.assert_allclose(target.state_vector(), expected, atol=1e-9)


def test_scipy_matrix_free_reweights_both_sides_between_passes():
    jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    measurement = np.array([1.0, 2.0, 3.2, 12.0])
    covariance = np.array(
        [
            [1.0, 0.2, 0.0, 0.1],
            [0.2, 1.4, -0.1, 0.0],
            [0.0, -0.1, 0.8, 0.15],
            [0.1, 0.0, 0.15, 1.2],
        ]
    )
    inverse_covariance = np.linalg.inv(covariance)
    first_state = np.linalg.solve(
        jacobian.T @ inverse_covariance @ jacobian,
        jacobian.T @ inverse_covariance @ measurement,
    )
    first_residual = measurement - jacobian @ first_state
    scale = np.abs(first_residual) / np.median(np.abs(first_residual))
    scale[scale < 1] = 1
    downweighted_information = (
        inverse_covariance / scale[:, np.newaxis] / scale[np.newaxis, :]
    )
    expected = np.linalg.solve(
        jacobian.T @ downweighted_information @ jacobian,
        jacobian.T @ downweighted_information @ measurement,
    )
    target = _LinearTarget(jacobian, measurement, covariance)

    minimizer = SciPyMinimizer(
        jacobian_mode="matrix_free",
        matrix_free_diagnostics="none",
        num_passes=2,
        max_nfev=20,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    minimizer.retrieve("measurement", _OperatorForwardModel(), target)

    np.testing.assert_allclose(target.state_vector(), expected, atol=1e-9)


def test_forward_model_refreshes_active_linearization_metadata():
    class Element:
        def __init__(self, size: int):
            self.enabled = True
            self._state = np.zeros(size)

        def state(self):
            return self._state

    class StateVector:
        def __init__(self):
            self.state_elements = [Element(1), Element(2)]

        def check_linearization_product_support(self):
            pass

        def linearization_parameter_names(self, _template):
            return tuple(
                f"parameter_{index}"
                for index, element in enumerate(self.state_elements)
                if element.enabled
            )

    class Linearization:
        backends = {"jvp": "native", "vjp": "native"}
        tangent_template = xr.Dataset(
            {
                "parameter_0": xr.DataArray([0.0], dims=["first"]),
                "parameter_1": xr.DataArray([0.0, 0.0], dims=["second"]),
            }
        )
        value = xr.DataArray(
            np.ones((1, 1, 1)),
            dims=["wavelength", "los", "stokes"],
            coords={"wavelength": [500.0], "los": [0], "stokes": ["I"]},
        )

        def jvp(self, _tangent):
            return self.value

        def vjp(self, _cotangent, parameters=None):
            return self.tangent_template[list(parameters)]

    class TestForwardModel(StandardForwardModel):
        def _construct_model_geometry(self):
            pass

        def _construct_model_wavelength(self):
            pass

        def _construct_viewing_geo(self):
            pass

        def _construct_inst_model(self):
            pass

    state_vector = StateVector()
    forward_model = object.__new__(TestForwardModel)
    forward_model._state_vector = state_vector
    forward_model._engine = {"measurement": object()}
    forward_model._linearization_tangent_templates = {}
    forward_model._linearize = lambda _key: Linearization()
    forward_model._append_instrument_result = (
        lambda l1, key, radiance: l1.__setitem__(key, radiance)
    )

    first = forward_model.calculate_linearized_radiance()
    assert first["measurement"].n_state == 3

    state_vector.state_elements[1].enabled = False
    second = forward_model.calculate_linearized_radiance()
    assert second["measurement"].n_state == 1
