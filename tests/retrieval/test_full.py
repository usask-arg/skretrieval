from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk2

import skretrieval as skr
from skretrieval.retrieval.observation import SimulatedNadirObservation
from skretrieval.util import configure_log


@pytest.mark.parametrize(
    ("minimizer", "minimizer_kwargs"),
    [
        pytest.param("rodgers", {}, id="rodgers"),
        pytest.param("scipy", {}, id="scipy"),
        pytest.param(
            "scipy",
            {"materialized_jacobian_source": "linearization"},
            id="scipy-linearization",
        ),
        pytest.param("scipy_lsmr", {}, id="scipy-lsmr"),
    ],
)
def test_simulated_retrieval(minimizer: str, minimizer_kwargs: dict):
    needs_linearization = (
        minimizer == "scipy_lsmr"
        or minimizer_kwargs.get("materialized_jacobian_source") == "linearization"
    )
    if needs_linearization and not hasattr(sk2.Engine, "linearize"):
        pytest.skip("SASKTRAN2 does not provide Engine.linearize")

    configure_log()
    obs = SimulatedNadirObservation(
        cos_sza=0.6,
        cos_viewing_zenith=1.0,
        reference_latitude=20,
        reference_longitude=0,
        sample_wavelengths=np.arange(280, 350, 0.5),
        state_adjustment_factors={
            "o3": {"vmr": {"scale": 2}},
            "lambertian_albedo": {"albedo": {"scale": 0.5}},
        },
    )

    ret = skr.Retrieval(
        obs,
        minimizer=minimizer,
        minimizer_kwargs=minimizer_kwargs,
        state_kwargs={
            "altitude_grid": np.arange(0, 70000, 1000),
            "absorbers": {
                "o3": {
                    "prior_influence": 5e0,
                    "tikh_factor": 1e-2,
                    "log_space": False,
                    "min_value": 0,
                    "max_value": 1,
                    "prior": {"type": "mipas"},
                },
            },
            "surface": {
                "lambertian_albedo": {
                    "prior_influence": 0,
                    "tikh_factor": 1e-2,
                    "log_space": False,
                    "wavelengths": np.array([280, 360]),
                    "initial_value": 0.5,
                }
            },
        },
    )

    result = ret.retrieve()

    if minimizer == "scipy_lsmr":
        assert "solution_covariance" not in result["minimizer"]


def test_retrieval_keeps_active_state_through_output_and_restores_flags(monkeypatch):
    class Element:
        enabled = True

    class StateVector:
        def __init__(self):
            self.sv = {"inactive": Element(), "active": Element()}

        def describe(self, _result):
            assert not self.sv["inactive"].enabled
            assert self.sv["active"].enabled
            return "state"

    class MeasurementVector:
        enabled = True

    class Target:
        def update_state_slices(self):
            pass

    class ForwardModel:
        def __init__(self, state_vector):
            self._state_vector = state_vector

        def calculate_radiance(self):
            assert not self._state_vector.sv["inactive"].enabled
            return "simulated"

    minimizer_options = {}

    class Minimizer:
        def __init__(self, **kwargs):
            minimizer_options.update(kwargs)

        def retrieve(self, *_args):
            return {}

    monkeypatch.setattr(
        "skretrieval.retrieval.processing.SciPyMinimizer",
        Minimizer,
    )

    retrieval = skr.Retrieval.__new__(skr.Retrieval)
    retrieval._minimizer = "scipy_lsmr"
    retrieval._minimizer_kwargs = {}
    retrieval._state_vector = StateVector()
    retrieval._measurement_vector = {"measurement": MeasurementVector()}
    retrieval._target = Target()
    retrieval._forward_model = ForwardModel(retrieval._state_vector)
    retrieval._obs_l1 = "measured"

    result = retrieval.retrieve(enabled_state_elements=["active"])

    assert minimizer_options["matrix_free_diagnostics"] == "none"
    assert result["state"] == "state"
    assert result["simulated_l1"] == "simulated"
    assert all(element.enabled for element in retrieval._state_vector.sv.values())
    assert retrieval._measurement_vector["measurement"].enabled
