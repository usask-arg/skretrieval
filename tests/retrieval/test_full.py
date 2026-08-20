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

    _ = ret.retrieve()
