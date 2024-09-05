from __future__ import annotations

import numpy as np

from skretrieval.core.lineshape import DeltaFunction
from skretrieval.retrieval.usarm.forwardmodel import IdealViewingSpectrograph
from skretrieval.retrieval.usarm.observation import SimulatedNadirObservation
from skretrieval.retrieval.usarm.processing import USARMRetrieval
from skretrieval.util import configure_log


def test_simulated_retrieval():
    configure_log()
    obs = SimulatedNadirObservation(
        cos_sza=0.6,
        cos_viewing_zenith=1.0,
        reference_latitude=20,
        reference_longitude=0,
        sample_wavelengths=np.arange(280, 350, 0.5),
    )

    ret = USARMRetrieval(
        obs,
        forward_model_class=IdealViewingSpectrograph,
        forward_model_kwargs={"lineshape_fn": lambda _: DeltaFunction()},
        minimizer="rodgers",
        target_kwargs={"rescale_state_space": True},
        minimizer_kwargs={
            "lm_damping_method": "fletcher",
            "lm_damping": 0.1,
            "max_iter": 10,
            "lm_change_factor": 2,
            "iterative_update_lm": True,
            "retreat_lm": False,
            "apply_cholesky_scaling": True,
            "convergence_factor": 0.5,
            "convergence_check_method": "dcost",
        },
        state_kwargs={
            "altitude_grid": np.arange(0, 70000, 1000),
            "absorbers": {
                "o3": {
                    "prior_influence": 5e-1,
                    "tikh_factor": 1e-2,
                    "log_space": False,
                    "min_value": 0,
                    "max_value": 1,
                    "prior": {"type": "mipas"},
                },
            },
            "aerosols": {
                "stratospheric": {
                    "type": "extinction_profile",
                    "nominal_wavelength": 745,
                    "retrieved_quantities": {
                        "extinction_per_m": {
                            "prior_influence": 1e-1,
                            "tikh_factor": 1e-1,
                            "min_value": 0,
                            "max_value": 1e-3,
                        },
                        "median_radius": {
                            "prior_influence": 1e-6,
                            "tikh_factor": 1e-2,
                            "min_value": 10,
                            "max_value": 900,
                        },
                    },
                    "prior": {
                        "extinction_per_m": {"type": "testing"},
                        "median_radius": {"type": "constant", "value": 80},
                    },
                    "enabled": False,
                }
            },
        },
    )

    ret.retrieve()
