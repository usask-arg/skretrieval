---
file_format: mystnb
---

(_example_nadir_ozone)=
# Backscatter Nadir Ozone Retrieval

Here we create a Nadir viewing backscatter measurement in the UV and retrieve the ozone concentration.
We also include a Lambertian surface in the state vector.

```{code-cell}
import skretrieval as skr
import numpy as np

observation = skr.observation.SimulatedNadirObservation(
    cos_sza=0.6,
    cos_viewing_zenith=1.0,
    reference_latitude=20,
    reference_longitude=0,
    sample_wavelengths=np.arange(280, 350, 0.5),
    state_adjustment_factors={"o3": 1.5, "lambertian_albedo": 2},
)


ret = skr.Retrieval(
    observation,
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

results = ret.retrieve()

skr.plotting.plot_state(results, "o3_vmr", show=True)
```
