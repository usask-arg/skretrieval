---
file_format: mystnb
---

(_example_nadir_ozone)=

```{code-cell}
from skretrieval import usarm
import numpy as np

observation = usarm.observation.SimulatedNadirObservation(cos_sza=0.6,
                                                          cos_viewing_zenith=1.0,
                                                          reference_latitude=20,
                                                          reference_longitude=0,
                                                          sample_wavelengths=np.arange(280, 800, 0.5),
                                                          state_adjustment_factors={"o3": 1.5})
ret = usarm.processing.USARMRetrieval(  observation,
                                        forward_model_class=usarm.forwardmodel.IdealViewingSpectrograph,
                                        state_kwargs={
                                            "altitude_grid": np.arange(0, 70000, 1000),
                                            "absorbers": {
                                            "o3": {
                                                "prior_influence": 5e0,
                                                "tikh_factor": 1e-2,
                                                "log_space": False,
                                                "min_value": 0,
                                                "max_value": 1,
                                                "prior": {"type": "mipas"}
                                            },
                                        },
                                        }
                        )

```
