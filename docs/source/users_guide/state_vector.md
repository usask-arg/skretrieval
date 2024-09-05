(_state_vector)=
# The State Vector
The state vector is the set of atmsopheric parameters that we are trying to retrieve.
In `usarm` the default behaviour is to define the state vector entirely through configuration options,
however it is also possible to override this behaviour for more complicated applications.

## Specifying the State Vector Through Configuration
The main processing class, {py:class}`skretrieval.usarm.processing.USARMRetrieval`, takes in an argument
`state_kwargs` which is responsible for defining the state vector.  An example of this parameter is

```
state_kwargs={
            "altitude_grid": np.arange(0, 70000, 1000),
            "absorbers": {
            "o3": {
                "prior_influence": 5e5,
                "tikh_factor": 1e-2,
                "log_space": False,
                "min_value": 0,
                "max_value": 1,
                "prior": {"type": "mipas"}
            },
        },
        "aerosols": {
            "stratospheric": {
                "type": "extinction_profile",
                "nominal_wavelength": 745,
                "retrieved_quantities": {
                    "extinction_per_m": {
                        "prior_influence": 1e0,
                        "tikh_factor": 1e-2,
                        "min_value": 0,
                        "max_value": 1e-3
                    },
                    "median_radius": {
                        "prior_influence": 1e2,
                        "tikh_factor": 1e-4,
                        "min_value": 10,
                        "max_value": 900
                    }
                },
                "prior": {
                    "extinction_per_m": {"type": "testing"},
                    "median_radius": {"type": "constant", "value": 80}
                }
            }
        }
        }
```
