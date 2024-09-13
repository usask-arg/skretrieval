(_state_vector)=
# The State Vector
The state vector is the set of atmsopheric parameters that we are trying to retrieve.
In `skretrieval` the default behaviour is to define the state vector entirely through configuration options,
however it is also possible to override this behaviour for more complicated applications.

## Specifying the State Vector Through Configuration
The main processing class, {py:class}`skretrieval.Retrieval`, takes in an argument
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
            },
            "no2: {
                "prior_influence": 5e5,
                "tikh_factor": 1e-2,
                "log_space": False,
                "min_value": 0,
                "max_value": 1,
            }
        },
        "aerosols": {
            "stratospheric_aerosol": {
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
        }
```

`skretrieval` provides default methods to construct the state vector for absorbers, aerosols, and surface parameters.

## Defining Your own State Vector Elements
If the default `skretrieval` methods to construct the state vector are not sufficient, you can manually supply your own.
This is done through the decorator method,

```python
import skretrieval as skr

@skr.Retrieval.register_state("category", "name")
def state_vector_element(self, name: str, native_alt_grid: np.array, cfg: dict):
    # calculate state vector element here
    sv_ele = ...

    return sv_ele

```

here "category" should be replaced by one of ["absorbers", "aerosols", "surface", "other"], and "name" will be the
name of the state vector element in the configuration.  For example, we could register a state vector element for
SO2 absorption through

```python
@skr.Retrieval.register_state("absorbers", "so2")
def state_vector_element(self, name: str, native_alt_grid: np.array, cfg: dict):
    # calculate state vector element here
    sv_ele = ...

    return sv_ele

```

When we construct `state_kwargs` we would then add a `so2` section under `absorbers`, and any information added
there will be passed through to the `state_vector_element` function above through the `cfg` parameter.
