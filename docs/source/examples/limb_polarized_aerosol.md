---
file_format: mystnb
---

(_example_limb_scatter_aerosol)=
# Limb Scatter Polarized Aerosol Retrieval

Here we set up a limb scatter aerosol retrieval similar to the previous example, except in this
example we are both going to

- Include a wavelength dependent Lambertian surface in the retrieval state vector
- Modify the instrument to measure two orthogonal polarizations


We will start by setting up our observation.  This is very similar to the previous example except
in this case we are scaling the particle size separately from the extinction, only adjusting it in the
20-25 km altitude range.

```{code-cell}
import numpy as np
import sasktran2 as sk
import skretrieval as skr
import matplotlib.pyplot as plt


@skr.Retrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_prop(*args, **kwargs):
    refrac = sk.mie.refractive.H2SO4()
    dist = sk.mie.distribution.LogNormalDistribution().freeze(mode_width=1.6)

    db = sk.database.MieDatabase(
        dist, refrac, np.array([469.0, 750.0]), median_radius=np.arange(10, 200, 10.0)
    )

    return db


# Measurement tangent altitudes
tan_alts = np.arange(10000, 45000, 1000)
altitude_grid = np.arange(0, 70000, 1000)

aerosol_scale = np.ones(len(altitude_grid))
aerosol_scale[20:25] = 1.5


sample_wavel = np.array([470.0, 745.0])

# Set up a simulated observation with our tangent altitudes, wavelengths
obs = skr.observation.SimulatedLimbObservation(
    cos_sza=0.6,
    relative_azimuth=0,
    observer_altitude=200000,
    reference_latitude=20,
    reference_longitude=20,
    tangent_altitudes=tan_alts,
    sample_wavelengths=sample_wavel,
    state_adjustment_factors={
        "stratospheric_aerosol": {"extinction_per_m": 1.5, "median_radius": aerosol_scale}
    },  # Simulate with 1.5x the prior
)

```


Next we will do the standard retrieval, but will add in a spectrally varying albedo into the state vector.

```{code-cell}
# Set up the retrieval object
ret = skr.Retrieval(
    obs,
    minimizer="scipy",
    state_kwargs={
        "altitude_grid": altitude_grid,
        "aerosols": {
            "stratospheric_aerosol": {
                "type": "extinction_profile",
                "nominal_wavelength": 745,
                "retrieved_quantities": {
                    "extinction_per_m": {
                        "prior_influence": 1e-6,
                        "tikh_factor": 1e-1,
                        "min_value": 0,
                        "max_value": 1e-3,
                    },
                    "median_radius": {
                        "prior_influence": 1e0,
                        "tikh_factor": 1e-4,
                        "min_value": 10,
                        "max_value": 190,
                    },
                },
                "prior": {
                    "extinction_per_m": {"type": "testing"},
                    "median_radius": {"type": "constant", "value": 80},
                },
            },
        },
        "surface": {
            "lambertian_albedo": {
                "prior_influence": 0,
                "tikh_factor": 1e-4,
                "log_space": False,
                "wavelengths": sample_wavel,
                "initial_value": 0.3,
                "out_bounds_mode": "extend",
            },
        },
    },
    model_kwargs={
        "num_stokes": 3,
        "multiple_scatter_source": sk.MultipleScatterSource.DiscreteOrdinates,
        "num_streams": 8,
    },
)

results = ret.retrieve()
```

And take a look at the resulting retrieved particle size

```{code-cell}
skr.plotting.plot_state(results, "stratospheric_aerosol_median_radius", show=False)
plt.subplot(1, 2, 1)
plt.plot(80 * aerosol_scale, altitude_grid, "k--")
plt.legend(["Retrieved", "Prior", "Truth"])
plt.show()
```

Note that there are some errors, we can also check the retrieved albedo

```{code-cell}
results["state"]["lambertian_albedo"]
```

And we see that it differs from the simulated value of 0.3.

Next we will repeat the calculation using a polarized instrument that measures `(I+Q)/2` and `(I-Q)/2` simultaneously.

```{code-cell}
# Set up the retrieval object
ret = skr.Retrieval(
    obs,
    minimizer="scipy",
    target_kwargs={"rescale_state_space": False},
    state_kwargs={
        "altitude_grid": altitude_grid,
        "aerosols": {
            "stratospheric_aerosol": {
                "type": "extinction_profile",
                "nominal_wavelength": 745,
                "retrieved_quantities": {
                    "extinction_per_m": {
                        "prior_influence": 1e-6,
                        "tikh_factor": 1e-1,
                        "min_value": 0,
                        "max_value": 1e-3,
                    },
                    "median_radius": {
                        "prior_influence": 1e0,
                        "tikh_factor": 1e-4,
                        "min_value": 10,
                        "max_value": 190,
                    },
                },
                "prior": {
                    "extinction_per_m": {"type": "testing"},
                    "median_radius": {"type": "constant", "value": 80},
                },
            },
        },
        "surface": {
            "lambertian_albedo": {
                "prior_influence": 0,
                "tikh_factor": 1e-4,
                "log_space": False,
                "wavelengths": sample_wavel,
                "initial_value": 0.3,
                "out_bounds_mode": "extend",
            },
        },
    },
    model_kwargs={
        "num_stokes": 3,
        "multiple_scatter_source": sk.MultipleScatterSource.DiscreteOrdinates,
        "num_streams": 8,
    },
    forward_model_cfg={
        "*": {
            "kwargs": {"stokes_sensitivities": {"vert": np.array([0.5, 0.5, 0, 0]), "horiz": np.array([0.5, -0.5, 0, 0])}}
        }
    },
)

results = ret.retrieve()

skr.plotting.plot_state(results, "stratospheric_aerosol_median_radius", show=False)
plt.subplot(1, 2, 1)
plt.plot(80 * aerosol_scale, altitude_grid, "k--")
plt.legend(["Retrieved", "Prior", "Truth"])
plt.show()
```

And we see that the results are better, with a more consistent surface albedo.


```{code-cell}
results["state"]["lambertian_albedo"]
```

To enable polarization we added the lines

```
    forward_model_cfg={
        "*": {
            "kwargs": {"stokes_sensitivities": {"vert": np.array([0.5, 0.5, 0, 0]), "horiz": np.array([0.5, -0.5, 0, 0])}}
        }
    },
```

Which can be a bit confusing, so we can unpack what is going on.  The first level of the dictionary `"*"` indicates that
this configuration applies to every measurement in our observation.  This is important when our observation contains
multiple measurements that require separate forward models and we want to configure them separately.

The `"kwargs"` dictionary is the list of options that is passed to our forward model. In this case we are using
the default forward model, which uses the  {py:class}`skretrieval.retrieval.forwardmodel.SpectrometerMixin` instrument model.  If we look at that
forward model we see it has a `stokes_sensitivities` input argument that we can set.
