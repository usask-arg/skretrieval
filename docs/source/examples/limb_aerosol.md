---
file_format: mystnb
---

(_example_limb_scatter_aerosol)=
# Limb Scatter Aerosol Retrieval

Here we set up a limb scatter aerosol retrieval. This is similar to the ozone retrieval example
except for aerosol we have to pass in information on the optical parameters of the aerosol
we are trying to retrieve.

```{code-cell}
import numpy as np
import sasktran2 as sk
import skretrieval as skr
```

To register the optical property for our aerosol, we use the decorator method to register it with
the retrieval.

```{code-cell}
@skr.Retrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_prop(*args, **kwargs):
    refrac = sk.mie.refractive.H2SO4()
    dist = sk.mie.distribution.LogNormalDistribution().freeze(
        mode_width=1.6, median_radius=80
    )

    db = sk.database.MieDatabase(
        dist,
        refrac,
        np.arange(250.0, 1501.0, 50.0)
    )

    return db
```

And then we can proceed with the aerosol retrieval as normal

```{code-cell}
# Measurement tangent altitudes
tan_alts = np.arange(10000, 45000, 1000)

# Triplets
triplets = {
    "745": {
        "wavelength": [745],
        "weights": [1],
        "altitude_range": [0, 40000],
        "normalization_range": [40000, 45000],
    },
}

# Get the wavelengths required for the triplets
wavel = np.unique(np.concatenate([triplets[t]["wavelength"] for t in triplets])).astype(float)

# Set up a simulated observation with our tangent altitudes, wavelengths, and use 1.5x the initial guess for ozone
obs = skr.observation.SimulatedLimbObservation(
    cos_sza=0.2,
    relative_azimuth=0,
    observer_altitude=200000,
    reference_latitude=20,
    reference_longitude=20,
    tangent_altitudes=tan_alts,
    sample_wavelengths=wavel,
    state_adjustment_factors={"stratospheric_aerosol": 1.5},  # Simulate with 1.5x the prior
)

# Construct our measurement vectors
meas_vec = {}
for name, t in triplets.items():
    meas_vec[name] = skr.measvec.Triplet(
        t["wavelength"], t["weights"], t["altitude_range"], t["normalization_range"], normalize=False
    )

# Set up the retrieval object
ret = skr.Retrieval(
    obs,
    measvec=meas_vec,
    minimizer="rodgers",
    target_kwargs={"rescale_state_space": True},
    state_kwargs={
        "altitude_grid": np.arange(0, 70000, 1000),
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
                },
                "prior": {
                    "extinction_per_m": {"type": "testing"},
                }
            },
        },
    },
)

# Do the retrieval
results = ret.retrieve()

# Plot the results
skr.plotting.plot_state(results, "stratospheric_aerosol_extinction_per_m", show=True)
```

## Retrieving Aerosol Microphysical Parameters
To include extra parameters in the retrieval, we have to adjust two things. We need to recreate the optical
property database to include variations of that parameter, and we also need to adjust our state configuration object
to indicate that we want to retrieve it.

We will start by making the database depend on `median_radius`

```{code-cell}
@skr.Retrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_prop(*args, **kwargs):
    refrac = sk.mie.refractive.H2SO4()
    dist = sk.mie.distribution.LogNormalDistribution().freeze(
        mode_width=1.6
    )

    db = sk.database.MieDatabase(
        dist,
        refrac,
        np.array([469.0, 750.0]),
        median_radius = np.arange(10, 200, 10.0)
    )

    return db
```

Then we will repeat the retrieval code, adding a second triplet so that we have two wavelengths in our measurements, and
updating the state config to indicate that we want to retrieve median radius alongside extinction.

```{code-cell}
# Measurement tangent altitudes
tan_alts = np.arange(10000, 45000, 1000)

# Triplets
triplets = {
    "745": {
        "wavelength": [745],
        "weights": [1],
        "altitude_range": [0, 40000],
        "normalization_range": [40000, 45000],
    },
    "470": {
        "wavelength": [470],
        "weights": [1],
        "altitude_range": [0, 40000],
        "normalization_range": [40000, 45000],
    },
}

# Get the wavelengths required for the triplets
wavel = np.unique(np.concatenate([triplets[t]["wavelength"] for t in triplets])).astype(float)

# Set up a simulated observation with our tangent altitudes, wavelengths, and use 1.5x the initial guess for ozone
obs = skr.observation.SimulatedLimbObservation(
    cos_sza=0.2,
    relative_azimuth=0,
    observer_altitude=200000,
    reference_latitude=20,
    reference_longitude=20,
    tangent_altitudes=tan_alts,
    sample_wavelengths=wavel,
    state_adjustment_factors={"stratospheric_aerosol": 1.5},  # Simulate with 1.5x the prior
)

# Construct our measurement vectors
meas_vec = {}
for name, t in triplets.items():
    meas_vec[name] = skr.measvec.Triplet(
        t["wavelength"], t["weights"], t["altitude_range"], t["normalization_range"], normalize=False
    )

# Set up the retrieval object
ret = skr.Retrieval(
    obs,
    measvec=meas_vec,
    minimizer="rodgers",
    target_kwargs={"rescale_state_space": True},
    state_kwargs={
        "altitude_grid": np.arange(0, 70000, 1000),
        "aerosols": {
            "stratospheric_aerosol": {
                "type": "extinction_profile",
                "nominal_wavelength": 745,
                "retrieved_quantities": {
                    "extinction_per_m": {
                        "prior_influence": 1e-1,
                        "tikh_factor": 1e-1,
                        "min_value": 0,
                        "max_value": 1e-3
                    },
                    "median_radius": {
                        "prior_influence": 1e0,
                        "tikh_factor": 1e-4,
                        "min_value": 10,
                        "max_value": 900
                    }
                },
                "prior": {
                    "extinction_per_m": {"type": "testing"},
                    "median_radius": {"type": "constant", "value": 80}
                }
            },
        },
    },
)

# Do the retrieval
results = ret.retrieve()

# Plot the results
skr.plotting.plot_state(results, "stratospheric_aerosol_extinction_per_m", show=True)

skr.plotting.plot_state(results, "stratospheric_aerosol_median_radius", show=True)

```
