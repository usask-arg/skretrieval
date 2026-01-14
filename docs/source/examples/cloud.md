---
file_format: mystnb
---

(_example_limb_scatter_cloud)=
# Limb Scatter Cloud Retrieval

Here we demonstrate the retrieval of cloud height, width, and optical depth.
Similar to other examples we must provide information on the optical parameters.

```{code-cell}
import numpy as np
import sasktran2 as sk
import skretrieval as skr
```

In this example we use a Mie distribution with a fixed particle size to
define our cloud optical properties.

```{code-cell}
@skr.Retrieval.register_optical_property("cloud")
def cloud_optical_prop(*args, **kwargs):
    refrac = sk.mie.refractive.H2SO4()
    dist = sk.mie.distribution.LogNormalDistribution().freeze(
        mode_width=1.6, median_radius=80.0,
    )

    db = sk.database.MieDatabase(
        dist,
        refrac,
        np.array([469.0, 750.0]),
    )

    return db
```

We configure the state vector for the three parameters (`vertical_optical_depth`, `width_fwhm_m`, `height_m`)
required by the Gaussian extinction profile constituent.

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
    state_adjustment_factors={
        "cloud": {
            "vertical_optical_depth": 1.5,
            "width_fwhm_m": 1.5,
            "height_m": 1.5,
        }
    },
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
            "cloud": {
                "type": "gaussian_extinction_profile",
                "nominal_wavelength": 745,
                "retrieved_quantities": {
                    "vertical_optical_depth": {
                        "prior_influence": 1e0,
                        "tikh_factor": 1e-2,
                        "min_value": 1e-9,
                        "max_value": 10
                    },
                    "width_fwhm_m": {
                        "prior_influence": 1e0,
                        "tikh_factor": 1e-2,
                        "min_value": 1e-9,
                        "max_value": 20000
                    },
                    "height_m": {
                        "prior_influence": 1e0,
                        "tikh_factor": 1e-2,
                        "min_value": 0,
                        "max_value": 25000
                    },
                },
                "prior": {
                    "vertical_optical_depth": {"type": "constant", "value": 0.05},
                    "width_fwhm_m": {"type": "constant", "value": 5000},
                    "height_m": {"type": "constant", "value": 15000},
                }
            },
        },
    },
)

# Do the retrieval
results = ret.retrieve()

# Display the results
print(results["state"])
```
