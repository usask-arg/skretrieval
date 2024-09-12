---
file_format: mystnb
---

(_example_limb_scatter_ozone)=
# Limb Scatter Ozone Retrieval

Here we set up a limb scatter ozone retrieval.  The core ideas here are that we use the
{py:class}`skr.observation.SimulatedLimbObservation` to define our observation, and that instead of using
the full spectrum in the retrieval we define discrete wavelength "triplets" as the measurement vector.

```{code-cell}
import numpy as np
import skretrieval as skr


# Measurement tangent altitudes
tan_alts = np.arange(10000, 66000, 1000)

# Triplets
triplets = {
    "uv_1": {
        "wavelength": [302, 351],
        "weights": [1, -1],
        "altitude_range": [45000, 60000],
        "normalization_range": [60000, 65000],
    },
    "uv_2": {
        "wavelength": [312, 351],
        "weights": [1, -1],
        "altitude_range": [40000, 60000],
        "normalization_range": [60000, 65000],
    },
    "uv_3": {
        "wavelength": [322, 351],
        "weights": [1, -1],
        "altitude_range": [30000, 50000],
        "normalization_range": [60000, 65000],
    },
    "vis": {
        "wavelength": [525, 600, 675],
        "weights": [-0.5, 1, -0.5],
        "altitude_range": [0, 35000],
        "normalization_range": [35000, 40000],
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
    state_adjustment_factors={"o3": 1.5},  # Simulate with 1.5x the ozone of the prior
)

# Construct our measurement vectors
meas_vec = {}
for name, t in triplets.items():
    meas_vec[name] = skr.measvec.Triplet(
        t["wavelength"], t["weights"], t["altitude_range"], t["normalization_range"]
    )

# Set up the retrieval object
ret = skr.Retrieval(
    obs,
    measvec=meas_vec,
    minimizer="rodgers",
    target_kwargs={"rescale_state_space": True},
    state_kwargs={
        "altitude_grid": np.arange(0, 70000, 1000),
        "absorbers": {
            "o3": {
                "prior_influence": 1e-1,
                "tikh_factor": 1e-1,
                "log_space": False,
                "min_value": 0,
                "max_value": 1,
                "prior": {"type": "mipas"},
            },
        },
        "aerosols": {},
    },
)

# Do the retrieval
results = ret.retrieve()

# Plot the results
skr.plotting.plot_state(results, "o3_vmr", show=True)

```
