(_observation)=
# Observations
An observation in `skretrieval` is typically radiance measurements from one (or multiple) instruments viewing the atmosphere.
These can be real measurements from actual instruments, or simulated measurements.
In the case of real measurements defining an observation is usually a matter of loading in the data and manipulating the format.
In the case of simulated observations, it can involve specifying additional information about the exact simulation that is being performed.
In all cases, an observation follows the base class interface {py:class}`skretrieval.observation.Observation`.

## Core Radiance Format
While `skretrieval` is flexible in allowing the user to define any format they wish for the observed radiances, several aspects of the retrieval depend on this format.
The forward model must produce radiances in the same format as the observations, and the measurement vector calculation has to be aware of the format the radiances are in.
`skretrieval` defines a generic radiance format that (almost) all atmospheric measurements are able to fit under which we call the "Core Radiance Format".

Briefly, this format is a `dict` of {py:class}`skretrieval.core.radianceformat.RadianceGridded` objects.
A `RadianceGridded` object is basically just a 2-D array where the radiance has dimension (spectral, line of sight).
The primary purpose of the `Observation` object is to convert the observations into this format.
We have found that measurements from almost every remote sensing instrument can fit into this format.
For example:

- A nadir viewing spectrometer with a single look direction consists of a one `RadianceGridded` object with dimension (spectral, 1).
- A limb scanning spectrometer fits in one `RadianceGridded` object with dimension (spectral, tangent_altitude)
- A limb imaging spectrometer fits in one `RadianceGidded` object with dimension (spectral, tangent_altitude)
- A limb imaging instrument that discretely picks different wavelengths fits in multiple `RadianceGridded` objects (one for each wavelength) with dimensions (1, N) where N spans the detector

In some cases this format is the most convenient format for the instrument in question, in some cases it is not the most convenient.
However, even if inconvenient, we recommend converting your data into this format so that components of `skretrieval` can be used.

### Extra Data
The "Core Radiance Format" defines the minimal set of information required for a basic retrieval problem, however in many cases you may need to provide more information.
The `Observation` can add any extra information it wants to the `RadianceGridded` object that can be accessed in other parts of the retrieval, however this information
is not automatically added in to data calculated by the forward model.
In this case the method `append_information_to_l1()` can be implemented in the observation, which will be called after the simulated L1 data is calculated.


## Defining Real Observations
The first step to defining a real observation is to create a child class which inherits from {py:class}`skretrieval.observation.Observation`.
The method `skretrieval_l1()` must then be implemented which returns back radiances in the "Core Radiance Format".
The second responsibility is providing any extra information that the forward model may need in order to simulate the viewing geometry of the observations.
Exactly what information this is depends on the forward model being used, for example, if you are using the `skretrieval.forwardmodel.IdealViewingSpectrograph`
forward model, it requires that the `Observation` implement the `sasktran_geometry()` method which returns back a `SASKTRAN2` viewing geometry
to use the simulation.

## Defining Simulated Observations
Simulated observations can be performed by simulated real measurements outside of `skretrieval`, creating data files, and using the same processing code you would for real observations.
However `skretrieval` also provides several convenient simulated observation classes that can be used for quick calculations.
These simulations typically piggy back off the forward model and state vector defined in the retrieval, modifying them in slight ways to simulate observations.
Each simulation class is slightly different, and will require different information to use.

### Simulation Classes
```{eval-rst}
.. autosummary::
    skretrieval.observation.SimulatedLimbObservation
    skretrieval.observation.SimulatedNadirObservation
```

## Defining Real Observations
Real observations are created through creating your own class that inherits from {py:class}`skretrieval.observation.Observation`. You must then
implement the `skretrieval_l1` and `sasktran_geometry` methods which return back the "Core radiance format" as well as information
on the ideal measurement geometry.
