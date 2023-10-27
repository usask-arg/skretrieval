.. _forwardmodel:

******************
The Forward Model
******************
The forward model is one of the key concepts in the retrieval process, and is the core object for simulating measurements.
The job of the forward model is relatively simple, simulate L1 data.
While the internals of the forward model are usually not important for the retrieval, most problems tend to be fall into a similar structure,
where the forward model is composed of

1. A :py:class:`Sensor <skretrieval.core.sensor.Sensor>`
2. An instance of the `SASKTRAN` radiative transfer model
3. Information on the measurement geometry
4. A measurement simulator

The Sensor
==========
The :py:class:`Sensor <skretrieval.core.sensor.Sensor>` object converts radiance that is incident on the front aperture
of an instrument to L1 data.

SASKTRAN
========
The forward model is responsible for both configuring SASKTRAN and running it.
Usually the foward model takes the measurement geometry, which is the geometry of each individual measurement
of an instrument, and constructs the model geometry, a high resolution geometry that spans the field of view of all
of the measurements.
A similar thing is done on the wavelengths to create model wavelengths.
SASKTRAN is then run on the model geometry and wavelengths to generate model radiances intended to be used
at the front aperture of a :py:class:`Sensor <skretrieval.core.sensor.Sensor>`.

The Measurement Geometry
========================
The measurement geometry is the geometry of the actual measurements of each individual measurement of an
instrument.
It is represented as a list of :py:class:`OpticalGeometry <skretrieval.core.OpticalGeometry>` objects, which each
contain

 - A look vector
 - A local up direction, used to define orientation
 - An observer position
 - A time

The Measurement Simulator
=========================
The measurement simulator is responsible for interfacing all of the above components together to create
an effective forward model.  Usually the core part of this is taking the measurement geometry information,
and calculating model geometries.  These are usually calculated based upon options specified by the user, such
as the desired resolution to integrate over in field of view calculations.  The measurement simulator is often
specialized for a single type of measurement geometry, e.g., information from a scanning satellite.
The measurement simulator may also be specialized for an individual :py:class:`Sensor <skretrieval.core.sensor.Sensor>`,
e.g., a simulator for CATS may need to set up SASKTRAN differently because of the fact that there are five slits.
