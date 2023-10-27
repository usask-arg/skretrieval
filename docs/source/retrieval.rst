.. retrieval

**************
The Retrieval
**************
A retrieval problem is formed based upon three key concepts:

1. The Measurement L1: L1 Data representing a real (or simulated) measurement
2. The Forward Model: the ability to simulate the measurements of 1.
3. The Retrieval Target: A definition of the quantity to be retrieved

The Measurement L1
==================
Measurement L1 is data, typically radiances, that the instrument sees, and is an instance of
:py:class:`RadianceBase <skretrieval.core.radianceformat.RadianceBase>`.
Typically the measurement L1 will be a specialized derived class of :py:class:`RadianceBase <skretrieval.core.radianceformat.RadianceBase>`.
An example is :py:class:`RadianceGridded <skretrieval.core.radianceformat.RadianceGridded>` which specifies that
the radiance may be expressed as a 2D array of (wavelength, line of sight), such as what is recorded by a scanning
spectrograph.  L1 data need not only contain radiance values, it may also contain information on the precision of
the measurements, or other things such as field of view.

The format of the measurement L1 is coupled with the choice of retrieval target (see below).  For example, one
may write a `CATSNO2` retrieval target class, that is only compatible with L1 data in a specific form, such as
`RadianceCATS`.  Typically each instrument would have it's own L1 format, unless it falls under one of the generic
formats already available.

Measurement L1 could be loaded in from a file, if it is actual data recorded from a instrument.
Or it could be created through simulations.

The Forward Model
=================
The :py:class:`ForwardModel <skretrieval.retrieval.ForwardModel>` is a class which has been presetup to simulate
the exact same data specified by the measurement L1 (in the same format).
Sometimes it is possible that the data produced by the :py:class:`ForwardModel <skretrieval.retrieval.ForwardModel>` will
have additional fields, such is the case when performing a retrieval that uses weighting functions; the weighting functions
are included in the L1 data output by the :py:class:`ForwardModel <skretrieval.retrieval.ForwardModel>`.

The Retrieval Target
====================
The :py:class:`RetrievalTarget <skretrieval.retrieval.RetrievalTarget>` defines the quantity that is to be retrieved.
:py:class:`OzoneRetrieval <skretrieval.retrieval.ozone.OzoneRetrieval>` provides an example for a simple ozone
retrieval from limb style measurements.
The :py:class:`RetrievalTarget <skretrieval.retrieval.RetrievalTarget>` is responsible for a few things

 - Defining the state vector, the quantity to be retrieved
 - Defining the measurement vector, a transformation of the input L1 radiances, and the equivalent transformation
   of weighting functions if necessary
 - Updating the state vector.  This must be propagated to the :py:class:`ForwardModel <skretrieval.retrieval.ForwardModel>`
 - Specifying the a priori state (and covariance).

Typically each :py:class:`RetrievalTarget <skretrieval.retrieval.RetrievalTarget>` is instrument specific.

Doing the Retrieval
-------------------
With the three above concepts defined and available, a retrieval can be performed.
The generic class that does this operation is the :py:class:`Minimizer <skretrieval.retrieval.Minimizer>`, and
a specific implementation can be found in :py:class:`Rodgers <skretrieval.retrieval.rodgers.Rodgers>`.
Most limb inverse problem type retrievals operate in a similar fashion:

1. Simulate measurements using the :py:class:`ForwardModel <skretrieval.retrieval.ForwardModel>`
2. Calculate the measurement vector for both the observation and the simulation using the :py:class:`RetrievalTarget <skretrieval.retrieval.RetrievalTarget>`
3. The :py:class:`Minimizer <skretrieval.retrieval.Minimizer>` determines how much the quantity to be retrieved should change by
4. Use the :py:class:`RetrievalTarget <skretrieval.retrieval.RetrievalTarget>` to update the :py:class:`ForwardModel <skretrieval.retrieval.ForwardModel>` with the new parameters
5. Iterate until convergence
