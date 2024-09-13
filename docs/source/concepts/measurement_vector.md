(_measurement_vector)=
# Measurement Vectors
The observation provides what we call the "Core Radiance Format", or colloquially, L1 data.
Often it is desired to not directly use L1 data in the retrieval process, but rather some transformation or subset of the L1 data.
This could be something like a microwindow, high-altitude normalized radiances, wavelength normalized radiances, etc.
We call the result of this transformation the measurement vector, and sometimes even refer to the process of going from L1 data
to the quantities used in the retrieval the measurement vector.

Measurement vectors are passed into the `Retrieval` object as a dictionary of {py:class}`skretrieval.measvec.MeasurementVector`
objects.

## Pre-defined Measurement Vectors
The simplest way to define the measurement vector is to use some of `skretrieval`'s pre-built measurement vectors.

```{eval-rst}
.. autosummary::
    skretrieval.measvec.Triplet
```

## Measurement Vector Filtering
A `MeasurementVector` consists of a function that transforms the L1 data, and a parameter that
is named `apply_to_filter` that defaults to "*".  The filter can be used to restrict the measurement
vector to apply to one certain measurements.  If the L1 data contains multiple `RadianceGridded` objects
such as `{"one": ..., "two": ...}` then if we set `apply_to_filter="one"` the measurement vector
will only apply to the first `RadianceGridded` object.  This can be useful in synergistic applications where
you may want to define multiple measurement vectors for different instruments.

## Manually Defining the Measurement Vector
In some cases you may want to define your own measurement vector, in these cases you can supply your own
user created function that transform the L1 data, however this can be more complicated than it seems.
In addition to transforming the radiances, the transformation must also be applied to the Jacobian matrix, performing standard derivative propagation rules.
The measurement error covariance matrix must also be propagated using Gaussian error propagation rules.
To make this simpler, `skretrieval` provides convenience methods to handle these transformations for you.
They can be composed to create more complicated measurement vector transformations.

See [Measurement Vector API](_api_measurement_vector) for more details.
