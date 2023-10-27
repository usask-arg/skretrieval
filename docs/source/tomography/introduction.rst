.. _tomography:

##########
Tomography
##########
A tomographic retrieval combines subsequent measurements together in order to account for the horizontal
variation of the atmosphere.  The philosophy of the inverse problem is identical to the one-dimensional case, we want
to find the state vector that best fits the measurement vector given some prior information.  The additional complexity
of tomography comes in with the definition of the state vector and the measurement vector.
The state vector is a two-dimensional profile, typically in altitude and angle along
the orbit plane, of the quantity of interest instead of only containing altitude variations.
The measurement vector contains information from subsequent measurements together, the number of measurements to
combine together can vary but often is an entire orbits worth.

The `tomography` module provides a framework to perform the tomographic retrieval and abstract away many of
the inconveniences in working in two-dimensions.  This involves constructing the orbital plane grid, modelling the
radiance with a 2d atmosphere along the orbital track, and doing the retrieval.

.. toctree::
   :maxdepth: 2

   orbitalgrid
   radtran
   retrievaltarget
