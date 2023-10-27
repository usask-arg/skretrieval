.. _tomography_target:

#############################
Tomographic Retrieval Aspects
#############################
Fundamentally the retrieval equations are identical for a one-dimensional retrieval and a two-dimensional retrieval.
The challenges for tomography are handling the additional complexity of the state and measurement vectors.  This is
compounded by the fact that the state and measurement vectors are, in fact, vectors.  The two-dimensional nature
of the problem is lost once the measurement and state vectors are flattened.  To add to the complications, often the
measurement vector cannot even be written as a two-dimensional matrix since the shape of the measurement vector
may be different for every image along the retrieval track.

The tomography module is designed so that the class :class:`~.TwoDimTarget` abstracts away much of the annoyances
with handling two-dimensional data.  A tomographically retrieved target species will inherit from the :class:`~.TwoDimTarget`
and implement several abstract functions which are necessary to perform the retrieval.  The goal is so that, as much
as possible, the two-dimensional nature of the problem can be ignored.  For example, user code only has to implement
the calculation of the measurement vector for a single images worth of data, the combination of the images is handled
by the :class:`~.TwoDimTarget`.

Format of Radiance (L1) Data
============================
Radiances used in tomography are expressed as a :class:`~.RadianceOrbit`.  This class is essentially just a list of
other radiance objects, one for each image from the instrument.  The class is necessary rather than using a list since
special care has to be used in handling the weighting functions.  The weighting function matrix cannot be stored
densely for every image because of memory requirements, and is thus stored sparsely.  However sparse matrices cannot
be placed inside an xarray (or netcdf) object and require special handling.  It is usually quite easy to convert
the output of :class:`~.EngineHRTwoDim` to a :class:`~.RadianceOrbit` object within the forward model.

Weighting Functions
-------------------
When calculating the measurement vector it is very convenient to have the weighting functions and radiances be
part of the same xarray object since it makes computations of the jacobian simpler.  For this reason when accessing
single image L1 data the weighting function is converted from sparse storage to dense storage and added in to the L1
object.  While it is not possible to store the entire weighting function densely, we can store it densely for a single
image.

Grids and Edges in the Retrieval
================================
A typical ``retrieval'' grid can have an upper and lower bound that varies along the retrieval track.
However, the observed radiance is dependent on values of the target outside the edges of this grid.
One common way to deal with this is to retrieve only on the ``good'' points, and then extrapolate the scaling factor
beyond the edges of the grid.  This method works well, but results in poor convergence at the edges of the grid since
the jacobian does not contain all of the information on how changing these edge points scales the profile outside
of the retrieval grid.  The jacobian could be modified to include this information, but it requires cognitively complicated
transformations.  An alternative way (and the preferred way) to deal with this problem is to retrieve on the full portion of the atmosphere
that affects the measurements, but enforce the shape outside the good retrieval points using apriori information.

Grid Definitions
----------------
Before dealing with how a-priori boundary scaling is implemented we introduce definitions for two grids.

The portion of the atmosphere that affects the measurements is called the *state vector grid*.  The state vector
is specified on this grid, and weighting functions are calculated on it.  This grid can always be written as a
two-dimensional array (the same number of vertical points for each horizontal point).  The climatology contains
values on this grid, and typically they match one-to-one.

The subset of the state vector grid that we care about, and believe we can actually retrieve at, is called the *retrieval grid*.
This can be determined from the state vector grid, along with the lower and upper bound as a function of grid point.
The usually cannot be written as a two-dimensional array. :py:class:`TwoDimTarget <skretrieval.tomography.target.TwoDimTarget>`
implements a function providing a mask to indicate what elements of the state vector grid make up the retrieval grid.

a-priori Boundary Scaling
-------------------------
Our goal is to use a-priori information so that every point in the retrieval grid is (to first order) unaffected
by the a-priori information, and that every grid point in the state vector grid outside the retrieval grid is pinned
to the closest edge.  Here pinning means that for any retrieval iteration it is scaled by the same factor.
The proper form of a-priori information is a constraint on the first derivative of the profile beginning at the edge
and extending out to the end of the grid.  If the state vector is specified as logarithm of number density including
this term is relatively straight forward.

As mentioned, a-priori boundary scaling improves the convergence of the retrieval, but there are other benefits.
The averaging kernel properly contains information outside the retrieval grid and error propagation is more robust.

What the User has to Implement
==============================
Derived classes of :py:class:`TwoDimTarget <skretrieval.tomography.target.TwoDimTarget>` must implement several derived functions for the tomographic retrieval to work.

Calculation of the Measurement Vector for a Single Image
--------------------------------------------------------
The target must implement :func:`~skretrieval.tomography.target.TwoDimTarget._image_measurement_vector` which
takes as input the L1 data for a single image and returns back the measurement vector (and jacobian/error estimates if applicable).
This function can be essentially identical to that of a one-dimensional retrieval.  The weighting functions
included in the L1 data will contain information for the entire grid, but generally the number of perturbations
in the weighting function does not have an impact on how the jacobian is calculated.

Determination of Upper and Lower Bounds
---------------------------------------
Derived classes have to specify lower and upper bounds for the retrieval, and this is one of the aspects of the problem
where the two-dimensional nature of the retrieval cannot be completely hidden.  The two functions
:func:`~skretrieval.tomography.target.TwoDimTarget._lower_wf_bound` and :func:`~skretrieval.tomography.target.TwoDimTarget._upper_wf_bound`
are fairly simple to implement and determine the grid which makes up the state vector.  Since it is recommended
to use apriori bounding to handle edges, usually there is no reason to not have the lower_wf_bound be 0 m, and the
upper_wf_bound to be as high as the profile is non-zero at.

The two functions :func:`~skretrieval.tomography.target.TwoDimTarget._lower_retrieval_bound` and :func:`~skretrieval.tomography.target.TwoDimTarget._upper_retrieval_bound`
determine the actual altitudes that are useful for the retrieval and are functions of *grid* index.  Simple retrieval
problems may have these as constant, or may determine them from outside sources such as the tropopause altitude.
More complicated retrieval problems can set the upper and lower bounds based upon the actual image radiances themselves.
In this case it is necessary to calculate upper and lower bounds for the images, and then interpolate them to the
grid in order to determine upper and lower boundaries.
