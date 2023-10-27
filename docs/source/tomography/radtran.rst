.. _tomography_radtran:

#################################
Radiative Transfer for Tomography
#################################
Fundamentally radiative transfer for tomography requires two-dimensional radiative transfer.  SASKTRAN-HR supports
two-dimensional radiative transfer, but the support is limited to when the atmosphere variation forms a plane in
the along sight dimension.  As we have seen due to the Earth's rotation the tomographic grid does not form a perfect
plane, however locally it does.  This means that for each radiative transfer calculation we must transform the orbital
plane grid to a local plane, perform the calculation there, and possibly transform calculated weighting functions
back to the orbital plane grid.

Tomography also offers an interesting potential speedup in radiative transfer relative to a one-dimensional calculation.
The slow part of the radiative transfer calculation is calculation of the multiple scatter source, which is independent
of the number of lines of sight included in the calculation.  This means that if we combine multiple images into the same
radiative transfer calculation it is faster than calculating them separately.  The problem with this is that subsequent
images happen at different instances of time and thus the sun is at a different location.  However, if the images are
not too far apart in time there is little error introduced by assuming a mean sun position.

Both the local gridding and image consolidation are handled by the class :class:`~.EngineHRTwoDim`.


Specifying the Atmosphere
=========================
Specifying the atmosphere for the two-dimensional calculation is relatively easy and is handled through a normal
climatology object, :class:`~.OrbitalPlaneClimatology`.  Relative to a normal one-dimensional calculation
there is nothing new that needs to be done.

Modelling the Radiance
======================
Modelling the radiance is usually straightforward.  The interface of :class:`~.EngineHRTwoDim` is similar to that of
a regular Engine object.


Combining Subsequent Images
===========================
As mentioned earlier subsequent images can be combined into the radiative transfer calculation in order to speed up the
modelling of an orbit.  This is performed automatically and is controlled by the `maxdifference_seconds` option in
:class:`~.EngineHRTwoDim`.  The default value of 100 s is approximately the time it takes for the sun to move the apparent
length of the solar disk.

There is a second issue with combining images together, over the course of a single radiative transfer calculation
the orbital plane grid is approximated locally as a plane.  The longer the orbital track spanned by the images
forming the calculation the worse this approximation will be.
