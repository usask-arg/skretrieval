.. _controlframes:


Control Frames
===============
The ``skretrieval.platforms`` package uses the following coordinate systems during its analysis

- Instrument Control Frame (ICF)
- Platform Control Frame (PCF)
- Geodetic Control Frame (GCF)
- Geographic Geocentric (ECEF)

Instruments coordinates are defined in the Instrument Control Frame with :math:`\hat{x}` along the optical-axis or boresight
of the instrument. Fields of view and lines of sight emanating from the instrument are canonically described in this
coordinate frame. The goals of the transforms are,

 - Convert unit vectors from the instrument control frame to the geographic geocentric control frame for easy use by radiative transfer code.
 - Provide methods to orient the instrument so it is looking at specific points on the Earth's surface or atmosphere.

The **instrument** is mounted on a **platform** and by default the instrument's optical axis is aligned
with the platform's :math:`\hat{x}` unit vector, which is often the *forward* direction but can be arbitrary.
The instrument's mounting orientation on the platform is changed by setting the instrument's *azimuth*, *elevation* and *roll*.

The platform is initially configured so its *forward* unit vector, :math:`\hat{x}`, is looking North, regardless of location.
The platform can be rotated to point in other directions by setting its *yaw*, *pitch* and *roll*. Simulator code may avoid setting
yaw, pitch and roll directly and instead invoke helper functions to set them so the instrument is looking towards a specific point
on the Earth's surface or atmosphere.

The platform pointing coordinates are converted from geodetic coordinates to geographic-geocentric coordinates once the platform's
location in space is determined by specifying *latitude*, *longitude* and *height*.

.. _icf:

Instrument Control Frame (ICF)
------------------------------------
The Instrument Control Frame (ICF) specifies a 3 axis right-handed system that is fixed to the instrument. Lines of sight
and field of views are defined in this control frame. These entities are typically rotated into the platform control frame
and from there rotated to look at a given target point.

  .. image:: figures/ICF_3axes.png
     :scale: 50 %
     :alt: Instrument Control Frame

The :math:`\hat{x}` unit vector of the ICF is arranaged so it parallel to the nominal optic axis or boresight of the
instrument and points away from the instrument.  The :math:`\hat{y}` and :math:`\hat{z}` unit vectors are in the plane
of the entrance aperture and :math:`\hat{z}` is placed so it has an upwards component when the instrument is sitting on
a table in a lab. The :math:`\hat{y}` is chosen to form the third axis of a right-handed system but is often placed so
it is horizontal.


 .. image:: figures/ICF_initial_orientation.png
   :scale: 100 %
   :alt: Instrument Control Frame

The instrument control frame is initialized so it is aligned upwards with resepect to the platform control frame with the
boresight of the instrument :math:`\hat{x}_{ICF}` parallel to :math:`\hat{x}_{PCF}`. The respective :math:`\hat{y}` and :math:`\hat{z}`
unit vectors are anti-parallel.

All rotations, typically azimuth and elevation, that are applied to place the instrument into its mounted position in the platform
control frame should start by assuming the instrument control frame is in its initial orientation with respect to the
platform control frame.

.. _pcf:

Platform Control Frame (PCF)
----------------------------
The Platform Control Frame (PCF) specifies the coordinate system used by the platform that the instrument is mounted too.
Typical platforms are satellites, aircraft, gondola as well as observatories on the ground.  Platforms will define a 3 axis
right-handed system. For aircraft the :math:`\hat{x}` is parallel to the body of the plane and points forward,
:math:`\hat{z}` is perpendicular to the body and wings and points downward when the plane is flying level, :math:`\hat{y}`
is parallel to the wings and points to the startboard side. Satellites and balloon gondolas may choose other ways to specify
the 3 axis system but keep in mind that all rotations are right-handed and using other coordinate systems, where
:math:`\hat{z}` is up for example, may generate counter-intuitive rotations when using yaw, pitch, azimuth or elevation.

 .. image:: figures/PCF_3axes.png
   :scale: 50 %
   :alt: Platform Control Frame

All platforms are able to express the 3 unit vectors in terms of local geodetic coordinates (west,south up) or
geographic geocentric coordinates.

The Platform Control Frame is initialized so :math:`\hat{x}_{PCF}` is co-aligned with local North, :math:`\hat{y}_{PCF}`
is pointing East and :math:`\hat{z}_{PCF}` is pointing downwards.

 .. image:: figures/PCF_initial_orientation.png
   :scale: 100 %
   :alt: Platform Control Frame


.. _gcf:

Geodetic Control Frame (GCF)
----------------------------
The geodetic control frame is a 3 axis right-handed system defined at any location by the geodetic latitude and
longitude.

 .. image:: figures/GCF_3axes.png
   :scale: 50 %
   :alt: Geodetic Control Frame

The :math:`\hat{x}` is given by local east, :math:`\hat{y}` is given by local north and :math:`\hat{z}` is
given by local up. This coordinate system is based upon an oblate spheroid geoid where :math:`\hat{z}` is
perpendicular to the surface of the oblate spheroid and does not usually pass through the center of the Earth.


..  _ecef:

Geocentric Control Frame (ECEF)
-------------------------------
The geocentric control frame is a geographic coordinate system with its origin at the centre of the (oblate spheroid)
Earth. The system is synonomous with the `ITRF <https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame>`_ system
using the WGS84 reference sphere. The :math:`\hat{z}` is parallel to the rotation axis of Earth and points from the center through the North pole.
The :math:`\hat{x}` points in the plane of the equator from the center of the Earth to the Greenwich meridian (in the
Atlantic ocean just off the cosat of Africa). The :math:`\hat{y}` forms the third axis of a right-handed system and points
from the center in the plane of the equator to the 90E meridian (in the Indian ocean west of Sumatra). The system rotates
with the Earth.
