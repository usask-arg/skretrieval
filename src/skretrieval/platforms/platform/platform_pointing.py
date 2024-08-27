from __future__ import annotations

import logging
import math

import numpy as np

from skretrieval.geodetic import geodetic

from .rotationmatrix import RotationMatrix, UnitVectors


# -----------------------------------------------------------------------------
#           Class PlatformPointing
# -----------------------------------------------------------------------------
class PlatformPointing:

    """
    A class used to manage the pointing requirements of an **instrument** mounted on a **platform**. The primary objective
    is to

    * Mount an instrument on a platform, eg balloon, aircraft, satellite, ground.
    * Place and orient the platform at a location in space, on the ground or in the atmosphere.
    * Generate unit vectors at the front aperture of the instrument representing lines of sight and similar that can be
      used by the radiative transfer codes.

    The PlatformPointing class generates rotation matrices to transform between the different standard control frames as well as
    methods to mount the instrument on the platform, position the platform and rotate the platform.
    """

    # -----------------------------------------------------------------------------
    #           __init__
    # -----------------------------------------------------------------------------

    def __init__(self):
        self._ICF_Init_PCF: RotationMatrix = RotationMatrix(
            array=[[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        )  # Initial matrix to get ICF properly aligned with PCF. ICFx is +PCFx, ICFy is -PCFy and ICFz is -PCFz
        self._PCF_Init_GCF: RotationMatrix = RotationMatrix(
            array=[[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        )  # Initial matrix to get PCF properly aligned with GCF. PCF X is GCFy (due North), PCF y is GCFx (due East), PCF Z is -GCFz (upwards)
        self._IRM: RotationMatrix = (
            RotationMatrix()
        )  # Instrument Rotation Matrix. Applies instrument internal mirrors and turntables tec,
        self._ICF_to_PCF: RotationMatrix = (
            RotationMatrix()
        )  # Rotation Matrix from Instrument Control Frame to Platform Control Frame. Applies monutnig information
        self._PCF_to_GCF: RotationMatrix = (
            RotationMatrix()
        )  # Rotation Matrix from Platform Control Frame to Geodetic Control Frame. E.G. Applies the Yaw, Pitch, Roll to orient the platform in local geodetic space.
        self._GCF_to_ECEF: RotationMatrix = (
            RotationMatrix()
        )  # Rotation Matrix to convert local geodetic coordinates to geographic geocentric. ECEF
        self._utc: np.datetime64 = (
            None  #: Platform pointing also stores the UTC time as a convenience
        )
        self._geo: geodetic = geodetic()  #: An instance of sasktran Geodetic object.
        self._geolocation: np.ndarray = np.zeros([3])
        self._local_west: np.ndarray = None
        self._local_south: np.ndarray = None
        self._local_up: np.ndarray = None
        self._nadirmode: bool = False

    # ------------------------------------------------------------------------------
    #           mount_instrument_on_platform
    # ------------------------------------------------------------------------------
    def mount_instrument_on_platform(
        self,
        azimuth_degrees: float,
        elevation_degrees: float,
        roll_degrees: float = 0.0,
    ):
        """
        Mounts an instrument on a platform by setting up the azimuth, elevation and roll of the :ref:`icf`
        with respect to the :ref:`pcf`. Note that we try to use the aircraft definitions of
        azimuth, elevation and roll and we apply the three rotations in the order, azimuth first, elevation second and
        roll third,

        * azimuth is 0.0 when :math:`\\hat{x}_{ICF}` is pointing along the fuselage towards the nose,  i.e. it is parallel to :math:`\\hat{x}_{PCF}`. It is positive towards starboard (right hand).
        * elevation is 0.0 along the fuselage towards the nose when :math:`\\hat{x}_{ICF}` is parallel to :math:`\\hat{x}_{PCF}` and :math:`\\hat{z}_{ICF}` is pointing upwards. It is positive as :math:`\\hat{x}_{ICF}` tilts upwards.
        * roll is 0.0 when :math:`\\hat{y}_{ICF}` is parallel to the wings pointing to port-side (left) and is anti-parallel to :math:`\\hat{y}_{PCF}`. It is positive as :math:`\\hat{y}_{ICF}` tilts upwards and the aircraft leans to the right.

        The instrument control frame is initally configured so the instrument is mounted with azimuth, elevation and roll
        all equal to 0:  :math:`\\hat{x}_{ICF}` is parallel to :math:`\\hat{x}_{PCF}` and :math:`\\hat{z}_{ICF}` is
        anti-parallel to :math:`\\hat{z}_{PCF}`

        Parameters
        ----------
        azimuth_degrees : float
            The azimuth of the instrument boresight in the platform control frame in degrees. Azimuth is zero along the aircraft
            fuselage towards the nose and increases towards the right/starboard
        elevation_degrees : float
            The elevation of the instrument boresight in the platform control frame in degrees. Elevation is
            0.0 along the fuselage towards the nose and increases in the upward direction.
        roll_degrees : float
            The right-handed roll in degrees of the instrument around the final resting point of the boresight, :math:`\\hat{x}_{ICF}`, in the platform control frame. The rotation
            is a right-handed roll around the final. Default is 0.0
        """

        self._ICF_to_PCF.from_yaw_pitch_roll(
            math.radians(azimuth_degrees),
            math.radians(elevation_degrees),
            math.radians(roll_degrees),
        )

    # ------------------------------------------------------------------------------
    #               rotate_instrument_in_icf
    # ------------------------------------------------------------------------------

    def rotate_instrument_in_icf(
        self,
        azimuth_degrees: float,
        elevation_degrees: float,
        roll_degrees: float = 0.0,
    ):
        """
        Sets the instrument rotation matrix so it rotates the instrument control frame, :math:`(\\hat{x}_{ICF}, \\hat{y}_{ICF}, \\hat{z}_{ICF})`,
        through the given azimuth, elevation and roll angles. This method is intended to simulate tilting mirrors and rotation stages
        attached to the instrument. The rotation is applied in the order, azimuth, elevation, roll.

        Parameters
        ----------
        azimuth_degrees : float
            The azimuth of the instrument boresight in the platform control frame in degrees. Azimuth is left handed rotation
            around :math:`\\hat{z}_{ICF}`.
        elevation_degrees : float
            The elevation of the instrument boresight in the platform control frame in degrees. Elevation is
            a left-handed rotation around :math:`\\hat{y}_{ICF}`.
        roll_degrees : float
            The roll in degrees of the instrument. It is a right-handed rotation around :math:`\\hat{x}_{ICF}`. Default is 0.0
        """

        self._IRM.from_azimuth_elevation_roll(
            math.radians(azimuth_degrees),
            math.radians(elevation_degrees),
            math.radians(roll_degrees),
        )

    # -----------------------------------------------------------------------------
    #           orient_platform_in_space
    # -----------------------------------------------------------------------------

    def orient_platform_in_space(
        self, yaw_degrees: float, pitch_degrees: float, roll_degrees: float
    ):
        """
        Sets the platforms yaw, pitch and roll to orient the platform in space at its location. The yaw, pitch
        and roll are applied in a geodetic system so angles are always about the local geodetic system regardless of
        the platform location and the values stay in effect until explicitly changed.

        This function is typically not required in simulators (although it can be set if you wish) as the platform pointing
        in these cases is overridden by the need to look at a "simulated" target location. The function is useful for
        attitude solutions from actual platforms where yaw, pitch and roll are a common method of specifying platform
        orientation

        This function will orient the entire platform in space. The platform control frame is initially set so the
        interpretation of yaw, pitch and roll are sensible, see :ref:`pcf`.

        Parameters
        ----------
        yaw_degrees : float
            The yaw or geographic bearing of the platform control frame in degrees. This is a geographic bearing, N= 0, E= 90, S= 180, W = 270.
        pitch_degrees : float
            The pitch of the platform control frame in degrees. Pitch is right-handed rotation around the starboard pointing :math:`\\hat{y}_{PCF}`.
            A positive pitch will tip the :math:`\\hat{x}_{PCF}` unit vector upwards. Negative values will tip it downwards.
        roll_degrees : float
            The roll of the platform control frame in degrees. Roll is the right-handed rotation around :math:`\\hat{x}_{PCF}`
            after yaw and pitch have been applied. A pilot in a plane looking out of the nose along :math:`\\hat{x}_{PCF}`
            will roll to his right side, (right shoulder falls, left shoulder rises) assuming he is not flying upside down.

        """

        self._PCF_to_GCF.from_yaw_pitch_roll(
            math.radians(yaw_degrees),
            math.radians(pitch_degrees),
            math.radians(roll_degrees),
        )

    # ------------------------------------------------------------------------------
    #           update_location
    # ------------------------------------------------------------------------------
    def set_platform_location(
        self,
        xyzt: tuple[float, float, float, np.datetime64] | None = None,
        latlonheightandt: tuple[float, float, float, np.datetime64] | None = None,
    ):
        """
        Sets the location of the platform to the specified location.

        Parameters
        ----------
        xyzt : Tuple[float, float, float, np.datetime64]
            Set the platform location using geographic geocentric coordinates
        latlonheight : Tuple[float, float, float]
            The the platform location using geodetic corodinates. The tuple is a 4 element array, [0] is latitue, [1] is longitude
            [2] is height in meters, [3] is platform_utc as float representing MJD or numpy.datetime64
        """
        if xyzt is not None:
            self._geo.from_xyz((xyzt[0], xyzt[1], xyzt[2]))
            self._utc = xyzt[3]

        elif latlonheightandt is not None:
            self._geo.from_lat_lon_alt(
                latlonheightandt[0], latlonheightandt[1], latlonheightandt[2]
            )
            self._utc = latlonheightandt[3]
        else:
            logging.warning(
                "set_platform_location, nothing done as neither xyz or latlonheight were set"
            )

        self._geolocation = self._geo.location
        self._local_west = self._geo.local_west
        self._local_south = self._geo.local_south
        self._local_up = self._geo.local_up
        self._GCF_to_ECEF.from_transform_to_destination_coordinates(
            -self._local_west, -self._local_south, self._local_up
        )  #: Transform local geodetic coordinates to geographic geocentric.

    # -----------------------------------------------------------------------------
    #           location
    # -----------------------------------------------------------------------------
    def location(self):
        "returns the geocentric location of the platform. Only valid after a successfull call to set_platform_location"
        return self._geolocation

    # -----------------------------------------------------------------------------
    #           platform_utc
    # -----------------------------------------------------------------------------
    def utc(self):
        return self._utc

    # -----------------------------------------------------------------------------
    #           local_west
    # -----------------------------------------------------------------------------

    def local_west(self):
        """
        Returns the geocentric unit vector of west at the current location of the platform. Only valid after a successful call to set_platform_location
        """
        return self._local_west

    # -----------------------------------------------------------------------------
    #           local_south
    # -----------------------------------------------------------------------------

    def local_south(self):
        """
        Returns the geocentric unit vector of south at the current location of the platform. Only valid after a successful call to set_platform_location
        """
        return self._local_south

    # -----------------------------------------------------------------------------
    #           local_up
    # -----------------------------------------------------------------------------

    def local_up(self):
        """
        Returns the geocentric unit vector of up at the current location of the platform. Only valid after a successful call to set_platform_location
        """
        return self._local_up

    # -----------------------------------------------------------------------------
    #               reset_icf_rotation_matrices
    # -----------------------------------------------------------------------------
    def reset_icf_rotation_matrices(self):
        """
        Resets the IRM and ICF_to_PCF matrices to unity.
        :return:
        """
        self._IRM = RotationMatrix(RotationMatrix.IUnit())
        self._ICF_to_PCF = RotationMatrix(RotationMatrix.IUnit())

    # -----------------------------------------------------------------------------
    #           force_pcf_rotation_matrix
    # -----------------------------------------------------------------------------
    def force_pcf_rotation_matrix(self, GEO: UnitVectors):
        """
        Defines the rotation matrix applied to the platform control frame so the :ref:`icf` unit vectors axes are
        aligned to the ECEF unit-vectors passed in after the full stack of rotation matrices are applied. Note this
        rotation is not applied to the Instrument Rotation Matrix (_IRM) as it is intended to force the primary
        boresight of the instrument control frame point towards a specific location.

        The problem is a linear algebra problem where we have G=RX, where G is the 3x3 Unitvector array passed in,
        X is the initial 3x3 unit vectors in the instrument control frame and R is the stack of rotation matrices. The
        matrix R can be expanded and we get a matrix equation of the form (A )(PCF_to_GCF) ( B )= G where we know
        everything except PCF_to_GCF. Thus we get  PCF_to_GCF = (A-1) G (B-1)


        Parameters
        ----------
        GEO : UnitVectors
            A 3x3 array of column unit vectors. These unit vectors specify the desired orientation of the instrument control frame
            vectors after rotation to the :ref:`ecef`.
        """
        B = RotationMatrix(
            self._PCF_Init_GCF.R  # B is the rotation matrics from instrument to Start of geodetic control frome
            @ self._ICF_to_PCF.R
            @ self._ICF_Init_PCF.R
        )

        A = (
            self._GCF_to_ECEF
        )  # A is the rotation matrix from geodetic control frome to GEO

        self._PCF_to_GCF.from_rotation_matrix(
            A.RInv @ GEO.R @ B.RInv
        )  # get the platform "yaw,pitch roll " rotation matrix

    # -----------------------------------------------------------------------------
    #               convert_icf_to_ecef
    # -----------------------------------------------------------------------------
    def convert_icf_to_ecef(self, v: np.ndarray) -> np.ndarray:
        """
        Converts vectors expressed in the instrument coordinate frame to geographic geocentric ECEF vectors using the
        current rotation matrices.

        Parameters
        ----------
        v : np.ndarray
           An array (3xN) of N vectors expressed in the instrument control frame.

           * v[0,:] is the :math:`\\hat{x}_{ICF}` component of each vector,
           * v[1,:] is the :math:`\\hat{y}_{ICF}` component,
           * v[2,:] is the :math:`\\hat{z}_{ICF}` component.

        Returns
        -------
        np.ndarray
            An array (3xN) of N vectors expressed in the geographic geocentric control frame,

            * v[0,:] is the :math:`\\hat{x}_{ECEF}` component,
            * v[1,:] is the :math:`\\hat{y}_{ECEF}` component,
            * v[2,:] is the :math:`\\hat{z}_{ECEF}` component.
        """
        R = self.get_icf_to_ecef_matrix()
        return R.R @ v

    # ------------------------------------------------------------------------------
    #               convert_icf_to_ecef(self, v: np.ndarray)->np.ndarray:
    # ------------------------------------------------------------------------------

    def convert_icf_to_gcf(self, v: np.ndarray) -> np.ndarray:
        """
        Converts vectors expressed in the instrument coordinate frame to geodetic control frame vectors using the
        current rotation matrices.

        Parameters
        ----------
        v : np.ndarray
            An array (3xN) of N vectors expressed in the instrument control frame.

            * v[0,:] is the :math:`\\hat{x}_{ICF}` component of each vector,
            * v[1,:] is the :math:`\\hat{y}_{ICF}` component,
            * v[2,:] is the :math:`\\hat{z}_{ICF}` component.

        Returns
        -------
        np.ndarray
            An array (3xN) of N vectors expressed in the geodetic control frame,

            * v[0,:] is the :math:`\\hat{x}_{GCF}` (West) component,
            * v[1,:] is the :math:`\\hat{y}_{GCF}` (South) component ,
            * v[2,:] is the :math:`\\hat{z}_{GCF}` (Up) component.
        """

        R = (
            self._PCF_to_GCF.R
            @ self._PCF_Init_GCF.R
            @ self._ICF_to_PCF.R
            @ self._ICF_Init_PCF.R
            @ self._IRM.R
        )
        return R @ v

    # -----------------------------------------------------------------------------
    #           convert_gcf_to_ecef
    # -----------------------------------------------------------------------------

    def convert_gcf_to_ecef(self, v: np.ndarray) -> np.ndarray:
        """
        Converts vectors expressed in the geodetic control frame to geographic geocentric vectors using the
        current rotation matrices.

        Parameters
        ----------
        v : np.ndarray
            An array (3xN) of N vectors expressed in the geodetic control frame.

            * v[0,:] is the :math:`\\hat{x}_{GCF}` component of each vector,
            * v[1,:] is the :math:`\\hat{y}_{GCF}` component,
            * v[2,:] is the :math:`\\hat{z}_{GCF}` component.

        Returns
        -------
        np.ndarray
           An array (3xN) of N vectors expressed in the geodetic control frame,

           * v[0,:] is the :math:`\\hat{x}_{ECEF}` component,
           * v[1,:] is the :math:`\\hat{y}_{ECEF}` component,
           * v[2,:] is the :math:`\\hat{z}_{ECEF}` component.
        """

        return self._GCF_to_ECEF.R @ v

    # -----------------------------------------------------------------------------
    #           get_icf_to_ecef_matrix
    # -----------------------------------------------------------------------------
    def get_icf_to_ecef_matrix(self) -> RotationMatrix:
        """
        Returns the rotation matrix that converts vectors expressed in the instrument control frame to vectors in the geographic geocentric control frame
        If you are inspecting the source code for this method note that the
        rotation matrices are applied in reverse order, ie. right-most or last array in the list is the first rotation applied and left-most or first in the list is the last
        rotation/operation.

        Returns
        -------
        np.ndarray
            An (3x3) rotation matrix. The array is used to transform vectors from the ICF instrument control frame to
            the ECEF geocentric control frame. The matrix should be applied as :math:`\\boldsymbol{V_{GEO}} = \\boldsymbol{R} @ \\boldsymbol{V_{ICF}}`
        """

        full_rotation = (
            self._GCF_to_ECEF.R  # 6) Convert the geodetic vectors to geocentric vectors.
            @ self._PCF_to_GCF.R  # 5) Apply the spatial orientation of the platform in geodetic coordinates
            @ self._PCF_Init_GCF.R  # 4) Rotate the platform control frame so its properly aligned with the geodetic corodinates
            @ self._ICF_to_PCF.R  # 3) Rotate the instrument boresight so its properly positioned on the platform
            @ self._ICF_Init_PCF.R  # 2) Rotate instrument control frame so it is in its initial orientation in the platform control frame
            @ self._IRM.R
        )  # 1) Apply Instrument internal rotations due to mirrors and turntables

        return RotationMatrix(array=full_rotation)
