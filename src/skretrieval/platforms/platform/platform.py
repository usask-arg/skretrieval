from __future__ import annotations

import numpy as np

from ...core import OpticalGeometry
from ..orientation_techniques import OrientationTechniques
from .platform_pointing import PlatformPointing
from .platformlocator import PlatformLocation
from .positionandorientationarray import PositionAndOrientationArray
from .rotationmatrix import RotationMatrix


# -----------------------------------------------------------------------------
#           class Platform
# -----------------------------------------------------------------------------
class Platform:
    """
    The purpose of the Platform class is to capture the details of the physical platform carrying an instrument. Common
    examples of platforms used in atmospheric research are spacecraft, aircraft, balloons and ground-based sites. There
    are a myriad of details surrounding real platforms but the Platform class's only concern is to generate a list
    of instrument *time*, *location* and rotation matrices that properly orient :ref:`icf` for each simulated measurement.
    This list of platform state information is internally generated and then passed onto the next stage of analysis or retrieval.

    The general form of usage is to perform the following steps,

    #. Optional. Specify how an instrument is mounted on the platform. This defines the :ref:`icf`.
    #. Specify the universal times for a set of measurements
    #. Specify the position of the platform for a set of measurements using a variety of positioning techniques, see :ref:`positioning_technique` and :meth:`~.add_measurement_set`
    #. Specify the orientation of the platform for a set of measurements using a variety of orientation techniques, see :ref:`pointing_technique` and :meth:`~.add_measurement_set`.
    #. The orientations define the :ref:`pcf` and the positions define the :ref:`gcf`.
    #. Create the internal :class:`~.PositionAndOrientationArray` for the measurement set, see :meth:`~.make_observation_policy`.
    #. Convert the :class:`~.PositionAndOrientationArray` to arrays of position and instrument look vectors suitable for retrieval code.
    """

    # -----------------------------------------------------------------------------
    #           __init__
    # -----------------------------------------------------------------------------

    def __init__(
        self,
        observation_policy: PositionAndOrientationArray = None,
        platform_locator: PlatformLocation = None,
    ):
        """

        Parameters
        ----------
        platform_locator: PlatformLocation
            Default None. This optional parameter can be used to add a platform location object such as a satellite,
            aircraft or ground site.

        observation_policy: PositionAndOrientationArray
            Default None. This optional parameter can use an existing :ref:`observationpolicy_class` object rather than
            create a new empty instance.
        """

        # super().__init__()
        self._platform_pointing: PlatformPointing = PlatformPointing()
        self._platform_location: PlatformLocation = platform_locator
        self._orientationtechniques: OrientationTechniques = OrientationTechniques()
        self._position_and_orientation_array: PositionAndOrientationArray = (
            observation_policy
            if observation_policy is not None
            else PositionAndOrientationArray()
        )

    # ------------------------------------------------------------------------------
    #           platform_pointing
    # ------------------------------------------------------------------------------
    @property
    def platform_pointing(self) -> PlatformPointing:
        """
        Gets the internal :class:`~.PlatformPointing` object. This object manages all the rotation matrices used to
        transform between various frames.
        """
        return self._platform_pointing

    # ------------------------------------------------------------------------------
    #           platform_locator
    # ------------------------------------------------------------------------------
    @property
    def platform_locator(self) -> PlatformLocation:
        """
        Sets and gets the internal :class:`~.PlatformLocation` object. This field is set to None by default but can
        be set to a valid class if the user wishes to use a specialized class to get the platform position or orientation
        as a function of time.
        """
        return self._platform_location

    @platform_locator.setter
    def platform_locator(self, value):
        self._platform_location = value

    # ------------------------------------------------------------------------------
    #           observation_policy
    # ------------------------------------------------------------------------------
    @property
    def observation_policy(self) -> PositionAndOrientationArray:
        """
        Returns the current internal :ref:`observationpolicy_class` object.

        Returns
        -------
        PositionAndOrientationArray
            Returns the current internal :ref:`observationpolicy_class` object.
        """
        if (self._position_and_orientation_array is None) or (
            self._orientationtechniques._isdirty
        ):
            self.make_position_and_orientation_array()
        return self._position_and_orientation_array

    # ------------------------------------------------------------------------------
    #           platform_ecef_positions
    # ------------------------------------------------------------------------------
    @property
    def platform_ecef_positions(self) -> np.ndarray:
        """
        Returns the position of the platform for each RT calculation

        Returns
        -------
        np.ndarray(3,N)
            The X,Y,Z location of the platform for each RT calculation. The coordinates are in meters form the center of the Earth,
        """
        return self._position_and_orientation_array.ecef_positions()

    # ------------------------------------------------------------------------------
    #           icf_to_ecef_rotation_matrices
    # ------------------------------------------------------------------------------
    @property
    def icf_to_ecef_rotation_matrices(self) -> list[RotationMatrix]:
        """
        Returns the platform rotation matrices, one  matrix for each RT calculation

        Returns
        -------
        np.ndarray(3,N)
            The X,Y,Z icf_to_ecef_rotation_matrices of the platform.
        """
        return self._position_and_orientation_array.icf_to_ecef_rotation_matrix()

    # ------------------------------------------------------------------------------
    #           num_exposures
    # ------------------------------------------------------------------------------
    @property
    def numsamples(self) -> int:
        """
        Returns the number of samples/exposures in the current observation set

        Returns
        -------
        int
            The number of samples in the current observation set
        """
        return self.observation_policy.numsamples_in_observationset()

    # ------------------------------------------------------------------------------
    #           add_measurement_set
    # ------------------------------------------------------------------------------

    def add_measurement_set(
        self, utc, platform_position, platform_orientation, icf_orientation=None
    ):
        """
        Adds a set of *N* measurements definitions to the internal list of measurement sets. An overview of position and
        orientation techniques is given in :ref:`platforms_model`.

        Parameters
        ----------
        utc:Array, sequence or scalar of string, float, datetime or datetime64
            The set of universal times for the set of measurements. The array should be an array or sequence of any type that can be
            coerced by package sktime into a numpy array (N,) of datetime64. This includes strings, datetime, datetime64. Floating
            point values are also converted and are assumed to represent modified julian date. The number of universal time values
            defines the number of measurements in this set unless it is a scalar. In this case the time value is duplicated into an array with
            the same number of measurements as parameter *observer_positions*. All elements of the array should be of the same
            type and should represent universal time. Be wary of using datetime objects with explicit time-zones.
        observer_positions: Tuple[str, sequence]
            A two element tuple. The first element specifies the :ref:`positioning_technique` to be used to position the
            platform. The second element of the tuple is an array of arrays of float. The array of arrays can be any Python, list of lists, tuples or arrays
            that can be coerced into a two dimensional array of shape **(N, numparams)** where *N* is the number of
            measurements and must match the size of the *utc* parameter parameters and *numparams* is the number of parameters
            required by the chosen :ref:`positioning_technique`. The second element of the tuple can be dropped if the *positioning technique*
            is *from_platform*.
        platform_orientation:
            A three element tuple. The first element of the tuple is a string that specifies the :ref:`pointing_technique`. The second element
            of the tuple specifies :ref:`rollcontrol` and the third element is an array of arrays of float. The array of arrays can be any Python, list of lists, tuples or arrays
            that can be coerced into a two dimensional array of shape **(N, numparams)** where *N* is the number of
            measurements and must match the size of the *utc* parameter and *numparams* is the number of parameters
            required by the chosen :ref:`pointing_technique`
        instrument_internal_rotation:
            Optional. An array that specifies the internal rotation of the instrument within the :ref:`icf`. This is intended to
            provide support for tilting mirrors and turntables attached to the instrument that redirect the instrument boresight independently
            of the platform. The array specifies the azimuth, elevation and roll of the instrument boresight in the :ref:`icf`.
            The array is a sequence or array that can be sensibly coerced into an array of size (N,2) or (N,3) where N is the number of measurements.
            N can be 1 in which case the array size is broadcast to match the number of measurements inferred from the other parameters. Elements [:,0] is the azimuth in degrees  of the
            instrument boresight in the instrument control frame, left handed rotation around :math:`\\hat{z}_{ICF}`. Elements [:,1] are the elevation in
            degrees of the instrument boresight in the instrument control frame, left handed rotation around the rotated :math:`\\hat{y}_{ICF}` axis.
            Elements[:,2], which are are optional, are the roll of the instrument boresight in degrees, right handed rotation around the rotated  :math:`\\hat{x}_{ICF}` axis.
            The roll defaults to 0.0 if not supplied.

        """
        self._orientationtechniques.add_measurement_set(
            utc,
            platform_position,
            platform_orientation,
            icf_orientation=icf_orientation,
        )

    # ------------------------------------------------------------------------------
    #           make_observation_policy
    # ------------------------------------------------------------------------------

    def make_position_and_orientation_array(self) -> PositionAndOrientationArray:
        """
        Takes all the measurements from previous calls to :meth:`~.Platform.add_measurement_set` and converts them into
        a list of measurements stored inside the internal instance of :class:`~.PositionAndOrientationArray` object. This list of
        measurements can be converted to other formats, such as **OpticalGeometry** required by other parts of the
        ``skretrieval`` package. This method clears all the measurements that have been previously added.
        """

        self._position_and_orientation_array.clear()
        self._orientationtechniques.make_observation_set(self)
        return self._position_and_orientation_array

    # ------------------------------------------------------------------------------
    #           make_optical_geometry
    # ------------------------------------------------------------------------------
    def make_optical_geometry(self) -> list[OpticalGeometry]:
        """
        Takes all the measurements from previous calls to :meth:`~.Platform.add_measurement_set` and returns them as
        a list of OpticalGeometry values which can be used by various retrievals.  The internal :class:`~.PositionAndOrientationArray` is
        also created at this time using method :meth:`~Platform.make_observation_policy`.
        """

        position_and_orientation_array = self.make_position_and_orientation_array()
        return position_and_orientation_array.to_optical_geometry()

    # -----------------------------------------------------------------------------
    #               make_ecef_position_and_lookvector
    # -----------------------------------------------------------------------------
    def make_ecef_position_and_lookvector(
        self, icf_lookvectors=None, one_to_one: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position_and_orientation_array = self.make_position_and_orientation_array()
        return position_and_orientation_array.to_ecef_position_and_look(
            icf_lookvectors=icf_lookvectors, one_to_one=one_to_one
        )

    # -----------------------------------------------------------------------------
    #               def add_current_state(self  )->int:
    # -----------------------------------------------------------------------------
    def add_current_platform_state(self) -> int:
        """
        Fetches the current platform position and orientation and uses this as a new sample in the observation
        policy.

        Returns
        -------
        int
            The number of samples in the current observation policy

        """
        return self._position_and_orientation_array.add_current_state(
            self._platform_pointing
        )

    # -----------------------------------------------------------------------------
    #               clear_states(self):
    # -----------------------------------------------------------------------------
    def clear_states(self):
        """
        Clears all of the internally cached measurement states. This should be called when
        re-using a platform object to create a new measurement set.
        """
        self._orientationtechniques.clear()
        self._position_and_orientation_array.clear()

    # ------------------------------------------------------------------------------
    #           icf_to_ecef
    # ------------------------------------------------------------------------------
    def icf_to_ecef(self, los_icf: np.ndarray) -> np.ndarray:
        """
        Returns the lines of sight in geographic geocentric ECEF coordinates of the lines of sight specified
        in the instrument control frame.

        Parameters
        ----------
        los_icf : np.ndarray( 3,Nlos)
            A 2-D array of N unit vectors expressed in the instrument control frame

        Returns
        -------
        np.ndarray (3,Nlos, Ntime)
            A numpy array is returned with the 3 element geographic, geocentric, line of sight unit vector for each
            input lines of sight and each time in the current observation set.
        """
        Nlos = los_icf.shape[1]
        obs = (
            self.observation_policy
        )  # Get the observation set used for this back_end_radiance calculation                                                                                    # Get the number of instantaneous lines of sight from the second dimension of the array. LOS are specified in the instrument control frame
        Nt = (
            obs.numsamples_in_observationset()
        )  # Get the number of exposures/samples in the observation set
        rot = (
            obs.icf_to_ecef_rotation_matrix()
        )  # Copy the rotation matrices for each sample in the observation set.
        all_los = np.full(
            (3, Nlos, Nt), np.nan
        )  # Create an array to hold the lines of sight for all instantaneous directions and all times
        for i in range(Nt):  # for each time
            R = rot[
                i
            ]  # Get the platform rotation matrix that converts ICF vectors to ECEF vectors
            instantaneouslos = (
                R.R @ los_icf
            )  # Convert all the instrument control frame instantaneous lines of sight for this sample to geocentric geographic lines of sight
            all_los[
                :, :, i
            ] = instantaneouslos  # and save the geo unit vectors in the array.
        return all_los
