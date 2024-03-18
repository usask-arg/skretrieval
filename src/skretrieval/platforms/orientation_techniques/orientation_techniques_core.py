from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from ..platform import Platform

import numpy as np

from skretrieval.geodetic import geodetic
from skretrieval.time import ut_to_datetime64

# from ..platform import Platform
from ..platform.rotationmatrix import UnitVectors
from .standard_orientation_techniques import (
    _technique_set_boresight_look_at_location_llh,
    _technique_set_boresight_look_at_location_orbitangle,
    _technique_set_boresight_look_at_location_xyz,
    _technique_set_boresight_pointing_from_unitvectors,
    _technique_set_icf_look_from_xyz,
    _technique_set_icf_orientation_from_azi_elev,
    _technique_set_icf_orientation_no_operation,
    _technique_set_limb_look_vectors_from_unit_xyz,
    _technique_set_look_vectors_from_tangent_altitude,
    _technique_set_observer_to_look_in_azi_elev,
    _technique_set_platform_pointing_from_platform,
    _technique_set_platform_position_from_llh,
    _technique_set_platform_position_from_observer_looking_at_llh,
    _technique_set_platform_position_from_platform,
    _technique_set_platform_position_from_xyz,
)


# ------------------------------------------------------------------------------
#           class PointingAlgorithms:
# ------------------------------------------------------------------------------
class PointingAlgorithms:
    def __init__(self, platform: Platform):
        self._geo = geodetic()
        self.platform: Platform = platform

    # ------------------------------------------------------------------------------
    #           geo
    # ------------------------------------------------------------------------------
    @property
    def geo(self):
        return self._geo

    # -----------------------------------------------------------------------------
    #           set_limb_boresight_to_look_at_tangent_altitude
    # -----------------------------------------------------------------------------

    def set_limb_boresight_to_look_at_tangent_altitude(
        self,
        observer: np.ndarray,
        target_tangent_altitude: float,
        target_geographic_bearing: float,
        roll_control: str,
        roll_angle_degrees: float,
    ) -> bool:
        """
        Sets the rotation matrix of the system so the boresight of the instrument is looking at a tangent point
        at the given tangent altitude and bearing. The tangent altitude must be below the observer and above 5 km below
        the ground.

        Parameters
        ----------
        observer: np.ndarray[3,]
            The x,y,z location of the observer in meters in the :ref:`ecef` reference frame.
        target_tangent_altitude : float
            The tangent altitude of the target location in meteres above sea-level. The tangent altitude  must be below the
            observer and above 5 km below the ground otherwise it is not used.
        target_geographic_bearing: float
            The geographic bearing in degrres of the tangent altitude measured the current platform location. N=0, E=90, S=180, W=270.
        roll_control: str
            The string describing roll control, see: :ref:`rollcontrol`
        roll_angle_degrees: float
            The roll angle in diegrees in the clockwise direction from the zero point set by `roll_control`

        Returns
        -------
        bool: True if successful.
        """

        self._geo.from_xyz(observer)
        north = -self._geo.local_south
        east = -self._geo.local_west
        maxaltitude = self._geo.altitude
        ok = (target_tangent_altitude < maxaltitude) and (
            target_tangent_altitude > -5000.0
        )  # make sure the target tangent altitude is below the observer and no more than 5 km below the surface.
        if ok:
            bearing = math.radians(target_geographic_bearing)
            horiz = math.cos(bearing) * north + math.sin(bearing) * east
            look = self._geo.from_tangent_altitude(
                target_tangent_altitude, observer, horiz
            )
            target = self._geo.location
            G = self.apply_roll_control(
                look,
                target,
                observer,
                roll_control,
                roll_angle_degrees,
                horizontalunitvector=horiz,
            )
            self.platform.platform_pointing.force_pcf_rotation_matrix(G)
        else:
            logging.warning(
                "PointingAlgorithms.set_limb_boresight_to_look_at_tangent_altitude, the target altitude {} is not between the observer altitude {} and 5 km below the ground".format(  # noqa: G001
                    target_tangent_altitude, maxaltitude
                )
            )
        return ok

    # -----------------------------------------------------------------------------
    #           set_limb_boresight_from_lookvector
    # -----------------------------------------------------------------------------

    def set_limb_boresight_from_lookvector(
        self,
        observer: np.ndarray,
        lookvector: np.ndarray,
        roll_control: str,
        roll_angle_degrees: float,
    ) -> bool:
        """
        Sets the platform rotation matrices so the boresight of the instrument is looking in the direction given by lookvector. The
        tangent point, implied by the look vector, must be in front of the observer and above 5 km below the ground.

        Parameters
        ----------
        observer: np.ndarray[3,]
            The x,y,z location of the observer in meters in the :ref:`ecef` reference frame.
        lookvector: np.ndarray[3]
            The x,y,z look unit vector expressed in the :ref:`ecef` reference frame, looking from the platform towards the limb
        roll_control: str
            The string describing roll control, see: :ref:`rollcontrol`
        roll_angle_degrees: float
            The roll angle in diegrees in the clockwise direction from the zero point set by `roll_control`

        Returns
        -------
        bool: True if successful.
        """
        self._geo.from_tangent_point(observer, lookvector)
        targetlocation = self._geo.location
        altitude = self._geo.altitude
        delta = targetlocation - observer
        ok = (np.dot(delta, lookvector) > 0.0) and (altitude > -5000.0)
        if ok:
            G = self.apply_roll_control(
                lookvector, targetlocation, observer, roll_control, roll_angle_degrees
            )
            self.platform.platform_pointing.force_pcf_rotation_matrix(G)
        else:
            logging.warning(
                "PointingAlgorithms.set_limb_boresight_from_lookvector, the tangent point is not in front of the observer or is more than 5000m below the ground [{}]. It has been discarded".format(  # noqa: G001
                    altitude
                )
            )
        return ok

    # ------------------------------------------------------------------------------
    #           get_unitvectors_at_target_location
    # ------------------------------------------------------------------------------

    def set_boresight_to_look_at_geocentric_location(
        self,
        observer_location: np.ndarray,
        targetlocation: np.ndarray,
        roll_control: str,
        roll_angle_degrees: float,
    ) -> bool:
        """
        Sets the rotation matrix of the platform so the boresight of the instrument will look at the given geocentric
        location. The target location must be more than 10 cm from the observer's location.

        Parameters
        ----------
        observer_location: np.ndarray[3,]
            The x,y,z location of the observer in meters in the :ref:`ecef` reference frame.
        targetlocation: np.ndarray[3,]
            The x,y,z location of the target point in meters in the :ref:`ecef` reference frame
        roll_control: str
            The string describing roll control, see: :ref:`rollcontrol`
        roll_angle_degrees: float
            The roll angle in diegrees in the clockwise direction from the zero point set by `roll_control`

        Returns
        -------
        bool: True if successful.
        """
        lookvector = (
            targetlocation - observer_location
        )  # Get the look vector from current location to the target location
        dist = np.linalg.norm(
            lookvector
        )  # Get the distance of the target from the observer
        ok = dist > 0.01  # make sure its more than 10 cm
        if ok:  # and if we are good
            lookvector /= dist  # the normalize the lookvector
            G = self.apply_roll_control(
                lookvector,
                targetlocation,
                observer_location,
                roll_control,
                roll_angle_degrees,
            )  # get the the desired unit vectors
            self.platform.platform_pointing.force_pcf_rotation_matrix(
                G
            )  # and calculate the new rotation matrix from the platform control frame.
        else:
            logging.warning(
                "PointingAlgorithms.set_boresight_to_look_at_geocentric_location, the target location is located within 10 cm of the observer. It is too close. It has been discarded"
            )
        return ok

    # ------------------------------------------------------------------------------
    #           get_unitvectors_at_target_location
    # ------------------------------------------------------------------------------
    def set_boresight_pointing_from_unitvectors(self, G: UnitVectors) -> bool:
        """
        Sets the rotation matrix of the platform to the given UnitVector array. The array defines the required x, y and
        z unit vectors in the :ref:`ecef`

        Parameters
        ----------
        G: UnitVectors
            The class that represents a (3x3) matrix. Each column of the matrix is the desired x,y,z unit vector
            in the :ref:`ecef` reference frame.

        Returns
        -------
        bool: True if successful.
        """
        self.platform.platform_pointing.force_pcf_rotation_matrix(G)
        return True

    # ------------------------------------------------------------------------------
    #           set_observer_to_look_in_azi_elev
    # ------------------------------------------------------------------------------
    def set_observer_to_look_in_azi_elev(
        self,
        observer: np.ndarray,
        elevation: float,
        azimuth: np.ndarray,
        roll_control: str,
        roll_angle_degrees: float,
    ) -> bool:
        """
        Sets the rotation matrix of the platform so the boresight of the instrument will look at the elevation
        and azimuth defined in the horizontal plane at the observers location.

        Parameters
        ----------
        observer: np.ndarray[3,]
            The x,y,z location of the observer in meters in the :ref:`ecef` reference frame.
        elevation: float
            The elevation in degrees of the look vector angle from the local horizontal plane at the observer.
            Horizontal is 0 degrees. vertical up is 90, vertical down is -90. It is a left handed rotation around the
            local west unit vector
        azimuth: float
            The azimuth in degrees of the look vector in the local horizontal plane of the observer. Azimuth is measured
            clockwise, like a compass, with 0 in the due North direction. 0=N, 90=E, 180=S, 270=W. It is a left handed rotation
            around the local up vector
        roll_control: str
            The string describing roll control, see: :ref:`rollcontrol`
        roll_angle_degrees: float
            The roll angle in degrees in the clockwise direction from the zero point set by `roll_control`

        Returns
        -------
        bool: True if successful.
        """
        self._geo.from_xyz(observer)
        up = self._geo.local_up
        north = -self._geo.local_south
        east = -self._geo.local_west
        bearing = math.radians(azimuth)
        theta = math.radians(elevation)
        horiz = math.cos(bearing) * north + math.sin(bearing) * east
        look = math.cos(theta) * horiz + math.sin(theta) * up
        G = self.apply_roll_control(
            look,
            observer,
            observer,
            roll_control,
            roll_angle_degrees,
            horizontalunitvector=horiz,
        )
        self.platform.platform_pointing.force_pcf_rotation_matrix(G)
        return True

    # ------------------------------------------------------------------------------
    #           config_roll_control
    # ------------------------------------------------------------------------------

    def apply_roll_control(
        self,
        lookvector: np.ndarray,
        target_location: np.ndarray,
        observer_location: np.ndarray,
        roll_cntl: str,
        roll_angle_degrees: float,
        horizontalunitvector: np.ndarray = None,
    ) -> UnitVectors:
        """
        Applys roll control to the look vector and returns the (3x3) array of Unit Vectors in the final :ref:`ecef` reference
        frame.

        Parameters
        ----------
        lookvector : np.ndarray(3,)
            Three element look unit vector away from the observer's position specified in the :ref:`ecef` coordinate system
        target_location : np.ndarray(3,)
            Three element position vector of the target location  specified in meters in the :ref:`ecef` coordinate system
        observer_location: np.ndarray(3,)
             Three element position vector of the observer's location specified in meters in the :ref:`ecef` coordinate system
        roll_cntl : str
            The class of required roll control. 'limb', 'nadir' and 'standard' are currently supported.
        roll_angle_degrees: float
            The required roll angle required using the current control frame.
        horizontalunitvector: np.ndarray(3,)
            Optional. This is the horizontal unit vector in the horizontal plane of the look vector. It is only used in standard
            roll control and allows the roll of the instrument unit vectors to stay well defined even when the looking straight up or down.

        Returns
        -------
        UnitVectors:
            Returns the 3x3 array of unit vectors required in the final :ref:`ecef` reference frame. The platform rotation
            matrix should be modified so that unit vectors in the :ref:`icf` will map to these unit vectors after rotation.
        """

        if roll_cntl == "limb":  # roll control is limb
            self._geo.from_xyz(
                target_location
            )  # so zero roll occurs when the Instrument Z axis, is as parallel as possible to the local up unit vector
            zunit = (
                self._geo.local_up
            )  # We want to choose a z unit vector. Get local up at the tangent point
            ythreshold = math.sin(math.radians(0.5))
            xunit = lookvector  # Get the look vector unit vector as the X direction
            yunit = np.cross(
                zunit, xunit
            )  # Get the y unit vector which points towards the left as seen from the oobserver.
            ymag = np.linalg.norm(
                yunit
            )  # Get the magnitude of the cross product. This becomes
            if (
                ymag < ythreshold
            ):  # if we are in an unstable region where the first choice of Z is too parallel to the look vector then use the second option  (ie  local Z axis and lookvector are with 0.57 degrees of each other)
                zunit2 = -self._geo.local_south  # so arbitrarily try local North
                yunit = np.cross(
                    zunit2, xunit
                )  # Get the y unit vector which points towards the left as seen from the oobserver.
                ymag = np.linalg.norm(
                    yunit
                )  # Get the magnitude of the cross product. This becomes
                yunit /= ymag  # get the magnitude of the Y unit vector
                zunit = np.cross(
                    xunit, yunit
                )  # now get the z unit vector perpendicular to the look vector and the y unit vector

        elif roll_cntl == "nadir":
            self._geo.from_xyz(
                target_location
            )  # so zero roll occurs when the Instrument Z axis, is as parallel as possible to the local north
            zunit = (
                -self._geo.local_south
            )  # We want to choose a z unit vector. Get local north at the target location
            ythreshold = math.sin(math.radians(0.5))
            xunit = lookvector  # Get the look vector unit vector as the X direction
            yunit = np.cross(
                zunit, xunit
            )  # Get the y unit vector which points towards the left as seen from the oobserver.
            ymag = np.linalg.norm(
                yunit
            )  # Get the magnitude of the cross product. This becomes
            if (
                ymag < ythreshold
            ):  # if we are in an unstable region where the first choice of Z is too parallel to the look vector then use the second option  (ie  local Z axis and lookvector are with 0.57 degrees of each other)
                zunit2 = -self._geo.local_up  # so arbitrarily use local north.
                yunit = np.cross(
                    zunit2, xunit
                )  # Get the y unit vector which points towards the left as seen from the oobserver.
                ymag = np.linalg.norm(
                    yunit
                )  # Get the magnitude of the cross product. This becomes
                yunit /= ymag  # get the magnitude of the Y unit vector
                zunit = np.cross(
                    xunit, yunit
                )  # now get the z unit vector perpendicular to the look vector and the y unit vector

        elif roll_cntl == "standard":
            self._geo.from_xyz(
                observer_location
            )  # so zero roll occurs when the Instrument Z axis, is as parallel as possible to the local up unit vector
            xunit = lookvector  # get the instrument x unit vector
            if (
                horizontalunitvector is not None
            ):  # If the caller has passed in the horizontal unit vector of the line of sight
                zunit = (
                    self._geo.local_up
                )  # then we can use it with good stability, so get the z unit vector
                yunit = np.cross(
                    zunit, horizontalunitvector
                )  # the zero point for the instrument y unit vector is in the horizontal plane perpendicular to the look vector. This always works even when the look vector is straight up.
                zunit = np.cross(
                    xunit, yunit
                )  # The zero point for the instrument z unit vector is the cross product of the x and y instrument unit vectors
            else:  # if the user cannot provide a horizontal bearing
                zunit = (
                    self._geo.local_up
                )  # then we try our best. We want to choose a z unit vector. Get local up at the tangent point
                z = np.dot(
                    lookvector, zunit
                )  # Get the vertical component of the look vector
                horiz = (
                    lookvector - z * zunit
                )  # And subtract it to get the horizontal component
                if (
                    math.fabs(z) > 0.9999619
                ):  # If we have almost no horizontal component( within 0.5 degrees of straight up/down)
                    horiz = (
                        -self._geo.local_south
                    )  # then arbitrarily use the north direction
                horiz = horiz / np.linalg.norm(
                    horiz
                )  # get the horizontal unit vector parallel to the horizontal component of the look vector
                yunit = np.cross(zunit, horiz)  # now get the instrument y unit vector
                zunit = np.cross(xunit, yunit)  # and the instrument z unit vector.
        else:
            msg = f"Unsupported value {roll_cntl} of roll control"
            raise ValueError(msg)
        if (
            roll_angle_degrees != 0.0
        ):  # Positive roll angle is clockwise as seen from theobserver looking along the line of sight
            theta = math.radians(
                -roll_angle_degrees
            )  # Apply the positive roll if the user has set it to non zero value
            costheta = math.cos(theta)
            sintheta = math.sin(theta)
            zprime = zunit * costheta + yunit * sintheta
            yprime = zunit * (-sintheta) + yunit * costheta
            zunit = zprime
            yunit = yprime
        return UnitVectors(vectors=(xunit, yunit, zunit))


# -----------------------------------------------------------------------------
#           OrientationTechniques
# -----------------------------------------------------------------------------
class OrientationTechniques:
    """
    A helper class that allows users to specify position and orientations using a set of position techniques
    and orientation techniques.
    """

    def __init__(self):
        """
        Creates a new empty instance.
        """
        self._position_convertors: dict[
            str,
            tuple[Callable[[Platform, np.datetime64, np.ndarray], bool], list[int]],
        ] = (
            {}
        )  # A dictionary of positioning tecniques. Each entry of the form [name, function]
        self._platform_orientation_convertors: dict[
            str, tuple[Callable[[PointingAlgorithms, np.ndarray, str], bool], list[int]]
        ] = (
            {}
        )  # A dictionary of orientation techniques. Each entry of the form [name, function]
        self._icf_look_vector_convertors: dict[
            str,
            tuple[Callable[[Platform, np.datetime64, np.ndarray], bool], list[int]],
        ] = (
            {}
        )  # A dictionary of ICF look vector techniques. Each entry of the form [name, function]
        self._position_definitions: list[
            tuple[str, np.ndarray | int]
        ] = (
            []
        )  # A list of various representations of position specifications. Each entry of the form [technique, data_array]
        self._platform_orientation_definitions: list[
            tuple[str, np.ndarray | int, str]
        ] = (
            []
        )  # A list of various representations of platform orientation techniques and data . Each entry of the form [technique, data_array, roll_control]
        self._icf_lookvector_definitions: list[
            tuple[str, np.ndarray | int, str]
        ] = []  # A list of various representations of ICF look vector specifications.
        self._utc: list[
            np.ndarray
        ] = []  # A list of arrays of measurement times of numpy.datetime64['us']
        self._isdirty = False

        self.add_position_convertor(
            "xyz", _technique_set_platform_position_from_xyz, [3]
        )
        self.add_position_convertor(
            "llh", _technique_set_platform_position_from_llh, [3]
        )
        self.add_position_convertor(
            "from_platform", _technique_set_platform_position_from_platform, [0]
        )
        self.add_position_convertor(
            "looking_at_llh",
            _technique_set_platform_position_from_observer_looking_at_llh,
            [5],
        )

        self.add_platform_orientation_convertor(
            "tangent_xyz_look", _technique_set_limb_look_vectors_from_unit_xyz, [3, 4]
        )  # Set look vectors using ITRF/ecef xyz location vectors
        self.add_platform_orientation_convertor(
            "tangent_altitude",
            _technique_set_look_vectors_from_tangent_altitude,
            [2, 3],
        )  # specifies
        self.add_platform_orientation_convertor(
            "tangent_from_orbitplane",
            _technique_set_boresight_look_at_location_orbitangle,
            [2, 3],
        )
        self.add_platform_orientation_convertor(
            "location_xyz", _technique_set_boresight_look_at_location_xyz, [3, 4]
        )
        self.add_platform_orientation_convertor(
            "location_llh", _technique_set_boresight_look_at_location_llh, [3, 4]
        )
        self.add_platform_orientation_convertor(
            "unit_vectors", _technique_set_boresight_pointing_from_unitvectors, [6]
        )
        self.add_platform_orientation_convertor(
            "azi_elev", _technique_set_observer_to_look_in_azi_elev, [2, 3]
        )
        self.add_platform_orientation_convertor(
            "yaw_pitch_roll", _technique_set_observer_to_look_in_azi_elev, [2, 3]
        )
        self.add_platform_orientation_convertor(
            "from_platform", _technique_set_platform_pointing_from_platform, [0]
        )

        self.add_icf_look_vector_convertor(
            "azi_elev", _technique_set_icf_orientation_from_azi_elev, [2]
        )
        self.add_icf_look_vector_convertor(
            "xyz", _technique_set_icf_look_from_xyz, [1, 3]
        )
        self.add_icf_look_vector_convertor(
            "nop", _technique_set_icf_orientation_no_operation, [0]
        )

    # ------------------------------------------------------------------------------
    #           add_platform_orientation_convertor
    # ------------------------------------------------------------------------------
    def add_platform_orientation_convertor(
        self,
        orientationtype: str,
        convertor_function: Callable[[PointingAlgorithms, np.ndarray, str], bool],
        num_user_params: list[int],
    ):
        key = orientationtype.lower()
        self._platform_orientation_convertors[key] = (
            convertor_function,
            num_user_params,
        )

    # ------------------------------------------------------------------------------
    #           add_position_convertor
    # ------------------------------------------------------------------------------
    def add_position_convertor(
        self,
        positiontype: str,
        convertor_function: Callable[[Platform, np.datetime64, np.ndarray], bool],
        num_user_params: list[int],
    ):
        key = positiontype.lower()
        self._position_convertors[key] = (convertor_function, num_user_params)

    # ------------------------------------------------------------------------------
    #           add_position_convertor
    # ------------------------------------------------------------------------------
    def add_icf_look_vector_convertor(
        self,
        lookvectorype: str,
        convertor_function: Callable[[Platform, np.datetime64, np.ndarray], bool],
        num_user_params: list[int],
    ):
        key = lookvectorype.lower()
        self._icf_look_vector_convertors[key] = (convertor_function, num_user_params)

    # -----------------------------------------------------------------------------
    #           numsamples
    # -----------------------------------------------------------------------------
    def num_measurements(self) -> int:
        """
        Returns the number of samples in this observation policy

        Returns
        -------
        int
            The number of samples in the observation policy.
        """
        n = 0
        for entry in self._utc:
            n += entry.size
        return n

    # -----------------------------------------------------------------------------
    #           clear
    # -----------------------------------------------------------------------------
    def clear(self):
        """
        Clears the current list of platform states inside the observation policy. This method should be called
        before starting a new observation policy
        """
        self._position_definitions.clear()
        self._platform_orientation_definitions.clear()
        self._icf_lookvector_definitions.clear()
        self._utc.clear()

    # ------------------------------------------------------------------------------
    #           check_position_converter_key
    # ------------------------------------------------------------------------------
    def _check_position_converter_key(self, key: str) -> bool:
        ok = self._position_convertors.get(key) is not None
        if not ok:
            msg = "The position converter keyword <{:s}> is not in the position converter dictionary. You will need to manually add this function with method OrientationTechniques.add_position_convertor".format(
                key
            )
            raise ValueError(msg)
        return ok

    # ------------------------------------------------------------------------------
    #           _check_platform_orientation_converter_key
    # ------------------------------------------------------------------------------
    def _check_platform_orientation_converter_key(self, key: str) -> bool:
        ok = self._platform_orientation_convertors.get(key) is not None
        if not ok:
            msg = "The platform orientation converter keyword <{:s}> is not in the platform orientation converter dictionary. You will need to manually add this function with method OrientationTechniques.add_platform_orientation_convertor".format(
                key
            )
            raise ValueError(msg)
        return ok

    # ------------------------------------------------------------------------------
    #           _check_icf_look_vector_converter_key
    # ------------------------------------------------------------------------------
    def _check_icf_look_vector_converter_key(self, key: str) -> bool:
        ok = self._icf_look_vector_convertors.get(key) is not None
        if not ok:
            msg = "The ICF look vector converter keyword <{}> is not in the platform orientation converter dictionary. You will need to manually add this function with method OrientationTechniques.add_icf_look_vector_convertor".format(
                key
            )
            raise ValueError(msg)
        return ok

    # ------------------------------------------------------------------------------
    #           _coerce_to_vector
    # ------------------------------------------------------------------------------
    def _coerce_to_vector(
        self, data: np.ndarray | Sequence[float], acceptable_n: list[int]
    ) -> np.ndarray:
        if isinstance(data, Sequence):
            data = np.array(data, dtype=np.float64)
        if type(data) is np.ndarray:
            if data.dtype != np.float64:
                data = np.array(data, dtype=np.float64)
            s = data.shape
            N = s[len(s) - 1]
            ok = N in acceptable_n
            if not ok:
                msg = "Incoming data must be an array or sequence with a trailing dimension being one of {}. Your array shape was {}".format(
                    acceptable_n, s
                )
                raise ValueError(msg)
            answer = np.copy(np.reshape(data, [data.size // N, N]))
        else:
            answer = None
            if data is not None:
                msg = "Incoming data must be a numpy array, sequence or None"
                raise ValueError(msg)
        return answer

    # ------------------------------------------------------------------------------
    #           _decode_position_entry
    # ------------------------------------------------------------------------------
    def _decode_position_entry(self, observer_positions):
        if isinstance(observer_positions, str):  # if we have just a string
            key = observer_positions  # then it is the technique key. It should be 'from_platform'
            data = None  # and there is no position data.
        else:  # otherwise it must be a sequence
            key = observer_positions[0]  # where the technique key is the first element
            data = (
                observer_positions[1] if len(observer_positions) > 1 else None
            )  # and the data sequence is the second element, or None if there is no second element
        key = key.lower()  # Convert the technique key to lower case
        if key == "from_platform":
            data = None
        self._check_position_converter_key(
            key
        )  # check that the technique key is supported
        num_params = self._position_convertors[key][1]
        positiondata = self._coerce_to_vector(
            data, num_params
        )  # coerce the vector data into array[N,3]
        numpositions = positiondata.shape[0] if (positiondata is not None) else 0
        return numpositions, positiondata, key

    # ------------------------------------------------------------------------------
    #           _decode_platform_orientation_entry
    # ------------------------------------------------------------------------------
    def _decode_platform_orientation_entry(self, platform_orientation_data):
        roll_control = "undefined"  # default value of roll if no value passed in
        data = None  # default value of data is no data are passed in
        if isinstance(platform_orientation_data, str):
            key = platform_orientation_data
        else:  # otherwise a dictionary has been passed
            key = platform_orientation_data[
                0
            ]  # get the technique key from the first element
            if len(platform_orientation_data) > 1:
                roll_control = platform_orientation_data[1]
                data = platform_orientation_data[2]
            else:
                data = None  # get the technique parameter data from the second element. None if there is no second element.
                roll_control = "undefined"
        key = key.lower()  # convert the technique key to lower case
        if key == "from_platform":
            data = None
            roll_control = "undefined"
        self._check_platform_orientation_converter_key(
            key
        )  # check that the look-vector technique is supported
        num_params = self._platform_orientation_convertors[key][
            1
        ]  # get the number of parameters expected from this look vector technique
        lookdata = self._coerce_to_vector(
            data, num_params
        )  # coerce the technique parameter data into a 2d-array of [num_params,N]
        numlooks = lookdata.shape[0] if (lookdata is not None) else 0
        return numlooks, lookdata, key, roll_control

    # ------------------------------------------------------------------------------
    #           _decode_instrument_internal_rotation
    # ------------------------------------------------------------------------------
    def _decode_icf_look_vector_entry(self, icf_look_vectors):
        data = None  # default value of data is no data are passed in
        if isinstance(icf_look_vectors, str):
            key = icf_look_vectors
        else:  # otherwise a dictionary has been passed
            if icf_look_vectors is not None:
                key = icf_look_vectors[
                    0
                ]  # get the technique key from the first element
                if len(icf_look_vectors) > 1:
                    data = icf_look_vectors[1]
            else:
                data = None  # get the technique parameter data from the second element. None if there is no second element.
                key = "nop"

            self._check_icf_look_vector_converter_key(
                key
            )  # check that the look-vector technique is supported
            num_params = self._icf_look_vector_convertors[key][
                1
            ]  # get the number of parameters expected from this look vector technique
            lookdata = self._coerce_to_vector(
                data, num_params
            )  # coerce the technique parameter data into a 2d-array of [num_params,N]
            numlooks = lookdata.shape[0] if (lookdata is not None) else 1
        return numlooks, lookdata, key

    # ------------------------------------------------------------------------------
    #           _decode_instrument_internal_rotation
    # ------------------------------------------------------------------------------
    def _decode_instrument_internal_rotation(self, instrument_internal_rotation):
        turntabledata = self._coerce_to_vector(
            instrument_internal_rotation, [2, 3]
        )  # coerce the technique parameter data into a 2d-array of [num_params,N]
        if turntabledata is None:  # If no turntable data are passed in
            turntabledata = np.zeros(
                [1, 3]
            )  # then create a 1 measurement array full of zeros
        numturntable = turntabledata.shape[
            0
        ]  # get the number of measurements from the first dimension
        numv = turntabledata.shape[1]  # get the number of parameters
        if numv != 3:  # if its not 3
            values = np.zeros(
                [numturntable, 3]
            )  # then resize the array to 3 with the last element equal to zero
            values[:, 0] = turntabledata[:, 0]  # copy over the first set of parameters
            values[:, 1] = turntabledata[:, 1]  # and the second set
            turntabledata = values  # and assign to the turntabledata
        return numturntable, turntabledata  # and we are done.

    # ------------------------------------------------------------------------------
    #           add_measurement_set
    # ------------------------------------------------------------------------------
    def add_measurement_set(
        self,
        utc: float
        | (
            str
            | (
                np.datetime64
                | (
                    datetime
                    | (
                        np.ndarray
                        | Sequence[
                            float | (str | (np.datetime64 | (datetime | np.ndarray)))
                        ]
                    )
                )
            )
        ),
        platform_position: tuple[str, Sequence | np.ndarray],
        platform_orientation: tuple[str, str, Sequence | np.ndarray],
        icf_orientation: tuple[str, str, Sequence | np.ndarray] | None = None,
    ):
        """
        Adds a set of measurements to the parent :class:~.Instrument`. The call adds **N** distinct measurements with
        **N* distinct settings of *time*, *position* and *orientation*. For convenience, settings for one variable can be represented
        with one value and it will be automatically expanded to the number of measurements represented in the other variables, i.e.
        borrowing the idea of **broadcasting** from *numpy*. Multiple measurement sets can be added to the :class:`~.Instrument`

        Parameters
        ----------
        utc: Array, sequence or scalar of string, float, datetime or datetime64
            The universal times of the set of measurements. It can be an array, sequence or scalar and each element
            should be a representation of universal time. If it is a scalar value then the same time value is assigned to every
            position given in `observer_positions`. All elements of the array should be of the same type and should represent
            universal time.
        roll_control: str
            The technique used to specify the location of zero roll in the :ref:`ecef` system. A description of values can
            be found in :ref:`rollcontrol`.
        platform_position: Tuple[str, sequence]
            A two element tuple. The first element of the tuple is a string that specifies the positioning technique. The second element
            of the tuple is a sequence or array (or None) that contains parameter data for the selected technique. The array should be of size (Npts, NParam)
            where  *Npts* is the number of points and *Nparam* is the number of parameters for the technique, e.g. (N,3), in the measurement set. The
            second element may be None if the technique is `from_platform`.
        platform_orientation:
            A three element tuple. The first element of the tuple is a string that specifies the orientation technique. The second element
            of the tuple is a sequence or array (or None) that contains parameter data for the selected technique. The array should be of size (Npts, NParam)
            where  *Npts* is the number of points and *Nparam* is the number of parameters for the technique, e.g. (N,3), in the measurement set. The
            second element may be None if the technique is `from_platform`.
        icf_orientation:
            Optional. A two element tuuple.  This option allows used to specify an array of instantaneous look vectors in the :ref:`icf`.
            The first element of the tuple is a string that specifies the ref:`icf` look vector  technique. The second element
            of the tuple is a sequence or array (or None) that contains parameter data for the selected technique. The array should be of size (Numlookvector, NParam)
            where  *Numlookvector* is the number of instantaneous look vectors points and *Nparam* is the number of parameters for the look vector technique, e.g. (N,3),
        """

        utdata = ut_to_datetime64(
            utc
        )  # Convert the incoming array of times to datetime64 for numpy
        if isinstance(
            utdata, np.datetime64
        ):  # If a scalar value comes back then we resize the times to math the positions
            utdata = np.array([utdata])  # make it into an array

        numtimes = utdata.size  # Get the number of time
        numpositions, positiondata, positionkey = self._decode_position_entry(
            platform_position
        )  # Get the number of positions, the key
        (
            numorientations,
            orientationdata,
            orientkey,
            roll_control,
        ) = self._decode_platform_orientation_entry(platform_orientation)
        numicflook, icflookdata, icfkey = self._decode_icf_look_vector_entry(
            icf_orientation
        )
        maxpts = np.max((numtimes, numpositions, numorientations, numicflook))

        if numtimes == 1:
            utdata = np.resize(
                utdata, [maxpts]
            )  # then resize the times array by simply resizing the current array
            numtimes = maxpts  # and reset the number of times
        else:
            if numtimes != maxpts:
                msg = "The number of time parameters given {} does not equal 1 or the number of looks {} and  and orientations given {}".format(
                    numtimes, numpositions, numorientations
                )
                raise ValueError(msg)

        if (
            positionkey == "from_platform"
        ):  # if the technique parameter array returns as None
            positiondata = maxpts  # set the technique parameter data as the number of times, used when passed to the technique algorithm
            numpositions = (
                maxpts  # and add the number of times to the number of positions
            )
        elif numpositions == 1:  # otherwise we have valied technique parameter data
            positiondata = np.tile(positiondata, [maxpts, 1])
            numpositions = (
                maxpts  # so get the number of points as the first element of the shape
            )
        else:
            if numpositions != maxpts:
                msg = "The number of position parameters given {} does not equal 1 or the number of times {} and orientations given {}".format(
                    numpositions, numtimes, numorientations
                )
                raise ValueError(msg)

        if (
            orientkey == "from_platform"
        ):  # if the technique parameter array returns as None
            orientationdata = maxpts  # set the technique parameter data as the number of times, used when passed to the technique algorithm
        elif numorientations == 1:  # otherwise we have valied technique parameter data
            orientationdata = np.tile(orientationdata, [maxpts, 1])
        else:
            if numorientations != maxpts:
                msg = "The number of orientation parameters given {} does not equal 1 or the number of times {} and positions {} given".format(
                    numorientations, numtimes, numpositions
                )
                raise ValueError(msg)

        if (numicflook == 1) and (icflookdata is not None):
            icflookdata = np.tile(icflookdata, [maxpts, 1])
        self._utc.append(
            utdata
        )  # Add the UTC times now that they are finalized in size
        self._position_definitions.append(
            (positionkey, positiondata)
        )  # if it is then append the definitions to the internal measurement set within this object
        self._platform_orientation_definitions.append(
            (orientkey, orientationdata, roll_control)
        )  # append these look vectors definitions to the internal measurement set within thgis object.
        self._icf_lookvector_definitions.append((icfkey, icflookdata))
        self._isdirty = True

    # ------------------------------------------------------------------------------
    #           make_observation_set
    # ------------------------------------------------------------------------------
    def make_observation_set(self, platform: Platform):
        """
        Converts the internal list of measurements from previous calls to :meth:`~.add_measurement_set` into an
        an internally cached :class:`~.PositionAndOrientationArray`. This method is normally called by the :class:`~.Platform` class.
        The internal list of measurement sets has only been cached up to this point. This code goes through the measurements
        and generates all the necessary platform positions and rotation matrices using all the specified techniques.

        Parameters
        ----------
        platform : Platform
            The :class:`~.Platform` object that will be updated with the measurements defined in this object.
        """
        pointingalgorithms = PointingAlgorithms(platform)
        postype = "----"
        orientationtype = "----"
        ilookseg = -1
        numlook = -1
        ilook = 0
        iposseg = -1
        numpos = -1
        ipos = 0
        iutc = 0
        iturn = 0
        numturn = -1
        iturnseg = -1
        allok = True

        for iseg in range(
            len(self._utc)
        ):  # Loop through all of the time segements. There is usually just one
            currentutc = self._utc[iseg]  # Get the current array of universal times
            numutc = (
                currentutc.size
            )  # Get the number of universal times in this segment
            for it in range(
                numutc
            ):  # Loop over all of the time values, we have already validated that the total number of positions and look arrays are the same size as the time array
                iutc += 1  # increase the time index to the next time measurement in the set of segments
                utc = currentutc[it]  # get the time. This should be np.datetime64

                iturn += 1  # Step to the next position entry in the current segment of internal turntable orientation entries
                if (
                    iturn >= numturn
                ):  # if we have exceeded the bounds of the current segment
                    iturnseg += 1  # then step to the next segment (it whould work as we have already validated data sets)
                    icflooktype, currentturn = self._icf_lookvector_definitions[
                        ilookseg
                    ]  # get the techniquename  to calculate ICF look vectors and the parameter data for this setting
                    icflookfunc = self._icf_look_vector_convertors[icflooktype][
                        0
                    ]  # get the ICF look vector technique function
                    iturn = 0  # Reset our position in the current segment
                    assert (currentturn is None) or (
                        type(currentturn) is np.ndarray
                    )  # We must have an array. There is no "from_platform" option
                    numturn = (
                        currentturn.shape[0] if currentturn is not None else 1
                    )  # then get the number of positions in this segment
                thisturn = (
                    currentturn[iturn, :] if currentturn is not None else None
                )  # get the technique parameter data for this measurement point.
                ok3 = icflookfunc(
                    platform, thisturn
                )  # Call the positioning technique function. Its sets the Platform position State. It returns false if it fails.

                ipos += 1  # Step to the next position entry in the current segment of position entries
                if (
                    ipos >= numpos
                ):  # if we have exceeded the bounds of the current segment
                    iposseg += 1  # then step to the next segment (it whould work as we have already validated data sets)
                    postype, currentpos = self._position_definitions[
                        iposseg
                    ]  # get the name of the position technique and the current position technique data for this segment
                    posfunc = self._position_convertors[postype][
                        0
                    ]  # Get the function we should call fo rthis technique
                    ipos = 0  # Reset our position in the current segment
                    if (
                        type(currentpos) is np.ndarray
                    ):  # If the position technique parameter is an array
                        numpos = currentpos.shape[
                            0
                        ]  # then get the number of positions ion this segment
                    else:  # otherwise its the special case, e.g. 'from_platform'
                        numpos = currentpos  # and the number of points is internally stored as the technique parameter data
                        currentpos = None  # and the real parameter data is None

                thispos = (
                    currentpos[ipos, :] if currentpos is not None else None
                )  # get the technique parameter data for this measurement point.
                ok1 = posfunc(
                    platform, utc, thispos
                )  # Call the positioning technique function. Its sets the Platform position State. It returns false if it fails.

                ilook += 1  # Step to the next position entry in the current list of orientation technique entries
                if (
                    ilook >= numlook
                ):  # if we have exceeded the bounds of the current segment
                    ilookseg += (
                        1  # then step to the next segment. It should always work
                    )
                    (
                        orientationtype,
                        currentlook,
                        roll_control,
                    ) = self._platform_orientation_definitions[
                        ilookseg
                    ]  # get the technique, parameter data and roll control for this settings
                    lookfunc = self._platform_orientation_convertors[orientationtype][
                        0
                    ]  # get the look technique function

                    ilook = 0
                    if type(currentlook) is np.ndarray:
                        numlook = currentlook.shape[0]
                    else:
                        numlook = currentlook
                        currentlook = None
                thislook = currentlook[ilook, :] if currentlook is not None else None
                ok2 = lookfunc(pointingalgorithms, thislook, roll_control)

                if ok1 and ok2 and ok3:
                    platform.add_current_platform_state()
                else:
                    logging.warning(
                        "Rejecting point {} from the observation policy. Positioning technique={} status = {}, Pointing Technique={} status = {}, Internal Rotation status = {}".format(  # noqa: G001
                            iutc, postype, ok1, orientationtype, ok2, ok3
                        )
                    )
                    allok = False
        if not allok:
            msg = "Platform.make_observation_set failed to create a valid observation set. Warning messages have been written to logging.warning"
            raise RuntimeError(msg)
        self._isdirty = False
