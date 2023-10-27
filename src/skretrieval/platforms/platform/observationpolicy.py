from __future__ import annotations

import numpy as np

import skretrieval.time as sktime

from ...core import OpticalGeometry
from .platform_pointing import PlatformPointing
from .rotationmatrix import RotationMatrix


# -----------------------------------------------------------------------------
#           ObservationSet
# -----------------------------------------------------------------------------
class ObservationPolicy:
    """
    Implements the Observation Policy object for the :class:`~.Platform`. This object stores a list of
    platform states. Any observation policy breaks down into an array of samples where the platform position,
    orientation and time are specified for each sample.

    The end user will often want call the following methods:

    * :meth:`~.add_current_state` to add a new sample to the policy object
    * :meth:`~.clear` to clear the current list of samples in the policy object

    The instrument model uses the current list of platform states in the platform's :ref:`observationpolicy_class` as one of its inputs
    to the radiative transfer engine. The user must add at least one platform state to the :ref:`observationpolicy_class` for the
    radiative transfer to calculate anything.

    Real observation policies for real instruments may be much more detailed than simply saving platform position,
    orientation and time. For example, mirrors and filters may need to be adjusted, integration times and detector binning
    modes may need to be varied. These aspects are beyond the scope of the platform's observation policy and are left
    for the user to manage and implement.
    """

    def __init__(self):
        """
        Creates a new empty instance.
        """
        self._position: list[
            np.ndarray
        ] = []  # A list [Numsamples] of  array (3) platform_ecef_positions
        self._orientation: list[
            RotationMatrix
        ] = []  # A list [Numsamples] rotation matrices
        self._utc: list[
            float | np.datetime64
        ] = []  # A list [Numsamples] of numpy.datetime64['us']

    # -----------------------------------------------------------------------------
    #           platform_ecef_positions
    # -----------------------------------------------------------------------------

    def ecef_positions(self):
        """
        Returns the geocentric geographic (ECEF) position of the platform for each sample in the observation set.

        Returns
        -------
        np.ndarray( 3,Nt)
            Returns the geographic geocentric position as an array of ECEF vectors, one three element column for each sample in the observation set.
            The position is specified in meters from the centre of the Earth.
        """
        return np.array(self._position).transpose()

    # -----------------------------------------------------------------------------
    #           icf_to_ecef_rotation_matrix
    # -----------------------------------------------------------------------------

    def icf_to_ecef_rotation_matrix(self):
        """
        Returns the orientation of the platform for each sample in the observation set.

        Returns
        -------
        List[RotationMatrix]
            Returns the rotation matrix for each sample. The rotation matric will convert Instrument Control Frame unit
            vectors to geocentric geographic unit vectors. The matrix equation is `vgeo = R.R @ vicf`
        """
        return self._orientation

    # -----------------------------------------------------------------------------
    #           platform_utc
    # -----------------------------------------------------------------------------

    def utc(self):
        """
        Returns the UTC of each of the samples in the observation set. The UTC is either an array of
        numpy.datetime64 values or an array of floats representing MJD.  Note that the datetime64 are detected and converted
        to MJD inside the FrontEndRadianceGenerator before calling Sasktran.
        """
        return np.array(self._utc)

    # -----------------------------------------------------------------------------
    #           numsamples
    # -----------------------------------------------------------------------------

    def numsamples_in_observationset(self) -> int:
        """
        Returns the number of samples in this observation policy

        Returns
        -------
        int
            The number of samples in the observation policy.
        """
        return len(self._utc)

    # -----------------------------------------------------------------------------
    #           clear
    # -----------------------------------------------------------------------------

    def clear(self):
        """
        Clears the current list of platform states inside the observation policy. This method should be called
        before starting a new observation policy
        """
        self._position.clear()
        self._orientation.clear()
        self._utc.clear()

    # -----------------------------------------------------------------------------
    #           add_current_state
    # -----------------------------------------------------------------------------

    def add_current_state(self, platform_attitude: PlatformPointing) -> int:
        """
        Adds the current state of the platform to the observation policy. A typical scenario is for the end-user to
        set the position, orientation and time of the platform using one of many available methods and to then capture the
        platform state for further use within the radiative transfer engine.

        Parameters
        ----------
        platform_attitude: PlatformPointing
            The object used to define the platform position, orientation and time.

        Returns
        -------
        int
            The number of samples in the observation policy.

        """

        R = platform_attitude.get_icf_to_ecef_matrix()
        pos = platform_attitude.location()
        t = platform_attitude.utc()
        self._position.append(pos)
        self._orientation.append(R)
        self._utc.append(t)
        return self.numsamples_in_observationset()

    # ------------------------------------------------------------------------------
    #           to_optical_geometry
    # ------------------------------------------------------------------------------
    def to_optical_geometry(self) -> list[OpticalGeometry]:
        """
        Converts the platform orientations defined in thos ObservationPolicy to a list of OpticalGeometry which can be
        used in retrieval code. This conversion assumes the instrument is looking along the x axis of the :ref:`icf` the instrument
        control frame and the ``local_up`` used by the retrieval is in the direction of the Z axis in the :ref:`icf`.

        Returns
        -------
        List[OpticalGeometry]
            The list of measurement definitions used by retrieval code.
        """

        optical_geometry = []
        for i in range(self.numsamples_in_observationset()):
            obs_pos = self.ecef_positions()[:, i]
            R = self.icf_to_ecef_rotation_matrix()[i].R
            look_vector = R @ np.array([1, 0, 0])
            local_up = R @ np.array([0, 0, 1])
            mjd = sktime.ut_to_mjd(self.utc()[i])
            optical_geometry.append(
                OpticalGeometry(obs_pos, look_vector, local_up, mjd)
            )
        return optical_geometry
