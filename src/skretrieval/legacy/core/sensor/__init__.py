from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sasktran import Geometry

import skretrieval.core.radianceformat as radianceformat
from skretrieval.core import OpticalGeometry


class Sensor(ABC):
    """
    A sensor is an object which takes in radiances at the front of an aperture, and returns backend radiances
    observed by the sensor.
    """

    @abstractmethod
    def model_radiance(
        self,
        optical_geometry: OpticalGeometry,
        model_wavel_nm: np.array,
        model_geometry: Geometry,
        radiance: np.array,
        wf=None,
    ) -> radianceformat.RadianceBase:
        """
        Takes in high resolution radiances at the front of the aperture and converts them to what is observed by the
        sensor.

        Parameters
        ----------
        optical_geometry : OpticalGeometry
            The orientation, look vector, time, position of the sensor
        model_wavel_nm : np.array
            Array of radiances that correspond to radiance
        model_geometry : Geometry
            The Geometry object that corresponds to radiance
        radiance : np.array
            An array (wavelength, line of sight) of input radiances
        wf : np.array, optional
            An array (wavelength, line of sight, perturbation) corresponding to weighting functions. Default None.

        Returns
        -------
        radianceformat.RadianceBase
            Output L1 radiances in a format specific to the sensor.  The format is defined by `radiance_format`

        """

    @abstractmethod
    def measurement_geometry(self, optical_geometry: OpticalGeometry):
        """
        Takes in the sensors orientation and returns back a Geometry object corresonding to the central
        point of each measurement.

        Parameters
        ----------
        optical_geometry: OpticalGeometry
            Sensor orientation

        Returns
        -------
        Geometry
        """

    @abstractmethod
    def measurement_wavelengths(self):
        """

        Returns
        -------
        np.array
            Central wavelengths of the measurement
        """

    @staticmethod
    @abstractmethod
    def radiance_format():
        """
        Returns the format that model_radiance returns data in

        Returns
        -------
        A specialized instance of radianceformat.RadianceBase
        """
