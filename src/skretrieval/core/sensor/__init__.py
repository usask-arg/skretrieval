from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import skretrieval.core.radianceformat as radianceformat
from skretrieval.core import OpticalGeometry
from skretrieval.core.sasktranformat import SASKTRANRadiance


class Sensor(ABC):
    """
    A sensor is an object which takes in radiances at the front of an aperture, and returns backend radiances
    observed by the sensor.
    """

    @abstractmethod
    def model_radiance(
        self,
        radiance: SASKTRANRadiance,
        orientation: OpticalGeometry,
    ) -> radianceformat.RadianceBase:
        """
        Takes in high resolution radiances at the front of the aperture and converts them to what is observed by the
        sensor.

        Parameters
        ----------
        radiance : SasktranFormat
            Input radiances in the format of the SASKTRAN radiative transfer model

        orientation : OpticalGeometry
            Orientation of the sensor

        Returns
        -------
        radianceformat.RadianceBase
            Output L1 radiances in a format specific to the sensor.  The format is defined by `radiance_format`

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
