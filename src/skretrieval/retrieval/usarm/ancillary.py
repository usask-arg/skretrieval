from __future__ import annotations

import abc

import numpy as np
import sasktran2 as sk2


class Ancillary:
    """
    Ancillary data for the forward model. Typically this is things like temperature/pressure profiles, etc.

    This class is responsible for adding this information to the atmosphere object.
    """

    @abc.abstractmethod
    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        """
        Adds the ancillary data to the atmosphere object

        Parameters
        ----------
        atmo : sk2.Atmosphere
        """


class GenericAncillary(Ancillary):
    def __init__(
        self,
        altitudes_m: np.array,
        pressure_pa: np.array,
        temperature_k: np.array,
        rayleigh_scattering=True,
    ) -> None:
        """
        A generic ancillary object that can be used to add temperature and pressure profiles to the atmosphere
        as well as include rayleigh scattering.

        Parameters
        ----------
        altitudes_m : np.array
            Altitudes in [m]
        pressure_pa : np.array
            Pressure in [pa]
        temperature_k : np.array
            Temperature in [K]
        rayleigh_scattering : bool, optional
            Whether to add Rayleigh scattering, by default True
        """
        self._altitudes_m = altitudes_m
        self._pressure_pa = pressure_pa
        self._temperature_k = temperature_k
        self._rayleigh_scattering = rayleigh_scattering

    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        if self._rayleigh_scattering:
            atmo["rayleigh"] = sk2.constituent.Rayleigh()

        atmo.pressure_pa = np.interp(
            atmo.model_geometry.altitudes(), self._altitudes_m, self._pressure_pa
        )
        atmo.temperature_k = np.interp(
            atmo.model_geometry.altitudes(), self._altitudes_m, self._temperature_k
        )


class US76Ancillary(GenericAncillary):
    def __init__(self):
        """
        US76 Standard Atmosphere Ancillary data
        """
        super().__init__(
            sk2.climatology.us76._ALTS,
            sk2.climatology.us76._PRESSURE * 1e4,
            sk2.climatology.us76.celsius_to_kelvin(sk2.climatology.us76._TEMPERATURE_C),
        )
