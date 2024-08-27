from __future__ import annotations

import abc

import numpy as np
import sasktran2 as sk2


@abc.ABC
class Ancillary:
    @abc.abstractmethod
    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        pass


class GenericAncillary(Ancillary):
    def __init__(
        self,
        altitudes_m: np.array,
        pressure_pa: np.array,
        temperature_k: np.array,
        rayleigh_scattering=True,
        o2o2=True,
    ) -> None:
        self._altitudes_m = altitudes_m
        self._pressure_pa = pressure_pa
        self._temperature_k = temperature_k
        self._rayleigh_scattering = rayleigh_scattering
        self._o2o2 = o2o2

    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        if self._rayleigh_scattering:
            atmo["rayleigh"] = sk2.constituent.Rayleigh()

        if self._o2o2:
            atmo["o2o2"] = sk2.constituent.CollisionInducedAbsorber(
                sk2.optical.HITRANCollision("O2O2"), "O2O2"
            )

        atmo.pressure_pa = np.interp(
            atmo.model_geometry.altitudes(), self._altitudes_m, self._pressure_pa
        )
        atmo.temperature_k = np.interp(
            atmo.model_geometry.altitudes(), self._altitudes_m, self._temperature_k
        )
