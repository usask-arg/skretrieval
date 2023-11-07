from __future__ import annotations

import sasktran as sk
from sasktran.exceptions import wrap_skif_functionfail


class MieAerosolOrbit(sk.OpticalProperty):
    """
    Specialized OpticalProperty which supports Mie Aerosol calculations.

    Parameters
    ----------
    particlesize_climatology : sasktran.Climatology

    species : str
        Molecule to use, one of ['H2SO4', 'ICE', 'WATER']
    """

    def __init__(self, particlesize_climatology, species: str):
        super().__init__("MIEAEROSOL_" + species.upper())

        self._species = species
        self._particlesize_climatology = particlesize_climatology

    @wrap_skif_functionfail
    def _update_opticalproperty(self, **kwargs):
        self._iskopticalproperty.SetProperty(
            "SetParticleSizeClimatology",
            self._particlesize_climatology.skif_object(**kwargs),
        )

    @property
    def particlesize_climatology(self):
        return self._particlesize_climatology

    @particlesize_climatology.setter
    def particlesize_climatology(self, value):
        self._particlesize_climatology = value

    def skif_object(self, **kwargs):
        self._update_opticalproperty(**kwargs)
        return super().skif_object()
