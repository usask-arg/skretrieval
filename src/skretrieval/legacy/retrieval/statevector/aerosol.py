from __future__ import annotations

from copy import copy

import numpy as np
import sasktran as sk
import xarray as xr

from skretrieval.legacy.retrieval.statevector.profile import StateVectorProfile


class AerosolGaussianPressure(StateVectorProfile):
    def __init__(
        self,
        clim_alts: np.array,
        vertical_aod: float,
        aod_wavelength: float,
        layer_height: float,
        layer_width: float,
        species_name: str,
        optical_property: sk.OpticalProperty,
        prior_aod: float,
        prior_layer_height: float,
        prior_layer_width: float,
        prior_aod_sigma: float,
        prior_layer_height_sigma: float,
        prior_layer_width_sigma: float,
        background_climatology: sk.ClimatologyUserDefined,
    ):
        """
        Component of the state vector that is an arbitrary aerosol type with a vmr profile that
        has a Gaussian shape in pressure space.

        The state vector contains 3 elements, the vertical aod at a reference wavelength, the layer height, and
        the layer width.  The layer height and layer width are specified with a vertical coordinate of
        x = pressure / (surface pressure)


        Parameters
        ----------
        clim_alts: np.array
            Internal climatology altitudes to represent the profile on
        vertical_aod: float
            Vertical optical depth at aod_wavelength
        aod_wavelength: float
            Wavelength in nm that the vertical_aod is specified at
        layer_height: float
            Central layer height in units of pressure / (surface pressure)
        layer_width: float
            Gaussian 1sigma width of the layer in units of pressure / (surface pressure)
        species_name: str
            Name of the species
        optical_property: sk.OpticalProperty
            Optical property of the species
        prior_aod: float
            Prior value for the AOD
        prior_layer_height: float
            Prior value for the layer height
        prior_layer_width: float
            Prior value for the layer width
        prior_aod_sigma: float
            Prior 1sigma on the vertical aod
        prior_layer_height_sigma: float
            Prior 1sigma on the layer height
        prior_layer_width_sigma: float
            Prior 1sigma on the layer width
        background_climatology: sk.Climatology
            Climatology containing 'SKCLIMATOLOGY_PRESSURE_PA' and 'SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3'
        """
        self._background_climatology = background_climatology
        self._background_pressure = np.asarray(
            self._background_climatology["SKCLIMATOLOGY_PRESSURE_PA"]
        )
        self._background_numberdensity = np.asarray(
            self._background_climatology["SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3"]
        )

        # State params are vertical AOD, layer height, and layer width
        self._clim_alts = clim_alts
        self._vertical_aod = vertical_aod
        self._layer_height = layer_height
        self._layer_width = layer_width
        self._aod_wavelength = aod_wavelength

        self._prior_aod = prior_aod
        self._prior_layer_height = prior_layer_height
        self._prior_layer_width = prior_layer_width
        self._prior_aod_sigma = prior_aod_sigma
        self._prior_layer_height_sigma = prior_layer_height_sigma
        self._prior_layer_width_sigma = prior_layer_width_sigma

        self._xs = optical_property.calculate_cross_sections(
            sk.MSIS90(), 0, 0, 10000, 54372, [self._aod_wavelength]
        ).total[0]

        super().__init__(species_name, optical_property)

    def inverse_apriori_covariance(self) -> np.ndarray:
        n = len(self.state())
        inv_apriori_covariance = np.zeros((n, n))

        inv_apriori_covariance[0, 0] = 1 / (self._prior_aod_sigma**2)
        inv_apriori_covariance[1, 1] = 1 / (self._prior_layer_height_sigma**2)
        inv_apriori_covariance[2, 2] = 1 / (self._prior_layer_width_sigma**2)

        return inv_apriori_covariance

    def apriori_state(self) -> np.array:
        return np.array(
            [self._prior_aod, self._prior_layer_height, self._prior_layer_width]
        )

    def _climatology_altitudes_and_density(self):
        return self._clim_alts, self._nd(
            self._vertical_aod, self._layer_height, self._layer_width
        )

    def _nd(self, aod, layer_height, layer_width):
        # vmr gaussian
        gauss = np.exp(
            -0.5
            * (
                (
                    self._background_pressure / self._background_pressure[0]
                    - layer_height
                )
                / (2 * layer_width)
            )
            ** 2
        )

        # convert to number density space
        gauss *= self._background_numberdensity

        # Convert profile to have the correct vertical aod
        integral = np.trapz(gauss, self._clim_alts)

        # xs in cm2, integral is over m to we have to divide by 100
        nd = gauss / integral * aod / self._xs / 100

        return copy(nd)

    def state(self) -> np.array:
        return np.array(
            [
                copy(self._vertical_aod),
                copy(self._layer_height),
                copy(self._layer_width),
            ]
        )

    def propagate_wf(self, radiance) -> np.ndarray:
        num_factor = 1.01

        d_aod = self._nd(
            self._vertical_aod * num_factor, self._layer_height, self._layer_width
        )
        d_height = self._nd(
            self._vertical_aod, self._layer_height * num_factor, self._layer_width
        )
        d_width = self._nd(
            self._vertical_aod, self._layer_height, self._layer_width * num_factor
        )

        base = self._nd(self._vertical_aod, self._layer_height, self._layer_width)

        d_aod = (d_aod - base) / (self._vertical_aod * (num_factor - 1))
        d_height = (d_height - base) / (self._layer_height * (num_factor - 1))
        d_width = (d_width - base) / (self._layer_width * (num_factor - 1))

        matrix = np.vstack([d_aod, d_height, d_width])

        old_wf = radiance["wf_" + self.name()]

        return old_wf @ xr.DataArray(
            data=matrix.T,
            dims=["perturbation", "x"],
            coords={"perturbation": old_wf["perturbation"].to_numpy()},
        )

    def update_state(self, x: np.array):
        self._vertical_aod = copy(x[0])
        self._layer_height = copy(x[1])
        self._layer_width = copy(x[2])
        self._internal_update()
