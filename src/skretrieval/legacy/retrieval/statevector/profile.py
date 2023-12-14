from __future__ import annotations

import abc
from copy import copy

import numpy as np
import sasktran as sk
import xarray as xr

from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.tikhonov import (
    two_dim_vertical_first_deriv,
    two_dim_vertical_second_deriv,
)
from skretrieval.util import linear_interpolating_matrix


class StateVectorProfile(StateVectorElement):
    def __init__(self, species_name: str, optical_property: sk.OpticalProperty):
        """
        A state vector component that represents a profile of a scattering or absorbing species within the atmosphere.
        This is a base class which other classes can derive


        Parameters
        ----------
        species_name: str
            Name of the species
        optical_property: sk.OpticalProperty
            Optical property of the species
        """
        self._species_name = species_name

        self._optical_property = optical_property
        self._species = None
        self._internal_update()

    def _internal_update(self):
        """
        Updates the internal climatology
        """
        alts, density = self._climatology_altitudes_and_density()
        if self._species is None:
            self._species = sk.Species(
                self._optical_property,
                sk.ClimatologyUserDefined(alts, {self._species_name: density}),
                self._species_name,
            )
        else:
            self._species.climatology[self._species_name] = density

    @abc.abstractmethod
    def _climatology_altitudes_and_density(self):
        pass

    def name(self) -> str:
        return self._species_name

    def add_to_atmosphere(self, atmosphere: sk.Atmosphere):
        """
        Adds the species to the atmosphere

        Parameters
        ----------
        atmosphere: sk.Atmosphere

        """
        atmosphere[self._species_name] = self._species

        if atmosphere.wf_species is None:
            atmosphere.wf_species = [self._species_name]

        if self._species_name not in atmosphere.wf_species:
            old_wf_species = copy(atmosphere.wf_species)
            old_wf_species.append(self._species_name)
            atmosphere.wf_species = old_wf_species


class StateVectorProfilePPM(StateVectorProfile):
    def propagate_wf(self, radiance) -> np.ndarray:
        interp_matrix = linear_interpolating_matrix(
            self._altitudes_m, self._species.climatology.altitudes
        )

        # We have dI/dn, but we need dI/dx, so we have to sum over the deltas that each ppm delta introduces
        backgroundstate = np.asarray(
            self._background_climatology["SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3"]
        )

        wf_name = "wf_" + self.name()
        if wf_name not in radiance:
            msg = "WF not calculated"
            raise ValueError(msg)

        # inlclude ppm conversion here
        new_wf = radiance[wf_name] * backgroundstate / 1e6
        return new_wf @ xr.DataArray(
            data=interp_matrix,
            dims=["perturbation", "x"],
            coords={"perturbation": radiance["perturbation"].to_numpy()},
        )

    def apriori_state(self) -> np.array:
        return self._prior_ppm

    def inverse_apriori_covariance(self) -> np.ndarray:
        return np.linalg.inv(self._prior_ppm_covar)

    def __init__(
        self,
        altitudes_m: np.array,
        values_ppm: np.array,
        species_name: str,
        optical_property: sk.OpticalProperty,
        background_climatology: sk.ClimatologyUserDefined,
        prior_ppm: np.array,
        priori_ppm_covar: np.ndarray,
    ):
        """
        State vector element which defines a vertical profile based on a secondary vertical profile of ppm values.
        The secondary profile can be on any vertical grid (coarser or equal to the regular atmospheric grid, not
        finer)

        Parameters
        ----------
        altitudes_m: np.array
            Altitudes of the PPM grid
        values_ppm: np.array
            ppm of the species
        species_name: np.array
            Name of the species within the atmosphere
        optical_property: sk.OpticalProperty
            Optical property of the species in question
        background_climatology: sk.Climatology
            Climatology which contains a key 'SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3' that is the background atmospheric
            number density
        prior_ppm: np.array
            Apriori ppm values for the profile
        priori_ppm_covar: np.ndarray
            Apriori covariance matrix
        """
        self._altitudes_m = np.atleast_1d(altitudes_m)
        self._values_ppm = np.atleast_1d(values_ppm)

        self._prior_ppm = prior_ppm
        self._prior_ppm_covar = priori_ppm_covar

        self._background_climatology = background_climatology

        super().__init__(species_name, optical_property)

    def _climatology_altitudes_and_density(self):
        return self._background_climatology.altitudes, self._convert_to_numden()

    def _convert_to_numden(self):
        background_values = self._background_climatology[
            "SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3"
        ]
        background_altitudes = self._background_climatology.altitudes

        species_fraction = self._values_ppm / 1e6  # Convert from ppm

        interp_fraction = np.interp(
            background_altitudes, self._altitudes_m, species_fraction
        )

        return np.asarray(interp_fraction * background_values)

    def state(self) -> np.array:
        return copy(self._values_ppm)

    def update_state(self, x: np.array):
        self._values_ppm = copy(np.atleast_1d(x))
        self._internal_update()


class StateVectorProfileScale(StateVectorProfile):
    def propagate_wf(self, radiance) -> np.ndarray:
        interp = np.ones((len(self._species.climatology.altitudes), 1))

        wf_name = "wf_" + self.name()

        old_wf = radiance[wf_name]

        return (
            old_wf * self._values[np.newaxis, :] * self._scale_factor
        ) @ xr.DataArray(
            data=interp,
            dims=["perturbation", "x"],
            coords={"perturbation": old_wf["perturbation"].to_numpy()},
        )

    def __init__(
        self,
        scale_factor: float,
        species_name: str,
        optical_property: sk.OpticalProperty,
        altitudes_m: np.array,
        values: np.array,
        scale_factor_sigma: float | None = None,
        scale_factor_prior: float = 1.0,
    ):
        """
        State vector element which defines a vertical profile based on a single scale factor of another profile

        Parameters
        ----------
        scale_factor: float
            Scale factor
        species_name: str
            Name of the species within the atmosphere object
        optical_property: sk.OpticalProperty
            Optical property of the species
        altitudes_m: np.array
            Array of altitudes that define the atmospheric grid
        values: np.array
            Profile values of the underlying profile that is to be scaled
        scale_factor_sigma: float, optional
            1sigma prior value, defualt is None which indicates no prior
        scale_factor_prior: float, optional
            Prior value on the scale factor, default is 1
        """
        self._scale_factor = scale_factor
        self._scale_factor_prior = scale_factor_prior
        self._scale_factor_sigma = scale_factor_sigma
        self._altitudes_m = altitudes_m
        self._values = values

        super().__init__(species_name, optical_property)

    def _climatology_altitudes_and_density(self):
        return self._altitudes_m, self._values * self._scale_factor

    def state(self) -> np.array:
        return [copy(self._scale_factor)]

    def update_state(self, x: np.array):
        self._scale_factor = copy(float(x))
        self._internal_update()

    def described_state(self, stdev: np.array):
        return xr.Dataset(
            {
                f"{self.name()}_scale_factor": float(self._scale_factor),
                f"{self.name()}_scale_factor_stdev": float(stdev),
            }
        )

    def apriori_state(self) -> np.array:
        return [copy(self._scale_factor_prior)]

    def inverse_apriori_covariance(self) -> np.ndarray:
        inv_covar = np.zeros((1, 1), np.float64)
        if self._scale_factor_sigma is not None:
            inv_covar[0, 0] = 1 / (self._scale_factor_sigma**2)

        return inv_covar


class StateVectorProfileLogND(StateVectorProfile):
    def propagate_wf(self, radiance) -> xr.Dataset:
        wf = radiance["wf_" + self.name()]
        new_wf = wf * np.exp(self._values)[np.newaxis, np.newaxis, :]
        return (
            new_wf.isel(perturbation=~self._zero_mask)
            .drop("perturbation")
            .rename({"perturbation": "x"})
        )

    def __init__(
        self,
        altitudes_m: np.array,
        values: np.array,
        species_name: str,
        optical_property: sk.OpticalProperty,
        lowerbound: float,
        upperbound: float,
        second_order_tikhonov_factor: float,
        bounding_factor=1e4,
        zero_factor: float = 1e-20,
        max_update_factor: float = 5,
    ):
        """
        A state vector element which defines a vertical profile as the logarithm of number density.  The profile
        is bounded above and below by the upperbound and lowerbound through the use of the apriori.  A second
        order tikhonov smoothing is included.  Number dennsities less than a certain factor (zero_factor) are treated as
        identically equal to 0 within the RTM to prevent numerical precision problems

        Parameters
        ----------
        altitudes_m: np.array
            Altitudes of the profile, should match weighting function altitudes
        values: np.array
            log(number density) for the profile
        species_name: str
            Name of the species in the atmosphere
        optical_property: sk.OpticalProperty
            Optical property of the species
        lowerbound: float
            Lowerbound of the retrieval in m
        upperbound: float
            Upperbound of the retrieval in m
        second_order_tikhonov_factor: float
            Second order tikhonov factor
        bounding_factor: float, optional
            Bounding factor used for the apriori upper/lower bounding. Default 1e4
        zero_factor: float, optional
            Number densities less than this number are treated as identically 0.  Default 1e-20
        max_update_factor: float, optional
            Maximum factor to update any element of the state vector every iteration. Default 5
        """
        self._altitudes_m = altitudes_m
        self._values = values
        self._zero_factor = zero_factor
        self._zero_mask = np.exp(self._values) < self._zero_factor
        self._lowerbound = lowerbound
        self._upperbound = upperbound
        self._second_order_tikhonov_factor = second_order_tikhonov_factor
        self._bounding_factor = bounding_factor
        self._max_update_factor = max_update_factor

        super().__init__(species_name, optical_property)

        self._initial_state = copy(self.state())
        self._compute_apriori_covariance()

    def _climatology_altitudes_and_density(self):
        values = np.exp(self._values)
        values[self._zero_mask] = 0.0

        return self._altitudes_m, values

    def _compute_apriori_covariance(self):
        n = len(self._values[~self._zero_mask])
        # Link above
        gamma = two_dim_vertical_first_deriv(1, n, factor=self._bounding_factor)
        bounded_altitudes = self._altitudes_m[~self._zero_mask] > self._upperbound
        first_idx = np.nonzero(bounded_altitudes)[0][0]
        bounded_altitudes[first_idx - 1] = True
        gamma[~bounded_altitudes, :] = 0
        self._inverse_Sa_bounding = gamma.T @ gamma

        # Link below
        gamma = two_dim_vertical_first_deriv(1, n, factor=self._bounding_factor)
        bounded_altitudes = self._altitudes_m[~self._zero_mask] < self._lowerbound
        gamma[~bounded_altitudes, :] = 0

        self._inverse_Sa_bounding += gamma.T @ gamma

        gamma = two_dim_vertical_second_deriv(
            1, n, factor=self._second_order_tikhonov_factor
        )
        self._inverse_Sa = gamma.T @ gamma

    def state(self) -> np.array:
        return copy(self._values[~self._zero_mask])

    def update_state(self, x: np.array):
        m_factors = np.exp(x - (self._values[~self._zero_mask]))
        m_factors[m_factors > self._max_update_factor] = self._max_update_factor
        m_factors[m_factors < 1 / self._max_update_factor] = 1 / self._max_update_factor
        self._values[~self._zero_mask] = np.log(
            copy(m_factors * np.exp(self._values[~self._zero_mask]))
        )
        self._internal_update()

    def apriori_state(self) -> np.array:
        x_a_bounding = self._initial_state
        full_inv_S_a = self._inverse_Sa + self._inverse_Sa_bounding
        return np.linalg.solve(full_inv_S_a, self._inverse_Sa_bounding @ x_a_bounding)

    def inverse_apriori_covariance(self) -> np.ndarray:
        return self._inverse_Sa_bounding + self._inverse_Sa

    def species(self):
        return self._species


class StateVectorProfileND(StateVectorProfile):
    def propagate_wf(self, radiance) -> xr.Dataset:
        interp_matrix = linear_interpolating_matrix(
            self._altitudes_m[self._state_mask], self._altitudes_m
        )
        wf_name = "wf_" + self.name()
        if wf_name not in radiance:
            msg = "WF not calculated"
            raise ValueError(msg)

        new_wf = radiance[wf_name]
        new_matrix = (
            self._values[:, np.newaxis]
            * interp_matrix
            / self._values[self._state_mask][np.newaxis, :]
        )
        return new_wf @ xr.DataArray(
            data=new_matrix,
            dims=["perturbation", "x"],
            coords={"perturbation": radiance["perturbation"].to_numpy()},
        )

    def __init__(
        self,
        altitudes_m: np.array,
        values: np.array,
        species_name: str,
        optical_property: sk.OpticalProperty,
        lowerbound: float,
        upperbound: float,
        second_order_tikhonov_factor: float,
        max_update_factor: float = 5,
    ):
        """

        Parameters
        ----------
        altitudes_m: np.array
            Altitudes of the profile, should match weighting function altitudes
        values: np.array
            log(number density) for the profile
        species_name: str
            Name of the species in the atmosphere
        optical_property: sk.OpticalProperty
            Optical property of the species
        lowerbound: float
            Lowerbound of the retrieval in m
        upperbound: float
            Upperbound of the retrieval in m
        second_order_tikhonov_factor: float
            Second order tikhonov factor
        max_update_factor: float, optional
            Maximum factor to update any element of the state vector every iteration. Default 5
        """
        self._altitudes_m = altitudes_m
        self._values = values
        self._lowerbound = lowerbound
        self._upperbound = upperbound

        self._state_mask = (self._altitudes_m > lowerbound) & (
            self._altitudes_m < upperbound
        )

        self._second_order_tikhonov_factor = second_order_tikhonov_factor
        self._max_update_factor = max_update_factor

        super().__init__(species_name, optical_property)

        self._initial_state = copy(self.state())
        self._compute_apriori_covariance()

    def _climatology_altitudes_and_density(self):
        values = copy(self._values)

        return self._altitudes_m, values

    def _compute_apriori_covariance(self):
        n = len(self._values[self._state_mask])
        gamma = two_dim_vertical_second_deriv(
            1, n, factor=self._second_order_tikhonov_factor
        )
        self._inverse_Sa = gamma.T @ gamma

    def state(self) -> np.array:
        return copy(self._values[self._state_mask])

    def update_state(self, x: np.array):
        m_factors = x / (self._values[self._state_mask])
        if self._max_update_factor is not None:
            m_factors[m_factors > self._max_update_factor] = self._max_update_factor
            m_factors[m_factors < 1 / self._max_update_factor] = (
                1 / self._max_update_factor
            )

        m_factors = np.interp(
            self._altitudes_m, self._altitudes_m[self._state_mask], m_factors
        )

        self._values = copy(m_factors * self._values)
        self._internal_update()

    def apriori_state(self) -> np.array:
        return self._initial_state

    def inverse_apriori_covariance(self) -> np.ndarray:
        return self._inverse_Sa

    def species(self):
        return self._species
