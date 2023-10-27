from __future__ import annotations

import numpy as np
import sasktran as sk
from scipy import constants, interpolate

from skretrieval.core.radianceformat import RadianceBase, RadianceGridded
from skretrieval.retrieval import ForwardModel, RetrievalTarget
from skretrieval.retrieval.tikhonov import two_dim_vertical_second_deriv


class AirDensityRetrieval(RetrievalTarget):
    """
    Limb retrieval for air number density.  Radiances should be supplied in the
    RadianceGridded format.
    """

    def __init__(
        self,
        air_species: sk.Species,
        ret_wavel=350,
        high_alt_normalize=False,
        tikh_factor=1,
    ):
        """

        Parameters
        ----------
        air_species : sk.Species
            Species for air used in the forward model
        ret_wavel : float, optional
            Wavelength to do the retrieval at, default 350 nm
        high_alt_normalize: bool, optional
            If true, do high altitude normalization.  This is not recommended, default False.
        tikh_factor: float, optional
            Tikhonov regularization parameter
        """
        self._air_species = air_species

        self._retrieval_altitudes = np.arange(30500, 80500, 1000)
        self._atmosphere_altitudes = air_species.climatology.altitudes

        self.ret_wavel = ret_wavel

        self.high_alt_normalize = high_alt_normalize

        self._tikh_factor = tikh_factor

    def measurement_vector(self, l1_data: RadianceBase):
        """
        The measurement vector is the logarithm of radiance at ret_wavel
        """
        if not isinstance(l1_data, RadianceGridded):
            msg = "Class OzoneRetrieval only supports data in the form RadianceGridded"
            raise ValueError(msg)

        result = {}

        tangent_locations = l1_data.tangent_locations()
        l1_data.data["tangent_altitude"] = tangent_locations.altitude

        triplet_values = l1_data.data.where(
            (l1_data.data["tangent_altitude"] > self._retrieval_altitudes[0] - 500)
            & (l1_data.data["tangent_altitude"] < self._retrieval_altitudes[-1]),
            drop=False,
        ).sel(wavelength=self.ret_wavel, method="nearest")

        if self.high_alt_normalize:
            # Find the highest non nan tangent altitude
            highest_non_nan_idx = np.nanargmax(triplet_values.tangent_altitude.values)

            triplet_values["normed_radiance"] = triplet_values[
                "radiance"
            ] / triplet_values["radiance"].sel(los=highest_non_nan_idx)

            # Remove the highest altitude from consideration
            triplet_values["normed_radiance"] = triplet_values["normed_radiance"].where(
                triplet_values.tangent_altitude
                != triplet_values.tangent_altitude.to_numpy()[highest_non_nan_idx]
            )

        if "wf" in triplet_values:
            jac = (
                triplet_values["wf"]
                / triplet_values["radiance"]
                * np.exp(self.state_vector())
            )

            # if self.high_alt_normalize:
            #    jac -= triplet_values['wf'].sel(los=highest_non_nan_idx) / triplet_values['radiance'].sel(los=highest_non_nan_idx) * np.exp(self.state_vector())

            result["jacobian"] = jac.to_numpy()

        if self.high_alt_normalize:
            result["y"] = np.log(triplet_values["normed_radiance"].values)
        else:
            result["y"] = np.log(triplet_values["radiance"].values)

        return result

    def state_vector(self):
        """
        State vector is the logarithm of air number density
        """
        return np.log(
            self._air_species.climatology.get_parameter(
                self._air_species.species, 0, 0, self._retrieval_altitudes, 54372
            )
        )

    def update_state(self, x: np.ndarray):
        deltas = x - self.state_vector()

        # Numeric stability
        deltas[deltas < -10] = -10
        deltas[deltas > 10] = 10

        mult_factors = np.exp(deltas)

        mult_factors = np.interp(
            self._atmosphere_altitudes,
            self._retrieval_altitudes,
            mult_factors,
            left=mult_factors[0],
            right=mult_factors[-1],
        )

        mult_factors[mult_factors < 0.2] = 0.2
        mult_factors[mult_factors > 5] = 5

        self._air_species.climatology[self._air_species.species] *= mult_factors

    def temperature(self, hires_spacing=100, T0=200, earth_radius=6372000):
        """
        Integrates the air number density using hydrostatic balance and the ideal gas law.

        Parameters
        ----------
        hires_spacing : float, optional
            Internal high resolution spacing to use for the integral. Default 100 m
        T0 : float, optional
            Upper altitude pin temperature, this is at self._retrieval_altitudes[-1]. Default 200 K
        earth_radius: float, optional
            Earth radius in m, necessary to approximate gravity. Default 6372000

        Returns
        -------
        np.array
            Temperature in K on self._retrieval_altitudes
        """
        boltzmann_k = constants.Boltzmann
        avogadro = constants.Avogadro

        mean_mass_air = 28.97 / avogadro / 1000

        hires_grid = np.arange(
            self._retrieval_altitudes[0],
            self._retrieval_altitudes[-1] + hires_spacing,
            hires_spacing,
        )

        hires_n = (
            self._air_species.climatology.get_parameter(
                self._air_species.species, 0, 0, self._retrieval_altitudes, 54372
            )
            * 1e9
        )

        hires_n = np.exp(
            interpolate.interp1d(
                self._retrieval_altitudes,
                np.log(hires_n),
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )(hires_grid)
        )

        g = constants.g * (earth_radius / (earth_radius + hires_grid)) ** 2

        integrand = (g * hires_n * mean_mass_air)[::-1]

        integral = np.array(
            [
                np.trapz(integrand[:low], hires_grid[:low])
                for low in range(len(integrand))
            ]
        )[::-1]

        hires_temperature = (hires_n[-1] * T0 + integral / boltzmann_k) / hires_n

        return np.interp(self._retrieval_altitudes, hires_grid, hires_temperature)

    def apriori_state(self):
        return None

    def inverse_apriori_covariance(self):
        gamma = two_dim_vertical_second_deriv(
            1, len(self._retrieval_altitudes), self._tikh_factor
        )
        return gamma.T @ gamma

    def initialize(self, forward_model: ForwardModel, meas_l1: RadianceBase):
        return super().initialize(forward_model, meas_l1)
