from __future__ import annotations

from copy import copy

import numpy as np
import sasktran2 as sk2
import xarray as xr
from scipy.linalg import block_diag, toeplitz

from skretrieval.retrieval.tikhonov import two_dim_vertical_first_deriv

from . import StateVectorElement


class StateVectorElementConstituent(
    StateVectorElement, sk2.constituent.base.Constituent
):
    def __init__(
        self,
        constituent: sk2.constituent.base.Constituent,
        constituent_name: str,
        property_names: list[str],
        min_value=None,
        max_value=None,
        prior_influence=0,
        first_order_tikh=None,
        log_space=False,
        enabled=True,
    ):
        """
        A state vector element that is a sasktran2.constituent

        Parameters
        ----------
        constituent : sk2.constituent.base.Constituent
            The sasktran2 constituent
        constituent_name : str
            A name for the constituent
        property_names : list[str]
            Property names of the constituent that will be retrieved
        min_value : dict, optional
            Minimum values for the property names as a dictionary, by default {}
        max_value : dict, optional
            maximumum values for the property names as a dictionary, by default {}
        prior_influence : int, optional
            Prior influence for the property names as a dictionary, by default 0
        first_order_tikh : dict, optional
            First order tikhonov factors for the property names as a dictionary, by default {}
        log_space : bool, optional
            If true then the state elements will be rescaled to logarithmic space, by default False
        """
        if first_order_tikh is None:
            first_order_tikh = {}
        if max_value is None:
            max_value = {}
        if min_value is None:
            min_value = {}
        self._log_space = log_space
        self._constituent = constituent
        self._property_names = property_names
        self._constituent_name = constituent_name
        self._min_value = min_value
        self._max_value = max_value

        self._prior = copy(self.state())
        self._prior_influence = prior_influence
        self._first_order_tikh = first_order_tikh

        self._prior_dict = {}
        for property_name in self._property_names:
            self._prior_dict[property_name] = copy(
                getattr(self._constituent, property_name)
            )
        super().__init__(enabled)

    def state(self) -> np.array:
        data = []

        for property_name in self._property_names:
            data.append(getattr(self._constituent, property_name))

        if self._log_space:
            return np.log(np.hstack(data))
        return np.hstack(data)

    def lower_bound(self) -> np.array:
        data = []
        for property_name in self._property_names:
            x = getattr(self._constituent, property_name)

            bound = self._min_value.get(property_name, -np.inf)

            data.append(np.ones(len(x)) * bound)

        if self._log_space:
            return np.log(np.hstack(data))
        return np.hstack(data)

    def upper_bound(self) -> np.array:
        data = []
        for property_name in self._property_names:
            x = getattr(self._constituent, property_name)

            bound = self._max_value.get(property_name, np.inf)

            data.append(np.ones(len(x)) * bound)

        if self._log_space:
            return np.log(np.hstack(data))
        return np.hstack(data)

    def inverse_apriori_covariance(self) -> np.ndarray:
        prior_mats = []

        for property_name in self._property_names:
            val = getattr(self._constituent, property_name)

            corel = np.zeros_like(val)
            corel[0] = 1
            # corel[1] = 0.7
            # corel[2] = 0.5
            # corel[3] = 0.3
            # corel[4] = 0.1

            if self._log_space:
                prior_mats.append(
                    np.linalg.inv(
                        toeplitz(corel) * self._prior_influence[property_name] ** 2
                    )
                )
            else:
                prior_mats.append(
                    np.linalg.inv(
                        toeplitz(corel)
                        * np.outer(
                            self._prior_dict[property_name],
                            self._prior_dict[property_name],
                        )
                        * self._prior_influence[property_name] ** 2
                    )
                )

            if property_name == "vmr" and self._constituent_name == "so2":
                prior_mats[-1][:20, :20] *= 1e10
                prior_mats[-1][-40:, -40:] *= 1e10

            if property_name in self._first_order_tikh:
                gamma = two_dim_vertical_first_deriv(1, len(val)) * (
                    1 / self._first_order_tikh[property_name]
                )

                if not self._log_space:
                    gamma /= self._prior_dict[property_name][np.newaxis, :]

                prior_mats[-1] += gamma.T @ gamma

        return block_diag(*prior_mats)

    def apriori_state(self) -> np.array:
        return self._prior

    def name(self) -> str:
        return self._constituent_name

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        if "extinction_per_m" in self._property_names:
            radiance = radiance.rename(
                {
                    f"wf_{self._constituent_name}_extinction": f"wf_{self._constituent_name}_extinction_per_m"
                }
            )
        wfs = []
        for property_name in self._property_names:
            wfs.append(
                radiance[f"wf_{self._constituent_name}_{property_name}"].rename(
                    {
                        radiance[f"wf_{self._constituent_name}_{property_name}"].dims[
                            0
                        ]: "x"
                    }
                )
            )

            if self._log_space:
                x = getattr(self._constituent, property_name)
                wfs[-1].values *= x[:, np.newaxis, np.newaxis, np.newaxis]

        return xr.concat(wfs, dim="x")

    def update_state(self, x: np.array):
        start = 0
        for property_name in self._property_names:
            current = getattr(self._constituent, property_name)
            property_length = len(current)
            if self._log_space:
                sv = np.exp(x[start : start + property_length])
                sv[np.isnan(sv)] = self._max_value[property_name]
            else:
                sv = x[start : start + property_length]
            if property_name in self._min_value:
                sv[sv < self._min_value[property_name]] = self._min_value[property_name]
            if property_name in self._max_value:
                sv[sv > self._max_value[property_name]] = self._max_value[property_name]

            self._constituent.__setattr__(property_name, sv)

            start += property_length

    def modify_input_radiance(self, radiance: xr.Dataset):
        return radiance

    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        return self._constituent.add_to_atmosphere(atmo)

    def register_derivative(self, atmo: sk2.Atmosphere, name: str):
        return self._constituent.register_derivative(atmo, name)
