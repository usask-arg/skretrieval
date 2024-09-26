from __future__ import annotations

import numpy as np
import sasktran2 as sk2
import xarray as xr
from scipy.linalg import block_diag

from skretrieval.retrieval.prior import BasePrior

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
        prior: dict[BasePrior] | None = None,
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
        prior : dict, optional
            Prior objects for each property name, by default {}
        log_space : bool, optional
            If true then the state elements will be rescaled to logarithmic space, by default False
        """
        if prior is None:
            prior = {}
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

        self._prior = prior

        start = 0
        for property_name in self._property_names:
            if property_name in self._prior:
                n = len(getattr(self._constituent, property_name))
                self._prior[property_name].init(self, slice(start, start + n))
                start += n
            else:
                self._prior[property_name] = BasePrior()

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
            inv_S_a = self._prior[property_name].inverse_covariance

            if self._log_space:
                prior_mats.append(inv_S_a)
            else:
                prior_mats.append(
                    inv_S_a
                    / np.outer(
                        self._prior[property_name].state,
                        self._prior[property_name].state,
                    )
                )

        return block_diag(*prior_mats)

    def apriori_state(self) -> np.array:
        return np.concatenate(
            [self._prior[property].state for property in self._property_names]
        )

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

    def adjust_constituent_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if k.lower() == "scale":
                        setattr(
                            self._constituent, key, getattr(self._constituent, key) * v
                        )
                    if k.lower() == "set":
                        setattr(self._constituent, key, v)
            else:
                setattr(self._constituent, key, getattr(self._constituent, key) * value)

    def describe(self, **kwargs) -> xr.Dataset | None:
        ds = xr.Dataset()

        if (
            type(self._constituent)
            is sk2.constituent.brdf.lambertiansurface.LambertianSurface
        ):
            albedo = getattr(self._constituent, self._property_names[0])

            ds[self._constituent_name] = xr.DataArray(
                albedo,
                dims=[self._constituent._interp_var],
                coords={self._constituent._interp_var: self._constituent._x},
            )
            ds[self._constituent_name + "_1sigma_error"] = xr.DataArray(
                np.sqrt(np.diag(kwargs["covariance"])),
                dims=[self._constituent._interp_var],
                coords={self._constituent._interp_var: self._constituent._x},
            )

        else:
            start = 0
            for property_name in self._property_names:
                end = start + len(getattr(self._constituent, property_name))

                ds[self._constituent_name + "_" + property_name] = xr.DataArray(
                    getattr(self._constituent, property_name), dims=["altitude"]
                )

                ds[
                    self._constituent_name + "_" + property_name + "_prior"
                ] = xr.DataArray(self._prior[property_name].state, dims=["altitude"])

                if "covariance" in kwargs:
                    if self._log_space:
                        ds[
                            self._constituent_name
                            + "_"
                            + property_name
                            + "_1sigma_error"
                        ] = xr.DataArray(
                            np.sqrt(np.diag(kwargs["covariance"])[start:end])
                            * getattr(self._constituent, property_name),
                            dims=["altitude"],
                        )
                    else:
                        ds[
                            self._constituent_name
                            + "_"
                            + property_name
                            + "_1sigma_error"
                        ] = xr.DataArray(
                            np.sqrt(np.diag(kwargs["covariance"])[start:end]),
                            dims=["altitude"],
                        )

                if "averaging_kernel" in kwargs:
                    ds[
                        self._constituent_name
                        + "_"
                        + property_name
                        + "_averaging_kernel"
                    ] = xr.DataArray(
                        kwargs["averaging_kernel"][start:end, start:end],
                        dims=["altitude", "altitude_2"],
                    )

                start = end

        return ds
