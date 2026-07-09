from __future__ import annotations

import numpy as np
import sasktran2 as sk2
import xarray as xr
from scipy.linalg import block_diag

from skretrieval.retrieval.prior import BasePrior

from . import StateVectorElement


def _as_scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _altitude_grids_match(altitude_grid: np.ndarray, model_altitude_grid) -> bool:
    if model_altitude_grid is None:
        return False

    model_altitude_grid = np.asarray(model_altitude_grid)
    return altitude_grid.shape == model_altitude_grid.shape and np.allclose(
        altitude_grid, model_altitude_grid
    )


def _physical_1sigma(
    covariance: np.ndarray,
    state_slice: slice,
    scale_factor: float,
    property_values,
    log_space: bool,
) -> np.ndarray:
    sigma = np.sqrt(np.diag(covariance)[state_slice])
    if log_space:
        return sigma * np.asarray(property_values)
    return sigma / scale_factor


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
        scale_factor: float = 1.0,
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
        scale_factor : float, optional
            Constant multiplicative factor between constituent properties and retrieval state.
            A state value of 1 corresponds to a constituent property value of 1 / scale_factor,
            by default 1.0
        """
        if prior is None:
            prior = {}
        if max_value is None:
            max_value = {}
        if min_value is None:
            min_value = {}
        if scale_factor <= 0:
            msg = "scale_factor must be positive"
            raise ValueError(msg)

        self._log_space = log_space
        self._scale_factor = float(scale_factor)
        self._constituent = constituent
        self._property_names = property_names
        self._constituent_name = constituent_name
        self._min_value = min_value
        self._max_value = max_value

        self._prior = prior

        start = 0
        for property_name in self._property_names:
            if property_name in self._prior:
                n = len(np.atleast_1d(getattr(self._constituent, property_name)))
                self._prior[property_name].init(self, slice(start, start + n))
                start += n
            else:
                self._prior[property_name] = BasePrior()

        super().__init__(enabled)

    def state(self) -> np.array:
        data = []

        for property_name in self._property_names:
            data.append(getattr(self._constituent, property_name) * self._scale_factor)

        if self._log_space:
            return np.log(np.hstack(data))
        return np.hstack(data)

    def lower_bound(self) -> np.array:
        data = []
        for property_name in self._property_names:
            x = getattr(self._constituent, property_name)

            bound = self._min_value.get(property_name, -np.inf)

            data.append(np.ones(len(x)) * bound * self._scale_factor)

        if self._log_space:
            return np.log(np.hstack(data))
        return np.hstack(data)

    def upper_bound(self) -> np.array:
        data = []
        for property_name in self._property_names:
            x = getattr(self._constituent, property_name)

            bound = self._max_value.get(property_name, np.inf)

            data.append(np.ones(len(x)) * bound * self._scale_factor)

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
            else:
                wfs[-1].values = wfs[-1].values / self._scale_factor

        return xr.concat(wfs, dim="x")

    def update_state(self, x: np.array):
        start = 0
        for property_name in self._property_names:
            current = getattr(self._constituent, property_name)
            property_length = len(np.atleast_1d(current))
            if self._log_space:
                sv = np.exp(x[start : start + property_length]) / self._scale_factor
                if np.sum(np.isnan(sv)) > 0:
                    sv[np.isnan(sv)] = self._max_value[property_name]
            else:
                sv = x[start : start + property_length] / self._scale_factor
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

    def _constituent_altitude_grid(self, property_length: int) -> np.ndarray | None:
        for attr_name in ("altitudes_m", "_altitudes_m", "_altitude_grid_m"):
            if hasattr(self._constituent, attr_name):
                altitude_grid = np.asarray(getattr(self._constituent, attr_name))
                if altitude_grid.shape == (property_length,):
                    return altitude_grid

        nested_constituent = getattr(self._constituent, "_constituent", None)
        if nested_constituent is not None and hasattr(
            nested_constituent, "altitudes_m"
        ):
            altitude_grid = np.asarray(nested_constituent.altitudes_m)
            if altitude_grid.shape == (property_length,):
                return altitude_grid

        return None

    def _profile_dims_and_coords(
        self, property_length: int, model_altitude_grid
    ) -> tuple[str, str, dict[str, np.ndarray]]:
        altitude_grid = self._constituent_altitude_grid(property_length)

        if altitude_grid is None and model_altitude_grid is not None:
            model_altitude_grid = np.asarray(model_altitude_grid)
            if model_altitude_grid.shape == (property_length,):
                altitude_grid = model_altitude_grid

        if altitude_grid is None or _altitude_grids_match(
            altitude_grid, model_altitude_grid
        ):
            altitude_dim = "altitude"
            altitude_dim_2 = "altitude_2"
        else:
            altitude_dim = f"{self._constituent_name}_altitude"
            altitude_dim_2 = f"{altitude_dim}_2"

        coords = {}
        if altitude_grid is not None:
            coords[altitude_dim] = altitude_grid
            coords[altitude_dim_2] = altitude_grid

        return altitude_dim, altitude_dim_2, coords

    def describe(self, **kwargs) -> xr.Dataset | None:
        ds = xr.Dataset()
        model_altitude_grid = kwargs.get("model_altitude_grid")

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
                _physical_1sigma(
                    kwargs["covariance"],
                    slice(None),
                    self._scale_factor,
                    albedo,
                    self._log_space,
                ),
                dims=[self._constituent._interp_var],
                coords={self._constituent._interp_var: self._constituent._x},
            )

        else:
            start = 0
            for property_name in self._property_names:
                end = start + len(
                    np.atleast_1d(getattr(self._constituent, property_name))
                )

                if self._log_space:
                    prior_values = (
                        np.exp(self._prior[property_name].state) / self._scale_factor
                    )
                else:
                    prior_values = self._prior[property_name].state / self._scale_factor

                if end - start == 1:  # scalar property
                    ds[self._constituent_name + "_" + property_name] = xr.DataArray(
                        _as_scalar(getattr(self._constituent, property_name))
                    )

                    ds[self._constituent_name + "_" + property_name + "_prior"] = (
                        _as_scalar(prior_values)
                    )

                    if "covariance" in kwargs:
                        ds[
                            self._constituent_name
                            + "_"
                            + property_name
                            + "_1sigma_error"
                        ] = _as_scalar(
                            _physical_1sigma(
                                kwargs["covariance"],
                                slice(start, end),
                                self._scale_factor,
                                getattr(self._constituent, property_name),
                                self._log_space,
                            )
                        )

                    if "averaging_kernel" in kwargs:
                        ds[
                            self._constituent_name
                            + "_"
                            + property_name
                            + "_averaging_kernel"
                        ] = _as_scalar(kwargs["averaging_kernel"][start:end, start:end])
                else:
                    altitude_dim, altitude_dim_2, coords = (
                        self._profile_dims_and_coords(end - start, model_altitude_grid)
                    )
                    profile_coords = (
                        {altitude_dim: coords[altitude_dim]}
                        if altitude_dim in coords
                        else None
                    )
                    ak_coords = (
                        {
                            altitude_dim: coords[altitude_dim],
                            altitude_dim_2: coords[altitude_dim_2],
                        }
                        if altitude_dim in coords and altitude_dim_2 in coords
                        else None
                    )

                    ds[self._constituent_name + "_" + property_name] = xr.DataArray(
                        getattr(self._constituent, property_name),
                        dims=[altitude_dim],
                        coords=profile_coords,
                    )

                    ds[self._constituent_name + "_" + property_name + "_prior"] = (
                        xr.DataArray(
                            prior_values,
                            dims=[altitude_dim],
                            coords=profile_coords,
                        )
                    )

                    if "covariance" in kwargs:
                        ds[
                            self._constituent_name
                            + "_"
                            + property_name
                            + "_1sigma_error"
                        ] = xr.DataArray(
                            _physical_1sigma(
                                kwargs["covariance"],
                                slice(start, end),
                                self._scale_factor,
                                getattr(self._constituent, property_name),
                                self._log_space,
                            ),
                            dims=[altitude_dim],
                            coords=profile_coords,
                        )

                    if "averaging_kernel" in kwargs:
                        ds[
                            self._constituent_name
                            + "_"
                            + property_name
                            + "_averaging_kernel"
                        ] = xr.DataArray(
                            kwargs["averaging_kernel"][start:end, start:end],
                            dims=[altitude_dim, altitude_dim_2],
                            coords=ak_coords,
                        )

                start = end

        return ds
