from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import sasktran2 as sk2
import xarray as xr
from scipy import sparse
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


def _physical_variance(
    variance: np.ndarray,
    state_slice: slice,
    scale_factor: float,
    property_values,
    log_space: bool,
) -> np.ndarray:
    selected = np.asarray(variance)[state_slice]
    if log_space:
        return selected * np.asarray(property_values) ** 2
    return selected / scale_factor**2


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
        log_space: bool | Mapping[str, bool] = False,
        enabled=True,
        scale_factor: float | Mapping[str, float] = 1.0,
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
        log_space : bool or mapping, optional
            If true then all properties are represented logarithmically. A
            mapping selects the parameterization independently for each
            property, by default False.
        scale_factor : float or mapping, optional
            Constant multiplicative factor between constituent properties and
            retrieval state. A mapping selects the factor independently for
            each property. A linear state value of 1 corresponds to a
            constituent property value of 1 / scale_factor, by default 1.0.
        """
        if prior is None:
            prior = {}
        if max_value is None:
            max_value = {}
        if min_value is None:
            min_value = {}
        self._constituent = constituent
        self._property_names = list(property_names)
        self._constituent_name = constituent_name
        self._min_value = min_value
        self._max_value = max_value

        self._log_space_by_property = self._property_mapping(
            log_space, "log_space", bool
        )
        self._scale_factor_by_property = self._property_mapping(
            scale_factor, "scale_factor", float
        )
        if any(value <= 0 for value in self._scale_factor_by_property.values()):
            msg = "scale_factor must be positive"
            raise ValueError(msg)

        self._prior = prior

        start = 0
        for property_name in self._property_names:
            if property_name in self._prior:
                n = self._property_state_size(property_name)
                self._prior[property_name].init(self, slice(start, start + n))
                start += n
            else:
                self._prior[property_name] = BasePrior()

        super().__init__(enabled)

    def _property_mapping(self, value, name: str, converter) -> dict:
        if not isinstance(value, Mapping):
            return {
                property_name: converter(value)
                for property_name in self._property_names
            }

        unknown = set(value) - set(self._property_names)
        missing = set(self._property_names) - set(value)
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if unknown:
                details.append(f"unknown {sorted(unknown)}")
            msg = f"{name} property mapping is invalid: {', '.join(details)}"
            raise ValueError(msg)
        return {
            property_name: converter(value[property_name])
            for property_name in self._property_names
        }

    def _property_uses_log_space(self, property_name: str) -> bool:
        return self._log_space_by_property[property_name]

    def _property_scale_factor(self, property_name: str) -> float:
        return self._scale_factor_by_property[property_name]

    def _transform_property(self, property_name: str, value) -> np.ndarray:
        scaled = np.asarray(value) * self._property_scale_factor(property_name)
        if self._property_uses_log_space(property_name):
            return np.log(scaled)
        return scaled

    def _transform_bound(
        self, property_name: str, bound: float, size: int
    ) -> np.ndarray:
        if np.isinf(bound):
            return np.full(size, bound)
        return self._transform_property(property_name, np.full(size, bound))

    def _property_state_size(self, property_name: str) -> int:
        return int(np.size(np.atleast_1d(getattr(self._constituent, property_name))))

    def averaging_kernel_row_sum_groups(self) -> np.ndarray:
        """Keep averaging-kernel row sums within each physical property."""
        return np.concatenate(
            [
                np.full(self._property_state_size(property_name), index, dtype=int)
                for index, property_name in enumerate(self._property_names)
            ]
        )

    def state(self) -> np.array:
        return np.concatenate(
            [
                self._transform_property(
                    property_name,
                    np.asarray(getattr(self._constituent, property_name)).reshape(-1),
                )
                for property_name in self._property_names
            ]
        )

    def lower_bound(self) -> np.array:
        data = []
        for property_name in self._property_names:
            x = np.asarray(getattr(self._constituent, property_name))

            bound = self._min_value.get(property_name, -np.inf)

            data.append(self._transform_bound(property_name, bound, x.size))

        return np.hstack(data)

    def upper_bound(self) -> np.array:
        data = []
        for property_name in self._property_names:
            x = np.asarray(getattr(self._constituent, property_name))

            bound = self._max_value.get(property_name, np.inf)

            data.append(self._transform_bound(property_name, bound, x.size))

        return np.hstack(data)

    def inverse_apriori_covariance(self) -> np.ndarray:
        prior_mats = []

        for property_name in self._property_names:
            inv_S_a = self._prior[property_name].inverse_covariance

            if self._property_uses_log_space(property_name):
                prior_mats.append(inv_S_a)
            else:
                prior_mats.append(
                    inv_S_a
                    / np.outer(
                        self._prior[property_name].state,
                        self._prior[property_name].state,
                    )
                )

        if any(sparse.issparse(matrix) for matrix in prior_mats):
            return sparse.block_diag(prior_mats, format="csr")
        return block_diag(*prior_mats)

    def prior_precision_factor(self):
        """Return the prior residual operator in retrieval-state coordinates."""
        factors = []
        for property_name in self._property_names:
            factor = self._prior[property_name].precision_factor
            if not self._property_uses_log_space(property_name):
                inverse_state = sparse.diags(
                    1 / np.asarray(self._prior[property_name].state).reshape(-1),
                    format="csr",
                )
                factor = factor @ inverse_state
            factors.append(factor)

        if any(sparse.issparse(factor) for factor in factors):
            return sparse.block_diag(factors, format="csr")
        return block_diag(*factors)

    def apriori_state(self) -> np.array:
        return np.concatenate(
            [
                np.asarray(self._prior[property].state).reshape(-1)
                for property in self._property_names
            ]
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
            wf = radiance[f"wf_{self._constituent_name}_{property_name}"]
            radiance_dims = set(radiance["radiance"].dims)
            parameter_dims = tuple(dim for dim in wf.dims if dim not in radiance_dims)

            if self._property_uses_log_space(property_name):
                x = np.asarray(getattr(self._constituent, property_name))
                factor = xr.DataArray(x, dims=parameter_dims)
                wf = wf * factor
            else:
                wf = wf / self._property_scale_factor(property_name)

            if len(parameter_dims) == 1:
                wf = wf.rename({parameter_dims[0]: "x"})
            else:
                wf = wf.stack(x=parameter_dims)
                wf = wf.transpose("x", *radiance["radiance"].dims)
            wfs.append(wf)

        return xr.concat(wfs, dim="x")

    def supports_linearization_products(self) -> bool:
        return True

    def _linearization_parameter_name(
        self, property_name: str, tangent_template: xr.Dataset
    ) -> str:
        candidates = [f"{self._constituent_name}_{property_name}"]
        if property_name == "extinction_per_m":
            candidates.append(f"{self._constituent_name}_extinction")

        for candidate in candidates:
            if candidate in tangent_template:
                return candidate

        msg = (
            "SASKTRAN2 linearization does not contain a derivative for "
            f"{self._constituent_name}.{property_name}. Tried: "
            f"{', '.join(candidates)}"
        )
        raise KeyError(msg)

    def linearization_parameter_names(
        self, tangent_template: xr.Dataset
    ) -> tuple[str, ...]:
        return tuple(
            self._linearization_parameter_name(property_name, tangent_template)
            for property_name in self._property_names
        )

    def _state_to_native_tangent(self, property_name: str, x: np.ndarray) -> np.ndarray:
        """Apply the derivative of constituent property with respect to state."""
        if self._property_uses_log_space(property_name):
            current = np.asarray(getattr(self._constituent, property_name)).reshape(-1)
            return current * x
        return x / self._property_scale_factor(property_name)

    def _native_gradient_to_state(
        self, property_name: str, gradient: np.ndarray
    ) -> np.ndarray:
        """Apply the adjoint of ``_state_to_native_tangent``."""
        if self._property_uses_log_space(property_name):
            current = np.asarray(getattr(self._constituent, property_name)).reshape(-1)
            return gradient * current
        return gradient / self._property_scale_factor(property_name)

    def add_to_linearization_tangent(
        self,
        tangent: dict[str, xr.DataArray],
        x: np.ndarray,
        tangent_template: xr.Dataset,
    ) -> None:
        start = 0
        for property_name in self._property_names:
            current = np.asarray(getattr(self._constituent, property_name))
            end = start + current.size
            parameter_name = self._linearization_parameter_name(
                property_name, tangent_template
            )
            template = tangent_template[parameter_name]
            values = self._state_to_native_tangent(property_name, x[start:end])
            direction = xr.DataArray(
                values.reshape(template.shape),
                dims=template.dims,
                coords=template.coords,
            )
            if parameter_name in tangent:
                tangent[parameter_name] = tangent[parameter_name] + direction
            else:
                tangent[parameter_name] = direction
            start = end

    def linearization_gradient(
        self,
        gradient: xr.Dataset,
        tangent_template: xr.Dataset,
    ) -> np.ndarray:
        parts = []
        for property_name in self._property_names:
            parameter_name = self._linearization_parameter_name(
                property_name, tangent_template
            )
            if parameter_name in gradient:
                native_gradient = np.asarray(gradient[parameter_name]).reshape(-1)
            else:
                native_gradient = np.zeros(
                    np.size(np.atleast_1d(getattr(self._constituent, property_name)))
                )
            parts.append(self._native_gradient_to_state(property_name, native_gradient))

        return np.concatenate(parts)

    def update_state(self, x: np.array):
        start = 0
        for property_name in self._property_names:
            current = np.asarray(getattr(self._constituent, property_name))
            property_length = current.size
            scale_factor = self._property_scale_factor(property_name)
            if self._property_uses_log_space(property_name):
                sv = np.exp(x[start : start + property_length]) / scale_factor
                if np.sum(np.isnan(sv)) > 0:
                    sv[np.isnan(sv)] = self._max_value[property_name]
            else:
                sv = x[start : start + property_length] / scale_factor
            if property_name in self._min_value:
                sv[sv < self._min_value[property_name]] = self._min_value[property_name]
            if property_name in self._max_value:
                sv[sv > self._max_value[property_name]] = self._max_value[property_name]

            self._constituent.__setattr__(property_name, sv.reshape(current.shape))

            start += property_length

    def modify_input_radiance(self, radiance: xr.Dataset):
        return radiance

    @property
    def volume_spatial_mode(self):
        """Preserve the wrapped constituent's spatial input convention."""
        return self._constituent.volume_spatial_mode

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
            property_name = self._property_names[0]

            ds[self._constituent_name] = xr.DataArray(
                albedo,
                dims=[self._constituent._interp_var],
                coords={self._constituent._interp_var: self._constituent._x},
            )
            if "covariance" in kwargs:
                ds[self._constituent_name + "_1sigma_error"] = xr.DataArray(
                    _physical_1sigma(
                        kwargs["covariance"],
                        slice(None),
                        self._property_scale_factor(property_name),
                        albedo,
                        self._property_uses_log_space(property_name),
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

                scale_factor = self._property_scale_factor(property_name)
                log_space = self._property_uses_log_space(property_name)
                if log_space:
                    prior_values = (
                        np.exp(self._prior[property_name].state) / scale_factor
                    )
                else:
                    prior_values = self._prior[property_name].state / scale_factor

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
                                scale_factor,
                                getattr(self._constituent, property_name),
                                log_space,
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
                                scale_factor,
                                getattr(self._constituent, property_name),
                                log_space,
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
