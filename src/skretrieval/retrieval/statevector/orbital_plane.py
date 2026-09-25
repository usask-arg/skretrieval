from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import sasktran2 as sk2
import xarray as xr
from scipy import sparse
from scipy.linalg import block_diag

from skretrieval.retrieval.statevector import StateVector
from skretrieval.retrieval.statevector.constituent import (
    StateVectorElementConstituent,
    _physical_1sigma,
    _physical_variance,
)


class OrbitalPlaneStateVectorElement(StateVectorElementConstituent):
    """A constituent state element labeled on an orbital-plane grid.

    Priors supplied to this class are defined directly in retrieval-state
    coordinates. In particular, a log-space prior state contains logarithms
    and its precision acts on logarithmic perturbations.
    """

    def __init__(
        self,
        constituent: sk2.constituent.base.Constituent,
        constituent_name: str,
        property_names: Iterable[str],
        *,
        geometry: sk2.OrbitalPlaneGeometry,
        property_dimensions: dict[str, tuple[str, ...]] | None = None,
        retrieval_masks: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        self._geometry = geometry
        self._property_dimensions = (
            {} if property_dimensions is None else dict(property_dimensions)
        )
        self._retrieval_masks = {} if retrieval_masks is None else dict(retrieval_masks)
        for property_name, retrieval_mask in self._retrieval_masks.items():
            value = np.asarray(getattr(constituent, property_name))
            mask = np.asarray(retrieval_mask, dtype=bool)
            if mask.shape != value.shape:
                msg = (
                    f"Retrieval mask for {property_name} has shape {mask.shape}; "
                    f"expected {value.shape}"
                )
                raise ValueError(msg)
            if not np.any(mask):
                msg = f"Retrieval mask for {property_name} selects no values"
                raise ValueError(msg)
            self._retrieval_masks[property_name] = mask
        super().__init__(
            constituent,
            constituent_name,
            list(property_names),
            **kwargs,
        )

    def _mask(self, property_name: str) -> np.ndarray:
        value = np.asarray(getattr(self._constituent, property_name))
        return self._retrieval_masks.get(
            property_name, np.ones(value.shape, dtype=bool)
        )

    def _property_state_size(self, property_name: str) -> int:
        return int(np.count_nonzero(self._mask(property_name)))

    def averaging_kernel_row_sum_groups(self) -> np.ndarray:
        """Separate physical properties and any non-spatial coordinates."""
        groups = []
        next_group = 0
        for property_name in self._property_names:
            value = np.asarray(getattr(self._constituent, property_name))
            labels = np.zeros(value.shape, dtype=int)
            dimensions = self._property_dimensions.get(property_name)
            if dimensions is not None:
                non_spatial_axes = [
                    axis
                    for axis, dimension in enumerate(dimensions)
                    if dimension not in {"orbital_position", "altitude"}
                ]
                if non_spatial_axes:
                    non_spatial_shape = tuple(
                        value.shape[axis] for axis in non_spatial_axes
                    )
                    index_grids = np.indices(non_spatial_shape)
                    local_labels = np.ravel_multi_index(
                        tuple(index_grids),
                        non_spatial_shape,
                    )
                    reshape = [1] * value.ndim
                    for grid_axis, value_axis in enumerate(non_spatial_axes):
                        reshape[value_axis] = non_spatial_shape[grid_axis]
                    labels = np.broadcast_to(
                        local_labels.reshape(reshape),
                        value.shape,
                    )
            selected = labels[self._mask(property_name)]
            groups.append(selected + next_group)
            next_group += int(np.max(selected)) + 1
        return np.concatenate(groups)

    def averaging_kernel_resolution_coordinates(self) -> dict[str, np.ndarray]:
        """Return altitude and ground-track distance for each retrieval state."""
        altitude_m = np.asarray(
            self._geometry.grid_dataset.altitude,
            dtype=float,
        )
        horizontal_m = float(self._geometry.earth_radius_m) * np.asarray(
            self._geometry.grid_dataset.along_track_angle_rad,
            dtype=float,
        )
        vertical_parts = []
        horizontal_parts = []
        for property_name in self._property_names:
            value = np.asarray(getattr(self._constituent, property_name))
            mask = self._mask(property_name)
            vertical = np.full(value.shape, np.nan)
            horizontal = np.full(value.shape, np.nan)
            if value.shape == self._geometry.shape:
                vertical = np.broadcast_to(
                    altitude_m[np.newaxis, :],
                    value.shape,
                )
            if value.ndim >= 1 and value.shape[0] == self._geometry.shape[0]:
                reshape = (horizontal_m.size,) + (1,) * (value.ndim - 1)
                horizontal = np.broadcast_to(horizontal_m.reshape(reshape), value.shape)
            vertical_parts.append(vertical[mask])
            horizontal_parts.append(horizontal[mask])
        return {
            "vertical_resolution_m": np.concatenate(vertical_parts),
            "horizontal_resolution_m": np.concatenate(horizontal_parts),
        }

    def state(self) -> np.ndarray:
        parts = []
        for property_name in self._property_names:
            value = np.asarray(getattr(self._constituent, property_name))
            parts.append(
                self._transform_property(
                    property_name, value[self._mask(property_name)]
                )
            )
        return np.concatenate(parts)

    def _bound(self, values: dict, default: float) -> np.ndarray:
        parts = []
        for property_name in self._property_names:
            size = np.count_nonzero(self._mask(property_name))
            parts.append(
                self._transform_bound(
                    property_name, values.get(property_name, default), size
                )
            )
        return np.concatenate(parts)

    def lower_bound(self) -> np.ndarray:
        return self._bound(self._min_value, -np.inf)

    def upper_bound(self) -> np.ndarray:
        return self._bound(self._max_value, np.inf)

    def add_to_linearization_tangent(
        self,
        tangent: dict[str, xr.DataArray],
        x: np.ndarray,
        tangent_template: xr.Dataset,
    ) -> None:
        start = 0
        for property_name in self._property_names:
            current = np.asarray(getattr(self._constituent, property_name))
            mask = self._mask(property_name)
            end = start + np.count_nonzero(mask)
            parameter_name = self._linearization_parameter_name(
                property_name, tangent_template
            )
            template = tangent_template[parameter_name]
            native = np.zeros(current.shape, dtype=float)
            if self._property_uses_log_space(property_name):
                native[mask] = current[mask] * x[start:end]
            else:
                native[mask] = x[start:end] / self._property_scale_factor(property_name)
            direction = xr.DataArray(
                native.reshape(template.shape),
                dims=template.dims,
                coords=template.coords,
            )
            tangent[parameter_name] = tangent.get(parameter_name, 0) + direction
            start = end

    def linearization_gradient(
        self,
        gradient: xr.Dataset,
        tangent_template: xr.Dataset,
    ) -> np.ndarray:
        parts = []
        for property_name in self._property_names:
            current = np.asarray(getattr(self._constituent, property_name))
            mask = self._mask(property_name)
            parameter_name = self._linearization_parameter_name(
                property_name, tangent_template
            )
            if parameter_name in gradient:
                native = np.asarray(gradient[parameter_name]).reshape(current.shape)
            else:
                native = np.zeros(current.shape)
            selected = native[mask]
            if self._property_uses_log_space(property_name):
                selected = selected * current[mask]
            else:
                selected = selected / self._property_scale_factor(property_name)
            parts.append(selected)
        return np.concatenate(parts)

    def update_state(self, x: np.ndarray) -> None:
        start = 0
        for property_name in self._property_names:
            current = np.asarray(
                getattr(self._constituent, property_name), dtype=float
            ).copy()
            mask = self._mask(property_name)
            end = start + np.count_nonzero(mask)
            selected = x[start:end]
            scale_factor = self._property_scale_factor(property_name)
            if self._property_uses_log_space(property_name):
                selected = np.exp(selected) / scale_factor
            else:
                selected = selected / scale_factor
            selected = np.maximum(selected, self._min_value.get(property_name, -np.inf))
            selected = np.minimum(selected, self._max_value.get(property_name, np.inf))
            current[mask] = selected
            setattr(self._constituent, property_name, current)
            start = end

    def inverse_apriori_covariance(self):
        """Return prior precision without legacy relative-profile rescaling."""
        matrices = [
            self._prior[property_name].inverse_covariance
            for property_name in self._property_names
        ]
        if any(sparse.issparse(matrix) for matrix in matrices):
            return sparse.block_diag(matrices, format="csr")
        return block_diag(*matrices)

    def prior_precision_factor(self):
        """Return sparse prior residuals in native orbital-state coordinates."""
        factors = [
            self._prior[property_name].precision_factor
            for property_name in self._property_names
        ]
        if any(sparse.issparse(factor) for factor in factors):
            return sparse.block_diag(factors, format="csr")
        return block_diag(*factors)

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        """Map a materialized Jacobian to the active orbital-plane state."""
        if "extinction_per_m" in self._property_names:
            radiance = radiance.rename(
                {
                    f"wf_{self._constituent_name}_extinction": f"wf_{self._constituent_name}_extinction_per_m"
                }
            )

        radiance_dims = tuple(radiance["radiance"].dims)
        wfs = []
        for property_name in self._property_names:
            wf = radiance[f"wf_{self._constituent_name}_{property_name}"]
            parameter_dims = tuple(dim for dim in wf.dims if dim not in radiance_dims)

            if self._property_uses_log_space(property_name):
                current = np.asarray(getattr(self._constituent, property_name))
                wf = wf * xr.DataArray(current, dims=parameter_dims)
            else:
                wf = wf / self._property_scale_factor(property_name)

            if len(parameter_dims) == 1:
                wf = wf.rename({parameter_dims[0]: "x"})
            else:
                wf = wf.stack(x=parameter_dims)
            active = np.flatnonzero(self._mask(property_name).reshape(-1))
            wfs.append(wf.isel(x=active).transpose("x", *radiance_dims))

        return xr.concat(wfs, dim="x")

    def _dims_and_coords(
        self,
        property_name: str,
        shape: tuple[int, ...],
        mask: np.ndarray,
    ) -> tuple[tuple[str, ...], dict]:
        if property_name in self._property_dimensions:
            dims = tuple(self._property_dimensions[property_name])
        elif shape == self._geometry.shape:
            dims = (
                "orbital_position",
                ("altitude" if np.all(mask) else f"{self._constituent_name}_altitude"),
            )
        elif shape == (self._geometry.shape[0],):
            dims = ("orbital_position",)
        elif shape == () or shape == (1,):
            dims = ()
        else:
            msg = (
                f"No orbital-plane dimensions were supplied for "
                f"{self._constituent_name}.{property_name} with shape {shape}"
            )
            raise ValueError(msg)

        if len(dims) != len(shape):
            msg = (
                f"Dimensions {dims} do not match "
                f"{self._constituent_name}.{property_name} shape {shape}"
            )
            raise ValueError(msg)

        available = self._geometry.grid_dataset.coords
        coords = {dim: available[dim] for dim in dims if dim in available}
        if shape == self._geometry.shape and not np.all(mask):
            altitude_mask = np.all(mask, axis=0)
            expected = np.broadcast_to(altitude_mask, shape)
            if not np.array_equal(mask, expected):
                msg = (
                    "Orbital-plane output currently requires a full grid or a "
                    "common altitude mask at every orbital position"
                )
                raise ValueError(msg)
            coords[dims[1]] = np.asarray(available["altitude"])[altitude_mask]
        return dims, coords

    @staticmethod
    def _output_name(constituent_name: str, property_name: str, constituent) -> str:
        if isinstance(constituent, sk2.constituent.LambertianSurface2D):
            return constituent_name
        return f"{constituent_name}_{property_name}"

    def describe(self, **kwargs) -> xr.Dataset:
        dataset = xr.Dataset()
        start = 0
        for property_name in self._property_names:
            full_physical = np.asarray(getattr(self._constituent, property_name))
            mask = self._mask(property_name)
            physical = full_physical[mask]
            if full_physical.shape == self._geometry.shape:
                physical = physical.reshape(
                    self._geometry.shape[0],
                    np.count_nonzero(np.all(mask, axis=0)),
                )
            elif np.all(mask):
                physical = physical.reshape(full_physical.shape)
            size = np.count_nonzero(mask)
            state_slice = slice(start, start + size)
            dims, coords = self._dims_and_coords(
                property_name, full_physical.shape, mask
            )
            output_name = self._output_name(
                self._constituent_name, property_name, self._constituent
            )

            prior = np.asarray(self._prior[property_name].state).reshape(physical.shape)
            scale_factor = self._property_scale_factor(property_name)
            log_space = self._property_uses_log_space(property_name)
            prior = np.exp(prior) / scale_factor if log_space else prior / scale_factor

            dataset[output_name] = xr.DataArray(physical, dims=dims, coords=coords)
            dataset[f"{output_name}_prior"] = xr.DataArray(
                prior, dims=dims, coords=coords
            )

            if "covariance" in kwargs:
                sigma = _physical_1sigma(
                    kwargs["covariance"],
                    state_slice,
                    scale_factor,
                    physical.reshape(-1),
                    log_space,
                ).reshape(physical.shape)
                dataset[f"{output_name}_1sigma_error"] = xr.DataArray(
                    sigma, dims=dims, coords=coords
                )

            if "posterior_variance" in kwargs:
                posterior_variance = _physical_variance(
                    kwargs["posterior_variance"],
                    state_slice,
                    scale_factor,
                    physical.reshape(-1),
                    log_space,
                ).reshape(physical.shape)
                posterior_1sigma = np.sqrt(
                    np.where(posterior_variance >= 0, posterior_variance, np.nan)
                )
                dataset[f"{output_name}_posterior_1sigma"] = xr.DataArray(
                    posterior_1sigma,
                    dims=dims,
                    coords=coords,
                )
                if "posterior_variance_standard_error" in kwargs:
                    variance_standard_error = _physical_variance(
                        kwargs["posterior_variance_standard_error"],
                        state_slice,
                        scale_factor,
                        physical.reshape(-1),
                        log_space,
                    ).reshape(physical.shape)
                    relative_error = variance_standard_error / np.maximum(
                        np.abs(posterior_variance),
                        np.finfo(float).tiny,
                    )
                    dataset[f"{output_name}_posterior_variance_relative_mc_error"] = (
                        xr.DataArray(
                            relative_error,
                            dims=dims,
                            coords=coords,
                        )
                    )

            if (
                "measurement_information" in kwargs
                and "measurement_information_standard_error" in kwargs
            ):
                information = np.asarray(kwargs["measurement_information"])[
                    state_slice
                ].reshape(physical.shape)
                information_standard_error = np.asarray(
                    kwargs["measurement_information_standard_error"]
                )[state_slice].reshape(physical.shape)
                dataset[f"{output_name}_measurement_information_relative_mc_error"] = (
                    xr.DataArray(
                        information_standard_error
                        / np.maximum(np.abs(information), np.finfo(float).tiny),
                        dims=dims,
                        coords=coords,
                    )
                )

            if "approximate_averaging_kernel_row_sum" in kwargs:
                row_sum = np.asarray(kwargs["approximate_averaging_kernel_row_sum"])[
                    state_slice
                ].reshape(physical.shape)
                dataset[f"{output_name}_approximate_averaging_kernel_row_sum"] = (
                    xr.DataArray(
                        row_sum,
                        dims=dims,
                        coords=coords,
                    )
                )

            if "averaging_kernel_row_sum" in kwargs:
                row_sum = np.asarray(kwargs["averaging_kernel_row_sum"])[
                    state_slice
                ].reshape(physical.shape)
                dataset[f"{output_name}_averaging_kernel_row_sum"] = xr.DataArray(
                    row_sum,
                    dims=dims,
                    coords=coords,
                )

            for resolution_name in (
                "approximate_averaging_kernel_vertical_resolution_m",
                "approximate_averaging_kernel_horizontal_resolution_m",
                "averaging_kernel_vertical_resolution_m",
                "averaging_kernel_horizontal_resolution_m",
            ):
                if resolution_name not in kwargs:
                    continue
                resolution_m = np.asarray(kwargs[resolution_name])[state_slice]
                if not np.any(np.isfinite(resolution_m)):
                    continue
                output_resolution_name = resolution_name.removesuffix("_m") + "_km"
                axis = (
                    "vertical"
                    if "vertical_resolution" in resolution_name
                    else "horizontal"
                )
                qualifier = (
                    "Approximate " if resolution_name.startswith("approximate_") else ""
                )
                if axis == "vertical" and full_physical.shape == self._geometry.shape:
                    local_spacing_m = np.broadcast_to(
                        np.abs(
                            np.gradient(
                                np.asarray(
                                    self._geometry.grid_dataset.altitude,
                                    dtype=float,
                                )
                            )
                        )[np.newaxis, :],
                        full_physical.shape,
                    )[mask]
                    resolution_m = np.maximum(resolution_m, local_spacing_m)
                elif (
                    axis == "horizontal"
                    and full_physical.ndim >= 1
                    and full_physical.shape[0] == self._geometry.shape[0]
                ):
                    horizontal_m = float(self._geometry.earth_radius_m) * np.asarray(
                        self._geometry.grid_dataset.along_track_angle_rad,
                        dtype=float,
                    )
                    local_spacing_m = np.abs(np.gradient(horizontal_m))
                    reshape = (local_spacing_m.size,) + (1,) * (full_physical.ndim - 1)
                    local_spacing_m = np.broadcast_to(
                        local_spacing_m.reshape(reshape),
                        full_physical.shape,
                    )[mask]
                    resolution_m = np.maximum(resolution_m, local_spacing_m)
                dataset[f"{output_name}_{output_resolution_name}"] = xr.DataArray(
                    (resolution_m / 1.0e3).reshape(physical.shape),
                    dims=dims,
                    coords=coords,
                    attrs={
                        "units": "km",
                        "long_name": (
                            f"{qualifier}Gaussian-equivalent averaging-kernel "
                            f"{axis} FWHM"
                        ),
                        "resolution_metric": (
                            "signed-moment Gaussian-equivalent FWHM, "
                            "lower-bounded by local state-grid spacing"
                        ),
                        "resolution_coordinate": (
                            "altitude"
                            if axis == "vertical"
                            else "orbital-plane ground-track arc distance"
                        ),
                    },
                )

            if "averaging_kernel" in kwargs:
                second_dims = tuple(f"{dim}_2" for dim in dims)
                second_coords = {
                    f"{dim}_2": np.asarray(coords[dim]) for dim in dims if dim in coords
                }
                kernel = kwargs["averaging_kernel"][state_slice, state_slice]
                dataset[f"{output_name}_averaging_kernel"] = xr.DataArray(
                    kernel.reshape((*physical.shape, *physical.shape)),
                    dims=(*dims, *second_dims),
                    coords={**coords, **second_coords},
                )

            start += size

        return dataset


class OrbitalPlaneStateVector(StateVector):
    """State vector whose constituents share one orbital-plane model grid."""

    def __init__(
        self,
        geometry: sk2.OrbitalPlaneGeometry,
        **elements: OrbitalPlaneStateVectorElement,
    ) -> None:
        self._geometry = geometry
        self._sv_eles = elements
        super().__init__(elements.values())

    @property
    def geometry(self) -> sk2.OrbitalPlaneGeometry:
        return self._geometry

    @property
    def sv(self) -> dict[str, OrbitalPlaneStateVectorElement]:
        return self._sv_eles

    @property
    def altitude_grid(self) -> np.ndarray:
        return self._geometry.altitudes()

    def add_to_atmosphere(self, atmo: sk2.Atmosphere) -> None:
        if atmo.model_geometry is not self._geometry:
            msg = "Orbital-plane state vector and atmosphere must share one geometry"
            raise ValueError(msg)
        for name, element in self._sv_eles.items():
            atmo[name] = element

    def post_process_sk2_radiances(self, radiance: xr.Dataset) -> xr.Dataset:
        """Map an explicitly materialized SASKTRAN2 Jacobian when requested."""
        weighting_functions = [
            element.propagate_wf(radiance)
            for element in self.state_elements
            if element.enabled
        ]
        for element in self.state_elements:
            if element.enabled:
                radiance = element.modify_input_radiance(radiance)

        if weighting_functions:
            old_names = [name for name in radiance if name.startswith("wf")]
            radiance = radiance.drop_vars(old_names)
            radiance["wf"] = xr.concat(weighting_functions, dim="x")
        return radiance

    def describe(self, minimizer_output: dict) -> xr.Dataset:
        state = super().describe(minimizer_output, model_geometry=self._geometry)
        return xr.merge([self._geometry.grid_dataset, state])
