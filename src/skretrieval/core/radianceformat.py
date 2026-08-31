from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

from skretrieval.geodetic import target_lat_lon_alt


@dataclass(frozen=True)
class VJPContribution:
    """A deferred cotangent and the shared source that will consume it.

    Keeping local adjoint transforms separate from the source VJP allows
    cotangents from multiple measurement vectors to be added first.
    """

    source: object
    cotangent: object
    pullback: Callable[[object], np.ndarray | list[VJPContribution]]


def evaluate_vjp_contributions(
    contributions: list[VJPContribution],
) -> np.ndarray:
    """Fuse cotangents at each shared node before continuing its pullback."""
    pending = contributions
    result = None
    while pending:
        grouped: dict[int, VJPContribution] = {}
        for contribution in pending:
            # Sources are grouped by identity because radiance containers do not
            # define value equality and represent one particular linearization.
            key = id(contribution.source)
            if key in grouped:
                current = grouped[key]
                grouped[key] = VJPContribution(
                    current.source,
                    current.cotangent + contribution.cotangent,
                    current.pullback,
                )
            else:
                grouped[key] = contribution

        # Deeper nodes are processed first so every branch reaches a shared
        # parent before that parent's pullback is evaluated.
        depth = max(getattr(item.source, "vjp_depth", 0) for item in grouped.values())
        pending = [
            item
            for item in grouped.values()
            if getattr(item.source, "vjp_depth", 0) < depth
        ]
        current = [
            item
            for item in grouped.values()
            if getattr(item.source, "vjp_depth", 0) == depth
        ]
        for contribution in current:
            value = contribution.pullback(contribution.cotangent)
            if isinstance(value, list):
                pending.extend(value)
            else:
                value = np.asarray(value)
                result = value if result is None else result + value

    if result is None:
        msg = "At least one VJP contribution is required"
        raise ValueError(msg)
    return result


def _hashable_indexer(value):
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, xr.DataArray):
        value = value.to_numpy()
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, list | tuple):
        return tuple(_hashable_indexer(item) for item in value)
    return value


def select_dataset(
    ds: xr.Dataset,
    indexers: dict,
    *,
    method: str | None = None,
) -> xr.Dataset:
    """Select labels, including slices on repeated one-dimensional indexes.

    Xarray cannot apply a label slice to a non-monotonic index. Orbital limb
    data naturally repeats the tangent-altitude sequence for every image, so
    convert such slices to positional masks while retaining ordinary xarray
    selection for monotonic and scalar indexes.
    """
    label_indexers = {}
    positional_masks: dict[str, np.ndarray] = {}
    positional_steps: dict[str, int] = {}
    for name, indexer in indexers.items():
        coordinate = ds.coords.get(name)
        xindex = ds.xindexes.get(name)
        if coordinate is None or coordinate.ndim != 1:
            label_indexers[name] = indexer
            continue

        if isinstance(indexer, slice):
            if xindex is not None:
                pandas_index = xindex.to_pandas_index()
                if (
                    pandas_index.is_monotonic_increasing
                    or pandas_index.is_monotonic_decreasing
                ):
                    label_indexers[name] = indexer
                    continue

            values = np.asarray(coordinate)
            if indexer.start is None:
                start_mask = np.ones(values.shape, dtype=bool)
            elif indexer.stop is not None and indexer.start > indexer.stop:
                start_mask = values <= indexer.start
            else:
                start_mask = values >= indexer.start
            if indexer.stop is None:
                stop_mask = np.ones(values.shape, dtype=bool)
            elif indexer.start is not None and indexer.start > indexer.stop:
                stop_mask = values >= indexer.stop
            else:
                stop_mask = values <= indexer.stop
            mask = start_mask & stop_mask
            if indexer.step is not None:
                positional_steps[coordinate.dims[0]] = indexer.step
        elif xindex is None:
            values = np.asarray(coordinate)
            requested = np.asarray(indexer)
            if method == "nearest" and requested.ndim == 0:
                mask = np.zeros(values.shape, dtype=bool)
                mask[int(np.argmin(np.abs(values - requested.item())))] = True
            elif requested.ndim == 0:
                mask = values == requested.item()
            else:
                mask = np.isin(values, requested)
        else:
            label_indexers[name] = indexer
            continue
        dimension = coordinate.dims[0]
        if dimension in positional_masks:
            positional_masks[dimension] &= mask
        else:
            positional_masks[dimension] = mask

    positional_indexers = {}
    for dimension, mask in positional_masks.items():
        positions = np.flatnonzero(mask)
        if dimension in positional_steps:
            positions = positions[:: positional_steps[dimension]]
        positional_indexers[dimension] = positions

    selected = ds.isel(positional_indexers)
    if label_indexers:
        selected = selected.sel(**label_indexers, method=method)
    return selected


@dataclass(frozen=True)
class _SelectionPlan:
    """Positional selection and scatter information for a fixed radiance grid."""

    indexers: dict
    flat_indices: np.ndarray
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def _compile_selection_plan(
    ds: xr.Dataset, indexers: dict, method: str | None
) -> _SelectionPlan:
    """Resolve label-based selection to reusable integer positions."""
    augmented = ds.copy(deep=False)
    position_names = {}
    for dim, size in ds.sizes.items():
        name = f"__skretrieval_position_{dim}"
        position_names[dim] = name
        augmented[name] = (dim, np.arange(size))

    selected = select_dataset(augmented, indexers, method=method)
    positional_indexers = {}
    for dim, name in position_names.items():
        positions = np.asarray(selected[name])
        if positions.ndim == 0:
            positional_indexers[dim] = int(positions)
        elif not np.array_equal(positions, np.arange(ds.sizes[dim])):
            positional_indexers[dim] = positions

    radiance = ds["radiance"]
    radiance_indexers = {
        dim: value for dim, value in positional_indexers.items() if dim in radiance.dims
    }
    flat = xr.DataArray(
        np.arange(radiance.size).reshape(radiance.shape), dims=radiance.dims
    )
    flat_indices = flat.isel(radiance_indexers).to_numpy().reshape(-1)
    output_shape = tuple(ds.isel(positional_indexers)["radiance"].shape)
    return _SelectionPlan(
        positional_indexers,
        flat_indices,
        tuple(radiance.shape),
        output_shape,
    )


class RadianceBase(ABC):
    """
    Base functionality that every radiance format must support
    """

    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    @property
    def data(self):
        return self._ds

    @data.setter
    def data(self, value):
        self._ds = value

    def tangent_locations(self):
        """
        Calculates tangent locations for all lines of sight.  If the line of sight does not have a tangent location
        the ground intersection is returned instead
        Returns
        -------
        xr.Dataset
            Dataset containing 'latitude', 'longitude', and 'altitude' of the tangent locations

        """
        los_dims = [dim for dim in self._ds["los_vectors"].dims if dim != "xyz"]

        stacked_los = self._ds["los_vectors"].stack(temp_dim=los_dims)  # noqa: PD013
        stacked_obs = self._ds["observer_position"].stack(  # noqa: PD013
            temp_dim=los_dims
        )

        latitudes = []
        longitudes = []
        altitudes = []
        for idx in stacked_los["temp_dim"]:
            one_los = stacked_los.sel(temp_dim=idx)
            one_obs = stacked_obs.sel(temp_dim=idx)

            lat, lon, alt = target_lat_lon_alt(one_los.to_numpy(), one_obs.to_numpy())

            latitudes.append(lat)
            longitudes.append(lon)
            altitudes.append(alt)

        result = xr.Dataset(
            {
                "latitude": (["temp_dim"], latitudes),
                "longitude": (["temp_dim"], longitudes),
                "altitude": (["temp_dim"], altitudes),
            },
            coords=stacked_los.coords,
        )

        return result.unstack("temp_dim").drop("xyz")  # noqa: PD010

    @abstractmethod
    def to_raw(self):
        pass


class RadianceRaw(RadianceBase):
    def __init__(self, ds: xr.Dataset):
        """
        Raw radiance measurements.  This is the simplest structure that can hold measurements from a wide variety of
        instruments.  Each measurement is represented by a line of sight (look vector, observer position, time) and
        a single radiance value.  Optionally, additional parameters may be added such as the estimated noise.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the variables ['radiance', 'los_vector', 'observer_position'] and the coordinates ['meas']
            'meas' is an index that goes across all measurements (could be wavelength or los changes)
        """
        super().__init__(ds)

    def to_raw(self):
        return self

    def _validate_format(self):
        return True


class RadianceGridded(RadianceBase):
    def __init__(self, ds: xr.Dataset):
        """
        A specific radiance format where the radiance data can be represented on a (wavelength, line of sight) grid.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the variable ['radiance', 'los_vector', 'observer_position'] and the coordinates
            ['wavelength', 'los'] where 'los' is a simple indexer that goes across changes in the line of sight.
        """
        super().__init__(ds)

    def _validate_format(self):
        return True

    def to_raw(self):
        new_ds = self.data.stack(meas=["wavelength", "los"])  # noqa: PD013

        return RadianceRaw(new_ds)


class LinearizedRadianceGridded(RadianceGridded):
    """Gridded radiance with Jacobian and adjoint product callbacks."""

    def __init__(
        self,
        ds: xr.Dataset,
        matvec: Callable[[np.ndarray], xr.DataArray],
        rmatvec: Callable[[xr.DataArray], np.ndarray],
        n_state: int,
        pullback: Callable[[np.ndarray], list[VJPContribution]] | None = None,
        selection_cache: dict | None = None,
        selection_path: tuple = (),
        vjp_depth: int = 0,
    ) -> None:
        super().__init__(ds)
        self._matvec = matvec
        self._rmatvec = rmatvec
        self._n_state = n_state
        self._pullback = pullback
        self._selection_cache = {} if selection_cache is None else selection_cache
        self._selection_path = selection_path
        self._vjp_depth = vjp_depth

    @property
    def n_state(self) -> int:
        return self._n_state

    @property
    def vjp_depth(self) -> int:
        return self._vjp_depth

    def jvp(self, x: np.ndarray) -> xr.DataArray:
        return self._matvec(x)

    def vjp(self, cotangent: xr.DataArray) -> np.ndarray:
        return evaluate_vjp_contributions(self.vjp_contributions(cotangent))

    def vjp_contributions(self, cotangent: xr.DataArray) -> list[VJPContribution]:
        cotangent = np.asarray(cotangent).reshape(self.data["radiance"].shape)
        return [VJPContribution(self, cotangent, self._continue_vjp)]

    def _continue_vjp(
        self, cotangent: np.ndarray
    ) -> np.ndarray | list[VJPContribution]:
        if self._pullback is not None:
            return self._pullback(cotangent)

        template = self.data["radiance"]
        labeled_cotangent = xr.DataArray(
            np.asarray(cotangent).reshape(template.shape),
            dims=template.dims,
            coords=template.coords,
        )
        return self._rmatvec(labeled_cotangent)

    def with_data(self, ds: xr.Dataset) -> LinearizedRadianceGridded:
        return type(self)(
            ds,
            self._matvec,
            self._rmatvec,
            self._n_state,
            pullback=self.vjp_contributions,
            selection_cache=self._selection_cache,
            selection_path=self._selection_path,
            vjp_depth=self._vjp_depth + 1,
        )

    def selection_plan(self, *, method: str | None = None, **kwargs) -> _SelectionPlan:
        """Return a cached positional plan for a label-based selection."""
        selection_key = self._selection_key(method, kwargs)
        plan = self._selection_cache.get(selection_key)
        if plan is None or plan.input_shape != self.data["radiance"].shape:
            plan = _compile_selection_plan(self.data, kwargs, method)
            self._selection_cache[selection_key] = plan
        return plan

    def _selection_key(self, method: str | None, indexers: dict) -> tuple:
        return (
            self._selection_path,
            method,
            tuple(
                sorted(
                    (key, _hashable_indexer(value)) for key, value in indexers.items()
                )
            ),
        )

    def selection_values(
        self, plan: _SelectionPlan, name: str = "radiance"
    ) -> np.ndarray:
        """Gather values using a plan whose source has radiance-shaped variables."""
        return (
            np.asarray(self.data[name])
            .reshape(-1)[plan.flat_indices]
            .reshape(plan.output_shape)
        )

    def selection_jvp(self, plan: _SelectionPlan, x: np.ndarray) -> np.ndarray:
        return (
            np.asarray(self.jvp(x))
            .reshape(-1)[plan.flat_indices]
            .reshape(plan.output_shape)
        )

    def selection_vjp_contributions(
        self, plan: _SelectionPlan, cotangent: np.ndarray
    ) -> list[VJPContribution]:
        values = np.asarray(cotangent).reshape(-1)
        full = np.zeros(plan.input_shape, dtype=values.dtype)
        np.add.at(full.reshape(-1), plan.flat_indices, values)
        return self.vjp_contributions(full)

    def select(
        self, *, assign_coords: dict | None = None, method=None, **kwargs
    ) -> LinearizedRadianceGridded:
        plan = self.selection_plan(method=method, **kwargs)
        selection_key = self._selection_key(method, kwargs)

        original_selected_ds = self.data.isel(plan.indexers)
        selected_ds = original_selected_ds
        if assign_coords is not None:
            selected_ds = selected_ds.assign_coords(**assign_coords)

        selected_template = selected_ds["radiance"]

        def matvec(x: np.ndarray) -> xr.DataArray:
            values = self.selection_jvp(plan, x)
            return xr.DataArray(
                values.reshape(selected_template.shape),
                dims=selected_template.dims,
                coords=selected_template.coords,
            )

        def rmatvec(cotangent: xr.DataArray) -> np.ndarray:
            return evaluate_vjp_contributions(
                self.selection_vjp_contributions(plan, cotangent)
            )

        def pullback(cotangent: xr.DataArray) -> list[VJPContribution]:
            return self.selection_vjp_contributions(plan, cotangent)

        return type(self)(
            selected_ds,
            matvec,
            rmatvec,
            self._n_state,
            pullback=pullback,
            selection_cache=self._selection_cache,
            selection_path=selection_key,
            vjp_depth=self._vjp_depth + 1,
        )


class RadianceSpectralImage(RadianceGridded):
    def __init__(self, ds, num_columns: int | None = None, num_rows: int | None = None):
        """
        A Specific radiance format to hold a (wavelength x columns x rows) grid of data. Only one of `num_columns` or
        `num_rows` should be specified.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the variable ['radiance', 'los_vector', 'observer_position'] and the coordinates
            ['wavelength', 'los'] where 'los' is a simple indexer that goes across changes in the line of sight.
        num_columns : int
            Number of columns in the radiance grid. Optional, inferred from num_rows if not provided.
        num_rows : int
            Number of rows in the radiance grid. Optional, inferred from num_columns if not provided.
        """
        if (num_columns is None) and (num_rows is None):
            msg = "either num_columns or num_rows must be specified"
            raise ValueError(msg)

        if num_rows is None:
            num_rows = int(len(ds.los) / num_columns)
        elif num_columns is None:
            num_columns = int(len(ds.los) / num_rows)

        if num_rows * num_columns != len(ds.los):
            msg = "number of pixels must equal the number of lines of sight"
            raise ValueError(msg)

        nx = np.arange(0, num_columns)
        ny = np.arange(0, num_rows)
        mi = pd.MultiIndex.from_product([ny, nx], names=["ny", "nx"])
        dsc = ds.copy()
        dsc.coords["los"] = mi
        dsc = dsc.unstack("los")  # noqa: PD010

        super().__init__(dsc)


class RadianceOrbit:
    """
    A collection of other RadianceFormats that combine together to create an entire orbit of L1 data

    For example an entire orbit of OMPS data can be created either as a single RadianceGridded for the entire orbit
    or as a List of single RadianceGridded for each image
    """

    def __init__(
        self,
        data: list[RadianceBase],
        wf: list[sparse.spmatrix] | None = None,
        wf_names=None,
    ):
        self._data = data
        self._wf = wf
        self._wf_names = wf_names

        # Create the image slices
        self._slices = []
        cur_idx = 0
        for rad in self._data:
            self._slices.append(slice(cur_idx, cur_idx + len(rad.data.los)))
            cur_idx += len(rad.data.los)

    def derived_type(self):
        return type(self._data[0])

    @property
    def wf(self):
        return self._wf

    @property
    def wf_names(self):
        return self._wf_names

    def image_radiance(self, index, dense_wf=False):
        radiance = self._data[index]
        if self._wf is not None:
            if dense_wf:
                if self._wf_names is None:
                    wfs = np.array(
                        [wf[self._slices[index], :].toarray() for wf in self._wf]
                    )
                    radiance.data["wf"] = (["wavelength", "los", "perturbation"], wfs)
                else:
                    for wf_name, specieswf in zip(self._wf_names, self._wf):
                        wfs = np.array(
                            [wf[self._slices[index], :].toarray() for wf in specieswf]
                        )
                        radiance.data[wf_name] = (
                            ["wavelength", "los", "perturbation"],
                            wfs,
                        )
                return radiance
            if self._wf_names is None:
                wfs = np.array([wf[self._slices[index], :] for wf in self._wf])
                return radiance, wfs
            wfs = []
            for specieswf in self._wf:
                wfs.append(np.array([wf[self._slices[index], :] for wf in specieswf]))
            return radiance, wfs
        return radiance

    def del_wf(self, index):
        if self._wf is not None:
            for wf in self._wf_names:
                self._data[index].data = self._data[index].data.drop(wf)

    @property
    def num_images(self):
        return len(self._data)
