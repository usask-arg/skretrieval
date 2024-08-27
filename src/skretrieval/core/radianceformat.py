from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

from skretrieval.geodetic import target_lat_lon_alt


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
