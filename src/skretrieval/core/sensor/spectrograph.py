from __future__ import annotations

from copy import copy

import numpy as np
import xarray as xr

import skretrieval.core.radianceformat as radianceformat
from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import LineShape
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.core.sensor import Sensor


class Spectrograph(Sensor):
    def __init__(
        self,
        wavelength_nm: np.array,
        pixel_shape: list[LineShape],
        vert_fov: LineShape,
        horiz_fov: LineShape,
        spectral_native_coordinate: str = "wavelength_nm",
    ):
        """
        A spectrograph is a 1D array of pixels

        Parameters
        ----------
        wavelength_nm : np.array
            Central wavelengths for each pixel
        pixel_shape: LineShape
            Wavelength line shape
        vert_fov: LineShape
            Vertical field of view
        horiz_fov: LineShape
            Horizontal field of view
        """

        self._wavelength_nm = wavelength_nm
        self._wavenumber_cminv = 1e7 / wavelength_nm
        self._pixel_shape = pixel_shape

        self._vert_fov = vert_fov
        self._horiz_fov = horiz_fov

        self._cached_wavel_interp = None
        self._cached_wavel_interp_wavel = None

        self._spectral_native_coordinate = spectral_native_coordinate

    def _construct_interpolators(self, orientation, los_vectors, model_spectral_grid):
        x_axis = np.array(orientation.look_vector)
        vert_normal = np.cross(np.array(x_axis), np.array(orientation.local_up))
        vert_normal = vert_normal / np.linalg.norm(vert_normal)
        vert_y_axis = np.cross(vert_normal, x_axis)

        horiz_y_axis = vert_normal

        horiz_angle = []
        vert_angle = np.arctan2(
            np.dot(los_vectors, vert_y_axis), np.dot(los_vectors, x_axis)
        )

        horiz_angle = np.arctan2(
            np.dot(los_vectors, horiz_y_axis), np.dot(los_vectors, x_axis)
        )

        horiz_interpolator = self._horiz_fov.integration_weights(
            0, np.array(horiz_angle)
        )
        vert_interpolator = self._vert_fov.integration_weights(0, np.array(vert_angle))

        los_interpolator = np.zeros(len(vert_interpolator))
        los_interpolator = horiz_interpolator * vert_interpolator
        los_interpolator /= np.nansum(los_interpolator)

        los_interpolator = los_interpolator.reshape(-1, 1)

        if not np.array_equal(model_spectral_grid, self._cached_wavel_interp_wavel):
            wavel_interp = []
            for cw, p in zip(self._wavelength_nm, self._pixel_shape):
                weights = p.integration_weights(cw, model_spectral_grid)

                wavel_interp.append(weights / weights.sum())

            wavel_interp = np.vstack(wavel_interp)
            self._cached_wavel_interp = wavel_interp
            self._cached_wavel_interp_wavel = copy(model_spectral_grid)

        return self._cached_wavel_interp, los_interpolator

    def model_radiance(
        self,
        radiance: SASKTRANRadiance,
        orientation: OpticalGeometry,
    ) -> radianceformat.RadianceGridded:
        wavel_interp, los_interp = self._construct_interpolators(
            orientation,
            radiance.data["look_vectors"].to_numpy(),
            radiance.data["wavelength_nm"].to_numpy(),
        )

        modelled_radiance = np.einsum(
            "ij,jk...,kl",
            wavel_interp,
            radiance.data["radiance"].to_numpy(),
            los_interp,
            optimize=True,
        )

        data = xr.Dataset(
            {
                "radiance": (["wavelength", "los"], modelled_radiance),
                "mjd": (["los"], [orientation.mjd]),
                "los_vectors": (
                    ["los", "xyz"],
                    orientation.look_vector.reshape((1, 3)),
                ),
                "observer_position": (
                    ["los", "xyz"],
                    orientation.observer.reshape((1, 3)),
                ),
            },
            coords={
                "wavelength": self._wavelength_nm,
                "xyz": ["x", "y", "z"],
            },
        )
        for key in list(radiance.data):
            if key.startswith("wf"):
                modelled_wf = np.einsum(
                    "ij,ljk,km->iml",
                    wavel_interp,
                    radiance.data[key].to_numpy(),
                    los_interp,
                    optimize=True,
                )

                data[key] = (
                    ["wavelength", "los", radiance.data[key].dims[0]],
                    modelled_wf,
                )

        return radianceformat.RadianceGridded(data)

    def radiance_format(self) -> type[radianceformat.RadianceGridded]:
        return radianceformat.RadianceGridded


class SpectrographOnlySpectral(Sensor):
    def __init__(
        self,
        wavelength_nm: np.array,
        pixel_shape: list[LineShape],
        spectral_native_coordinate: str = "wavelength_nm",
        assign_coord: str = "wavelength",
    ):
        """
        Similar to a spectrograph but does not perform convolution in spatial space, just wavelength

        Parameters
        ----------
        wavelength_nm : np.array
            Central wavelengths for each pixel
        pixel_shape: LineShape
            Wavelength line shape
        """

        self._wavelength_nm = wavelength_nm
        self._wavenumber_cminv = 1e7 / wavelength_nm
        self._pixel_shape = pixel_shape

        self._cached_wavel_interp = None
        self._cached_wavel_interp_wavel = None

        self._spectral_native_coordinate = spectral_native_coordinate
        self._assign_coord = assign_coord

        if spectral_native_coordinate == "wavelength_nm":
            self._assign_vals = self._wavelength_nm
        else:
            self._assign_vals = self._wavenumber_cminv

    def _construct_interpolators(self, model_spectral_grid):
        if not np.array_equal(model_spectral_grid, self._cached_wavel_interp_wavel):
            wavel_interp = []
            for cw, p in zip(self._assign_vals, self._pixel_shape):
                weights = p.integration_weights(cw, model_spectral_grid)

                wavel_interp.append(weights / weights.sum())

            wavel_interp = np.vstack(wavel_interp)
            self._cached_wavel_interp = wavel_interp
            self._cached_wavel_interp_wavel = copy(model_spectral_grid)

        return self._cached_wavel_interp

    def model_radiance(
        self,
        radiance: SASKTRANRadiance,
        orientation: OpticalGeometry,  # noqa: ARG002
    ) -> radianceformat.RadianceGridded:
        wavel_interp = self._construct_interpolators(
            radiance.data[self._spectral_native_coordinate].to_numpy()
        )

        modelled_radiance = np.einsum(
            "ij,jk...",
            wavel_interp,
            radiance.data["radiance"].isel(stokes=0).to_numpy(),
            optimize=True,
        )

        data = xr.Dataset(
            {
                "radiance": ([self._assign_coord, "los"], modelled_radiance),
            },
            coords={
                self._assign_coord: self._assign_vals,
                "xyz": ["x", "y", "z"],
            },
        )

        if "look_vectors" in radiance.data:
            data["los_vectors"] = radiance.data["look_vectors"]

        if "observer_position" in radiance.data:
            data["observer_position"] = radiance.data["observer_position"]

        for key in list(radiance.data):
            if key.startswith("wf"):
                modelled_wf = np.einsum(
                    "ij,ljk->ikl",
                    wavel_interp,
                    radiance.data[key].isel(stokes=0).to_numpy(),
                    optimize=True,
                )

                data[key] = (
                    [self._assign_coord, "los", radiance.data[key].dims[0]],
                    modelled_wf,
                )

        return radianceformat.RadianceGridded(data)

    def radiance_format(self) -> type[radianceformat.RadianceGridded]:
        return radianceformat.RadianceGridded


def _set_join(lower_bounds, upper_bounds):
    final_sets = [[lower_bounds[0], upper_bounds[0]]]

    for lower, upper in zip(lower_bounds, upper_bounds):
        new_set = True
        for set in final_sets:
            if _in_set(set, lower) and not _in_set(set, upper):
                set[1] = upper
                new_set = False
                break
            elif not _in_set(set, lower) and _in_set(set, upper):  # noqa: RET508
                set[0] = lower
                new_set = False
                break
            elif _in_set(set, lower) and _in_set(set, upper):
                new_set = False
        if new_set:
            final_sets.append([lower, upper])
    return final_sets


def _in_set(set, val):
    if val >= set[0] and val <= set[1]:
        return True
    return None
