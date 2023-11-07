from __future__ import annotations

from collections.abc import Iterable
from copy import copy

import numpy as np
import xarray as xr
from sasktran import Geometry, LineOfSight

import skretrieval.core.radianceformat as radianceformat
from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import LineShape
from skretrieval.legacy.core.sensor import Sensor
from skretrieval.legacy.core.sensor.pixel import Pixel


class SpectrographPixelArray(Sensor):
    """
    A spectrograph created from an array of Pixels
    """

    def __init__(
        self,
        wavelength_nm: np.array,
        pixel_shape: LineShape,
        vert_fov: LineShape,
        horiz_fov: LineShape,
    ):
        """

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
        self._pixels = []

        if isinstance(pixel_shape, Iterable):
            for w, p_shape in zip(wavelength_nm, pixel_shape):
                self._pixels.append(Pixel(w, p_shape, vert_fov, horiz_fov))
        else:
            for w in wavelength_nm:
                self._pixels.append(Pixel(w, pixel_shape, vert_fov, horiz_fov))

        self.wavelength_nm = wavelength_nm

    def measurement_geometry(self, optical_geometry: OpticalGeometry):
        return [
            LineOfSight(
                optical_geometry.mjd,
                optical_geometry.observer,
                optical_geometry.look_vector,
            )
        ]

    def model_radiance(
        self,
        optical_geometry: OpticalGeometry,
        model_wavel_nm: np.array,
        model_geometry: Geometry,
        radiance: np.array,
        wf=None,
    ):
        data = xr.concat(
            [
                p.model_radiance(
                    optical_geometry, model_wavel_nm, model_geometry, radiance, wf
                ).data
                for p in self._pixels
            ],
            dim="wavelength",
            data_vars="minimal",
        )

        return radianceformat.RadianceGridded(data)

    @staticmethod
    def radiance_format():
        return radianceformat.RadianceGridded

    def measurement_wavelengths(self):
        return self.wavelength_nm

    def required_wavelengths(self, res_nm: float) -> np.array:
        """
        Recommended wavelengths for high resolution calculations

        Parameters
        ----------
        res_nm: float
            Resolution in nm

        Returns
        -------
        np.array
        """
        lower_bounds = np.zeros(len(self._pixels))
        upper_bounds = np.zeros(len(self._pixels))

        for i, p in enumerate(self._pixels):
            lower_bounds[i], upper_bounds[i] = p._pixel_shape.bounds()

            upper_bounds[i] += p._wavelength_nm
            lower_bounds[i] += p._wavelength_nm

        bounding_sets = _set_join(lower_bounds, upper_bounds)

        wavel = np.concatenate(
            [np.arange(set[0], set[1], res_nm) for set in bounding_sets]
        )

        if len(wavel) == 0:
            wavel = np.array(self.wavelength_nm)

        return wavel


class Spectrograph(SpectrographPixelArray):
    """
    Same functionality as Spectrogaph, but coded to not consist of individual pixels for speed
    """

    def __init__(
        self,
        wavelength_nm: np.array,
        pixel_shape: LineShape,
        vert_fov: LineShape,
        horiz_fov: LineShape,
    ):
        """

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
        super().__init__(wavelength_nm, pixel_shape, vert_fov, horiz_fov)

        self._cached_wavel_interp = None
        self._cached_wavel_interp_wavel = None

    def _construct_interpolators(
        self, model_geometry, optical_geometry, model_wavel_nm
    ):
        los_interp = self._pixels[0]._construct_los_interpolator(
            model_geometry, optical_geometry
        )

        if not np.array_equal(model_wavel_nm, self._cached_wavel_interp_wavel):
            wavel_interp = []
            for p in self._pixels:
                wavel_interp.append(
                    p._construct_wavelength_interpolator(model_wavel_nm)
                )

            wavel_interp = np.vstack(wavel_interp)
            self._cached_wavel_interp = wavel_interp
            self._cached_wavel_interp_wavel = copy(model_wavel_nm)

        return self._cached_wavel_interp, los_interp

    def model_radiance(
        self,
        optical_geometry: OpticalGeometry,
        model_wavel_nm: np.array,
        model_geometry: Geometry,
        radiance: np.array,
        wf=None,
    ):
        wavel_interp, los_interp = self._construct_interpolators(
            model_geometry, optical_geometry, model_wavel_nm
        )

        modelled_radiance = np.einsum(
            "ij,jk...,kl", wavel_interp, radiance, los_interp, optimize="optimal"
        )

        data = xr.Dataset(
            {
                "radiance": (["wavelength", "los"], modelled_radiance),
                "mjd": (["los"], [optical_geometry.mjd]),
                "los_vectors": (
                    ["los", "xyz"],
                    optical_geometry.look_vector.reshape((1, 3)),
                ),
                "observer_position": (
                    ["los", "xyz"],
                    optical_geometry.observer.reshape((1, 3)),
                ),
            },
            coords={
                "wavelength": self.measurement_wavelengths(),
                "xyz": ["x", "y", "z"],
            },
        )

        if wf is not None:
            modelled_wf = np.einsum(
                "ij,jkl,km->iml", wavel_interp, wf, los_interp, optimize="optimal"
            )

            data["wf"] = (["wavelength", "los", "perturbation"], modelled_wf)

        return radianceformat.RadianceGridded(data)


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
