from __future__ import annotations

import numpy as np
import xarray as xr
from sasktran import Geometry, LineOfSight

import skretrieval.core.radianceformat as radianceformat
from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import LineShape
from skretrieval.legacy.core.sensor import Sensor


class Pixel(Sensor):
    """
    A single pixel with corresponding horizontal field of view, vertical field of view, and wavelength line shape.
    """

    def __init__(
        self,
        mean_wavel_nm: float,
        pixel_shape: LineShape,
        vert_fov: LineShape,
        horiz_fov: LineShape,
    ):
        """
        Parameters
        ----------
        mean_wavel_nm : float
            Mean wavelength of the pixel
        pixel_shape: LineShape
            Lineshape for the wavelength
        vert_fov: LineShape
            Vertical field of view
        horiz_fov: LineShape
            Horizontal field of view
        """
        self._wavelength_nm = mean_wavel_nm
        self._pixel_shape = pixel_shape
        self._vert_fov = vert_fov
        self._horiz_fov = horiz_fov

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
        wavel_interp, los_interp = self._construct_interpolation(
            hires_wavel_nm=model_wavel_nm,
            model_geometry=model_geometry,
            optical_axis=optical_geometry,
        )

        modelled_radiance = np.einsum("ij,jk...,kl", wavel_interp, radiance, los_interp)

        meas_los = self.measurement_geometry(optical_geometry)[0]

        data = xr.Dataset(
            {
                "radiance": (["wavelength", "los"], modelled_radiance),
                "mjd": (["los"], [meas_los.mjd]),
                "los_vectors": (
                    ["los", "xyz"],
                    np.array(meas_los.look_vector).reshape((1, 3)),
                ),
                "observer_position": (
                    ["los", "xyz"],
                    np.array(meas_los.observer).reshape((1, 3)),
                ),
            },
            coords={
                "wavelength": [self.measurement_wavelengths()],
                "xyz": ["x", "y", "z"],
            },
        )

        if wf is not None:
            modelled_wf = np.einsum("ij,jkl,k...", wavel_interp, wf, los_interp)

            data["wf"] = (["wavelength", "los", "perturbation"], modelled_wf)

        return radianceformat.RadianceGridded(data)

    def required_wavelengths(self, res_nm: float) -> np.array:
        """
        Recommended wavelengths to use in the high resolution input radiance

        Parameters
        ----------
        res_nm: float
            Desired model resolution

        Returns
        -------
        np.array
            Array of wavelengths
        """
        bounds = self._pixel_shape.bounds()
        return np.arange(bounds[0], bounds[1], res_nm) + self._wavelength_nm

    def measurement_wavelengths(self):
        """
        Central wavelength of the pixel
        """
        return self._wavelength_nm

    @staticmethod
    def radiance_format():
        return radianceformat.RadianceGridded

    def _construct_wavelength_interpolator(self, hires_wavel_nm):
        """
        Internally constructs the matrix used for the wavelength interpolation
        """
        wavel_interpolator = np.zeros((1, len(hires_wavel_nm)))

        wavel_interpolator[0, :] = self._pixel_shape.integration_weights(
            self._wavelength_nm, hires_wavel_nm
        )

        return wavel_interpolator

    def _construct_los_interpolator(
        self, model_geometry, optical_axis: OpticalGeometry
    ):
        """
        Internally constructs the matrix used for the line of sight interpolation
        """
        x_axis = np.array(optical_axis.look_vector)
        vert_normal = np.cross(np.array(x_axis), np.array(optical_axis.local_up))
        vert_normal = vert_normal / np.linalg.norm(vert_normal)
        vert_y_axis = np.cross(vert_normal, x_axis)

        horiz_y_axis = vert_normal

        horiz_angle = []
        vert_angle = []

        for los in model_geometry.lines_of_sight:
            vert_angle.append(
                np.arctan2(
                    np.dot(los.look_vector, vert_y_axis),
                    np.dot(los.look_vector, x_axis),
                )
            )
            horiz_angle.append(
                np.arctan2(
                    np.dot(los.look_vector, horiz_y_axis),
                    np.dot(los.look_vector, x_axis),
                )
            )

        horiz_interpolator = self._horiz_fov.integration_weights(
            0, np.array(horiz_angle)
        )
        vert_interpolator = self._vert_fov.integration_weights(0, np.array(vert_angle))

        # TODO: Only valid for exponential distributions?
        los_interpolator = np.zeros((len(vert_interpolator), 1))
        los_interpolator[:, 0] = horiz_interpolator * vert_interpolator
        los_interpolator[:, 0] /= np.nansum(los_interpolator)

        return los_interpolator

    def _construct_interpolation(
        self, hires_wavel_nm, model_geometry, optical_axis: OpticalGeometry
    ):
        """
        Creates both the wavelength and line of sight interpolators.  This is done so we can write

        output_radiance = (wavel_interp) @ (input_radiance) @ (los_interp)
        """
        # Radiance is [wavel, los], which gets collapsed down to a single measurement

        wavel_interpolator = self._construct_wavelength_interpolator(hires_wavel_nm)
        los_interpolator = self._construct_los_interpolator(
            model_geometry, optical_axis
        )

        # Collapsed radiance is is wavel_interpolator @ radiance @ los_interpolator
        return wavel_interpolator, los_interpolator
