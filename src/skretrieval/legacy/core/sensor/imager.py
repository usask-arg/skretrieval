from __future__ import annotations

import numpy as np
import xarray as xr
from sasktran import Geometry, LineOfSight

from skretrieval.core.lineshape import LineShape
from skretrieval.core.radianceformat import RadianceSpectralImage
from skretrieval.legacy.core.sensor import OpticalGeometry, Sensor
from skretrieval.legacy.core.sensor.spectrograph import Spectrograph


class SpectralImager(Sensor):
    """
    Basic ideal instrument for spectral images on a rectangular 2d grid.
    Contains an array of Spectrograph instances to do the calculation for each pixel.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Array of central wavelengths that the sensor measures.
    spectral_lineshape : LineShape
        Spectral point spread function of the detector. Each pixel is assumed to have the same PSF
    pixel_vert_fov : LineShape
        Vertical angular point spread function of each pixel. Each pixel is assumed to have the same PSF
    pixel_horiz_fov : LineShape
        Horizontal angular point spread function of each pixel. Each pixel is assumed to have the same PSF
    image_horiz_fov : float
        horizontal field of view of the entire detector in radians
    image_vert_fov : float
        vertical field of view of the entire detector in radians
    num_columns: int
        Number of columns in the sensor
    num_rows : int
        Number of rows in the sensor
    """

    def __init__(
        self,
        wavelength_nm: np.ndarray,
        spectral_lineshape: LineShape,
        pixel_vert_fov: LineShape,
        pixel_horiz_fov: LineShape,
        image_horiz_fov: float,
        image_vert_fov: float,
        num_columns: int,
        num_rows: int,
    ):
        self.horizontal_fov = image_horiz_fov
        self.vertical_fov = image_vert_fov
        self._num_columns = num_columns
        self._num_rows = num_rows
        self._wavelenth_nm = wavelength_nm
        self._pixels = self._create_pixels(
            wavelength_nm, spectral_lineshape, pixel_vert_fov, pixel_horiz_fov
        )

    def _create_pixels(
        self,
        wavelength_nm: np.ndarray,
        spectral_lineshape: LineShape,
        pixel_vert_fov: LineShape,
        pixel_horiz_fov: LineShape,
    ) -> list[Spectrograph]:
        pixels = []

        for _i in range(self._num_columns):
            for _j in range(self._num_rows):
                pixels.append(
                    Spectrograph(
                        wavelength_nm=wavelength_nm,
                        pixel_shape=spectral_lineshape,
                        vert_fov=pixel_vert_fov,
                        horiz_fov=pixel_horiz_fov,
                    )
                )

        return pixels

    def model_radiance(
        self,
        optical_geometry: OpticalGeometry,
        model_wavel_nm: np.array,
        model_geometry: Geometry,
        radiance: np.array,
        wf=None,
    ) -> RadianceSpectralImage:
        optical_axes = self.pixel_optical_axes(
            optical_geometry,
            self.horizontal_fov,
            self.vertical_fov,
            self._num_columns,
            self._num_rows,
        )

        model_values = [
            p.model_radiance(oa, model_wavel_nm, model_geometry, radiance, wf=wf)
            for p, oa in zip(self._pixels, optical_axes)
        ]
        return RadianceSpectralImage(
            xr.concat([m.data for m in model_values], dim="los"),
            num_columns=self._num_columns,
        )

    def radiance_format(self) -> type[RadianceSpectralImage]:
        return RadianceSpectralImage

    def measurement_geometry(
        self,
        optical_geometry: OpticalGeometry,
        num_horiz_samples: int | None = None,
        num_vertical_samples: int | None = None,
    ) -> list[LineOfSight]:
        if num_vertical_samples is None:
            num_vertical_samples = self._num_rows
        if num_horiz_samples is None:
            num_horiz_samples = self._num_columns

        optical_axes = self.pixel_optical_axes(
            optical_geometry,
            self.horizontal_fov,
            self.vertical_fov,
            num_horiz_samples,
            num_vertical_samples,
        )
        return [
            LineOfSight(mjd=oa.mjd, observer=oa.observer, look_vector=oa.look_vector)
            for oa in optical_axes
        ]

    def measurement_wavelengths(self) -> np.ndarray:
        return np.unique(np.array([p.measurement_wavelengths() for p in self._pixels]))

    def required_wavelengths(self, res_nm: float) -> np.ndarray:
        return np.unique(
            np.array([p.required_wavelengths(res_nm) for p in self._pixels])
        )

    @staticmethod
    def pixel_optical_axes(
        optical_axis: OpticalGeometry,
        hfov: float,
        vfov: float,
        num_columns: int,
        num_rows: int,
    ) -> list[OpticalGeometry]:
        """
        Get the optical geometry at the center of each pixel in the sensor.

        Parameters
        ----------
        optical_axis : OpticalGeometry
            The optical axis of the center of the sensor
        hfov : float
            horizontal field of view in radians
        vfov : float
            vertical field of view in radians
        num_columns :
            Number of colums in the sensor
        num_rows :
            Number of rows in the sensor

        Returns
        -------
        List[OpticalGeometry]
            The optical geoemetry of each pixel in the sensor as a row-major list.
        """
        pixel_geometry = []
        look = optical_axis.look_vector
        up = optical_axis.local_up
        mjd = optical_axis.mjd
        obs = optical_axis.observer

        dist = 1e6
        htan = np.tan(hfov / 2 * np.pi / 180) * dist
        vtan = np.tan(vfov / 2 * np.pi / 180) * dist
        x = np.linspace(-htan, htan, num_columns)
        y = np.linspace(-vtan, vtan, num_rows)

        nx, ny = np.meshgrid(x, y)
        center = obs + look * dist
        xaxis = np.cross(up, look)
        xaxis /= np.linalg.norm(xaxis)
        yaxis = up
        for (
            x,
            y,
        ) in zip(nx.flatten(), ny.flatten()):
            pos = center + xaxis * x + yaxis * y
            los = pos - obs
            los /= np.linalg.norm(los)
            pixel_yaxis = np.cross(los, up)
            pixel_yaxis /= np.linalg.norm(pixel_yaxis)
            pixel_up = np.cross(pixel_yaxis, los)
            pixel_up /= np.linalg.norm(pixel_up)
            pixel_geometry.append(
                OpticalGeometry(
                    observer=obs, look_vector=los, local_up=pixel_up, mjd=mjd
                )
            )

        return pixel_geometry
