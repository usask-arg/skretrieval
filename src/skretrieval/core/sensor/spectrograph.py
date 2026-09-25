from __future__ import annotations

from copy import copy

import numpy as np
import xarray as xr

import skretrieval.core.radianceformat as radianceformat
from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import LineShape
from skretrieval.core.radianceformat import LinearizedRadianceGridded
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.core.sensor import Sensor


def _scalar_stokes(array: xr.DataArray, sensor_name: str) -> xr.DataArray:
    if "stokes" not in array.dims:
        return array
    if array.sizes["stokes"] != 1:
        msg = f"{sensor_name} only supports scalar radiances"
        raise ValueError(msg)
    return array.squeeze("stokes", drop=True)


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
        self._linearized_selection_cache = {}

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

        scalar_radiance = _scalar_stokes(radiance.data["radiance"], type(self).__name__)
        modelled_radiance = np.einsum(
            "ij,jk...,kl",
            wavel_interp,
            scalar_radiance.to_numpy(),
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
                scalar_wf = _scalar_stokes(radiance.data[key], type(self).__name__)
                modelled_wf = np.einsum(
                    "ij,ljk,km->iml",
                    wavel_interp,
                    scalar_wf.to_numpy(),
                    los_interp,
                    optimize=True,
                )

                data[key] = (
                    ["wavelength", "los", scalar_wf.dims[0]],
                    modelled_wf,
                )

        if hasattr(radiance, "jvp") and hasattr(radiance, "vjp"):

            def matvec(x, *, rad=radiance, w=wavel_interp, los=los_interp, tmpl=data):
                raw = _scalar_stokes(rad.jvp(x), type(self).__name__)
                values = np.einsum(
                    "ij,jk...,kl",
                    w,
                    raw.to_numpy(),
                    los,
                    optimize=True,
                )
                return xr.DataArray(
                    values,
                    dims=tmpl["radiance"].dims,
                    coords=tmpl["radiance"].coords,
                )

            def raw_cotangent(
                cotangent, *, rad=radiance, w=wavel_interp, los=los_interp
            ):
                raw = rad.data["radiance"]
                # Transpose the spectral and spatial convolutions in matvec.
                values = np.einsum(
                    "ij,il...,kl",
                    w,
                    np.asarray(cotangent).reshape(data["radiance"].shape),
                    los,
                    optimize=True,
                )
                if "stokes" in raw.dims:
                    values = np.expand_dims(values, raw.get_axis_num("stokes"))
                return values

            def rmatvec(cotangent, *, rad=radiance, adjoint=raw_cotangent):
                return rad.vjp(adjoint(cotangent))

            def pullback(cotangent, *, rad=radiance, adjoint=raw_cotangent):
                raw = adjoint(cotangent)
                if hasattr(rad, "vjp_contributions"):
                    return rad.vjp_contributions(raw)
                return [radianceformat.VJPContribution(rad, raw, rad.vjp)]

            return LinearizedRadianceGridded(
                data,
                matvec,
                rmatvec,
                radiance.n_state,
                pullback=pullback,
                selection_cache=self._linearized_selection_cache,
                selection_path=("spectrograph",),
                vjp_depth=getattr(radiance, "vjp_depth", 0) + 1,
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
        stokes_sensitivity: dict[str, np.array] | None = None,
    ):
        """
        Similar to a spectrograph but does not perform convolution in spatial space, just wavelength

        Parameters
        ----------
        wavelength_nm : np.array
            Central wavelengths for each pixel
        pixel_shape: LineShape
            Wavelength line shape
        spectral_native_coordinate: str
            Coordinate the lineshape is assumed to be a function of
        assign_coord: str
            Resulting coordinate name in the output L1 dataset
        stokes_sensitivity: dict
            Dictionary of stokes sensitivity matrices. The default is {"I": np.array([1.0, 0, 0, 0])}
            Can set to multiple sensitivies, e.g., {"vert": np.array([0.5, 0.5, 0, 0]), "horiz": np.array([0.5, -0.5, 0, 0])}
            When set to multiple sensitivities, the output will be a dictionary of RadianceGridded objects corresponding to each sensitivity
        """

        self._wavelength_nm = wavelength_nm
        self._wavenumber_cminv = 1e7 / wavelength_nm
        self._pixel_shape = pixel_shape

        self._cached_wavel_interp = None
        self._cached_wavel_interp_wavel = None
        self._linearized_selection_cache = {}

        self._spectral_native_coordinate = spectral_native_coordinate
        self._assign_coord = assign_coord

        if spectral_native_coordinate == "wavelength_nm":
            self._assign_vals = self._wavelength_nm
        else:
            self._assign_vals = self._wavenumber_cminv

        if stokes_sensitivity is None:
            self._stokes_sensitivity = {"I": np.array([1.0, 0, 0, 0])}
        else:
            self._stokes_sensitivity = stokes_sensitivity

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

        result = {}

        for k, mueller in self._stokes_sensitivity.items():
            stokes_applied_radiance = radiance.data["radiance"] @ xr.DataArray(
                mueller,
                dims=["stokes"],
                coords={"stokes": ["I", "Q", "U", "V"]},
            )

            modelled_radiance = np.einsum(
                "ij,jk...",
                wavel_interp,
                stokes_applied_radiance.to_numpy(),
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

            for key in list(radiance.data):
                if key.startswith("wf"):
                    stokes_applied_wf = radiance.data[key] @ xr.DataArray(
                        mueller,
                        dims=["stokes"],
                        coords={"stokes": ["I", "Q", "U", "V"]},
                    )

                    modelled_wf = np.einsum(
                        "ij,ljk->ikl",
                        wavel_interp,
                        stokes_applied_wf.to_numpy(),
                        optimize=True,
                    )

                    data[key] = (
                        [self._assign_coord, "los", radiance.data[key].dims[0]],
                        modelled_wf,
                    )
                else:
                    # Copy all of the extra variables
                    if key != "radiance":
                        data[key] = radiance.data[key]

            if hasattr(radiance, "jvp") and hasattr(radiance, "vjp"):
                # Bind this channel's Mueller vector and output template.
                def matvec(
                    x,
                    *,
                    rad=radiance,
                    w=wavel_interp,
                    m=mueller,
                    tmpl=data,
                ):
                    raw = rad.jvp(x)
                    stokes_applied = raw @ xr.DataArray(
                        m,
                        dims=["stokes"],
                        coords={"stokes": ["I", "Q", "U", "V"]},
                    )
                    values = np.einsum(
                        "ij,jk...",
                        w,
                        stokes_applied.to_numpy(),
                        optimize=True,
                    )
                    return xr.DataArray(
                        values,
                        dims=tmpl["radiance"].dims,
                        coords=tmpl["radiance"].coords,
                    )

                def raw_cotangent(
                    cotangent,
                    *,
                    rad=radiance,
                    w=wavel_interp,
                    m=mueller,
                ):
                    raw = rad.data["radiance"]
                    # The adjoint first undoes spectral convolution, then the
                    # Mueller projection by restoring the Stokes dimension.
                    stokes_weight = (
                        xr.DataArray(
                            m,
                            dims=["stokes"],
                            coords={"stokes": ["I", "Q", "U", "V"]},
                        )
                        .reindex(stokes=raw["stokes"], fill_value=0)
                        .to_numpy()
                    )
                    return np.einsum(
                        "ij,ik...,l->jkl",
                        w,
                        np.asarray(cotangent),
                        stokes_weight,
                        optimize=True,
                    )

                def rmatvec(cotangent, *, rad=radiance, adjoint=raw_cotangent):
                    return rad.vjp(adjoint(cotangent))

                def pullback(cotangent, *, rad=radiance, adjoint=raw_cotangent):
                    raw = adjoint(cotangent)
                    if hasattr(rad, "vjp_contributions"):
                        return rad.vjp_contributions(raw)
                    return [radianceformat.VJPContribution(rad, raw, rad.vjp)]

                result[k] = LinearizedRadianceGridded(
                    data,
                    matvec,
                    rmatvec,
                    radiance.n_state,
                    pullback=pullback,
                    selection_cache=self._linearized_selection_cache,
                    selection_path=("spectrograph", k),
                    vjp_depth=getattr(radiance, "vjp_depth", 0) + 1,
                )
            else:
                result[k] = radianceformat.RadianceGridded(data)

        return result

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
