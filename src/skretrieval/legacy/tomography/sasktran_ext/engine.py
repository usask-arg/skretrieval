from __future__ import annotations

from copy import copy

import numpy as np
import sasktran as sk
import xarray as xr
from scipy import sparse

from skretrieval.tomography.grids import OrbitalPlaneGrid


class EngineHRTwoDim:
    def __init__(
        self,
        geometry: list[sk.Geometry],
        atmosphere: sk.atmosphere,
        grid: OrbitalPlaneGrid,
        wavelength: np.ndarray = None,
        max_difference_seconds=100,
        common_options: dict | None = None,
        grid_spacing: float = 1000,
    ):
        """
        Radiative transfer model using the HR engine that calculates radiances for an entire orbit.

        Radiative transfer calculations are split into chunks depending on the time delta between subsequent images.

        Parameters
        ----------
        geometry : List[sk.Geometry]
            A list of geometries for subsequent images/scans
        atmosphere : sk.Atmosphere
            The atmosphere
        grid : OrbitalPlaneGrid
            The grid defining the geometry of the orbit
        wavelength : np.ndarray
            Wavelengths to perform the calculation at
        max_difference_seconds : float, optional
            Time delta in seconds used to determine which radiative transfer calculations should be combined together.
            Default is 100s which roughly corresponds to the sun moving an amount equivalent to the width of the solar
            disk
        common_options : dict
            RTM options
        grid_spacing : float, optional
            Grid spacing of the model, default 1000 m
        """
        self._atmo = atmosphere
        self._grid = grid
        self._wavelength = wavelength
        self._geometry = geometry
        self._max_difference_seconds = max_difference_seconds
        self._grid_spacing = grid_spacing

        self._construct_segments()
        if common_options is not None:
            self._common_options = common_options
        else:
            self._common_options = {}

    def calculate_radiance(
        self, full_stokes_vector=False, stokes_orientation="geographic"
    ):
        """
        Calculates the radiance for the entire orbit.

        Data is returned as a dictionary with keys 'radiance' that contains a list of radiances for each input
        image/scan, and weighting function keys if calculated.
        """
        # Calculate the radiance separately for every geo_segment
        output = {}
        all_segment_radiances = []
        all_angleidx = []
        for segment in self._geo_segments:
            options = copy(self._common_options)
            new_geometry = self._combined_geometry(segment)
            geodetic = sk.Geodetic()
            geodetic.from_lat_lon_alt(
                new_geometry.reference_point[0],
                new_geometry.reference_point[1],
                new_geometry.reference_point[1],
            )

            local_angles, angleidx, normalandreference = self._grid.get_local_plane(
                geodetic.location
            )
            options["opticalanglegrid"] = np.rad2deg(local_angles)
            options["opticalnormalandreference"] = normalandreference

            engine = sk.EngineHR(new_geometry, self._atmo, self._wavelength, options)
            engine.atmosphere_dimensions = 2
            engine.grid_spacing = self._grid_spacing

            segment_radiance = engine.calculate_radiance(
                "xarray",
                full_stokes_vector=full_stokes_vector,
                stokes_orientation=stokes_orientation,
            )

            all_segment_radiances.append(segment_radiance.drop("wf_brdf"))
            all_angleidx.append(angleidx)

        if self._common_options.get("calcwf", 0) == 3:
            # Calculating weighting functions and we have to convert the weighting functions to the overall grid
            num_los = [len(ds["los"]) for ds in all_segment_radiances]
            num_angles = len(self._grid._angles)

            if "wfheights" in self._common_options:
                num_alts = len(self._common_options["wfheights"])
            else:
                num_alts = 100  # Default

            var_names = np.array(list(all_segment_radiances[0].keys()))
            wf_names = var_names[[name.startswith("wf_") for name in var_names]]

            for name in wf_names:
                # Full WF matrix is going to be [num_wavel, num_los, num_angles * num_alts]
                sparse_wf = [
                    sparse.lil_matrix((np.nansum(num_los), num_angles * num_alts))
                    for w in self._wavelength
                ]

                for angleidx, segment_radiance, los_end, nl in zip(
                    all_angleidx, all_segment_radiances, np.cumsum(num_los), num_los
                ):
                    # Raw wf's are stored with altitude being the fastest varying dimension
                    pert_slice = slice(
                        angleidx[0] * num_alts, (angleidx[-1] + 1) * num_alts
                    )
                    los_slice = slice(los_end - nl, los_end)

                    for w_idx in range(len(self._wavelength)):
                        sparse_wf[w_idx][los_slice, pert_slice] = segment_radiance[
                            name
                        ].to_numpy()[w_idx, :, :]
                for w_idx, _wf in enumerate(sparse_wf):
                    sparse_wf[w_idx] = sparse_wf[w_idx].tocsc()
                output[name] = sparse_wf

                for idx, segment in enumerate(all_segment_radiances):
                    all_segment_radiances[idx] = segment.drop([name])

        concat_rad = xr.concat(all_segment_radiances, dim="los")

        image_radiances = []
        cur_idx = 0
        for geo in self._geometry:
            image_radiances.append(
                concat_rad.isel(los=slice(cur_idx, cur_idx + len(geo.lines_of_sight)))
            )
            cur_idx += len(geo.lines_of_sight)

        output["radiance"] = image_radiances

        return output

    def _construct_segments(self):
        # calculate the mean mjds of every geometry object
        mjds = []
        for geo in self._geometry:
            geo_mjd = [los.mjd for los in geo.lines_of_sight]
            mjds.append(np.nanmean(geo_mjd))

        mjds = np.asarray(mjds)

        geo_segment_idx = []
        temp_idx = [0]

        current_start_seconds = 0
        mjd_differences = np.cumsum(np.diff(mjds) * 3600 * 24)
        for idx, difference in enumerate(mjd_differences):
            if (difference - current_start_seconds) > self._max_difference_seconds:
                current_start_seconds = difference
                geo_segment_idx.append(temp_idx)
                temp_idx = [idx + 1]
            else:
                temp_idx.append(idx + 1)
        geo_segment_idx.append(temp_idx)

        self._geo_segments = geo_segment_idx

    def _combined_geometry(self, segment):
        all_lines_of_sight = np.concatenate(
            [g.lines_of_sight for g in [self._geometry[s] for s in segment]]
        )

        new_geometry = sk.Geometry()
        new_geometry.lines_of_sight = all_lines_of_sight

        mean_lat = np.nanmean(
            [los.tangent_location().latitude for los in all_lines_of_sight]
        )
        mean_lon = np.nanmean(
            [los.tangent_location().longitude for los in all_lines_of_sight]
        )
        mean_mjd = np.nanmean([los.mjd for los in all_lines_of_sight])

        new_reference_point = [mean_lat, mean_lon, 0, mean_mjd]
        new_geometry.reference_point = new_reference_point

        return new_geometry
