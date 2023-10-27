from __future__ import annotations

import logging

import numpy as np
import sasktran as sk
import scipy as sp
from scipy import interpolate

from skretrieval.core import OpticalGeometry
from skretrieval.util import rotation_matrix


class OrbitalPlaneGrid:
    """
    Class which constructs a two dimensional grid out of a set of tangent
    points and look directions.  The core idea is that when the earths rotation
    is corrected for, the tangent points should form a plane.  A uniform in angle
    grid is constructed in this plane.  The approximation made is that over the
    course of one radiative transfer calculation, the actual grid is almost the
    same as this rotation corrected grid.

    The inputs to this function should be the low resolution values

    Parameters
    ----------
    tp : list or 2d numpy array
        Tangent points, should be convertible to a numpy array of shape(n,3)
    look : list or 2d numpy array
        Look directions, should be convertible to a numpy array of shape(n,3)
    mjd : List or numpy array
        Modified julian dates
    scanslices : list
        Each list element should be a range indicating which line of sights
        belong to which scan. e.g. scanslices[0] could be range(0,60) indicating
        that lines of sight 0-60 are in the first scan
    numinbetween : scalar, optional
        The ratio of scans / retrieval profiles if placementtype is set to uniform
    cutoff : scalar, optional
        Angle in degrees to cutoff of each side of the grid if placementype is
        set to uniform
    extend : scalar, optional
        Angle in degrees to extend each side of the grid if placementtype is set to uniform. Default 0. Do not use
        both extend and cutoff
    placementtype : string, optional
        If set to 'scan', retrieval profiles are placed at the average tp of
        every scan.  If set to uniform, a uniform grid is constructed between
        the first and last scan
    geoid_mode : string, optional
        The geoid to use when calculating latitude/longitude etc.

    """

    def __init__(
        self,
        opt_geo: list[OpticalGeometry],
        grid_altitudes,
        numinbetween=1.0,
        cutoff=0,
        extend=0,
        placementtype="scan",
        geoid_mode="wgs84",
    ):
        self._geoid_mode = geoid_mode
        self._numinbetween = numinbetween  # The number of scans / ret profiles
        self.cutoff = np.deg2rad(cutoff)
        self._extend = np.deg2rad(extend)
        self._numextend = 0

        # Find the tangent point of each measurements
        self._averagescantp = []
        self._averagescanlook = []
        self._averagescanmjd = []
        self._altitudes = grid_altitudes

        (
            self._averagescantp,
            self._averagescanlook,
            self._averagescanmjd,
        ) = self._tangent_points(opt_geo)

        self.averagescanangles = None
        self._xaxis = None
        self._yaxis = None
        self._numretprof = None
        self._angles = None
        self._retprofilemjd = None
        self._retprofileavglook = None
        self._retprofileloc = None

        self._placementtype = placementtype

        self._make_retrieval_profile_locations()

    def _make_retrieval_profile_locations(self):
        """
        Takes the measurement grid and constructs the retrieval grid.
        """
        rottp = self._correct_for_earths_rotation(
            self._averagescantp, self._averagescanmjd, self._averagescanmjd[0]
        )
        localangles = np.zeros(np.size(self._averagescanmjd))
        xaxis = rottp[0, :] / np.linalg.norm(rottp[0, :])
        locallook = rottp[np.min([10, np.size(self._averagescanmjd)])] - rottp[0, :]
        locallook /= np.linalg.norm(locallook)
        normal = np.cross(locallook, xaxis)
        normal /= np.linalg.norm(normal)
        yaxis = np.cross(xaxis, normal)
        yaxis /= np.linalg.norm(yaxis)
        twopiaddition = 0
        for idx in range(1, np.size(localangles)):
            localangles[idx] = (
                np.arctan2(yaxis.dot(rottp[idx, :]), xaxis.dot(rottp[idx, :]))
                + twopiaddition
            )
            if localangles[idx] < (localangles[idx - 1] - np.pi):
                twopiaddition += 2 * np.pi
                localangles[idx] += 2 * np.pi

        self.averagescanangles = localangles
        self._xaxis = xaxis
        self._yaxis = yaxis

        totalanglechange = np.max(localangles)
        averageanglechange = (
            totalanglechange / np.size(self._averagescanmjd) * self._numinbetween
        )
        possibleretangles = np.linspace(
            0, totalanglechange, int(np.floor(totalanglechange / averageanglechange))
        )

        if self._placementtype == "scan":
            possibleretangles = localangles
        elif self._placementtype == "uniform":
            possibleretangles = possibleretangles[
                (possibleretangles > self.cutoff)
                & (possibleretangles < (totalanglechange - self.cutoff))
            ]

            if self._extend > 0:
                delta_angle = possibleretangles[1] - possibleretangles[0]
                num_extend = int(np.floor(self._extend / delta_angle))
                self._numextend = num_extend

                lower_add = (
                    np.arange(num_extend, 0.5, -1) * delta_angle * -1
                    + possibleretangles[0]
                )
                upper_add = (
                    np.arange(1, num_extend + 0.5, 1) * delta_angle
                    + possibleretangles[-1]
                )

                possibleretangles = np.concatenate(
                    (lower_add, possibleretangles, upper_add)
                )
        else:
            msg = "Placement type is not one of ['scan', 'uniform']"
            raise ValueError(msg)

        self._retprofileloc = np.zeros((np.size(possibleretangles), 3))
        self._retprofileavglook = np.zeros((np.size(possibleretangles), 3))
        self._retprofilemjd = np.zeros(np.size(possibleretangles))
        self._numretprof = np.size(possibleretangles)
        self._angles = possibleretangles

        for idx in range(self._numretprof):
            self._retprofileloc[idx, :] = rotation_matrix(
                normal, -1 * possibleretangles[idx]
            ).dot(rottp[0, :])

        self._retprofilemjd = np.interp(
            possibleretangles, localangles, self._averagescanmjd
        )
        f = interpolate.interp1d(
            localangles, self._averagescanlook, axis=0, fill_value="extrapolate"
        )  # Extrapolate in case we are extending the profiles
        self._retprofileavglook = f(possibleretangles)

        self._retprofileloc = self._correct_for_earths_rotation(
            self._retprofileloc, -1 * self._retprofilemjd, -1 * self._averagescanmjd[0]
        )

    def get_local_plane(self, loc: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Returns back the local plane at a specific location.

        Parameters
        ----------
        loc : numpy array
            Shape (3,) tangent point

        Returns
        -------
        angles : numpy array
            Angles in radians in the plane which correspond to the profile locations
            angles are measured with respect to loc
        angleidx : numpy array
            Indicies of the angles in the grid
        normalandreference : numpy array
            The normal and reference of the returned local plane, this is intended
            to be passed into ISKEngine with the 'normalandreference' option
            Result is shape(6,) with the first 3 elements being the normal
            and the last 3 being the reference
        """
        anglesbetween = np.zeros(self._numretprof)
        for idx in range(self._numretprof):
            anglesbetween[idx] = np.arccos(
                loc.dot(self._retprofileloc[idx, :])
                / np.linalg.norm(loc)
                / np.linalg.norm(self._retprofileloc[idx, :])
            )

        minidx = np.argmin(anglesbetween)
        # Use loc as the reference (xaxis), and calculate the normal from the
        # tangent points near
        reference = loc / np.linalg.norm(loc)
        # TODO: check that this gradient is good?
        avglook = (
            np.gradient(self._retprofileloc, axis=0)
            / np.linalg.norm(np.gradient(self._retprofileloc, axis=0), axis=1)[
                :, np.newaxis
            ]
        )[minidx]
        normal = np.cross(avglook, reference)
        normal /= np.linalg.norm(normal)
        yaxis = np.cross(reference, normal)
        yaxis /= np.linalg.norm(yaxis)

        # Rotate the profile locations by the earth's rotation to find the
        # angles in the tangent plane
        # rottp = correct_for_earths_rotation( self.retprofileloc, self.retprofilemjd, self.retprofilemjd[minidx])
        # NEW METHOD: Project the retrieval profile locations onto the plane
        rottp = self.project_onto_plane(self._retprofileloc, normal)
        allangles = np.zeros(self._numretprof)
        twopiaddition = 0
        for idx in range(self._numretprof):
            allangles[idx] = np.arctan2(
                yaxis.dot(rottp[idx, :]), reference.dot(rottp[idx, :])
            )
            if idx > 0 and allangles[idx] < (allangles[idx - 1] - np.pi):
                twopiaddition += 2 * np.pi
                allangles[idx] += 2 * np.pi

        # Have to be careful because we do not assume periodicity at the edges
        # TODO: make this user set
        maxanglediff = np.deg2rad(20)

        lowangleidx = ((allangles >= -1 * maxanglediff) & (allangles < 0)).nonzero()[0]
        lowangleidx = 0 if np.size(lowangleidx) == 0 else lowangleidx[0]
        highangleidx = (allangles[minidx:] > maxanglediff).nonzero()[0]
        if np.size(highangleidx) == 0:
            highangleidx = self._numretprof
        else:
            highangleidx = highangleidx[0] + minidx

        angleidx = np.arange(lowangleidx, highangleidx)
        if np.size(angleidx) == 0:
            angleidx = minidx

        localangles = allangles[angleidx]
        sortidx = np.argsort(localangles)
        localangles = localangles[sortidx]
        angleidx = angleidx[sortidx]

        # Return back what we need
        return localangles, angleidx, np.concatenate((normal, reference))

    def interpolate_onto_grid(
        self,
        prof_lon: np.ndarray,
        prof_lat: np.ndarray,
        prof_height: np.ndarray,
        prof: np.ndarray,
        do_height_interp=True,
    ) -> np.ndarray:
        """
        Converts a lat/lon prof that is defined on any grid to the retrieval grid
        Parameters
        ----------
        prof_lon:
            Longitude for the profile
        prof_lat:
            Latitude for the profile
        prof_height:
            Heights for the profile.  If the profile is only two dimensions this is not used
        prof:
            The profile. (lat, lon, height)
        do_height_interp:
            True if you want to do height interpolation, false otherwise.

        Returns
        -------
        (lat, lon, height) profile on the retrieval grid
        """
        retlat, retlon, retheight = self._lat_lon_alts(self._retprofileloc)
        retlon[retlon > 180] -= 360
        prof_lon = np.asarray(prof_lon)
        prof_lon[prof_lon > 180] -= 360
        retheight = self._altitudes

        # First get vertical profiles at the retrieval locations
        s = np.shape(prof)

        if len(s) > 1:
            # The profile is a 2d profile
            temp_prof = np.zeros((len(retlat), s[1]))
            # profile is (location,height), so loop over heights
            for idx in range(s[1]):
                temp_prof[:, idx] = self.distance_interpolate(
                    prof_lat, prof_lon, retlat, retlon, prof[:, idx]
                )

            if do_height_interp:
                # Now construct the retrieval profile by interpolating over height
                ret_prof = np.zeros((len(retlat), len(retheight)))
                for idx in range(len(retlat)):
                    ret_prof[idx, :] = np.interp(
                        retheight, prof_height, temp_prof[idx, :]
                    )

                return ret_prof
            return temp_prof

        return self.distance_interpolate(prof_lat, prof_lon, retlat, retlon, prof)

    def interpolate_from_grid(
        self,
        prof_lon: np.ndarray,
        prof_lat: np.ndarray,
        prof_height: np.ndarray,
        ret_prof: np.ndarray,
        do_height_interp=True,
    ) -> np.ndarray:
        """
        Converts a profile that is specified on the retrieval grid to a new lat/lon grid
        Parameters
        ----------
        prof_lon:
            Longitude to interpolate onto
        prof_lat:
            Latitude to interpolate onto
        prof_height:
            Heights to interpolate onto
        ret_prof:
            Profile that is specified on the retrieval grid
        do_height_interp:
            True if height interpolation should be done

        Returns
        -------
        Profile on the new grid
        """
        retlat, retlon, retheight = self._lat_lon_alts(self._retprofileloc)
        retlon[retlon > 180] -= 360
        prof_lon = np.asarray(prof_lon)
        prof_lon[prof_lon > 180] -= 360
        retheight = self._altitudes

        # First get vertical profiles at the retrieval locations
        s = np.shape(ret_prof)

        if len(s) > 1:
            # The profile is a 2d profile
            temp_prof = np.zeros((len(retlat), s[1]))
            # profile is (location,height), so loop over heights
            for idx in range(s[1]):
                temp_prof[:, idx] = self.distance_interpolate(
                    retlat, retlon, prof_lat, prof_lon, ret_prof[:, idx]
                )
            if do_height_interp:
                # Now construct the retrieval profile by interpolating over height
                meas_prof = np.zeros((len(prof_lat), len(prof_height)))
                for idx in range(len(prof_lat)):
                    meas_prof[idx, :] = np.interp(
                        prof_height, retheight, temp_prof[idx, :]
                    )

                return meas_prof
            return temp_prof

        return self.distance_interpolate(retlat, retlon, prof_lat, prof_lon, ret_prof)

    @staticmethod
    def distance_interpolate(
        from_lat: np.ndarray,
        from_lon: np.ndarray,
        to_lat: np.ndarray,
        to_lon: np.ndarray,
        profile: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolates a profile in spherical distance.

        Parameters
        ----------
        from_lat : np.ndarray
            Array length (n) of latitudes in degrees for the grid to interpolate from
        from_lon : np.ndarray
            Array length (n) of longitudes in degrees for the grid to interpolate from
        to_lat : np.ndarray
            Array length (m) of latitudes in degrees for the grid to interpolate to
        to_lon : np.ndarray
            Array length (m) of longitudes in degrees for the grid to interpolate to
        profile : np.ndarray
            Profile of size (n) that is to be interpolated

        Returns
        -------
        np.ndarray
            Profile of size (m)
        """
        new_profile = np.zeros(len(to_lat))

        end_lon = from_lon[-1]
        end_lat = from_lat[-1]
        dist_grid_to_end = np.cos(np.deg2rad(end_lat)) * np.cos(
            np.deg2rad(from_lat)
        ) * np.cos(np.deg2rad(from_lon - end_lon)) + np.sin(
            np.deg2rad(from_lat)
        ) * np.sin(
            np.deg2rad(end_lat)
        )
        dist_grid_to_end[dist_grid_to_end > 1] = 1
        dist_grid_to_end[dist_grid_to_end < -1] = -1
        dist_grid_to_end = np.arccos(dist_grid_to_end)

        for idx, (lat, lon) in enumerate(zip(to_lat, to_lon)):
            # Calculate the distance from the this point to all old locations
            # assuming a spherical earth
            distance = np.arccos(
                np.cos(np.deg2rad(lat))
                * np.cos(np.deg2rad(from_lat))
                * np.cos(np.deg2rad(from_lon - lon))
                + np.sin(np.deg2rad(from_lat)) * np.sin(np.deg2rad(lat))
            )

            # Check if we are outside the grid
            if np.all(np.diff(distance) > 0):
                new_profile[idx] = profile[0]
                continue
            if np.all(np.diff(distance) < 0):
                new_profile[idx] = profile[-1]
                continue

            distance *= np.sign(np.gradient(distance))
            useful_idx = np.abs(distance) < np.pi / 2

            if not (
                np.all(np.diff(distance[useful_idx]) > 0)
                or np.all(np.diff(distance[useful_idx]) < 0)
            ):
                logging.warning(
                    "Distance between points is not monotonic, using the nearest point instead"
                )
                new_profile[idx] = sp.interpolate.interp1d(
                    distance, profile, kind="nearest"
                )(0.0)

            if distance[useful_idx][1] < distance[useful_idx][0]:
                new_profile[idx] = np.interp(
                    [0.0], distance[useful_idx][::-1], profile[useful_idx][::-1]
                )
            else:
                new_profile[idx] = np.interp(
                    [0.0], distance[useful_idx], profile[useful_idx]
                )

        return new_profile

    def _tangent_points(self, opt_geo: list[OpticalGeometry]):
        geo = sk.Geodetic(self._geoid_mode)

        tan_locs = []
        tan_looks = []
        tan_mjds = []

        for o in opt_geo:
            geo.from_tangent_point(o.observer, o.look_vector)

            tan_locs.append(geo.location)
            tan_looks.append(o.look_vector)
            tan_mjds.append(o.mjd)

        return np.asarray(tan_locs), np.asarray(tan_looks), np.asarray(tan_mjds)

    @staticmethod
    def _correct_for_earths_rotation(tp, mjd, refmjd=None):
        """
        Takes a set of tangent points, corresponding mjds and rotates them by
        the Earth's rotation.    If you take a set of satellite tangent points
        and mjd's and apply this function, the end result should be a plane

        Parameters
        ----------
        tp: np.ndarray
            (n, 3) of tangent points
        mjd: np.ndarray
            (n,) modified julian dates
        refmjd : float
            The reference mjd where no rotation is applied.  If set to None the mean MJD is used

        Returns
        -------
        2d numpy array
            Shape (n,3) of rotated tangent points
        """
        if refmjd is None:
            refmjd = np.nanmean(mjd)
        omega = 2 * np.pi

        mjddiff = mjd - refmjd
        rottp = np.zeros(np.shape(tp))

        for idx in range(np.size(mjddiff)):
            theta = omega * mjddiff[idx]
            rot_mat = np.array(
                [
                    [np.cos(theta), -1 * np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )

            rottp[idx, :] = rot_mat.dot(tp[idx, :])

        return rottp

    @staticmethod
    def project_onto_plane(tp: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        Takes a list of tangent point vectors and returns their value projected onto a plane specified by normal.
        Note that the projected value returned has the same magnitude as the original input tangent point.  The only
        thing different is the direction
        Parameters
        ----------
        tp: np.ndarray
            (:, 3) array of tangent points
        normal: np.ndarray
            (3,) normal unit vector for the plane

        Returns
        -------
        projected: np.ndarray
            The tangent points projected into the plane, same size as tp
        """
        projected = np.zeros(np.shape(tp))
        for idx, t in enumerate(tp):
            n = np.linalg.norm(t)
            projected[idx, :] = t - np.dot(t, normal) * normal
            projected[idx, :] *= n / np.linalg.norm(projected[idx, :])

        return projected

    def _lat_lon_alts(self, loc):
        geo = sk.Geodetic(self._geoid_mode)

        lats = []
        lons = []
        alts = []
        for lo in loc:
            geo.from_xyz(lo)

            lats.append(geo.latitude)
            lons.append(geo.longitude)
            alts.append(geo.altitude)

        return np.asarray(lats), np.asarray(lons), np.asarray(alts)

    @property
    def altitudes(self):
        """
        Altitudes of the grid in m
        """
        return self._altitudes

    @property
    def numhoriz(self):
        """
        Number of horizontal divisions in the grid
        """
        return len(self._retprofileloc)

    def is_extended(self, angle_idx):
        """
        True if this angle index is outside the range of the input grid points.
        """
        if angle_idx < self._numextend:
            return True
        if angle_idx >= self.numhoriz - self._numextend:
            return True
        return False

    @property
    def numextended(self):
        """
        Number of extended grid points on each side of the grid
        """
        return self._numextend

    @property
    def grid_latitudes(self):
        """
        Latitudes of the grid
        """
        return self._lat_lon_alts(self._retprofileloc)[0]

    @property
    def grid_longitudes(self):
        """
        Longitudes of the grid
        """
        return self._lat_lon_alts(self._retprofileloc)[1]

    @property
    def image_latitudes(self):
        """
        Input image latitudes
        """
        return self._lat_lon_alts(self._averagescantp)[0]

    @property
    def image_longitudes(self):
        """
        Input image longitudes
        """
        return self._lat_lon_alts(self._averagescantp)[1]
