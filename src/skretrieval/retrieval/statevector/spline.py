from __future__ import annotations

from copy import copy

import numpy as np
import xarray as xr
from scipy.interpolate import UnivariateSpline

from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.tikhonov import (
    two_dim_horizontal_second_deriv,
    two_dim_vertical_first_deriv,
)


class MultiplicativeSpline(StateVectorElement):
    def __init__(
        self,
        num_los: int,
        low_wavelength_nm: float,
        high_wavelength_nm: float,
        num_wv: int,
        s: float,
        order=3,
        min_value=0.1,
        max_value=3,
    ):
        self._wv = np.linspace(
            low_wavelength_nm, high_wavelength_nm, num_wv, endpoint=True
        )
        self._x = np.ones((num_los, len(self._wv)))
        self._low_wavelength_nm = low_wavelength_nm
        self._high_wavelength_nm = high_wavelength_nm
        self._s = s
        self._order = order
        self._min_value = min_value
        self._max_value = max_value

    def state(self) -> np.array:
        return self._x.flatten()

    def lower_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._min_value

    def upper_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._max_value

    def update_state(self, x: np.array):
        self._x = x.reshape(self._x.shape)

    def inverse_apriori_covariance(self) -> np.ndarray:
        # TODO: Fix
        gamma = two_dim_horizontal_second_deriv(  # noqa: F841
            self._x.shape[0], self._x.shape[1], factor=0.01
        )
        return np.eye(len(self.state())) * 1e2

    def name(self) -> str:
        return f"spline_{self._low_wavelength_nm}_{self._high_wavelength_nm}"

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        # Calculate the derivative of the spline
        spline_deriv = np.zeros(
            (self._x.shape[0], self._x.shape[1], len(radiance["wavelength"]))
        )

        wv = radiance["wavelength"].to_numpy()
        good = (wv > self._low_wavelength_nm) & (wv < self._high_wavelength_nm)

        for i in range(self._x.shape[0]):
            bx = copy(self._x[i])
            base_spline = UnivariateSpline(self._wv, bx, s=self._s, k=self._order)
            base_vals = base_spline(wv[good])

            for j in range(len(bx)):
                bx[j] += 1e-2
                p_vals = UnivariateSpline(self._wv, bx, s=self._s, k=self._order)(
                    wv[good]
                )
                bx[j] -= 1e-2

                spline_deriv[i, j, good] = (p_vals - base_vals) / 1e-2

        full_deriv = np.zeros(
            (
                spline_deriv.shape[0],
                spline_deriv.shape[1],
                radiance["radiance"].shape[2],
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
            )
        )

        for i in range(self._x.shape[0]):
            full_deriv[i, :, :, :, i] = (
                spline_deriv[i, :, :, np.newaxis]
                * radiance["radiance"].to_numpy()[:, i, :]
            ).transpose([0, 2, 1])

        return xr.DataArray(
            full_deriv.reshape(
                (
                    -1,
                    radiance["radiance"].shape[2],
                    radiance["radiance"].shape[0],
                    radiance["radiance"].shape[1],
                )
            ),
            dims=["x", "stokes", "wavelength", "los"],
        )

    def modify_input_radiance(self, radiance: xr.Dataset):
        wv = radiance["wavelength"].to_numpy()
        vals = np.ones((len(wv), self._x.shape[0]))
        good = (wv > self._low_wavelength_nm) & (wv < self._high_wavelength_nm)

        for i in range(self._x.shape[0]):
            base_spline = UnivariateSpline(
                self._wv, self._x[i], s=self._s, k=self._order
            )
            vals[good, i] = base_spline(wv[good])

        radiance *= xr.DataArray(vals, dims=["wavelength", "los"])

        return radiance

    def apriori_state(self) -> np.array:
        return np.ones_like(self.state())


class AdditiveSpline(StateVectorElement):
    def __init__(
        self,
        num_los: int,
        low_wavelength_nm: float,
        high_wavelength_nm: float,
        num_wv: int,
        s: float,
        order=0,
        min_value=-np.inf,
        max_value=np.inf,
    ):
        self._wv = np.linspace(
            low_wavelength_nm, high_wavelength_nm, num_wv, endpoint=True
        )
        self._x = np.ones((num_los, len(self._wv)))
        self._low_wavelength_nm = low_wavelength_nm
        self._high_wavelength_nm = high_wavelength_nm
        self._s = s
        self._order = order
        self._min_value = min_value
        self._max_value = max_value

    def state(self) -> np.array:
        return self._x.flatten()

    def lower_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._min_value

    def upper_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._max_value

    def update_state(self, x: np.array):
        self._x = x.reshape(self._x.shape)

    def inverse_apriori_covariance(self) -> np.ndarray:
        # TODO: Fix
        gamma = two_dim_vertical_first_deriv(  # noqa: F841
            self._x.shape[0], self._x.shape[1], factor=1e6
        )
        return np.eye(len(self.state())) * 1e-20

    def name(self) -> str:
        return f"add_spline_{self._low_wavelength_nm}_{self._high_wavelength_nm}"

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        # Calculate the derivative of the spline
        spline_deriv = np.zeros(
            (self._x.shape[0], self._x.shape[1], len(radiance["wavelength"]))
        )

        wv = radiance["wavelength"].to_numpy()
        good = (wv > self._low_wavelength_nm) & (wv < self._high_wavelength_nm)

        for i in range(self._x.shape[0]):
            bx = copy(self._x[i])
            base_spline = UnivariateSpline(self._wv, bx, s=self._s, k=self._order)
            base_vals = base_spline(wv[good])

            for j in range(len(bx)):
                bx[j] += 1e-2
                p_vals = UnivariateSpline(self._wv, bx, s=self._s, k=self._order)(
                    wv[good]
                )
                bx[j] -= 1e-2

                spline_deriv[i, j, good] = (p_vals - base_vals) / 1e-2

        full_deriv = np.zeros(
            (
                spline_deriv.shape[0],
                spline_deriv.shape[1],
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )

        for i in range(self._x.shape[0]):
            full_deriv[i, :, :, :, i] = spline_deriv[i, :, np.newaxis, :]

        return xr.DataArray(
            full_deriv.reshape(
                (
                    -1,
                    radiance["radiance"].shape[0],
                    radiance["radiance"].shape[1],
                    radiance["radiance"].shape[2],
                )
            ),
            dims=["x", "stokes", "wavelength", "los"],
        )

    def modify_input_radiance(self, radiance: xr.Dataset):
        wv = radiance["wavelength"].to_numpy()
        vals = np.ones((len(wv), self._x.shape[0]))
        good = (wv > self._low_wavelength_nm) & (wv < self._high_wavelength_nm)

        for i in range(self._x.shape[0]):
            base_spline = UnivariateSpline(
                self._wv, self._x[i], s=self._s, k=self._order
            )
            vals[good, i] = base_spline(wv[good])

        radiance["radiance"] += xr.DataArray(vals, dims=["wavelength", "los"])

        return radiance

    def apriori_state(self) -> np.array:
        return np.zeros_like(self.state())
