from __future__ import annotations

from copy import copy

import numpy as np
import xarray as xr

from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.tikhonov import (
    two_dim_vertical_first_deriv,
)


class WavenumberShift(StateVectorElement):
    def __init__(
        self,
        num_los: int,
        tikh_factor: float,
        prior_factor: float = 0,
        numerical_delta=0.0001,
        min_shift=-0.1,
        max_shift=0.1,
        apply_to_measurement: str | None = None,
    ):
        """
        Implements a wavenumber shift for every modelled LOS in the forward model

        Parameters
        ----------
        num_los : int
        numerical_delta : float, optional
            _description_, by default 0.0001
        """
        self._shifts = np.zeros(num_los)
        self._numerical_delta = numerical_delta
        self._tikh_factor = tikh_factor
        self._prior_factor = prior_factor

        self._min_shift = min_shift
        self._max_shift = max_shift
        self._apply_to_measurement = apply_to_measurement

        super().__init__(True)

    def state(self) -> np.array:
        return copy(self._shifts)

    def name(self) -> str:
        if self._apply_to_measurement:
            return f"wavenumber_shifts_{self._apply_to_measurement}"
        return "wavenumber_shifts"

    def lower_bound(self) -> np.array:
        return np.ones_like(self._shifts.flatten()) * self._min_shift

    def upper_bound(self) -> np.array:
        return np.ones_like(self._shifts.flatten()) * self._max_shift

    def inverse_apriori_covariance(self) -> np.ndarray:
        gamma = two_dim_vertical_first_deriv(
            1, len(self._shifts), factor=self._tikh_factor
        )
        return gamma.T @ gamma + np.eye(len(self._shifts)) * self._prior_factor

    def apriori_state(self) -> np.array:
        return np.zeros_like(self.state())

    def propagate_wf(self, radiance) -> np.ndarray:
        wf = np.zeros(
            (
                len(self._shifts),
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )

        for i in range(len(self._shifts)):
            new_rad = radiance.isel(los=i, stokes=0)["radiance"].interp(
                wavelength=radiance.wavelength + self._numerical_delta,
                kwargs={"fill_value": "extrapolate"},
            )

            drad = (
                new_rad - radiance.isel(los=i, stokes=0)["radiance"]
            ) / self._numerical_delta
            wf[i, :, i, 0] = drad
        return xr.DataArray(wf, dims=["x", "wavelength", "los", "stokes"])

    def update_state(self, x: np.array):
        self._shifts = copy(x)

    def modify_input_radiance(self, radiance: xr.Dataset):
        for i in range(len(self._shifts)):
            shift = self._shifts[i]
            for var in list(radiance):
                if "wavelength" in radiance[var].dims:
                    if len(radiance[var].dims) == 3:
                        radiance[var].to_numpy()[:, i, 0] = radiance.isel(
                            los=i, stokes=0
                        )[var].interp(
                            wavelength=radiance.wavelength + shift,
                            kwargs={"fill_value": "extrapolate"},
                        )
                    else:
                        radiance[var].to_numpy()[:, :, i, 0] = radiance.isel(
                            los=i, stokes=0
                        )[var].interp(
                            wavelength=radiance.wavelength + shift,
                            kwargs={"fill_value": "extrapolate"},
                        )

        return radiance
