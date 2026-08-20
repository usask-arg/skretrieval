from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

from skretrieval.core.radianceformat import (
    VJPContribution,
    evaluate_vjp_contributions,
)


class SASKTRANRadiance:
    def __init__(self, ds: xr.Dataset, collapse_scalar_stokes: bool = True) -> None:
        """
        Handles conversions between Sasktran2 and Sasktran legacy radiance formats, as well as potentially
        other RTM radiance output formats.

        The base SASKTRAN radiance container contains the following variables:

        - radiance: containing dimensions [stokes, los, spectral_grid]

        And the following coordinates:

        - wavelength_nm: wavelength in nm, corresponding to the spectral_grid dimension
        - wavenumber_cminv: wavenumber in cm^-1, corresponding to the spectral_grid dimension

        The "stokes" dimension can either be scalar, or contain any number of the parameters
        ["I", "Q", "U", "V"] for polarized calculations

        These are the minimum required variables and coordinates, others may be added in as needed.

        Weighting functions are included in the dataset as separate variables prefixed by wf_.
        For example wf_ozone_vmr.  These variables must have the dimensions

        - wf: [stokes, los, spectral_grid, ...]

        where the ... may be any number of dimensions and could be different for each wf.

        If collapse_scalar_stokes is set to true, upon construction, the Stokes dimension will be collapsed
        if it is 1 element long.  This is the default behavior.

        Parameters
        ----------
        ds : xr.Dataset

        collapse_scalar_stokes : bool
            If true, then if a scalar calculation is performed the stokes dimension will be removed. By default True
        """
        self._data = ds

        self._validate_ds()

        if collapse_scalar_stokes and self.data.stokes.size == 1:
            self._data = self.data.squeeze("stokes")

    @property
    def data(self) -> xr.Dataset:
        return self._data

    @classmethod
    def from_sasktran_legacy_xr(
        cls, sasktran_legacy_radiance: xr.Dataset
    ) -> SASKTRANRadiance:
        # Check if we have polarized radiances or unpolarized
        if "radiance" in sasktran_legacy_radiance:
            # Scalar
            sasktran_legacy_radiance["radiance"] = sasktran_legacy_radiance[
                "radiance"
            ].expand_dims("stokes")

            for var in list(sasktran_legacy_radiance.variables):
                if var.startswith("wf_"):
                    sasktran_legacy_radiance[var] = sasktran_legacy_radiance[
                        var
                    ].expand_dims("stokes")
        else:
            # Polarized
            sasktran_legacy_radiance["radiance"] = xr.concat(
                [
                    sasktran_legacy_radiance["I"],
                    sasktran_legacy_radiance["Q"],
                    sasktran_legacy_radiance["U"],
                    sasktran_legacy_radiance["V"],
                ],
                dim="stokes",
            )

            sasktran_legacy_radiance = sasktran_legacy_radiance.drop(
                ["I", "Q", "U", "V"]
            )

        if "wf_brdf" in sasktran_legacy_radiance:
            sasktran_legacy_radiance = sasktran_legacy_radiance.drop("wf_brdf")

        sasktran_legacy_radiance = sasktran_legacy_radiance.rename(
            {"wavelength": "wavelength_nm"}
        )
        sasktran_legacy_radiance = sasktran_legacy_radiance.swap_dims(
            {"wavelength_nm": "spectral_grid"}
        )
        sasktran_legacy_radiance.coords["wavenumber_cminv"] = (
            1e7 / sasktran_legacy_radiance["wavelength_nm"]
        )

        return cls(sasktran_legacy_radiance)

    @classmethod
    def from_sasktran2(
        cls, sasktran2_radiance: xr.Dataset, collapse_scalar_stokes=False
    ) -> SASKTRANRadiance:
        sasktran2_radiance = sasktran2_radiance.rename({"wavelength": "wavelength_nm"})
        sasktran2_radiance = sasktran2_radiance.swap_dims(
            {"wavelength_nm": "spectral_grid"}
        )
        sasktran2_radiance.coords["wavenumber_cminv"] = (
            1e7 / sasktran2_radiance["wavelength_nm"]
        )
        return cls(sasktran2_radiance, collapse_scalar_stokes=collapse_scalar_stokes)

    def _validate_ds(self):
        assert "radiance" in self.data.variables
        assert "wavelength_nm" in self.data.coords
        assert "wavenumber_cminv" in self.data.coords

        assert "stokes" in self.data.radiance.dims
        assert "los" in self.data.radiance.dims
        assert "spectral_grid" in self.data.radiance.dims

        for var in list(self.data.variables):
            if var.startswith("wf_"):
                assert "stokes" in self.data[var].dims
                assert "los" in self.data[var].dims
                assert "spectral_grid" in self.data[var].dims


class LinearizedSASKTRANRadiance(SASKTRANRadiance):
    """SASKTRAN radiance backed by a SASKTRAN2 linearization object."""

    def __init__(
        self,
        ds: xr.Dataset,
        matvec: Callable[[np.ndarray], xr.DataArray],
        rmatvec: Callable[[xr.DataArray], np.ndarray],
        n_state: int,
        sk2_radiance_template: xr.DataArray,
    ) -> None:
        super().__init__(ds, collapse_scalar_stokes=False)
        self._matvec = matvec
        self._rmatvec = rmatvec
        self._n_state = n_state
        self._sk2_radiance_template = sk2_radiance_template

    @classmethod
    def from_sasktran2_linearization(
        cls,
        radiance: xr.DataArray,
        matvec: Callable[[np.ndarray], xr.DataArray],
        rmatvec: Callable[[xr.DataArray], np.ndarray],
        n_state: int,
    ) -> LinearizedSASKTRANRadiance:
        converted = SASKTRANRadiance.from_sasktran2(
            xr.Dataset({"radiance": radiance}), collapse_scalar_stokes=False
        )
        return cls(
            converted.data,
            matvec,
            rmatvec,
            n_state,
            sk2_radiance_template=radiance,
        )

    @property
    def n_state(self) -> int:
        return self._n_state

    @property
    def vjp_depth(self) -> int:
        return 0

    def jvp(self, x: np.ndarray) -> xr.DataArray:
        result = self._matvec(x)
        converted = SASKTRANRadiance.from_sasktran2(
            xr.Dataset({"radiance": result}), collapse_scalar_stokes=False
        )
        return converted.data["radiance"]

    def _as_sasktran2_cotangent(self, cotangent: np.ndarray) -> xr.DataArray:
        radiance = self.data["radiance"]
        template = self._sk2_radiance_template
        return xr.DataArray(
            np.asarray(cotangent).reshape(radiance.shape).reshape(template.shape),
            dims=template.dims,
            coords=template.coords,
        )

    def vjp_contributions(self, cotangent: xr.DataArray) -> list[VJPContribution]:
        cotangent = np.asarray(cotangent).reshape(self.data["radiance"].shape)
        return [VJPContribution(self, cotangent, self._continue_vjp)]

    def _continue_vjp(self, cotangent: np.ndarray) -> np.ndarray:
        return self._rmatvec(self._as_sasktran2_cotangent(cotangent))

    def vjp(self, cotangent: xr.DataArray) -> np.ndarray:
        return evaluate_vjp_contributions(self.vjp_contributions(cotangent))
