from __future__ import annotations

from copy import copy
from typing import Callable

import numpy as np
import sasktran as sk
import sasktranif.sasktranif as skif
from sasktran.exceptions import wrap_skif_functionfail

from skretrieval.core.lineshape import LineShape


class OpticalPropertyLineShape(sk.OpticalProperty):
    """
    An optical property that is convolved with an arbitrary line shape
    """

    def __init__(
        self,
        optical_property: sk.OpticalProperty,
        line_shape_fn: Callable[[float], LineShape],
        wavel_to_calc_for=None,
    ):
        super().__init__(optical_property._name)

        self._unconv_optical_prop = optical_property
        self._line_shape_fn = line_shape_fn
        self._valid_wavelengths = []
        self._wavel_to_calc_for = wavel_to_calc_for

        if self._wavel_to_calc_for is not None:
            self._iskopticalproperty = self.convolved_optical_property(
                self._unconv_optical_prop, self._line_shape_fn, self._wavel_to_calc_for
            )

    def skif_object(self, **kwargs):
        if len(self._valid_wavelengths) != len(
            kwargs["engine"].wavelengths
        ) or not np.allclose(self._valid_wavelengths, kwargs["engine"].wavelengths):
            self._valid_wavelengths = kwargs["engine"].wavelengths

            calc_wavelengths = copy(self._valid_wavelengths)
            calc_wavelengths = np.concatenate(
                (
                    [np.nanmin(calc_wavelengths) - 1],
                    calc_wavelengths,
                    [np.nanmax(calc_wavelengths) + 1],
                )
            )

            self._iskopticalproperty = self.convolved_optical_property(
                self._unconv_optical_prop, self._line_shape_fn, calc_wavelengths
            )
        return self._iskopticalproperty

    def __setstate__(self, state):
        super().__setstate__(state)
        self._iskopticalproperty = self.convolved_optical_property(
            self._unconv_optical_prop, self._line_shape_fn, self._wavel_to_calc_for
        )

    def __repr__(self):
        return f"SasktranIF Optical Property: {self._name}_Convolved"

    @staticmethod
    @wrap_skif_functionfail
    def convolved_optical_property(
        hires_optprop: sk.OpticalProperty,
        line_shape_fn: Callable[[float], LineShape],
        wavel_to_calc_for,
    ):
        """
        Convolves down the hi-resolution cross sections and creates a user defined optical property. Convolution assumes
        the hi-resolution version has infinite resolution

        Parameters
        ----------
        hires_optprop : sasktran.OpticalProperty
            sasktran OpticalProperty that will be convolved down to desired resolution.
        Returns
        -------
        optprop : ISKOpticalProperty
            Convolved optical property
        """

        import contextlib

        optprop = skif.ISKOpticalProperty("USERDEFINED_TABLES")
        with contextlib.suppress(skif._sasktranif.functionfail):
            optprop.SetProperty("WavelengthTruncation", 1)

        # Temperatures of the hi-resolution xsect
        temp = hires_optprop.info["temperatures"]
        # Ranges in which each temperature has data
        temp_range = hires_optprop.info["wavelength_range"]

        spec_sampling = hires_optprop.info["spectral sampling"] = 0.01

        for T, ra in zip(temp, temp_range):
            # Get the high resolution DBM cross section
            clim_temp = sk.ClimatologyUserDefined(
                [0, 10000], {"SKCLIMATOLOGY_TEMPERATURE_K": [T, T]}, interp="linear"
            )
            wavel_hr = np.arange(ra[0], ra[1], spec_sampling)

            if wavel_to_calc_for is None:
                wavel_calc = wavel_hr
            else:
                wavel_calc = wavel_to_calc_for
                wavel_calc = wavel_calc[(wavel_calc >= ra[0]) & (wavel_calc <= ra[1])]

            xsect = hires_optprop.calculate_cross_sections(
                clim_temp, 0, 0, 1000, 54372, wavel_hr
            )
            xsect = xsect.absorption

            wavelconv = np.zeros(np.shape(wavel_hr))
            xsectconv = np.zeros(np.shape(wavel_hr))

            for widx, w in enumerate(wavel_calc):
                line_shape = line_shape_fn(w)
                weights = line_shape.integration_weights(w, wavel_hr)
                wavelconv[widx] = np.dot(wavel_hr, weights)
                xsectconv[widx] = np.dot(xsect, weights)

            good = xsectconv > 0
            if np.nansum(good) > 0:
                full_xsect = np.interp(
                    wavel_hr,
                    wavelconv[good],
                    xsectconv[good],
                    left=np.nan,
                    right=np.nan,
                )

                optprop.AddUserDefined(
                    T,
                    wavel_hr[~np.isnan(full_xsect)],
                    full_xsect[~np.isnan(full_xsect)],
                )

        return optprop
