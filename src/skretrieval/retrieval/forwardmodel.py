from __future__ import annotations

import abc
from typing import Callable

import numpy as np
import sasktran2 as sk

from skretrieval.core.lineshape import DeltaFunction, LineShape
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.core.sensor.spectrograph import SpectrographOnlySpectral
from skretrieval.geodetic import geodetic
from skretrieval.retrieval import ForwardModel

from .ancillary import Ancillary
from .measvec import MeasurementVector
from .observation import FilteredObservation, Observation
from .statevector.altitude import AltitudeNativeStateVector


class StandardForwardModel(ForwardModel):
    def __init__(
        self,
        observation: Observation,
        state_vector: AltitudeNativeStateVector,
        meas_vec: MeasurementVector,
        ancillary: Ancillary,
        engine_config: sk.Config,
        **kwargs,
    ) -> None:
        """
        A forward model for the Retrieval class.  This is a base class that should be inherited from.


        Parameters
        ----------
        observation : Observation
            The observation
        state_vector : AltitudeNativeStateVector
            The State Vector
        ancillary : Ancillary
            The Ancillary Object
        engine_config : sk.Config
            Configuration for the engine
        """
        ForwardModel.__init__(self)

        self._state_vector = state_vector
        self._engine_config = engine_config
        self._ancillary = ancillary
        self._observation = observation
        self._meas_vec = meas_vec

        self._viewing_geo = self._construct_viewing_geo()
        self._model_geometry = self._construct_model_geometry()
        self._model_wavelength = self._construct_model_wavelength()

        self._atmosphere = self._construct_atmosphere()

        self._engine = self._construct_engine()

        self._inst_model = self._construct_inst_model()

        self._solar_model = self._construct_solar_model()

    @abc.abstractmethod
    def _construct_model_geometry(self):
        pass

    @abc.abstractmethod
    def _construct_model_wavelength(self):
        pass

    @abc.abstractmethod
    def _construct_viewing_geo(self):
        pass

    @abc.abstractmethod
    def _construct_inst_model(self):
        pass

    def _construct_solar_model(self):
        pass

    def _construct_atmosphere(self):
        atmo = {}

        for key in self._model_geometry:
            atmo[key] = sk.Atmosphere(
                self._model_geometry[key],
                self._engine_config,
                wavelengths_nm=self._model_wavelength[key],
                pressure_derivative=False,
                temperature_derivative=False,
            )

            self._state_vector.add_to_atmosphere(atmo[key])
            self._ancillary.add_to_atmosphere(atmo[key])

        return atmo

    def _construct_engine(self):
        engines = {}

        for key in self._model_geometry:
            engines[key] = sk.Engine(
                self._engine_config, self._model_geometry[key], self._viewing_geo[key]
            )

        return engines

    def calculate_radiance(self):
        l1 = {}
        for key in self._engine:
            sk2_rad = self._engine[key].calculate_radiance(self._atmosphere[key])
            sk2_rad = self._state_vector.post_process_sk2_radiances(sk2_rad)
            sk2_rad = SASKTRANRadiance.from_sasktran2(sk2_rad)

            model_result = self._inst_model[key].model_radiance(sk2_rad, None)

            if isinstance(model_result, dict):
                if len(model_result) == 1:
                    l1[key] = next(iter(model_result.values()))

                for k, v in model_result.items():
                    l1[f"{key}_{k}"] = v
            else:
                l1[key] = model_result

            self._observation.append_information_to_l1(l1)

        return l1


class SpectrometerMixin:
    def __init__(
        self,
        lineshape_fn: Callable[[float], LineShape] | None = None,
        model_res_nm=0.02,
        model_res_cminv=0.02,
        spectral_native_coordinate="wavelength_nm",
        round_decimal=2,
        stokes_sensitivities=None,
    ) -> None:
        """
        Mixin for adding a spectrometer to the forward model

        Parameters
        ----------
        lineshape_fn : Callable[[float], LineShape] | None, optional
            Function that takes in wavelength in nm and returns back a LineShape, by default None
        model_res_nm : float, optional
            Model Resolution to use in [nm], by default 0.02
        model_res_cminv : float, optional
            Model Resolution to use in [cm^-1], by default 0.02
        spectral_native_coordinate : str, optional
            The native coordinate for the spectral axis, by default "wavelength_nm", can also be
            "wavenumber_cminv"
        round_decimal : int, optional
            Decimal points to round the wavelengths to in the radiative transfer calculation, by default 2
        stokes_sensitivities : dict, optional
            Dictionary of stokes sensitivities, by default None. Can be set to multiple measurements, e.g.,
            {"I": np.array([1, 0, 0, 0]), "Q": np.array([0, 1, 0, 0]), "U": np.array([0, 0, 1, 0])} to measure
            multiple stokes parameters separately.
        """
        if lineshape_fn is None:
            self._lineshape_fn = lambda _: DeltaFunction()
        else:
            self._lineshape_fn = lineshape_fn

        self._model_res_nm = model_res_nm
        self._model_res_cminv = model_res_cminv
        self._round_decimal = round_decimal
        self._spectral_native_coordinate = spectral_native_coordinate
        self._stokes_sensitivities = stokes_sensitivities

    def _get_required_wavelength(self):
        obs_samples = self._observation.sample_wavelengths()
        mv_required_samples = {}
        for key, val in self._meas_vec.items():
            mv_required_samples[key] = val.required_sample_wavelengths(obs_samples)

        sample_wavelengths = {}
        for key in obs_samples:
            sample_wavelengths[key] = np.unique(
                np.concatenate([d[key] for d in mv_required_samples.values()])
            )

        return sample_wavelengths

    def _construct_model_wavelength(self):
        """
        Evaluates the lineshape at the sample wavelengths and returns back the model wavelengths
        spaced by the model resolution
        """
        sample_wavelengths = self._get_required_wavelength()

        ws = {}
        for k, v in sample_wavelengths.items():
            if self._spectral_native_coordinate == "wavelength_nm":
                bounds = [
                    self._lineshape_fn(w).bounds(center=0)
                    if self._lineshape_fn(w).zero_centered()
                    else self._lineshape_fn(w).bounds(center=-w)
                    for w in v
                ]
                ws[k] = np.unique(
                    np.concatenate(
                        [
                            np.around(
                                np.arange(
                                    a, b + self._model_res_nm / 2, self._model_res_nm
                                )
                                + w,
                                self._round_decimal,
                            )
                            for (a, b), w in zip(bounds, v)
                        ]
                    )
                )
            else:
                bounds = [
                    self._lineshape_fn(w).bounds(center=0)
                    if self._lineshape_fn(w).zero_centered()
                    else self._lineshape_fn(w).bounds(center=-1e7 / w)
                    for w in v
                ]
                ws[k] = (
                    1e7
                    / np.unique(
                        np.concatenate(
                            [
                                np.around(
                                    np.arange(
                                        1e7 / (b + w),
                                        1e7 / (a + w) + self._model_res_cminv / 2,
                                        self._model_res_cminv,
                                    ),
                                    self._round_decimal,
                                )
                                for (a, b), w in zip(bounds, v)
                            ]
                        )
                    )[::-1]
                )

        return ws

    def _construct_inst_model(self):
        """
        Constructs the instrument model
        """
        sample_wavelengths = self._get_required_wavelength()

        inst_models = {}

        for key in sample_wavelengths:
            inst_models[key] = SpectrographOnlySpectral(
                sample_wavelengths[key],
                [self._lineshape_fn(x) for x in sample_wavelengths[key]],
                spectral_native_coordinate=self._spectral_native_coordinate,
                assign_coord="wavelength"
                if self._spectral_native_coordinate == "wavelength_nm"
                else "wavenumber",
                stokes_sensitivity=self._stokes_sensitivities,
            )

        return inst_models


class IdealViewingMixin:
    def __init__(self, observation: Observation, model_altitude_grid: np.array) -> None:
        """
        Mixin for adding an ideal viewing geometry to the forward model. This means
        that a single line of sight is used by the forward model for each observation rather
        than using a spatial PSF.

        Parameters
        ----------
        observation : Observation
        model_altitude_grid : np.array
        """
        self._model_altitude_grid = model_altitude_grid
        self._obs = observation

    def _construct_viewing_geo(self):
        return self._obs.sk2_geometry()

    def _construct_model_geometry(self):
        # Construct the model geometry

        # State vector tells us the engine altitude grid
        altitude_grid_m = self._model_altitude_grid

        # Observation tells us the reference point
        cos_sza = self._obs.reference_cos_sza()
        ref_lat = self._obs.reference_latitude()
        ref_lon = self._obs.reference_longitude()

        geometry = {}

        geo = geodetic()
        for key in cos_sza:
            geo.from_lat_lon_alt(ref_lat[key], ref_lon[key], 0.0)
            earth_radius_m = np.linalg.norm(geo.location)
            geometry[key] = sk.Geometry1D(
                cos_sza[key],
                0.0,
                earth_radius_m,
                altitude_grid_m,
                sk.InterpolationMethod.LinearInterpolation,
                sk.GeometryType.Spherical,
            )

        return geometry


class IdealViewingSpectrograph(
    IdealViewingMixin, SpectrometerMixin, StandardForwardModel
):
    def __init__(
        self,
        observation: Observation,
        state_vector: AltitudeNativeStateVector,
        meas_vec: MeasurementVector,
        ancillary: Ancillary,
        engine_config: sk.Config,
        **kwargs,
    ) -> None:
        """
        A forward model for the retrieval that uses an ideal viewing geometry and a spectrometer

        Parameters
        ----------
        observation : Observation
        state_vector : AltitudeNativeStateVector
        ancillary : Ancillary
        engine_config : sk.Config
        kwargs : dict
            Additional arguments passed to `SpectrometerMixin`
        """
        IdealViewingMixin.__init__(self, observation, state_vector.altitude_grid)
        SpectrometerMixin.__init__(
            self,
            lineshape_fn=kwargs.get("lineshape_fn", lambda _: DeltaFunction()),
            model_res_cminv=kwargs.get("model_res_cminv", 0.02),
            model_res_nm=kwargs.get("model_res_nm", 0.02),
            spectral_native_coordinate=kwargs.get(
                "spectral_native_coordinate", "wavelength_nm"
            ),
            stokes_sensitivities=kwargs.get("stokes_sensitivities"),
        )
        StandardForwardModel.__init__(
            self,
            observation,
            state_vector,
            meas_vec,
            ancillary,
            engine_config,
            **kwargs,
        )


class ForwardModelHandler(ForwardModel):
    def __init__(
        self,
        cfg: dict,
        observation: Observation,
        state_vector: AltitudeNativeStateVector,
        meas_vec: MeasurementVector,
        ancillary: Ancillary,
        engine_config: sk.Config,
        **kwargs,
    ):
        super().__init__()

        # Construct the internal forward models
        self._forward_models = {}
        for key, val in cfg.items():
            self._forward_models[key] = val.get("class", IdealViewingSpectrograph)(
                FilteredObservation(observation, key),
                state_vector,
                meas_vec,
                ancillary,
                engine_config,
                **val.get("kwargs", {}),
            )

    def calculate_radiance(self):
        result = {}
        for _, v in self._forward_models.items():
            result.update(v.calculate_radiance())
        return result
