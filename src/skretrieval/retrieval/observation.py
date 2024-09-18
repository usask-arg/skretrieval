from __future__ import annotations

import abc
from copy import copy
from fnmatch import fnmatch

import numpy as np
import sasktran2 as sk

from skretrieval.core.radianceformat import RadianceGridded


class Observation:
    """
    Abstract base class which defines the interface for an observation
    """

    @abc.abstractmethod
    def sk2_geometry(self, **kwargs) -> dict[sk.ViewingGeometry]:
        """
        The "Ideal" viewing geometry for the observation. One viewing ray for every
        line of sight of the instrument

        Returns
        -------
        dict[sk.ViewingGeometry]
            _description_
        """

    @abc.abstractmethod
    def skretrieval_l1(self, **kwargs) -> dict[RadianceGridded]:
        """
        The L1 data for the observation in the "Core Radiance Format"

        Returns
        -------
        dict[RadianceGridded]
        """

    @abc.abstractmethod
    def sample_wavelengths(self) -> dict[np.array]:
        """
        The sample wavelengths for the observation in [nm]

        Returns
        -------
        dict[np.array]
        """

    @abc.abstractmethod
    def reference_cos_sza(self) -> dict[float]:
        """
        The reference cosine of the solar zenith angle for the observation

        Returns
        -------
        dict[float]
        """

    @abc.abstractmethod
    def reference_latitude(self) -> dict[float]:
        """
        The reference latitude for the observation

        Returns
        -------
        dict[float]
        """

    @abc.abstractmethod
    def reference_longitude(self) -> dict[float]:
        """
        The reference longitude for the observation

        Returns
        -------
        dict[float]
        """

    def append_information_to_l1(self, l1: dict[RadianceGridded], **kwargs) -> None:
        """
        A method that allows for the observation to append information to the L1 data
        simulated by the forward model. Useful for adding things that are in the real L1 data
        to the simulations that may be useful inside the measurement vector.

        Parameters
        ----------
        l1 : dict[RadianceGridded]
        """

    def __add__(self, other: Observation) -> Observation:
        return CombinedObservation(self, other)


class FilteredObservation(Observation):
    def __init__(self, obs: Observation, filter: str):
        """
        An internal class that allows for filtering of observation data.
        Only data that matches the filter will be passed through
        the various methods

        Parameters
        ----------
        obs : Observation
        filter : str
            A string that can be used with fnmatch to filter the keys
        """
        self._obs = obs
        self._filter = filter

    def sk2_geometry(self, **kwargs) -> dict[sk.ViewingGeometry]:
        return {
            k: v
            for k, v in self._obs.sk2_geometry(**kwargs).items()
            if fnmatch(k, self._filter)
        }

    def skretrieval_l1(self, **kwargs) -> dict[RadianceGridded]:
        return {
            k: v
            for k, v in self._obs.skretrieval_l1(**kwargs).items()
            if fnmatch(k, self._filter)
        }

    def sample_wavelengths(self) -> dict[np.array]:
        return {
            k: v
            for k, v in self._obs.sample_wavelengths().items()
            if fnmatch(k, self._filter)
        }

    def reference_cos_sza(self) -> dict[float]:
        return {
            k: v
            for k, v in self._obs.reference_cos_sza().items()
            if fnmatch(k, self._filter)
        }

    def reference_latitude(self) -> dict[float]:
        return {
            k: v
            for k, v in self._obs.reference_latitude().items()
            if fnmatch(k, self._filter)
        }

    def reference_longitude(self) -> dict[float]:
        return {
            k: v
            for k, v in self._obs.reference_longitude().items()
            if fnmatch(k, self._filter)
        }

    def append_information_to_l1(self, l1: dict[RadianceGridded], **kwargs) -> None:
        self._obs.append_information_to_l1(l1, **kwargs)


class CombinedObservation(Observation):
    def __init__(self, obs1: Observation, obs2: Observation):
        """
        An internal class that allows for combining two observations into a single
        observation.

        Parameters
        ----------
        obs1 : Observation
        obs2 : Observation
        """
        self._obs1 = obs1
        self._obs2 = obs2

    def sk2_geometry(self, **kwargs) -> dict[sk.ViewingGeometry]:
        return {
            **self._obs1.sk2_geometry(**kwargs),
            **self._obs2.sk2_geometry(**kwargs),
        }

    def skretrieval_l1(self, *args, **kwargs) -> dict[RadianceGridded]:
        return {
            **self._obs1.skretrieval_l1(*args, **kwargs),
            **self._obs2.skretrieval_l1(*args, **kwargs),
        }

    def sample_wavelengths(self) -> dict[np.array]:
        return {**self._obs1.sample_wavelengths(), **self._obs2.sample_wavelengths()}

    def reference_cos_sza(self) -> dict[float]:
        return {**self._obs1.reference_cos_sza(), **self._obs2.reference_cos_sza()}

    def reference_latitude(self) -> dict[float]:
        return {**self._obs1.reference_latitude(), **self._obs2.reference_latitude()}

    def reference_longitude(self) -> dict[float]:
        return {**self._obs1.reference_longitude(), **self._obs2.reference_longitude()}

    def append_information_to_l1(self, l1: dict[RadianceGridded], **kwargs) -> None:
        self._obs1.append_information_to_l1(l1, **kwargs)
        self._obs2.append_information_to_l1(l1, **kwargs)


class SimulatedObservation(Observation):
    def __init__(
        self,
        geo: sk.ViewingGeometry,
        sample_wavelengths: np.array,
        name="measurement",
        state_adjustment_factors=None,
    ):
        """
        Common funnctionality for ideal simulated observations that consist of a single
        RadianceGridded object

        Parameters
        ----------
        geo : sk.ViewingGeometry
        sample_wavelengths : np.array
        name : str, optional
            Name to use in the measurement dectionaries, by default "measurement"
        state_adjustment_factors : _type_, optional
            Factors {key: val} where state elements indexed by key will be multiplied by val inside the simulation, by default None
        """
        self._geo = geo
        self._sample_wavelengths = sample_wavelengths
        self._name = name
        if state_adjustment_factors is None:
            self._state_adjustment_factors = {}
        else:
            self._state_adjustment_factors = state_adjustment_factors

    def sk2_geometry(self, **kwargs) -> dict[sk.ViewingGeometry]:
        return {self._name: self._geo}

    def _append_information_to_l1(self, l1: dict[RadianceGridded], **kwargs) -> None:
        pass

    def skretrieval_l1(
        self, forward_model, state_vector, l1_kwargs, **kwargs  # noqa: ARG002
    ) -> dict[RadianceGridded]:
        old_x = {}
        for k, v in self._state_adjustment_factors.items():
            old_x[k] = copy(state_vector.sv[k].state())
            if isinstance(v, dict):
                state_vector.sv[k].adjust_constituent_attributes(**v)
            else:
                state_vector.sv[k].update_state(state_vector.sv[k].state() * v)

        l1 = forward_model.calculate_radiance()
        self._append_noise_to_l1(l1)

        for k, _ in self._state_adjustment_factors.items():
            state_vector.sv[k].update_state(old_x[k])

        return l1

    def sample_wavelengths(self) -> np.array:
        return {self._name: self._sample_wavelengths}

    def _append_noise_to_l1(self, l1: dict[RadianceGridded]) -> None:
        for _, v in l1.items():
            v.data["radiance_noise"] = v.data["radiance"] * 0.01


class SimulatedNadirObservation(SimulatedObservation):
    def __init__(
        self,
        cos_sza: float,
        cos_viewing_zenith: float,
        reference_latitude: float,
        reference_longitude: float,
        sample_wavelengths: np.array,
        name: str = "measurement",
        state_adjustment_factors=None,
        noise_fn=None,
    ):
        """
        A simulated nadir observation

        Parameters
        ----------
        cos_sza : float
            Cosine of the solar zenith angle at the ground
        cos_viewing_zenith : float
            Cosone of the viewing zenith angle, 1.0 for pure nadir viewing
        reference_latitude : float
            Latitude of the ground point
        reference_longitude : float
            Longitude of the ground point
        sample_wavelengths : np.array
        name : str, optional
            , by default "measurement"
        state_adjustment_factors : _type_, optional
            , by default None
        """
        self._cos_sza = cos_sza
        self._cos_viewing_zenith = cos_viewing_zenith
        self._reference_latitude = reference_latitude
        self._reference_longitude = reference_longitude
        self._noise_fn = noise_fn

        geo = sk.ViewingGeometry()
        geo.add_ray(sk.GroundViewingSolar(cos_sza, 0, cos_viewing_zenith, 200000))

        super().__init__(
            geo,
            sample_wavelengths,
            name,
            state_adjustment_factors=state_adjustment_factors,
        )

    def reference_cos_sza(self) -> float:
        return {self._name: self._cos_sza}

    def reference_latitude(self) -> float:
        return {self._name: self._reference_latitude}

    def reference_longitude(self) -> float:
        return {self._name: self._reference_longitude}

    def _append_noise_to_l1(self, l1: dict[RadianceGridded]) -> None:
        if self._noise_fn is not None:
            for _, v in l1.items():
                v.data["radiance_noise"] = self._noise_fn(v.data)
        else:
            super()._append_noise_to_l1(l1)


class SimulatedLimbObservation(SimulatedObservation):
    def __init__(
        self,
        cos_sza: float,
        relative_azimuth: float,
        observer_altitude: float,
        reference_latitude: float,
        reference_longitude: float,
        tangent_altitudes: np.array,
        sample_wavelengths: np.array,
        name: str = "measurement",
        state_adjustment_factors=None,
        noise_fn=None,
    ):
        """
        A simulated limb observation

        Parameters
        ----------
        cos_sza : float
            cos of the solar zenith angle for all lines of sight
        relative_azimuth : float
            Relative azimuth angle for all lines of sight
        observer_altitude : float
            Altitude of the observer
        reference_latitude : float
        reference_longitude : float
        tangent_altitudes : np.array
            Tangent altitudes in [m]
        sample_wavelengths : np.array
            Sample wavelengths in [nm]
        name : str, optional
        state_adjustment_factors : _type_, optional
        """
        self._cos_sza = cos_sza
        self._relative_azimuth = relative_azimuth
        self._observer_altitude = observer_altitude
        self._reference_latitude = reference_latitude
        self._reference_longitude = reference_longitude
        self._tan_alts = tangent_altitudes
        self._noise_fn = noise_fn

        geo = sk.ViewingGeometry()
        for tan_alt in tangent_altitudes:
            geo.add_ray(
                sk.TangentAltitudeSolar(
                    tan_alt, relative_azimuth, observer_altitude, cos_sza
                )
            )

        super().__init__(
            geo,
            sample_wavelengths,
            name,
            state_adjustment_factors=state_adjustment_factors,
        )

    def reference_cos_sza(self) -> float:
        return {self._name: self._cos_sza}

    def reference_latitude(self) -> float:
        return {self._name: self._reference_latitude}

    def reference_longitude(self) -> float:
        return {self._name: self._reference_longitude}

    def append_information_to_l1(self, l1: dict[RadianceGridded], **kwargs) -> None:
        if self._name in l1:
            l1[self._name].data.coords["tangent_altitude"] = (["los"], self._tan_alts)

            l1[self._name].data = l1[self._name].data.set_xindex("tangent_altitude")

    def _append_noise_to_l1(self, l1: dict[RadianceGridded]) -> None:
        if self._noise_fn is not None:
            for _, v in l1.items():
                v.data["radiance_noise"] = self._noise_fn(v.data)
        else:
            super()._append_noise_to_l1(l1)
