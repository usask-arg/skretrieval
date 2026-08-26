from __future__ import annotations

import gc
import operator
from collections.abc import Callable
from types import MappingProxyType

import numpy as np
import sasktran2 as sk2
import xarray as xr

from skretrieval.core.lineshape import DeltaFunction
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.forwardmodel import SpectrometerMixin, StandardForwardModel
from skretrieval.retrieval.processing import Retrieval
from skretrieval.retrieval.statevector.orbital_plane import OrbitalPlaneStateVector


def _split_time_groups(
    viewing_geometry: sk2.OrbitalPlaneViewingGeometry,
    *,
    time_group_duration_s: float,
    max_time_groups_per_engine: int,
) -> tuple[sk2.OrbitalPlaneViewingGeometry, ...]:
    """Split complete native time groups into bounded contiguous chunks."""
    if isinstance(max_time_groups_per_engine, bool | np.bool_):
        msg = "max_time_groups_per_engine must be a positive integer"
        raise TypeError(msg)
    try:
        max_time_groups_per_engine = operator.index(max_time_groups_per_engine)
    except TypeError as error:
        msg = "max_time_groups_per_engine must be a positive integer"
        raise TypeError(msg) from error
    if max_time_groups_per_engine < 1:
        msg = "max_time_groups_per_engine must be positive"
        raise ValueError(msg)

    duration_ns_float = float(time_group_duration_s) * 1.0e9
    if (
        not np.isfinite(duration_ns_float)
        or duration_ns_float < 0.5
        or duration_ns_float > np.iinfo(np.int64).max
    ):
        msg = "time_group_duration_s must be finite, positive, and at least 1 ns"
        raise ValueError(msg)
    duration_ns = int(np.floor(duration_ns_float + 0.5))

    vertical_slice = np.asarray(viewing_geometry.vertical_slice)
    times_ns = (
        np.asarray(viewing_geometry.times).astype("datetime64[ns]").astype(np.int64)
    )
    slice_runs = np.flatnonzero(
        np.r_[True, vertical_slice[1:] != vertical_slice[:-1], True]
    )
    origin_ns = int(
        np.asarray(viewing_geometry.time_bin_origin)
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    slice_bins = []
    for slice_index in range(len(slice_runs) - 1):
        start = int(slice_runs[slice_index])
        stop = int(slice_runs[slice_index + 1])
        reference_time_ns = sum(int(value) for value in times_ns[start:stop]) // (
            stop - start
        )
        slice_bins.append((reference_time_ns - origin_ns) // duration_ns)
    if np.any(np.diff(slice_bins) < 0):
        msg = "Vertical slices must be ordered by non-decreasing reference time"
        raise ValueError(msg)

    group_slice_runs = np.flatnonzero(np.r_[True, np.diff(slice_bins) != 0, True])
    num_groups = len(group_slice_runs) - 1
    group_bins = np.asarray(slice_bins, dtype=np.int64)[group_slice_runs[:-1]]
    contiguous_group_runs = np.flatnonzero(np.r_[True, np.diff(group_bins) > 1, True])
    if num_groups <= max_time_groups_per_engine and len(contiguous_group_runs) == 2:
        return (viewing_geometry,)

    chunks = []
    for run_index in range(len(contiguous_group_runs) - 1):
        run_start = int(contiguous_group_runs[run_index])
        run_stop = int(contiguous_group_runs[run_index + 1])
        for first_group in range(run_start, run_stop, max_time_groups_per_engine):
            final_group = min(first_group + max_time_groups_per_engine, run_stop)
            first_slice = int(group_slice_runs[first_group])
            final_slice = int(group_slice_runs[final_group])
            start = int(slice_runs[first_slice])
            stop = int(slice_runs[final_slice])
            chunks.append(viewing_geometry.isel(slice(start, stop)))
    return tuple(chunks)


def _concat_radiance(values: list[xr.DataArray]) -> xr.DataArray:
    result = xr.concat(values, dim="los")
    return result.assign_coords(los=np.arange(result.sizes["los"]))


def _count_time_groups(
    viewing_geometry: sk2.OrbitalPlaneViewingGeometry,
    time_group_duration_s: float,
) -> int:
    vertical_slice = np.asarray(viewing_geometry.vertical_slice)
    times_ns = (
        np.asarray(viewing_geometry.times).astype("datetime64[ns]").astype(np.int64)
    )
    slice_runs = np.flatnonzero(
        np.r_[True, vertical_slice[1:] != vertical_slice[:-1], True]
    )
    origin_ns = int(
        np.asarray(viewing_geometry.time_bin_origin)
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    duration_ns = int(np.floor(float(time_group_duration_s) * 1.0e9 + 0.5))
    bins = []
    for slice_index in range(len(slice_runs) - 1):
        start = int(slice_runs[slice_index])
        stop = int(slice_runs[slice_index + 1])
        reference_time_ns = sum(int(value) for value in times_ns[start:stop]) // (
            stop - start
        )
        bins.append((reference_time_ns - origin_ns) // duration_ns)
    return 0 if not bins else int(1 + np.count_nonzero(np.diff(bins)))


class _ChunkedOrbitalPlaneLinearization:
    """One linearization assembled from contiguous orbital-engine chunks."""

    def __init__(self, linearizations: list) -> None:
        if not linearizations:
            msg = "At least one orbital linearization is required"
            raise ValueError(msg)
        self._linearizations = tuple(linearizations)
        self._los_sizes = tuple(
            linearization.value.sizes["los"] for linearization in self._linearizations
        )
        self._value = _concat_radiance(
            [linearization.value for linearization in self._linearizations]
        )
        self._tangent_template = self._linearizations[0].tangent_template
        self._backends = MappingProxyType(dict(self._linearizations[0].backends))

        for linearization in self._linearizations[1:]:
            if not linearization.tangent_template.identical(self._tangent_template):
                msg = "Orbital-engine chunks have incompatible parameter grids"
                raise ValueError(msg)
            if dict(linearization.backends) != dict(self._backends):
                msg = "Orbital-engine chunks selected different derivative backends"
                raise ValueError(msg)

    @property
    def value(self) -> xr.DataArray:
        return self._value

    @property
    def tangent_template(self) -> xr.Dataset:
        return self._tangent_template.copy(deep=True)

    @property
    def backends(self):
        return self._backends

    @property
    def jacobian(self) -> xr.Dataset:
        return xr.concat(
            [linearization.jacobian for linearization in self._linearizations],
            dim="los",
        ).assign_coords(los=np.arange(self._value.sizes["los"]))

    def jvp(self, tangent: xr.Dataset) -> xr.DataArray:
        return _concat_radiance(
            [linearization.jvp(tangent) for linearization in self._linearizations]
        )

    def vjp(self, cotangent: xr.DataArray, parameters=None) -> xr.Dataset:
        start = 0
        gradients = []
        for linearization, size in zip(
            self._linearizations, self._los_sizes, strict=True
        ):
            chunk = cotangent.isel(los=slice(start, start + size)).assign_coords(
                los=linearization.value.coords["los"]
            )
            gradients.append(linearization.vjp(chunk, parameters=parameters))
            start += size
        if not gradients or not gradients[0].data_vars:
            return xr.Dataset()
        return sum(gradients[1:], start=gradients[0])


class _StreamingChunkedOrbitalPlaneLinearization:
    """Rebuild bounded batches of orbital chunks for matrix-free products."""

    def __init__(self, owner, atmosphere, prepare_parameters=None) -> None:
        self._owner = owner
        self._atmosphere = atmosphere
        self._prepare_parameters = prepare_parameters
        self._los_sizes = []
        values = []
        tangent_template = None
        backends = None

        products = self._evaluate_batches(
            lambda _index, linearization: (
                linearization.value.copy(deep=True),
                linearization.tangent_template.copy(deep=True),
                dict(linearization.backends),
            )
        )
        for value, template, selected_backends in products:
            values.append(value)
            self._los_sizes.append(value.sizes["los"])
            if tangent_template is None:
                tangent_template = template
                backends = selected_backends
            else:
                if not template.identical(tangent_template):
                    msg = "Orbital-engine chunks have incompatible parameter grids"
                    raise ValueError(msg)
                if selected_backends != backends:
                    msg = "Orbital-engine chunks selected different derivative backends"
                    raise ValueError(msg)
        if tangent_template is None or backends is None:
            msg = "At least one orbital-engine chunk is required"
            raise ValueError(msg)
        self._value = _concat_radiance(values)
        self._los_sizes = tuple(self._los_sizes)
        self._los_offsets = tuple(
            int(value) for value in np.r_[0, np.cumsum(self._los_sizes)]
        )
        self._tangent_template = tangent_template
        self._backends = MappingProxyType(backends)

    @property
    def value(self) -> xr.DataArray:
        return self._value

    @property
    def tangent_template(self) -> xr.Dataset:
        return self._tangent_template.copy(deep=True)

    @property
    def backends(self):
        return self._backends

    def _evaluate_batches(self, evaluator):
        results = []
        for chunk_indices in self._owner.streaming_chunk_batches:
            internal_atmosphere = self._atmosphere.internal_object()
            try:
                results.extend(
                    self._owner._evaluate_chunk_batch(
                        chunk_indices,
                        self._atmosphere,
                        internal_atmosphere=internal_atmosphere,
                        prepare_parameters=self._prepare_parameters,
                        evaluator=evaluator,
                    )
                )
            finally:
                del internal_atmosphere
                gc.collect()
        return results

    @property
    def jacobian(self) -> xr.Dataset:
        values = self._evaluate_batches(
            lambda _index, linearization: linearization.jacobian.copy(deep=True)
        )
        return xr.concat(values, dim="los").assign_coords(
            los=np.arange(self._value.sizes["los"])
        )

    def jvp(self, tangent: xr.Dataset) -> xr.DataArray:
        values = self._evaluate_batches(
            lambda _index, linearization: linearization.jvp(tangent).copy(deep=True)
        )
        return _concat_radiance(values)

    def vjp(self, cotangent: xr.DataArray, parameters=None) -> xr.Dataset:
        def evaluate(chunk_index, linearization):
            start = self._los_offsets[chunk_index]
            stop = self._los_offsets[chunk_index + 1]
            chunk = cotangent.isel(los=slice(start, stop)).assign_coords(
                los=linearization.value.coords["los"]
            )
            return linearization.vjp(chunk, parameters=parameters).copy(deep=True)

        gradients = self._evaluate_batches(evaluate)
        if not gradients or not gradients[0].data_vars:
            return xr.Dataset()
        return sum(gradients[1:], start=gradients[0])


class _ChunkedOrbitalPlaneEngine:
    """Keep a bounded number of native time groups in each orbital engine."""

    def __init__(
        self,
        config: sk2.Config,
        geometry: sk2.OrbitalPlaneGeometry,
        viewing_geometry: sk2.OrbitalPlaneViewingGeometry,
        *,
        max_time_groups_per_engine: int,
        chunk_execution: str = "resident",
        streaming_chunk_workers: int = 1,
        **engine_kwargs,
    ) -> None:
        if chunk_execution not in {"resident", "streaming"}:
            msg = "chunk_execution must be 'resident' or 'streaming'"
            raise ValueError(msg)
        if isinstance(streaming_chunk_workers, bool | np.bool_):
            msg = "streaming_chunk_workers must be a positive integer"
            raise TypeError(msg)
        try:
            streaming_chunk_workers = operator.index(streaming_chunk_workers)
        except TypeError as error:
            msg = "streaming_chunk_workers must be a positive integer"
            raise TypeError(msg) from error
        if streaming_chunk_workers < 1:
            msg = "streaming_chunk_workers must be positive"
            raise ValueError(msg)
        if streaming_chunk_workers != 1:
            msg = (
                "streaming_chunk_workers must be 1 because SASKTRAN orbital "
                "objects are bound to their constructing Python thread"
            )
            raise ValueError(msg)
        duration_s = engine_kwargs.get("time_group_duration_s", 60.0)
        self._viewing_chunks = _split_time_groups(
            viewing_geometry,
            time_group_duration_s=duration_s,
            max_time_groups_per_engine=max_time_groups_per_engine,
        )
        self._config = config
        self._geometry = geometry
        self._engine_kwargs = dict(engine_kwargs)
        self._chunk_execution = chunk_execution
        self._streaming_chunk_workers = streaming_chunk_workers
        self._engines = (
            tuple(
                sk2.OrbitalPlaneEngine(
                    config,
                    geometry,
                    viewing_chunk,
                    **engine_kwargs,
                )
                for viewing_chunk in self._viewing_chunks
            )
            if chunk_execution == "resident"
            else ()
        )

    @property
    def num_chunks(self) -> int:
        return len(self._viewing_chunks)

    @property
    def streaming_chunk_batches(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                range(
                    start,
                    min(start + self._streaming_chunk_workers, self.num_chunks),
                )
            )
            for start in range(0, self.num_chunks, self._streaming_chunk_workers)
        )

    @property
    def num_groups(self) -> int:
        if self._engines:
            return sum(engine.num_groups for engine in self._engines)
        return sum(
            _count_time_groups(
                chunk,
                self._engine_kwargs.get("time_group_duration_s", 60.0),
            )
            for chunk in self._viewing_chunks
        )

    def _linearize_engine(
        self,
        engine,
        atmosphere,
        *,
        internal_atmosphere=None,
        prepare_parameters=None,
    ):
        engine._validate_orbital_atmosphere(atmosphere)
        if internal_atmosphere is None:
            internal_atmosphere = atmosphere.internal_object()
        engine._prepare_refraction(atmosphere)
        engine._prepare_surface(atmosphere)
        state_generation = engine._engine.state_generation()

        def validate_session(
            *,
            owner=engine,
            generation=state_generation,
        ) -> None:
            owner._validate_linearization_session(generation)

        return sk2.Engine._linearize(
            engine,
            atmosphere,
            internal_atmosphere=internal_atmosphere,
            validate_session=validate_session,
            prepare_parameters=(
                None
                if engine.derivative_execution == "streaming"
                else prepare_parameters
            ),
        )

    def _construct_chunk_engine(
        self,
        chunk_index: int,
    ):
        return sk2.OrbitalPlaneEngine(
            self._config,
            self._geometry,
            self._viewing_chunks[chunk_index],
            **self._engine_kwargs,
        )

    def _evaluate_chunk_batch(
        self,
        chunk_indices: tuple[int, ...],
        atmosphere,
        *,
        internal_atmosphere,
        prepare_parameters,
        evaluator,
    ):
        def evaluate(chunk_index):
            engine = self._construct_chunk_engine(chunk_index)
            linearization = self._linearize_engine(
                engine,
                atmosphere,
                internal_atmosphere=internal_atmosphere,
                prepare_parameters=prepare_parameters,
            )
            try:
                return evaluator(chunk_index, linearization)
            finally:
                del linearization, engine

        return tuple(evaluate(index) for index in chunk_indices)

    def _linearize_chunk(
        self,
        atmosphere,
        chunk_index: int,
        *,
        internal_atmosphere=None,
        prepare_parameters=None,
    ):
        engine = self._construct_chunk_engine(chunk_index)
        return engine, self._linearize_engine(
            engine,
            atmosphere,
            internal_atmosphere=internal_atmosphere,
            prepare_parameters=prepare_parameters,
        )

    def linearize(self, atmosphere, *, prepare_parameters=None):
        if self._chunk_execution == "streaming":
            return _StreamingChunkedOrbitalPlaneLinearization(
                self,
                atmosphere,
                prepare_parameters=prepare_parameters,
            )
        for engine in self._engines:
            engine._validate_orbital_atmosphere(atmosphere)
        internal_atmosphere = atmosphere.internal_object()
        linearizations = []
        for engine in self._engines:
            linearizations.append(
                self._linearize_engine(
                    engine,
                    atmosphere,
                    internal_atmosphere=internal_atmosphere,
                    prepare_parameters=prepare_parameters,
                )
            )
        return _ChunkedOrbitalPlaneLinearization(linearizations)


class OrbitalPlaneSpectrograph(SpectrometerMixin, StandardForwardModel):
    """Matrix-free spectrograph forward model for an orbital-plane atmosphere."""

    def __init__(
        self,
        observation,
        state_vector: OrbitalPlaneStateVector,
        meas_vec,
        ancillary,
        engine_config: sk2.Config,
        *,
        model_geometry: dict[str, sk2.OrbitalPlaneGeometry],
        engine_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        self._provided_model_geometry = model_geometry
        self._orbital_engine_kwargs = (
            {} if engine_kwargs is None else dict(engine_kwargs)
        )
        self._prepare_linearization_parameters = bool(
            self._orbital_engine_kwargs.pop("prepare_linearization_parameters", True)
        )
        SpectrometerMixin.__init__(
            self,
            lineshape_fn=kwargs.get("lineshape_fn", lambda _: DeltaFunction()),
            model_res_cminv=kwargs.get("model_res_cminv", 0.02),
            model_res_nm=kwargs.get("model_res_nm", 0.02),
            round_decimal=kwargs.get("round_decimal", 2),
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

    def _linearize(self, key: str, *, prepare_parameters=None):
        if not self._prepare_linearization_parameters:
            prepare_parameters = None
        return super()._linearize(key, prepare_parameters=prepare_parameters)

    def _construct_viewing_geo(self):
        viewing = self._observation.sk2_geometry()
        if any(
            not isinstance(value, sk2.OrbitalPlaneViewingGeometry)
            for value in viewing.values()
        ):
            msg = "OrbitalPlaneSpectrograph requires orbital-plane viewing geometry"
            raise TypeError(msg)
        return viewing

    def _construct_model_geometry(self):
        missing = set(self._viewing_geo) - set(self._provided_model_geometry)
        if missing:
            msg = f"Missing orbital-plane model geometry for {sorted(missing)}"
            raise KeyError(msg)
        return {key: self._provided_model_geometry[key] for key in self._viewing_geo}

    def _construct_atmosphere(self):
        atmospheres = {}
        for key, geometry in self._model_geometry.items():
            atmosphere = sk2.Atmosphere(
                geometry,
                self._engine_config,
                wavelengths_nm=self._model_wavelength[key],
                pressure_derivative=False,
                temperature_derivative=False,
            )
            self._state_vector.add_to_atmosphere(atmosphere)
            self._ancillary.add_to_atmosphere(atmosphere)
            atmospheres[key] = atmosphere
        return atmospheres

    def _construct_engine(self):
        kwargs = dict(self._orbital_engine_kwargs)
        max_time_groups_per_engine = kwargs.pop("max_time_groups_per_engine", None)
        chunk_execution = kwargs.pop("chunk_execution", "resident")
        streaming_chunk_workers = kwargs.pop("streaming_chunk_workers", 1)
        if max_time_groups_per_engine is None and chunk_execution != "resident":
            msg = "chunk_execution='streaming' requires max_time_groups_per_engine"
            raise ValueError(msg)
        if "solar_handler" not in kwargs and "sun_vectors_ecef" not in kwargs:
            kwargs["solar_handler"] = sk2.solar.SolarGeometryHandlerAstropy()
        engines = {}
        for key in self._model_geometry:
            if max_time_groups_per_engine is None:
                engines[key] = sk2.OrbitalPlaneEngine(
                    self._engine_config,
                    self._model_geometry[key],
                    self._viewing_geo[key],
                    **kwargs,
                )
            else:
                engines[key] = _ChunkedOrbitalPlaneEngine(
                    self._engine_config,
                    self._model_geometry[key],
                    self._viewing_geo[key],
                    max_time_groups_per_engine=max_time_groups_per_engine,
                    chunk_execution=chunk_execution,
                    streaming_chunk_workers=streaming_chunk_workers,
                    **kwargs,
                )
        return engines

    def calculate_radiance(self):
        """Calculate radiance without requesting a stitched eager Jacobian."""
        l1 = {}
        for key in self._engine:
            value = self._linearize(key).value
            radiance = SASKTRANRadiance.from_sasktran2(
                value.to_dataset(name="radiance")
            )
            self._append_instrument_result(l1, key, radiance)
        return l1


class OrbitalPlaneRetrieval(Retrieval):
    """Build and retrieve a structured state along an observed orbital track.

    The wrapper constructs the SASKTRAN2 atmosphere grid before invoking the
    user-supplied ``state_vector_factory``. This resolves the ordering required
    by native 2D constituents while retaining the standard skretrieval target,
    measurement-vector, and minimizer interfaces.
    """

    def __init__(
        self,
        observation,
        *,
        altitude_grid_m: np.ndarray,
        along_track_angle_delta: float,
        state_vector_factory: Callable[
            [sk2.OrbitalPlaneGeometry], OrbitalPlaneStateVector
        ],
        path_padding_angle: float = np.deg2rad(5.0),
        interpolation_method: sk2.InterpolationMethod = sk2.InterpolationMethod.LinearInterpolation,
        orbital_engine_kwargs: dict | None = None,
        forward_model_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        viewing = observation.sk2_geometry()
        if len(viewing) != 1:
            msg = "OrbitalPlaneRetrieval currently requires one observation stream"
            raise ValueError(msg)
        key, viewing_geometry = next(iter(viewing.items()))
        if not isinstance(viewing_geometry, sk2.OrbitalPlaneViewingGeometry):
            msg = "Observation must provide OrbitalPlaneViewingGeometry"
            raise TypeError(msg)

        geometry = viewing_geometry.construct_atmosphere_geometry(
            np.asarray(altitude_grid_m, dtype=float),
            along_track_angle_delta,
            interpolation_method,
            path_padding_angle=path_padding_angle,
        )
        self._orbital_plane_geometry = geometry
        self._provided_state_vector = state_vector_factory(geometry)
        if not isinstance(self._provided_state_vector, OrbitalPlaneStateVector):
            msg = "state_vector_factory must return OrbitalPlaneStateVector"
            raise TypeError(msg)

        if forward_model_cfg is None:
            engine_kwargs = {
                "time_group_duration_s": 60.0,
                "derivative_execution": "streaming",
            }
            if orbital_engine_kwargs is not None:
                engine_kwargs.update(orbital_engine_kwargs)
            forward_model_cfg = {
                key: {
                    "class": OrbitalPlaneSpectrograph,
                    "kwargs": {
                        "model_geometry": {key: geometry},
                        "engine_kwargs": engine_kwargs,
                    },
                }
            }

        super().__init__(
            observation,
            forward_model_cfg=forward_model_cfg,
            state_kwargs={},
            **kwargs,
        )

    @property
    def model_geometry(self) -> sk2.OrbitalPlaneGeometry:
        return self._orbital_plane_geometry

    @property
    def state_vector(self) -> OrbitalPlaneStateVector:
        """Return the structured state used by this retrieval."""
        return self._provided_state_vector

    def _construct_state_vector(self):
        return self._provided_state_vector
