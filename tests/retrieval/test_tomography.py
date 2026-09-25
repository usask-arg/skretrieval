from __future__ import annotations

from types import MappingProxyType

import numpy as np
import xarray as xr

from skretrieval.retrieval.tomography import (
    OrbitalPlaneSpectrograph,
    _ChunkedOrbitalPlaneEngine,
    _ChunkedOrbitalPlaneLinearization,
    _split_time_groups,
    _StreamingChunkedOrbitalPlaneLinearization,
)


class _ViewingGeometry:
    def __init__(self, times, vertical_slice, *, origin=None):
        self.times = np.asarray(times, dtype="datetime64[ns]")
        self.vertical_slice = np.asarray(vertical_slice)
        self.time_bin_origin = (
            self.times[0] if origin is None else np.datetime64(origin, "ns")
        )

    def isel(self, selection):
        return type(self)(
            self.times[selection],
            self.vertical_slice[selection],
            origin=self.time_bin_origin,
        )


class _Linearization:
    def __init__(self, size: int, factor: float):
        coords = {
            "wavelength": [869.0],
            "los": np.arange(size),
            "stokes": ["I"],
        }
        self.value = xr.DataArray(
            np.full((1, size, 1), factor),
            dims=("wavelength", "los", "stokes"),
            coords=coords,
        )
        self.tangent_template = xr.Dataset(
            {"extinction": ("state", np.zeros(2))},
            coords={"state": np.arange(2)},
        )
        self.backends = MappingProxyType({"jvp": "native", "vjp": "native"})
        self._factor = factor

    @property
    def jacobian(self):
        values = np.full((*self.value.shape, 2), self._factor)
        return xr.Dataset(
            {
                "extinction": (
                    ("wavelength", "los", "stokes", "state"),
                    values,
                )
            },
            coords={**self.value.coords, "state": np.arange(2)},
        )

    def jvp(self, tangent):
        return xr.full_like(
            self.value,
            float(tangent.extinction.sum()) * self._factor,
        )

    def vjp(self, cotangent, parameters=None):
        assert cotangent.coords["los"].equals(self.value.coords["los"])
        if parameters is not None and "extinction" not in parameters:
            return xr.Dataset()
        value = float(cotangent.sum()) * self._factor
        return xr.Dataset(
            {"extinction": ("state", np.full(2, value))},
            coords={"state": np.arange(2)},
        )


def test_split_time_groups_keeps_complete_native_slices():
    origin = np.datetime64("2022-01-01T00:00:00", "ns")
    offsets_s = np.repeat([10, 70, 130, 190, 250, 310], 2)
    viewing = _ViewingGeometry(
        origin + offsets_s.astype("timedelta64[s]"),
        np.repeat(np.arange(6), 2),
        origin=origin,
    )

    chunks = _split_time_groups(
        viewing,
        time_group_duration_s=120.0,
        max_time_groups_per_engine=2,
    )

    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0].vertical_slice, np.repeat(range(4), 2))
    np.testing.assert_array_equal(chunks[1].vertical_slice, np.repeat(range(4, 6), 2))
    assert all(chunk.time_bin_origin == origin for chunk in chunks)


def test_split_time_groups_breaks_at_missing_native_bins():
    origin = np.datetime64("2022-01-01T00:00:00", "ns")
    offsets_s = np.repeat([10, 70, 610, 670], 2)
    viewing = _ViewingGeometry(
        origin + offsets_s.astype("timedelta64[s]"),
        np.repeat(np.arange(4), 2),
        origin=origin,
    )

    chunks = _split_time_groups(
        viewing,
        time_group_duration_s=120.0,
        max_time_groups_per_engine=6,
    )

    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0].vertical_slice, np.repeat(range(2), 2))
    np.testing.assert_array_equal(chunks[1].vertical_slice, np.repeat(range(2, 4), 2))
    assert all(chunk.time_bin_origin == origin for chunk in chunks)


def test_chunked_linearization_concatenates_products_and_sums_vjp():
    linearization = _ChunkedOrbitalPlaneLinearization(
        [_Linearization(2, 1.0), _Linearization(3, 2.0)]
    )

    np.testing.assert_array_equal(linearization.value.los, np.arange(5))
    np.testing.assert_allclose(
        linearization.jvp(
            xr.Dataset({"extinction": ("state", [3.0, 4.0])})
        ).values.ravel(),
        [7.0, 7.0, 14.0, 14.0, 14.0],
    )
    gradient = linearization.vjp(xr.ones_like(linearization.value))
    np.testing.assert_allclose(gradient.extinction, [8.0, 8.0])
    assert linearization.jacobian.sizes["los"] == 5


def test_resident_chunks_share_one_materialized_atmosphere():
    class Atmosphere:
        def __init__(self):
            self.calls = 0

        def internal_object(self):
            self.calls += 1
            return object()

    class Engine:
        def _validate_orbital_atmosphere(self, atmosphere):
            assert isinstance(atmosphere, Atmosphere)

    owner = object.__new__(_ChunkedOrbitalPlaneEngine)
    owner._chunk_execution = "resident"
    owner._engines = (Engine(), Engine())
    received_internal_atmospheres = []

    def linearize_engine(
        engine,
        atmosphere,
        *,
        internal_atmosphere=None,
        prepare_parameters=None,
    ):
        assert isinstance(engine, Engine)
        assert isinstance(atmosphere, Atmosphere)
        assert prepare_parameters == ("extinction",)
        received_internal_atmospheres.append(internal_atmosphere)
        return _Linearization(2, 1.0)

    owner._linearize_engine = linearize_engine
    atmosphere = Atmosphere()
    owner.linearize(atmosphere, prepare_parameters=("extinction",))

    assert atmosphere.calls == 1
    assert len(received_internal_atmospheres) == 2
    assert received_internal_atmospheres[0] is received_internal_atmospheres[1]


def test_orbital_spectrograph_can_skip_jvp_parameter_preparation():
    class Engine:
        def __init__(self):
            self.prepared = []

        def linearize(self, atmosphere, *, prepare_parameters=None):
            assert atmosphere == "atmosphere"
            self.prepared.append(prepare_parameters)
            return "linearization"

    engine = Engine()
    model = object.__new__(OrbitalPlaneSpectrograph)
    model._engine = {"measurement": engine}
    model._atmosphere = {"measurement": "atmosphere"}
    model._prepare_linearization_parameters = False

    assert (
        model._linearize("measurement", prepare_parameters=("extinction",))
        == "linearization"
    )
    assert engine.prepared == [None]


def test_streaming_chunked_linearization_rebuilds_bounded_batches_for_vjp():
    class Atmosphere:
        def __init__(self):
            self.internal_calls = 0

        def internal_object(self):
            self.internal_calls += 1
            return f"internal-{self.internal_calls}"

    class Owner:
        num_chunks = 5
        streaming_chunk_batches = ((0, 1, 2), (3, 4))

        def __init__(self):
            self.calls = []

        def _evaluate_chunk_batch(
            self,
            chunk_indices,
            _atmosphere,
            *,
            internal_atmosphere,
            prepare_parameters,
            evaluator,
        ):
            self.calls.append(("batch", chunk_indices))
            results = []
            for engine in chunk_indices:
                self.calls.append(
                    ("linearize", engine, internal_atmosphere, prepare_parameters)
                )
                results.append(
                    evaluator(engine, _Linearization(engine + 2, engine + 1.0))
                )
            return tuple(results)

    owner = Owner()
    atmosphere = Atmosphere()
    linearization = _StreamingChunkedOrbitalPlaneLinearization(
        owner,
        atmosphere,
        prepare_parameters=("extinction",),
    )

    np.testing.assert_allclose(
        linearization.value.values.ravel(),
        np.repeat(np.arange(1.0, 6.0), np.arange(2, 7)),
    )
    gradient = linearization.vjp(xr.ones_like(linearization.value))
    np.testing.assert_allclose(gradient.extinction, [70.0, 70.0])
    assert owner.calls == [
        ("batch", (0, 1, 2)),
        ("linearize", 0, "internal-1", ("extinction",)),
        ("linearize", 1, "internal-1", ("extinction",)),
        ("linearize", 2, "internal-1", ("extinction",)),
        ("batch", (3, 4)),
        ("linearize", 3, "internal-2", ("extinction",)),
        ("linearize", 4, "internal-2", ("extinction",)),
        ("batch", (0, 1, 2)),
        ("linearize", 0, "internal-3", ("extinction",)),
        ("linearize", 1, "internal-3", ("extinction",)),
        ("linearize", 2, "internal-3", ("extinction",)),
        ("batch", (3, 4)),
        ("linearize", 3, "internal-4", ("extinction",)),
        ("linearize", 4, "internal-4", ("extinction",)),
    ]
    assert atmosphere.internal_calls == 4


def test_streaming_chunk_batches_limit_resident_engines():
    owner = object.__new__(_ChunkedOrbitalPlaneEngine)
    owner._viewing_chunks = tuple(range(7))
    owner._streaming_chunk_workers = 1

    assert owner.streaming_chunk_batches == tuple((index,) for index in range(7))
