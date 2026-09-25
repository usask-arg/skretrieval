from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk2

from skretrieval.retrieval.ancillary import Ancillary
from skretrieval.retrieval.observation import Observation
from skretrieval.retrieval.prior import TwoDimensionalTikhonov
from skretrieval.retrieval.statevector.orbital_plane import (
    OrbitalPlaneStateVector,
    OrbitalPlaneStateVectorElement,
)
from skretrieval.retrieval.tomography import OrbitalPlaneRetrieval


class _TransmissionObservation(Observation):
    def __init__(self):
        radius = 6_372_000.0
        angles = np.repeat([-0.1, 0.1], 3)
        tangent_radius = radius + np.tile([10_000.0, 20_000.0, 30_000.0], 2)
        up = np.column_stack((np.sin(angles), np.zeros(6), np.cos(angles)))
        forward = np.column_stack((np.cos(angles), np.zeros(6), -np.sin(angles)))
        tangents = tangent_radius[:, np.newaxis] * up
        distance = np.sqrt((radius + 700_000.0) ** 2 - tangent_radius**2)
        observers = tangents - distance[:, np.newaxis] * forward
        times = np.datetime64("2026-01-01T00:00:00", "ns") + np.repeat(
            [0, 120], 3
        ).astype("timedelta64[s]")
        self.viewing = sk2.OrbitalPlaneViewingGeometry.from_tangent_locations(
            times,
            observers,
            tangents,
            vertical_slice=np.repeat([0, 1], 3),
            geoid=sk2.SphericalGeoid(radius),
        )

    def sk2_geometry(self, **_kwargs):
        return {"measurement": self.viewing}

    def sample_wavelengths(self):
        return {"measurement": np.array([500.0])}

    def skretrieval_l1(self, forward_model, state_vector, _kwargs):
        element = state_vector.sv["aerosol"]
        initial = element.state().copy()
        element.update_state(initial + np.log(1.3))
        try:
            result = forward_model.calculate_radiance()
            for radiance in result.values():
                radiance.data["radiance_noise"] = 0.01 * radiance.data["radiance"]
            return result
        finally:
            element.update_state(initial)


def _state_vector(geometry):
    optics = sk2.optical.HenyeyGreenstein.from_parameters(
        wavelength_nm=np.array([400.0, 600.0]),
        xs_total=np.full(2, 1e-12),
        ssa=np.full(2, 0.8e-12),
        g=np.zeros(2),
        max_num_moments=4,
    )
    constituent = sk2.constituent.ExtinctionScatterer2D(
        optics, np.full(geometry.shape, 1e-6), 500.0
    )
    element = OrbitalPlaneStateVectorElement(
        constituent,
        "aerosol",
        ["extinction_per_m"],
        geometry=geometry,
        min_value={"extinction_per_m": 1e-8},
        max_value={"extinction_per_m": 1e-4},
        log_space=True,
        prior={
            "extinction_per_m": TwoDimensionalTikhonov(
                geometry.shape,
                diagonal_factor=1.0,
                vertical_factor=0.5,
                horizontal_factor=0.5,
            )
        },
    )
    return OrbitalPlaneStateVector(geometry, aerosol=element)


def _retrieve(minimizer, chunk_execution=None):
    engine_options = {"solar_handler": None}
    if chunk_execution is not None:
        engine_options.update(
            max_time_groups_per_engine=1, chunk_execution=chunk_execution
        )
    options = {"max_nfev": 100, "ftol": 1e-10, "verbose": 0}
    if minimizer == "scipy_lbfgsb":
        # Small cost changes can still move weakly constrained state entries.
        # Converge tightly enough for the state comparison across platforms.
        options.update(max_nfev=300, ftol=1e-13)
        options["minimize_options"] = {"gtol": 1e-9}
    if minimizer == "scipy":
        options["materialized_jacobian_source"] = "linearization"
    retrieval = OrbitalPlaneRetrieval(
        _TransmissionObservation(),
        altitude_grid_m=np.array([0.0, 10_000.0, 25_000.0, 45_000.0, 60_000.0]),
        along_track_angle_delta=0.1,
        path_padding_angle=0.2,
        state_vector_factory=_state_vector,
        ancillary=Ancillary(),
        minimizer=minimizer,
        minimizer_kwargs=options,
        target_kwargs={"rescale_state_space": True},
        orbital_engine_kwargs=engine_options,
        model_kwargs={
            "single_scatter_source": sk2.SingleScatterSource.NoSource,
            "multiple_scatter_source": sk2.MultipleScatterSource.NoSource,
            "occultation_source": sk2.OccultationSource.Standard,
            "num_singlescatter_moments": 4,
        },
    )
    return retrieval.retrieve()


@pytest.fixture(scope="module")
def materialized_orbital_result():
    return _retrieve("scipy")


@pytest.mark.parametrize("minimizer", ["scipy_lsmr", "scipy_lbfgsb"])
@pytest.mark.parametrize("chunk_execution", ["resident", "streaming"])
def test_orbital_retrieval_matches_materialized_solution(
    minimizer, chunk_execution, materialized_orbital_result
):
    result = _retrieve(minimizer, chunk_execution)
    reference = materialized_orbital_result

    assert result["minimizer"]["minimizer"].success
    assert reference["minimizer"]["minimizer"].success
    assert (
        result["minimizer"]["objective_history"][-1]
        < 0.01 * result["minimizer"]["objective_history"][0]
    )
    np.testing.assert_allclose(
        result["minimizer"]["minimizer"].cost,
        reference["minimizer"]["minimizer"].cost,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result["state"]["aerosol_extinction_per_m"],
        reference["state"]["aerosol_extinction_per_m"],
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        result["simulated_l1"]["measurement"].data["radiance"],
        reference["simulated_l1"]["measurement"].data["radiance"],
        rtol=1e-5,
    )
