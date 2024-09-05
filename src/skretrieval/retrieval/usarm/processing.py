from __future__ import annotations

from copy import copy

import numpy as np
import sasktran2 as sk2

import skretrieval.retrieval.usarm.prior as prior
from skretrieval.retrieval.rodgers import Rodgers
from skretrieval.retrieval.scipy import SciPyMinimizer, SciPyMinimizerGrad
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent

from .ancillary import US76Ancillary
from .forwardmodel import USARMForwardModel
from .measvec import MeasurementVector, select
from .observation import Observation
from .spline import MultiplicativeSpline
from .statevector import USARMStateVector
from .target import USARMTarget


class USARMRetrieval:
    def __init__(
        self,
        observation: Observation,
        measvec: dict[MeasurementVector] | None = None,
        minimizer="rodgers",
        l1_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        minimizer_kwargs: dict | None = None,
        target_kwargs: dict | None = None,
        state_kwargs: dict | None = None,
        forward_model_kwargs: dict | None = None,
        forward_model_class: type = USARMForwardModel,
        **kwargs,
    ) -> None:
        """
        The main processing script that handles the usarm retrieval

        Parameters
        ----------
        observation : Observation
        measvec : dict[MeasurementVector] | None, optional
            Measurement vectors to use, by default will use the raw radiances, by default None
        minimizer : str, optional
            Selects which minimizer to use, default is "rodgers", by default "rodgers"
        l1_kwargs : dict | None, optional
            Additional arguments passed to the observation when constructing the L1, by default None
        model_kwargs : dict | None, optional
            Additional arguments passed to the SASKTRAN2 engine, by default None
        minimizer_kwargs : dict | None, optional
            Additional arguments passed to the minimizer, by default None
        target_kwargs : dict | None, optional
            Additional arguments passed to the retrieval target, by default None
        state_kwargs : dict | None, optional
            Arguments to construct the state vector, by default None
        forward_model_kwargs : dict | None, optional
            Additional arguments passed to the forward model, by default None
        forward_model_class : type, optional
            Class to use when constructing the forward model, by default USARMForwardModel
        """
        if state_kwargs is None:
            state_kwargs = {}
        if target_kwargs is None:
            target_kwargs = {}
        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if l1_kwargs is None:
            l1_kwargs = {}
        if forward_model_kwargs is None:
            forward_model_kwargs = {}
        self._options = kwargs
        self._l1_kwargs = l1_kwargs
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs
        self._target_kwargs = target_kwargs
        self._state_kwargs = state_kwargs
        self._model_kwargs = model_kwargs
        self._forward_model_class = forward_model_class
        self._forward_model_kwargs = forward_model_kwargs
        self._measurement_vector = measvec

        self._measurement_vector = self._construct_measurement_vector()

        self._observation = observation

        self._anc = self._construct_ancillary()
        self._state_vector = self._construct_state_vector()
        self._forward_model = self._construct_forward_model()
        self._target = self._construct_target()

        self._obs_l1 = self._observation.skretrieval_l1(
            self._forward_model, self._state_vector, self._l1_kwargs
        )

    def _construct_measurement_vector(self):
        if self._measurement_vector is not None:
            return self._measurement_vector
        return {
            "measurement": MeasurementVector(lambda l1, **kwargs: select(l1, **kwargs))
        }

    def _construct_forward_model(self):
        engine_config = sk2.Config()

        return self._forward_model_class(
            self._observation,
            self._state_vector,
            self._anc,
            engine_config,
            **self._forward_model_kwargs,
        )

    def _const_from_mipas(
        self,
        alt_grid,
        species_name,
        optical,
        prior_infl=1e-2,
        tikh=1e8,
        log_space=False,
        min_val=0,
        max_val=1,
    ):
        const = sk2.climatology.mipas.constituent(species_name, optical)

        new_vmr = np.interp(alt_grid, const._altitudes_m, const.vmr)

        new_const = sk2.constituent.VMRAltitudeAbsorber(
            optical, alt_grid, new_vmr, out_of_bounds_mode="extend"
        )

        min_val = 1e-40 if log_space else min_val

        return StateVectorElementConstituent(
            new_const,
            species_name,
            ["vmr"],
            min_value={"vmr": min_val},
            max_value={"vmr": max_val},
            prior={
                "vmr": tikh * prior.VerticalTikhonov(1)
                + prior_infl * prior.ConstantDiagonalPrior()
            },
            log_space=log_space,
        )

    def _optical_property(self, species_name: str):
        if species_name.lower() == "o3":
            return sk2.optical.O3DBM()
        if species_name.lower() == "no2":
            return sk2.optical.NO2Vandaele()
        if species_name.lower() == "bro":
            return sk2.optical.HITRANUV("BrO")
        if species_name.lower() == "so2":
            return sk2.optical.HITRANUV("SO2")
        if species_name.lower() == "o2":
            return sk2.optical.HITRANTabulated("O2")
        if species_name.lower() == "h2o":
            return sk2.optical.HITRANTabulated("H2O")
        return None

    def _construct_state_vector(self):
        native_alt_grid = self._state_kwargs["altitude_grid"]

        absorbers = {}

        for name, options in self._state_kwargs["absorbers"].items():
            if options["prior"]["type"] == "mipas":
                absorbers[name] = self._const_from_mipas(
                    native_alt_grid,
                    name,
                    self._optical_property(name),
                    tikh=options["tikh_factor"],
                    prior_infl=options["prior_influence"],
                    log_space=options["log_space"],
                    min_val=options["min_value"],
                    max_val=options["max_value"],
                )

            absorbers[name].enabled = options.get("enabled", True)

        surface = {}
        if "albedo" in self._state_kwargs:
            options = self._state_kwargs["albedo"]

            albedo_wavel = options["wavelengths"]
            albedo_start = np.ones(len(albedo_wavel)) * options["initial_value"]

            albedo_const = sk2.constituent.LambertianSurface(
                albedo_start, albedo_wavel, options.get("out_of_bounds_mode", "extend")
            )
            surface["albedo"] = StateVectorElementConstituent(
                albedo_const,
                "albedo",
                ["albedo"],
                min_value={"albedo": 0},
                max_value={"albedo": 1},
                prior={
                    "albedo": options["tikh_factor"] * prior.VerticalTikhonov(1)
                    + options["prior_influence"] * prior.ConstantDiagonalPrior()
                },
                log_space=False,
            )
            surface["albedo"].enabled = options.get("enabled", True)

        aerosols = {}
        for name, aerosol in self._state_kwargs.get("aerosols", {}).items():
            if aerosol["type"] == "extinction_profile":
                aero_const = sk2.test_util.scenarios.test_aerosol_constituent(
                    native_alt_grid
                )

                ext = copy(aero_const.extinction_per_m)
                ext[ext == 0] = 1e-15

                scale_factor = aerosol.get("scale_factor", 1)

                secondary_kwargs = {
                    name: np.ones_like(native_alt_grid)
                    * aerosol["prior"][name]["value"]
                    for name in aerosol["prior"]
                    if name != "extinction_per_m"
                }

                refrac = sk2.mie.refractive.H2SO4()
                dist = sk2.mie.distribution.LogNormalDistribution().freeze(
                    mode_width=1.6
                )

                db = sk2.database.MieDatabase(
                    dist,
                    refrac,
                    np.arange(250.0, 1501.0, 50.0),
                    median_radius=np.arange(10.0, 901.0, 50.0),
                )

                aero_const = sk2.constituent.ExtinctionScatterer(
                    db,
                    native_alt_grid,
                    ext * scale_factor,
                    745,
                    "extend",
                    **secondary_kwargs,
                )

                aerosols[f"aerosol_{name}"] = StateVectorElementConstituent(
                    aero_const,
                    f"aerosol_{name}",
                    aerosol["retrieved_quantities"].keys(),
                    min_value={
                        name: val["min_value"]
                        for name, val in aerosol["retrieved_quantities"].items()
                    },
                    max_value={
                        name: val["max_value"]
                        for name, val in aerosol["retrieved_quantities"].items()
                    },
                    prior={
                        name: val["tikh_factor"] * prior.VerticalTikhonov(1)
                        + val["prior_influence"] * prior.ConstantDiagonalPrior()
                        for name, val in aerosol["retrieved_quantities"].items()
                    },
                    log_space=False,
                )

                aerosols[f"aerosol_{name}"].enabled = aerosol.get("enabled", True)

        splines = {}
        for name, spline in self._state_kwargs.get("splines", {}).items():
            splines[f"spline_{name}"] = MultiplicativeSpline(
                len(self._os.sasktran_geometry().lines_of_sight),
                spline["min_wavelength"],
                spline["max_wavelength"],
                spline["num_knots"],
                s=spline["smoothing"],
                order=spline["order"],
            )

            splines[f"spline_{name}"].enabled = spline.get("enabled", True)

        return USARMStateVector(
            native_alt_grid, **absorbers, **surface, **aerosols, **splines
        )

    def _construct_ancillary(self):
        return US76Ancillary()

    def _construct_target(self):
        return USARMTarget(
            self._state_vector, self._measurement_vector, **self._target_kwargs
        )

    def _construct_output(self, rodgers_output: dict):
        return rodgers_output

    def retrieve(self):
        if self._minimizer == "rodgers":
            minimizer = Rodgers(**self._minimizer_kwargs)
        elif self._minimizer == "scipy":
            minimizer = SciPyMinimizer(**self._minimizer_kwargs)
        elif self._minimizer == "scipy_grad":
            minimizer = SciPyMinimizerGrad()

        min_results = minimizer.retrieve(
            self._obs_l1, self._forward_model, self._target
        )

        # Post process
        final_l1 = self._forward_model.calculate_radiance()
        meas_l1 = self._obs_l1

        results = {}

        results["minimizer"] = min_results
        results["meas_l1"] = meas_l1
        results["simulated_l1"] = final_l1

        results["state"] = self._state_vector.describe(min_results)

        return self._construct_output(results)
