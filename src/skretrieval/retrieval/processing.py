from __future__ import annotations

from copy import copy
from typing import ClassVar

import numpy as np
import sasktran2 as sk2

import skretrieval.retrieval.prior as prior
from skretrieval.retrieval.rodgers import Rodgers
from skretrieval.retrieval.scipy import SciPyMinimizer, SciPyMinimizerGrad
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent

from .ancillary import Ancillary, US76Ancillary
from .forwardmodel import ForwardModelHandler, IdealViewingSpectrograph
from .measvec import MeasurementVector, select
from .observation import Observation
from .statevector.altitude import AltitudeNativeStateVector
from .target.mvtarget import MeasVecTarget


class Retrieval:
    _optical_property_fns: ClassVar[dict[str, callable]] = {}
    _prior_fns: ClassVar[dict[str, callable]] = {}
    _state_fns: ClassVar[dict[str, dict[str, callable]]] = {
        "absorbers": {},
        "aerosols": {},
        "splines": {},
        "surface": {},
        "other": {},
    }

    def _context_fn(_):
        return {}

    @classmethod
    def register_context(cls):
        def decorator(context_fn: callable):
            cls._context_fn = context_fn
            return context_fn

        return decorator

    @classmethod
    def register_optical_property(cls, species_name: str):
        def decorator(optical_property_fn: callable):
            cls._optical_property_fns[species_name] = optical_property_fn
            return optical_property_fn

        return decorator

    @classmethod
    def register_prior(cls, species_name: str):
        def decorator(prior_fn: callable):
            cls._prior_fns[species_name] = prior_fn
            return prior_fn

        return decorator

    @classmethod
    def register_state(cls, category: str, species_name: str):
        def decorator(state_fn: callable):
            cls._state_fns[category][species_name] = state_fn
            return state_fn

        return decorator

    def __init__(
        self,
        observation: Observation,
        measvec: dict[MeasurementVector] | None = None,
        forward_model_cfg: dict | None = None,
        minimizer="rodgers",
        ancillary: Ancillary | None = None,
        l1_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        minimizer_kwargs: dict | None = None,
        target_kwargs: dict | None = None,
        state_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """
        The main processing script that handles the retrieval

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
        forward_model_cfg : dict | None, optional
            Additional arguments passed to the forward model, by default None
        """
        if minimizer.lower() == "rodgers":
            # Override the default Rodgers options
            self._minimizer_kwargs = {
                "lm_damping_method": "fletcher",
                "lm_damping": 0.1,
                "max_iter": 30,
                "lm_change_factor": 2,
                "iterative_update_lm": True,
                "retreat_lm": False,
                "apply_cholesky_scaling": True,
                "convergence_factor": 1e-2,
                "convergence_check_method": "dcost",
            }
        else:
            self._minimizer_kwargs = {}
        if minimizer_kwargs is not None:
            self._minimizer_kwargs.update(minimizer_kwargs)

        if state_kwargs is None:
            state_kwargs = {}
        if target_kwargs is None:
            target_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if l1_kwargs is None:
            l1_kwargs = {}
        if forward_model_cfg is None:
            forward_model_cfg = {"*": {"class": IdealViewingSpectrograph}}
        self._options = kwargs
        self._l1_kwargs = l1_kwargs
        self._minimizer = minimizer
        self._target_kwargs = target_kwargs
        self._state_kwargs = state_kwargs
        self._model_kwargs = model_kwargs
        self._measurement_vector = measvec
        self._forward_model_cfg = forward_model_cfg

        self._ancillary = ancillary

        self._measurement_vector = self._construct_measurement_vector()

        self._observation = observation

        self._anc = self._construct_ancillary()

        self._context = self._context_fn()

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
            "measurement": MeasurementVector(
                lambda l1, ctxt, **kwargs: select(l1, **kwargs)  # noqa: ARG005
            )
        }

    def _construct_forward_model(self):
        engine_config = sk2.Config()

        for k, v in self._model_kwargs.items():
            setattr(engine_config, k, v)

        return ForwardModelHandler(
            self._forward_model_cfg,
            self._observation,
            self._state_vector,
            self._measurement_vector,
            self._anc,
            engine_config,
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
        return self._optical_property_fns[species_name]()

    @staticmethod
    def _default_state_absorber(self, name: str, native_alt_grid: np.array, cfg: dict):
        const = self._const_from_mipas(
            native_alt_grid,
            name,
            self._optical_property(name),
            tikh=cfg["tikh_factor"],
            prior_infl=cfg["prior_influence"],
            log_space=cfg["log_space"],
            min_val=cfg["min_value"],
            max_val=cfg["max_value"],
        )

        const.enabled = cfg.get("enabled", True)
        return const

    @staticmethod
    def _default_state_surface(
        self, name: str, native_alt_grid: np.array, cfg: dict  # noqa: ARG004
    ):
        msg = f"Surface {name} does not have a default implementation"
        raise ValueError(msg)

    @staticmethod
    def _default_state_spline(
        self, name: str, native_alt_grid: np.array, cfg: dict  # noqa: ARG004
    ):
        msg = f"Spline {name} does not have a default implementation"
        raise ValueError(msg)

    @staticmethod
    def _default_state_aerosol(
        self, name: str, native_alt_grid: np.array, cfg: dict  # noqa: ARG004
    ):
        msg = f"aerosol {name} does not have a default implementation"
        raise ValueError(msg)

    def _construct_state_vector(self):
        native_alt_grid = self._state_kwargs["altitude_grid"]

        absorbers = {}

        for name, options in self._state_kwargs.get("absorbers", {}).items():
            absorbers[name] = self._state_fns["absorbers"].get(
                name, self._default_state_absorber
            )(self, name, native_alt_grid, options)

        surface = {}
        for name, options in self._state_kwargs.get("surface", {}).items():
            surface[name] = self._state_fns["surface"].get(
                name, self._default_state_surface
            )(self, name, native_alt_grid, options)

        aerosols = {}
        for name, aerosol in self._state_kwargs.get("aerosols", {}).items():
            aerosols[f"{name}"] = self._state_fns["aerosols"].get(
                aerosol["type"], self._default_state_aerosol
            )(self, name, native_alt_grid, aerosol)

        splines = {}
        for name, spline in self._state_kwargs.get("splines", {}).items():
            splines[name] = self._state_fns["splines"].get(
                name, self._default_state_spline
            )(self, name, native_alt_grid, spline)

        return AltitudeNativeStateVector(
            native_alt_grid, **absorbers, **surface, **aerosols, **splines
        )

    def _construct_ancillary(self):
        if self._ancillary is None:
            return US76Ancillary()
        return self._ancillary

    def _construct_target(self):
        return MeasVecTarget(
            self._state_vector,
            self._measurement_vector,
            self._context,
            **self._target_kwargs,
        )

    def _construct_output(self, rodgers_output: dict):
        return rodgers_output

    def retrieve(
        self,
        enabled_state_elements: list[str] | None = None,
        enabled_measurement_vectors: list[str] | None = None,
    ):
        if self._minimizer == "rodgers":
            minimizer = Rodgers(**self._minimizer_kwargs)
        elif self._minimizer == "scipy":
            minimizer = SciPyMinimizer(**self._minimizer_kwargs)
        elif self._minimizer == "scipy_grad":
            minimizer = SciPyMinimizerGrad()

        if enabled_state_elements is not None:
            for key, val in self._state_vector.sv.items():
                if key in enabled_state_elements:
                    val.enabled = True
                else:
                    val.enabled = False

        if enabled_measurement_vectors is not None:
            for key, val in self._measurement_vector.items():
                if key in enabled_measurement_vectors:
                    val.enabled = True
                else:
                    val.enabled = False

        self._target.update_state_slices()

        min_results = minimizer.retrieve(
            self._obs_l1, self._forward_model, self._target
        )

        # Reset the enabled flag
        for _, val in self._state_vector.sv.items():
            val.enabled = True

        for _, val in self._measurement_vector.items():
            val.enabled = True

        # Post process
        final_l1 = self._forward_model.calculate_radiance()
        meas_l1 = self._obs_l1

        results = {}

        results["minimizer"] = min_results
        results["meas_l1"] = meas_l1
        results["simulated_l1"] = final_l1

        results["state"] = self._state_vector.describe(min_results)

        return self._construct_output(results)


# Register all the default optical properties
@Retrieval.register_optical_property("o3")
def o3_optical_property(*args, **kwargs):
    return sk2.optical.O3DBM()


@Retrieval.register_optical_property("no2")
def no2_optical_property(*args, **kwargs):
    return sk2.optical.NO2Vandaele()


@Retrieval.register_optical_property("bro")
def bro_optical_property(*args, **kwargs):
    return sk2.optical.HITRANUV("BrO")


@Retrieval.register_optical_property("so2")
def so2_optical_property(*args, **kwargs):
    return sk2.optical.HITRANUV("SO2")


# Register the default Lambertian surface state
@Retrieval.register_state("surface", "lambertian_albedo")
def lambertian_state(self, name, native_alt_grid: np.array, cfg: dict):  # noqa: ARG001
    albedo_wavel = cfg["wavelengths"]
    albedo_start = np.ones(len(albedo_wavel)) * cfg["initial_value"]

    albedo_const = sk2.constituent.LambertianSurface(
        albedo_start, albedo_wavel, cfg.get("out_of_bounds_mode", "extend")
    )
    sv_ele = StateVectorElementConstituent(
        albedo_const,
        name,
        ["albedo"],
        min_value={"albedo": 0},
        max_value={"albedo": 1},
        prior={
            "albedo": cfg["tikh_factor"] * prior.VerticalTikhonov(1)
            + cfg["prior_influence"] * prior.ConstantDiagonalPrior()
        },
        log_space=False,
    )
    sv_ele.enabled = cfg.get("enabled", True)

    return sv_ele


@Retrieval.register_state("aerosols", "extinction_profile")
def aerosol_extinction_profile(self, name: str, native_alt_grid: np.array, cfg: dict):
    aero_const = sk2.test_util.scenarios.test_aerosol_constituent(native_alt_grid)

    ext = copy(aero_const.extinction_per_m)

    low_boundary = np.nonzero(ext)[0][0]

    ext[:low_boundary] = ext[low_boundary]

    ext[ext == 0] = 1e-15

    scale_factor = cfg.get("scale_factor", 1)

    secondary_kwargs = {
        name: np.ones_like(native_alt_grid) * cfg["prior"][name]["value"]
        for name in cfg["prior"]
        if name != "extinction_per_m"
    }

    db = self._optical_property(name)

    aero_const = sk2.constituent.ExtinctionScatterer(
        db,
        native_alt_grid,
        ext * scale_factor,
        745,
        "extend",
        **secondary_kwargs,
    )

    sv_ele = StateVectorElementConstituent(
        aero_const,
        f"{name}",
        cfg["retrieved_quantities"].keys(),
        min_value={
            name: val["min_value"] for name, val in cfg["retrieved_quantities"].items()
        },
        max_value={
            name: val["max_value"] for name, val in cfg["retrieved_quantities"].items()
        },
        prior={
            name: val["tikh_factor"]
            * prior.VerticalTikhonov(
                1, prior_state=secondary_kwargs.get(name, ext * scale_factor)
            )
            + val["prior_influence"] * prior.ConstantDiagonalPrior()
            for name, val in cfg["retrieved_quantities"].items()
        },
        log_space=False,
    )

    sv_ele.enabled = cfg.get("enabled", True)

    return sv_ele
