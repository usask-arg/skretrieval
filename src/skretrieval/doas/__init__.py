from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import sasktran2 as sk
import xarray as xr
from scipy.optimize import least_squares
from sklearn.decomposition import PCA

from skretrieval.core.lineshape import Gaussian


def _normalized_filter_weights(
    filter_weights: list[float] | np.ndarray | None,
) -> np.ndarray:
    if filter_weights is None:
        return np.array([1.0], dtype=float)

    weights = np.asarray(filter_weights, dtype=float).reshape(-1)
    if weights.size == 0:
        msg = "filter_weights must contain at least one value"
        raise ValueError(msg)
    if not np.all(np.isfinite(weights)):
        msg = "filter_weights must be finite"
        raise ValueError(msg)

    total = float(np.sum(weights))
    if abs(total) < 1e-15:
        msg = "filter_weights must not sum to zero"
        raise ValueError(msg)

    return weights / total


def _filter_dataarray_wavelength(
    data: xr.DataArray, weights: np.ndarray
) -> xr.DataArray:
    if "wavelength" not in data.dims:
        return data
    if len(weights) == 1 and np.isclose(weights[0], 1.0):
        return data

    ordered_dims = [dim for dim in data.dims if dim != "wavelength"] + ["wavelength"]
    reordered = data.transpose(*ordered_dims)
    values = np.asarray(reordered.to_numpy(), dtype=float)

    filtered = np.apply_along_axis(
        lambda row: np.convolve(row, weights, mode="same"),
        -1,
        values,
    )

    out = xr.DataArray(
        filtered,
        dims=reordered.dims,
        coords={
            dim: reordered.coords[dim]
            for dim in reordered.dims
            if dim in reordered.coords
        },
        attrs=reordered.attrs,
    )
    return out.transpose(*data.dims)


def _filter_measurement_input(
    radiances: xr.DataArray | xr.Dataset,
    weights: np.ndarray,
) -> xr.DataArray | xr.Dataset:
    if len(weights) == 1 and np.isclose(weights[0], 1.0):
        return radiances

    if isinstance(radiances, xr.DataArray):
        return _filter_dataarray_wavelength(radiances, weights)

    filtered = radiances.copy()
    if "radiance" in filtered:
        filtered["radiance"] = _filter_dataarray_wavelength(
            filtered["radiance"], weights
        )
    if "wf" in filtered:
        filtered["wf"] = _filter_dataarray_wavelength(filtered["wf"], weights)
    return filtered


def _temperature_cross_section_components(
    optical_quantity,
    atmo,
    temperatures_k: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    temps = np.atleast_1d(np.asarray(temperatures_k, dtype=float))
    if temps.size == 0:
        msg = "At least one temperature must be provided for absorber components"
        raise ValueError(msg)

    original_temperature = np.asarray(atmo.temperature_k, dtype=float).copy()
    num_wavel = int(atmo.num_wavel)
    components: list[np.ndarray] = []

    try:
        for temp in temps:
            atmo.temperature_k = np.full_like(
                original_temperature, float(temp), dtype=float
            )

            extinction = np.asarray(
                optical_quantity.atmosphere_quantities(atmo).extinction,
                dtype=float,
            )

            if extinction.ndim == 1:
                spectrum = extinction
            else:
                wavelength_axes = [
                    idx
                    for idx, size in enumerate(extinction.shape)
                    if size == num_wavel
                ]
                if not wavelength_axes:
                    msg = "Unable to identify wavelength axis in optical extinction"
                    raise ValueError(msg)

                wavelength_axis = wavelength_axes[-1]
                if wavelength_axis != extinction.ndim - 1:
                    extinction = np.moveaxis(extinction, wavelength_axis, -1)

                reshaped = extinction.reshape(-1, extinction.shape[-1])
                finite_rows = np.all(np.isfinite(reshaped), axis=1)
                reshaped = reshaped[finite_rows]
                if reshaped.size == 0:
                    msg = "Optical extinction does not contain finite samples"
                    raise ValueError(msg)

                spectrum = np.nanmean(reshaped, axis=0)

            if spectrum.shape[-1] != num_wavel:
                msg = (
                    "Absorber component wavelength grid does not match atmosphere grid"
                )
                raise ValueError(msg)

            components.append(np.asarray(spectrum, dtype=float))
    finally:
        atmo.temperature_k = original_temperature

    result = np.vstack(components)
    variance_ratio = np.ones(result.shape[0], dtype=float) / result.shape[0]
    return result, variance_ratio


def _extract_constituent_profile(
    atmo,
    absorber_name: str,
) -> tuple[str, np.ndarray]:
    if absorber_name not in atmo:
        msg = (
            f"Absorber '{absorber_name}' is not present in atmosphere; "
            "cannot build radiative-transfer basis"
        )
        raise ValueError(msg)

    constituent = atmo[absorber_name]
    if hasattr(constituent, "vmr"):
        return "vmr", np.asarray(constituent.vmr, dtype=float).copy()
    if hasattr(constituent, "number_density"):
        return (
            "number_density",
            np.asarray(constituent.number_density, dtype=float).copy(),
        )

    msg = (
        f"Absorber '{absorber_name}' does not expose vmr or number_density; "
        "cannot build radiative-transfer basis"
    )
    raise ValueError(msg)


def _set_constituent_profile(
    atmo,
    absorber_name: str,
    profile_kind: str,
    values: np.ndarray,
) -> None:
    if profile_kind == "vmr":
        atmo[absorber_name].vmr[:] = values
        return
    if profile_kind == "number_density":
        atmo[absorber_name].number_density[:] = values
        return

    msg = f"Unsupported constituent profile kind '{profile_kind}'"
    raise ValueError(msg)


def _mean_spectrum_from_radiance(
    radiance: xr.DataArray | xr.Dataset,
    num_wavel: int,
) -> np.ndarray:
    rad = _measurement_radiance(radiance)
    values = np.asarray(rad.to_numpy(), dtype=float)

    if values.ndim == 1:
        spectrum = values
    else:
        wavelength_axes = [
            idx for idx, size in enumerate(values.shape) if size == num_wavel
        ]
        if not wavelength_axes:
            msg = "Unable to identify wavelength axis in radiance output"
            raise ValueError(msg)

        wavelength_axis = wavelength_axes[-1]
        if wavelength_axis != values.ndim - 1:
            values = np.moveaxis(values, wavelength_axis, -1)

        reshaped = values.reshape(-1, values.shape[-1])
        finite_rows = np.all(np.isfinite(reshaped), axis=1)
        reshaped = reshaped[finite_rows]
        if reshaped.size == 0:
            msg = "Radiance output does not contain finite samples"
            raise ValueError(msg)

        spectrum = np.nanmean(reshaped, axis=0)

    if spectrum.shape[-1] != num_wavel:
        msg = "Radiance spectrum wavelength grid does not match atmosphere grid"
        raise ValueError(msg)

    return np.asarray(spectrum, dtype=float)


def _extract_species_wf_array(
    radiance: xr.DataArray | xr.Dataset,
    species_name: str,
) -> np.ndarray | None:
    if not isinstance(radiance, xr.Dataset):
        return None

    species_key = species_name.lower()

    # Prefer explicit species WF variables when available, e.g.
    # wf_o3, wf_o3_vmr, wf_o3_number_density.
    wf_variable_candidates = [
        name
        for name in radiance.data_vars
        if name.lower().startswith("wf_") and species_key in name.lower()
    ]
    if wf_variable_candidates:
        wf_arrays = [
            np.asarray(radiance[name].to_numpy(), dtype=float)
            for name in wf_variable_candidates
        ]
        if len(wf_arrays) == 1:
            return wf_arrays[0]

        # If multiple channels exist for the same species, average them so we
        # still provide one stable predictor.
        stacked = np.stack(wf_arrays, axis=0)
        return np.nanmean(stacked, axis=0)

    if "wf" not in radiance:
        return None

    wf = radiance["wf"]
    if "x" not in wf.dims:
        return None

    x_coord = np.asarray(wf.coords["x"].to_numpy())
    matches = np.array([species_key in str(value).lower() for value in x_coord])
    if not np.any(matches):
        return None

    selected = wf.isel(x=np.where(matches)[0])
    return np.asarray(selected.to_numpy(), dtype=float)


def _wf_to_mean_spectrum(
    wf_values: np.ndarray,
    num_wavel: int,
) -> np.ndarray:
    values = np.asarray(wf_values, dtype=float)
    if values.ndim == 1:
        if values.shape[0] != num_wavel:
            msg = "Weighting function spectrum wavelength size mismatch"
            raise ValueError(msg)
        return values

    wavelength_axes = [
        idx for idx, size in enumerate(values.shape) if size == num_wavel
    ]
    if not wavelength_axes:
        msg = "Unable to identify wavelength axis in weighting function output"
        raise ValueError(msg)

    wavelength_axis = wavelength_axes[-1]
    if wavelength_axis != values.ndim - 1:
        values = np.moveaxis(values, wavelength_axis, -1)

    reshaped = values.reshape(-1, values.shape[-1])
    finite_rows = np.all(np.isfinite(reshaped), axis=1)
    reshaped = reshaped[finite_rows]
    if reshaped.size == 0:
        msg = "Weighting function output does not contain finite samples"
        raise ValueError(msg)

    return np.nanmean(reshaped, axis=0)


def _species_log_radiance_derivative_component(
    radiance: xr.DataArray | xr.Dataset,
    species_name: str,
    num_wavel: int,
    min_radiance: float = 1e-30,
) -> np.ndarray | None:
    wf_values = _extract_species_wf_array(radiance, species_name)
    if wf_values is None:
        return None

    d_i = _wf_to_mean_spectrum(wf_values, num_wavel)
    i_mean = _mean_spectrum_from_radiance(radiance, num_wavel)
    safe_i = np.maximum(i_mean, min_radiance)
    return np.asarray(d_i / safe_i, dtype=float)


def _rt_cross_section_components(
    absorber_name: str,
    atmo,
    engine,
    temperatures_k: list[float] | np.ndarray,
    min_radiance: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    temps = np.atleast_1d(np.asarray(temperatures_k, dtype=float))
    if temps.size == 0:
        msg = "At least one temperature must be provided for absorber components"
        raise ValueError(msg)

    original_temperature = np.asarray(atmo.temperature_k, dtype=float).copy()
    profile_kind, original_profile = _extract_constituent_profile(atmo, absorber_name)
    num_wavel = int(atmo.num_wavel)
    components: list[np.ndarray] = []

    try:
        for temp in temps:
            atmo.temperature_k = np.full_like(
                original_temperature,
                float(temp),
                dtype=float,
            )

            _set_constituent_profile(
                atmo,
                absorber_name,
                profile_kind,
                np.zeros_like(original_profile),
            )
            without_species = _mean_spectrum_from_radiance(
                engine.calculate_radiance(atmo),
                num_wavel,
            )

            _set_constituent_profile(
                atmo,
                absorber_name,
                profile_kind,
                original_profile,
            )
            with_species = _mean_spectrum_from_radiance(
                engine.calculate_radiance(atmo),
                num_wavel,
            )

            safe_without = np.maximum(without_species, min_radiance)
            safe_with = np.maximum(with_species, min_radiance)
            components.append(np.log(safe_without) - np.log(safe_with))
    finally:
        atmo.temperature_k = original_temperature
        _set_constituent_profile(
            atmo,
            absorber_name,
            profile_kind,
            original_profile,
        )

    result = np.vstack(components)
    variance_ratio = np.ones(result.shape[0], dtype=float) / result.shape[0]
    return result, variance_ratio


def _convolve_template(
    calc_wavel: np.ndarray,
    template: np.ndarray,
    output_wavel: np.ndarray,
    fwhm: np.ndarray,
) -> np.ndarray:
    templates = np.asarray(template, dtype=float)
    is_single_template = templates.ndim == 1
    if is_single_template:
        templates = templates[np.newaxis, :]

    if templates.shape[-1] != len(calc_wavel):
        msg = "Template wavelength dimension must match calc_wavel"
        raise ValueError(msg)

    # Build integration weights once and apply to all templates that share the same grid.
    weights = np.empty((len(output_wavel), len(calc_wavel)), dtype=float)
    for idx, (center, width) in enumerate(zip(output_wavel, fwhm, strict=False)):
        gaussian = Gaussian(fwhm=max(float(width), 1e-6))
        weights[idx] = gaussian.integration_weights(float(center), calc_wavel)

    convolved = templates @ weights.T
    if is_single_template:
        return convolved[0]
    return convolved


def _manual_predictor_to_components(
    predictor_name: str,
    predictor,
    calc_wavel: np.ndarray,
) -> np.ndarray:
    calc_wavel_arr = np.asarray(calc_wavel, dtype=float)

    predictor_wavel = None
    predictor_values = None

    if isinstance(predictor, xr.DataArray):
        predictor_values = np.asarray(predictor.to_numpy(), dtype=float)
        if "wavelength" in predictor.coords:
            predictor_wavel = np.asarray(
                predictor.coords["wavelength"].to_numpy(),
                dtype=float,
            )
    elif isinstance(predictor, dict):
        if "values" not in predictor:
            msg = (
                f"manual predictor '{predictor_name}' dict must contain a "
                "'values' entry"
            )
            raise ValueError(msg)
        predictor_values = np.asarray(predictor["values"], dtype=float)
        if "wavelength" in predictor:
            predictor_wavel = np.asarray(predictor["wavelength"], dtype=float)
    elif isinstance(predictor, tuple) and len(predictor) == 2:
        predictor_wavel = np.asarray(predictor[0], dtype=float)
        predictor_values = np.asarray(predictor[1], dtype=float)
    else:
        predictor_values = np.asarray(predictor, dtype=float)

    if predictor_values is None:
        msg = f"manual predictor '{predictor_name}' could not be parsed"
        raise ValueError(msg)

    if predictor_values.ndim == 1:
        predictor_values = predictor_values[np.newaxis, :]
    elif predictor_values.ndim > 2:
        msg = (
            f"manual predictor '{predictor_name}' must be 1D or 2D "
            "(components, wavelength)"
        )
        raise ValueError(msg)

    if predictor_wavel is None:
        if predictor_values.shape[-1] != calc_wavel_arr.size:
            msg = (
                f"manual predictor '{predictor_name}' does not provide "
                "wavelengths and its length does not match calc grid"
            )
            raise ValueError(msg)
        return predictor_values

    predictor_wavel = np.asarray(predictor_wavel, dtype=float).reshape(-1)
    if predictor_wavel.size != predictor_values.shape[-1]:
        msg = (
            f"manual predictor '{predictor_name}' wavelength/value size mismatch "
            f"({predictor_wavel.size} vs {predictor_values.shape[-1]})"
        )
        raise ValueError(msg)

    valid = np.isfinite(predictor_wavel)
    if not np.any(valid):
        msg = f"manual predictor '{predictor_name}' has no finite wavelengths"
        raise ValueError(msg)

    predictor_wavel = predictor_wavel[valid]
    predictor_values = predictor_values[:, valid]

    order = np.argsort(predictor_wavel)
    predictor_wavel = predictor_wavel[order]
    predictor_values = predictor_values[:, order]

    unique_wavel, unique_idx = np.unique(predictor_wavel, return_index=True)
    predictor_wavel = unique_wavel
    predictor_values = predictor_values[:, unique_idx]

    if predictor_wavel.size < 2:
        msg = (
            f"manual predictor '{predictor_name}' must have at least two unique "
            "wavelengths"
        )
        raise ValueError(msg)

    return np.vstack(
        [
            np.interp(
                calc_wavel_arr,
                predictor_wavel,
                np.nan_to_num(component, nan=0.0),
                left=np.nan_to_num(component[0], nan=0.0),
                right=np.nan_to_num(component[-1], nan=0.0),
            )
            for component in predictor_values
        ]
    )


def _design_matrix(
    wavelengths: np.ndarray,
    calc_wavel: np.ndarray,
    xs: dict[str, np.ndarray],
    irrad: np.ndarray,
    poly_order: int,
    shift: float,
    stretch: float,
    fwhm_zero: float,
    fwhm_slope: float,
    manual_predictor_offsets: Mapping[str, float] | None = None,
    nonlinear_orders: dict[str, tuple[int, ...]] | None = None,
    residual_basis: np.ndarray | None = None,
    tilt_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    band_center = float(np.mean(wavelengths))
    transformed_wavel = (
        band_center + (1.0 + stretch) * (wavelengths - band_center) + shift
    )
    normalized_coord = (transformed_wavel - band_center) / max(
        np.ptp(wavelengths) / 2.0,
        1e-6,
    )
    fwhm = np.maximum(fwhm_zero + fwhm_slope * normalized_coord, 1e-6)

    # Convolve all XS components and irradiance in one batched call since they
    # share the same calculation and output wavelength grids.
    template_names: list[tuple[str, int]] = []
    stacked_templates: list[np.ndarray] = []
    for name, values in xs.items():
        values_arr = np.asarray(values, dtype=float)
        if values_arr.ndim == 1:
            values_arr = values_arr[np.newaxis, :]
        template_names.append((name, values_arr.shape[0]))
        stacked_templates.append(values_arr)

    stacked_templates.append(np.asarray(irrad, dtype=float)[np.newaxis, :])
    all_templates = np.vstack(stacked_templates)
    convolved_all = _convolve_template(
        calc_wavel, all_templates, transformed_wavel, fwhm
    )

    convolved_xs: dict[str, np.ndarray] = {}
    start = 0
    for name, count in template_names:
        convolved_xs[name] = convolved_all[start : start + count]
        start += count
    convolved_irrad = convolved_all[start]

    # Manual predictor wavelength offsets are global (shared across LOS).
    for name, offset_nm in (manual_predictor_offsets or {}).items():
        if name not in convolved_xs:
            continue

        shift_nm = float(offset_nm)
        if abs(shift_nm) < 1e-12:
            continue

        convolved_xs[name] = np.vstack(
            [
                np.interp(
                    transformed_wavel - shift_nm,
                    transformed_wavel,
                    component,
                    left=component[0],
                    right=component[-1],
                )
                for component in np.asarray(convolved_xs[name], dtype=float)
            ]
        )

    def _standardize_column(column: np.ndarray) -> np.ndarray:
        col = np.asarray(column, dtype=float)
        finite = np.isfinite(col)
        if not np.any(finite):
            return np.zeros_like(col)

        centered = col.copy()
        centered[finite] = centered[finite] - np.mean(centered[finite])
        std = np.std(centered[finite])
        if std > 0:
            centered[finite] = centered[finite] / std
        return centered

    basis_columns = []
    for name in xs:
        basis_columns.extend(
            _standardize_column(component) for component in convolved_xs[name]
        )
        for order in (nonlinear_orders or {}).get(name, ()):
            basis_columns.extend(
                _standardize_column(component**order)
                for component in convolved_xs[name]
            )
    basis_columns.append(_standardize_column(convolved_irrad))
    basis_columns.extend(normalized_coord**order for order in range(poly_order + 1))
    if residual_basis is not None:
        basis_columns.extend(
            _standardize_column(component) for component in residual_basis
        )
    if tilt_basis is not None:
        basis_columns.extend(_standardize_column(component) for component in tilt_basis)

    design_matrix = np.column_stack(basis_columns)
    return design_matrix, convolved_xs, convolved_irrad, transformed_wavel


def _solve_linear_coefficients(
    design_matrix: np.ndarray,
    log_radiances: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.empty(
        (log_radiances.shape[0], design_matrix.shape[1]), dtype=float
    )
    fitted = np.empty_like(log_radiances)

    for idx, spectrum in enumerate(log_radiances):
        spectrum_mask = valid_mask[idx]

        if not np.any(spectrum_mask):
            coefficients[idx] = 0.0
            fitted[idx] = np.nan
            continue

        coeff, _, _, _ = np.linalg.lstsq(
            design_matrix[spectrum_mask],
            spectrum[spectrum_mask],
            rcond=None,
        )
        if not np.all(np.isfinite(coeff)):
            coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
        coeff = np.clip(coeff, -1e12, 1e12)
        coefficients[idx] = coeff
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            projected = design_matrix @ coeff
        projected[~np.isfinite(projected)] = np.nan
        fitted[idx] = projected

    return coefficients, fitted


def _measurement_radiance(radiances: xr.DataArray | xr.Dataset) -> xr.DataArray:
    if isinstance(radiances, xr.DataArray):
        return radiances

    if "radiance" not in radiances:
        msg = "DOAS fit requires a 'radiance' variable when a Dataset is provided"
        raise ValueError(msg)

    return radiances["radiance"]


def _measurement_wf(
    radiances: xr.DataArray | xr.Dataset,
    sample_dims: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(radiances, xr.Dataset) or "wf" not in radiances:
        return None

    wf = radiances["wf"]
    if "x" not in wf.dims:
        return None

    # Collapse unsupported dimensions (e.g. stokes) so we are left with sample, wavelength, x.
    for dim in list(wf.dims):
        if dim not in {"x", "wavelength", *sample_dims}:
            wf = wf.isel({dim: 0}, drop=True)

    for dim in sample_dims:
        if dim not in wf.dims:
            return None

    wf = wf.transpose(*sample_dims, "wavelength", "x")
    wf_values = np.asarray(wf.to_numpy(), dtype=float)
    wf_values = wf_values.reshape((-1, wf_values.shape[-2], wf_values.shape[-1]))

    x_values = np.asarray(wf.coords["x"].to_numpy())
    return wf_values, x_values


class DOASFitter:
    def __init__(
        self,
        radiances: xr.DataArray | xr.Dataset,
        anc,
        *,
        optical: dict[str, object] | None = None,
        num_pca: dict[str, int] | None = None,
        absorber_basis_method: str | dict[str, str] = "optical",
        absorber_cross_section_predictors: dict[str, bool] | None = None,
        absorber_derivative_predictors: dict[str, bool] | None = None,
        absorber_temperatures: dict[str, float | list[float] | str] | None = None,
        absorber_nonlinear_orders: dict[str, list[int]] | None = None,
        manual_predictors: Mapping[str, object] | None = None,
        cos_sza: np.ndarray | xr.DataArray | None = None,
        radiance_filter: list[float] | np.ndarray | None = None,
        filter: list[float] | np.ndarray | None = None,
        poly_order: int = 3,
        calc_margin: float = 5.0,
        calc_spacing: float = 0.001,
        initial_params: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        residual_pca_components: int = 0,
        tilt_pca_components: int = 0,
    ):
        self._optical = optical or {
            "o3": sk.optical.O3DBM(),
            "bro": sk.optical.HITRANUV("BrO"),
            "no2": sk.optical.NO2Vandaele(),
            "o2o2": sk.optical.HITRANCollision("o2o2"),
            # "oclo": sk.optical.OClOGeisa(),
            # "so2": sk.optical.HITRANUV("SO2")
        }
        self._num_pca = num_pca or {
            "o3": 3,
            "no2": 2,
            "o2o2": 2,
            "bro": 1,
            "oclo": 1,
            "SO2": 1,
        }
        default_absorber_temperatures: dict[str, float | list[float] | str] = {
            "bro": 228.0,
            "o3": [203.0, 223.0, 243.0],
            "no2": 220.0,
            "o2o2": 220.0,
            "oclo": 220.0,
            "so2": 220.0,
        }
        if absorber_temperatures is not None:
            default_absorber_temperatures.update(absorber_temperatures)
        self._absorber_temperatures = default_absorber_temperatures

        if isinstance(absorber_basis_method, str):
            self._absorber_basis_method = dict.fromkeys(
                self._optical,
                absorber_basis_method,
            )
        else:
            self._absorber_basis_method = {
                name: absorber_basis_method.get(name, "optical")
                for name in self._optical
            }

        if absorber_cross_section_predictors is None:
            self._absorber_cross_section_predictors = dict.fromkeys(self._optical, True)
        else:
            self._absorber_cross_section_predictors = {
                name: bool(absorber_cross_section_predictors.get(name, True))
                for name in self._optical
            }

        if absorber_derivative_predictors is None:
            self._absorber_derivative_predictors = dict.fromkeys(self._optical, False)
        else:
            self._absorber_derivative_predictors = {
                name: bool(absorber_derivative_predictors.get(name, False))
                for name in self._optical
            }

        self._absorber_derivative_component_count: dict[str, int] = dict.fromkeys(
            self._optical, 0
        )

        self._absorber_nonlinear_orders: dict[str, tuple[int, ...]] = {}
        if absorber_nonlinear_orders is not None:
            for name, orders in absorber_nonlinear_orders.items():
                if name not in self._optical:
                    msg = f"Unknown absorber '{name}' in absorber_nonlinear_orders"
                    raise ValueError(msg)

                valid_orders = []
                for order in orders:
                    if int(order) < 2:
                        msg = (
                            "absorber_nonlinear_orders entries must be >= 2 "
                            "(e.g., [2, 3])"
                        )
                        raise ValueError(msg)
                    valid_orders.append(int(order))

                unique_sorted = tuple(sorted(set(valid_orders)))
                if unique_sorted:
                    self._absorber_nonlinear_orders[name] = unique_sorted

        self._xs_by_sample: dict[str, np.ndarray] = {}
        self._manual_predictors: dict[str, object] = dict(manual_predictors or {})
        self._manual_predictor_names = tuple(self._manual_predictors)
        self._manual_predictor_param_index = {
            name: 4 + idx for idx, name in enumerate(self._manual_predictor_names)
        }
        self._manual_predictor_offsets: dict[str, float] = dict.fromkeys(
            self._manual_predictor_names, 0.0
        )
        self._cos_sza_input = cos_sza

        raw_filter = radiance_filter if radiance_filter is not None else filter
        self._radiance_filter_weights = _normalized_filter_weights(raw_filter)
        self._poly_order = poly_order
        base_initial = np.array(
            [0.0, 0.0, 0.2, 0.0] if initial_params is None else initial_params,
            dtype=float,
        )
        if bounds is None:
            base_lower = np.array([-0.5, -0.05, 1e-4, -1.0], dtype=float)
            base_upper = np.array([0.5, 0.05, 3.0, 1.0], dtype=float)
        else:
            base_lower = np.asarray(bounds[0], dtype=float)
            base_upper = np.asarray(bounds[1], dtype=float)

        num_manual = len(self._manual_predictor_names)
        if num_manual == 0:
            self._initial_params = base_initial
            self._bounds = (base_lower, base_upper)
        else:
            total_params = 4 + num_manual
            if base_initial.size == 4:
                base_initial = np.concatenate(
                    [base_initial, np.zeros(num_manual, dtype=float)]
                )
            elif base_initial.size != total_params:
                msg = (
                    "initial_params must contain either 4 base values or "
                    f"{total_params} values including manual predictor offsets"
                )
                raise ValueError(msg)

            default_offset_bound_nm = 0.25
            if base_lower.size == 4:
                base_lower = np.concatenate(
                    [
                        base_lower,
                        np.full(num_manual, -default_offset_bound_nm, dtype=float),
                    ]
                )
            if base_upper.size == 4:
                base_upper = np.concatenate(
                    [
                        base_upper,
                        np.full(num_manual, default_offset_bound_nm, dtype=float),
                    ]
                )

            if base_lower.size != total_params or base_upper.size != total_params:
                msg = (
                    "bounds must contain either 4 base values or "
                    f"{total_params} values including manual predictor offsets"
                )
                raise ValueError(msg)

            self._initial_params = base_initial
            self._bounds = (base_lower, base_upper)
        self._residual_pca_components = max(0, int(residual_pca_components))
        self._tilt_pca_components = max(0, int(tilt_pca_components))
        self._residual_basis: np.ndarray | None = None
        self._residual_variance_ratio = np.array([], dtype=float)
        self._convolved_modelled_radiance: np.ndarray | None = None
        self._tilt_pca_basis: np.ndarray | None = None
        self._tilt_pca_variance_ratio = np.array([], dtype=float)
        self._cached_design_matrix: np.ndarray | None = None
        self._cached_convolved_xs: dict[str, np.ndarray] | None = None
        self._cached_convolved_irrad: np.ndarray | None = None
        self._cached_fit_wavelength: np.ndarray | None = None

        reference_radiance = _measurement_radiance(radiances)
        self._reference_wavelengths = reference_radiance.wavelength.to_numpy()
        self._cos_sza = self._prepare_cos_sza(reference_radiance)

        min_wavel = reference_radiance.wavelength.min()
        max_wavel = reference_radiance.wavelength.max()
        self._calc_wavelength = np.arange(
            min_wavel - calc_margin, max_wavel + calc_margin, calc_spacing
        )

        if len(reference_radiance.tangent_altitude.to_numpy()) < 2:
            msg = "Not enough tangent altitudes for retrieval"
            raise ValueError(msg)

        model_geo = sk.Geometry1D(
            0.45,
            0.0,
            6371000,
            reference_radiance.tangent_altitude.to_numpy(),
        )
        atmo = sk.Atmosphere(
            model_geo,
            sk.Config(),
            self._calc_wavelength,
            calculate_derivatives=any(self._absorber_derivative_predictors.values()),
        )
        anc.add_to_atmosphere(atmo)
        atmo["o3"].vmr[:] *= 0.0
        model_altitudes = np.asarray(atmo.model_geometry.altitudes(), dtype=float)
        model_temperatures = np.asarray(atmo.temperature_k, dtype=float)
        tangent_altitudes = np.asarray(
            reference_radiance.tangent_altitude.to_numpy(),
            dtype=float,
        )
        self._tangent_temperatures = np.interp(
            tangent_altitudes,
            model_altitudes,
            model_temperatures,
        )

        viewing_geo = sk.ViewingGeometry()

        for talt in reference_radiance.tangent_altitude.to_numpy():
            viewing_geo.add_ray(sk.TangentAltitudeSolar(talt, 0.0, 200000, 0.45))

        engine = sk.Engine(sk.Config(), model_geo, viewing_geo)

        modelled_radiance = engine.calculate_radiance(atmo)

        self._xs = {}
        self._xs_variance_ratio = {}
        for name, optical_quantity in self._optical.items():
            configured_temperatures = self._absorber_temperatures.get(name, 220.0)
            include_cross_section_predictor = bool(
                self._absorber_cross_section_predictors.get(name, True)
            )

            if not include_cross_section_predictor:
                self._xs[name] = np.zeros((0, int(atmo.num_wavel)), dtype=float)
                self._xs_variance_ratio[name] = np.array([], dtype=float)
                continue

            configured_basis_method = (
                str(self._absorber_basis_method.get(name, "optical")).strip().lower()
            )
            if configured_basis_method in {
                "optical",
                "optically_thin",
                "optically-thin",
                "thin",
            }:
                basis_from_rt = False
            elif configured_basis_method in {
                "radiative_transfer",
                "radiative-transfer",
                "rt",
                "full_rt",
                "full-rt",
            }:
                basis_from_rt = True
            else:
                msg = (
                    f"Unsupported absorber_basis_method '{configured_basis_method}' "
                    f"for absorber '{name}'"
                )
                raise ValueError(msg)

            if isinstance(configured_temperatures, str):
                if configured_temperatures.lower() != "tangent":
                    msg = "absorber_temperatures supports string mode 'tangent' only"
                    raise ValueError(msg)

                sample_components = []
                sample_variance = []
                for temperature in self._tangent_temperatures:
                    if basis_from_rt:
                        components, variance_ratio = _rt_cross_section_components(
                            name,
                            atmo,
                            engine,
                            [float(temperature)],
                        )
                    else:
                        components, variance_ratio = (
                            _temperature_cross_section_components(
                                optical_quantity,
                                atmo,
                                [float(temperature)],
                            )
                        )
                    sample_components.append(components)
                    sample_variance.append(variance_ratio)

                self._xs_by_sample[name] = np.stack(sample_components, axis=0)
                self._xs[name] = np.mean(self._xs_by_sample[name], axis=0)
                self._xs_variance_ratio[name] = np.mean(
                    np.stack(sample_variance, axis=0),
                    axis=0,
                )
                continue

            if isinstance(configured_temperatures, int | float):
                configured_temperatures = [float(configured_temperatures)]

            if basis_from_rt:
                components, variance_ratio = _rt_cross_section_components(
                    name,
                    atmo,
                    engine,
                    configured_temperatures,
                )
            else:
                components, variance_ratio = _temperature_cross_section_components(
                    optical_quantity,
                    atmo,
                    configured_temperatures,
                )
            self._xs[name] = components
            self._xs_variance_ratio[name] = variance_ratio

        for predictor_name, predictor in self._manual_predictors.items():
            if predictor_name in self._xs:
                msg = (
                    f"manual predictor '{predictor_name}' conflicts with an existing "
                    "absorber name"
                )
                raise ValueError(msg)

            components = _manual_predictor_to_components(
                predictor_name,
                predictor,
                self._calc_wavelength,
            )
            self._xs[predictor_name] = components
            self._xs_variance_ratio[predictor_name] = (
                np.ones(
                    components.shape[0],
                    dtype=float,
                )
                / components.shape[0]
            )

        if any(self._absorber_derivative_predictors.values()):
            for name, enabled in self._absorber_derivative_predictors.items():
                if not enabled:
                    continue

                derivative_component = _species_log_radiance_derivative_component(
                    modelled_radiance,
                    name,
                    int(atmo.num_wavel),
                )
                if derivative_component is None:
                    continue

                derivative_component = np.asarray(derivative_component, dtype=float)[
                    np.newaxis, :
                ]
                self._absorber_derivative_component_count[name] = 1

                self._xs[name] = np.vstack([self._xs[name], derivative_component])
                if name in self._xs_by_sample:
                    num_samples = self._xs_by_sample[name].shape[0]
                    repeated = np.repeat(
                        derivative_component[np.newaxis, :, :],
                        num_samples,
                        axis=0,
                    )
                    self._xs_by_sample[name] = np.concatenate(
                        [self._xs_by_sample[name], repeated],
                        axis=1,
                    )

                old_ratio = np.asarray(self._xs_variance_ratio[name], dtype=float)
                new_ratio = np.append(old_ratio, 0.0)
                ratio_sum = np.sum(new_ratio)
                if ratio_sum > 0:
                    new_ratio = new_ratio / ratio_sum
                self._xs_variance_ratio[name] = new_ratio

        self._irrad = sk.solar.SolarModel().irradiance(self._calc_wavelength)

        filtered_radiances = _filter_measurement_input(
            radiances,
            self._radiance_filter_weights,
        )
        calibration_radiance, _, calibration_mask, calibration_log = self._fit_inputs(
            filtered_radiances
        )

        def residual(params: np.ndarray) -> np.ndarray:
            manual_offsets = {
                name: float(params[idx])
                for name, idx in self._manual_predictor_param_index.items()
            }
            design_matrices, _, _, _ = self._design_matrices_for_samples(
                calibration_radiance.wavelength.to_numpy(),
                calibration_log.shape[0],
                shift=float(params[0]),
                stretch=float(params[1]),
                fwhm_zero=float(params[2]),
                fwhm_slope=float(params[3]),
                manual_predictor_offsets=manual_offsets,
                residual_basis=self._residual_basis,
                tilt_basis=None,
            )
            _, fitted = self._solve_linear_coefficients_for_samples(
                design_matrices,
                calibration_log,
                calibration_mask,
            )
            return (
                calibration_log[calibration_mask] - fitted[calibration_mask]
            ).ravel()

        self._nonlinear_fit = least_squares(
            residual,
            self._initial_params,
            bounds=self._bounds,
        )

        if self._residual_pca_components > 0:
            initial_design_matrices, _, _, _ = self._design_matrices_for_samples(
                calibration_radiance.wavelength.to_numpy(),
                calibration_log.shape[0],
                shift=float(self._nonlinear_fit.x[0]),
                stretch=float(self._nonlinear_fit.x[1]),
                fwhm_zero=float(self._nonlinear_fit.x[2]),
                fwhm_slope=float(self._nonlinear_fit.x[3]),
                manual_predictor_offsets={
                    name: float(self._nonlinear_fit.x[idx])
                    for name, idx in self._manual_predictor_param_index.items()
                },
                residual_basis=None,
                tilt_basis=None,
            )
            _, initial_fitted = self._solve_linear_coefficients_for_samples(
                initial_design_matrices,
                calibration_log,
                calibration_mask,
            )
            calibration_residual = np.where(
                calibration_mask,
                calibration_log - initial_fitted,
                0.0,
            )

            num_components = min(
                self._residual_pca_components,
                calibration_residual.shape[0],
                calibration_residual.shape[1],
            )
            if num_components > 0:
                residual_pca = PCA(n_components=num_components, svd_solver="full")
                residual_pca.fit(calibration_residual)
                self._residual_basis = residual_pca.components_
                self._residual_variance_ratio = residual_pca.explained_variance_ratio_

                self._nonlinear_fit = least_squares(
                    residual,
                    self._nonlinear_fit.x,
                    bounds=self._bounds,
                )

        self._shift = float(self._nonlinear_fit.x[0])
        self._stretch = float(self._nonlinear_fit.x[1])
        self._fwhm_zero = float(self._nonlinear_fit.x[2])
        self._fwhm_slope = float(self._nonlinear_fit.x[3])
        self._manual_predictor_offsets = {
            name: float(self._nonlinear_fit.x[idx])
            for name, idx in self._manual_predictor_param_index.items()
        }

        self._convolved_modelled_radiance = self._convolve_modelled_radiance(
            modelled_radiance
        )
        if self._convolved_modelled_radiance is not None:
            _, tilt_spectrum = self._calculate_tilt_spectrum(
                self._convolved_modelled_radiance,
                self._reference_wavelengths,
                self._cos_sza,
            )
            self._tilt_pca_basis, self._tilt_pca_variance_ratio = (
                self._tilt_pca_from_spectrum(
                    tilt_spectrum,
                    self._tilt_pca_components,
                )
            )

        # Cache the design matrix after initialization. Subsequent fit calls
        # reuse it because the fitted nonlinear params and wavelengths are fixed.
        if not self._uses_sample_specific_xs:
            self._build_cached_design_matrix()

    @property
    def _uses_sample_specific_xs(self) -> bool:
        return bool(self._xs_by_sample)

    def _validate_sample_count(self, num_samples: int) -> None:
        if not self._uses_sample_specific_xs:
            return

        expected = len(self._tangent_temperatures)
        if num_samples != expected:
            msg = (
                "DOASFitter configured with tangent-temperature absorbers requires "
                f"{expected} samples, got {num_samples}"
            )
            raise ValueError(msg)

    def _sample_xs(self, sample_index: int) -> dict[str, np.ndarray]:
        sample_xs: dict[str, np.ndarray] = {}
        for name, components in self._xs.items():
            by_sample = self._xs_by_sample.get(name)
            sample_xs[name] = (
                by_sample[sample_index] if by_sample is not None else components
            )
        return sample_xs

    def _design_matrices_for_samples(
        self,
        wavelengths: np.ndarray,
        num_samples: int,
        *,
        shift: float,
        stretch: float,
        fwhm_zero: float,
        fwhm_slope: float,
        manual_predictor_offsets: Mapping[str, float] | None,
        residual_basis: np.ndarray | None,
        tilt_basis: np.ndarray | None,
    ) -> tuple[
        list[np.ndarray],
        list[dict[str, np.ndarray]],
        np.ndarray,
        np.ndarray,
    ]:
        if not self._uses_sample_specific_xs:
            design_matrix, convolved_xs, convolved_irrad, fit_wavelength = (
                _design_matrix(
                    wavelengths,
                    self._calc_wavelength,
                    self._xs,
                    self._irrad,
                    self._poly_order,
                    shift=shift,
                    stretch=stretch,
                    fwhm_zero=fwhm_zero,
                    fwhm_slope=fwhm_slope,
                    manual_predictor_offsets=manual_predictor_offsets,
                    nonlinear_orders=self._absorber_nonlinear_orders,
                    residual_basis=residual_basis,
                    tilt_basis=tilt_basis,
                )
            )
            return [design_matrix], [convolved_xs], convolved_irrad, fit_wavelength

        self._validate_sample_count(num_samples)
        design_matrices: list[np.ndarray] = []
        convolved_xs_list: list[dict[str, np.ndarray]] = []
        convolved_irrad: np.ndarray | None = None
        fit_wavelength: np.ndarray | None = None

        for sample_idx in range(num_samples):
            (
                design_matrix,
                convolved_xs,
                sample_convolved_irrad,
                sample_fit_wavelength,
            ) = _design_matrix(
                wavelengths,
                self._calc_wavelength,
                self._sample_xs(sample_idx),
                self._irrad,
                self._poly_order,
                shift=shift,
                stretch=stretch,
                fwhm_zero=fwhm_zero,
                fwhm_slope=fwhm_slope,
                manual_predictor_offsets=manual_predictor_offsets,
                nonlinear_orders=self._absorber_nonlinear_orders,
                residual_basis=residual_basis,
                tilt_basis=tilt_basis,
            )
            design_matrices.append(design_matrix)
            convolved_xs_list.append(convolved_xs)
            if convolved_irrad is None:
                convolved_irrad = sample_convolved_irrad
            if fit_wavelength is None:
                fit_wavelength = sample_fit_wavelength

        if convolved_irrad is None or fit_wavelength is None:
            msg = "Unable to build sample-specific design matrices"
            raise ValueError(msg)

        return design_matrices, convolved_xs_list, convolved_irrad, fit_wavelength

    def _solve_linear_coefficients_for_samples(
        self,
        design_matrices: list[np.ndarray],
        log_radiances: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(design_matrices) == 1:
            return _solve_linear_coefficients(
                design_matrices[0],
                log_radiances,
                valid_mask,
            )

        coeff_rows = []
        fitted_rows = []
        for sample_idx, design_matrix in enumerate(design_matrices):
            coeff, fitted = _solve_linear_coefficients(
                design_matrix,
                log_radiances[sample_idx : sample_idx + 1],
                valid_mask[sample_idx : sample_idx + 1],
            )
            coeff_rows.append(coeff[0])
            fitted_rows.append(fitted[0])

        return np.vstack(coeff_rows), np.vstack(fitted_rows)

    def _build_cached_design_matrix(self) -> None:
        design_matrix, convolved_xs, convolved_irrad, fit_wavelength = _design_matrix(
            self._reference_wavelengths,
            self._calc_wavelength,
            self._xs,
            self._irrad,
            self._poly_order,
            shift=self._shift,
            stretch=self._stretch,
            fwhm_zero=self._fwhm_zero,
            fwhm_slope=self._fwhm_slope,
            manual_predictor_offsets=self._manual_predictor_offsets,
            nonlinear_orders=self._absorber_nonlinear_orders,
            residual_basis=self._residual_basis,
            tilt_basis=self._tilt_pca_basis,
        )
        self._cached_design_matrix = design_matrix
        self._cached_convolved_xs = convolved_xs
        self._cached_convolved_irrad = convolved_irrad
        self._cached_fit_wavelength = fit_wavelength

    def fit(self, radiances: xr.DataArray | xr.Dataset) -> xr.Dataset:
        filtered_radiances = _filter_measurement_input(
            radiances,
            self._radiance_filter_weights,
        )
        radiance_data, radiance_values, valid_mask, log_radiances = self._fit_inputs(
            filtered_radiances
        )

        sample_dims = [dim for dim in radiance_data.dims if dim != "wavelength"]

        if self._uses_sample_specific_xs:
            design_matrices, convolved_xs_list, convolved_irrad, fit_wavelength = (
                self._design_matrices_for_samples(
                    radiance_data.wavelength.to_numpy(),
                    log_radiances.shape[0],
                    shift=self._shift,
                    stretch=self._stretch,
                    fwhm_zero=self._fwhm_zero,
                    fwhm_slope=self._fwhm_slope,
                    manual_predictor_offsets=self._manual_predictor_offsets,
                    residual_basis=self._residual_basis,
                    tilt_basis=self._tilt_pca_basis,
                )
            )
            coefficients, fitted_log_radiance = (
                self._solve_linear_coefficients_for_samples(
                    design_matrices,
                    log_radiances,
                    valid_mask,
                )
            )
        else:
            if (
                self._cached_design_matrix is None
                or self._cached_convolved_xs is None
                or self._cached_convolved_irrad is None
                or self._cached_fit_wavelength is None
            ):
                self._build_cached_design_matrix()

            design_matrices = [self._cached_design_matrix]
            convolved_xs_list = [self._cached_convolved_xs]
            convolved_irrad = self._cached_convolved_irrad
            fit_wavelength = self._cached_fit_wavelength
            coefficients, fitted_log_radiance = _solve_linear_coefficients(
                self._cached_design_matrix,
                log_radiances,
                valid_mask,
            )

        basis_names = []
        for name, components in self._xs.items():
            deriv_count = self._absorber_derivative_component_count.get(name, 0)
            base_count = max(components.shape[0] - deriv_count, 0)
            basis_names.extend(f"{name}_pca_{idx}" for idx in range(base_count))
            basis_names.extend(f"{name}_dlogI_{idx}" for idx in range(deriv_count))
            for order in self._absorber_nonlinear_orders.get(name, ()):
                basis_names.extend(
                    f"{name}_nl_pca_{idx}_pow_{order}"
                    for idx in range(components.shape[0])
                )
        basis_names.extend(
            ["irrad"] + [f"poly_{idx}" for idx in range(self._poly_order + 1)]
        )
        if self._residual_basis is not None:
            basis_names.extend(
                f"residual_pca_{idx}" for idx in range(self._residual_basis.shape[0])
            )
        if self._tilt_pca_basis is not None:
            basis_names.extend(
                f"tilt_pca_{idx}" for idx in range(self._tilt_pca_basis.shape[0])
            )

        fit_dims = [dim for dim in radiance_data.dims if dim != "wavelength"]
        if not fit_dims:
            fit_dims = ["sample"]

        coords = {
            dim: radiance_data.coords[dim]
            for dim in fit_dims
            if dim in radiance_data.coords
        }
        if "sample" in fit_dims:
            coords["sample"] = [0]
        elif fit_dims[0] not in coords:
            coords[fit_dims[0]] = np.arange(radiance_values.shape[0])

        for coord_name, coord in radiance_data.coords.items():
            if coord_name not in coords and fit_dims[0] in coord.dims:
                coords[coord_name] = coord

        fit_coord_name = fit_dims[0]
        wavelengths = radiance_data.wavelength.to_numpy()
        band_center = float(np.mean(wavelengths))
        fitted_fwhm = np.maximum(
            self._fwhm_zero
            + self._fwhm_slope
            * ((fit_wavelength - band_center) / max(np.ptp(wavelengths) / 2.0, 1e-6)),
            1e-6,
        )

        bro_indices = [
            idx for idx, name in enumerate(basis_names) if name.startswith("bro_pca_")
        ]
        bro_index = bro_indices[0] if bro_indices else None
        if bro_indices:
            if len(design_matrices) == 1:
                bro_contribution = (
                    coefficients[:, bro_indices] @ design_matrices[0][:, bro_indices].T
                )
            else:
                bro_contribution = np.zeros_like(fitted_log_radiance)
                for sample_idx, sample_design_matrix in enumerate(design_matrices):
                    bro_contribution[sample_idx] = (
                        coefficients[sample_idx, bro_indices]
                        @ sample_design_matrix[:, bro_indices].T
                    )
        else:
            bro_contribution = np.zeros_like(fitted_log_radiance)

        bro_coefficient = (
            coefficients[:, bro_index]
            if bro_index is not None
            else np.zeros(coefficients.shape[0], dtype=float)
        )

        oclo_indices = [
            idx for idx, name in enumerate(basis_names) if name.startswith("oclo_pca_")
        ]
        oclo_index = oclo_indices[0] if oclo_indices else None
        oclo_coefficient = (
            coefficients[:, oclo_index]
            if oclo_index is not None
            else np.zeros(coefficients.shape[0], dtype=float)
        )

        # Propagate fit residual variance into linear coefficient uncertainty.
        # Var(c) ~= s^2 * (A^T A)^-1 for each LOS sample.
        bro_coefficient_sigma = np.full(coefficients.shape[0], np.nan, dtype=float)
        oclo_coefficient_sigma = np.full(coefficients.shape[0], np.nan, dtype=float)
        for sample_idx in range(coefficients.shape[0]):
            sample_mask = valid_mask[sample_idx]
            if not np.any(sample_mask):
                continue

            sample_design_matrix = (
                design_matrices[0]
                if len(design_matrices) == 1
                else design_matrices[sample_idx]
            )
            a_matrix = sample_design_matrix[sample_mask]
            n_obs, n_param = a_matrix.shape
            if n_obs == 0:
                continue

            sample_residual = (
                log_radiances[sample_idx, sample_mask]
                - fitted_log_radiance[sample_idx, sample_mask]
            )
            finite_resid = np.isfinite(sample_residual)
            if not np.any(finite_resid):
                continue

            residual_ss = float(np.sum(sample_residual[finite_resid] ** 2))
            dof = max(int(np.count_nonzero(finite_resid)) - n_param, 1)
            s2 = max(residual_ss / dof, 0.0)

            ata = a_matrix.T @ a_matrix
            coeff_cov = s2 * np.linalg.pinv(ata, rcond=1e-10)
            coeff_sigma = np.sqrt(np.clip(np.diag(coeff_cov), 0.0, np.inf))

            if bro_index is not None and bro_index < coeff_sigma.size:
                bro_coefficient_sigma[sample_idx] = coeff_sigma[bro_index]
            if oclo_index is not None and oclo_index < coeff_sigma.size:
                oclo_coefficient_sigma[sample_idx] = coeff_sigma[oclo_index]

        bro_coefficient_sigma = np.where(
            np.isfinite(bro_coefficient_sigma) & (bro_coefficient_sigma > 0),
            bro_coefficient_sigma,
            1e-3,
        )
        oclo_coefficient_sigma = np.where(
            np.isfinite(oclo_coefficient_sigma) & (oclo_coefficient_sigma > 0),
            oclo_coefficient_sigma,
            1e-3,
        )

        bro_coefficient_wf = None
        oclo_coefficient_wf = None
        x_coord = None
        wf_data = _measurement_wf(filtered_radiances, sample_dims)
        if wf_data is not None and (bro_index is not None or oclo_index is not None):
            wf_values, x_values = wf_data
            x_coord = x_values

            num_sample = coefficients.shape[0]
            num_x = wf_values.shape[-1]
            if bro_index is not None:
                bro_coefficient_wf = np.zeros((num_sample, num_x), dtype=float)
            if oclo_index is not None:
                oclo_coefficient_wf = np.zeros((num_sample, num_x), dtype=float)
            radiance_floor = 1e-6

            for sample_idx in range(num_sample):
                sample_mask = valid_mask[sample_idx]
                if not np.any(sample_mask):
                    continue

                sample_design_matrix = (
                    design_matrices[0]
                    if len(design_matrices) == 1
                    else design_matrices[sample_idx]
                )
                a_matrix = sample_design_matrix[sample_mask]
                if a_matrix.shape[0] == 0:
                    continue

                # coeff = (X^T X)^-1 X^T Y, with Y = log(I)
                # dcoeff/dx = (X^T X)^-1 X^T (dY/dx), and dY/dx = (1/I) dI/dx.
                linear_map = np.linalg.pinv(a_matrix, rcond=1e-10)

                base_radiance = np.maximum(
                    radiance_values[sample_idx, sample_mask], radiance_floor
                )
                d_i_dx = wf_values[sample_idx, sample_mask, :]
                d_y_dx = np.zeros_like(d_i_dx)
                denom = np.broadcast_to(base_radiance[:, np.newaxis], d_i_dx.shape)
                finite_mask = np.isfinite(d_i_dx) & np.isfinite(denom)
                d_y_dx[finite_mask] = d_i_dx[finite_mask] / denom[finite_mask]

                dcoeff_dx = linear_map @ d_y_dx
                if not np.all(np.isfinite(dcoeff_dx)):
                    dcoeff_dx = np.linalg.lstsq(a_matrix, d_y_dx, rcond=None)[0]
                if bro_index is not None and bro_coefficient_wf is not None:
                    bro_coefficient_wf[sample_idx, :] = dcoeff_dx[bro_index, :]
                if oclo_index is not None and oclo_coefficient_wf is not None:
                    oclo_coefficient_wf[sample_idx, :] = dcoeff_dx[oclo_index, :]

        residual = np.where(valid_mask, log_radiances - fitted_log_radiance, np.nan)
        residual_with_bro_added = np.where(
            valid_mask, residual + bro_contribution, np.nan
        )

        dataset = xr.Dataset(
            data_vars={
                "log_radiance": (
                    (fit_coord_name, "wavelength"),
                    np.where(valid_mask, log_radiances, np.nan),
                ),
                "fitted_log_radiance": (
                    (fit_coord_name, "wavelength"),
                    fitted_log_radiance,
                ),
                "residual": (
                    (fit_coord_name, "wavelength"),
                    residual,
                ),
                "bro_contribution": (
                    (fit_coord_name, "wavelength"),
                    np.where(valid_mask, bro_contribution, np.nan),
                ),
                "bro_coefficient": (
                    (fit_coord_name,),
                    bro_coefficient,
                ),
                "bro_coefficient_sigma": (
                    (fit_coord_name,),
                    bro_coefficient_sigma,
                ),
                "oclo_coefficient": (
                    (fit_coord_name,),
                    oclo_coefficient,
                ),
                "oclo_coefficient_sigma": (
                    (fit_coord_name,),
                    oclo_coefficient_sigma,
                ),
                "residual_with_bro_added": (
                    (fit_coord_name, "wavelength"),
                    residual_with_bro_added,
                ),
                "coefficients": (
                    (fit_coord_name, "basis"),
                    coefficients,
                ),
                "convolved_irrad": ("wavelength", convolved_irrad),
                "fit_wavelength": ("wavelength", fit_wavelength),
                "fitted_fwhm": ("wavelength", fitted_fwhm),
                "shift": self._shift,
                "stretch": self._stretch,
                "fwhm_zero": self._fwhm_zero,
                "fwhm_slope": self._fwhm_slope,
                "cost": self._nonlinear_fit.cost,
            },
            coords={
                **coords,
                "calc_wavelength": self._calc_wavelength,
                "wavelength": wavelengths,
                "basis": basis_names,
            },
        )

        if self._manual_predictor_names:
            dataset = dataset.assign_coords(
                manual_predictor=np.array(self._manual_predictor_names, dtype=str)
            )
            dataset["manual_predictor_spectral_offset_nm"] = (
                ("manual_predictor",),
                np.array(
                    [
                        self._manual_predictor_offsets.get(name, 0.0)
                        for name in self._manual_predictor_names
                    ],
                    dtype=float,
                ),
            )

        if bro_coefficient_wf is not None and x_coord is not None:
            dataset = dataset.assign_coords(x=x_coord)
            dataset["bro_coefficient_wf"] = (
                (fit_coord_name, "x"),
                bro_coefficient_wf,
            )

        if oclo_coefficient_wf is not None and x_coord is not None:
            if "x" not in dataset.coords:
                dataset = dataset.assign_coords(x=x_coord)
            dataset["oclo_coefficient_wf"] = (
                (fit_coord_name, "x"),
                oclo_coefficient_wf,
            )

        modelled_conv = None
        if self._convolved_modelled_radiance is not None:
            modelled_conv = self._convolved_modelled_radiance
            if modelled_conv.shape[0] == 1 and radiance_values.shape[0] > 1:
                modelled_conv = np.repeat(
                    modelled_conv, radiance_values.shape[0], axis=0
                )
            if modelled_conv.shape[0] != radiance_values.shape[0]:
                modelled_conv = None

        if modelled_conv is not None:
            dataset["convolved_modelled_radiance"] = (
                (fit_coord_name, "wavelength"),
                modelled_conv,
            )

            tilt_polynomial, tilt_spectrum = self._calculate_tilt_spectrum(
                modelled_conv,
                wavelengths,
                self._cos_sza,
            )

            dataset["tilt_polynomial"] = (
                (fit_coord_name, "wavelength"),
                tilt_polynomial,
            )
            dataset["tilt_spectrum"] = (
                (fit_coord_name, "wavelength"),
                tilt_spectrum,
            )

        if self._tilt_pca_basis is not None:
            dataset = dataset.assign_coords(
                tilt_component=np.arange(self._tilt_pca_basis.shape[0])
            )
            dataset["tilt_pca_basis"] = (
                ("tilt_component", "wavelength"),
                self._tilt_pca_basis,
            )
            dataset["tilt_pca_variance_ratio"] = (
                ("tilt_component",),
                self._tilt_pca_variance_ratio,
            )

        for name in self._xs:
            if len(convolved_xs_list) == 1:
                values = convolved_xs_list[0][name]
            else:
                values = np.stack(
                    [
                        sample_convolved_xs[name]
                        for sample_convolved_xs in convolved_xs_list
                    ],
                    axis=0,
                )

            component_dim = f"{name}_component"
            num_components = self._xs[name].shape[0]
            dataset = dataset.assign_coords({component_dim: np.arange(num_components)})
            dataset[f"xs_pca_{name}"] = (
                (component_dim, "calc_wavelength"),
                self._xs[name],
            )
            dataset[f"xs_pca_variance_ratio_{name}"] = (
                (component_dim,),
                self._xs_variance_ratio[name],
            )
            if len(convolved_xs_list) == 1:
                dataset[f"convolved_xs_{name}"] = (
                    (component_dim, "wavelength"),
                    values,
                )
            else:
                dataset[f"convolved_xs_{name}"] = (
                    (fit_coord_name, component_dim, "wavelength"),
                    values,
                )

        if self._residual_basis is not None:
            dataset = dataset.assign_coords(
                residual_component=np.arange(self._residual_basis.shape[0])
            )
            dataset["residual_pca_basis"] = (
                ("residual_component", "wavelength"),
                self._residual_basis,
            )
            dataset["residual_pca_variance_ratio"] = (
                ("residual_component",),
                self._residual_variance_ratio,
            )

        return dataset

    def _fit_inputs(
        self,
        radiances: xr.DataArray | xr.Dataset,
    ) -> tuple[xr.DataArray, np.ndarray, np.ndarray, np.ndarray]:
        radiance_data = _measurement_radiance(radiances)
        wavelengths = radiance_data.wavelength.to_numpy()
        if not np.array_equal(wavelengths, self._reference_wavelengths):
            msg = "DOASFitter.fit requires wavelengths to match the initialization radiances"
            raise ValueError(msg)

        radiance_values = np.asarray(radiance_data.to_numpy(), dtype=float)
        if radiance_values.ndim == 1:
            radiance_values = radiance_values[np.newaxis, :]

        valid_mask = np.isfinite(radiance_values) & (radiance_values > 0)
        if not np.any(valid_mask):
            msg = "No finite positive radiances are available for DOAS fitting"
            raise ValueError(msg)

        safe_radiance_values = np.where(valid_mask, radiance_values, 1.0)
        return radiance_data, radiance_values, valid_mask, np.log(safe_radiance_values)

    def _prepare_cos_sza(self, radiance_data: xr.DataArray) -> np.ndarray:
        values = np.asarray(radiance_data.to_numpy(), dtype=float)
        expected_samples = 1 if values.ndim == 1 else int(np.prod(values.shape[:-1]))

        if self._cos_sza_input is None:
            return np.ones(expected_samples, dtype=float)

        if isinstance(self._cos_sza_input, xr.DataArray):
            cos_values = np.asarray(self._cos_sza_input.to_numpy(), dtype=float)
        else:
            cos_values = np.asarray(self._cos_sza_input, dtype=float)

        cos_values = cos_values.reshape(-1)
        finite = np.isfinite(cos_values)
        if not np.any(finite):
            return np.ones(expected_samples, dtype=float)

        cos_values = cos_values[finite]
        if cos_values.size == 1:
            return np.full(expected_samples, float(cos_values[0]), dtype=float)
        if cos_values.size != expected_samples:
            msg = (
                "cos_sza must be scalar or match the number of non-wavelength "
                f"samples ({expected_samples}); got {cos_values.size}"
            )
            raise ValueError(msg)

        return cos_values

    def _convolve_modelled_radiance(
        self,
        modelled_radiance: xr.DataArray | xr.Dataset,
    ) -> np.ndarray | None:
        modelled_data = _measurement_radiance(modelled_radiance)
        modelled_values = np.asarray(modelled_data.to_numpy(), dtype=float)
        if "wavelength" in modelled_data.coords:
            modelled_wavelength = modelled_data.wavelength.to_numpy()
        else:
            modelled_wavelength = self._calc_wavelength

        if modelled_values.ndim == 1:
            modelled_values = modelled_values[np.newaxis, :]

        wavelength_axes = [
            idx
            for idx, size in enumerate(modelled_values.shape)
            if size == len(modelled_wavelength)
        ]
        if wavelength_axes and wavelength_axes[-1] != modelled_values.ndim - 1:
            modelled_values = np.moveaxis(modelled_values, wavelength_axes[-1], -1)

        if modelled_values.ndim == 3 and modelled_values.shape[-2] <= 4:
            modelled_values = modelled_values[:, 0, :]
        elif modelled_values.ndim > 2:
            modelled_values = modelled_values.reshape(-1, modelled_values.shape[-1])

        if modelled_values.shape[-1] != len(modelled_wavelength):
            if modelled_values.shape[0] == len(modelled_wavelength):
                modelled_values = modelled_values.T
            else:
                return None

        fit_wavelength = self._reference_wavelengths
        band_center = float(np.mean(fit_wavelength))
        transformed_wavel = (
            band_center
            + (1.0 + self._stretch) * (fit_wavelength - band_center)
            + self._shift
        )
        normalized_coord = (transformed_wavel - band_center) / max(
            np.ptp(fit_wavelength) / 2.0,
            1e-6,
        )
        fitted_fwhm = np.maximum(
            self._fwhm_zero + self._fwhm_slope * normalized_coord,
            1e-6,
        )

        return _convolve_template(
            modelled_wavelength,
            modelled_values,
            transformed_wavel,
            fitted_fwhm,
        )

    def _calculate_tilt_spectrum(
        self,
        modelled_conv: np.ndarray,
        wavelengths: np.ndarray,
        cos_sza: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            modelled_conv.ndim != 2
            or modelled_conv.shape[0] == 0
            or modelled_conv.shape[1] < 2
            or wavelengths.size < 2
        ):
            empty = np.full_like(modelled_conv, np.nan, dtype=float)
            return empty, empty.copy()

        safe_modelled = np.where(modelled_conv > 0, modelled_conv, np.nan)
        log_modelled = np.log(safe_modelled)

        ref_idx = -2 if modelled_conv.shape[0] >= 2 else -1
        log_modelled_ref = log_modelled[ref_idx]

        if cos_sza is None:
            cos_sza_arr = np.ones(modelled_conv.shape[0], dtype=float)
        else:
            cos_sza_arr = np.asarray(cos_sza, dtype=float).reshape(-1)
            if cos_sza_arr.size == 1:
                cos_sza_arr = np.full(
                    modelled_conv.shape[0], cos_sza_arr[0], dtype=float
                )
            elif cos_sza_arr.size != modelled_conv.shape[0]:
                cos_sza_arr = np.full(
                    modelled_conv.shape[0], np.nanmedian(cos_sza_arr), dtype=float
                )

        safe_cos = np.where(
            np.isfinite(cos_sza_arr) & (np.abs(cos_sza_arr) > 1e-6), cos_sza_arr, np.nan
        )
        ref_cos = safe_cos[ref_idx]
        if not np.isfinite(ref_cos):
            ref_cos = 1.0
        airmass_scale = ref_cos / safe_cos
        airmass_scale = np.where(np.isfinite(airmass_scale), airmass_scale, 1.0)

        tilt_raw = (log_modelled - log_modelled_ref) * airmass_scale[:, np.newaxis]

        band_center = float(np.mean(wavelengths))
        normalized_wavelength = (wavelengths - band_center) / max(
            np.ptp(wavelengths) / 2.0, 1e-6
        )

        tilt_polynomial = np.full_like(tilt_raw, np.nan)
        tilt_spectrum = np.full_like(tilt_raw, np.nan)

        for los_idx in range(tilt_raw.shape[0]):
            valid_tilt = np.isfinite(tilt_raw[los_idx])
            if np.count_nonzero(valid_tilt) < 2:
                continue

            degree = min(self._poly_order, np.count_nonzero(valid_tilt) - 1)
            poly_coeff = np.polyfit(
                normalized_wavelength[valid_tilt],
                tilt_raw[los_idx, valid_tilt],
                deg=degree,
            )
            poly_fit = np.polyval(poly_coeff, normalized_wavelength)
            tilt_polynomial[los_idx] = poly_fit
            tilt_spectrum[los_idx] = tilt_raw[los_idx] - poly_fit

        return tilt_polynomial, tilt_spectrum

    def _tilt_pca_from_spectrum(
        self,
        tilt_spectrum: np.ndarray,
        n_components: int,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if (
            n_components <= 0
            or tilt_spectrum.ndim != 2
            or tilt_spectrum.shape[0] == 0
            or tilt_spectrum.shape[1] < 2
        ):
            return None, np.array([], dtype=float)

        finite_counts = np.count_nonzero(np.isfinite(tilt_spectrum), axis=0)
        finite_columns = finite_counts >= max(2, tilt_spectrum.shape[0] // 3)
        if np.count_nonzero(finite_columns) < 2:
            return None, np.array([], dtype=float)

        pca_input = tilt_spectrum[:, finite_columns]
        if not np.all(np.isfinite(pca_input)):
            column_means = np.nanmean(pca_input, axis=0)
            column_means = np.nan_to_num(column_means, nan=0.0)
            nan_locs = ~np.isfinite(pca_input)
            pca_input = pca_input.copy()
            pca_input[nan_locs] = column_means[np.where(nan_locs)[1]]

        actual_components = min(
            n_components,
            pca_input.shape[0],
            pca_input.shape[1],
        )
        if actual_components <= 0:
            return None, np.array([], dtype=float)

        tilt_pca = PCA(n_components=actual_components, svd_solver="full")
        tilt_pca.fit(pca_input)

        full_components = np.zeros(
            (actual_components, tilt_spectrum.shape[1]), dtype=float
        )
        full_components[:, finite_columns] = tilt_pca.components_
        return full_components, tilt_pca.explained_variance_ratio_


def doas_fit(
    radiances: xr.DataArray | xr.Dataset,
    anc=None,
    **kwargs,
) -> xr.Dataset:
    fitter = kwargs.pop("fitter", None)
    if fitter is None:
        if anc is None:
            msg = "doas_fit requires either anc or fitter"
            raise ValueError(msg)
        fitter = DOASFitter(radiances, anc, **kwargs)

    return fitter.fit(radiances)
