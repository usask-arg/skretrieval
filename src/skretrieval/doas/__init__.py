from __future__ import annotations

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
        absorber_temperatures: dict[str, float | list[float]] | None = None,
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
        default_absorber_temperatures: dict[str, float | list[float]] = {
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
        raw_filter = radiance_filter if radiance_filter is not None else filter
        self._radiance_filter_weights = _normalized_filter_weights(raw_filter)
        self._poly_order = poly_order
        self._initial_params = np.array(
            [0.0, 0.0, 0.2, 0.0] if initial_params is None else initial_params,
            dtype=float,
        )
        self._bounds = bounds or (
            np.array([-0.5, -0.05, 1e-4, -1.0], dtype=float),
            np.array([0.5, 0.05, 3.0, 1.0], dtype=float),
        )
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
            calculate_derivatives=False,
        )
        anc.add_to_atmosphere(atmo)
        atmo["o3"].vmr[:] *= 0.0

        viewing_geo = sk.ViewingGeometry()

        for talt in reference_radiance.tangent_altitude.to_numpy():
            viewing_geo.add_ray(sk.TangentAltitudeSolar(talt, 0.0, 200000, 0.45))

        engine = sk.Engine(sk.Config(), model_geo, viewing_geo)

        modelled_radiance = engine.calculate_radiance(atmo)

        self._xs = {}
        self._xs_variance_ratio = {}
        for name, optical_quantity in self._optical.items():
            configured_temperatures = self._absorber_temperatures.get(name, 220.0)
            if isinstance(configured_temperatures, int | float):
                configured_temperatures = [float(configured_temperatures)]

            components, variance_ratio = _temperature_cross_section_components(
                optical_quantity,
                atmo,
                configured_temperatures,
            )
            self._xs[name] = components
            self._xs_variance_ratio[name] = variance_ratio

        self._irrad = sk.solar.SolarModel().irradiance(self._calc_wavelength)

        filtered_radiances = _filter_measurement_input(
            radiances,
            self._radiance_filter_weights,
        )
        calibration_radiance, _, calibration_mask, calibration_log = self._fit_inputs(
            filtered_radiances
        )

        def residual(params: np.ndarray) -> np.ndarray:
            design_matrix, _, _, _ = _design_matrix(
                calibration_radiance.wavelength.to_numpy(),
                self._calc_wavelength,
                self._xs,
                self._irrad,
                self._poly_order,
                shift=params[0],
                stretch=params[1],
                fwhm_zero=params[2],
                fwhm_slope=params[3],
                residual_basis=self._residual_basis,
                tilt_basis=None,
            )
            _, fitted = _solve_linear_coefficients(
                design_matrix,
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
            initial_design, _, _, _ = _design_matrix(
                calibration_radiance.wavelength.to_numpy(),
                self._calc_wavelength,
                self._xs,
                self._irrad,
                self._poly_order,
                shift=float(self._nonlinear_fit.x[0]),
                stretch=float(self._nonlinear_fit.x[1]),
                fwhm_zero=float(self._nonlinear_fit.x[2]),
                fwhm_slope=float(self._nonlinear_fit.x[3]),
                residual_basis=None,
                tilt_basis=None,
            )
            _, initial_fitted = _solve_linear_coefficients(
                initial_design,
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

        self._convolved_modelled_radiance = self._convolve_modelled_radiance(
            modelled_radiance
        )
        if self._convolved_modelled_radiance is not None:
            _, tilt_spectrum = self._calculate_tilt_spectrum(
                self._convolved_modelled_radiance,
                self._reference_wavelengths,
            )
            self._tilt_pca_basis, self._tilt_pca_variance_ratio = (
                self._tilt_pca_from_spectrum(
                    tilt_spectrum,
                    self._tilt_pca_components,
                )
            )

        # Cache the design matrix after initialization. Subsequent fit calls
        # reuse it because the fitted nonlinear params and wavelengths are fixed.
        self._build_cached_design_matrix()

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

        if (
            self._cached_design_matrix is None
            or self._cached_convolved_xs is None
            or self._cached_convolved_irrad is None
            or self._cached_fit_wavelength is None
        ):
            self._build_cached_design_matrix()

        best_design_matrix = self._cached_design_matrix
        convolved_xs = self._cached_convolved_xs
        convolved_irrad = self._cached_convolved_irrad
        fit_wavelength = self._cached_fit_wavelength
        coefficients, fitted_log_radiance = _solve_linear_coefficients(
            best_design_matrix,
            log_radiances,
            valid_mask,
        )

        basis_names = []
        for name, components in self._xs.items():
            basis_names.extend(
                f"{name}_pca_{idx}" for idx in range(components.shape[0])
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
            bro_contribution = (
                coefficients[:, bro_indices] @ best_design_matrix[:, bro_indices].T
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

            a_matrix = best_design_matrix[sample_mask]
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

                a_matrix = best_design_matrix[sample_mask]
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

        for name, values in convolved_xs.items():
            component_dim = f"{name}_component"
            dataset = dataset.assign_coords({component_dim: np.arange(values.shape[0])})
            dataset[f"xs_pca_{name}"] = (
                (component_dim, "calc_wavelength"),
                self._xs[name],
            )
            dataset[f"xs_pca_variance_ratio_{name}"] = (
                (component_dim,),
                self._xs_variance_ratio[name],
            )
            dataset[f"convolved_xs_{name}"] = ((component_dim, "wavelength"), values)

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
        tilt_raw = log_modelled - log_modelled_ref

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
