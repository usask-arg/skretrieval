from __future__ import annotations

import abc
from collections.abc import Iterable

import numpy as np
import xarray as xr
from scipy import sparse

from skretrieval.retrieval.erroranalysis import information_sqrt


class StateVectorElement(abc.ABC):
    """
    A state vector element is a component of the full state vector used in the retrieval. Each state vector element
    has a state, and a prior state/covariance associated with it.  The state vector element must also be able
    to update itself, calculate the jacobian matrix for itself.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    @abc.abstractmethod
    def state(self) -> np.array:
        pass

    def inverse_apriori_covariance(self) -> np.ndarray:
        n = len(self.state())
        return np.zeros((n, n))

    def prior_precision_factor(self):
        """Return ``R`` such that ``R.T @ R`` is the prior precision."""
        return information_sqrt(
            self.inverse_apriori_covariance(), "A priori inverse covariance"
        )

    def apriori_state(self) -> np.array:
        return np.zeros_like(self.state())

    def lower_bound(self) -> np.array:
        n = len(self.state())
        return np.ones(n) * (-np.inf)

    def upper_bound(self) -> np.array:
        n = len(self.state())
        return np.ones(n) * (np.inf)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, e: bool):
        self._enabled = e

    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        pass

    @abc.abstractmethod
    def update_state(self, x: np.array):
        pass

    def modify_input_radiance(self, radiance: xr.Dataset):
        return radiance

    def supports_linearization_products(self) -> bool:
        return False

    def averaging_kernel_row_sum_groups(self) -> np.ndarray:
        """Label physical quantities within this state-vector element."""
        return np.zeros(len(self.state()), dtype=int)

    def averaging_kernel_resolution_coordinates(self) -> dict[str, np.ndarray]:
        """Return physical coordinates used for averaging-kernel moments."""
        return {}

    def linearization_parameter_names(
        self, tangent_template: xr.Dataset
    ) -> tuple[str, ...]:
        msg = (
            f"{type(self).__name__} does not support matrix-free linearization products"
        )
        raise NotImplementedError(msg)

    def add_to_linearization_tangent(
        self,
        tangent: dict[str, xr.DataArray],
        x: np.ndarray,
        tangent_template: xr.Dataset,
    ) -> None:
        msg = (
            f"{type(self).__name__} does not support matrix-free linearization products"
        )
        raise NotImplementedError(msg)

    def linearization_gradient(
        self,
        gradient: xr.Dataset,
        tangent_template: xr.Dataset,
    ) -> np.ndarray:
        msg = (
            f"{type(self).__name__} does not support matrix-free linearization products"
        )
        raise NotImplementedError(msg)

    def describe(self, **kwargs) -> xr.Dataset | None:
        return None


class StateVector:
    def __init__(self, elements: Iterable[StateVectorElement]):
        """
        A full state vector made up of a collection of state vector elements.

        Parameters
        ----------
        elements: Iterable[StateVectorElement]
            A collection of state vector elements
        """
        self._elements = elements

    @property
    def state_elements(self):
        return self._elements

    def update_sasktran_radiance(self, radiance: xr.Dataset, drop_old_wf: bool = False):
        """
        Modifies radiances output from sasktran based on the state vector elements if applicable, e.g., if a state
        vector element is a wavelength shift this will apply it.

        Propagates weighting functions from the sasktran radiance raw output to weighting functions for each
        state vector element.

        If drop_old_wf is set to true then the old weighting functions are removed from the radiance.

        Parameters
        ----------
        radiance: xr.Dataset
            Output from sk.Engine.calculate_radiance(output_format='xarray')
        drop_old_wf: bool, Optional
            If true then the old weighting functions are removed after being propagated to the state vector. Default
            False

        Returns
        -------
        radiance: xr.Dataset
            Modified radiance with a new key 'wf' that is the jacobian with respect to the full state vector.
        """
        all_jacobian = []
        for state_element in self._elements:
            all_jacobian.append(state_element.propagate_wf(radiance))
            radiance = state_element.modify_input_radiance(radiance)

        new_wf = xr.concat(all_jacobian, dim="x")
        radiance["wf"] = new_wf

        if drop_old_wf:
            wf_names = [key for key in radiance if key.startswith("wf_")]
            radiance = radiance.drop(wf_names)
        return radiance

    def check_linearization_product_support(self):
        """Reject enabled state elements that cannot map JVP/VJP products."""
        unsupported = [
            state_element.name()
            for state_element in self._elements
            if state_element.enabled
            and not state_element.supports_linearization_products()
        ]
        if unsupported:
            names = ", ".join(unsupported)
            msg = (
                "Matrix-free retrieval only supports product-aware state "
                f"elements. Unsupported enabled element(s): {names}"
            )
            raise NotImplementedError(msg)

    def prior_precision_factor(self):
        """Block together prior-residual factors for enabled state elements."""
        factors = [
            state_element.prior_precision_factor()
            for state_element in self._elements
            if state_element.enabled
        ]
        if not factors:
            return sparse.csr_matrix((0, 0))
        if any(sparse.issparse(factor) for factor in factors):
            return sparse.block_diag(factors, format="csr")

        rows = sum(factor.shape[0] for factor in factors)
        columns = sum(factor.shape[1] for factor in factors)
        result = np.zeros((rows, columns))
        row_start = 0
        column_start = 0
        for factor in factors:
            row_end = row_start + factor.shape[0]
            column_end = column_start + factor.shape[1]
            result[row_start:row_end, column_start:column_end] = factor
            row_start = row_end
            column_start = column_end
        return result

    def averaging_kernel_row_sum_groups(self) -> np.ndarray:
        """Build globally unique row-sum groups from enabled state elements."""
        groups = []
        next_group = 0
        for state_element in self._elements:
            if not state_element.enabled:
                continue
            local_groups = np.asarray(
                state_element.averaging_kernel_row_sum_groups()
            ).reshape(-1)
            if local_groups.shape != (len(state_element.state()),):
                msg = (
                    f"Averaging-kernel row-sum groups for {state_element.name()} "
                    "must match its state size"
                )
                raise ValueError(msg)
            _, normalized_groups = np.unique(local_groups, return_inverse=True)
            groups.append(normalized_groups + next_group)
            next_group += int(np.max(normalized_groups)) + 1
        if not groups:
            return np.array([], dtype=int)
        return np.concatenate(groups)

    def averaging_kernel_resolution_coordinates(self) -> dict[str, np.ndarray]:
        """Combine resolution coordinates from enabled state elements."""
        enabled_elements = [element for element in self._elements if element.enabled]
        local_coordinates = [
            element.averaging_kernel_resolution_coordinates()
            for element in enabled_elements
        ]
        coordinate_names = {
            name for coordinates in local_coordinates for name in coordinates
        }
        result = {}
        for name in coordinate_names:
            parts = []
            for element, coordinates in zip(
                enabled_elements,
                local_coordinates,
                strict=True,
            ):
                if name not in coordinates:
                    parts.append(np.full(len(element.state()), np.nan))
                    continue
                values = np.asarray(coordinates[name], dtype=float).reshape(-1)
                if values.shape != (len(element.state()),):
                    msg = (
                        f"Averaging-kernel resolution coordinate {name} for "
                        f"{element.name()} must match its state size"
                    )
                    raise ValueError(msg)
                parts.append(values)
            result[name] = np.concatenate(parts)
        return result

    def linearization_parameter_names(
        self, tangent_template: xr.Dataset
    ) -> tuple[str, ...]:
        """Return the active SASKTRAN2 derivative parameter names."""
        names = []
        for state_element in self._elements:
            if state_element.enabled:
                names.extend(
                    state_element.linearization_parameter_names(tangent_template)
                )
        return tuple(dict.fromkeys(names))

    def linearization_tangent(
        self, x: np.ndarray, tangent_template: xr.Dataset
    ) -> xr.Dataset:
        """Map a retrieval-state direction into SASKTRAN2 parameter space."""
        tangent: dict[str, xr.DataArray] = {}
        start = 0
        for state_element in self._elements:
            if not state_element.enabled:
                continue

            end = start + len(state_element.state())
            state_element.add_to_linearization_tangent(
                tangent, x[start:end], tangent_template
            )
            start = end

        return xr.Dataset(tangent)

    def linearization_gradient(
        self, gradient: xr.Dataset, tangent_template: xr.Dataset
    ) -> np.ndarray:
        """Map a SASKTRAN2 VJP result back into retrieval-state space."""
        parts = []
        for state_element in self._elements:
            if state_element.enabled:
                parts.append(
                    state_element.linearization_gradient(gradient, tangent_template)
                )
        if not parts:
            return np.array([])
        return np.concatenate(parts)

    def describe(self, rodgers_output: dict, **kwargs) -> xr.Dataset:
        all_ds = []

        covar = rodgers_output.get("error_covariance_from_noise")
        averaging_kernel = rodgers_output.get("averaging_kernel")
        posterior_variance = rodgers_output.get("solution_covariance_diagonal")
        posterior_variance_standard_error = rodgers_output.get(
            "solution_covariance_diagonal_standard_error"
        )
        measurement_information = rodgers_output.get("measurement_information_diagonal")
        measurement_information_standard_error = rodgers_output.get(
            "measurement_information_diagonal_standard_error"
        )
        approximate_averaging_kernel_row_sum = rodgers_output.get(
            "approximate_averaging_kernel_row_sum"
        )
        averaging_kernel_row_sum = rodgers_output.get("averaging_kernel_row_sum")
        resolution_names = (
            "approximate_averaging_kernel_vertical_resolution_m",
            "approximate_averaging_kernel_horizontal_resolution_m",
            "averaging_kernel_vertical_resolution_m",
            "averaging_kernel_horizontal_resolution_m",
        )
        resolution_values = {
            name: rodgers_output[name]
            for name in resolution_names
            if name in rodgers_output
        }

        start = 0
        for state_element in self._elements:
            end = start + len(state_element.state())
            state_slice = slice(start, end)
            describe_kwargs = dict(kwargs)
            if covar is not None:
                describe_kwargs["covariance"] = covar[state_slice, state_slice]
            if averaging_kernel is not None:
                describe_kwargs["averaging_kernel"] = averaging_kernel[
                    state_slice, state_slice
                ]
            if posterior_variance is not None:
                describe_kwargs["posterior_variance"] = posterior_variance[state_slice]
            if posterior_variance_standard_error is not None:
                describe_kwargs["posterior_variance_standard_error"] = (
                    posterior_variance_standard_error[state_slice]
                )
            if measurement_information is not None:
                describe_kwargs["measurement_information"] = measurement_information[
                    state_slice
                ]
            if measurement_information_standard_error is not None:
                describe_kwargs["measurement_information_standard_error"] = (
                    measurement_information_standard_error[state_slice]
                )
            if approximate_averaging_kernel_row_sum is not None:
                describe_kwargs["approximate_averaging_kernel_row_sum"] = (
                    approximate_averaging_kernel_row_sum[state_slice]
                )
            if averaging_kernel_row_sum is not None:
                describe_kwargs["averaging_kernel_row_sum"] = averaging_kernel_row_sum[
                    state_slice
                ]
            for name, values in resolution_values.items():
                describe_kwargs[name] = values[state_slice]

            ds = state_element.describe(**describe_kwargs)
            if ds is not None:
                all_ds.append(ds)

            start = end

        return xr.merge(all_ds)
