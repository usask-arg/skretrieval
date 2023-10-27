from __future__ import annotations

import abc
from collections import defaultdict
from copy import copy

import numpy as np
import sasktran as sk
from scipy import sparse
from scipy.sparse.linalg import spsolve

from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval import RetrievalTarget
from skretrieval.retrieval.tikhonov import (
    two_dim_horizontal_second_deriv,
    two_dim_vertical_second_deriv,
)
from skretrieval.tomography.grids import OrbitalPlaneGrid


class TwoDimTarget(RetrievalTarget):
    def __init__(
        self,
        species: list[sk.Species],
        grid: OrbitalPlaneGrid,
        state_vector_space: list[str],
        horiz_tikh_factor=0,
        vert_tikh_factor=0,
        use_apriori_for_bounds=True,
        apriori_influence=None,
    ):
        """
        A generic two dimensional retrieval target.  This class implements the two-dimensional nature of the retrieval,
        derived classes implement the species specific retrieval information.

        Derived classes must implement functions to calculate the measurement vector/Jacobian for singular image/scans,
        as well as functions to determined the upper and lower bounds of the retrieval.

        Parameters
        ----------
        species : List[sk.Species]
            A list of species objects that are to be retrieved
        grid : OrbitalPlaneGrid
            The orbital plane grid that is used to define the retrieval grid
        horiz_tikh_factor : float, optional
            Horizontal tikhonov factor used for a second order tikhonov regularization.  Default 0
        vert_tikh_factor : float, optional
            Vertical tikhonov factor used for a second order tikhonov regularization.  Default 0
        state_vector_space: List[str]
            List of either 'log' or 'linear', 'linear' currently may not work.
        use_apriori_for_bounds: bool, optional
            If true, the retrieval is performed on a full grid and edges are handled by pinning the shape using
            the apriori covariance.
        apriori_influence: List[float]
            Constant inverse covariances for each species
        """
        self._species = species
        self._horiz_tikh_factor = horiz_tikh_factor
        self._vert_tikh_factor = vert_tikh_factor
        self._grid = grid
        self._state_vector_space = state_vector_space
        self._apriori_influence = apriori_influence

        self._use_apriori_for_bounds = use_apriori_for_bounds

        self._initial_state_vector = copy(self.state_vector())

        self._max_change_factor = 10
        self._apriori_changed = True

    def _internal_climatology(self):
        """
        List of climatologies for each species being retrieved
        """
        return [species.climatology for species in self._species]

    def _internal_species_id(self):
        """
        List of species identifiers for each species being retrieved.
        """
        return [species.species for species in self._species]

    def state_vector(self):
        """
        Full state vector
        """
        lowerbound = self._lower_wf_bound()
        upperbound = self._upper_wf_bound()
        grid_alts = self._grid.altitudes
        alt_idx = (grid_alts >= lowerbound) & (grid_alts <= upperbound)

        state_values = []
        for clim, species in zip(
            self._internal_climatology(), self._internal_species_id()
        ):
            state_values.append(copy(clim[species])[:, alt_idx])  # numangle, numalt

        for idspecies in range(len(self._internal_climatology())):
            if self._state_vector_space[idspecies] == "log":
                state_values[idspecies] = np.log(state_values[idspecies])

        return np.concatenate([state_value.flatten() for state_value in state_values])

    def _species_state_vector(self, idspecies):
        """
        State vector for an individual species
        """
        x = self.state_vector()
        len_species = int(len(x) / len(self._internal_climatology()))

        return x[idspecies * len_species : (idspecies + 1) * len_species]

    def state_vector_bounded_mask(self, mask=True):
        """
        A mask the same size as state_vector which is true if the state vector is bounded by the apriori
        """
        numangle = self._grid.numhoriz
        numalt = len(self.state_altitudes())

        x_mask = np.ones((numangle, numalt)) * np.nan

        for grid_index in range(numangle):
            upper_bound_altitude = self._upper_retrieval_bound(grid_index)
            lower_bound_altitude = self._lower_retrieval_bound(grid_index)

            # Bound above
            bounded_altitudes = self.state_altitudes() > upper_bound_altitude
            bounded_to = np.nonzero(bounded_altitudes)[0][0] - 1

            x_mask[grid_index, bounded_altitudes] = grid_index * numalt + bounded_to

            # Bound below
            bounded_altitudes = self.state_altitudes() < lower_bound_altitude

            bounded_to = np.nonzero(bounded_altitudes)[0][-1] + 1

            x_mask[grid_index, bounded_altitudes] = grid_index * numalt + bounded_to

            if self._grid.is_extended(grid_index):
                if grid_index <= self._grid.numextended:
                    x_mask[grid_index, :] = numalt * (
                        self._grid.numextended
                    ) + np.arange(0, len(self.state_altitudes()))
                else:
                    x_mask[grid_index, :] = numalt * (
                        numangle - self._grid.numextended
                    ) + np.arange(0, len(self.state_altitudes()))

        if mask:
            return ~np.isnan(x_mask.flatten())
        return x_mask.flatten()

    def zero_apriori_from_mask(self, gamma):
        """
        Takes a tikhonov regularization matrix and zeros out elements that are affected by a bounded state vector
        profile
        """
        bounded_mask = self.state_vector_bounded_mask()

        # Any row of gamma that has contribution from a bounded value should be zero'd out
        row_contrib = np.abs(gamma) @ (bounded_mask).astype(int)
        new_gamma = gamma.asformat("lil")
        new_gamma[row_contrib > 0, :] = 0

        return new_gamma.asformat("csr")

    def scale_apriori(self, gamma, idspecies):
        """
        Scales a tikhonov regularization matrix by the state vector
        """
        if self._state_vector_space[idspecies] == "log":
            return gamma
        x = self._species_state_vector(idspecies)
        x = sparse.diags(1 / x)
        return gamma * x

    def measurement_vector(self, l1_data: RadianceBase):
        output = defaultdict(list)

        for image_idx in range(l1_data.num_images):
            image_l1 = l1_data.image_radiance(image_idx, dense_wf=True)

            image_meas_vec = self._image_measurement_vector(image_l1, image_idx)

            for key, item in image_meas_vec.items():
                if len(item.shape) == 1:
                    output[key].append(item)
                else:
                    output[key].append(sparse.csc_matrix(item))

        y_lengths = [len(y) for y in output["y"]]
        output["y_image_lengths"] = y_lengths

        for key, item in output.items():
            if sparse.issparse(item[0]):
                output[key] = sparse.vstack(item)
            else:
                if hasattr(item[0], "shape") and len(item[0].shape) == 1:
                    output[key] = np.concatenate(item)
                else:
                    output[key] = np.vstack(item)

        if "jacobian" in output:
            diags = []
            for idspecies in range(len(self._internal_climatology())):
                x = np.exp(self._species_state_vector(idspecies))
                if self._state_vector_space[idspecies] == "log":
                    diags.append(x)
                else:
                    diags.append(np.ones_like(x))
            diags = np.concatenate(diags)
            x = sparse.diags(diags)
            output["jacobian"] = (output["jacobian"] * x).asformat("csc")

        return output

    def update_state(self, x: np.ndarray):
        current_state = self.state_vector()
        num_species = len(self._internal_climatology())

        for idspecies, (clim, species, cur_state, new_x) in enumerate(
            zip(
                self._internal_climatology(),
                self._internal_species_id(),
                np.split(current_state, num_species),
                np.split(x, num_species),
            )
        ):
            current_state_resized = cur_state.reshape((self._grid.numhoriz, -1))
            new_state = new_x.reshape((self._grid.numhoriz, -1))

            if self._state_vector_space[idspecies] == "log":
                current_state_resized = np.exp(current_state_resized)
                new_state = np.exp(new_state)

            mult_factor = new_state / current_state_resized
            if self._max_change_factor is not None:
                mult_factor[
                    mult_factor > self._max_change_factor
                ] = self._max_change_factor
                mult_factor[mult_factor < 1 / self._max_change_factor] = (
                    1 / self._max_change_factor
                )

            ret_alts = self.state_altitudes()
            all_mult_factors = np.ones_like(clim[species])
            for idx in range(self._grid.numhoriz):
                if not self._use_apriori_for_bounds:
                    good_low_alt = self._lower_retrieval_bound(idx)
                    low_idx = np.nonzero(ret_alts > good_low_alt)[0][0]

                    good_high_alt = self._upper_retrieval_bound(idx)
                    high_idx = np.nonzero(ret_alts > good_high_alt)[0][0]

                    all_mult_factors[idx, :] = np.interp(
                        self._grid.altitudes,
                        ret_alts[low_idx:high_idx],
                        mult_factor[idx, low_idx:high_idx],
                        left=mult_factor[idx, low_idx],
                        right=mult_factor[idx, high_idx],
                    )
                else:
                    all_mult_factors[idx, :] = np.interp(
                        self._grid.altitudes,
                        ret_alts,
                        mult_factor[idx, :],
                        left=mult_factor[idx, 0],
                        right=mult_factor[idx, -1],
                    )
            clim[species] *= all_mult_factors

    def _compute_apriori_parameters(self):
        if self._apriori_changed:
            num_species = len(self._internal_climatology())
            n = int(len(self.state_vector()) / num_species)

            all_S_a = []
            all_x_a = []
            for idspecies in range(num_species):
                inv_S_a = sparse.csc_matrix((n, n))

                numangle = self._grid.numhoriz
                numalt = n // numangle

                if self._horiz_tikh_factor is not None:
                    gamma = two_dim_horizontal_second_deriv(
                        numangle, numalt, self._horiz_tikh_factor, sparse=True
                    )
                    gamma = self.zero_apriori_from_mask(gamma)
                    gamma = self.scale_apriori(gamma, idspecies)

                    inv_S_a += gamma.T @ gamma
                if self._vert_tikh_factor is not None:
                    gamma = two_dim_vertical_second_deriv(
                        numangle, numalt, self._vert_tikh_factor, sparse=True
                    )
                    gamma = self.zero_apriori_from_mask(gamma)
                    gamma = self.scale_apriori(gamma, idspecies)

                    inv_S_a += gamma.T @ gamma

                if self._apriori_influence[idspecies] is not None:
                    gamma = sparse.diags(
                        np.ones(n) * self._apriori_influence[idspecies]
                    )
                    gamma = self.zero_apriori_from_mask(gamma)
                    gamma = self.scale_apriori(gamma, idspecies)

                    inv_S_a += gamma

                if self._use_apriori_for_bounds:
                    inv_S_a_bounding = sparse.csc_matrix((n, n))
                    high_factor = 1e3
                    # Create a very large relative apriori scaling
                    # First order tikhonov matrix is (n+1, n), create this by making two diagonals that are (numangle, numalt)
                    # and then flatten them
                    diagonal = np.ones((numangle, numalt))
                    off_diagonal = np.ones((numangle, numalt)) * -1
                    for grid_index in range(numangle):
                        upper_bound_altitude = self._upper_retrieval_bound(grid_index)
                        lower_bound_altitude = self._lower_retrieval_bound(grid_index)

                        # Bound above
                        bounded_altitudes = (
                            self.state_altitudes() > upper_bound_altitude
                        )
                        first_idx = np.nonzero(bounded_altitudes)[0][0]
                        bounded_altitudes[first_idx - 1] = True
                        diagonal[grid_index, ~bounded_altitudes] = 0
                        off_diagonal[grid_index, ~bounded_altitudes] = 0

                        # Bound below
                        bounded_altitudes = (
                            self.state_altitudes() < lower_bound_altitude
                        )
                        # last_idx = np.nonzero(~bounded_altitudes)[0][0]
                        # bounded_altitudes[last_idx] = True

                        diagonal[grid_index, bounded_altitudes] = 1
                        off_diagonal[grid_index, bounded_altitudes] = -1

                        # Unlink the last point to the next angular point
                        diagonal[grid_index, -1] = 0
                        off_diagonal[grid_index, -1] = 0
                    gamma = (
                        sparse.diags(
                            [diagonal.flatten(), off_diagonal.flatten()], offsets=[0, 1]
                        )
                        * high_factor
                    )
                    gamma = self.scale_apriori(gamma, idspecies)
                    inv_S_a_bounding += gamma.T @ gamma
                    self._vert_gamma = copy(gamma)

                    # Same thing but for horizontal regularization to enforce the edges
                    num_extended = self._grid.numextended
                    if num_extended > 0:
                        diagonal_horiz = np.zeros((numangle, numalt))
                        off_diagonal_horiz = np.zeros((numangle + 1, numalt))

                        # Forward difference from the left edge
                        diagonal_horiz[: num_extended + 1, :] = high_factor
                        off_diagonal_horiz[: num_extended + 1, :] = -high_factor

                        # Forward difference from the right edge
                        diagonal_horiz[-(num_extended + 1) : -1, :] = high_factor
                        off_diagonal_horiz[-(num_extended + 2) :, :] = -high_factor

                        gamma_horiz = sparse.diags(
                            [diagonal_horiz.flatten(), off_diagonal_horiz.flatten()],
                            [0, numalt],
                            shape=((numangle + 1) * numalt, numalt * numangle),
                        )

                        gamma_horiz = self.scale_apriori(gamma_horiz, idspecies)

                        self._horiz_gamma = copy(gamma_horiz)
                        inv_S_a_bounding += gamma_horiz.T @ gamma_horiz

                    full_inv_S_a = inv_S_a + inv_S_a_bounding
                    x_a_bounding = self.state_vector()[
                        (idspecies * n) : (idspecies + 1) * n
                    ]
                    x_a = self._initial_state_vector[
                        (idspecies * n) : (idspecies + 1) * n
                    ]

                    full_x_a = spsolve(
                        full_inv_S_a, inv_S_a_bounding @ x_a_bounding + inv_S_a @ x_a
                    )
                    all_x_a.append(full_x_a)

                    all_S_a.append(full_inv_S_a)

            self._x_a = np.concatenate(all_x_a)
            self._inv_S_a = sparse.block_diag(all_S_a, format="csc")
        self._apriori_changed = False

    def apriori_state(self) -> np.array:
        self._compute_apriori_parameters()
        return self._x_a

    def inverse_apriori_covariance(self):
        self._compute_apriori_parameters()
        return self._inv_S_a

    @abc.abstractmethod
    def _image_measurement_vector(self, l1_data: RadianceBase, image_index):
        """
        Calculate the measurement vector for a single image/scan

        Parameters
        ----------
        l1_data : RadianceBase
            l1_data for the image at image_index
        image_index: int
            Index of the image

        Returns
        -------
        dict
            Dictionary with keys 'y', and optionally 'y_error' and 'jacobian'.  Additional diagnostic keys may
            be included but are not used.
        """

    @abc.abstractmethod
    def _lower_wf_bound(self):
        """
        Lower weighting function altitude in m
        """

    @abc.abstractmethod
    def _upper_wf_bound(self):
        """
        Upper weighting function altitude in m
        """

    @abc.abstractmethod
    def _upper_retrieval_bound(self, grid_index):
        """
        Upper altitude in m at horizontal grid index grid_index

        """

    @abc.abstractmethod
    def _lower_retrieval_bound(self, grid_index):
        """
        Lower altitude in m at horizontal grid index grid_index
        """

    def wfwidths_and_alts(self):
        """
        weighting function widths and altitudes that are intended to be included as options for the HR model.
        """
        wf_alts = self.state_altitudes()
        wf_widths = np.gradient(wf_alts)

        return wf_widths, wf_alts

    def state_altitudes(self):
        """
        Altitudes that the weighting functions are calculated at, and thus deltas for the state vector are calculated
        at

        """
        lowerbound = self._lower_wf_bound()
        upperbound = self._upper_wf_bound()

        grid_alts = self._grid.altitudes

        alt_idx = (grid_alts >= lowerbound) & (grid_alts <= upperbound)

        return grid_alts[alt_idx]
