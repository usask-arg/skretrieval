from __future__ import annotations

import numpy as np

from skretrieval.core.radianceformat import RadianceBase, RadianceGridded
from skretrieval.retrieval.statevector import StateVector
from skretrieval.retrieval.target import GenericTarget


class USARMTarget(GenericTarget):
    def __init__(
        self,
        state_vector: StateVector,
        measurement_vectors,  # noqa: ARG002
        rescale_state_space: bool = False,
    ):
        super().__init__(state_vector, rescale_state_elements=rescale_state_space)

    def _internal_measurement_vector(self, l1_data: RadianceBase):
        pass


class OSIRISFPTarget(GenericTarget):
    def __init__(
        self,
        state_vector: StateVector,
        log_space: bool = False,
        include_high_alt_norm=True,
        rescale_state_space: bool = False,
        low_alt=15000,
    ):
        super().__init__(state_vector, rescale_state_space)
        self._log_space = log_space
        self._include_high_alt_norm = include_high_alt_norm
        self._low_alt = low_alt

    def _internal_measurement_vector(self, l1_data: RadianceGridded):
        result = {}

        if "radiance_noise" in l1_data.data:
            result["y"] = l1_data.data["radiance"].to_numpy().T.flatten()
            result["y_error"] = self._masked_osiris_noise(l1_data)

            if self._include_high_alt_norm:
                result["y_error"] *= 1e6

        else:
            result["y"] = l1_data.data["radiance"].to_numpy().flatten()

        if "wf" in l1_data.data:
            np_wf = l1_data.data["wf"].to_numpy()
            result["jacobian"] = np_wf.reshape(-1, np_wf.shape[2])

        if self._log_space:
            if "y_error" in result:
                result["y_error"] /= result["y"] ** 2

            if "jacobian" in result:
                result["jacobian"] /= np.abs(result["y"][:, np.newaxis])

            result["y"] = np.log(np.abs(result["y"]))

        if self._include_high_alt_norm:
            norm = l1_data.data.where(
                (l1_data.data.tangent_altitude > 35000)
                & (l1_data.data.tangent_altitude < 40000)
            ).mean(dim="los")

            useful = l1_data.data.where(
                (l1_data.data.tangent_altitude < 35000), drop=True
            )

            normalized = useful["radiance"] / norm["radiance"]

            norm = norm.where(
                (norm.wavelength > 290) & (norm.wavelength < 800), drop=True
            )
            useful = useful.where(
                (useful.wavelength > 290) & (useful.wavelength < 800), drop=True
            )
            normalized = normalized.where(
                (normalized.wavelength > 290) & (normalized.wavelength < 800), drop=True
            )

            if "y_error" in result:
                if self._log_space:
                    result["y_error"] = np.concatenate(
                        (
                            result["y_error"],
                            np.ones_like(normalized.to_numpy().T.flatten())
                            * (0.05**2),
                        )
                    )
                    result["y"] = np.concatenate(
                        (result["y"], np.log(normalized.to_numpy().T.flatten()))
                    )
                else:
                    error = (
                        (0.1 * useful["radiance"].to_numpy())
                        / norm["radiance"].to_numpy()[np.newaxis, :]
                    ) ** 2
                    result["y_error"] = np.concatenate(
                        (result["y_error"], error.T.flatten())
                    )
                    result["y"] = np.concatenate(
                        (result["y"], normalized.to_numpy().T.flatten())
                    )

            if "jacobian" in result:
                np_wf_useful = useful["wf"].to_numpy()
                np_wf_norm = norm["wf"].to_numpy()

                if self._log_space:
                    np_wf_useful /= useful["radiance"].to_numpy()[:, :, np.newaxis]
                    np_wf_norm /= norm["radiance"].to_numpy()[:, np.newaxis]
                    np_wf_useful -= np_wf_norm[:, np.newaxis, :]

                    result["y"] = np.concatenate(
                        (result["y"], np.log(normalized.to_numpy().flatten()))
                    )
                    result["jacobian"] = np.vstack(
                        (
                            result["jacobian"],
                            np_wf_useful.reshape(-1, np_wf_useful.shape[2]),
                        )
                    )
                else:
                    np_wf_useful /= norm["radiance"].to_numpy()[
                        :, np.newaxis, np.newaxis
                    ]
                    np_wf_norm /= norm["radiance"].to_numpy()[:, np.newaxis]
                    np_wf_useful -= (
                        np_wf_norm[:, np.newaxis, :]
                    ) * normalized.to_numpy()[:, :, np.newaxis]

                    result["y"] = np.concatenate(
                        (result["y"], normalized.to_numpy().flatten())
                    )
                    result["jacobian"] = np.vstack(
                        (
                            result["jacobian"],
                            np_wf_useful.reshape(-1, np_wf_useful.shape[2]),
                        )
                    )

        return result

    def _masked_osiris_noise(self, l1_data: RadianceGridded):
        f = 1e8
        # Mask stray light
        wavels = np.array(
            [250, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800, 850, 1020]
        )

        top_alts = np.array([65, 65, 65, 65, 60, 55, 50, 45, 40, 40, 40, 40]) * 1000
        top_alts = np.array([65, 65, 65, 65, 65, 65, 65, 65, 65, 40, 40, 40, 40]) * 1000

        noise = l1_data.data["radiance_noise"].to_numpy().T
        noise = l1_data.data["radiance"].to_numpy().T * 0.01

        # noise[noise < 1e-3] = 1e-3

        wavel = l1_data.data.wavelength.to_numpy()
        alts = l1_data.data.tangent_altitude.to_numpy()

        for i in range(len(wavel)):
            # if wavel[i] < 350:
            #    noise[i, noise[i, :] < 1e-3] = 1e-3

            top_alt = np.interp(wavel[i], wavels, top_alts)

            noise[i, alts > top_alt] *= f

        # Test mask
        noise[wavel < 290, :] *= f
        # noise[(wavel > 390) & (wavel < 400), :] *= f
        # noise[(wavel > 302) & (wavel < 308), :] *= f
        # noise[(wavel > 625) & (wavel < 640), :] *= f
        # noise[(wavel > 755) & (wavel < 772), :] *= f
        # noise[(wavel > 710) & (wavel < 740), :] *= f
        # noise[(wavel > 680) & (wavel < 700), :] *= f

        # noise[wavel < 500] *= f

        noise[:, alts < self._low_alt] *= f

        return noise.flatten() ** 2
