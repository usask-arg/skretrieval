from __future__ import annotations

from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.retrieval.measvec import (
    MeasurementVector,
    concat,
    post_process,
    pre_process,
)
from skretrieval.retrieval.statevector import StateVector
from skretrieval.retrieval.target import GenericTarget


class MeasVecTarget(GenericTarget):
    def __init__(
        self,
        state_vector: StateVector,
        measurement_vectors: dict[MeasurementVector],
        context: dict,
        rescale_state_space: bool = False,
    ):
        """
        A target where the measurement vector is calculated through MeasurementVector objects

        Parameters
        ----------
        state_vector : StateVector
        measurement_vectors : dict[MeasurementVector]
        rescale_state_space : bool, optional
            If true, the state vectors are internally scaled to be within their min and max values, by default False
        """
        super().__init__(state_vector, rescale_state_elements=rescale_state_space)
        self._measurement_vectors = measurement_vectors
        self._context = context

    def _internal_measurement_vector(self, l1_data: dict[RadianceGridded]):
        l1 = pre_process(l1_data, len(self.state_vector()))

        res = []
        for _, v in self._measurement_vectors.items():
            appl = v.apply(l1, self._context)
            if appl is not None:
                res.append(appl)

        return post_process(concat(res))
