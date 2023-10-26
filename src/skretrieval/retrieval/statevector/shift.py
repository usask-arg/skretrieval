import numpy as np
import xarray as xr

from skretrieval.retrieval.statevector import StateVectorElement
from copy import copy


class RadianceShift(StateVectorElement):
    def state(self) -> np.array:
        return [copy(self._shift_nm)]

    def name(self) -> str:
        return 'wavelength_shift'

    def propagate_wf(self, radiance) -> np.ndarray:
        x = 5

    def update_state(self, x: np.array):
        self._shift_nm = copy(float(x))

    def __init__(self, shift_nm: float):
        self._shift_nm = shift_nm

    def modify_input_radiance(self, radiance: xr.Dataset):
        pass