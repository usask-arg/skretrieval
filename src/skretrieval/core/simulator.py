import numpy as np
from typing import List
import sasktran as sk
from skretrieval.core.sensor import Sensor
from skretrieval.core import OpticalGeometry
import skretrieval.core.radianceformat as radianceformat
import xarray as xr
from skretrieval.retrieval import ForwardModel


class MeasurementSimulator(ForwardModel):
    """
    A simple measurement simulator which acts as a forward model.  This is not a generic class, this is moreso
    an example of what a MeasurementSimulator should do.
    """
    def __init__(self, sensor: Sensor, optical_axis: List[OpticalGeometry], atmosphere: sk.Atmosphere, options=None):
        self.sensor = sensor

        self.optical_axis = optical_axis
        self.model_geometry = sk.Geometry()

        self.measurement_geometry = np.hstack([self.sensor.measurement_geometry(o) for o in optical_axis])

        # Determine how to set up the model geometry
        self.model_geometry.lines_of_sight = self.measurement_geometry

        self.engine = sk.EngineHR(geometry=self.model_geometry, atmosphere=atmosphere, options=options)

        self.model_wavel_nm = self.sensor.required_wavelengths(0.1)

        self.engine.wavelengths = self.model_wavel_nm

    def calculate_radiance(self):
        radiance = self.engine.calculate_radiance()
        if hasattr(radiance, 'weighting_function'):
            wf = radiance.weighting_function
            radiance = radiance.radiance
        else:
            wf = None

        model_values = [self.sensor.model_radiance(o, self.model_wavel_nm, self.model_geometry, radiance, wf) for o
                        in self.optical_axis]

        # TODO: Figure out how to make the radiance format class do this concatenation rather than the simulator
        if self.sensor.radiance_format() == radianceformat.RadianceGridded:
            data = radianceformat.RadianceGridded(xr.concat([m.data for m in model_values], dim='los'))
        elif self.sensor.radiance_format() == radianceformat.RadianceRaw:
            data = radianceformat.RadianceRaw(xr.concat([m.data for m in model_values], dim='meas'))
        else:
            raise ValueError('Simulator does not support the format of the sensor')

        return data
