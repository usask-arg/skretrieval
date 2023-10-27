from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import skyfield
import skyfield.timelib
import tudatjson  # This is created in the ARG copy of Tudat Bundle, see tudatBundle\tudatApplications\python_json_dll\python_tudatjson_interface

from .satellitebase import SatelliteBase
from .satellitekepler import SatelliteKepler

jsontemplate = r"""
{
  "initialEpoch": 0,
  "finalEpoch": 86400,
  "globalFrameOrientation": "J2000",
  "spice": {
    "useStandardKernels": true,
    "preloadEphemeris": false
  },
  "bodies": {
    "Sun": {
      "useDefaultSettings": true
    },
    "Earth": {
      "useDefaultSettings": true,
      "gravityField": {
            "type": "sphericalHarmonic",
            "model": "ggm02c"
      }
    },
    "Moon": {
      "useDefaultSettings": true
    },
    "custom_satellite": {
      "initialState": {
        "type": "cartesian",
        "x": 1732339.13760243,
        "y": -6656846.40425629,
        "z": 723621.98106622,
        "vx": -775.84161121,
        "vy": -1026.88295639,
        "vz": -7488.07085716
      },
      "mass": 400,
      "referenceArea": 4,
      "atmosphere": {
            "type": "nrlmsise00"
       },
      "aerodynamics": {
        "forceCoefficients": [ 8.75, 0, 0 ]
      },
      "radiationPressure": {
        "Sun": {
          "radiationPressureCoefficient": 1.2,
          "occultingBodies": [ "Earth" ]
        }
      }
    }
  },
  "propagators": [
    {
      "integratedStateType": "translational",
      "centralBodies": [
        "Earth"
      ],
      "bodiesToPropagate": [
        "custom_satellite"
      ],
      "accelerations": {
        "custom_satellite": {
          "Earth": [
            {
              "type": "sphericalHarmonicGravity",
              "maximumDegree": 20,
              "maximumOrder": 20
            },
            {
              "type": "aerodynamic"
            }
          ],
          "Sun": [
            {
              "type": "pointMassGravity"
            },
            {
              "type": "cannonBallRadiationPressure"
            }
          ],
          "Moon": [
            {
              "type": "pointMassGravity"
            }
          ]
        }
      }
    }
  ],
  "integrator": {
    "type": "rungeKutta4",
    "stepSize": 10
  },
  "export": [
    {
      "file": "temporary_stateHistory.txt",
      "variables": [
        {
          "type": "state"
        }
      ]
    }
  ],
  "options": {
    "fullSettingsFile": "temporary_fullSettings.json"
  }
}
"""


# -----------------------------------------------------------------------------
#           StateCache
# -----------------------------------------------------------------------------
class StateCache:
    """
    Holds a cache of satellite state
    """

    def __init__(self):
        self.cache_starttime: float = None  # start time of first value in cache
        self.cache_endtime: float = None  # end time of last value in cache.
        self.entry_starttime: float = None  # start time of first value in current entry
        self.entry_endtime: float = None  # end time of last value in current entry.
        self.timeoffsets: np.ndarray = (
            None  # The array of time offsets in seconds since J2000.
        )
        self.positions: np.ndarray = None  # The array of platform_ecef_positions in the current simulation window
        self.velocities: np.ndarray = (
            None  # The array of velocities in the current simulation window
        )
        self.ts: skyfield.timelib.Timescale = (
            skyfield.api.load.timescale()
        )  # Load in the skyfield timescales
        self.j2000: skyfield.timelib.Time = self.ts.tt(
            2000, 1, 1, 12, 0, 0
        )  # Keep the time of J2000 cached
        self.kepler: SatelliteKepler = SatelliteKepler()

    # -----------------------------------------------------------------------------
    #           in_cache
    # -----------------------------------------------------------------------------

    def in_cache(self, tt_j2000: float) -> bool:
        return (tt_j2000 >= self.cache_starttime) and (tt_j2000 < self.cache_endtime)

    # -----------------------------------------------------------------------------
    #           is_valid
    # -----------------------------------------------------------------------------

    def is_valid(self) -> bool:
        return self.timeoffsets is not None

    # -----------------------------------------------------------------------------
    #           update_cache
    # -----------------------------------------------------------------------------
    def update_cache(self, tudatresults: np.ndarray):
        self.timeoffsets = tudatresults[:, 0]  # seconds since J2000.
        self.positions = tudatresults[:, 1:4]
        self.velocities = tudatresults[:, 4:7]
        self.cache_starttime = self.timeoffsets[0]
        self.cache_endtime = self.timeoffsets[-1]
        if self.cache_endtime < self.cache_starttime:
            np.flip(self.timeoffsets)
            np.flipud(self.positions)
            np.flipud(self.velocities)
            self.cache_starttime = self.timeoffsets[0]
            self.cache_endtime = self.timeoffsets[-1]
        self.entry_starttime = 1.0
        self.entry_endtime = 0.0

    # -----------------------------------------------------------------------------
    #           utc_to_tt_j2000
    # -----------------------------------------------------------------------------
    def datetime_to_ttj2000(self, mjd: datetime) -> float:
        t = self.ts.observation_policy.utc(
            mjd.year,
            mjd.month,
            mjd.day,
            mjd.hour,
            mjd.minute,
            mjd.second + mjd.microsecond / 1.0e6,
        )
        return (t.tt - self.j2000.tt) * 86400.0

    # -----------------------------------------------------------------------------
    #           ttj2000_to_datetime
    # -----------------------------------------------------------------------------
    def ttj2000_to_datetime(self, tt_j2000) -> datetime:
        t = self.ts.tt_jd(tt_j2000 / 86400.0 + self.j2000.tt)
        return t.utc_datetime()

    # -----------------------------------------------------------------------------
    #           update_entry
    # -----------------------------------------------------------------------------

    def update_entry(self, tt_j2000: float):
        ok = (tt_j2000 >= self.entry_starttime) and (tt_j2000 < self.entry_endtime)
        if not ok:
            index = np.searchsorted(self.timeoffsets, tt_j2000) - 1
            self.entry_starttime = self.timeoffsets[index]
            self.entry_endtime = self.timeoffsets[index + 1]
            r = self.positions[index, :]
            v = self.velocities[index, :]
            mjd0 = self.ttj2000_to_datetime(self.entry_starttime)
            self.kepler.from_state_vector(mjd0, 1, r, v)
            ok = True
        return ok

    # -----------------------------------------------------------------------------
    #           interpolate_current_entry
    # -----------------------------------------------------------------------------

    def interpolate_current_entry(
        self, tt_j2000: float
    ) -> tuple[np.ndarray, np.ndarray]:
        mjd = self.ttj2000_to_datetime(tt_j2000)
        self.kepler.update_eci_position(mjd)
        r = self.kepler.eciposition()
        v = self.kepler.ecivelocity()
        return (r, v)

    # -----------------------------------------------------------------------------
    #           interpolate_position_and_velocity
    # -----------------------------------------------------------------------------

    def interpolate_position_and_velocity(
        self, tt_j2000: float
    ) -> tuple[np.ndarray, np.ndarray]:
        ok = self.update_entry(tt_j2000)
        if ok:
            r, v = self.interpolate_current_entry(tt_j2000)
        else:
            r = None
            v = None
        return (r, v)


# ------------------------------------------------------------------------------
#           class SatelliteTudat
# ------------------------------------------------------------------------------
class SatelliteTudat(SatelliteBase):
    def __init__(self):
        super().__init__()
        self.cache: StateCache = StateCache()
        self.simulation_window: float = 86400.0  # Get the length of the simulation window in seconds. default is 1 day
        self.jsonobject: dict[str, Any] = json.loads(jsontemplate)

    # ---------------------------------------------------------------------------
    #              SatelliteKepler::orbital_period
    # ---------------------------------------------------------------------------

    def period(self) -> timedelta:
        """
        Return the orbital period of this orbit. Uses keplers third law.

        Returns
        -------
        datetime.timedelta
            The orbital period.
        """
        return self.cache.kepler.period()

    # -----------------------------------------------------------------------------
    #           calculate_cache
    # -----------------------------------------------------------------------------

    def calculate_cache(self, start_j2000, end_j2000, r: np.ndarray, v: np.ndarray):
        self.jsonobject["initialEpoch"] = start_j2000
        self.jsonobject["finalEpoch"] = end_j2000
        state = self.jsonobject["bodies"]["custom_satellite"]["initialState"]
        state["type"] = "cartesian"
        state["x"] = r[0]
        state["y"] = r[1]
        state["z"] = r[2]
        state["vx"] = v[0]
        state["vy"] = v[1]
        state["vz"] = v[2]
        jsonstring = json.dumps(self.jsonobject)
        self.jsonobject = tudatjson.execute_json(jsonstring)
        datafile = self.jsonobject["export"][0]["file"]
        settingsfile = self.jsonobject["options"]["fullSettingsFile"]
        results = np.loadtxt(datafile)
        Path.unlink(datafile)
        Path.unlink(settingsfile)
        self.cache.update_cache(results)

    # ---------------------------------------------------------------------------
    #      SatelliteKepler::update_eci_position
    # ---------------------------------------------------------------------------

    def update_eci_position(self, mjd: datetime):
        if (self._m_time is None) or (mjd != self._m_time):
            tt_j2000 = self.cache.datetime_to_ttj2000(mjd)
            ok = self.cache.in_cache(tt_j2000)  # this time is not in the current cache
            while not ok:  # so
                if self.cache.is_valid():  # If we have a valid cache
                    if (
                        tt_j2000 >= self.cache.cache_endtime
                    ):  # then if we are going forwards in time
                        startdt = (
                            self.cache.cache_endtime
                        )  # start at the end of the current cache
                        enddt = startdt + self.simulation_window
                        r = self.cache.positions[-1, :]
                        v = self.cache.velocities[-1, :]
                    else:  # otherwise we are going backwards in time
                        startdt = self.cache.cache_starttime
                        enddt = self.cache.cache_starttime - self.simulation_window
                        r = self.cache.positions[0, :]
                        v = self.cache.velocities[0, :]
                else:
                    msg = "You must initialize the tudat satellite propagator before trying to update the eci position"
                    raise Exception(msg)
                self.calculate_cache(startdt, enddt, r, v)
                ok = self.cache.in_cache(tt_j2000)
            self._m_time = mjd
            r, v = self.cache.interpolate_position_and_velocity(tt_j2000)
            self._m_location = r.copy()
            self._m_velocity = v.copy()

    # -----------------------------------------------------------------------------
    #           from_state_vector
    # -----------------------------------------------------------------------------

    def from_state_vector(
        self,
        mjd: datetime,
        orbitnumber: int,  # noqa: ARG002
        r: np.ndarray,
        v: np.ndarray,
    ):
        startdt = self.cache.datetime_to_ttj2000(mjd)
        enddt = startdt + self.simulation_window
        self.calculate_cache(startdt, enddt, r, v)
        self.update_eci_position(mjd)
