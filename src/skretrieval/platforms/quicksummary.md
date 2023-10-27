Sorry for the long message. Happy to hear any feedback that anybody cares to make. Ideas about how to manage the Z axis unit vector are welcome.

## Overview

I thought I would give a quick update about some of the proposed directions within the skretrievals.platforms package. The skretrievals.platforms package has reasonably detailed techniques for rotating and positioning vectors from one reference frame to another but it is lacking methods to help users setup observing scenarios quickly and easily.

I have spoken with Doug, who has had experience using the `plaforms` package and he has found that a flexible method is to specify observer positions and look vectors using several techniques.

Consequently I am developing a class called **MeasurementGeometry** within the `platforms` package that will take the different specifications of time, position and look vector and convert all the different specifications to the internal formats used by the **Platform** class and Dan's `skretrieval` code.

The **MeasurementGeometry** essentially acts as a user interface to the `Platform` class making it easier for the user to specify scans/observation sets. Much of the coding is checking inputs. Several techniques for specifying position and look vector are outlined below but these are just the start. More more specification techniques can be easily added. It will also be possible to add project specific techniques to the **MeasurementGeometry** class at run-time.

## Time specifications
All times will be specified as list or arrays of UTC. A time will be specified for each observation point. Times can be encoded in a variety of formats

## Position Specification Techniques
1. **xyz**,  specifies location as 3 element tuple (x,y,z).  Geocentric, geographic (ITRF) vector in meters
2. **llh**,  specifies location as 3 element tuple( latitude, longitude, height). angles in degrees, height in meters above sea-level.
3. **from_plaform** will query the platform object for its x,y,z geocentric position at a given UTC. This is useful for real instruments and simulators that make use of satellite, aircraft, balloon, ground site classes.

## Boresight Look Vector Specification Techniques
1. **xyz**, specifies boresight look unit vector with 3 element tuple(x,y,z).
2. **tangent_altitude**, specifies boresight unit vector with 2 element tuple( height, azimuth). Look at tangent point at given height and azimuth from observer
3. **location_xyz**, specifies boresight unit vector to look at the location given in 3 element tuple (x,y,z) geocentric in meters. Currently used in nadir configuration
4. **location_llh**, specifies boresight unit vector to look at the location given in 3 element tuple (lat,long,height). Currently used in nadir configuration
5. **from_platform**, will query the platform object for its boresight look vector at a given UTC. Useful for real instruments.

## Example
```
    from skretrieval.platforms import Platform
    from skretrieval.platforms import MeasurentGeometry

    def test_tangentalt_geometry():
        platform = Platform()
        measgeom = MeasurentGeometry()
        utc  = ['2020-09-24T12:15:36.123456', '2020-09-24T12:15:37.456123', '2020-09-24T12:15:38.654321', '2020-09-24T12:15:39.654321']
        pos  = [(52, -107, 600000),           (53, -107, 600001),           (54, -107, 600002),           (55,-107, 600003)]
        look = [(35000, 10),                  (27000, 5),                   (24000, 0),                   (21000, -5)]

        measgeom.add_measurement_set( utc,    ('llh', pos),      ('tangent_altitude',look) )
        measgeom.make_observation_set( platform )
```

## Z-Axis unit vector
The Z axis unit vector, which is perpendicular to the boresight, is not handled very consistently at the moment and I think we need to think about a better way. The current code in limb mode forces the Z axis unit vector to be perpendicular to the boresight and as parallel as possible to local up at the tangent point. This is a reasonable plan but it is fragile, for example, you dont get sensible results if we have look vectors in _limb_ mode that are looking upwards or into the ground.  Similarly nadir mode forces the Z axis unit vector to be rotated to an azimuth at the ground point. Again this is reasonable but is fragile. The current code has no scenario for calculating the Z axis unit vector for instruments looking upwards which seems like an omission to me that should be rectified.
