(_changelog)=
# Changelog

## Unreleased
- Matrix-free retrievals and orbital-plane tomography support (#61)
- CI and docs builds migrated to `uv`
- Python 3.13 and 3.14 are now tested
- Removed the `skretrieval.legacy` module, `skretrieval.core.platform`, and
  `SASKTRANRadiance.from_sasktran_legacy_xr`; support for the original SASKTRAN model has been dropped
- `jdcal` and `astropy` are no longer dependencies
- Fixed several bugs in the DOAS fitter, lineshapes, and `AdditiveSpline`
- Each `Retrieval` subclass now has its own optical property, prior, and state
  registries, so packages registering the same names on their own subclasses no
  longer replace each other. Subclasses still resolve registrations made on `Retrieval`
- Aerosol state elements now honour `log_space`, set per retrieved quantity or for
  the whole aerosol. Aerosol `initial_guess` values are given in physical units.
  Log-space absorber priors are now built in log coordinates, which slightly changes
  existing log-space absorber results. The Gaussian aerosol ignores `scale_factor`,
  which previously had no effect on the retrieval

## 2026.7.0
- Fix docs examples (#58)
- More options for DOAS (#59)
- Fix VMR error scaling and grid diagnostics (#60)

## 2026.6.0
- Allow injecting constituents from the input dict into the ancillary object (#50)
- Initial volume emission rate (VER) retrieval (#56)
- Cloud retrieval (#55)
- Initial DOAS/BrO code (#57)

## 2025.6.0
- Fix tests with newer versions of `sasktran2` (#45)

## 2025.02.1
- Miscellaneous updates (#43)

## 2025.02.0
- Simplified CI structure (#34)
- Add a non-log option to the triplet measurement vector (#38)
- Bug fixes (#40)

## 2025.01.0
- Multiple fixes and cleanups for SHOW (#32)
- Add polarization support to `IdealSpectrograph` (#33)

## 2024.10.0
- Add ancillary data, update state elements (#23)
- Add logistic bounds mapping (#24)
- Tweaks to the default aerosol state vector element (#25)
- Various improvements to the scipy minimizer (#26)
- Only the required wavelengths are calculated in the forward model (#27)
- Add the concept of a retrieval context (#28)
- Bug fixes (#29)
- Devcontainer support (#31)

## 2024.09.1
- Major release with large updates to the user interface

## 2024.09.0
- Fix nans in tomography grid

## 2024.08.1
- Disallow nan values in the tomography code

## 2024.08.0
- Some bug fixes in the tomography module

## 2024.06.0
- Update the internal platform module to match `skplatform` 0.2.6

## 2024.02.0
- First release with preliminary `sasktran2 support


## 0.3.0
- Last release before moving over to `sasktran2`
