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
- Line shapes are now applied with [rebasis](https://usask-arg.github.io/rebasis/) (`pyrebasis`) as sparse
  matrices, see `skretrieval.core.lineshape.measurement_matrix`.  The spectrographs, the forward model
  `lineshape_fn`, and DOAS template convolution accept `rebasis.Space` kernels.  Gaussian integrals are
  now exact rather than using a fast erf approximation (changes of ~1e-3 relative), and `Rectangle`
  edges on samples and `UserLineShape(mode="integrate")` are fixed.  The `LineShape` classes are kept for
  backwards compatibility; custom subclasses still use their `integration_weights`
- With `spectral_native_coordinate="wavenumber_cminv"`, the forward model grid now spans the line shape
  bounds in wavenumber.  Previously the bounds (in cm^-1) were applied as offsets in nm, making the grid
  ~5x too wide near 1.4 um (too narrow beyond ~3.2 um).  Results change slightly where the line shape
  has significant area outside its `bounds`, e.g. a `UserLineShape` with `integration_fraction < 1`

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
