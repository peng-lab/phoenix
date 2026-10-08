# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions
- `SpatialDataset(native_mpp=...)` to override the H&E pixel size.

### Fixed

- `SpatialDataset` now takes the H&E pixel size from the store's `spatialdata_io_reader` attributes (`source_mpp` for
  `xenium`, `source_he_mpp` for `he`). H&E-only stores, whose transformations are identity, were assumed to be 1.0
  um/px, so their 112 px crop covered ~25 um instead of 112 um. **Predictions on H&E-only stores change** and should be
  regenerated; Xenium stores are bit-identical. A store with neither the attributes nor a micron-scaled
  `nucleus_boundaries` now raises instead of silently assuming 1.0 um/px.
