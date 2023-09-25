<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2023-11-xx

### Added

- Added an example for 1D simulation of blood flow dynamics

### Changed

### Deprecated

### Removed

### Fixed

- Fixed an issue when loading a checkpoint which contains `scaler_state_dict`, but no `scaler` provided to update. 

### Security

### Dependencies

## [0.3.0] - 2023-09-21

### Added

- Added support for CUDA Graphs and AMP for the DLWP example
- Recipe for downloading ERA5 datasets for CDS API.

## [0.2.0] - 2023-08-10

### Added

- Added a CHANGELOG.md
- Ahmed body recipe
- Documentation for SFNO, GraphCast, vortex shedding, and Ahmed body
- Documentation for DLWP, and RNN examples

### Changed

- Updated the SFNO example
- Changed the default SFNO configs
- Header test to ignore .gitignore items
- Sample download scripts in the DLWP example

### Fixed

- Fixed training checkpoint function for updated static capture
- Brought back the dataset download script for vortex shedding that was accidentally removed

### Dependencies

- Updated the base container to latest PyTorch base container which is based on torch 2.0
- Container now supports CUDA 12, Python 3.10

## [0.1.0] - 2023-05-08

### Added

- Initial public release.
