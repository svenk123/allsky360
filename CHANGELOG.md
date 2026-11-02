# 📋 Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Support for JSON-based star detection overlays
- Basic command-line tool for aurora data retrieval from NOAA/ESA
- Optional debug image output for HDR merge function

### Changed
- Improved tone mapping algorithm for HDR output
- Updated default mask settings in star detection module

### Fixed
- Corrected JPEG loading bug with images >16 MB on Jetson Orin NX

---

## [1.0.0] – 2025-07-13

### Added
- Initial public release of the Allsky Kamera core modules
- C implementation of:
  - JPEG-based timelapse generator with optional NVENC + nvJPEG support
  - HDR image merging (5 exposures, float-based)
  - Tonemapping and linear autostretching
  - Star detection with JSON output for overlays
  - Background neutralization and white balance functions
- Full command-line interface with flexible parameters
- Modular directory structure with CMake-based build system
- Example input/output images and test scripts
- MIT license and documentation (README, CONTRIBUTING, etc.)
