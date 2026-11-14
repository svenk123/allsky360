# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- One-step installation `sudo ./install.sh` with preflight checks, configuration
  assistant (location, INDIGO server, camera, mount, API token, sensors) and
  self-test; `--configure` runs the assistant alone.
- `allsky360-sensors`: RS485/Modbus RTU sensors (weather sensor, anemometer,
  wind vane, control cabinet) as a continuous service; 10-minute wind vector
  mean, readings every 5 minutes in the daily database and in
  `latest_sensor_data.json`.
- `allsky360-keogram` and `allsky360-startrails` for every night, created by
  `endofnight.sh` and shown on the new page `night.html` (`/api/nights`).
- `allsky360-mount-position`: reads the INDIGO mount position (read-only) into
  `mount.json` with `status` and `timestamp`; `indigo-cameras -M` lists mounts.
- SQM estimate from the stars detected on the raw frame of HDR frame 0.
- Weather: 24 h Open-Meteo forecast, station sensor tiles, wind rose and wind
  direction chart, cloud cover per layer on `meteo.html`.
- Aurora: real-time solar wind (NOAA RTSW).
- Web interface: shared navigation, status page, quieter design, dashboard
  overlays that stay in place when zoomed, toggle between station time and UTC.
- Station time zone `location_timezone` for the time printed into the image.
- Archive retention (`retention_*`): limits by age and size.
- AI sky classification with Ultralytics YOLO26 (`sudo make yolo`).
- `make camera-run` runs the camera in the foreground like the service.
- Tone curve, Reinhard tone mapping and automatic gamma (`tone_curve`,
  `reinhard_tonemap`, `auto_gamma`).

### Changed
- Builds on JetPack 7.2 (INDIGO 3, CUDA 13, gcc 13); API on Node.js 24.
- Everything is stored in UTC (database timestamps, `images/YYYYMMDD`, file
  names); summer/winter time no longer affects stored data.
- HDR: the frames are fused raw and white-balanced once after the fusion;
  fused frame by frame (less memory); brightness no longer depends on the
  number of frames (`hdr_highlight_knee`); OpenMP-parallel fusion on the CPU.
- Meteo queries Open-Meteo every 30 minutes and passes the station altitude.
- Timelapse: mixed image sizes, robust JPEG loading, correct timestamps;
  settings for video files instead of live streams.
- Hardened systemd units; camera with systemd watchdog, `systemctl reload`
  (SIGHUP) and waiting for the INDIGO server; timers fire again after a restart.
- Service control from the web interface via polkit and API token.
- Removed the tools `indigo-list-cameras`, `indigo-mount-status` and
  `location` (replaced by `indigo-cameras` and `install.sh`).
- Removed the never implemented local histogram normalization from the config.

### Fixed
- Config parsing: hemisphere suffixes (N/S/E/W), truncated values.
- Aurora tool crash; missing space-weather values are stored as `null`.
- Raw frame size check, double free with 8-bit Bayer frames, CUDA libjpeg build.
- Correct sun and moon positions and rise/set times in the web interface.

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
