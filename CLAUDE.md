# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Allsky360 is an allsky camera system for continuous sky observation. It is a C application that connects to cameras via the [INDIGO](https://github.com/indigo-astronomy/indigo) astronomy protocol and processes images through a GPU-accelerated pipeline. The main binary is `bin/allsky360-camera`.

The INDIGO library is expected to be cloned as a sibling directory (`../indigo`) and built there. The Makefile links against `../indigo/build/lib/libindigo`.

## Build Commands

Install system dependencies (requires root):
```bash
sudo make deps
```

CPU-only build:
```bash
make
```

GPU-accelerated build (NVIDIA CUDA):
```bash
make USE_CUDA=1
```

Build all companion tools (timelapse, meteo, aurora, location, panorama, list-cameras):
```bash
make tools
```

Build a specific tool:
```bash
make timelapse   # or: meteo, aurora, location, list-cameras, panorama
```

Build the Node.js web API and bundle dependencies (requires Node.js 24, install with `sudo make node-deps`):
```bash
make node-build
```

Run tests:
```bash
make tests
```
Test sources are in `test/`; the built test binary goes to `bin/test_sun_calc`.

Clean build artifacts:
```bash
make clean
```

## Installation

One-step install (preflight, deps, build as the sudo user, `make install`, configuration assistant, optional YOLO, self-test): `sudo ./install.sh` (`--help` for options, `--configure` for the assistant only). It orchestrates the Makefile targets below.

Full system install (requires root, installs to `/opt/allsky360/`):
```bash
sudo make install
```

The install targets create a dedicated `allskyuser`/`allskygroup`, copy binaries to `/opt/allsky360/bin/`, install systemd services and timers (aurora, meteo, location, mount-position). Configuration is installed to `/home/allskyuser/config.ini`.

Run the camera in development mode (without installing):
```bash
make camera-run
```

Run the API server in development mode:
```bash
make node-run
```

View live service logs:
```bash
make log
```

## Architecture

### Main Application (`src/`, `include/allsky.h`)

The main process (`src/main.c`) connects to an INDIGO server over TCP (default port 7624) and drives the capture-process-save loop. Key subsystems:

- **Camera I/O**: `indigo_raw_to_rgbf.c` — converts raw INDIGO frames to float RGB
- **HDR Capture**: Up to 5 bracketed exposures (`MAX_FRAMES = 5`) merged via `hdr_merge.c`
- **Processing pipeline** (executed in this order after HDR merge):
  1. `hdr_merge_rgbf()` — combines multiple exposures
  2. `tone_map_rgbf()` — compresses HDR dynamic range
  3. `autostretch_rgbf()` — histogram stretch
  4. `white_balance_rgbf()` — color/white balance + background neutralization
  5. `acdnr_filter` — adaptive noise reduction
  6. `wavelet_sharpen` — multi-scale sharpening
  7. `clarity_filter` — local mid-scale contrast
  8. `dehaze_filter` — atmospheric haze removal
  9. `gamma_apply_rgbf()` — linear → sRGB conversion
  10. `saturation_rgbf()` — color intensity adjustment
  11. JPEG encode (CPU libjpeg or GPU nvjpeg)
- **Astronomy overlays**: `sun_calc.c`, `moon_calc.c`, `star_detector.c`, `image_overlay.c`, `image_metadata.c`
- **Autoexposure**: `autoexposure.c`
- **Debayering**: Three algorithms — `debayer_nni.c`, `debayer_bilinear.c`, `debayer_vng.c`
- **Database**: `daily_database.c` writes per-image metadata to SQLite
- **AI**: `scripts/postprocessing.sh` classifies sky conditions with an Ultralytics YOLO classification model (`/opt/allsky360/ai/models/weather.pt`, YOLO26 or YOLO11) via the venv `/home/allskyuser/venvs/yolo` (set up with `sudo make yolo`, check with `make yolo-check`). `ai_training.c` only saves image patches as training data

### CUDA Dual-Implementation Pattern

Every GPU-accelerated filter has three files:
- `foo_filter.c` / `foo_filter.h` — CPU fallback implementation
- `foo_filter_cuda.cu` / `foo_filter_cuda.h` — GPU CUDA kernel
- `foo_filter_cuda_wrapper.c` — thin C wrapper that calls the CUDA version

When `USE_CUDA=1`, the CUDA wrappers are compiled and linked; otherwise the CPU versions are used. The Makefile handles this conditional compilation automatically.

### Companion Tools (`tools/`)

Each tool has its own Makefile and produces a binary in its `bin/` subdirectory:
- `tools/timelapse/` — GPU-accelerated JPEG→MP4 timelapse (nvJPEG + NVENC)
- `tools/aurora/` — fetches NOAA/ESA aurora/Kp data → SQLite
- `tools/meteo/` — fetches Open-Meteo weather data → SQLite
- `tools/location/` — calculates magnetic declination and astronomical data
- `tools/panorama/` — panorama stitching
- `tools/indigo-list-cameras/` — lists INDIGO-connected cameras
- `tools/indigo-mount-status/` — mount status tool

### Web API (`allsky-api/`)

Node.js REST API (`allsky-api/index.js`, plain `http` module, Node.js >= 24) serving images, database data, and the panorama viewer. The astronomia library is bundled for browser use via esbuild (`npm run build`, entry `astro_bundle.js`) into `public/js/astronomia.bundle.min.js`; esbuild and astronomia are devDependencies and are pruned after `make install`. npm 11 only runs install scripts listed in `allowScripts` in `package.json` (sqlite3, esbuild) - bump those entries when upgrading these packages. Runs as systemd service `allsky360-api.service` on port 3000 (configurable via `api-port` in `config.ini`).

### Configuration

The main config file (`config.ini`) uses simple `key=value` INI format. `config.ini.default` is the reference template. Example hardware-specific configs are `config.ini.saturn-c` and `config.ini.uranus-c`. Key sections:
- Location coordinates (latitude/longitude/altitude)
- INDIGO server connection (IP, port, camera name)
- Camera parameters (exposure range, gain, debayer algorithm, binning, cooling)
- Image processing parameters (JPEG quality, HDR exposures, filter strengths)
- Directory paths for images, videos, and database

## Code Style

C code uses GNU11 (`-std=gnu11`), `-Wall -Wextra -O2`. Run `clang-format` before submitting patches. All processing operates on float RGBA images in linear 0.0–1.0 space; gamma is applied last before JPEG encoding. OpenMP is enabled (`-fopenmp`) for CPU parallelism.
