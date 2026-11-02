# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Small, independent C command-line tools for an NVIDIA Jetson (Orin, hostname `orinhsn8`) that talk to RS485/Modbus sensors and GPIO hardware. They are invoked by an external poller/agent (`poller/weather_poller.py` and a dashboard, not in this repo) that parses their stdout, so **output format and exit codes are a contract** (exit `0` success, `1` any error). READMEs are in German; code comments and identifiers are mixed German/English.

There is no top-level build, no test suite, no linter and the directory is not a git repo. Each tool is built on its own; compiled binaries (and `meteo/obj`, `meteo/bin`) are checked in alongside the sources.

## Build

Per tool, from its directory: `make`, `make clean`, `sudo make install` (installs to `/usr/local/bin`). Tools with a plain Makefile compile one `.c` with `-Wall -Wextra -O2 -lgpiod` (needs `libgpiod` dev package, v1 API-style `gpiod.h`).

- `meteo/` has a multi-file Makefile (`src/*.c` → `obj/` → `bin/allsky360-meteo`), needs `libcurl4-openssl-dev libjson-c-dev libsqlite3-dev`.
- There are no automated tests; verification is done by running against real hardware (see below).

## Architecture

- **RS485 Modbus-RTU sensor tools** (`weather-sensor`, `temp-humidity-sensor`, `modbus_tool`): each is a self-contained file implementing its own Modbus RTU framing/CRC over `/dev/ttyTHS1` (4800 baud, termios). The RS485 transceiver direction (DE/RTS) is **toggled manually via GPIO** (`/dev/gpiochip0`, line offset 112 = PR.04) around each transmit, not by kernel RS485 mode. These constants are duplicated across the three files — change them in all of them. Reads use retries (5 × 300 ms) and map failures to error codes.
- **`modbus_tool`**: scans addresses 1–247 (`--scan -r REG`) or rewrites a sensor's slave address (`--set-address -c OLD -n NEW -r 0x7D0`). The write goes to real hardware immediately; the config register deliberately has no default.
- **`weather-sensor`**: Halisense HLS-UWS-SDTHA (DFRobot-SEN0658-Registerkarte, Messwerte ab `0x01F4`, im Test auf Slave-Adresse 5), Block-Read mit Einzelregister-Fallback. Humidity/Temp/Noise/Pressure sind am Sensor verifiziert, Wind/PM/Licht/Regen nicht, die Solarstrahlungs-Adresse (`0x0202`) ist geraten. Output is a flat JSON object (not the `success`/`result` wrapper).
- **`magnetventil-steuern`**: controls up to 4 solenoid valves via a PCA9534 I2C GPIO expander (`gpiochip2`, valves 1–4 → offsets 3,2,1,0, active-HIGH). `--action ein` requires `--max_duration` (1–1800 s) and forks a detached **watchdog child** that opens/holds the GPIO line exclusively and forces it LOW on timeout, so the valve closes even if the caller dies. Parent only waits on a pipe for the child's confirmation. PID files live in `/run/magnetventil/` (must pre-exist; tmpfs, see its README for the tmpfiles.d rule); log at `/var/log/magnetventil_watchdog.log`. Preserve the invariant that `pipe()`+`fork()` happen before any GPIO access, so failure never opens a valve.
- **`wind-direction`, `wind-speed`**: single-sensor Modbus tools (register 1, address default 1) with the `success`/`result` JSON wrapper via `common/json_output.h`; `wind-direction` has `-o` mounting offset + 16-point compass, `wind-speed` has `-s` divisor (default 1, **unverified** against a known wind speed).
- **`modbus-scan`**: scans an address range at a selectable baud rate (default 1–247, 4800); exit 2 = nothing found (not an error). Default timeout 1200 ms × 2 tries because the sensors take ~1 s to answer.
- **`gpio-output`** (`gpio_out`): permanent on/off for OUT1–OUT4 (same PCA9534/`gpiochip2` as `magnetventil-steuern`). Because a HIGH line drops to LOW as soon as its requester exits, `ein` forks a detached holder process that renames itself `gpio_out_N` (found again via `/proc/*/comm`, no pid files); `aus` SIGTERMs it. No timer/watchdog.
- **`meteo/`**: vendored from `svenk123/allsky360` (`tools/meteo`), fetches Open-Meteo data (not a local sensor) and prints JSON; wind speeds are km/h. Local changes vs upstream: install path and `-a/--latitude`, `-o/--longitude` flags.
- **`common/json_output.h`**: shared header with `JSON_SUCCESS_*` / `JSON_ERROR` macros and `json_escape`. Currently only `magnetventil_steuern` uses it (`-I../common`); the other tools emit the same JSON shape themselves.

## Output / error conventions

Tools emit one-line JSON: `{"success": true, "tool": "...", "result": {...}, "timestamp": <unix>}` or `{"success": false, "tool": "...", "error": "...", "error_code": "...", "metadata": {"retry_safe": bool}}`. Shared error codes: `INVALID_ARGS`, `HARDWARE_NOT_FOUND`, `PERMISSION_DENIED`, `SENSOR_TIMEOUT` (retry_safe), `SENSOR_OUT_OF_RANGE`, `GPIO_WRITE_FAILED`, `WATCHDOG_INIT_FAILED`, `UNKNOWN_ERROR`. `weather-sensor` and `modbus_tool` use `-j` for JSON (plaintext default); `magnetventil_steuern` is always JSON. When adding a tool or error path, follow this shape and document the code in that tool's README table.
