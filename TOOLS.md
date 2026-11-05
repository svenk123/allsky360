# Allsky360 Tools

So, I've built a few command-line tools that work alongside the main `allsky360-camera` application. They're in the `tools/` directory and handle various tasks like fetching weather data, listing cameras, reading the mount position and the RS485 sensors, and creating timelapse videos. Each one is built separately and can be installed to `/opt/allsky360/bin/` or just run from wherever you built them.

---

## Overview

Here's what's in the `tools/` directory:

* **allsky360-aurora** – Grabs aurora forecast and space weather data from NOAA/ESA
* **indigo-cameras** – Lists the INDIGO cameras or reads the sensor data of one camera
* **allsky360-meteo** – Fetches weather data from Open-Meteo
* **timelapse-cuda-nv** – Creates timelapse videos from image sequences, with GPU acceleration
* **allsky360-sensors** – Reads the local RS485/Modbus sensors (weather sensor, anemometer, wind vane, control cabinet) into the daily database

Each tool has its own Makefile, so you can build them individually if you want.

---

## 1. allsky360-aurora

**Location:** `tools/aurora/`  
**Binary:** `bin/allsky360-aurora`

### What it does

This tool fetches aurora forecast data, the planetary Kp index, and solar wind information from space weather APIs. I use it to track space weather conditions and correlate them with my sky observations. All the data gets stored in a SQLite database so you can analyze it later or use it for overlays on your images.

### Running automatically

By default, this runs via systemd every 5 minutes. The service files are:

* **Timer file:** `/etc/systemd/system/allsky360-aurora.timer`
* **Service file:** `/etc/systemd/system/allsky360-aurora.service`

#### Managing the service

Here are the commands you'll need:

Start the timer:
```bash
sudo systemctl start allsky360-aurora.timer
```

Stop it:
```bash
sudo systemctl stop allsky360-aurora.timer
```

Check if it's running:
```bash
sudo systemctl status allsky360-aurora.timer
```

Enable it to start on boot:
```bash
sudo systemctl enable allsky360-aurora.timer
```

Or disable it:
```bash
sudo systemctl disable allsky360-aurora.timer
```

#### Changing how often it updates

If you want to change the update interval (default is 5 minutes), edit the timer file:

```bash
sudo nano /etc/systemd/system/allsky360-aurora.timer
```

Look for line 6 (`OnUnitActiveSec=5min`) and change it to whatever you want. For example:

* `OnUnitActiveSec=1min` – Every minute (probably overkill, but you do you)
* `OnUnitActiveSec=10min` – Every 10 minutes
* `OnUnitActiveSec=30min` – Every 30 minutes

After editing, reload systemd and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart allsky360-aurora.timer
```

### Configuration

**Important:** You need to have your longitude and latitude set correctly in the `config.ini` file. The tool uses these coordinates to fetch location-specific aurora forecasts, so make sure they're accurate.

### Manual usage

Even though it runs automatically, you can also run it manually from the command line if you want to test it or troubleshoot:

```bash
./bin/allsky360-aurora --config <config_file> [OPTIONS]
```

### Parameters

| Parameter | Short | Description |
|-----------|-------|-------------|
| `--config` | `-c` | **Required.** Path to your config file |
| `--longitude` | `-l` | Longitude of your site (e.g., `7.0` or `122.5W`) |
| `--latitude` | `-b` | Latitude of your site (e.g., `50.2` or `60.0N`) |
| `--timestamp` | `-t` | Optional UTC timestamp (format: `YYYYMMDDhhmmss`) |
| `--outfile` | `-f` | Write output to a file instead of stdout |
| `--databasedir` | `-d` | Where to put the database |
| `--kp` | | Include the planetary Kp index (1-minute estimate, falls back to the 3-hourly value) |
| `--mag` | | Include the solar wind magnetic field (Bt, Bz in GSM) |
| `--plasma` | | Include solar wind plasma data (density, speed, temperature) |
| `--debug` | `-v` | Enable debug output (useful when things go wrong) |
| `--help` | `-h` | Show the help message |

### Data sources (NOAA SWPC)

| Value | Endpoint |
|-------|----------|
| Aurora probability | `json/ovation_aurora_latest.json` (OVATION model) |
| Kp | `json/planetary_k_index_1m.json`, fallback `products/noaa-planetary-k-index.json` |
| Bt, Bz | `json/rtsw/rtsw_mag_1m.json` |
| Density, speed, temperature | `json/rtsw/rtsw_wind_1m.json` |

The real-time solar wind files contain one row per minute and spacecraft; only the spacecraft marked `active` (the one SWPC uses operationally) is used, and only values at most 30 minutes from the requested time. Values that are not requested or not available are `null` in the JSON and `NULL` in the database.

### Example

Here's how I test it manually:

```bash
./bin/allsky360-aurora --config /home/allskyuser/config.ini \
  --longitude 7.0 --latitude 50.2 \
  --kp --mag --plasma \
  --databasedir /home/allskyuser/database \
  --outfile /tmp/aurora_test.json \
  --debug
```

---

## 2. indigo-cameras

**Location:** `tools/indigo-cameras/`  
**Binary:** `bin/indigo_cameras` (installed to `/opt/allsky360/bin/indigo_cameras`)

### What it does

Connects to an INDIGO server and either lists the cameras it offers or reads
the sensor data of one camera (`CCD_INFO`: size, pixel size, binning, bit
depth, plus the sensor temperature). Use the list to find the exact device
name for `indigo_camera` in `config.ini`; `install.sh` uses it for the
camera choice in the configuration assistant. Agents such as the "Imager
Agent" are not listed. A camera that is not connected is connected for the
query and disconnected again.

### Usage

```bash
./bin/indigo_cameras -h <server> [-p <port>] [-l] [-c <camera>] [-t <secs>] [-j]
```

| Parameter | Description |
|---|---|
| `-h`, `--host` | INDIGO server (host name or IP) |
| `-p`, `--port` | port, default 7624 |
| `-l`, `--list-cameras` | list the cameras and exit (waits up to `-t`, stops 1 s after the last device answered) |
| `-c`, `--camera` | device name as listed, e.g. `"Poseidon-C PRO @ indigo_server"` |
| `-t`, `--timeout` | seconds to wait, default 10 |
| `-j`, `--json` | JSON output |

### Example

```bash
./bin/indigo_cameras -h localhost -l
./bin/indigo_cameras -h localhost -l -j
./bin/indigo_cameras -h localhost -c "Poseidon-C PRO @ indigo_server" -j
```

---

## 3. allsky360-meteo

**Location:** `tools/meteo/`  
**Binary:** `bin/allsky360-meteo`

### What it does

This fetches meteorological data (temperature, humidity, cloud cover, etc.) from the Open-Meteo API. I use it to correlate weather conditions with my sky observations – it's interesting to see how cloud cover affects what I can capture. The data gets stored in a SQLite database just like the aurora data.

One request per run fetches the hourly values from the current hour on for the
next 24 hours. The current hour goes into the database (table `meteo`) and
`latest_meteo_data.json`; with `--forecastfile` the 24-hour forecast is written
to `allsky-api/public/forecast_meteo.json` (overwritten on every run, not
stored in the database). `meteo.html` shows it as a timeline: cloud cover per
layer, precipitation, temperature/dew point and wind, night hours shaded, plus
the longest clear stretch of the coming night.

### Running automatically

This also runs via systemd by default, every 30 minutes (the weather models are recomputed every 1-3 hours, the current values come from the station sensors). The service files are:

* **Timer file:** `/etc/systemd/system/allsky360-meteo.timer`
* **Service file:** `/etc/systemd/system/allsky360-meteo.service`

#### Managing the service

Same commands as the aurora tool:

Start:
```bash
sudo systemctl start allsky360-meteo.timer
```

Stop:
```bash
sudo systemctl stop allsky360-meteo.timer
```

Status:
```bash
sudo systemctl status allsky360-meteo.timer
```

Enable on boot:
```bash
sudo systemctl enable allsky360-meteo.timer
```

Disable:
```bash
sudo systemctl disable allsky360-meteo.timer
```

#### Changing the update interval

Edit the timer file:
```bash
sudo nano /etc/systemd/system/allsky360-meteo.timer
```

Change `OnUnitActiveSec=30min` to your desired interval:

* `OnUnitActiveSec=15min` – Every 15 minutes
* `OnUnitActiveSec=60min` – Every hour

Then reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart allsky360-meteo.timer
```

### Configuration

**Important:** Like the aurora tool, this needs correct longitude and latitude in your `config.ini` file to fetch location-specific weather data.

### Manual usage

You can run it manually too:

```bash
./bin/allsky360-meteo --config <config_file> [OPTIONS]
```

### Parameters

| Parameter | Short | Description |
|-----------|-------|-------------|
| `--config` | `-c` | **Required.** Path to your config file |
| `--timestamp` | `-t` | Optional UTC timestamp (format: `YYYYMMDDhhmmss`) |
| `--outfile` | `-f` | Write output to a file instead of stdout |
| `--databasedir` | `-d` | Where to put the database |
| `--forecastfile` | `-F` | Write the hourly forecast of the next 24 hours as JSON to this file |
| `--debug` | `-v` | Enable debug output |
| `--help` | `-h` | Show help message |

### Example

```bash
./bin/allsky360-meteo --config /home/allskyuser/config.ini \
  --databasedir /home/allskyuser/database \
  --timestamp 20250101120000 \
  --outfile /tmp/meteo_test.json \
  --debug
```

---

## 4. timelapse-cuda-nv (allsky360-timelapse)

**Location:** `tools/timelapse/`  
**Binary:** `bin/timelapse-cuda-nv` or `bin/allsky360-timelapse`

### What it does

This is probably my favorite tool. It creates timelapse videos from a sequence of JPEG images, and it uses GPU acceleration for JPEG decoding and image resizing (the H.264 encoding runs on the CPU with libx264, the Jetson Orin Nano has no hardware video encoder). If you've got hundreds or thousands of images, this makes a huge difference. I've processed sequences with thousands of images in minutes instead of hours.

It also supports different stacking modes, which is really useful if you want to combine multiple images in interesting ways.

### Features

* GPU-accelerated JPEG decoding (using nvJPEG)
* GPU-accelerated image resizing (using NPP)
* H.264 video encoding with libx264 (CPU)
* Multiple stacking modes (average, max, min, sum, sigma-clip, difference, motion)
* Progress bar so you know it's actually working

### Usage

```bash
./bin/timelapse-cuda-nv -o <output.mp4> -w <width> -h <height> -p <fps> [OPTIONS] image1.jpg image2.jpg ...
```

### Parameters

| Parameter | Short | Description |
|-----------|-------|-------------|
| `-o` | | **Required.** Output video filename (MP4 format) |
| `-w` | | **Required.** Target video width in pixels |
| `-h` | | **Required.** Target video height in pixels |
| `-p` | | **Required.** Frame rate (frames per second) |
| `-v` | | Enable verbose output with progress bar (I always use this) |
| `-s` | | Save the final stacked result as a PNG file |
| `-m` | | Stacking mode: `avg`, `max`, `min`, `sum`, `sigma`, `diff`, or `motion` |
| `-t` | | Sigma clipping threshold (default: `2.0`, used with `-m sigma`) |

### Stacking modes

Here's what each mode does:

* `avg` – Averages all images (this is the default and usually what you want)
* `max` – Takes the maximum pixel values (good for star trails)
* `min` – Takes the minimum pixel values
* `sum` – Sums all pixels (can get very bright, watch out for overflow)
* `sigma` – Sigma-clipped average (removes outliers, great for noise reduction)
* `diff` – Shows the difference between frames (useful for detecting changes)
* `motion` – Motion detection and stacking (experimental, but interesting)

### Example

Here's how I usually run it:

```bash
./bin/timelapse-cuda-nv -o night_sky.mp4 -w 1920 -h 1080 -p 25 -v \
  -m avg -s stacked_result.png \
  image001.jpg image002.jpg image003.jpg ...
```

Or if you've got a bunch of images in a directory, you can use a wildcard:
```bash
./bin/timelapse-cuda-nv -o night_sky.mp4 -w 1920 -h 1080 -p 25 -v \
  -m avg images/*.jpg
```

### Requirements

You'll need:
* An NVIDIA GPU with CUDA support
* CUDA Toolkit (11.x or 12.x should work)
* FFmpeg development libraries
* NVIDIA libraries: nvJPEG, NPP
* FFmpeg with libx264

---

## Building the tools

You can build each tool individually from its directory:

```bash
cd tools/aurora && make
cd tools/meteo && make
cd tools/timelapse && make
cd tools/indigo-cameras && make
```

Or build all of them at once from the main directory:

```bash
make tools
```

The binaries will end up in each tool's `bin/` subdirectory.

---

## Installation

If you want to install all the tools to `/opt/allsky360/bin/`, use the main Makefile:

```bash
sudo make install
```

This installs all the tools along with the main `allsky360-camera` application.

---

## A few notes

* Most tools need a configuration file with your location coordinates and other settings. Make sure your `config.ini` is set up correctly.

* The database tools (aurora and meteo) store everything in SQLite format. You can query it later with sqlite3 or use it in your own scripts.

* The timelapse tool really needs an NVIDIA GPU to work well. If you try to run it without CUDA, it'll probably complain or be really slow.

* All the tools follow the same coding standards and license (MIT) as the main application. If you want to modify them or add features, go for it!

---

## allsky360-sensors (RS485 / Modbus sensors)

Reads the sensors on the station's RS485 bus once per run and stores one row
in the table `sensors` of the daily database (`database_YYYYMMDD.db`, UTC day,
same file as the camera data). The last reading is also written to
`allsky-api/public/latest_sensor_data.json` (served as `/latest_sensor_data.json`);
the history of a day comes from `GET /api/sensors?date=YYYY-MM-DD`.

The web page `meteo.html` (Weather) shows the latest reading as tiles (marked
*stale* when it is older than 15 minutes, condensation warning for the control
cabinet) and draws the station values next to the Open-Meteo forecast, plus
charts for rain/illuminance, air quality/noise and the control cabinet.

Check the last reading on the station:

```bash
cat /opt/allsky360/allsky-api/public/latest_sensor_data.json
curl -s http://localhost:3000/latest_sensor_data.json
sqlite3 -readonly -header -column /home/allskyuser/database/database_$(date -u +%Y%m%d).db \
  "SELECT datetime(timestamp,'unixepoch','localtime') AS time, temperature, humidity,
          pressure, wind_speed, wind_direction, cabinet_temperature
   FROM sensors ORDER BY timestamp DESC LIMIT 5"
```

| Sensor | Values | config.ini |
|---|---|---|
| Weather sensor Halisense HLS-UWS-SDTHA | temperature, humidity, dew point, pressure, wind speed/direction, noise, PM2.5/PM10, illuminance, rain | `modbus_weather_*` |
| Anemometer | wind speed (m/s = raw / scale) | `modbus_wind_speed_*` |
| Wind vane | wind direction (+ mounting offset) | `modbus_wind_direction_*` |
| Control cabinet sensor | temperature, humidity, dew point | `modbus_cabinet_*` |

### Configuration

All settings are `modbus_*` keys in `config.ini` (see the commented block in
`config.ini.default`): serial device and baud rate, the GPIO that switches the
RS485 transceiver direction (`modbus_gpio_line=-1` for adapters with automatic
direction control), timeout, attempts, and per sensor the slave address and
register. **Address `0` disables a sensor**; with all addresses `0` (the
default) the tool does nothing. `sudo ./install.sh --configure` asks for the
four addresses.

Values outside plausible ranges (e.g. humidity > 100 %, pressure outside
300-1100 hPa) are stored as `NULL`, as are values of sensors that did not
answer.

In calm air the wind vane turns at random. When the anemometer reports less
than `modbus_wind_calm_speed` (default `0.3` m/s, Beaufort 0; `0` switches it
off), the tool stores the last stored `wind_direction` (today's or yesterday's
database) instead of the measured one; without an earlier direction it stores
`NULL`. If the anemometer did not answer, the measured direction is kept.

### Running automatically

`allsky360-sensors.timer` starts the tool every 5 minutes (`OnUnitActiveSec=5min`,
change it with `sudo systemctl edit allsky360-sensors.timer`). Like the other timers
it is installed but not enabled by `make install`:

```bash
sudo systemctl enable --now allsky360-sensors.timer
journalctl -u allsky360-sensors.service -f
```

The service account needs the groups `dialout` (serial port) and `gpio`
(direction GPIO on the Jetson); `make install` adds them.

### Manual usage

```bash
allsky360-sensors -c /home/allskyuser/config.ini -n -v   # read and print, no database
allsky360-sensors -c config.ini -d /home/allskyuser/database -f latest_sensor_data.json
```

| Option | Meaning |
|---|---|
| `-c, --config FILE` | config.ini (required) |
| `-d, --databasedir DIR` | override `database_directory` |
| `-f, --outfile FILE` | also write the reading as JSON (atomic) |
| `-n, --dry-run` | read and print, do not write the database |
| `-v, --verbose` | print the configuration and every failed attempt with the received bytes |

Exit status: `0` at least one sensor read (or none configured), `2` no
configured sensor answered, `1` configuration, bus or database error.

### Table `sensors`

| Column | Unit | Source |
|---|---|---|
| `timestamp` | Unix time (UTC) | |
| `temperature`, `humidity`, `dew_point` | °C, %RH, °C | weather sensor (dew point calculated) |
| `pressure` | hPa | weather sensor |
| `ws_wind_speed`, `ws_wind_direction` | m/s, ° | weather sensor |
| `noise`, `pm25`, `pm10`, `illuminance`, `rainfall` | dB, µg/m³, µg/m³, lx, mm | weather sensor |
| `solar_irradiance_raw` | raw | weather sensor, register unverified |
| `wind_speed`, `wind_direction` | m/s, ° | anemometer, wind vane |
| `cabinet_temperature`, `cabinet_humidity`, `cabinet_dew_point` | °C, %RH, °C | control cabinet sensor |

The register maps are taken from the stand-alone test programs in
`examples/` (weather-sensor, wind-speed, wind-direction, temp-humidity-sensor).

