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
* **allsky360-keogram** – Builds a keogram (the north-south meridian of every image over time) of a night or any time range
* **allsky360-startrails** – Stacks the night images into star trails (brightest value of every pixel)
* **allsky360-mount-position** – Reads the position of the telescope mount into `mount.json` (read-only)

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

## 5. allsky360-keogram

A keogram shows a whole night in one picture: every column is the north-south line
through the zenith of one image, the columns run from left to right in time. Clouds,
dawn and dusk, the moon and aurora (as bright bands in the north) are visible at a glance.

### What it does

* Reads the images `image_directory/YYYYMMDD/image-YYYYMMDDHHMMSS.jpg` of the time range
  (UTC days, like everything stored).
* Takes from each image the meridian from the northern horizon (top) through the zenith
  to the southern horizon (bottom): bilinear, averaged over a few pixels across the line.
  The geometry is the calibration from `config.ini`: `image_zenith_x`, `image_zenith_y`,
  `image_horizon_radius` and `image_north_angle` (the same values as the azimuth grid of
  the dashboard).
* Places the columns by time, linearly from start to end: gaps in the capture stay black.
* Decodes the JPEGs reduced (1/2, 1/4 or 1/8 while decoding) and in parallel on all cores:
  a day of about 230 images of 4176x4176 pixels takes about 9 seconds on the Orin Nano.
* Skips unreadable images (for example the one being written) and images of another size
  than the newest image: the calibration fits only one image size (crop switched on or off).
* Writes the keogram as JPEG (temporary file + rename) and optionally a JSON description
  for the web page.

`scripts/endofnight.sh` runs it when the camera switches from night to day (night images of
the last 20 hours, `--night-only --trim`) and writes `keogram_YYYYMMDD.jpg/.json` next to the
timelapse videos; the page **Night** (`night.html`) shows it with a time axis. The code shared
with allsky360-startrails (config, time range, image list, JPEG) is in `tools/common/`.

### Usage

```bash
allsky360-keogram [options]
```

### Parameters

| Option | Description |
|--------|-------------|
| `-n`, `--night DATE` | The night after the evening of DATE (YYYYMMDD): from solar noon to the next solar noon at `location_longitude` |
| `-s`, `--start TIME` | Start (UTC): Unix seconds, `YYYYMMDD`, `"YYYY-MM-DD HH:MM[:SS]"` or `YYYY-MM-DDTHH:MM` |
| `-e`, `--end TIME` | End (default: start + 24 h); without `-n`/`-s` the last 24 hours |
| `--trim` | Shorten the range to the first and last image used (no black margins) |
| `-c`, `--config FILE` | config.ini (default `/home/allskyuser/config.ini`) |
| `-i`, `--images DIR` | Image directory (overrides `image_directory`) |
| `-o`, `--output FILE` | Keogram JPEG (default `keogram.jpg`) |
| `-j`, `--json FILE` | Also write a JSON description (time range, size, number of images) |
| `-W`, `--width N` | Width in pixels (default: one column per capture interval) |
| `-H`, `--height N` | Height in pixels, horizon to horizon (default 720) |
| `-w`, `--slice-width N` | Pixels averaged across the meridian (default 5) |
| `--night-only` | Only images with `night_mode` 1 in their JSON |
| `--zenith X,Y`, `--radius R`, `--north DEG` | Geometry instead of the values in config.ini (for testing a calibration) |
| `-q`, `--quality N` | JPEG quality (default 90) |
| `-v`, `--verbose` | Progress output |

Exit codes: 0 written, 1 usage or configuration error, 2 no images in the range, 3 write error.

### JSON description (`--json`)

```json
{"source":"allsky360-keogram","generated":1793718881,"start":1793689200,"end":1793721600,
 "width":268,"height":720,"seconds_per_column":120.9,"images":223,"images_unreadable":0,
 "images_other_size":11,"image_width":4176,"image_height":4176,"first_image":1793690196,
 "last_image":1793718817,"interval":121.0,"night_only":false,"top":"N","bottom":"S"}
```

Times are Unix seconds (UTC). Column `x` covers the time `start + x * seconds_per_column`.

### Example

```bash
# Last night (the evening of 3 November up to the next noon)
allsky360-keogram -n 20261103 -o /tmp/keogram.jpg -j /tmp/keogram.json -v

# Test a calibration without changing config.ini
allsky360-keogram -s 20261103 --zenith 2088,2088 --radius 2000 --north -11 -o /tmp/k.jpg
```

---

## 6. allsky360-startrails

Star trails: every pixel of the result is the brightest value of that pixel over all images of
the night ("lighten" stacking), so the stars draw their arcs around the celestial pole.

### What it does

* Uses the same image list, time range options and config keys as allsky360-keogram
  (shared code in `tools/common/`).
* Only night images (`night_mode` 1 in the image JSON) by default; `--all` also takes day images.
* `--max-brightness` leaves out images that are brighter on average (0..1). Off by default:
  the camera exposes night images to a target brightness, so their mean says little about
  how dark the sky was (here 0.13–0.35 within one evening). `-v` prints the brightness of
  every image to find a useful value.
* Skips unreadable images and images of another size than the newest one.
* Decodes reduced (`--scale`, default 1/2: 2088x2088 from 4176x4176) and in parallel, with one
  maximum buffer per thread.
* Writes the JPEG (temporary file + rename) and optionally a JSON description.

`scripts/endofnight.sh` writes `startrails_YYYYMMDD.jpg/.json` next to the timelapse videos;
the page **Night** shows it.

Known limit: the text printed into the images (date and time at the top left) is stacked as
well and becomes unreadable.

### Parameters

| Option | Description |
|--------|-------------|
| `-n`, `--night DATE` | The night after the evening of DATE (YYYYMMDD), solar noon to solar noon |
| `-s`, `--start TIME` / `-e`, `--end TIME` | Time range (UTC), as for allsky360-keogram |
| `-c`, `--config FILE` | config.ini (default `/home/allskyuser/config.ini`) |
| `-i`, `--images DIR` | Image directory (overrides `image_directory`) |
| `-o`, `--output FILE` | Star trails JPEG (default `startrails.jpg`) |
| `-j`, `--json FILE` | Also write a JSON description |
| `-b`, `--max-brightness F` | Leave out images brighter than F on average (default 1.0 = keep all) |
| `-S`, `--scale N` | Output at 1/N of the image size: 1, 2, 4 or 8 (default 2) |
| `--all` | Also day images |
| `-q`, `--quality N` | JPEG quality (default 92) |
| `-v`, `--verbose` | Progress and the brightness of every image |

Exit codes: 0 written, 1 usage or configuration error, 2 no (usable) images, 3 write error.

### JSON description (`--json`)

```json
{"source":"allsky360-startrails","generated":1793726141,"start":1793664000,"end":1793750400,
 "first_image":1793723899,"last_image":1793725956,"width":2088,"height":2088,"scale":2,
 "images":19,"images_too_bright":0,"images_unreadable":0,"images_other_size":0,
 "max_brightness":1.0,"night_only":true}
```

### Example

```bash
allsky360-startrails -n 20261103 -o /tmp/startrails.jpg -j /tmp/startrails.json -v
```

---

## 7. allsky360-mount-position

**Location:** `tools/indigo-mount-position/`
**Binary:** `bin/indigo_mount_position`

### What it does

Reads the position of the telescope mount `indigo_mount` from the INDIGO
server and writes it to `mount.json`, which the dashboard shows as an overlay
(checkbox *Mount*). The mount is only read: the tool never connects, moves or
parks it. Another client (AIN, KStars, …) must have connected the mount.

### Running automatically

`allsky360-mount-position.timer` runs the tool every minute:

```bash
sudo systemctl enable --now allsky360-mount-position.timer
```

The service writes `/opt/allsky360/allsky-api/public/mount.json`. Exit status
`2` (mount not available) counts as success for systemd, so a station
without a mount does not show a failed service.

### Output

```json
{
  "timestamp": 1794684438,
  "status": "ok",
  "mount": "Mount LX200 @ indigo_server",
  "equatorial": { "ra_h": 23.545, "ra_str": "23h 32m 42.00s",
                  "dec_deg": -0.0203, "dec_str": "-00° 01' 13.00\"" },
  "horizontal": { "alt_deg": 39.88, "az_deg": 179.56 },
  "side_of_pier": "west",
  "tracking": false
}
```

`timestamp` (Unix time, UTC) and `status` are always present. `status`:

| Value | Meaning |
|---|---|
| `ok` | Position read |
| `not_configured` | `indigo_mount` is empty |
| `disconnected` | The mount exists on the server but no client has connected it |
| `no_response` | The mount did not deliver a position in time |
| `server_unreachable` | The INDIGO server did not answer |

### Manual usage

```bash
indigo_mount_position -c /home/allskyuser/config.ini -j     # position as JSON
indigo_mount_position -h 127.0.0.1 -l                       # list the mounts
```

| Option | Meaning |
|---|---|
| `-c, --config FILE` | config.ini (server, port and mount) |
| `-h, --host SERVER` | INDIGO server |
| `-p, --port PORT` | port (default 7624) |
| `-m, --mount NAME` | mount device |
| `-t, --timeout SECS` | timeout (default 10) |
| `-l, --list-mounts` | list the mounts and exit |
| `-j, --json` | JSON output |
| `-o, --output FILE` | write the JSON to a file (implies `-j`) |

---

## 8. allsky360-sensors (RS485 / Modbus sensors)

Optional. Hardware, wiring, sensor addresses, first start and
troubleshooting are described in [SENSORS.md](SENSORS.md); this section is
the technical reference.

Runs continuously and stores one row every 5 minutes (`modbus_interval`)
in the table `sensors` of the daily database (`database_YYYYMMDD.db`, UTC day,
same file as the camera data). The last reading is also written to
`allsky-api/public/latest_sensor_data.json` (served as `/latest_sensor_data.json`);
the history of a day comes from `GET /api/sensors?date=YYYY-MM-DD`.
Besides the values the JSON file holds `configured` (`weather`, `anemometer`,
`wind_vane`, `cabinet`: `true` if the sensor has an address in `config.ini`);
`meteo.html` hides the tiles and chart series of the other sensors.

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

#### Wind: 10-minute mean

A single reading of the wind vane every 5 minutes is a random snapshot in
gusty or changing wind. As recommended by the WMO (CIMO guide, WMO-No. 8)
the tool therefore samples the anemometer and the wind vane together every
`modbus_wind_sample_interval` seconds (default `10`, i.e. 60 samples in
10 minutes; 1..60) and stores the mean over the last
`modbus_wind_average_minutes` (default `10`):

* `wind_speed`: scalar mean of the samples (m/s, two decimals).
* `wind_direction`: vector mean weighted by the wind speed of the same sample
  (the resultant wind), so 350° and 10° average to 0°, not 180°.

The samples live only in a ring buffer in RAM (one hour at most); nothing is
written per sample, the SD card sees one database row and one JSON file per
interval as before. After a (re)start the first row follows after at least two
minutes of samples, on the next multiple of `modbus_interval`. A wind sample
is requested only once (no retries, a missed sample costs little); failures
are counted in the journal line of the next row, a warning follows only when
a sensor stays silent for 60 s, and then a line when it answers again. The
weather sensor and the cabinet sensor are read once per row (instantaneous
values, with `modbus_retries`).

Calm: below `modbus_wind_calm_speed` (default `0.3` m/s, Beaufort 0; `0`
switches it off) the vane turns at random, so such samples do not count for
the direction. With less than 30 s of windy samples in the window (at least 3
samples) the last direction is kept (in RAM; after a restart the last stored one from today's
or yesterday's database, `NULL` if there is none). Without an anemometer
every vane sample has the same weight.

The journal line of every row shows the mean, the strongest sample, the
consistency of the direction (1 = steady, near 0 = variable) and how many
samples were windy:

```
2026-11-11 10:05:00 UTC: 4 of 4 sensors; wind 10 min: 2.31 m/s (max 4.2), 245 deg, consistency 0.82, 41 of 60 samples windy; failed reads: anemometer 0, vane 2
```

### Running automatically

`allsky360-sensors.service` runs `allsky360-sensors --daemon` continuously
(restarted after 30 s if the bus is not available). Like the camera it is
installed but not enabled by a fresh `make install`:

```bash
sudo systemctl enable --now allsky360-sensors.service
journalctl -u allsky360-sensors.service -f
```

Up to November 2026 the tool was started by `allsky360-sensors.timer` every
5 minutes. `make install` removes that timer and enables the service if the
timer was enabled.

The service account needs the groups `dialout` (serial port) and `gpio`
(direction GPIO on the Jetson); `make install` adds them.

### Manual usage

Without `--daemon` the tool reads every sensor once (instantaneous values,
calm rule against the last stored direction). The daemon holds the lock on the
serial port, so stop it first:

```bash
sudo systemctl stop allsky360-sensors.service
allsky360-sensors -c /home/allskyuser/config.ini -n -v   # read and print, no database
allsky360-sensors -c /home/allskyuser/config.ini -n -D -vv   # daemon, every wind sample, no database
sudo systemctl start allsky360-sensors.service
```

| Option | Meaning |
|---|---|
| `-c, --config FILE` | config.ini (required) |
| `-d, --databasedir DIR` | override `database_directory` |
| `-f, --outfile FILE` | also write the reading as JSON (atomic) |
| `-D, --daemon` | run continuously: wind every `modbus_wind_sample_interval` s into the RAM ring buffer, a stored row every `modbus_interval` s |
| `-n, --dry-run` | read and print, do not write the database |
| `-v, --verbose` | print the configuration and every failed attempt with the received bytes; twice: every wind sample (daemon) |

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

The register maps follow the manuals of the sensors listed above. Other
sensors with a different register layout need a change in
`tools/sensors/src/main.c`.

---

## Building the tools

You can build each tool individually from its directory:

```bash
cd tools/aurora && make
cd tools/meteo && make
cd tools/timelapse && make
cd tools/indigo-cameras && make
cd tools/keogram && make
cd tools/startrails && make
cd tools/indigo-mount-position && make
cd tools/sensors && make
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
