# Allsky360 API Reference

The Allsky360 API is a lightweight Node.js HTTP server that exposes sky camera data,
meteorological records, aurora indices, ADS-B aircraft data, and systemd service
controls to local network clients.  The server is started by the
`allsky360-api.service` systemd unit and listens on **port 3000** by default
(configurable via `api-port` in `config.ini`).

All API responses use `Content-Type: application/json` unless stated otherwise.
Dates are always expressed in **ISO 8601** format (`YYYY-MM-DD`).

**Time:** every `timestamp` is Unix time (UTC). The `date` parameter of the
day-based endpoints (`/api/image`, `/api/meteo`, `/api/aurora`,
`/api/sensors`, `/api/thumbnails`) is the **UTC day**: the daily databases
`database_YYYYMMDD.db` and the folders `images/YYYYMMDD` are named after it.
A calendar day in local time spans two UTC days; the web pages fetch both
and filter by timestamp (`fetchDayRows()` in `public/js/allsky-common.js`).
For display, `/api/config` returns `timezone` (the IANA zone of the station,
from `location_timezone`); summer/winter time comes from the time zone
database for each moment, not from a fixed offset.

---

## Base URL

```
http://<host>:<port>
```

Default: `http://localhost:3000`

---

## Endpoints

### Image Data

#### `GET /api/image`

Returns per-image capture metadata for a given observation day, ordered by
timestamp ascending.  Each row corresponds to one captured and processed frame.

**Query parameters**

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| `date`    | string | yes      | UTC day in `YYYY-MM-DD` format |

**Response**

HTTP 200 — JSON array of image records:

| Field             | Type    | Description                                              |
|-------------------|---------|----------------------------------------------------------|
| `timestamp`       | integer | Unix timestamp (UTC) of image capture                    |
| `timezone_offset` | integer | UTC offset of the station time zone at capture, seconds (DST included; informational) |
| `exposure`        | number  | Exposure time in seconds                                 |
| `gain`            | integer | Camera sensor gain                                       |
| `brightness`      | number  | Mean image brightness (0.0 – 1.0)                        |
| `mean_r`          | number  | Mean red channel value (0.0 – 1.0)                       |
| `mean_g`          | number  | Mean green channel value (0.0 – 1.0)                     |
| `mean_b`          | number  | Mean blue channel value (0.0 – 1.0)                      |
| `noise`           | number  | Estimated noise level                                    |
| `hdr`             | integer | Number of HDR exposures merged (1 = no HDR)              |
| `night_mode`      | integer | 1 if nighttime settings were active, 0 otherwise         |
| `stars`           | integer | Number of detected stars                                 |
| `sqm`             | number  | Sky Quality Meter estimate (mag/arcsec²)                 |
| `focus`           | number  | Focus quality metric                                     |

**Error responses**

| Status | Condition                                |
|--------|------------------------------------------|
| 400    | `date` parameter missing or malformed    |
| 404    | No database file found for the given date |
| 500    | Internal database error                  |

**Example**

```
GET /api/image?date=2025-07-22
```

```json
[
  {
    "timestamp": 1753142400,
    "timezone_offset": 7200,
    "exposure": 30.0,
    "gain": 150,
    "brightness": 0.12,
    "mean_r": 0.11,
    "mean_g": 0.12,
    "mean_b": 0.14,
    "noise": 0.008,
    "hdr": 3,
    "night_mode": 1,
    "stars": 214,
    "sqm": 20.4,
    "focus": 0.87
  }
]
```

---

### Meteorological Data

#### `GET /api/sensors`

Returns the readings of the local RS485/Modbus sensors for a given day
(table `sensors`, written every 5 minutes by `allsky360-sensors.service`).
Values of sensors that are not configured or did not answer are `null`.

**Query parameters**

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| `date`    | string | yes      | Date in `YYYY-MM-DD` format (UTC day) |

**Response**

HTTP 200 — JSON array of records with `timestamp` (Unix, UTC),
`temperature`, `humidity`, `dew_point`, `pressure` (hPa), `ws_wind_speed`
(m/s), `ws_wind_direction` (°), `noise` (dB), `pm25`, `pm10` (µg/m³),
`illuminance` (lx), `rainfall` (mm), `wind_speed` (m/s, anemometer),
`wind_direction` (°, wind vane), `cabinet_temperature`, `cabinet_humidity`,
`cabinet_dew_point`. See TOOLS.md for the sources.

**Error responses**

| Status | Condition                                |
|--------|------------------------------------------|
| 400    | `date` parameter missing or malformed    |
| 404    | no database or no sensor data for that day |

The latest reading is also available as `latest_sensor_data.json`.

---

#### `GET /api/meteo`

Returns hourly weather forecast/analysis records for a given day from the
Open-Meteo data fetched by the `allsky360-meteo.service`.

**Query parameters**

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| `date`    | string | yes      | UTC day in `YYYY-MM-DD` format |

**Response**

HTTP 200 — JSON array of meteorological records:

| Field               | Type    | Description                                  |
|---------------------|---------|----------------------------------------------|
| `timestamp`         | integer | Unix timestamp (UTC)                         |
| `temperature`       | number  | 2 m air temperature (°C)                     |
| `humidity`          | number  | Relative humidity at 2 m (%)                 |
| `dew_point`         | number  | Dew point temperature at 2 m (°C)            |
| `pressure_msl`      | number  | Mean sea level pressure (hPa)                |
| `surface_pressure`  | number  | Surface pressure (hPa)                       |
| `cloud_cover`       | number  | Total cloud cover (%)                        |
| `cloud_low`         | number  | Low-level cloud cover (%)                    |
| `cloud_mid`         | number  | Mid-level cloud cover (%)                    |
| `cloud_high`        | number  | High-level cloud cover (%)                   |
| `visibility`        | number  | Visibility (m)                               |
| `wind_speed_10m`    | number  | Wind speed at 10 m height (km/h)             |
| `wind_dir_10m`      | number  | Wind direction at 10 m height (degrees)      |
| `wind_speed_300hPa` | number  | Wind speed at 300 hPa pressure level (km/h)  |

**Error responses**

| Status | Condition                                |
|--------|------------------------------------------|
| 400    | `date` parameter missing or malformed    |
| 404    | No database file found for the given date |
| 500    | Internal database error                  |

**Example**

```
GET /api/meteo?date=2025-07-22
```

---

### Aurora Data

#### `GET /api/aurora`

Returns space weather and aurora probability records for a given day, sourced
from NOAA/ESA data fetched by the `allsky360-aurora.service`.

**Query parameters**

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| `date`    | string | yes      | UTC day in `YYYY-MM-DD` format |

**Response**

HTTP 200 — JSON array of aurora records:

| Field                | Type    | Description                                        |
|----------------------|---------|----------------------------------------------------|
| `timestamp`          | integer | Unix timestamp (UTC)                               |
| `probability_percent`| number  | Current aurora visibility probability (%)          |
| `probability_max`    | number  | Maximum probability in the observation window (%)  |
| `probability_avg`    | number  | Average probability in the observation window (%)  |
| `kp_index`           | number  | Planetary K-index (0 – 9)                          |
| `bt`                 | number  | Interplanetary magnetic field total (nT)            |
| `bz`                 | number  | IMF Bz component (nT); negative values favour aurora|
| `density`            | number  | Solar wind proton density (p/cm³)                  |
| `speed`              | number  | Solar wind speed (km/s)                            |
| `temperature`        | number  | Solar wind proton temperature (K)                  |

**Error responses**

| Status | Condition                                |
|--------|------------------------------------------|
| 400    | `date` parameter missing or malformed    |
| 404    | No database file found for the given date |
| 500    | Internal database error                  |

**Example**

```
GET /api/aurora?date=2025-07-22
```

---

### Thumbnails and Video Listings

#### `GET /api/thumbnails`

Lists the available image thumbnails or timelapse video files for a given day.

**Query parameters**

| Parameter | Type   | Required | Allowed values     | Description               |
|-----------|--------|----------|--------------------|---------------------------|
| `date`    | string | yes      | `YYYY-MM-DD`       | Observation date          |
| `type`    | string | yes      | `image` or `video` | Resource type to list     |

**Response**

HTTP 200 — JSON array of strings.

- **`type=image`**: Returns a sorted list of timestamp stems derived from thumbnail
  filenames, e.g. `["20250722123045", "20250722123145", ...]`.  The full thumbnail
  URL is constructed as `/images/{YYYYMMDD}/thumbnails/image-{stem}.jpg`.
- **`type=video`**: Returns a sorted list of video filename stems (without the
  `.mp4` extension) for files ending in `_{YYYYMMDD}.mp4`,
  e.g. `["timelapse_20250722"]`.  The full video URL is `/videos/{stem}.mp4`.

**Error responses**

| Status | Condition                                    |
|--------|----------------------------------------------|
| 400    | `date` or `type` parameter missing/invalid   |
| 404    | Thumbnail or video directory does not exist  |

**Example**

```
GET /api/thumbnails?date=2025-07-22&type=image
```

```json
["20250722210015", "20250722210215", "20250722210415"]
```

---

### Configuration

#### `GET /api/config`

Returns the public runtime configuration of the camera station as a JSON object.
The data is read from `public/images/config.json`, which is written by the camera
daemon at startup and updated on configuration changes.

**Response**

HTTP 200 — JSON object with station metadata and active processing parameters.
The exact fields mirror the settings in `config.ini` that are relevant to the
web frontend (location, lens, image dimensions, overlay parameters, etc.).

**Error responses**

| Status | Condition                          |
|--------|------------------------------------|
| 500    | `config.json` could not be read    |

**Example**

```
GET /api/config
```

---

### Latest Image

#### `GET /api/latest_image`

Returns metadata about the most recently captured and saved image.  The data is
read from `public/images/latest_image.json`, which is updated after every
successful capture cycle.

**Response**

HTTP 200 — JSON object:

| Field      | Type   | Description                                          |
|------------|--------|------------------------------------------------------|
| `date`     | string | Date portion of the image filename (`YYYYMMDD`)      |
| `image`    | string | Time portion of the image filename (`HHmmss`)        |
| `filetype` | string | File extension without dot, e.g. `jpg`               |

The full image path can be assembled as:

```
/images/{date}/image-{date}{image}.{filetype}
```

The corresponding panorama (if enabled) is available at:

```
/images/{date}/panorama-{date}{image}.{filetype}
```

**Error responses**

| Status | Condition                              |
|--------|----------------------------------------|
| 500    | `latest_image.json` could not be read  |

**Example**

```
GET /api/latest_image
```

```json
{
  "date": "20250722",
  "image": "213045",
  "filetype": "jpg"
}
```

---

### Aircraft / ADS-B

#### `GET /api/aircraft`

Proxies the live ADS-B aircraft JSON produced by `dump1090-mutability` from the
local file `/run/dump1090-mutability/aircraft.json`.

**Response**

HTTP 200 — Raw `aircraft.json` content as produced by dump1090.  The structure
follows the dump1090 aircraft JSON format and includes fields such as `now`,
`messages`, and an `aircraft` array with ICAO hex codes, positions, altitudes,
speeds, and squawk codes.

**Error responses**

| Status | Condition                                          |
|--------|----------------------------------------------------|
| 503    | `aircraft.json` not available (dump1090 not running or no data yet) |

---

#### `GET /api/aircraft/status`

Checks whether `dump1090-mutability` is currently running on the host.

**Response**

HTTP 200 — JSON object:

| Field       | Type    | Description                                  |
|-------------|---------|----------------------------------------------|
| `running`   | boolean | `true` if at least one dump1090 process is active |
| `processes` | array   | List of matching process description strings |

**Example**

```json
{ "running": true, "processes": ["1234 dump1090-mutability --net"] }
```

---

### Service Management

The following endpoints allow starting, stopping, restarting, and querying the
status of allsky360 systemd services. The API runs `systemctl` directly
(no sudo, no shell) as `allskyuser`. Querying the status needs no rights;
start/stop/restart are authorised by the polkit rule
`/etc/polkit-1/rules.d/50-allsky360.rules` (installed by `make install`),
which allows exactly these three actions on the units listed below.

**Allowed services**

| Service name                    | Description                       |
|---------------------------------|-----------------------------------|
| `allsky360-camera.service`      | Main camera capture daemon        |
| `allsky360-api.service`         | Node.js web API server            |
| `allsky360-meteo.service`       | Meteorological data fetcher       |
| `allsky360-aurora.service`      | Aurora / space weather fetcher    |
| `allsky360-mount-position.service` | Mount position service (timer-driven) |

Requests for any service not in the list above are rejected with HTTP 500.

**Authentication for start/stop/restart**

`start`, `stop` and `restart` require:

- the HTTP method `POST` (otherwise HTTP 405),
- the header `X-API-Token` matching `api-token` in `config.ini` (otherwise HTTP 401).

If `api-token` is empty in `config.ini`, these endpoints are disabled and return HTTP 403.
The web UI (`config.html`) asks for the token on the first control action and keeps it
in the browser's `sessionStorage` until the tab is closed. `status` needs no token.

```
curl -X POST -H "X-API-Token: <token>" \
  "http://allsky:3000/api/services/restart?service=allsky360-camera.service"
```

---

#### `GET /api/services/status`

Queries the active state of a systemd service.

**Query parameters**

| Parameter | Type   | Required | Description       |
|-----------|--------|----------|-------------------|
| `service` | string | yes      | Systemd unit name |

**Response**

HTTP 200 — JSON object:

| Field     | Type   | Description                                          |
|-----------|--------|------------------------------------------------------|
| `service` | string | Name of the queried service                          |
| `status`  | string | Always `"success"` on HTTP 200                       |
| `output`  | string | Systemd active state string: `active`, `inactive`, `failed`, etc. |

**Error responses**

| Status | Condition                                     |
|--------|-----------------------------------------------|
| 400    | `service` parameter missing                   |
| 500    | `systemctl` command failed or service not allowed |

**Example**

```
GET /api/services/status?service=allsky360-camera.service
```

```json
{
  "service": "allsky360-camera.service",
  "status": "success",
  "output": "active"
}
```

---

#### `POST /api/services/start`

Starts a systemd service.

**Query parameters**

| Parameter | Type   | Required | Description       |
|-----------|--------|----------|-------------------|
| `service` | string | yes      | Systemd unit name |

**Response**

HTTP 200 — JSON object:

| Field     | Type   | Description                     |
|-----------|--------|---------------------------------|
| `service` | string | Name of the service             |
| `action`  | string | Always `"start"`                |
| `status`  | string | `"success"` or `"error"`        |
| `output`  | string | stdout from systemctl           |

**Error responses**: see [Service Management](#service-management).

---

#### `POST /api/services/stop`

Stops a systemd service.

**Query parameters**

| Parameter | Type   | Required | Description       |
|-----------|--------|----------|-------------------|
| `service` | string | yes      | Systemd unit name |

**Response**

HTTP 200 — JSON object with fields `service`, `action` (`"stop"`), `status`, `output`.

---

#### `POST /api/services/restart`

Restarts a systemd service.

**Query parameters**

| Parameter | Type   | Required | Description       |
|-----------|--------|----------|-------------------|
| `service` | string | yes      | Systemd unit name |

**Response**

HTTP 200 — JSON object with fields `service`, `action` (`"restart"`), `status`, `output`.

---

## Static File Serving

The server exposes the `public/` directory for direct file access.  The following
file types are permitted: `.html`, `.js`, `.css`, `.ico`, `.png`, `.jpg`,
`.jpeg`, `.json`, `.mp4`.  All other extensions and all `.db` files are rejected
with HTTP 403.

### Common static paths

| Path pattern                                           | Description                        |
|--------------------------------------------------------|------------------------------------|
| `/` or `/dashboard.html`                               | Web dashboard (landing page)       |
| `/images/{YYYYMMDD}/image-{datetime}.jpg`              | Full-resolution allsky image       |
| `/images/{YYYYMMDD}/panorama-{datetime}.jpg`           | Horizon panorama image             |
| `/images/{YYYYMMDD}/thumbnails/image-{datetime}.jpg`   | Reduced-size thumbnail             |
| `/videos/{name}_{YYYYMMDD}.mp4`                        | Timelapse video                    |

---

## Error Format

All API error responses share the same JSON envelope:

```json
{ "error": "<human-readable message>" }
```

On service management errors an additional `stderr` field may be present with
the raw output from `systemctl`.

---

## Command-Line Client

A Bash helper script (`scripts/allsky360_api.sh`) wraps common API calls with
`curl` and `jq`.

```
Usage:
  allsky360_api.sh --api-url <URL> --target <target> [--mode json|download]
                   [--output <file>] [--date YYYY-MM-DD]

Targets:
  latest_image    Metadata of the most recent image (json or download)
  panorama        Download the current panorama image
  config          Station configuration JSON
  image           Per-image capture records for a date (requires --date)
  meteo           Meteorological records for a date (requires --date)
  aurora          Aurora/space weather records for a date (requires --date)
  aircraft        Live ADS-B aircraft data
  aircraft_status dump1090 process status
```

**Examples**

```bash
# Print latest image metadata as JSON
./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 \
    --target latest_image --mode json

# Download the current allsky image
./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 \
    --target latest_image --mode download --output latest.jpg

# Download the current panorama
./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 \
    --target panorama --mode download --output panorama.jpg

# Retrieve capture records for a specific night
./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 \
    --target image --date 2025-07-22

# Retrieve meteorological data for a specific day
./scripts/allsky360_api.sh --api-url http://192.168.3.62:3000 \
    --target meteo --date 2025-07-22
```

---

## Configuration Reference

The API server reads its configuration from `config.ini` (default path:
`/home/allskyuser/config.ini`; override with `--config <path>` on startup).

| Key        | Default | Description                        |
|------------|---------|------------------------------------|
| `api-port` | `3000`  | TCP port the HTTP server binds to  |

The `PORT` environment variable takes precedence over `api-port` in the
configuration file.
