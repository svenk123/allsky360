# Web Interface

The web interface runs on the station itself (service `allsky360-api`) and is
reached with any browser in the network:

```
http://<hostname>:3000/
```

The address opens the **Dashboard**. The port is set with `api-port`, the
listen address with `api-bind` (see [CONFIGURATION.md](CONFIGURATION.md)).
All pages only read data; the only actions are the service buttons on the
*System Status* page, which need the API token.

<!-- TODO: screenshots of the pages in docs/images/ -->

---

## Navigation and times

The navigation bar at the top leads to every page. The *Camera* entry opens
a second menu with *Statistics*, *HDR*, *Pipeline* and *Calibration*.

All data is stored in UTC. The pages show times either in the **station time
zone** (`location_timezone`) or in **UTC**; switch with the toggle in the
header. The choice is remembered in the browser. A day selected in a date
field is the local calendar day, even when it spans two UTC days.

---

## Pages

| Page | Menu | Shows |
|---|---|---|
| `index.html` | Home | Live image with SQM, weather and aurora summary |
| `dashboard.html` | Dashboard | Latest image with overlays and all current data |
| `archive.html` | Archive | All images of a day as thumbnails, with their data |
| `videos.html` | Videos | Timelapse videos of a day |
| `night.html` | Night | Keogram and star trails of every night |
| `meteo.html` | Weather | Station sensors, Open-Meteo values and forecast |
| `aurora.html` | Aurora | Aurora probability, Kp index, solar wind |
| `aircraft.html` | Aircraft | Aircraft received by an ADS-B receiver |
| `graphs.html` | Camera → Statistics | Exposure, gain, brightness, noise, SQM over a day |
| `hdr.html` | Camera → HDR | HDR exposures and fusion of the latest image |
| `processing.html` | Camera → Pipeline | Debug images of every processing stage |
| `calibration.html` | Camera → Calibration | Tool to calibrate the image geometry |
| `status.html` | System Status | Services, timers, disk, health checks |
| `config.html` | Config | Current configuration with help texts |
| `panorama.html` | – | Horizon panorama viewer |
| `lcd_1280x400.html` | – | Full-screen view for a 1280×400 display |

### Home

A compact live view: the latest image, the SQM value, the current weather and
the aurora values (Kp, Bz). Good as a start page on a tablet or phone.

### Dashboard

The main page. It shows the latest image and, next to it, panels with the
image data (exposure, gain, HDR frames, stars), sun and moon (altitude,
rise and set), the coming night, weather, aurora and – with a mount – the
mount position. The image updates automatically after every capture.

Overlays (checkboxes above the image):

| Checkbox | Overlay |
|---|---|
| Az Grid | Altitude/azimuth grid with cardinal directions |
| RA/DEC | Equatorial grid |
| Stars | Names of the brightest stars |
| Meteors | Radiants of the active meteor showers |
| Sun/Moon | Position of sun and moon |
| Detected stars | The stars found by the SQM detector |
| Mount | Where the telescope mount points (needs `indigo_mount`) |

The overlays are computed from the location and the image geometry
(`image_zenith_*`, `image_horizon_radius`, `image_north_angle`). If they do
not match the sky, calibrate on the *Calibration* page (see
[GETTING_STARTED.md](GETTING_STARTED.md#5-calibrate-the-image-geometry)).
Zoom with the buttons or the mouse wheel; the overlays stay in place.

### Archive

Choose a date to see all images of that day as thumbnails, newest or oldest
first. Hovering over a thumbnail shows its data (time, exposures of all HDR
frames, gain, brightness, stars, moon phase, day or night mode) and the
environment values; a click opens the full image.

### Videos

The timelapse videos of a day, created by `endofnight.sh` after every night.
Click a thumbnail to play the video and see its data.

### Night

One entry per night with the **keogram** (the north–south line of every
image side by side over time; top = north) and the **star trails** (all night
images stacked). Clicking in the keogram shows the image taken at that time.
Both are created by `endofnight.sh` when the night ends; see
[TOOLS.md](TOOLS.md) for the tools behind them.

### Weather

* **Station sensors** (only with [RS485 sensors](SENSORS.md)): tiles with the
  latest reading, marked *stale* when older than 15 minutes, and a
  condensation warning for the control cabinet. Tiles of sensors that are not
  configured are hidden.
* **Open-Meteo**: temperature, dew point, humidity, pressure, wind, cloud
  cover per layer (low, mid, high), visibility, jet stream and a 24 h
  forecast. Station values are drawn next to the forecast.
* **Wind**: speed, direction chart and wind rose.

The charts can be moved and resized.

### Aurora

Aurora probability for the station, Kp index and the real-time solar wind
(Bz, Bt, speed, density, temperature) from NOAA, for a selectable day.

### Aircraft

Aircraft received by a local ADS-B receiver (`dump1090-mutability`), with
statistics of the receiver. Without a receiver the page stays empty.

### Camera → Statistics

Charts of one day: exposure, gain, brightness per color channel, noise,
focus, number of HDR frames, stars and SQM, plus the histogram and the HDR
exposures of the latest image. Use it to see how the auto-exposure behaves
over a day and night.

### Camera → HDR

The single exposures of the latest image, the HDR settings, the measured
background noise and – with `hdr_weight_maps=1` – the weight maps of the
fusion. See [HDR-FUSION.md](HDR-FUSION.md).

### Camera → Pipeline

The image after every processing stage (autostretch, noise reduction,
sharpening, clarity, dehaze, gamma, saturation) with the keys of that stage.
The images are only created on request: send the camera the signal `SIGUSR2`
(`sudo systemctl kill -s USR2 allsky360-camera`); the next image then saves
every stage. See [POSTPROCESSING.md](POSTPROCESSING.md).

### Camera → Calibration

Overlays for the mask, the azimuthal grid and the equatorial grid on the
latest image. Adjust the values until the grids match the sky, then copy the
lines from *Config Output* into `config.ini` (without `projection_type`).
Step by step: [GETTING_STARTED.md](GETTING_STARTED.md#5-calibrate-the-image-geometry).

### System Status

* **Services & timers**: state of every Allsky360 service and timer, with
  *Start*, *Stop*, *Restart* and, for timers, *Run now*. These buttons need
  the API token (`api-token` in `config.ini`); the browser asks for it on the
  first click and keeps it until the tab is closed. Without a token in
  `config.ini` the API refuses these actions.
* **Health checks**: are the latest image, the weather and aurora data and
  the station sensors up to date.
* **Storage**: disk usage and the range of daily databases.
* **Hardware** and **Control cabinet**: host, uptime, memory and – with the
  cabinet sensor – temperature, humidity and the distance to the dew point.

A camera started by hand (`make camera-run`) is shown as *Running outside
systemd*.

### Config

All keys of the running configuration (`config.json`, written by the camera),
grouped by topic and with a help text. The page is read-only: change the
values in `/home/allskyuser/config.ini`. The INDIGO server address, the owner
and the API token are never shown.

### Panorama viewer

`panorama.html` shows the horizon panorama (`panorama=1`) as a scrollable
360° strip with the image data, location and weather. It has no menu entry;
open it directly.

### LCD display

`lcd_1280x400.html` is a full-screen layout for a small 1280×400 display
(e.g. in the observatory). It refreshes with the capture interval.

---

## Access from the internet

The web interface has no login. Leave `api-bind=0.0.0.0` only in a trusted
network. To make the station reachable from the internet, set
`api-bind=127.0.0.1` and put an HTTPS reverse proxy with authentication in
front of it; templates for Apache are in `etc/apache2/`. With
`publish_exact_location=0` the coordinates are rounded to about 1 km.

For your own pages and scripts use the REST API: [API.md](API.md).
