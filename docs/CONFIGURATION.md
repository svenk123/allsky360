# Configuration Reference

All settings of Allsky360 are in one file:

```
/home/allskyuser/config.ini
```

`make install` creates it from [`config.ini.default`](../config.ini.default) and
never overwrites it later. `sudo ./install.sh --configure` sets the most
important keys (location, INDIGO, camera, mount, sensors, API token)
interactively.

This page lists **every key** with its value in `config.ini.default` and a
short description. The linked pages explain the keys of a topic in detail.

---

## How the file works

* One setting per line: `key=value`. Lines starting with `#` are comments.
* Text with spaces goes in double quotes: `location_city="My Observatory"`.
* A key that is missing uses a built-in default. The built-in defaults do not
  always match `config.ini.default`; keep the keys you rely on in the file.
* The camera reads the file again **at the start of every capture cycle**:
  changes take effect with the next image, no restart needed.
  `sudo systemctl reload allsky360-camera` does the same immediately.
  Exceptions: `image_directory`, `database_directory`, `indigocamera`,
  `indigoserver`, `indigoservername` and `indigoport` are only read at
  startup. A change is reported in the log and applied with
  `sudo systemctl restart allsky360-camera`.
* The tools (meteo, aurora, sensors, keogram, …) read it at every start; the
  sensors daemon and the web API need a restart after changes:
  `sudo systemctl restart allsky360-sensors allsky360-api`.
* Unknown keys are reported once in the log (`unknown configuration key`), so
  typos do not go unnoticed. Keys that were removed are reported with a note.
* Edit the file as the service user, so it keeps its owner:
  `sudo -u allskyuser nano /home/allskyuser/config.ini`.

The *Config* page of the web interface shows the current values with help
texts (read-only).

---

## Location and time

Details: [GENERAL_SETTINGS.md](GENERAL_SETTINGS.md)

| Key | Default | Description |
|---|---|---|
| `location_latitude` | `51.2349N` | Latitude, decimal degrees, `N`/`S` suffix or signed |
| `location_longitude` | `7.4479E` | Longitude, decimal degrees, `E`/`W` suffix or signed |
| `location_altitude` | `114` | Altitude in meters; passed to Open-Meteo |
| `location_city` | `"Musterstadt/Germany"` | Name of the place (informational) |
| `location_owner` | `"Max Mustermann"` | Operator (informational, never published) |
| `location_timezone` | `Europe/Berlin` | Station time zone for display (IANA name); empty = system zone. Stored data is always UTC. |
| `publish_exact_location` | `1` | `0` = publish the coordinates rounded to ~1 km in `/api/config` |

## Capture and storage

Details: [GENERAL_SETTINGS.md](GENERAL_SETTINGS.md)

| Key | Default | Description |
|---|---|---|
| `capture_interval` | `120` | Seconds between two images |
| `image_directory` | `/home/allskyuser/images` | Images, one folder per UTC day (`YYYYMMDD`) |
| `video_directory` | `/home/allskyuser/videos` | Timelapse videos, keograms, star trails |
| `database_directory` | `/home/allskyuser/database` | SQLite databases, one per UTC day |
| `raw_dump_directory` | `/home/allskyuser/raw` | INDIGO raw frames saved on request (`SIGUSR1`) |

### Hook scripts

| Key | Default | Description |
|---|---|---|
| `postprocessing_script` | `/opt/allsky360/scripts/postprocessing.sh` | Runs after every image with the JPEG path as `$1` and `ALLSKY360_IMAGE_PATH`, `ALLSKY360_IMAGE_DIRECTORY`, `ALLSKY360_TODAY_STR`, `ALLSKY360_NOW_STR`, `ALLSKY360_IS_NIGHT` in the environment. The shipped script runs the AI sky classification. Empty = off. |
| `endofnight_script` | `/opt/allsky360/scripts/endofnight.sh` | Runs once when night turns into day, with `ALLSKY360_IMAGE_DIRECTORY`, `ALLSKY360_DATABASE_DIRECTORY`, `ALLSKY360_VIDEO_DIRECTORY`. The shipped script builds the timelapse, keogram and star trails of the night. Empty = off. |

Both scripts run detached, as `allskyuser`, and must be executable.

### Archive retention

| Key | Default | Description |
|---|---|---|
| `retention` | `1` | `0` = never delete anything |
| `retention_max_days` | `365` | Delete days older than this |
| `retention_max_size_gb` | `500` | Delete the oldest days while the archive (images + videos + databases) is larger than this (GiB). Leave headroom for the system. |
| `retention_keep_days` | `2` | The newest days that are never deleted (minimum 2) |
| `retention_dry_run` | `0` | `1` = only log what would be deleted |

A day is always deleted as a whole: image folder, videos, raw dumps and the
database of that day. Whichever limit is reached first wins. At
`capture_interval=120` a day takes several GB.

## Web interface and API

Details: [INSTALL.md](INSTALL.md) (*Set the API token*), [API.md](API.md)

| Key | Default | Description |
|---|---|---|
| `api-port` | `3000` | TCP port of the web server |
| `api-bind` | `0.0.0.0` | Listen address; `127.0.0.1` = local only (behind a reverse proxy) |
| `api-token` | empty | Secret for Start/Stop/Restart on the *System Status* page (`openssl rand -hex 32`). Empty = service control disabled. |

Restart the API after a change: `sudo systemctl restart allsky360-api`.

## INDIGO server

Details: [INDIGO_SETTINGS.md](INDIGO_SETTINGS.md)

| Key | Default | Description |
|---|---|---|
| `indigo_server` | `192.168.3.61` | Address of the INDIGO server (`127.0.0.1` = same computer) |
| `indigo_port` | `7624` | Port of the INDIGO server |
| `indigo_servername` | `indigo_server` | Connection name; device names end with ` @ ` plus this name |
| `indigo_camera` | `"Uranus-C @ indigo_server"` | Camera device (`indigo_cameras -l`) |
| `indigo_mount` | empty | Mount device, read-only (`indigo_cameras -M`); empty = no mount |
| `indigo_capture_timeout` | `30` | Seconds to wait for an image in addition to the exposure time |

## RS485 / Modbus sensors (optional)

Details: [SENSORS.md](SENSORS.md). All `modbus_*` keys are read by
`allsky360-sensors`; restart it after a change.

| Key | Default | Description |
|---|---|---|
| `modbus_device` | `/dev/ttyTHS1` | Serial port (Jetson UART1; USB adapter: `/dev/ttyUSB0`) |
| `modbus_baudrate` | `4800` | Baud rate of the bus (all sensors must use the same) |
| `modbus_gpio_chip` | `/dev/gpiochip0` | GPIO chip of the direction line |
| `modbus_gpio_line` | `112` | GPIO line that switches the transceiver direction (DE/RE); `-1` = automatic direction (USB adapters) |
| `modbus_timeout_ms` | `1200` | Wait for an answer |
| `modbus_retries` | `3` | Attempts per request |
| `modbus_function` | `3` | `3` = holding registers, `4` = input registers |
| `modbus_interval` | `300` | Seconds between two stored readings |
| `modbus_weather_address` | `0` | Address of the weather sensor; `0` = not connected |
| `modbus_weather_register` | `0x01F4` | First register of its data block |
| `modbus_weather_wind_direction_offset` | `0` | Mounting correction of its wind direction (degrees) |
| `modbus_wind_speed_address` | `0` | Address of the anemometer; `0` = not connected |
| `modbus_wind_speed_register` | `1` | Register of the wind speed |
| `modbus_wind_speed_scale` | `1.0` | Wind speed in m/s = raw value / scale |
| `modbus_wind_direction_address` | `0` | Address of the wind vane; `0` = not connected |
| `modbus_wind_direction_register` | `1` | Register of the direction |
| `modbus_wind_direction_offset` | `0` | Mounting correction (degrees, added modulo 360) |
| `modbus_wind_sample_interval` | `10` | Seconds between two wind samples (1 – 60) |
| `modbus_wind_average_minutes` | `10` | Averaging window of the wind (WMO: 10 minutes) |
| `modbus_wind_calm_speed` | `0.3` | Below this speed (m/s) samples do not count for the direction; `0` = off |
| `modbus_cabinet_address` | `0` | Address of the cabinet temperature/humidity sensor; `0` = not connected |
| `modbus_cabinet_register` | `0` | Register of the humidity (temperature at register + 1) |

## Camera

Details: [CAMERA_SETTINGS.md](CAMERA_SETTINGS.md)

| Key | Default | Description |
|---|---|---|
| `camera_max_exposure` | `60.0` | Longest exposure of the auto-exposure (s) |
| `camera_min_exposure` | `0.0007` | Shortest exposure the camera can do (s); also ends the HDR series |
| `camera_max_gain` | `200` | Highest gain of the auto-exposure |
| `camera_min_gain` | `0` | Lowest gain |
| `camera_binning2x2` | `0` | `1` = 2×2 binning in the camera |
| `camera_debayer` | `vng` | `nni`, `bilinear` or `vng` |
| `camera_monochrome` | `0` *(not in the file)* | `1` = monochrome camera: no debayering, white balance or saturation |

## Image geometry

Set these on the page *Camera → Calibration* (see
[GETTING_STARTED.md](GETTING_STARTED.md#5-calibrate-the-image-geometry)).
All positions are **pixels of the saved image**, i.e. after cropping.

| Key | Default | Description |
|---|---|---|
| `image_zenith_x`, `image_zenith_y` | `1925`, `1185` | Position of the zenith |
| `image_horizon_radius` | `1090` | Radius of the horizon circle |
| `image_north_angle` | `-11.0` | Rotation of north, degrees clockwise from up |
| `image_center_x`, `image_center_y` | `1925`, `1185` | Center of the black mask |
| `image_mask_radius` | `1090` | Everything outside this circle is set to black; `0` = no mask |
| `measure_brightness_area` | `0.55` | Auto-exposure measures inside this fraction of `image_horizon_radius` |
| `focus_radius` | `200` | Radius of the focus measurement around the zenith |
| `crop_image` | `1` | `1` = cut the region below out of the sensor image |
| `crop_width`, `crop_height` | `2280`, `2180` | Size of the region |
| `crop_x_offset`, `crop_y_offset` | `760`, `0` | Top left corner of the region on the sensor |

The keogram, the star trails, the dashboard overlays and the SQM use the
zenith, horizon radius and north angle.

## Image output

| Key | Default | Description |
|---|---|---|
| `jpeg_quality` | `100` | JPEG quality 0 – 100 |
| `thumbnail` | `1` *(not in the file)* | Save a thumbnail of every image |
| `thumbnail_width` | `100` *(not in the file)* | Width of the thumbnail in pixels |
| `panorama` | `1` | Create the horizon panorama (`latest_panorama.jpg`) |
| `panorama_center_x`, `panorama_center_y` | `1144`, `1180` | Center of the fisheye image for the unrolling |
| `panorama_width`, `panorama_height` | `3048`, `1090` | Size of the panorama |
| `panorama_horizontal_start` | `260` | Azimuth (degrees) at the left edge of the panorama |
| `font_path` | DejaVuSans *(not in the file)* | TrueType font for the text in the image |
| `font_size` | `20` *(not in the file)* | Font size of that text |

## Day and night

Details: [DAYTIME_NIGHTTIME_SETTINGS.md](DAYTIME_NIGHTTIME_SETTINGS.md)

| Key | Default | Description |
|---|---|---|
| `nighttime_sunangle` | `-6` | Sun altitude (degrees) below which the night settings apply |
| `daytime_capture` | `1` | `0` = no images by day |
| `daytime_startexposure` / `nighttime_startexposure` | `0.0045` / `10.0` | Exposure (s) when the mode starts |
| `daytime_startgain` / `nighttime_startgain` | `0` / `0` | Gain when the mode starts |
| `daytime_meanbrightness` / `nighttime_meanbrightness` | `0.20` / `0.15` | Target brightness of the auto-exposure (0 – 1) |
| `daytime_white_balance_red` / `nighttime_white_balance_red` | `1.0` / `1.0` | Red factor (green = 1.0) |
| `daytime_white_balance_blue` / `nighttime_white_balance_blue` | `1.4` / `2.0` | Blue factor |
| `daytime_white_balance_light_protect` / `nighttime_…` | `0.0` *(not in the file)* | Protects bright areas from the white balance; only without HDR |
| `daytime_move_blackpoint` / `nighttime_move_blackpoint` | `0.01` / `0.001` | Black point shift; `0` = off |
| `*_move_blackpoint_min_shift_pct` | `0.005` | Smallest shift |
| `*_move_blackpoint_max_shift_pct` | `0.02` | Largest shift |
| `*_move_blackpoint_dark_threshold` | `0.2` | Brightness below which the shift is gentler |
| `daytime_hdr_exposure_factor` / `nighttime_hdr_exposure_factor` | `2.0` / `2.0` | EV steps between two HDR frames |
| `nighttime_measure_noise_radius` | `250` | Radius of the night noise measurement around the zenith |
| `nighttime_multiscale_median_filter` | `1` | Median filter against noise at night |
| `nighttime_multiscale_median_filter_max_scale` | `2` | Size of the removed structures (1 – 3) |
| `nighttime_multiscale_median_filter_amount` | `0.9` | Blend with the original (0 – 1) |
| `scnr_filter` | `1` | Green reduction at night |
| `scnr_filter_amount` | `0.5` | Strength (0 – 1) |
| `scnr_filter_protection` | `average_neutral` | `none`, `average_neutral`, `maximum_neutral` |

## HDR

Details: [HDR-FUSION.md](HDR-FUSION.md)

| Key | Default | Description |
|---|---|---|
| `hdr` | `1` | Up to 5 exposures per image |
| `hdr_clipping_threshold` | `0.99998` | Brightness from which a pixel counts as clipped |
| `hdr_min_clipped_pixels` | `0.00001` | Fraction of clipped pixels that makes a frame over-exposed |
| `hdr_y_max_expected` | `1.0` | Brightness scale of the fusion weights; `0` = automatic |
| `hdr_weight_sigma` | `0.8` | Width of the "well exposed" weight curve |
| `hdr_weight_clip_factor` | `0.90` | Suppression of near-clipped pixels; `0` = automatic |
| `hdr_highlight_knee` | `0.8` | Start of the highlight shoulder after the fusion |
| `hdr_contrast_weight_strength` | `0.6` | Influence of local contrast on the weights |
| `hdr_noise_fusion_strength` | `1.0` | Down-weighting of noisy short exposures at night |
| `hdr_pyramid_levels_override` | `0` | Pyramid levels 1 – 6; `0` = automatic |
| `hdrmt` | `0` | Experimental HDR multiscale transform |
| `hdr_weight_maps` | `0` | Save the weight maps (debugging) |
| `hdr_weight_stats` | `1` | Log weight statistics |
| `hdr_save_exposures` | `1` | Save a preview of every exposure for the HDR page |

## Image processing

Details: [POSTPROCESSING.md](POSTPROCESSING.md)

| Key | Default | Description |
|---|---|---|
| `autostretch` | `1` | Histogram stretch |
| `autostretch_min_val`, `autostretch_max_val` | `0.01`, `0.99` | Percentiles that become black and white |
| `acdnr_filter` | `1` | Adaptive noise reduction |
| `acdnr_filter_lum_stddev`, `_amount`, `_iterations`, `_kernel_size` | `1.5`, `0.15`, `2`, `3` | Luminance part |
| `acdnr_filter_chrom_stddev`, `_amount`, `_iterations`, `_kernel_size` | `2.0`, `0.4`, `2`, `5` | Color part |
| `wavelet_sharpen` | `1` | Multi-scale sharpening |
| `wavelet_sharpen_gain_small`, `_medium`, `_large` | `2.0`, `0.0`, `0.0` | Gain per detail size; `0` = unchanged |
| `clarity_filter` | `0` | Local contrast |
| `clarity_filter_strength` | `0.35` | Strength |
| `clarity_filter_radius` | `24` | Size of the structures (pixels) |
| `clarity_filter_midtone_width` | `0.25` | Range around the mid-tones |
| `clarity_filter_preserve_highlights` | `1` | Protect bright areas |
| `clarity_filter_mask_mode` | `1` | Use a mid-tone mask |
| `dehaze_amount` | `0.0` | Haze removal; `0` = off |
| `dehaze_estimate` | `0.01` | Fraction of the darkest pixels used to estimate the haze |
| `gamma` | `0.9` | `output = input ^ gamma`; < 1 brightens |
| `auto_gamma` | `0` | `1` = gamma from the mean brightness |
| `auto_gamma_target`, `auto_gamma_min`, `auto_gamma_max` | `0.18`, `0.15`, `1.2` | Target and limits of the automatic gamma |
| `reinhard_tonemap` | `0` | Reinhard tone mapping after gamma |
| `reinhard_white_point` | `0.0` | `0` = automatic |
| `tone_curve` | `0` | S-curve on the luminance |
| `tone_curve_shadow_lift`, `tone_curve_midtone_contrast` | `0.20`, `0.30` | Shape of the curve (0 – 0.5) |
| `saturation` | `2.0` | Color saturation; `1.0` = unchanged |

## Sky quality (SQM)

The SQM is estimated from the number of stars detected on the raw first
exposure. It is only computed when that exposure is longer than 1 s and the
image is dark. The star list of every image is saved as
`images/YYYYMMDD/stars-*.json`; the dashboard shows the stars with the
checkbox *Detected stars*.

| Key | Default | Description |
|---|---|---|
| `sqm` | `1` | Estimate the SQM |
| `sqm_center_x`, `sqm_center_y` | `989`, `1238` | Center of the measured circle; `0` = the zenith |
| `sqm_radius` | `100` | Radius of the measured circle (pixels); `0` = no star detection |
| `sqm_max_stars` | `1000` | Maximum number of stars |
| `sqm_star_sigma` | `1.0` | Expected star size (Gaussian sigma = FWHM / 2.355) |
| `sqm_detect_sigma` | `5.0` | Detection threshold in noise sigma; lower = fainter stars and more false detections |
| `sqm_max_fwhm` | `8.0` | Wider objects are not stars (lights, lit clouds) |
| `sqm_max_elongation` | `0.5` | More elongated objects are not stars (0 = round, 1 = line) |
| `sqm_intercept` | `21.2` | SQM of the darkest sky; calibrate on a clear moonless night against an SQM meter or lightpollutionmap.info |
| `sqm_slope` | `1.5` | Sensitivity of the estimate to the star count |

After changing the detection keys, calibrate `sqm_intercept` again.

## Experimental and debugging

| Key | Default | Description |
|---|---|---|
| `spektrum_enabled` | `0` | Rough spectrum from the RGB values (experimental, uses estimated IMX585 response curves for every sensor, written as `qe_model` into the output file; color cameras only, skipped with `camera_monochrome=1`) |
| `spektrum_cx`, `spektrum_cy` | `-1`, `-1` | Measuring center; `-1` = image center |
| `spektrum_radius` | `50` | Measuring radius (1 – 100 pixels) |
| `spektrum_regularization` | `0.01` | Smoothing of the spectrum |
| `spektrum_output_path` | `images/spektrum.json` | Output file |
| `debug_raw_exposures` | `0` *(not in the file)* | Save every single exposure as JPEG into the day folder |

Signals for the running camera (`sudo systemctl kill -s <signal> allsky360-camera`):

| Signal | Effect |
|---|---|
| `HUP` | Re-read the configuration (same as `systemctl reload`) |
| `USR1` | Save the INDIGO raw frames of the next image into `raw_dump_directory` |
| `USR2` | Save the debug images of every processing stage for the next image |

## Removed keys

These keys never had an effect and were removed. An older `config.ini` may
still contain them: the camera prints a `NOTE` at startup, and you can delete
the lines.

| Key | Note |
|---|---|
| `camera_cooling`, `camera_cooling_temperature` | Sensor cooling is not set by Allsky360; set it in the INDIGO control panel |
| `rotate_image` | Image rotation was never implemented; the north direction is `image_north_angle` |
| `nighttime_measure_noise` | The night noise is always measured |
| `lens_name`, `lens_focallength`, `lens_focalratio`, `camera_r_factor`, `camera_g_factor`, `camera_b_factor`, `black_clipping_point`, `white_clipping_point`, `target_mean`, `nighttime_fullmoongain`, `working_directory` | Never used |

`camera_roi_*` is commented out in `config.ini.default`: ROI is not
implemented, the keys have no effect.
