# Getting Started

This guide takes you from a freshly installed station to good images with
correct overlays. It assumes that INDIGO and Allsky360 are installed
([INSTALL.md](INSTALL.md)) and that the camera is connected.

Work through the steps in this order: the image geometry (step 5) depends on
the crop (step 4), and the overlays depend on the location (step 3).

Throughout this guide:

* **Config file:** `/home/allskyuser/config.ini` – edit it with
  `sudo -u allskyuser nano /home/allskyuser/config.ini`. Changes take effect
  with the next image. All keys: [CONFIGURATION.md](CONFIGURATION.md).
* **Web interface:** `http://<hostname>:3000/` – the pages are described in
  [WEB_INTERFACE.md](WEB_INTERFACE.md).
* **Camera log:** `journalctl -u allsky360-camera -f`

---

## 1. Check that everything runs

Open the page **System Status** in the web interface. It shows every service
and timer, the disk space and the health checks.

| Service | Must be | Purpose |
|---|---|---|
| `allsky360-camera` | running | Captures and processes the images |
| `allsky360-api` | running | Web interface |
| `allsky360-meteo.timer` | active | Weather data every 30 minutes |
| `allsky360-aurora.timer` | active | Aurora and space weather |
| `allsky360-mount-position.timer` | active | Mount position (only useful with a mount) |
| `allsky360-sensors` | running | RS485 sensors (only useful with sensors) |
| `indigo-server` | running | INDIGO server, if it runs on this computer |

If `install.sh` did not start the services, start them on the status page or
with:

```bash
sudo systemctl enable --now allsky360-camera.service allsky360-meteo.timer allsky360-aurora.timer
```

✅ **Check:** the camera service is running and its log shows lines like
`===== Capturing HDR frame 0 / 5 …` without errors.

If the camera cannot connect, see the troubleshooting section of
[INDIGO_SETTINGS.md](INDIGO_SETTINGS.md).

---

## 2. The first image

After one capture cycle (`capture_interval`, default 120 s, plus the
exposure time) the **Dashboard** shows the first image.

✅ **Check:** an image is visible, the time printed in the image is the local
time of the station.

If the image is completely black or white, wait two or three cycles: the
auto-exposure starts from `daytime_startexposure` / `nighttime_startexposure`
and needs a few images to settle.

---

## 3. Location and time zone

Check these keys (`install.sh --configure` asks for them):

```ini
location_latitude=51.2349N
location_longitude=7.4479E
location_altitude=114
location_timezone=Europe/Berlin
```

They decide the day/night switch, the sun and moon positions, the weather
data and the overlays. Use at least four decimal places.

✅ **Check:** the *Sun & Moon* panel of the dashboard shows the right sunrise
and sunset times for your place.

Details: [GENERAL_SETTINGS.md](GENERAL_SETTINGS.md)

---

## 4. Camera limits and crop

### Exposure and gain

```ini
camera_min_exposure=0.0007   # the real minimum of your camera
camera_max_exposure=60.0
camera_min_gain=0
camera_max_gain=200
camera_debayer=vng
```

Take the minimum exposure and the useful gain range from the data sheet of
your camera. Details: [CAMERA_SETTINGS.md](CAMERA_SETTINGS.md)

### Crop

A fisheye lens draws a circle on a rectangular sensor. Cropping the unused
sides saves memory and processing time:

```ini
crop_image=1
crop_width=2280
crop_height=2180
crop_x_offset=760
crop_y_offset=0
```

Choose a region that contains the whole image circle. To start, set
`crop_image=0`, look at the full image on the dashboard, and read off where
the circle begins and ends.

> **Important:** all positions in the following steps are pixels of the
> **saved (cropped) image**. If you change the crop later, calibrate again.

---

## 5. Calibrate the image geometry

The overlays, the keogram, the star trails, the focus and noise measurement
and the SQM need to know where the zenith, the horizon and north are in the
image. Open **Camera → Calibration** in the web interface. It shows the
latest image with adjustable overlays; zoom with the buttons or the mouse
wheel and drag the image to move it.

1. **Mask** (*Black Mask*): switch on *Show mask overlay* and set
   *Center X/Y* and *Mask radius* so the circle encloses the sky and the
   horizon. Everything outside is set to black in the image.
2. **Zenith and horizon** (*Azimuthal Grid*): switch on *Show azimuthal grid*.
   Set *Zenith X/Y* to the point straight overhead (for a level camera: the
   center of the image circle) and *Horizon radius* so the outermost ring
   lies on the horizon.
3. **North** (*North direction*): switch on *Show cardinal directions* and
   turn the grid until N points to north – use a known landmark, the sun's
   position at noon (south in the northern hemisphere), or Polaris at night.
4. **Check against the stars** (at night): switch on *Show equatorial grid*.
   The celestial pole of the grid must sit on Polaris (or the south celestial
   pole). Fine-tune zenith and north until it does.
5. **Copy the values:** the box *Config Output* shows the lines for
   `config.ini`. Copy `image_horizon_radius`, `image_zenith_x`,
   `image_zenith_y`, `image_north_angle`, `image_mask_radius`,
   `image_center_x` and `image_center_y`.
   *Do not copy the line `projection_type` – it is only a display setting of
   the page; the camera reports it as an unknown key.*

✅ **Check:** at night, switch on *Stars* on the dashboard. The labels of the
bright stars sit on the stars in the image, over the whole sky.

### Panorama and SQM area

* **Panorama:** set `panorama_center_x`/`panorama_center_y` to the zenith
  position and `panorama_horizontal_start` to the azimuth that should be at
  the left edge. `panorama_width`/`panorama_height` are the size of the
  panorama image.
* **SQM:** `sqm_center_x=0` and `sqm_center_y=0` measure around the zenith;
  `sqm_radius` should cover a dark area without trees or buildings.

---

## 6. Day and night look

Look at the images of a whole day and night, then adjust:

| Problem | Key |
|---|---|
| Night images too dark or too bright | `nighttime_meanbrightness` (0.10 – 0.25) |
| Day images too dark or too bright | `daytime_meanbrightness` |
| Color cast at night (orange/green) | `nighttime_white_balance_red/blue`, then `scnr_filter` |
| Color cast by day | `daytime_white_balance_red/blue` |
| Too much noise at night | `nighttime_multiscale_median_filter`, `acdnr_filter_*`, lower `camera_max_gain` |
| Stars look "crunchy" | lower `wavelet_sharpen_gain_small` |
| Moon or sun burnt out | `nighttime_/daytime_hdr_exposure_factor` |
| Switch to night too early or late | `nighttime_sunangle` |

The pages *Camera → Statistics* and *Camera → HDR* show exposure, gain,
brightness and noise over time and per image.

Details: [DAYTIME_NIGHTTIME_SETTINGS.md](DAYTIME_NIGHTTIME_SETTINGS.md),
[HDR-FUSION.md](HDR-FUSION.md), [POSTPROCESSING.md](POSTPROCESSING.md)

---

## 7. After the first night

When night turns into day, the camera starts `endofnight.sh`. It builds the
timelapse video, the keogram and the star trails of the past night. This
takes a few minutes.

✅ **Check:** the page **Night** shows the keogram and the star trails, the
page **Videos** the timelapse.

If the keogram is shifted or slanted, the zenith or north angle of step 5 is
not right yet.

### Calibrate the SQM

On a clear, moonless night compare the SQM value on the dashboard with a
reference (an SQM-L meter or the value of your place on
lightpollutionmap.info) and correct `sqm_intercept` by the difference.

---

## 8. Optional modules

| Module | What you need | Guide |
|---|---|---|
| RS485 weather sensors | sensors + RS485 adapter | [SENSORS.md](SENSORS.md) |
| Telescope mount overlay | a mount on the INDIGO server | [INDIGO_SETTINGS.md](INDIGO_SETTINGS.md#indigo_mount) |
| AI sky classification | – (runs on the Jetson GPU) | [INSTALL.md](INSTALL.md) |
| Aircraft (ADS-B) | ADS-B receiver with `dump1090-mutability` | [API.md](API.md) |
| Service buttons in the browser | an API token | [INSTALL.md](INSTALL.md) |

---

## 9. Running the station

* **Disk space:** the archive is cleaned up automatically by age and size
  (`retention_*`). Check the limits once against your disk; run with
  `retention_dry_run=1` first to see what would be deleted.
* **Updates:** `cd ~/allsky360 && git pull && sudo ./install.sh` – your
  `config.ini` is kept.
* **Logs:** `journalctl -u allsky360-camera -f` (camera),
  `journalctl -u allsky360-api -f` (web interface, also `make log`).
* **Times:** everything is stored in UTC. The day folders
  `images/YYYYMMDD` therefore change at midnight UTC, not at local midnight.
