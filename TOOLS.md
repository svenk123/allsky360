# Allsky360 Tools

So, I've built a few command-line tools that work alongside the main `allsky360-camera` application. They're in the `tools/` directory and handle various tasks like fetching weather data, listing cameras, calculating location data, and creating timelapse videos. Each one is built separately and can be installed to `/opt/allsky360/bin/` or just run from wherever you built them.

---

## Overview

Here's what's in the `tools/` directory:

* **allsky360-aurora** – Grabs aurora forecast and space weather data from NOAA/ESA
* **indigo-list-cameras** – Lists all the INDIGO cameras you've got connected
* **location** – Calculates location-specific astronomical data (magnetic declination, etc.)
* **allsky360-meteo** – Fetches weather data from Open-Meteo
* **timelapse-cuda-nv** – Creates timelapse videos from image sequences, with GPU acceleration

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
| `--kp` | | Include planetary Kp index data |
| `--mag` | | Include solar wind magnetic field data |
| `--plasma` | | Include solar wind plasma data |
| `--debug` | `-v` | Enable debug output (useful when things go wrong) |
| `--help` | `-h` | Show the help message |

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

## 2. indigo-list-cameras

**Location:** `tools/indigo-list-cameras/`  
**Binary:** `bin/indigo-list-cameras`

### What it does

This is a simple utility that connects to your INDIGO server and lists all the CCD cameras it can find. I use this all the time when I'm setting up a new camera or troubleshooting connection issues. It's really handy to verify that INDIGO can see your camera before you try to configure the main application.

### Usage

```bash
./bin/indigo-list-cameras [OPTIONS]
```

It connects to the local INDIGO server by default (127.0.0.1:7624). If you need to connect to a remote server or change other options, check the source code or run it with `--help`.

### Example

Just run it:
```bash
./bin/indigo-list-cameras
```

It'll print out all the cameras it finds.

---

## 3. location (indigo-location)

**Location:** `tools/location/`  
**Binary:** `bin/indigo-location`

### What it does

This calculates location-specific astronomical data like magnetic declination and other geophysical parameters based on your geographic coordinates. I use this to get accurate magnetic declination for my location, which is useful for aurora forecasting and other calculations.

### Usage

```bash
./bin/indigo-location --config <config_file> [OPTIONS]
```

### Parameters

| Parameter | Short | Description |
|-----------|-------|-------------|
| `--config` | | **Required.** Path to your config file |
| `--outfile` | `-o` | Write output to a file |
| `--longitude` | | Longitude in decimal degrees |
| `--latitude` | | Latitude in decimal degrees |
| `--altitude` | | Altitude in meters |
| `--debug` | `-v` | Enable debug output |
| `--help` | `-h`, `-?` | Show help message |

### Example

```bash
./bin/indigo-location --config /home/allskyuser/config.ini \
  --longitude 7.0 --latitude 50.2 --altitude 200 \
  --outfile location_data.json
```

---

## 4. allsky360-meteo

**Location:** `tools/meteo/`  
**Binary:** `bin/allsky360-meteo`

### What it does

This fetches meteorological data (temperature, humidity, cloud cover, etc.) from the Open-Meteo API. I use it to correlate weather conditions with my sky observations – it's interesting to see how cloud cover affects what I can capture. The data gets stored in a SQLite database just like the aurora data.

### Running automatically

This also runs via systemd by default, every 10 minutes. The service files are:

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

Change line 6 (`OnUnitActiveSec=10min`) to your desired interval:

* `OnUnitActiveSec=5min` – Every 5 minutes
* `OnUnitActiveSec=15min` – Every 15 minutes
* `OnUnitActiveSec=30min` – Every 30 minutes

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

## 5. timelapse-cuda-nv (allsky360-timelapse)

**Location:** `tools/timelapse/`  
**Binary:** `bin/timelapse-cuda-nv` or `bin/allsky360-timelapse`

### What it does

This is probably my favorite tool. It creates timelapse videos from a sequence of JPEG images, and it uses GPU acceleration for everything – JPEG decoding, image resizing, and video encoding. If you've got hundreds or thousands of images, this makes a huge difference. I've processed sequences with thousands of images in minutes instead of hours.

It also supports different stacking modes, which is really useful if you want to combine multiple images in interesting ways.

### Features

* GPU-accelerated JPEG decoding (using nvJPEG)
* GPU-accelerated image resizing (using NPP)
* GPU-accelerated video encoding (using NVENC)
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
* NVIDIA libraries: nvJPEG, NPP, NVENC

---

## Building the tools

You can build each tool individually from its directory:

```bash
cd tools/aurora && make
cd tools/meteo && make
cd tools/timelapse && make
cd tools/location && make
cd tools/indigo-list-cameras && make
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
