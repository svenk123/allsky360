# Installation Guide

## Quick installation

Once INDIGO is built (Part 1 below), one script does the rest:

```bash
git clone <allsky360 repository> ~/allsky360   # INDIGO in ~/indigo next to it
cd ~/allsky360
sudo ./install.sh
```

`install.sh` runs from your normal user account (via sudo) and

1. checks the system: Jetson/JetPack, CUDA (GPU build is chosen
   automatically when `nvcc` is present), disk space, INDIGO build,
2. installs the packages (apt, CUDA libraries, Node.js 24),
3. builds allsky360 and the tools **as your user** (not as root),
4. installs everything to `/opt/allsky360` (`make install`),
5. runs the **configuration assistant**: location, INDIGO server, camera and
   mount (picked from the list the server reports), API token,
6. optionally installs the AI venv (Ultralytics YOLO26, `sudo make yolo`),
7. runs a self-test (binaries, web UI, polkit rule, INDIGO reachable),
8. asks whether to start the camera and the data timers - nothing is
   started without your confirmation.

Useful options (`./install.sh --help`):

| Option | Effect |
|---|---|
| `--cpu` / `--cuda` | force the build variant |
| `--with-ai` / `--no-ai` | install the YOLO venv without asking / skip it |
| `--indigo-root DIR` | INDIGO source tree if it is not in `../indigo` |
| `--skip-deps` | do not touch apt / Node.js |
| `--yes` | unattended: defaults, no assistant, services not started |
| `--configure` | only the configuration assistant (also later: `sudo ./install.sh --configure`) |

Running `sudo ./install.sh` again updates an existing installation; your
`config.ini` is kept.

The sections below describe the individual steps for manual installations
and troubleshooting.

---

The installation process has two main parts. Let me walk you through both.

## Part 1: Installing the INDIGO Server

The INDIGO server controls your camera. Here's the thing: it doesn't have to run on the same machine as allsky360. The server uses the INDIGO network protocol (similar to INDI) to talk to cameras and other equipment over TCP. This means you can run the server on a separate computer if that works better for your setup.

## Part 2: Installing allsky360

This is where you build and set up the allsky360 software and its tools. There are two ways to install it, depending on your hardware:

**CPU-based installation** - This is the standard version that runs on the CPU. It works on pretty much any system, including Raspberry Pi and other ARM-based boards. It's slower but more compatible.

**GPU-accelerated installation** - If you have an NVIDIA GPU in your system, you can use this version. It's significantly faster for image processing tasks, especially when dealing with high-resolution images or real-time processing. You'll need CUDA installed for this to work.

---

## Installing the INDIGO Server

You'll need a running `indigoserver` from the [INDIGO Astronomy Project](https://github.com/indigo-astronomy/indigo). It handles camera control and image retrieval over the network.

### Step 1: Install dependencies

On Ubuntu/Debian, you'll need these packages:

```bash
sudo apt update
sudo apt install build-essential libusb-1.0-0-dev libjpeg-dev cmake git

sudo apt-get install build-essential autoconf autotools-dev libtool cmake libudev-dev libavahi-compat-libdnssd-dev libusb-1.0-0-dev libcurl4-gnutls-dev libz-dev git curl bsdmainutils bsdextrautils patchelf

sudo apt-get remove libraw1394-dev

sudo apt install libavahi-compat-libdnssd-dev
sudo apt install libcurl4-openssl-dev
```

### Step 2: Get the source code

```bash
git clone https://github.com/indigo-astronomy/indigo.git
cd indigo
```

### Step 3: Build it

```bash
make
```

**Note:** If the build fails when compiling the QHY camera drivers and you stops at this error.

```bash
/usr/bin/ld: bin_externals/qhyccd/lib/Linux/arm64//libqhy.a(qhy5iii367base.o):qhy5iii367base.cpp:(.text+0x604): more undefined references to `pthread_yield' follow
collect2: error: ld returned 1 exit status
make[2]: *** [../../Makefile.drv:126: /home/user1/indigo/build/drivers/indigo_ccd_qhy] Error 1
make[2]: Leaving directory '/home/user1/indigo/indigo_drivers/ccd_qhy'
make[1]: *** [../Makefile.drvs:64: ccd_qhy/] Error 2
make[1]: Leaving directory '/home/user1/indigo/indigo_drivers'
make: *** [Makefile:151: all] Error 2
```

Here's the fix:

```bash
cd indigo_drivers/ccd_qhy/bin_externals/pthread_yield_compat/
make patchlib
cd -
make
```

Then finish the installation:

```bash
sudo make install
```

### Step 4: Starting the server

You have two options here. For a quick test, you can run it manually. For production use, set it up as a systemd service.

#### Manual start (for testing)

Good for checking if everything works:

```bash
indigo_server -v -c
```

The server listens on TCP port 7624 by default. You can connect to it from any machine on your network.

#### Running as a systemd service (recommended)

This is the way to go if you want the server to start automatically and keep running reliably. I've been running it this way for years without issues.

**Create the service file:**

Create `/etc/systemd/system/indigo-server.service` with this content:

```ini
[Unit]
Description=INDIGO Server

[Service]
User=user1
PAMName=login
ExecStart=/usr/local/bin/indigo_server indigo_ccd_asi indigo_ccd_playerone indigo_mount_lx200
#StandardOutput=file:/home/user1/indigo.log
#StandardError=file:/home/user1/indigo.log

[Install]
WantedBy=multi-user.target
```

**A few things to adjust:**

- Change `user1` to your actual username. This is the user the service will run as.
- The `ExecStart` line lists the drivers you need. The example shows ZWO ASI (`indigo_ccd_asi`) and Player One (`indigo_ccd_playerone`) drivers. You'll want to match this to your camera. Common options:
  - `indigo_ccd_asi` - ZWO ASI cameras
  - `indigo_ccd_playerone` - Player One cameras
  - `indigo_ccd_qhy` - QHY cameras
  - `indigo_ccd_atik` - Atik cameras
  - Add whatever else your camera needs

**Enable and start it:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable indigo-server.service
sudo systemctl start indigo-server.service
```

**Check if it's running:**

```bash
sudo systemctl status indigo-server.service
```

If something's not working, check the logs:

```bash
sudo journalctl -u indigo-server.service -f
```

The `-f` flag follows the log in real-time, which is handy when troubleshooting.

---

## Installing allsky360

There are two build variants available, depending on your hardware:

**CPU-only variant**: This version runs on the CPU and works on most systems, including Raspberry Pi and other ARM-based boards. It's slower but has broader compatibility.

**GPU-accelerated variant**: If you have an NVIDIA GPU with CUDA support, you can use this version. It's significantly faster for image processing, especially with high-resolution images or real-time processing.

### Step 1: Install dependencies

Before building, you need to install the required Ubuntu packages. Run this first:

```bash
make deps
```

This will install all the necessary dependencies for building allsky360.

### Step 2: Choose your build variant

#### CPU-only build (for Raspberry Pi and similar systems)

This is the simpler option. Just run:

```bash
make
```

This builds the CPU-only version that works on most Linux systems, including Raspberry Pi and other ARM boards. It doesn't require CUDA or any GPU libraries.

#### GPU-accelerated build (for systems with NVIDIA GPU)

If you have an NVIDIA GPU, build allsky360 with CUDA support (Step 2b). The
CUDA architecture is detected automatically (`-arch=native`); for a cross
build pass e.g. `CUDA_ARCH=sm_87`.

##### Step 2a (optional): Build GPUJPEG

GPUJPEG is **not required**: JPEG encoding uses nvjpeg from the CUDA toolkit.
It is only used when you build with `make USE_CUDA=1 USE_GPUJPEG=1`.

First, clone and build the GPUJPEG library:

```bash
git clone https://github.com/CESNET/GPUJPEG
cd GPUJPEG/

mkdir bld
cd bld
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUJPEG_CUDA_ARCHITECTURES="87" -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo make install
```

**Note**: The `GPUJPEG_CUDA_ARCHITECTURES` value should match your GPU's compute capability. For Jetson Orin NX, use `"87"`. For other GPUs, check NVIDIA's documentation for the correct architecture number.

After installation, update the library cache:

```bash
sudo ldconfig
```

You can verify the installation by checking if the library is found:

```bash
ldconfig -p | grep gpujpeg
```

##### Step 2b: Build allsky360 with CUDA support

Now build allsky360 with GPU acceleration enabled:

```bash
cd /path/to/allsky360
make USE_CUDA=1
```

This builds the GPU-accelerated version that uses CUDA for image processing tasks.

### Step 3: Install the binaries

The web API needs **Node.js 24** (LTS). `sudo make deps` sets it up from the
NodeSource repository; on an existing system with an older Node.js run:

```bash
sudo make node-deps   # configures NodeSource node_24.x and installs nodejs
make node-check       # prints the version, fails if it is older than 24
```

After building, install the binaries:

```bash
sudo make install
```

This installs allsky360 and its tools to `/opt/allsky360/bin/`.

`make install` only enables and starts the web API. On a fresh install the
camera service and the data timers (aurora, meteo, location, mount position)
are installed but **not** started: `config.ini` still contains the template
values. Enter at least the location (`location_latitude`,
`location_longitude`, `location_altitude`) and the INDIGO settings
(`indigo_server`, `indigo_port`, `indigo_camera`, `indigo_mount`), then start
everything manually:

```bash
sudo -u allskyuser nano /home/allskyuser/config.ini
sudo systemctl enable --now allsky360-camera.service \
    allsky360-aurora.timer allsky360-meteo.timer \
    allsky360-location.timer allsky360-mount-position.timer
tail -f /var/log/allsky360/camera.log
```

On an update (units already enabled) `make install` restarts a running
camera so the new binary is used; enabled timers keep running.

The configuration is installed to `/home/allskyuser/config.ini` (mode 640,
it contains the API token). Running `sudo make install` again for an update
never overwrites an existing `config.ini`; the current template is written to
`config.ini.new` instead, so you can compare it and copy new settings over.

### Step 4: Set the API token

The web configuration page has Start/Stop/Restart buttons for the allsky360
services. They call privileged API endpoints that are protected by a shared
secret, the `api-token` in `/home/allskyuser/config.ini`. **The token is empty
after installation, so the buttons do not work until you set one.**

1. Generate a random token:

   ```bash
   openssl rand -hex 32
   ```

2. Enter it in `/home/allskyuser/config.ini`:

   ```ini
   api-token=<the generated token>
   ```

3. Restart the API so it reads the new value:

   ```bash
   sudo systemctl restart allsky360-api.service
   ```

4. Click a service button in the web UI and enter the token when asked.
   The browser keeps it only until the tab is closed (`sessionStorage`).

Notes:

- Without a token the service-control endpoints answer with HTTP 403. All
  read-only pages (images, graphs, service status) work without a token.
- The token is only read by the API. The camera daemon ignores it and it is
  never published in `config.json`.
- The API speaks plain HTTP. In an untrusted network set `api-bind=127.0.0.1`
  and put an HTTPS reverse proxy in front of it (template in `etc/apache2`).
- Scripts can use the endpoints directly, see `API.md`:

  ```bash
  curl -X POST -H "X-API-Token: <token>" \
    "http://allsky:3000/api/services/restart?service=allsky360-camera.service"
  ```

### Step 5 (optional): AI sky classification with Ultralytics YOLO26

`scripts/postprocessing.sh` can classify every image (clear, cloudy, ...)
with an Ultralytics YOLO classification model. It needs a Python venv with
Ultralytics, PyTorch for CUDA 13, onnxruntime-gpu and the JetPack TensorRT
bindings. The Makefile installs it the way the Ultralytics
[quickstart](https://docs.ultralytics.com/quickstart) and the
[NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
(section JetPack 7.2) recommend:

```bash
sudo make yolo        # creates /home/allskyuser/venvs/yolo (several GB download)
make yolo-check       # prints versions, fails if torch does not see the GPU
```

Requirements: JetPack 7.2 (Ubuntu 24.04, Python 3.12, CUDA 13, TensorRT 10).
For other platforms override the variables, e.g.
`sudo make yolo TORCH_INDEX_URL=... ONNXRUNTIME_GPU_WHL=...`.

The model itself is not part of the repository. Copy your trained
classification model and the class list to `/opt/allsky360/ai/models/`
(`weather.pt`, `classes.txt`). Models trained with YOLO11 keep working with
the current Ultralytics release; for YOLO26 retrain on top of a YOLO26
classification checkpoint.

TensorRT engines (`*.engine`) are tied to the GPU and the TensorRT version.
After a JetPack upgrade export them again on the device:

```bash
/opt/allsky360/scripts/export_trt.sh /opt/allsky360/ai/models/weather.pt 640
```

The old venv `/home/allskyuser/venvs/yolo11` (Python 3.10, JetPack 6) is no
longer used and can be deleted.

