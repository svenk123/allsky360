# Installation Guide

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

If you have an NVIDIA GPU and want the faster GPU-accelerated version, you need to build the GPUJPEG library first, then build allsky360 with CUDA support.

##### Step 2a: Build GPUJPEG

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

After building, install the binaries:

```bash
sudo make install
```

This installs allsky360 and its tools to `/opt/allsky360/bin/`.
