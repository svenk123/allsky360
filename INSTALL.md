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

### Step 1: Get the source code

```bash
git clone https://github.com/indigo-astronomy/indigo.git
cd indigo
```

### Step 2: Install dependencies

On Ubuntu/Debian, you'll need these packages:

```bash
sudo apt update
sudo apt install build-essential libusb-1.0-0-dev libjpeg-dev cmake git

sudo apt-get install build-essential autoconf autotools-dev libtool cmake libudev-dev libavahi-compat-libdnssd-dev libusb-1.0-0-dev libcurl4-gnutls-dev libz-dev git curl bsdmainutils bsdextrautils patchelf

sudo apt-get remove libraw1394-dev

sudo apt install libavahi-compat-libdnssd-dev
sudo apt install libcurl4-openssl-dev
```

### Step 3: Build it

```bash
cd indigo
make
```

**Note:** If the build fails when compiling the QHY camera drivers (it happens sometimes), here's the fix:

```bash
cd indigo_drivers/ccd_qhy/bin_externals/pthread_yield_compat/
make patchlib
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
indigoserver -v -c
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

This section will be added as the installation process for allsky360 is documented.
