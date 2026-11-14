# INDIGO Server Settings

Allsky360 does not talk to the camera directly. It connects to an
[INDIGO](https://github.com/indigo-astronomy/indigo) server, which runs the
camera driver. These keys tell Allsky360 where the server is and which camera
(and optionally which mount) to use.

The server can run on the same computer (`127.0.0.1`) or on another computer
in the network. `sudo ./install.sh --configure` asks for the server and lets
you pick camera and mount from the list the server reports.

| Key | Example | Meaning |
|---|---|---|
| `indigo_server` | `127.0.0.1` | IP address or host name of the INDIGO server |
| `indigo_port` | `7624` | TCP port of the server |
| `indigo_servername` | `indigo_server` | Name under which Allsky360 connects; part of every device name |
| `indigo_camera` | `"Uranus-C @ indigo_server"` | Camera device |
| `indigo_mount` | empty | Mount device (optional) |
| `indigo_capture_timeout` | `30` | Extra seconds to wait for an image after the exposure time |

---

## Server connection

### indigo_server

IP address or host name of the INDIGO server. Default `127.0.0.1` (same
computer). For a server elsewhere in the network use its address, e.g.
`192.168.3.61`. A fixed IP address is more reliable than a `.local` name.

### indigo_port

TCP port of the server. INDIGO uses `7624` unless you changed it. If the
server runs on another computer, its firewall must allow this port.

### indigo_servername

INDIGO reports every device of a remote server as
`<device name> @ <server name>`, and the server name is the name Allsky360
uses for the connection. That is this key. `indigo_camera` and
`indigo_mount` must therefore end with ` @ ` plus exactly this value.

Default `indigo_server`. You only need to change it if you have several
INDIGO servers; then change the device names accordingly.

---

## Devices

### indigo_camera

The camera device, exactly as INDIGO reports it, including the
` @ <indigo_servername>` suffix. The name is case-sensitive; put it in quotes.

List the cameras of the server:

```bash
/opt/allsky360/bin/indigo_cameras -h 127.0.0.1 -l
```

The list shows the names with ` @ indigo_server`; if you changed
`indigo_servername`, replace that part.

Examples:

```ini
indigo_camera="Uranus-C @ indigo_server"
indigo_camera="ZWO ASI294MC Pro @ indigo_server"
```

### indigo_mount

The mount device, if a telescope mount is connected to the same INDIGO server
(empty = no mount). List the mounts with:

```bash
/opt/allsky360/bin/indigo_cameras -h 127.0.0.1 -M
```

The mount is **only read, never moved or connected**: the tool
`allsky360-mount-position` reads its position once a minute into `mount.json`,
and the dashboard shows it as an overlay (checkbox *Mount*). Another program
(AIN, KStars, …) must have connected the mount; otherwise `mount.json` reports
the status `disconnected`. Possible status values: `ok`, `not_configured`,
`disconnected`, `no_response`, `server_unreachable`.

### indigo_capture_timeout

How many seconds to wait **in addition to the exposure time** for an image to
arrive. With a 60 s exposure and the value 30, the capture fails after 90 s.
Default in `config.ini.default`: 30.

The extra time covers the download from the camera and, for a remote server,
the network transfer. 20 – 30 s is enough for USB cameras on the same
computer; raise it for large sensors over a slow network.

---

## Troubleshooting

If the camera does not start, its log (`journalctl -u allsky360-camera -f`) shows what failed:

1. **Server not reachable** – is the INDIGO server running
   (`systemctl status indigo-server`)? Can you reach the address and port
   (`nc -zv <server> 7624`)?
2. **Camera not found** – compare `indigo_camera` with the output of
   `indigo_cameras -l` character by character, including the ` @ …` suffix.
3. **Capture timeouts** – raise `indigo_capture_timeout`. If timeouts keep
   happening, check the USB cable and power supply of the camera and the
   INDIGO server log.

At start the camera service waits until the INDIGO server answers, so after
a reboot the start order of the services does not matter.
