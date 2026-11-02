# INDIGO Server Settings

These parameters tell allsky360 how to connect to your INDIGO server and which camera to use. INDIGO is the astronomy device framework that handles communication with your camera hardware.

If you're running the INDIGO server on the same machine as allsky360, you can usually leave most of these at their defaults. If your camera is connected to a different computer, you'll need to adjust the server address.

## Server Connection

### indigo_server

The IP address or hostname where your INDIGO server is running. This can be either a local address (like `127.0.0.1` or `localhost`) or a remote address on your network (like `192.168.1.100`).

The default is `127.0.0.1`, which means the server is running on the same machine. If your camera is connected to a different computer, you'll need to change this to that computer's IP address.

For example, if your INDIGO server is running on a Raspberry Pi at IP address `192.168.3.61`, you'd set this to `192.168.3.61`. You can also use a hostname if your network supports it, like `raspberry-pi.local` – but IP addresses are usually more reliable.

### indigo_port

The network port number that your INDIGO server is listening on. The default is `7624`, which is the standard INDIGO port. You probably won't need to change this unless you've configured your INDIGO server to use a different port.

If you're not sure what port to use, check your INDIGO server configuration. Most installations use 7624, so unless you've changed it, stick with the default.

## Camera Selection

### indigo_camera

The exact name of the camera device as it appears in INDIGO. This needs to match exactly what INDIGO reports, including any network location information.

The default is `"Uranus-C @ localhost"`, but your camera name will be different. To find the correct name, you can use the `indigo-list-cameras` tool that comes with allsky360, or check the INDIGO control panel.

Camera names in INDIGO typically include the camera model and where it's connected. For example:
- `"Uranus-C @ 192.168.3.61"` – a Player-One Uranus-C connected via network
- `"ASI294MC Pro @ localhost"` – a ZWO camera connected locally
- `"QHY268M @ indigoserver.net"` – a QHY camera connected via network

The name is case-sensitive and must be in quotes if it contains spaces or special characters. If you get connection errors, double-check that the name matches exactly what INDIGO shows – even small differences like extra spaces will cause problems.

## Capture Timeout

### indigo_capture_timeout

The maximum time in seconds to wait for a single image capture to complete. If the camera doesn't finish capturing within this time, the capture is considered failed and the system will move on.

The default is 20 seconds, which works well for most cameras and exposure times. For very long exposures (like 60 seconds or more), you'll want to increase this value. A good rule of thumb is to set it to at least 1.5 times your longest expected exposure time, plus a few seconds for overhead.

For example, if your maximum exposure is 60 seconds, setting `indigo_capture_timeout` to 90 or 100 seconds gives the system enough time to complete the capture and transfer the data.

On the other hand, if you're only doing short exposures (a few seconds), you can reduce this value to fail faster if something goes wrong. But honestly, leaving it at 20 or 30 seconds is usually fine – the timeout only matters if something actually goes wrong.

## Practical Tips

When setting up a new camera, start by finding the correct camera name. Run `indigo-list-cameras` or check your INDIGO control panel to see what cameras are available. Copy the exact name, including any network location information.

If you're having connection issues, check:
1. Is the INDIGO server running? You can test this by trying to connect with the INDIGO control panel.
2. Is the IP address correct? Try pinging the server address to make sure it's reachable.
3. Is the camera name exactly right? Even small typos will prevent connection.
4. Is the port correct? Most installations use 7624, but check your server config if you're not sure.

For network-connected cameras, make sure the firewall allows connections on port 7624. Some Linux distributions block this by default.

If captures are timing out, increase `indigo_capture_timeout`. But if you're consistently hitting timeouts, there might be a deeper issue – like network problems or the camera not responding properly. In that case, check your INDIGO server logs for more details.

