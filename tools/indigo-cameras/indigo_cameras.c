/*****************************************************************************
 *
 * Copyright (c) 2026 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

// Reads camera information from a running INDIGO server and prints
// CCD_INFO and CCD_TEMPERATURE to stdout.
//
// Usage: indigo_cameras -h <server> [-p <port>] [-c <camera_name>] [-t <secs>] [-l] [-j]
//   -h / --host        <server>      hostname or IP of the INDIGO server
//   -p / --port        <port>        TCP port (default: 7624)
//   -c / --camera      <camera_name> device name as reported by the server
//                                    (default: "CCD Imager Simulator")
//   -t / --timeout     <secs>        timeout waiting for camera response (default: 10)
//   -l / --list-cameras              list available camera device names and exit
//   -j / --json                      output as JSON
//
// Example:
//   indigo_cameras -h indigosky.local -l
//   indigo_cameras -h indigosky.local -l -j
//   indigo_cameras -h indigosky.local -p 7624 -c "QHY CCD QHY268M-XXXXXXXX"
//   indigo_cameras -h indigosky.local -p 7624 -c "QHY CCD QHY268M-XXXXXXXX" -j

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(INDIGO_LINUX) || defined(INDIGO_MACOS)
#include <unistd.h>
#include <getopt.h>
#endif
#include <indigo/indigo_bus.h>
#include <indigo/indigo_client.h>
#include <indigo/indigo_ccd_driver.h>

/* ------------------------------------------------------------------ */
/* Global state shared between callbacks and main()                   */
/* ------------------------------------------------------------------ */

static char g_camera_name[256] = "CCD Imager Simulator";
static bool g_finished         = false;
static bool g_connected        = false; /* true once camera is connected */
static bool g_we_connected     = false; /* true if we initiated the connection */

/* CCD_INFO */
static bool   g_got_info          = false;
static double g_width             = 0.0;
static double g_height            = 0.0;
static double g_pixel_size        = 0.0;
static double g_pixel_width       = 0.0;
static double g_pixel_height      = 0.0;
static double g_max_horiz_bin     = 0.0;
static double g_max_vert_bin      = 0.0;
static double g_bits_per_pixel    = 0.0;

/* CCD_TEMPERATURE */
static bool   g_got_temperature   = false;
static double g_temperature       = 0.0;

/* List-mode: collect camera device names */
static bool g_list_mode = false;
#define MAX_FOUND_CAMERAS 64
static char g_found_cameras[MAX_FOUND_CAMERAS][256];
static int  g_found_count = 0;
/* Number of INFO properties seen so far (list mode stops when it is quiet) */
static volatile int g_info_seen = 0;

static bool g_json_mode = false;

/* ------------------------------------------------------------------ */
/* JSON helper                                                         */
/* ------------------------------------------------------------------ */
static void json_escape(const char *src, char *buf, size_t len) {
	size_t out = 0;
	for (const char *p = src; *p && out + 2 < len; p++) {
		if (*p == '"' || *p == '\\')
			buf[out++] = '\\';
		buf[out++] = *p;
	}
	buf[out] = '\0';
}

/* ------------------------------------------------------------------ */
/* INDIGO client callbacks                                             */
/* ------------------------------------------------------------------ */
static indigo_result client_attach(indigo_client *client) {
	indigo_enumerate_properties(client, &INDIGO_ALL_PROPERTIES);
	return INDIGO_OK;
}

static void collect_camera(const char *device_name) {
	for (int i = 0; i < g_found_count; i++) {
		if (strcmp(g_found_cameras[i], device_name) == 0)
			return;
	}
	if (g_found_count < MAX_FOUND_CAMERAS) {
		strncpy(g_found_cameras[g_found_count], device_name, 255);
		g_found_cameras[g_found_count][255] = '\0';
		g_found_count++;
	}
}

static void read_ccd_info(indigo_property *property) {
	indigo_item *item;

	if ((item = indigo_get_item(property, CCD_INFO_WIDTH_ITEM_NAME)))
		g_width = item->number.value;
	if ((item = indigo_get_item(property, CCD_INFO_HEIGHT_ITEM_NAME)))
		g_height = item->number.value;
	if ((item = indigo_get_item(property, CCD_INFO_PIXEL_SIZE_ITEM_NAME)))
		g_pixel_size = item->number.value;
	if ((item = indigo_get_item(property, CCD_INFO_PIXEL_WIDTH_ITEM_NAME)))
		g_pixel_width = item->number.value;
	if ((item = indigo_get_item(property, CCD_INFO_PIXEL_HEIGHT_ITEM_NAME)))
		g_pixel_height = item->number.value;
	if ((item = indigo_get_item(property, CCD_INFO_MAX_HORIZONTAL_BIN_ITEM_NAME)))
		g_max_horiz_bin = item->number.value;
	if ((item = indigo_get_item(property, CCD_INFO_MAX_VERTICAL_BIN_ITEM_NAME)))
		g_max_vert_bin = item->number.value;
	if ((item = indigo_get_item(property, CCD_INFO_BITS_PER_PIXEL_ITEM_NAME)))
		g_bits_per_pixel = item->number.value;
	g_got_info = true;
}

static indigo_result client_define_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
	(void)client;
	(void)device;
	(void)message;
	if (g_list_mode) {
		if (strcmp(property->name, INFO_PROPERTY_NAME) == 0) {
			indigo_item *iface = indigo_get_item(property, INFO_DEVICE_INTERFACE_ITEM_NAME);
			long flags = iface ? atol(iface->text.value) : 0;
			/* Agents (e.g. "Imager Agent") also report the CCD interface,
			 * but they are not cameras */
			if ((flags & INDIGO_INTERFACE_CCD) && !(flags & INDIGO_INTERFACE_AGENT))
				collect_camera(property->device);
			g_info_seen++;
		}

		return INDIGO_OK;
	}

	if (strcmp(property->device, g_camera_name) != 0)
		return INDIGO_OK;

	if (strcmp(property->name, CONNECTION_PROPERTY_NAME) == 0) {
		if (indigo_get_switch(property, CONNECTION_CONNECTED_ITEM_NAME)) {
			g_connected = true;
		} else {
			/* Camera is disconnected — connect it and remember we did so */
			g_we_connected = true;
			indigo_device_connect(client, g_camera_name);
		}
	} else if (strcmp(property->name, CCD_INFO_PROPERTY_NAME) == 0) {
		read_ccd_info(property);
	} else if (strcmp(property->name, CCD_TEMPERATURE_PROPERTY_NAME) == 0) {
		indigo_item *item = indigo_get_item(property, CCD_TEMPERATURE_ITEM_NAME);
		if (item) {
			g_temperature     = item->number.value;
			g_got_temperature = true;
		}
	}

	if (g_got_info)
		g_finished = true;

	return INDIGO_OK;
}

static indigo_result client_update_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
	(void)client;
	(void)device;
	(void)message;
	if (g_list_mode)
		return INDIGO_OK;

	if (strcmp(property->device, g_camera_name) != 0)
		return INDIGO_OK;

	if (strcmp(property->name, CONNECTION_PROPERTY_NAME) == 0 &&
	    property->state == INDIGO_OK_STATE) {
		if (indigo_get_switch(property, CONNECTION_CONNECTED_ITEM_NAME)) {
			g_connected = true;
		}
	} else if (strcmp(property->name, CCD_INFO_PROPERTY_NAME) == 0) {
		read_ccd_info(property);
	} else if (strcmp(property->name, CCD_TEMPERATURE_PROPERTY_NAME) == 0) {
		indigo_item *item = indigo_get_item(property, CCD_TEMPERATURE_ITEM_NAME);
		if (item) {
			g_temperature     = item->number.value;
			g_got_temperature = true;
		}
	}

	if (g_got_info)
		g_finished = true;

	return INDIGO_OK;
}

static indigo_result client_detach(indigo_client *client) {
	(void)client;
	return INDIGO_OK;
}

static indigo_client g_client = {
	.name = "indigo_cameras",
	.is_remote = false,
	.client_context = NULL,
	.last_result = INDIGO_OK,
	.version = INDIGO_VERSION_CURRENT,
	.enable_blob_mode_records = NULL,
	.attach = client_attach,
	.define_property = client_define_property,
	.update_property = client_update_property,
	.delete_property = NULL,
	.send_message = NULL,
	.detach = client_detach,
};

static void print_usage(const char *prog) {
	fprintf(stderr, "Usage: %s -h <server> [-p <port>] [-c <camera_name>] [-t <secs>] [-l] [-j]\n", prog);
	fprintf(stderr, "  -h / --host          <server>      hostname or IP of the INDIGO server\n");
	fprintf(stderr, "  -p / --port          <port>        TCP port (default: 7624)\n");
	fprintf(stderr, "  -c / --camera        <camera_name> device name (default: \"CCD Imager Simulator\")\n");
	fprintf(stderr, "  -t / --timeout       <secs>        timeout in seconds (default: 10)\n");
	fprintf(stderr, "  -l / --list-cameras               list available cameras and exit\n");
	fprintf(stderr, "  -j / --json                        output as JSON\n");
}

int main(int argc, const char *argv[]) {
	indigo_main_argc = argc;
	indigo_main_argv = argv;

	const char *server = NULL;
	int port    = 7624;
	int timeout = 10;

	static struct option long_opts[] = {
		{ "host",          required_argument, NULL, 'h' },
		{ "port",          required_argument, NULL, 'p' },
		{ "camera",        required_argument, NULL, 'c' },
		{ "timeout",       required_argument, NULL, 't' },
		{ "list-cameras",  no_argument,       NULL, 'l' },
		{ "json",          no_argument,       NULL, 'j' },
		{ NULL, 0, NULL, 0 }
	};

	int opt;
	while ((opt = getopt_long(argc, (char *const *)argv, "h:p:c:t:lj", long_opts, NULL)) != -1) {
		switch (opt) {
			case 'h':
				server = optarg;
				break;
			case 'p':
				port = atoi(optarg);
				if (port <= 0 || port > 65535) {
					fprintf(stderr, "Invalid port: %s\n", optarg);
					return 1;
				}
				break;
			case 'c':
				strncpy(g_camera_name, optarg, sizeof(g_camera_name) - 1);
				g_camera_name[sizeof(g_camera_name) - 1] = '\0';
				break;
			case 't':
				timeout = atoi(optarg);
				if (timeout <= 0) {
					fprintf(stderr, "Invalid timeout: %s\n", optarg);
					return 1;
				}
				break;
			case 'l':
				g_list_mode = true;
				break;
			case 'j':
				g_json_mode = true;
				break;
			default:
				print_usage(argv[0]);
				return 1;
		}
	}

	if (!server) {
		fprintf(stderr, "Error: --host is required\n\n");
		print_usage(argv[0]);
		return 1;
	}

	indigo_set_log_level(INDIGO_LOG_ERROR);
	if (indigo_start() != INDIGO_OK) {
		fprintf(stderr, "Failed to start INDIGO.\n");
		return 2;
	}

	indigo_server_entry *server_entry;
	static bool connection_state = false;
	indigo_attach_client(&g_client);
	indigo_result rc = indigo_connect_server("indigo_server", /* same name as the camera daemon (indigo_servername) */ server, port, &server_entry, &connection_state);
	if (rc != INDIGO_OK) {
		fprintf(stderr, "Failed to connect to %s:%d (rc=%d)\n", server, port, rc);
		indigo_detach_client(&g_client);
		indigo_stop();
		return 1;
	}

	if (g_list_mode) {
		/* Collect the INFO properties of all devices. The server sends them
		 * in a burst after the connect, but a busy server can take a while:
		 * wait up to the timeout, and stop once a camera was found and no
		 * further device has shown up for 1 s (a fixed 3 s wait missed the
		 * camera at times, BUG-15). */
		int waited_ms = 0, quiet_ms = 0, last_seen = 0;
		while (waited_ms < timeout * 1000) {
			indigo_usleep(100000);
			waited_ms += 100;
			if (g_info_seen != last_seen) {
				last_seen = g_info_seen;
				quiet_ms = 0;
			} else {
				quiet_ms += 100;
			}
			if (g_found_count > 0 && quiet_ms >= 1000)
				break;
		}
	} else {
		/* Wait for CCD_INFO up to the configured timeout */
		int timeout_ms = timeout * 1000;
		while (!g_finished && timeout_ms > 0) {
			indigo_usleep(100000);
			timeout_ms -= 100;
		}

		/* Once CCD_INFO is in, wait briefly for optional temperature (max 1 s) */
		if (g_got_info && !g_got_temperature) {
			int wait_ms = 1000;
			while (!g_got_temperature && wait_ms > 0) {
				indigo_usleep(100000);
				wait_ms -= 100;
			}
		}

		/* Disconnect camera if we were the ones who connected it */
		if (g_we_connected)
			indigo_device_disconnect(&g_client, g_camera_name);
	}

	indigo_disconnect_server(server_entry);
	/* The disconnect is asynchronous: wait (max. 2 s) until the server
	 * device is gone, otherwise indigo_stop() complains that it is still
	 * attached */
	for (int i = 0; i < 20 && connection_state; i++)
		indigo_usleep(100000);
	indigo_detach_client(&g_client);
	indigo_stop();

	/* ---------------------------------------------------------------- */
	/* Output                                                            */
	/* ---------------------------------------------------------------- */

	if (g_list_mode) {
		if (g_found_count == 0) {
			if (g_json_mode)
				printf("{\"server\":\"%s\",\"port\":%d,\"cameras\":[]}\n", server, port);
			else
				fprintf(stderr, "No cameras found on %s:%d\n", server, port);
			return 2;
		}
		if (g_json_mode) {
			char esc[512];
			printf("{\"server\":\"%s\",\"port\":%d,\"cameras\":[", server, port);
			for (int i = 0; i < g_found_count; i++) {
				json_escape(g_found_cameras[i], esc, sizeof(esc));
				printf("%s\"%s\"", i > 0 ? "," : "", esc);
			}
			printf("]}\n");
		} else {
			printf("Available cameras on %s:%d:\n", server, port);
			for (int i = 0; i < g_found_count; i++)
				printf("  %s\n", g_found_cameras[i]);
		}
		return 0;
	}

	if (!g_got_info) {
		fprintf(stderr, "Error: no response from camera \"%s\" on %s:%d\n",
		        g_camera_name, server, port);
		fprintf(stderr, "Tip: use -l / --list-cameras to see available camera names.\n");
		return 2;
	}

	if (g_json_mode) {
		char esc[512];
		json_escape(g_camera_name, esc, sizeof(esc));
		printf("{\n");
		printf("  \"camera\": \"%s\",\n", esc);
		printf("  \"ccd_info\": {\n");
		printf("    \"width_px\": %.0f,\n", g_width);
		printf("    \"height_px\": %.0f,\n", g_height);
		printf("    \"pixel_size_um\": %.4f,\n", g_pixel_size);
		printf("    \"pixel_width_um\": %.4f,\n", g_pixel_width);
		printf("    \"pixel_height_um\": %.4f,\n", g_pixel_height);
		printf("    \"max_horizontal_bin\": %.0f,\n", g_max_horiz_bin);
		printf("    \"max_vertical_bin\": %.0f,\n", g_max_vert_bin);
		printf("    \"bits_per_pixel\": %.0f\n", g_bits_per_pixel);
		printf("  },\n");
		if (g_got_temperature)
			printf("  \"temperature_c\": %.2f\n", g_temperature);
		else
			printf("  \"temperature_c\": null\n");
		printf("}\n");
	} else {
		printf("Camera:              %s\n", g_camera_name);
		printf("Sensor size:         %.0f x %.0f px\n", g_width, g_height);
		printf("Pixel size:          %.4f µm  (%.4f x %.4f µm)\n",
		       g_pixel_size, g_pixel_width, g_pixel_height);
		printf("Max binning:         %.0f x %.0f\n", g_max_horiz_bin, g_max_vert_bin);
		printf("Bits per pixel:      %.0f\n", g_bits_per_pixel);
		if (g_got_temperature)
			printf("Temperature:         %.2f °C\n", g_temperature);
		else
			printf("Temperature:         not available\n");
	}

	return 0;
}
