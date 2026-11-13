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

// Reads the current mount position from a running INDIGO server and
// prints MOUNT_EQUATORIAL_COORDINATES and MOUNT_HORIZONTAL_COORDINATES
// to stdout.
//
// The tool only reads: it never connects, disconnects or moves the mount, so
// it does not disturb a client that uses it (and works on a device locked
// with an access token, which only blocks change requests). The driver
// defines the MOUNT_* properties only while the device is connected, so some
// client (or the server) must have connected the mount.
//
// With -o the file is always written, also when there is no position, so the
// web page never shows an old position as current:
//   "timestamp": Unix time (UTC) of this run
//   "status":    "ok"                 position read
//                "not_configured"     -c given, indigo_mount empty, no -m
//                "disconnected"       mount known to the server, not connected
//                "no_response"        mount not found / no coordinates
//                "server_unreachable" no connection to the INDIGO server
// Exit status: 0 ok or not configured, 2 mount not available (disconnected,
// no response; the systemd service counts this as success), 1 other errors
// (server unreachable, bad arguments, output file).
//
// Usage: indigo_mount_position [-c <config>] [-h <server>] [-p <port>] [-m <mount_name>] [-t <secs>] [-l] [-j] [-o <file>]
//   -c / --config      <file>        allsky360 config.ini (sets indigo_server, indigo_port, indigo_mount)
//   -h / --host        <server>     - hostname or IP of the INDIGO server
//   -p / --port        <port>       - TCP port (default: 7624)
//   -m / --mount       <mount_name> - device name as reported by the server
//                                     (default: "Mount LX200")
//   -t / --timeout     <secs>       - timeout waiting for mount response (default: 10)
//   -l / --list-mounts              - list available mount device names and exit
//   -j / --json                     - output as JSON
//   -o / --output      <file>       - write output to file instead of stdout (implies -j)
//
// CLI arguments override values from -c/--config.
//
// Example:
//   indigo_mount_position -c /home/allskyuser/config.ini -j -o /opt/allsky360/allsky-api/public/mount.json
//   indigo_mount_position -h indigosky.local -l
//   indigo_mount_position -h indigosky.local -l -j
//   indigo_mount_position -h indigosky.local -p 7624 -m "Mount Simulator"
//   indigo_mount_position -h indigosky.local -p 7624 -m "Mount Simulator" -j

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#if defined(INDIGO_LINUX) || defined(INDIGO_MACOS)
#include <unistd.h>
#include <getopt.h>
#endif
#if defined(INDIGO_WINDOWS)
#include <windows.h>
#pragma warning(disable:4996)
#endif
#include <indigo/indigo_bus.h>
#include <indigo/indigo_client.h>
#include <indigo/indigo_names.h>

/* ------------------------------------------------------------------ */
/* Global state shared between callbacks and main()                   */
/* ------------------------------------------------------------------ */

static char g_mount_name[256] = "Mount LX200";
/* Name under which the server is connected. INDIGO reports remote devices as
 * "<device> @ <server name>", so this must match the camera daemon
 * (indigo_servername in config.ini, default "indigo_server") - otherwise the
 * same indigo_mount value cannot fit both. */
static char g_server_name[INDIGO_NAME_SIZE] = "indigo_server";
static bool g_got_equatorial  = false;
static bool g_got_horizontal  = false;
static bool g_finished        = false;

static double g_ra  = 0.0;   /* decimal hours  */
static double g_dec = 0.0;   /* decimal degrees */
static double g_alt = 0.0;   /* decimal degrees */
static double g_az  = 0.0;   /* decimal degrees */

static bool g_got_side_of_pier = false;
static bool g_pier_east        = false; /* true = EAST, false = WEST */

static bool g_got_tracking     = false;
static bool g_tracking_on      = false; /* true = ON, false = OFF */

/* CONNECTION property of the mount: without a connection the driver does not
 * define the coordinate properties, so there is nothing to wait for */
static bool g_mount_disconnected = false;

/* List-mode: collect mount device names */
static bool g_list_mode = false;
static bool g_json_mode = false;
#define MAX_FOUND_MOUNTS 64
static char g_found_mounts[MAX_FOUND_MOUNTS][256];
static int  g_found_count = 0;

/* ------------------------------------------------------------------ */
/* Coordinate formatting helpers                                       */
/* ------------------------------------------------------------------ */

/* Formats a value in decimal hours as  H h MM m SS.s s  */
static void format_ra(double hours, char *buf, size_t len) {
	int h   = (int)hours;
	double rem = (hours - h) * 60.0;
	int m   = (int)rem;
	double s = (rem - m) * 60.0;
	snprintf(buf, len, "%02dh %02dm %05.2fs", h, m, s);
}

/* Formats a value in decimal degrees as  ±DD° MM' SS.s"  */
static void format_dec(double deg, char *buf, size_t len) {
	char sign = (deg < 0.0) ? '-' : '+';
	double abs_deg = fabs(deg);
	int d   = (int)abs_deg;
	double rem = (abs_deg - d) * 60.0;
	int m   = (int)rem;
	double s = (rem - m) * 60.0;
	snprintf(buf, len, "%c%02d° %02d' %05.2f\"", sign, d, m, s);
}

/* Writes a JSON-escaped string (without surrounding quotes) to buf */
static void json_escape(const char *src, char *buf, size_t len) {
	size_t out = 0;
	for (const char *p = src; *p && out + 2 < len; p++) {
		if (*p == '"' || *p == '\\') {
			buf[out++] = '\\';
		}
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

static void collect_mount(const char *device_name) {
	for (int i = 0; i < g_found_count; i++) {
		if (strcmp(g_found_mounts[i], device_name) == 0)
			return; /* already known */
	}
	if (g_found_count < MAX_FOUND_MOUNTS) {
		strncpy(g_found_mounts[g_found_count], device_name, 255);
		g_found_mounts[g_found_count][255] = '\0';
		g_found_count++;
	}
}

static void process_property(indigo_property *property) {
	if (g_list_mode) {
		/* Identify mounts via INFO.DEVICE_INTERFACE bitmask */
		if (strcmp(property->name, INFO_PROPERTY_NAME) == 0) {
			indigo_item *iface = indigo_get_item(property, INFO_DEVICE_INTERFACE_ITEM_NAME);
			if (iface) {
				long bits = atol(iface->text.value);
				/* Agents (e.g. "Mount Agent") report the mount interface
				 * too, but they are not mounts */
				if ((bits & INDIGO_INTERFACE_MOUNT) && !(bits & INDIGO_INTERFACE_AGENT))
					collect_mount(property->device);
			}
		}
		return;
	}

	if (strcmp(property->device, g_mount_name) != 0)
		return;

	if (strcmp(property->name, CONNECTION_PROPERTY_NAME) == 0) {
		/* BUSY = another client is just connecting: keep waiting */
		g_mount_disconnected = property->state != INDIGO_BUSY_STATE &&
		                       !indigo_get_switch(property, CONNECTION_CONNECTED_ITEM_NAME);
		if (g_mount_disconnected)
			g_finished = true;
		return;
	} else if (strcmp(property->name, MOUNT_EQUATORIAL_COORDINATES_PROPERTY_NAME) == 0) {
		indigo_item *ra_item  = indigo_get_item(property, MOUNT_EQUATORIAL_COORDINATES_RA_ITEM_NAME);
		indigo_item *dec_item = indigo_get_item(property, MOUNT_EQUATORIAL_COORDINATES_DEC_ITEM_NAME);
		if (ra_item && dec_item) {
			g_ra  = ra_item->number.value;
			g_dec = dec_item->number.value;
			g_got_equatorial = true;
		}
	} else if (strcmp(property->name, MOUNT_HORIZONTAL_COORDINATES_PROPERTY_NAME) == 0) {
		indigo_item *alt_item = indigo_get_item(property, MOUNT_HORIZONTAL_COORDINATES_ALT_ITEM_NAME);
		indigo_item *az_item  = indigo_get_item(property, MOUNT_HORIZONTAL_COORDINATES_AZ_ITEM_NAME);
		if (alt_item && az_item) {
			g_alt = alt_item->number.value;
			g_az  = az_item->number.value;
			g_got_horizontal = true;
		}
	} else if (strcmp(property->name, MOUNT_SIDE_OF_PIER_PROPERTY_NAME) == 0) {
		g_pier_east        = indigo_get_switch(property, MOUNT_SIDE_OF_PIER_EAST_ITEM_NAME);
		g_got_side_of_pier = true;
	} else if (strcmp(property->name, MOUNT_TRACKING_PROPERTY_NAME) == 0) {
		g_tracking_on  = indigo_get_switch(property, MOUNT_TRACKING_ON_ITEM_NAME);
		g_got_tracking = true;
	}

	/* Done once we have at least equatorial coordinates */
	if (g_got_equatorial)
		g_finished = true;
}

static indigo_result client_define_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
	(void)client;
	(void)device;
	(void)message;
	process_property(property);
	return INDIGO_OK;
}

static indigo_result client_update_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
	(void)client;
	(void)device;
	(void)message;
	process_property(property);
	return INDIGO_OK;
}

static indigo_result client_detach(indigo_client *client) {
	(void)client;
	return INDIGO_OK;
}

static indigo_client g_client = {
	.name = "indigo_mount_position",
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

/* ------------------------------------------------------------------ */
/* Minimal INI config parser (indigo_server/port/servername/mount)  */
/* ------------------------------------------------------------------ */

static void parse_config_file(const char *path,
                               char *srv_buf, int srv_size,
                               int  *port_out,
                               char *mnt_buf, int mnt_size,
                               bool *mnt_set) {
	FILE *f = fopen(path, "r");
	if (!f) {
		fprintf(stderr, "Warning: cannot open config file: %s\n", path);
		return;
	}
	char line[512];
	while (fgets(line, sizeof(line), f)) {
		char *p = line;
		while (*p == ' ' || *p == '\t') p++;
		if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
		char *eq = strchr(p, '=');
		if (!eq) continue;
		*eq = '\0';
		char *key = p;
		char *val = eq + 1;
		/* trim key trailing whitespace */
		char *ke = key + strlen(key) - 1;
		while (ke > key && (*ke == ' ' || *ke == '\t')) *ke-- = '\0';
		/* trim val leading whitespace */
		while (*val == ' ' || *val == '\t') val++;
		/* trim val trailing whitespace + newline */
		char *ve = val + strlen(val) - 1;
		while (ve >= val && (*ve == '\n' || *ve == '\r' || *ve == ' ' || *ve == '\t')) *ve-- = '\0';
		/* strip surrounding quotes from val */
		if (*val == '"') {
			val++;
			ve = val + strlen(val) - 1;
			if (ve >= val && *ve == '"') *ve = '\0';
		}
		if (strcmp(key, "indigo_server") == 0)
			snprintf(srv_buf, srv_size, "%s", val);
		else if (strcmp(key, "indigo_port") == 0 && atoi(val) > 0)
			*port_out = atoi(val);
		else if (strcmp(key, "indigo_mount") == 0 && *val) {
			snprintf(mnt_buf, mnt_size, "%s", val);
			*mnt_set = true;
		}
		else if (strcmp(key, "indigo_servername") == 0 && *val)
			snprintf(g_server_name, sizeof(g_server_name), "%s", val);
	}
	fclose(f);
}

/* ------------------------------------------------------------------ */
/* Status file without a position (-o only)                            */
/* ------------------------------------------------------------------ */

/* Writes {"timestamp","status","mount",...: null} atomically to path, so the
 * web page replaces the last position instead of showing it as current */
static int write_status_file(const char *path, const char *status, time_t now) {
	char tmp[PATH_MAX];
	snprintf(tmp, sizeof(tmp), "%s.tmp", path);
	FILE *f = fopen(tmp, "w");
	if (!f) {
		fprintf(stderr, "Error: cannot open output file: %s\n", tmp);
		return -1;
	}
	char esc_mount[512];
	json_escape(g_mount_name, esc_mount, sizeof(esc_mount));
	fprintf(f, "{\n");
	fprintf(f, "  \"timestamp\": %lld,\n", (long long)now);
	fprintf(f, "  \"status\": \"%s\",\n", status);
	if (strcmp(status, "not_configured") == 0)
		fprintf(f, "  \"mount\": null,\n");
	else
		fprintf(f, "  \"mount\": \"%s\",\n", esc_mount);
	fprintf(f, "  \"equatorial\": null,\n");
	fprintf(f, "  \"horizontal\": null,\n");
	fprintf(f, "  \"side_of_pier\": null,\n");
	fprintf(f, "  \"tracking\": null\n");
	fprintf(f, "}\n");
	if (fclose(f) != 0 || rename(tmp, path) != 0) {
		fprintf(stderr, "Error: cannot write %s\n", path);
		remove(tmp);
		return -1;
	}
	return 0;
}

/* ------------------------------------------------------------------ */
/* Usage / main                                                        */
/* ------------------------------------------------------------------ */

static void print_usage(const char *prog) {
	fprintf(stderr, "Usage: %s [-c <config>] [-h <server>] [-p <port>] [-m <mount_name>] [-t <secs>] [-l] [-j] [-o <file>]\n", prog);
	fprintf(stderr, "  -c / --config      <file>       allsky360 config.ini (provides server/port/mount defaults)\n");
	fprintf(stderr, "  -h / --host        <server>     hostname or IP of the INDIGO server\n");
	fprintf(stderr, "  -p / --port        <port>       TCP port (default: 7624)\n");
	fprintf(stderr, "  -m / --mount       <mount_name> device name (default: \"Mount LX200\")\n");
	fprintf(stderr, "  -t / --timeout     <secs>       timeout in seconds (default: 10)\n");
	fprintf(stderr, "  -l / --list-mounts              list available mounts and exit\n");
	fprintf(stderr, "  -j / --json                     output as JSON\n");
	fprintf(stderr, "  -o / --output      <file>       write JSON output to file (implies -j)\n");
}

int main(int argc, const char *argv[]) {
	indigo_main_argc = argc;
	indigo_main_argv = argv;

	/* Sentinel values: NULL / -1 / false = "not set on CLI" */
	const char *cli_server  = NULL;
	int         cli_port    = -1;
	const char *cli_mount   = NULL;
	const char *config_path = NULL;
	const char *output_path = NULL;
	int timeout = 10;

	static struct option long_opts[] = {
		{ "config",      required_argument, NULL, 'c' },
		{ "host",        required_argument, NULL, 'h' },
		{ "port",        required_argument, NULL, 'p' },
		{ "mount",       required_argument, NULL, 'm' },
		{ "timeout",     required_argument, NULL, 't' },
		{ "list-mounts", no_argument,       NULL, 'l' },
		{ "json",        no_argument,       NULL, 'j' },
		{ "output",      required_argument, NULL, 'o' },
		{ NULL, 0, NULL, 0 }
	};

	int opt;
	while ((opt = getopt_long(argc, (char *const *)argv, "c:h:p:m:t:ljo:", long_opts, NULL)) != -1) {
		switch (opt) {
			case 'c':
				config_path = optarg;
				break;
			case 'h':
				cli_server = optarg;
				break;
			case 'p':
				cli_port = atoi(optarg);
				if (cli_port <= 0 || cli_port > 65535) {
					fprintf(stderr, "Invalid port: %s\n", optarg);
					return 1;
				}
				break;
			case 'm':
				cli_mount = optarg;
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
			case 'o':
				output_path = optarg;
				g_json_mode = true; /* -o implies -j */
				break;
			default:
				print_usage(argv[0]);
				return 1;
		}
	}

	/* Effective connection parameters: start with defaults */
	static char srv_buf[256] = "";
	int port = 7624;

	/* Apply config file values as defaults (CLI args override below) */
	bool mount_configured = false;
	if (config_path) {
		parse_config_file(config_path, srv_buf, sizeof(srv_buf), &port, g_mount_name, sizeof(g_mount_name), &mount_configured);
	}

	/* CLI args override config file */
	const char *server = cli_server ? cli_server : (srv_buf[0] ? srv_buf : NULL);
	if (cli_port > 0) port = cli_port;
	if (cli_mount) {
		strncpy(g_mount_name, cli_mount, sizeof(g_mount_name) - 1);
		g_mount_name[sizeof(g_mount_name) - 1] = '\0';
	}

	if (!server) {
		fprintf(stderr, "Error: INDIGO server not specified. Use --host or --config.\n\n");
		print_usage(argv[0]);
		return 1;
	}

	time_t now = time(NULL);

	/* indigo_mount empty in config.ini = no mount: nothing to ask, and no
	 * failed service run every minute */
	if (config_path && !mount_configured && !cli_mount && !g_list_mode) {
		fprintf(stderr, "No mount configured (indigo_mount is empty in %s)\n", config_path);
		if (output_path && write_status_file(output_path, "not_configured", now) != 0)
			return 1;
		return 0;
	}

	indigo_set_log_level(INDIGO_LOG_DEBUG);
	if (indigo_start() != INDIGO_OK) {
		fprintf(stderr, "Failed to start INDIGO.\n");
		return 2;
	}

	indigo_server_entry *server_entry;
	static bool connection_state = false;
	indigo_attach_client(&g_client);
	indigo_result rc = indigo_connect_server(g_server_name, server, port, &server_entry, &connection_state);
	if (rc != INDIGO_OK) {
		fprintf(stderr, "Failed to connect to %s:%d (rc=%d)\n", server, port, rc);
		indigo_detach_client(&g_client);
		indigo_stop();
		if (output_path && !g_list_mode)
			write_status_file(output_path, "server_unreachable", now);
		return 1;
	}

	/* indigo_connect_server() returns before the TCP connection is up and
	 * does not report a refused connection: the server was reachable if
	 * connection_state was true at some point while waiting */
	bool server_answered = false;

	if (g_list_mode) {
		/* Collect INFO properties from all devices for 3 seconds */
		for (int i = 0; i < 30; i++)
			indigo_usleep(100000);
	} else {
		/* Wait for equatorial coordinates up to the configured timeout */
		int timeout_ms = timeout * 1000;
		while (!g_finished && timeout_ms > 0) {
			indigo_usleep(100000); /* 100 ms */
			timeout_ms -= 100;
			if (connection_state)
				server_answered = true;
		}
		if (connection_state || g_finished)
			server_answered = true;

		/* Once equatorial coords are in, wait briefly for the optional
		   properties: horizontal, side-of-pier, tracking (max 1 s) */
		if (g_got_equatorial) {
			int wait_ms = 1000;
			while (wait_ms > 0 &&
			       (!g_got_horizontal || !g_got_side_of_pier || !g_got_tracking)) {
				indigo_usleep(100000);
				wait_ms -= 100;
			}
		}
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

	/* Redirect stdout to file if -o was given (atomic via tmp+rename) */
	char output_tmp[PATH_MAX] = "";
	FILE *out_fp = stdout;
	if (output_path) {
		snprintf(output_tmp, sizeof(output_tmp), "%s.tmp", output_path);
		out_fp = fopen(output_tmp, "w");
		if (!out_fp) {
			fprintf(stderr, "Error: cannot open output file: %s\n", output_tmp);
			return 1;
		}
	}

#define OUTF(...) fprintf(out_fp, __VA_ARGS__)

	if (g_list_mode) {
		if (g_found_count == 0) {
			if (g_json_mode)
				OUTF("{\"server\":\"%s\",\"port\":%d,\"mounts\":[]}\n", server, port);
			else
				fprintf(stderr, "No mounts found on %s:%d\n", server, port);
			if (out_fp != stdout) { fclose(out_fp); rename(output_tmp, output_path); }
			return 2;
		}
		if (g_json_mode) {
			char esc[512];
			OUTF("{\"server\":\"%s\",\"port\":%d,\"mounts\":[", server, port);
			for (int i = 0; i < g_found_count; i++) {
				json_escape(g_found_mounts[i], esc, sizeof(esc));
				OUTF("%s\"%s\"", i > 0 ? "," : "", esc);
			}
			OUTF("]}\n");
		} else {
			OUTF("Available mounts on %s:%d:\n", server, port);
			for (int i = 0; i < g_found_count; i++)
				OUTF("  %s\n", g_found_mounts[i]);
		}
		if (out_fp != stdout) { fclose(out_fp); rename(output_tmp, output_path); }
		return 0;
	}

	if (!g_got_equatorial) {
		const char *status;
		int ret = 2;
		if (g_mount_disconnected) {
			status = "disconnected";
			fprintf(stderr, "Mount \"%s\" on %s:%d is not connected (no client has connected it)\n",
			        g_mount_name, server, port);
		} else if (!server_answered) {
			status = "server_unreachable";
			ret = 1;
			fprintf(stderr, "Error: INDIGO server %s:%d not reachable\n", server, port);
		} else {
			status = "no_response";
			fprintf(stderr, "Error: no response from mount \"%s\" on %s:%d\n",
			        g_mount_name, server, port);
			fprintf(stderr, "Tip: list the mount names with indigo_cameras -h %s -p %d -M\n", server, port);
		}
		if (out_fp != stdout) {
			fclose(out_fp);
			remove(output_tmp);
			if (write_status_file(output_path, status, now) != 0)
				return 1;
		}
		return ret;
	}

	char ra_str[32], dec_str[32];
	format_ra(g_ra, ra_str, sizeof(ra_str));
	format_dec(g_dec, dec_str, sizeof(dec_str));

	if (g_json_mode) {
		char esc_mount[512], esc_dec[64];
		json_escape(g_mount_name, esc_mount, sizeof(esc_mount));
		json_escape(dec_str, esc_dec, sizeof(esc_dec));
		OUTF("{\n");
		OUTF("  \"timestamp\": %lld,\n", (long long)now);
		OUTF("  \"status\": \"ok\",\n");
		OUTF("  \"mount\": \"%s\",\n", esc_mount);
		OUTF("  \"equatorial\": {\n");
		OUTF("    \"ra_h\": %.8f,\n", g_ra);
		OUTF("    \"ra_str\": \"%s\",\n", ra_str);
		OUTF("    \"dec_deg\": %.8f,\n", g_dec);
		OUTF("    \"dec_str\": \"%s\"\n", esc_dec);
		OUTF("  },\n");
		if (g_got_horizontal) {
			OUTF("  \"horizontal\": {\n");
			OUTF("    \"alt_deg\": %.8f,\n", g_alt);
			OUTF("    \"az_deg\": %.8f\n", g_az);
			OUTF("  },\n");
		} else {
			OUTF("  \"horizontal\": null,\n");
		}
		if (g_got_side_of_pier)
			OUTF("  \"side_of_pier\": \"%s\",\n", g_pier_east ? "east" : "west");
		else
			OUTF("  \"side_of_pier\": null,\n");
		if (g_got_tracking)
			OUTF("  \"tracking\": %s\n", g_tracking_on ? "true" : "false");
		else
			OUTF("  \"tracking\": null\n");
		OUTF("}\n");
	} else {
		OUTF("Mount:         %s\n", g_mount_name);
		OUTF("Equatorial:    RA  %s  (%.6f h)\n", ra_str, g_ra);
		OUTF("               DEC %s  (%.6f °)\n", dec_str, g_dec);
		if (g_got_horizontal)
			OUTF("Horizontal:    ALT %.4f °   AZ %.4f °\n", g_alt, g_az);
		else
			OUTF("Horizontal:    not available\n");
		if (g_got_side_of_pier)
			OUTF("Side of pier:  %s\n", g_pier_east ? "East" : "West");
		else
			OUTF("Side of pier:  not available\n");
		if (g_got_tracking)
			OUTF("Tracking:      %s\n", g_tracking_on ? "On" : "Off");
		else
			OUTF("Tracking:      not available\n");
	}

	if (out_fp != stdout) {
		fclose(out_fp);
		if (rename(output_tmp, output_path) != 0) {
			fprintf(stderr, "Error: cannot rename %s -> %s\n", output_tmp, output_path);
			return 1;
		}
	}

#undef OUTF

	return 0;
}
