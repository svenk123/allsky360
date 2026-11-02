/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
/*
 * Simple INDIGO client: query mount equatorial coordinates (RA/DEC)
 * and print them as JSON to stdout.
 *
 * Usage:
 *   ./get_mount_pos <host> <port>
 *
 * Example:
 *   ./get_mount_pos localhost 7624
 *
 * Only JSON is printed to stdout, errors go to stderr.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdbool.h>
 #include <string.h>
 #include <time.h>
 #include <math.h>
 
 #include <indigo/indigo_bus.h>
 #include <indigo/indigo_client.h>
 #include <indigo/indigo_names.h>
 
 /* Property and item names used by INDIGO mount drivers */
 #define EQ_COORDS_PROPERTY_NAME        "MOUNT_EQUATORIAL_COORDINATES"
 #define EQ_COORDS_RA_ITEM_NAME         "MOUNT_EQUATORIAL_COORDINATES_RA"
 #define EQ_COORDS_DEC_ITEM_NAME        "MOUNT_EQUATORIAL_COORDINATES_DEC"
 
 /* Timeout in seconds for waiting RA/DEC */
 #define TIMEOUT_SECONDS 5.0
 
 /* Global state guarded only by simple rules:
  * INDIGO calls callbacks from its own worker threads, main thread only reads.
  * For this small tool, we accept the tiny race; once coords_received is set,
  * values are treated as final.
  */
 static volatile bool coords_received = false;
 static double global_ra = NAN;   /* decimal hours */
 static double global_dec = NAN;  /* degrees */
 static char   mount_device_name[INDIGO_NAME_SIZE] = {0};
 
 /* Forward declaration of client callbacks */
 static indigo_result client_attach(indigo_client *client);
 static indigo_result client_detach(indigo_client *client);
 static indigo_result client_define_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message);
 static indigo_result client_update_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message);
 static indigo_result client_delete_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message);
 static indigo_result client_send_message(indigo_client *client, indigo_device *device, indigo_property *property, const char *message);
 
 /* Designated initializers avoid depending on the struct layout */
 static indigo_client mount_client = {
   .name = "get_mount_pos_client",
   .is_remote = false,
   .client_context = NULL,
   .last_result = INDIGO_OK,
   .version = INDIGO_VERSION_CURRENT,
   .enable_blob_mode_records = NULL,
   .attach = client_attach,
   .define_property = client_define_property,
   .update_property = client_update_property,
   .delete_property = client_delete_property,
   .send_message = client_send_message,
   .detach = client_detach,
 };
 
 /* Helper: extract RA/DEC from a property, if present */
 static void handle_equatorial_property(indigo_property *property) {
   if (coords_received)
     return;
 
   if (strcmp(property->name, EQ_COORDS_PROPERTY_NAME) != 0)
     return;
 
   double ra = NAN;
   double dec = NAN;
 
   for (int i = 0; i < property->count; i++) {
     indigo_item *item = &property->items[i];
     if (strcmp(item->name, EQ_COORDS_RA_ITEM_NAME) == 0) {
       ra = item->number.value;
     } else if (strcmp(item->name, EQ_COORDS_DEC_ITEM_NAME) == 0) {
       dec = item->number.value;
     }
   }
 
   if (!isnan(ra) && !isnan(dec)) {
     global_ra = ra;
     global_dec = dec;
     coords_received = true;
   }
 }
 
 /* Client callback: called when client is attached to INDIGO bus */
 static indigo_result client_attach(indigo_client *client) {
   /* Request all properties; BLOBs are disabled by default for clients. */
   indigo_enumerate_properties(client, &INDIGO_ALL_PROPERTIES);
   return INDIGO_OK;
 }
 
 /* Client callback: called when client is detached */
 static indigo_result client_detach(indigo_client *client) {
   (void)client;
   return INDIGO_OK;
 }
 
 /* Called when a new property appears on the bus */
 static indigo_result client_define_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
   (void)client;
   (void)device;
   (void)message;
 
   /* Remember mount device name when we first see its equatorial coordinates. */
   if (!coords_received && strcmp(property->name, EQ_COORDS_PROPERTY_NAME) == 0) {
     if (mount_device_name[0] == '\0') {
       strncpy(mount_device_name, property->device, sizeof(mount_device_name) - 1);
       mount_device_name[sizeof(mount_device_name) - 1] = '\0';
 
       /* Ensure device is connected (idempotent if already connected). */
       indigo_device_connect(&mount_client, mount_device_name);
     }
     handle_equatorial_property(property);
   }
 
   return INDIGO_OK;
 }
 
 /* Called when an existing property is updated */
 static indigo_result client_update_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
   (void)client;
   (void)device;
   (void)message;
 
   handle_equatorial_property(property);
   return INDIGO_OK;
 }
 
 /* Called when a property is deleted (not relevant here) */
 static indigo_result client_delete_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
   (void)client;
   (void)device;
   (void)property;
   (void)message;
   return INDIGO_OK;
 }
 
 /* Called when a message is sent from a device to client (ignored) */
 static indigo_result client_send_message(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
   (void)client;
   (void)device;
   (void)property;
   (void)message;
   return INDIGO_OK;
 }
 
 /* Simple monotonic time helper (seconds as double) */
 static double now_seconds(void) {
   struct timespec ts;
 #if defined(CLOCK_MONOTONIC)
   clock_gettime(CLOCK_MONOTONIC, &ts);
 #else
   clock_gettime(CLOCK_REALTIME, &ts);
 #endif
   return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
 }
 
 int main(int argc, char **argv) {
   if (argc != 3) {
     fprintf(stderr, "Usage: %s <host> <port>\n", argv[0]);
     return 1;
   }
 
   const char *host = argv[1];
   int port = atoi(argv[2]);
   if (port <= 0 || port > 65535) {
     fprintf(stderr, "Error: invalid port: %s\n", argv[2]);
     return 1;
   }
 
   if (indigo_start() != INDIGO_OK) {
     fprintf(stderr, "Error: indigo_start() failed\n");
     return 1;
   }
 
   if (indigo_attach_client(&mount_client) != INDIGO_OK) {
     fprintf(stderr, "Error: indigo_attach_client() failed\n");
     indigo_stop();
     return 1;
   }
 
   /* Connect to remote INDIGO server (host:port) */
   indigo_server_entry *server = NULL;
   static bool connection_state = false;
   if (indigo_connect_server("indigo_server", /* same name as the camera daemon (indigo_servername) */ host, port, &server, &connection_state) != INDIGO_OK) {
     fprintf(stderr, "Error: indigo_connect_server(%s,%d) failed\n", host, port);
     indigo_detach_client(&mount_client);
     indigo_stop();
     return 1;
   }
 
   /* Wait for RA/DEC for up to TIMEOUT_SECONDS */
   double start = now_seconds();
   while (!coords_received) {
     double elapsed = now_seconds() - start;
     if (elapsed >= TIMEOUT_SECONDS)
       break;
     /* Sleep a bit to avoid busy spinning; INDIGO uses its own worker threads. */
     indigo_usleep(100000); /* 100 ms */
   }
 
   int exit_code = 0;
 
   if (!coords_received) {
     fprintf(stderr, "Error: timed out (%d s) waiting for mount coordinates\n", (int)TIMEOUT_SECONDS);
     exit_code = 1;
   } else {
     /* Only JSON to stdout */
     printf("{\"ra\": %.8f, \"dec\": %.8f}\n", global_ra, global_dec);
     fflush(stdout);
   }
 
   if (server)
     indigo_disconnect_server(server);
   indigo_detach_client(&mount_client);
   indigo_stop();
 
   return exit_code;
 }
