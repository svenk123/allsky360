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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <inttypes.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_client.h>
#include <indigo/indigo_version.h>
#include <indigo/indigo_ccd_driver.h> // CCD_INFO_PROPERTY_NAME

#define PROG_VERSION "1.0"

#define DEFAULT_HOST "127.0.0.1"
#define DEFAULT_PORT 7624
#define DEFAULT_TIMEOUT_MS 5000

typedef struct camera_entry {
  char name[INDIGO_NAME_SIZE];
  indigo_property_state state; // last known CCD_INFO state
  struct camera_entry *next;
} camera_entry;

volatile sig_atomic_t running = 1;
int json;
int timeout_ms;
char host[256];
int port;
camera_entry *cams;
indigo_server_entry *server = NULL;

// --- linked list helpers ---
static camera_entry *find_cam(const char *name) {
  for (camera_entry *p = cams; p; p = p->next) {
    if (strcmp(p->name, name) == 0) 
      return p;
  }

  return NULL;
}

static camera_entry *add_cam(const char *name) {
  camera_entry *e = find_cam(name);
  if (e) 
    return e;
  
  e = (camera_entry*)calloc(1, sizeof(camera_entry));
  if (!e) 
    return NULL;
  
  strncpy(e->name, name, sizeof(e->name)-1);
  e->state = INDIGO_IDLE_STATE;
  e->next = cams;
  cams = e;
  
  return e;
}

static void clear_cams(void) {
  camera_entry *p = cams;
  
  while (p) { 
    camera_entry *n = p->next; 
    free(p); 
    p = n; 
  }

  cams = NULL;
}

// --- signal handler ---
static void cleanup_resources()
{
	if (server)
		indigo_disconnect_server(server);

	indigo_stop();

}

void cleanup_and_exit(int signum)
{
	printf("\nReceived signal %d, cleaning up...\n", signum);

	running = 0;

	cleanup_resources();

	if (signum == SIGINT || signum == SIGTERM)
	{
		exit(EXIT_SUCCESS);
	}
}

void setup_signal_handlers()
{
	struct sigaction sa;
	sa.sa_handler = cleanup_and_exit;
	sa.sa_flags = SA_RESTART; // Restart interrupted system calls
	sigemptyset(&sa.sa_mask);

	sigaction(SIGINT, &sa, NULL);  // CTRL+C
	sigaction(SIGTERM, &sa, NULL); // Normale termination (kill <pid>)
}

// --- state to string ---
static const char* state_str(indigo_property_state s) {
  switch (s) {
    case INDIGO_OK_STATE: 
      return "OK";
    case INDIGO_BUSY_STATE: 
      return "BUSY";
    case INDIGO_ALERT_STATE: 
      return "ALERT";
    case INDIGO_IDLE_STATE: 
      return "IDLE";
    default: 
      return "UNKNOWN";
  }
}

// Check if the property is a CCD_INFO property
static int is_ccd_info(indigo_property *property) {
  if (!property || strcmp(property->name, "INFO") != 0) 
    return 0;

  // Search for the item "DEVICE_INTERFACE"
  for (int i = 0; i < property->count; i++) {
    indigo_item *it = &property->items[i];
    
    if (!strcmp(it->name, "DEVICE_INTERFACE")) {
      uint32_t iface = (uint32_t)it->number.value;

      // Is CCD-Bit set?
      return (iface & INDIGO_INTERFACE_CCD) != 0;
    }
  }
  return 0;
}

// --- INDIGO client callbacks ---
static indigo_result client_attach(indigo_client *client) {
  indigo_log("attached to INDIGO bus...");
  indigo_enumerate_properties(client, NULL);

  return INDIGO_OK;
}
static indigo_result client_detach(indigo_client *client) {
  indigo_log("detached from INDIGO bus...");

  return INDIGO_OK;
}
static indigo_result client_define_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
  (void)client; 
  (void)device; 
  (void)message;

  // 1) First INFO: also lists "inactive" cameras
  if (is_ccd_info(property)) {
    camera_entry *e = add_cam(property->device);
    if (e)
      e->state = property->state; // INFO-State (typically IDLE)

    return INDIGO_OK;
  }

  // 2) Optional: If CCD_INFO comes later/updated, get the state
  if (property && strcmp(property->name, CCD_INFO_PROPERTY_NAME) == 0) {
    camera_entry *e = add_cam(property->device);
    if (e) 
      e->state = property->state;
  }

  return INDIGO_OK;
}

static indigo_result client_update_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
  (void)client; 
  (void)device; 
  (void)message;

  // If INFO comes again, or CCD_INFO is updated → update the state
  if (is_ccd_info(property) || (property && strcmp(property->name, CCD_INFO_PROPERTY_NAME) == 0)) {
    camera_entry *e = add_cam(property->device);
    if (e)
      e->state = property->state;
  }

  return INDIGO_OK;
}
static indigo_result client_delete_property(indigo_client *client, indigo_device *device, indigo_property *property, const char *message) {
  (void)client;
  (void)device;
  (void)property;
  (void)message;

  return INDIGO_OK;
}

// --- output ---
static void print_result(int json, int timeout_ms, char *host, int port, camera_entry *cams) {
  if (json) {
    printf("{\"host\":\"%s\",\"port\":%d,\"timeout_ms\":%d,\"cameras\":[",
           host, port, timeout_ms);
    int first = 1;
 
    for (camera_entry *p = cams; p; p = p->next) {
      if (!first)
        printf(",");

      printf("{\"name\":\"%s\",\"state\":\"%s\"}", p->name, state_str(p->state));
      first = 0;
    }
    printf("]}\n");
  } else {
    int count = 0;

    for (camera_entry *p = cams; p; p = p->next)
      count++;

    printf("Found %d camera(s):\n", count);
    for (camera_entry *p = cams; p; p = p->next)
      printf("- Device: %s (state: %s)\n", p->name, state_str(p->state));
  }
}

static void usage(const char *argv0) {
  fprintf(stderr,
    "indigo-list-cameras v%s\n"
    "Copyright (c) 2025 Sven Kreiensen\n\n"
    "Usage: %s [--host HOST] [--port PORT] [--timeout MS] [--json]\n"
    "Defaults: --host %s --port %d --timeout %d\n",
    PROG_VERSION, argv0, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT_MS);
}

int main(int argc, char **argv) {
  running = 1;
  strncpy(host, DEFAULT_HOST, sizeof(host)-1);
  port = DEFAULT_PORT;
  timeout_ms = DEFAULT_TIMEOUT_MS;
  json = 0;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--host") && i+1 < argc) {
      strncpy(host, argv[++i], sizeof(host)-1);
    }
    else if (!strcmp(argv[i], "--port") && i+1 < argc) {
      port = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "--timeout") && i+1 < argc) {
      timeout_ms = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "--json")) {
      json = 1;
    }
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { 
      usage(argv[0]); 
      return 0;
    }
    else { 
      fprintf(stderr, "Unknown arg: %s\n", argv[i]); 
      usage(argv[0]); 
      return 2;
    }
  }

  setup_signal_handlers();
  indigo_set_log_level(INDIGO_LOG_INFO);

  static indigo_client client = {
    .name = "indigo_list_cameras",
    .client_context = NULL,
    .version = INDIGO_VERSION_CURRENT,
    .attach = client_attach,
    .detach = client_detach,
    .define_property = client_define_property,
    .delete_property = client_delete_property,
    .update_property = client_update_property
  };

  if (indigo_start() != INDIGO_OK) {
    fprintf(stderr, "Failed to start INDIGO.\n");
    return 2;
  }

  if (indigo_attach_client(&client) != INDIGO_OK) {
    fprintf(stderr, "Failed to attach INDIGO client.\n");
    indigo_stop();
    return 2;
  }

  indigo_result rc = indigo_connect_server("remote", host, port, NULL);
  if (rc != INDIGO_OK) {
    fprintf(stderr, "Failed to connect to %s:%d (rc=%d)\n", host, port, rc);
    indigo_detach_client(&client);
    indigo_stop();
    return 1;
  }

  // IMPORTANT: enumerate properties again after successful connect
  //indigo_enumerate_properties(&client, NULL);
  
  // Wait for properties to arrive
  const int step_ms = 50;
  int waited = 0;
  while (running && waited < timeout_ms) {
    indigo_usleep(step_ms * 1000);
    waited += step_ms;
  }

  print_result(json, timeout_ms, host, port, cams);

//  if (server)
    indigo_disconnect_server(server);

  indigo_detach_client(&client);

  indigo_stop();
  clear_cams();
  return 0;
}
