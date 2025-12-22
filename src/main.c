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
#include <cairo.h>
#include <fcntl.h>
#include <json-c/json.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>

#include <indigo/indigo_bus.h>
#include <indigo/indigo_ccd_driver.h>
#include <indigo/indigo_client.h>

#include "acdnr_filter.h"
#include "ai_training.h"
#include "allsky.h"
#include "allsky_panorama.h"
#include "autoexposure.h"
#include "clarity_filter.h"
#include "color_calibration.h"
#include "compute_noise.h"
#include "config.h"
#include "daily_database.h"
#include "debug_pipeline.h"
#include "dehaze_filter.h"
#include "focus_measure.h"
#include "gpu_functions.h"
#include "hdr_merge.h"
#include "hdrmt.h"
#include "image_metadata.h"
#include "image_overlay.h"
#include "indigo_raw_to_rgbf.h"
#include "jpeg_functions.h"
#include "jpeg_to_cairo.h"
#include "lights_and_shadows.h"
#include "mask_image.h"
#include "median_filter.h"
#include "moon_calc.h"
#include "png_to_cairo.h"
#include "png_to_rgbf.h"
#include "rgb16_to_cairo.h"
#include "rotate_image.h"
#include "scnr_filter.h"
#include "star_detector.h"
#include "sun_calc.h"
#include "thumbnail.h"
#include "wavelet_sharpen.h"
#include "white_balance.h"

/******* Global variables and constants **********/

int allsky_debug;

#define MAX_FRAMES 5

struct HdrFrames frames[MAX_FRAMES];

float *hdr_image = NULL;
size_t hdr_image_size = 0;
int debug_pipeline_images = 0;

// #define AI_TRAINING 0

#define MIN_EXPOSURE 0.000032

/******* Global indigo variables **********/

indigo_server_entry *server = NULL;
char *indigo_camera = NULL;
double exposure_sec = 0.0;
double gain = 0;
static bool connected = false;
static int count = 1;
unsigned char *raw_buffer = NULL;
size_t raw_buffer_size = 0;

/**
 * Configure 2x2 binning on an already connected camera
 * @param binning_val: 0 for 1x1 binning, 1 for 2x2 binning
 * @param client: indigo_client
 * @param device_name: name of the device to configure
 */
void set_camera_binning(indigo_client *client, const char *device_name,
                        int binning_val) {
  /* Allocate a property for CCD_BIN */
  indigo_property *binning = indigo_init_number_property(
      NULL,                  // no existing property
      device_name,           // the target device name
      CCD_BIN_PROPERTY_NAME, // "CCD_BIN"
      MAIN_GROUP,            // group name, often "Main" or "Image"
      CCD_BIN_PROPERTY_NAME, // label, can be same as name
      INDIGO_OK_STATE,       // initial state
      INDIGO_RW_PERM,        // read/write
      2                      // number of items (horizontal + vertical)
  );

  if (binning == NULL) {
    fprintf(stderr, "Failed to allocate CCD_BIN property\n");
    return;
  }

  if (binning_val > 1)
    return;

  binning_val++;

  /* Add the items: horizontal and vertical binning */
  indigo_init_number_item(&binning->items[0], CCD_BIN_HORIZONTAL_ITEM_NAME,
                          "Horizontal binning", 1, 4, 1,
                          binning_val); // min=1, max=4, step=1, value=2
  indigo_init_number_item(&binning->items[1], CCD_BIN_VERTICAL_ITEM_NAME,
                          "Vertical binning", 1, 4, 1, binning_val);

  /* Send property change request to the server */
  indigo_change_property(client, binning);

  printf("set_camera_binning: %.0fx%.0f\n", binning->items[0].number.value,
         binning->items[1].number.value);

  /* Clean up */
  indigo_release_property(binning);
}

/**
 *  Indigo client attach function
 * @param client: indigo_client
return: INDIGO_OK
 */
static indigo_result client_attach(indigo_client *client) {
  indigo_log("attached to INDIGO bus...");
  indigo_enumerate_properties(client, &INDIGO_ALL_PROPERTIES);
  return INDIGO_OK;
}

/**
 * Indigo client define property function
 * @param client: indigo_client
 * @param device: indigo_device
 * @param property: indigo_property
 * @param message: message
 * @return: INDIGO_OK
 */
static indigo_result client_define_property(indigo_client *client,
                                            indigo_device *device,
                                            indigo_property *property,
                                            const char *message) {
  (void)message;
  // printf("client_define_property(device = %s, name = %s)\n",
  // property->device, property->name);

  if (strcmp(property->device, indigo_camera))
    return INDIGO_OK;

  if (!strcmp(property->name, CONNECTION_PROPERTY_NAME)) {
    if (indigo_get_switch(property, CONNECTION_CONNECTED_ITEM_NAME)) {
      connected = true;
      indigo_log("already connected...");

      static const char *gain_items[] = {CCD_GAIN_ITEM_NAME};
      double gain_values[] = {gain};
      indigo_change_number_property(client, indigo_camera,
                                    CCD_GAIN_PROPERTY_NAME, 1, gain_items,
                                    gain_values);

      static const char *bpp_items[] = {CCD_FRAME_BITS_PER_PIXEL_ITEM_NAME};
      double bpp_values[] = {16};
      indigo_change_number_property(client, indigo_camera,
                                    CCD_FRAME_PROPERTY_NAME, 1, bpp_items,
                                    bpp_values);

      static const char *exposure_items[] = {CCD_EXPOSURE_ITEM_NAME};
      double exposure_values[] = {exposure_sec};
      indigo_change_number_property(client, indigo_camera,
                                    CCD_EXPOSURE_PROPERTY_NAME, 1,
                                    exposure_items, exposure_values);
    } else {
      indigo_device_connect(client, indigo_camera);
      return INDIGO_OK;
    }
  }

  if (!strcmp(property->name, "FILE_NAME")) {
    char value[1024] = {0};
    static const char *items[] = {"PATH"};
    static const char *values[1];
    values[0] = value;
    for (int i = 0; i < 1023; i++)
      value[i] = '0' + i % 10;
    indigo_change_text_property(client, indigo_camera, "FILE_NAME", 1, items,
                                values);
  }
  if (!strcmp(property->name, CCD_IMAGE_PROPERTY_NAME)) {
    if (device->version >= INDIGO_VERSION_2_0)
      indigo_enable_blob(client, property, INDIGO_ENABLE_BLOB_URL);
    else
      indigo_enable_blob(client, property, INDIGO_ENABLE_BLOB_ALSO);
  }

  /* Gain */
  if (!strcmp(property->name, CCD_GAIN_PROPERTY_NAME)) {
    static const char *gain_items[] = {CCD_GAIN_ITEM_NAME};
    double gain_values[] = {gain};
    indigo_change_number_property(client, indigo_camera, CCD_GAIN_PROPERTY_NAME,
                                  1, gain_items, gain_values);
  }

  /* RAW */
  if (!strcmp(property->name, CCD_IMAGE_FORMAT_PROPERTY_NAME)) {
    static const char *items[] = {CCD_IMAGE_FORMAT_RAW_ITEM_NAME};
    static bool values[] = {true};
    indigo_change_switch_property(client, indigo_camera,
                                  CCD_IMAGE_FORMAT_PROPERTY_NAME, 1, items,
                                  values);
  }

  return INDIGO_OK;
}

/**
 *  Indigo client update property function
 * @param client: indigo_client
 * @param device: indigo_device
 * @param property: indigo_property
 * @param message: message
 * @return: INDIGO_OK
 */
static indigo_result client_update_property(indigo_client *client,
                                            indigo_device *device,
                                            indigo_property *property,
                                            const char *message) {
  (void)device;
  (void)message;
  // printf("client_update_property(device = %s, name = %d), %d\n",
  // property->device, property->name, strcmp(property->device, indigo_camera));

  if (strcmp(property->device, indigo_camera))
    return INDIGO_OK;

  /* Exposure */
  static const char *exposure_items[] = {CCD_EXPOSURE_ITEM_NAME};
  double exposure_values[] = {exposure_sec};

  /* Gain */
  static const char *gain_items[] = {CCD_GAIN_ITEM_NAME};
  double gain_values[] = {gain};

  if (!strcmp(property->name, CONNECTION_PROPERTY_NAME) &&
      property->state == INDIGO_OK_STATE) {
    if (indigo_get_switch(property, CONNECTION_CONNECTED_ITEM_NAME)) {
      if (!connected) {
        connected = true;
        indigo_log("connected...");

        indigo_change_number_property(client, indigo_camera,
                                      CCD_GAIN_PROPERTY_NAME, 1, gain_items,
                                      gain_values);
        indigo_change_number_property(client, indigo_camera,
                                      CCD_EXPOSURE_PROPERTY_NAME, 1,
                                      exposure_items, exposure_values);
      }
    } else {
      if (connected) {
        indigo_log("disconnected...");
        connected = false;
      }
    }
    return INDIGO_OK;
  }

  if (!strcmp(property->name, CCD_IMAGE_PROPERTY_NAME) &&
      property->state == INDIGO_OK_STATE) {
    /* URL blob transfer is available only in client - server setup.
       This will never be called in case of a client loading a driver. */
    if (*property->items[0].blob.url &&
        indigo_populate_http_blob_item(&property->items[0]))
      indigo_log("image URL received (%s, %d bytes)...",
                 property->items[0].blob.url, property->items[0].blob.size);

    if (property->items[0].blob.value) {
      /* Load raw image data into memory */
      size_t needed_raw_buffer_size = (size_t)property->items[0].blob.size;

      /* Check if target buffer size is large enough and reallocate if necessary
       */
      if (needed_raw_buffer_size > raw_buffer_size) {
        unsigned char *temp_raw_buffer = (unsigned char *)allsky_safe_realloc(
            raw_buffer, needed_raw_buffer_size);
        if (!temp_raw_buffer) {
          fprintf(stderr,
                  "realloc() failed (raw_buffer_size=%ld bytes, new_size=%ld "
                  "bytes)",
                  raw_buffer_size, needed_raw_buffer_size);
          allsky_safe_free(raw_buffer);
          raw_buffer = NULL;
          return INDIGO_OK;
        }

        raw_buffer = temp_raw_buffer;
      }
      raw_buffer_size = needed_raw_buffer_size;

      /* Copy image data to raw buffer */
      memcpy(raw_buffer, property->items[0].blob.value, raw_buffer_size);

      /* In case we have URL BLOB transfer we need to release the blob ourselves
       */
      if (*property->items[0].blob.url) {
        free(property->items[0].blob.value);
        property->items[0].blob.value = NULL;
      }
    }
  }
  if (!strcmp(property->name, CCD_EXPOSURE_PROPERTY_NAME)) {
    if (property->state == INDIGO_BUSY_STATE) {
      printf(".");
      fflush(stdout);
    } else if (property->state == INDIGO_OK_STATE) {
      printf("exposure done...\n");
      if (--count > 0) {
        indigo_change_number_property(client, indigo_camera,
                                      CCD_EXPOSURE_PROPERTY_NAME, 1,
                                      exposure_items, exposure_values);
      } else {
        //				indigo_device_disconnect(client,
        // indigo_camera);
      }
    }
    return INDIGO_OK;
  }
  return INDIGO_OK;
}

/**
 *  Indigo client detach function
 * @param client: indigo_client
 * @return: INDIGO_OK
 */
static indigo_result client_detach(indigo_client *client) {
  (void)client;
  indigo_log("detached from INDIGO bus...");
  return INDIGO_OK;
}

/* Indigo client client structure */
static indigo_client client = {
    "Allsky camera client", false, NULL,          INDIGO_OK,
    INDIGO_VERSION_CURRENT, NULL,  client_attach, client_define_property,
    client_update_property, NULL,  NULL,          client_detach};

/**
 *  Capture image function
 * @param exposure: exposure time
 * @param capture_timeout: capture timeout
 * @return: 0 if successful, 1 if failed
 */
static int capture_image(double exposure, int capture_timeout) {
  int ret = 0;

  exposure_sec = exposure;
  indigo_attach_client(&client);

  time_t exposure_sec_begin = time(NULL);

  /* Wait until capture is finished */
  while (count > 0) {
    indigo_usleep(ONE_SECOND_DELAY);

    time_t now_sec = time(NULL);
    if (difftime(now_sec, exposure_sec_begin) >= (exposure + capture_timeout)) {
      /* Exposure takes too long */
      printf("Capture timeout: elapsed %.0f seconds\n",
             difftime(now_sec, exposure_sec_begin));
      ret = 1;
      break;
    }
  }

  connected = false;
  count = 1;
  indigo_detach_client(&client);
  return ret;
}

/***** Signal handlers **********/

volatile sig_atomic_t keep_running =
    1; // Set to 0 when program should be terminated

/**
 *  Cleanup resources function
 */
static void cleanup_resources() {
  if (server) {
    indigo_device_disconnect(&client, indigo_camera);

    indigo_disconnect_server(server);
  }
  indigo_stop();

  if (raw_buffer)
    allsky_safe_free(raw_buffer);

  if (hdr_image)
    allsky_safe_free(hdr_image);

  for (int i = 0; i < MAX_FRAMES; i++) {
    allsky_safe_free(frames[i].image);
    frames[i].image = NULL;
  }
}

/**
 *  Cleanup and exit function
 * @param signum: signal number
 */
void cleanup_and_exit(int signum) {
  printf("\nReceived signal %d, cleaning up...\n", signum);

  keep_running = 0; // Stop main loop

  cleanup_resources();

  if (signum == SIGINT || signum == SIGTERM) {
    exit(EXIT_SUCCESS);
  }
}

/**
 *  Debug pipeline handler
 * @param signum: signal number
 */
void debug_pipeline_handler(int signum) {
  (void)signum; // Suppress unused parameter warning
  printf("\nReceived SIGUSR2, enabling debug pipeline images...\n");
  debug_pipeline_images = 1;
}

/* Setup signal handlers */
void setup_signal_handlers() {
  struct sigaction sa;
  sa.sa_handler = cleanup_and_exit;
  sa.sa_flags = SA_RESTART; // Restart interrupted system calls
  sigemptyset(&sa.sa_mask);

  sigaction(SIGINT, &sa, NULL);  // CTRL+C
  sigaction(SIGTERM, &sa, NULL); // Normale termination (kill <pid>)

  /* Setup SIGUSR2 handler for debug pipeline images */
  sa.sa_handler = debug_pipeline_handler;
  sigaction(SIGUSR2, &sa, NULL);
}

/************************************************/

/**
 *  Run end-of-night tasks asynchronously via systemd-run + sudo
 *
 * @return:
 *   0 = background job successfully spawned
 *   2 = fork error
 *   3 = waitpid error on intermediate child
 *   127 = execl failed
 */
int run_end_of_night_tasks(void) {
  pid_t pid = fork();
  if (pid < 0) {
    // First fork failed
    return 2;
  }

  if (pid > 0) {
    // Parent: wait for the first child (avoid zombie)
    int status;
    if (waitpid(pid, &status, 0) < 0) {
      return 3;
    }
    return 0; // parent returns immediately
  }

  // ========== FIRST CHILD PROCESS ==========
  // second fork to fully detach
  pid_t pid2 = fork();
  if (pid2 < 0) {
    _exit(2); // fork error
  }

  if (pid2 > 0) {
    // first child exits immediately
    _exit(0);
  }

  // ========== GRANDCHILD PROCESS (REAL WORKER) ==========
  // Fully detached now
  if (setsid() < 0) {
    // not fatal
  }

  // Close all file descriptors except stdin/out/err
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
    for (int fd = 3; fd < (int)rl.rlim_cur; fd++) {
      close(fd);
    }
  }

  // Redirect stdin/out/err to /dev/null
  int devnull = open("/dev/null", O_RDWR);
  if (devnull >= 0) {
    dup2(devnull, STDIN_FILENO);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    if (devnull > 2)
      close(devnull);
  }

  // Execute sudo/systemd-run
  execl("/usr/bin/sudo", "sudo", "/usr/bin/systemd-run",
        "--unit=allsky360-endofnight-%j",
        "--description=Allsky360 End of Night Tasks", "--collect",
        "/opt/allsky360/scripts/endofnight.sh", (char *)NULL);

  // If execl returns, something failed
  _exit(127);
}

/**
 * Run AI processing asynchronously via systemd-run + sudo.
 * 
 * This function:
 *  - validates the image_path,
 *  - double-forks to avoid zombies and detach from the parent,
 *  - redirects stdio to /dev/null in the final child,
 *  - executes:
 *      sudo /usr/bin/systemd-run --unit=ai-processing-%j \
 *           --description=Allsky360 AI Processing --collect \
 *           /opt/allsky360/scripts/run_ai_processing.sh <image_path>
 * 
 * @param image_path: path to the image
 * @return: 
 *   0 = background job successfully spawned
 *   1 = parameter/usage error
 *   2 = fork error
 *   3 = waitpid error on intermediate child
 *   127 = execl failed (something went wrong)
 */
int run_ai_processing(const char *image_path) {
  if (!image_path || image_path[0] == '\0') {
    // Parameter/usage error
    return 1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    // fork error
    return 2;
  }

  if (pid > 0) {
    // Parent: wait for first child so it does not become a zombie
    int status;
    pid_t w = waitpid(pid, &status, 0);
    if (w < 0) {
      // waitpid failed; not fatal for functionality, but report error
      return 3;
    }
    // Parent returns immediately; actual work is in grandchild
    return 0;
  }

  // First child process here
  // Do second fork to fully detach from the original parent
  pid_t pid2 = fork();
  if (pid2 < 0) {
    _exit(2); // fork error in child
  }

  if (pid2 > 0) {
    // First child exits immediately, grandchild continues
    _exit(0);
  }

  // Grandchild process here: fully detached worker
  // Optional: create new session to detach from controlling terminal
  if (setsid() < 0) {
    // Not fatal, continue anyway
  }

  // Close all inherited file descriptors except stdin/stdout/stderr
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
    for (int fd = 3; fd < (int)rl.rlim_cur; fd++) {
      close(fd);
    }
  }

  // Redirect stdin, stdout, stderr to /dev/null
  int devnull = open("/dev/null", O_RDWR);
  if (devnull >= 0) {
    dup2(devnull, STDIN_FILENO);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    if (devnull > 2) {
      close(devnull);
    }
  }

  // Execute sudo + systemd-run + script
  // Assumes sudoers allows this without password for the running user.
  execl("/usr/bin/sudo", "sudo", "/usr/bin/systemd-run",
        "--unit=ai-processing-%j", "--description=Allsky360 AI Processing",
        "--collect", "/opt/allsky360/scripts/run_ai_processing.sh", image_path,
        (char *)NULL);

  // If execl returns, something went wrong
  _exit(127);
}

/**
 *  Get timezone offset seconds
 * @param utc_timestamp: UTC timestamp
 * Positive values mean east of UTC (e.g. +3600 for CET, +7200 for CEST).
 * Negative values mean west of UTC (e.g. -18000 for US Eastern Standard Time).
 * Works on glibc / Linux.
 * @return: timezone offset in seconds
 */
long get_timezone_offset_seconds(time_t utc_timestamp) {
  struct tm local_tm;

  if (localtime_r(&utc_timestamp, &local_tm) == NULL)
    return 0; // fallback: assume UTC if conversion fails

#if defined(__GLIBC__)
  return (long)local_tm.tm_gmtoff;
#else
  // Portable fallback if tm_gmtoff not available
  struct tm gm_tm;
  gmtime_r(&utc_timestamp, &gm_tm);
  time_t local_epoch = mktime(&local_tm);
  time_t gm_epoch = mktime(&gm_tm);
  return (long)difftime(local_epoch, gm_epoch);
#endif
}

/* Compute exposure time for HDR bracketing.
 *
 * t0        = longest exposure time in seconds (float)
 * ev_steps  = EV distance between exposures (e.g. 1 or 2)
 * index     = image index: 0 = longest exposure, 1 = next shorter, etc.
 *
 * The formula:
 * t(index) = t0 / powf(2.0f, index * ev_steps)
 */

float calc_exposure_time(float t0, int ev_steps, int index) {
  if (index <= 0) {
    // index 0 = longest exposure → no reduction
    return t0;
  }

  float exponent = (float)(index * ev_steps);
  float factor = powf(2.0f, exponent);

  return t0 / factor;
}

/***** Main function **********/

int main(int argc, const char *argv[]) {
  int i = 0;
  const char *config_path = NULL;
  const char *images_dir = NULL;
  const char *database_dir = NULL;
  const char *img0 = NULL;
  const char *img1 = NULL;
  const char *img2 = NULL;
  const char *img3 = NULL;
  const char *img4 = NULL;
  config_t config;

  image_metadata_t image_metadata = {0};
  allsky_debug = ALLSKY_DEBUG_DISABLED;

  /* Parse command line parameters */
  for (i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0)) {
      if (i + 1 < argc) {
        config_path = argv[i + 1];
        i++;
      } else {
        fprintf(stderr, "Error: --config requires a file path\n");
        exit(1);
      }
    } else if ((strcmp(argv[i], "--start-exposure") == 0 ||
                strcmp(argv[i], "-e") == 0)) {
      exposure_sec = atof(argv[i + 1]);
      i++;
    } else if ((strcmp(argv[i], "--start-gain") == 0 ||
                strcmp(argv[i], "-g") == 0)) {
      gain = atof(argv[i + 1]);
      i++;
    } else if (strncmp(argv[i], "--verbose", 9) == 0) {
      // Long form: --verbose [level]
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        allsky_debug = atoi(argv[i + 1]);
        i++;
      } else {
        allsky_debug = 1;
      }
    } else if (argv[i][0] == '-' && argv[i][1] == 'v') {
      // Short form: -v, -vv, -vvv etc.
      int count = 0;
      for (int j = 1; argv[i][j] == 'v'; j++) {
        count++;
      }
      if (count > 0) {
        allsky_debug = count;
      }
    } else if (strcmp(argv[i], "--img0") == 0) {
      if (i + 1 < argc) {
        img0 = argv[i + 1];
        i++;
      } else {
        fprintf(stderr, "Error: --img0 requires a file path\n");
        exit(1);
      }
    } else if (strcmp(argv[i], "--img1") == 0) {
      if (i + 1 < argc) {
        img1 = argv[i + 1];
        i++;
      } else {
        fprintf(stderr, "Error: --img1 requires a file path\n");
        exit(1);
      }
    } else if (strcmp(argv[i], "--img2") == 0) {
      if (i + 1 < argc) {
        img2 = argv[i + 1];
        i++;
      } else {
        fprintf(stderr, "Error: --img2 requires a file path\n");
        exit(1);
      }
    } else if (strcmp(argv[i], "--img3") == 0) {
      if (i + 1 < argc) {
        img3 = argv[i + 1];
        i++;
      } else {
        fprintf(stderr, "Error: --img3 requires a file path\n");
        exit(1);
      }
    } else if (strcmp(argv[i], "--img4") == 0) {
      if (i + 1 < argc) {
        img4 = argv[i + 1];
        i++;
      } else {
        fprintf(stderr, "Error: --img4 requires a file path\n");
        exit(1);
      }
    } else if ((strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0 ||
                strcmp(argv[i], "-?") == 0)) {
      printf("%s v%s\n", argv[0], PROG_VERSION);
      printf("Copyright (c) 2025 Sven Kreiensen\n");
      printf("\nCommand line parameters:\n");
      printf("--config file|-c file          Configuration file\n");
      printf("--start-exposure|-e exposure   Start exposuse [s]\n");
      printf("--start-gain|-g gain           Start gain\n");
      printf("--verbose [n]|-v[n]            Verbosity level (e.g., -vv or "
             "--verbose 2)\n");
      printf("--img0 file                    Optional image file 0\n");
      printf("--img1 file                    Optional image file 1\n");
      printf("--img2 file                    Optional image file 2\n");
      printf("--img3 file                    Optional image file 3\n");
      printf("--img4 file                    Optional image file 4\n");
      printf("--help|-h|-?                   Command help\n");
      printf("\n");
      exit(0);
    }
  }

  /* Program called without configuration file */
  if (!config_path) {
    fprintf(stderr, "ERROR: No configuration file. Call: %s --config <path>\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  /* Reap all child processes automatically */
  signal(SIGCHLD, SIG_IGN);

  /* Load configuration */
  if (load_config(config_path, &config) != 0) {
    fprintf(stderr, "ERROR: Failed to load configuration file %s.\n",
            config_path);
    return EXIT_FAILURE;
  }
  printf("-----------------------\n\n");

  setup_signal_handlers();

  /* Initialize frames array */
  for (i = 0; i < MAX_FRAMES; i++) {
    frames[i].image = NULL;
    frames[i].image_size = 0;
    frames[i].exposure = 0.0;
    frames[i].gain = 0.0;
    frames[i].median_r = 0.0f;
    frames[i].median_g = 0.0f;
    frames[i].median_b = 0.0f;
    frames[i].brightness = 0.0f;
    frames[i].sigma_noise = 0.0f;
  }

  /* Get image directory */
  images_dir = allsky_safe_strdup(config.image_directory);
  if (allsky_check_directory(images_dir) > 0) {
    fprintf(
        stderr,
        "ERROR: images directory (%s) does not exist or is not writeable.\n",
        images_dir);
    return EXIT_FAILURE;
  }

  /* Get database directory */
  database_dir = allsky_safe_strdup(config.database_directory);
  if (allsky_check_directory(database_dir) > 0) {
    fprintf(stderr,
            "ERROR: database directory (%s) does not exist or is not "
            "writeable.\n",
            database_dir);
    return EXIT_FAILURE;
  }

  /* Start INDIGO API */
  indigo_camera = allsky_safe_strdup(config.indigocamera);
  indigo_set_log_level(INDIGO_LOG_INFO);
  if (indigo_start() != INDIGO_OK) {
    fprintf(stderr, "Failed to start INDIGO.\n");
    return EXIT_FAILURE;
  }

  /* Check if it is night or day */
  int is_night =
      is_sun_below_6_degrees(time(NULL), config.longitude, config.latitude);
  printf("Night mode: %s\n", is_night ? "on" : "off");

  /* Set exposure und gain initially */
  if (exposure_sec == 0.0)
    exposure_sec = is_night ? config.nighttime_startexposure
                            : config.daytime_startexposure;

  if (gain == 0.0)
    gain = is_night ? config.nighttime_startgain : config.daytime_startgain;

  /* Verbosity */
  if (allsky_debug == ALLSKY_DEBUG_DISABLED) {
    int nullfd = open("/dev/null", O_RDWR);
    if (nullfd >= 0) {
      dup2(nullfd, STDIN_FILENO);
      dup2(nullfd, STDOUT_FILENO);
      dup2(nullfd, STDERR_FILENO);
      if (nullfd > 2)
        close(nullfd);
    }
  }

  int time_1 = time(NULL) - config.capture_interval - 1;
  int capture_no = 1; // Capture counter
  int failed_no = 0;  // Failed captures

  printf("\nWaiting");
  fflush(stdout);

  /* Loop forever and taking pictures */
  while (keep_running) {
    time_t now1 = time(NULL);

    /* Capture interval reached, start taking picture */
    if (now1 > time_1 + config.capture_interval) {
      time_1 = now1;

      /* Read configuration again */
      if (load_config(config_path, &config) != 0) {
        fprintf(stderr, "ERROR: Failed to load configuration file %s.\n",
                config_path);
        cleanup_resources();
        return EXIT_FAILURE;
      }

      if (allsky_debug >= 2) {
        show_config(&config);
        printf("\n\n");
      }

      /* Write actual config.json */
      char config_json_path[PATH_MAX + 1];
      snprintf(config_json_path, sizeof(config_json_path), "%s/config.json",
               images_dir);
      config_to_json(config_json_path, &config);

      /* Set day or night mode */
      int was_night = is_night;
      is_night =
          is_sun_below_6_degrees(time(NULL), config.longitude, config.latitude);

      if (was_night && !is_night) {
        printf(
            "It is night and it is now day. Starting end of night tasks...\n");
        if (run_end_of_night_tasks() != 0) {
          fprintf(stderr, "ERROR: Failed to run end of night tasks.\n");
        }
        printf("End of night tasks started\n");
      }

      /* Don't capture if it is day and daytime_capture is off */
      if (!is_night && !config.daytime_capture) {
        printf("Camera is off during day\n");
        continue;
      }

      printf("\n====== Start taking picture series %d (%d failed) "
             "===============\n",
             capture_no, failed_no);
      capture_no++;

      /* Capture timestamp (UTC) */
      time_t now = time(NULL);
      int timezone_offset = get_timezone_offset_seconds(now);

      /* Today string YYYYMMDD (UTC) */
      struct tm *tm_info = gmtime(&now);
      char today_str[9];
      allsky_safe_snprintf(today_str, sizeof(today_str), "%04d%02d%02d",
                           tm_info->tm_year + 1900, tm_info->tm_mon + 1,
                           tm_info->tm_mday);

      /* Now string hhmmss (UTC) */
      char now_str[7];
      snprintf(now_str, sizeof(now_str), "%02d%02d%02d", tm_info->tm_hour,
               tm_info->tm_min, tm_info->tm_sec);

      /* Create image directory images/YYYYMMDD */
      char today_directory[NAME_MAX + 1];
      if (allsky_safe_snprintf(today_directory, sizeof(today_directory),
                               "%s/%s", images_dir, today_str))
        fprintf(stderr, "WARNING: String %s truncated\n", today_directory);
      if (allsky_safe_mkdir(today_directory, 0777))
        continue;

      /* Create thumbnails sub-directory images/YYYYMMDD/thumbnails */
      char thumbnail_directory[NAME_MAX + 1];
      if (allsky_safe_snprintf(thumbnail_directory, sizeof(thumbnail_directory),
                               "%s/thumbnails", today_directory))
        fprintf(stderr, "WARNING: String %s truncated.\n", thumbnail_directory);
      if (allsky_safe_mkdir(thumbnail_directory, 0777))
        continue;

      /* Create debug sub-directory images/YYYYMMDD/debug */
      char debug_directory[NAME_MAX + 1];
      if (allsky_safe_snprintf(debug_directory, sizeof(debug_directory),
                               "%s/debug", today_directory))
        fprintf(stderr, "WARNING: String %s truncated\n", debug_directory);
      if (allsky_safe_mkdir(debug_directory, 0777))
        continue;

#if AI_TRAINING
      /* Create training data sub-directory images/YYYYMMDD/dataset */
      char training_data_directory[NAME_MAX + 1];
      if (allsky_safe_snprintf(training_data_directory,
                               sizeof(training_data_directory), "%s/dataset",
                               today_directory))
        fprintf(stderr, "WARNING: String %s truncated\n",
                training_data_directory);
      if (allsky_safe_mkdir(training_data_directory, 0777))
        continue;
#endif

      /* Reset frames array, but keep image data */
      for (i = 0; i < MAX_FRAMES; i++) {
        // Don't touch frames[i].image = NULL...
        // Don't touch frames[i].image_size...
        frames[i].exposure = 0.0;
        frames[i].gain = 0.0;
        frames[i].median_r = 0.0f;
        frames[i].median_g = 0.0f;
        frames[i].median_b = 0.0f;
        frames[i].brightness = 0.0f;
        frames[i].sigma_noise = 0.0f;
      }

      double f = is_night ? config.nighttime_hdr_exposure_factor
                          : config.daytime_hdr_exposure_factor;
      int use_max_images = 0;
      int overexposed_images = 0;
      int width = 0, height = 0; // Image width and height

      /* Set debayer algorithm */
      int debayer_alg = DEBAYER_ALG_NNI;
      if (!strcmp(config.debayer, "bilinear"))
        debayer_alg = DEBAYER_ALG_BILINEAR;
      if (!strcmp(config.debayer, "vng"))
        debayer_alg = DEBAYER_ALG_VNG;

      clock_t function_start = 0;
      clock_t function_end;

      printf("\n");

      /* Capture frames */
      int abort_captures = 0;

      for (i = 0; i < MAX_FRAMES; i++) {
        if (abort_captures)
          break;

        printf("===== Capturing HDR frame %d / %d (night: %s)=====\n", i,
               MAX_FRAMES, is_night ? "on" : "off");

        frames[i].gain = gain;

        /* Calculate exposure */
        if (i == 0)
          // Reference exposure
          frames[i].exposure = exposure_sec;
        else {

          /* Calculate exposure */
          frames[i].exposure =
              calc_exposure_time(frames[0].exposure,
                                 is_night ? config.nighttime_hdr_exposure_factor
                                          : config.daytime_hdr_exposure_factor,
                                 i);

          // frames[i].exposure = frames[i - 1].exposure / f;

          if (frames[i].exposure < MIN_EXPOSURE) {
            frames[i].exposure = MIN_EXPOSURE;

            /* If last frame was minimum exposure. Give up here. */
            if (i > 0) {
              if (frames[i - 1].exposure == config.camera_min_exposure) {
                fprintf(stderr, "WARNING: Calculated exposure is below minimum "
                                "exposure. Sorry we give up here.\n");
                break; // Next frame
              }
            }
            fprintf(stderr, "WARNING: Calculated exposure is below minimum "
                            "exposure. Taking next frame.\n");
            break; // Next frame
          }
        }

        /* Check if image file parameter is provided for this frame index */
        const char *img_file = NULL;
        if (i == 0 && img0 != NULL)
          img_file = img0;
        else if (i == 1 && img1 != NULL)
          img_file = img1;
        else if (i == 2 && img2 != NULL)
          img_file = img2;
        else if (i == 3 && img3 != NULL)
          img_file = img3;
        else if (i == 4 && img4 != NULL)
          img_file = img4;

        /* Exit conditions */
        if (img0 != NULL && i==1 && img1 == NULL) {

          fprintf(stderr, "ERROR: Image file 1 is not provided. Exiting...\n");
          abort_captures = 1;
          failed_no++;
          continue;
        }
        if (img0 != NULL && i==2 && img2 == NULL) {
          fprintf(stderr, "ERROR: Image file 2 is not provided. Exiting...\n");
          abort_captures = 1;
          failed_no++;
          continue;
        }
        if (img0 != NULL && i==3 && img3 == NULL) {
          fprintf(stderr, "ERROR: Image file 3 is not provided. Exiting...\n");
          abort_captures = 1;
          failed_no++;
          continue;
        }
        if (img0 != NULL && i==4 && img4 == NULL) {
          fprintf(stderr, "ERROR: Image file 4 is not provided. Exiting...\n");
          abort_captures = 1;
          failed_no++;
          continue;
        }

        if (img_file != NULL) {
          /* Load image file from disk */
          FILE *file = fopen(img_file, "rb");
          if (!file) {
            fprintf(stderr, "ERROR: Failed to open image file: %s\n", img_file);
            abort_captures = 1;
            failed_no++;
            continue;
          }

          /* Get file size */
          fseek(file, 0, SEEK_END);
          long file_size = ftell(file);
          fseek(file, 0, SEEK_SET);

          if (file_size <= 0) {
            fprintf(stderr, "ERROR: Invalid file size for image file: %s\n",
                    img_file);
            fclose(file);
            abort_captures = 1;
            failed_no++;
            continue;
          }

          /* Check if target buffer size is large enough and reallocate if
           * necessary */
          size_t needed_raw_buffer_size = (size_t)file_size;
          if (needed_raw_buffer_size > raw_buffer_size) {
            unsigned char *temp_raw_buffer = (unsigned char *)allsky_safe_realloc(
                raw_buffer, needed_raw_buffer_size);
            if (!temp_raw_buffer) {
              fprintf(stderr,
                      "ERROR: realloc() failed (raw_buffer_size=%ld bytes, "
                      "new_size=%ld bytes)\n",
                      raw_buffer_size, needed_raw_buffer_size);
              fclose(file);
              allsky_safe_free(raw_buffer);
              raw_buffer = NULL;
              abort_captures = 1;
              failed_no++;
              continue;
            }

            raw_buffer = temp_raw_buffer;
          }
          raw_buffer_size = needed_raw_buffer_size;

          /* Read file content into raw buffer */
          size_t bytes_read = fread(raw_buffer, 1, raw_buffer_size, file);
          fclose(file);

          if (bytes_read != raw_buffer_size) {
            fprintf(stderr,
                    "ERROR: Failed to read complete file: %s (read %ld of %ld "
                    "bytes)\n",
                    img_file, bytes_read, raw_buffer_size);
            abort_captures = 1;
            failed_no++;
            continue;
          }

          printf("Loaded image file %s (%ld bytes) into raw_buffer\n", img_file,
                 raw_buffer_size);
        } else {
          /* Connect to Indigo server */
          if (!server)
            indigo_connect_server(config.indigoservername, config.indigoserver,
                                  config.indigoport,
                                  &server); // Check correct host name in 2nd arg!!!

          /* Set camera binning */
          set_camera_binning(&client, indigo_camera, config.camera_binning2x2);

          /* Capturing a single frame */
          if (capture_image(frames[i].exposure, config.indigo_capture_timeout)) {
            abort_captures = 1;
            exposure_sec = frames[0].exposure;

            indigo_device_disconnect(&client, indigo_camera);
            indigo_disconnect_server(server);
            server = NULL;

            continue;
          }
        }

        /* Get raw image width and height */
        int raw_width, raw_height;
        indigo_raw_get_width(raw_buffer, &raw_width);
        indigo_raw_get_height(raw_buffer, &raw_height);
        if (raw_width == 0 || raw_height == 0) {
          fprintf(stderr,
                  "ERROR: taking HDR frame (raw_width or raw_height is 0)\n");
          abort_captures = 1;
          exposure_sec = frames[0].exposure;
          failed_no++;
          continue;
        }

        /* Crop image */
        width = config.crop_image ? config.crop_width : raw_width;
        height = config.crop_image ? config.crop_height : raw_height;
        printf("Indigo raw image (header data): %dx%d px (%ld bytes)\n",
               raw_width, raw_height,
               raw_width * raw_height * sizeof(unsigned short));
        printf("Processing image buffer (float, RGBA): %dx%d px (%ld bytes)\n",
               width, height, width * height * CHANNELS * sizeof(float));

        /* Check if target buffer size is large enough */
        size_t est_buffer_size = width * height * sizeof(unsigned short);
        if (est_buffer_size > raw_buffer_size) {
          fprintf(stderr,
                  "ERROR: taking HDR frame (image header size: %dx%d > target "
                  "buffer size %dx%d)\nPlease decrease crop.width and "
                  "crop.height in your config.ini\n",
                  raw_width, raw_height, width, height);

          abort_captures = 1;
          exposure_sec = frames[0].exposure;
          failed_no++;
          continue;
        }

        /* Load raw image data into memory */
        size_t processing_image_size =
            (size_t)(width * height * CHANNELS * sizeof(float));
        if (processing_image_size > frames[i].image_size) {
          float *temp_image = (float *)allsky_safe_realloc(
              frames[i].image, processing_image_size);
          if (!temp_image) {
            fprintf(stderr, "HDR frame %d / %d: image buffer realloc failed\n",
                    i, MAX_FRAMES);
            allsky_safe_free(frames[i].image);
            frames[i].image = NULL;
            frames[i].image_size = 0;
            continue;
          }

          printf("HDR frame %d / %d: image buffer allocated: %ld bytes\n", i,
                 MAX_FRAMES, processing_image_size);

          frames[i].image = temp_image;
        }
        frames[i].image_size = processing_image_size;

        /* Load camera image buffer into RGBA (float) */
        TIMED_BLOCK("indigo_raw_to_rgbf1", {
          if (indigo_raw_to_rgbf1(frames[i].image, raw_width, raw_height,
                                  raw_buffer, debayer_alg, width, height,
                                  config.crop_image ? config.crop_x_offset : 0,
                                  config.crop_image ? config.crop_y_offset
                                                    : 0)) {
            fprintf(stderr, "ERROR: Loading raw data to rgbf array.\n");

            abort_captures = 1;
            failed_no++;
            continue;
          }
        });

        printf(
            "------ HDR frame %d / %d (exposure: %.6fs, gain: %.1f) captured "
            "--------\n\n",
            i, MAX_FRAMES, frames[i].exposure, gain);

        /* Mask areas outside of the image circle */
        if (config.image_mask_radius > 0) {
          TIMED_BLOCK("mask_image_circle_rgbf1", {
            mask_image_circle_rgbf1(
                frames[i].image, width, height, config.image_center_x,
                config.image_center_y, config.image_mask_radius);
          });

          if (debug_pipeline_images) {
            save_debug_pipeline_image(frames[i].image, width, height,
                                      i * 100 + 1, images_dir, 1);
          }
        }
#if 0
#define CAL_REGION_X0 123
#define CAL_REGION_Y0 987
#define CAL_REGION_X1 133
#define CAL_REGION_Y1 997
		float scale_r, scale_b;
		estimate_gray_scaling_rgbf1(rgba16_data, width, height,
                      CAL_REGION_X0, CAL_REGION_Y0, CAL_REGION_X1, CAL_REGION_Y1,
                      &scale_r, &scale_b);
#endif

        /* RGB channel scaling */
        TIMED_BLOCK("white_balance_rgbf1", {
          white_balance_rgbf1(frames[i].image, width, height,
                              is_night ? config.nighttime_white_balance_red
                                       : config.daytime_white_balance_red,
                              1.0f,
                              is_night ? config.nighttime_white_balance_blue
                                       : config.daytime_white_balance_blue,
                              config.daytime_white_balance_light_protect);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(frames[i].image, width, height, i * 100 + 2,
                                    images_dir, 1);
        }

        /* Save raw exposures */
        if (config.debug_raw_exposures) {
          char re_filename[PATH_MAX + 1];
          if (allsky_safe_snprintf(re_filename, sizeof(re_filename),
                                   "%s/image-%s%s_t%d.jpg", today_directory,
                                   today_str, now_str, i))
            fprintf(stderr, "WARNING: String %s truncated.\n", re_filename);

          // Save jpeg
          save_jpeg_rgbf1(frames[i].image, width, height, 9, 0.25f,
                          re_filename);

          printf("Debug raw HDR frame %d / %d image %s saved.\n", i, MAX_FRAMES,
                 re_filename);
        }

        /* Check for clipped pixels */
        int overexposed = 0;
        if (hdr_check_overexposure_rgbf1(
                frames[i].image, width, height, config.hdr_clipping_threshold,
                config.hdr_min_clipped_pixels, &overexposed) > 0) {
          fprintf(stderr, "ERROR: Over exposure check failed.\n");
        }

        if (overexposed) {
          printf("------------ HDR frame %d / %d is overexposed!\n", i,
                 MAX_FRAMES);

          if (overexposed_images < MAX_FRAMES)
            overexposed_images++;
        }

        use_max_images++;
        printf("Can use %d of %d HDR frames for HDR fusion\n", use_max_images,
               MAX_FRAMES);

        /* No HDR */
        if (!config.hdr) {
          overexposed_images = 0;
          break;
        }

        /* Stop taking more frames when this image is not over-exposed */
        if (!overexposed)
          break;
      } // Take next frame

      if (use_max_images == 0 || abort_captures == 1) {
        // Abort here
        continue;
      }

      for (i = 0; i < use_max_images; i++) {
        printf("===== Processing HDR frame %d / %d =====\n", i, use_max_images);

#if AI_TRAINING
        /* Save AI training data */
        int ai_training_data = 1;

        // Define patch positions
        int patch_positions[13][2] = {{1515, 345},  {1584, 660},  {1965, 708},
                                      {2355, 729},  {2673, 909},  {2283, 1056},
                                      {1848, 1137}, {2202, 1230}, {2610, 1284},
                                      {2208, 1356}, {1992, 1578}, {2460, 1638},
                                      {2091, 1752}};

        if (ai_training_data && i < 2 && is_night && gain > 0.0f) {
          char dataset_directory[NAME_MAX + 1];
          if (allsky_safe_snprintf(dataset_directory, sizeof(dataset_directory),
                                   "%s/%s", training_data_directory,
                                   i == 0 ? "target" : "input"))
            fprintf(stderr, "WARNING: String %s truncated\n",
                    training_data_directory);
          if (allsky_safe_mkdir(dataset_directory, 0777))
            continue;

          // Create metadata object
          struct json_object *meta = json_object_new_object();
          json_object_object_add(meta, "exposure",
                                 json_object_new_double(frames[i].exposure));
          json_object_object_add(meta, "gain", json_object_new_double(gain));
          json_object_object_add(meta, "sigma_noise",
                                 json_object_new_double(frames[i].sigma_noise));
          json_object_object_add(meta, "note",
                                 json_object_new_string("Test metadata"));

          // Call patch saving function
          int patch_size = 256;

          char file_prefix[PATH_MAX + 1];
          snprintf(file_prefix, sizeof(file_prefix), "image-%s", now_str);

          save_patches_with_metadata(frames[i].image, width, height, patch_size,
                                     patch_positions, 13, dataset_directory,
                                     file_prefix, meta);

          json_object_put(meta);
        }
#endif

        /* Compute (median) brightness */
        TIMED_BLOCK("compute_filtered_median_brightness_rgbf1", {
          compute_filtered_median_brightness_rgbf1(
              frames[i].image, width, height, config.measure_brightness_area,
              &frames[i].median_r, &frames[i].median_g, &frames[i].median_b);
        });

        /* Measure background noise at night and at higher gains */
        if (is_night && gain > 0.0) {
          TIMED_BLOCK("compute_background_noise_mad_rgbf1", {
            compute_background_noise_mad_rgbf1(
                frames[i].image, width, height, frames[i].median_g,
                &frames[i].sigma_noise, config.image_zenith_x,
                config.image_zenith_y, config.nighttime_measure_noise_radius);
          });
        }

        /* SCNR filter to reduce artificial light sources */
        if (config.scnr_filter && is_night) {
          /* Set protection mode */
          int scnr_protection = SCNR_PROTECT_NONE;
          if (!strcmp(config.scnr_filter_protection, "average_neutral"))
            scnr_protection = SCNR_PROTECT_AVERAGE_NEUTRAL;
          if (!strcmp(config.scnr_filter_protection, "maximum_neutral"))
            scnr_protection = SCNR_PROTECT_MAXIMUM_NEUTRAL;

          TIMED_BLOCK("scnr_green_filter_rgbf1", {
            scnr_green_filter_rgbf1(frames[i].image, width, height,
                                    config.scnr_filter_amount, scnr_protection);
          });
        }

        /* Reduce noise at higher gain */
        if (config.nighttime_multiscale_median_filter) {
          // Don't touch frame 0
          // Only at night
          // Gain must be > 0.0
          // Only for short exposures
          if (is_night && frames[i].exposure > 1.0 && gain > 1.0 && i > 0) {
            TIMED_BLOCK("multiscale_median_filter_rgbf1", {
              /* Adaptive multiscale median filter: Max scale is dynamic based
               * on the number of HDR frames */
              multiscale_median_filter_rgbf1(
                  frames[i].image, width, height,
                  config.nighttime_multiscale_median_filter_max_scale * (i + 1),
                  config.nighttime_multiscale_median_filter_amount);
            });

            if (debug_pipeline_images) {
              save_debug_pipeline_image(frames[i].image, width, height,
                                        i * 100 + 3, images_dir, 1);
            }
          }
        }

        /* Save single exposures */
        if (config.hdr_save_exposures) {
          char hdr_frame_filename[PATH_MAX + 1];
          if (allsky_safe_snprintf(hdr_frame_filename,
                                   sizeof(hdr_frame_filename),
                                   "%s/latest_image_t%d.jpg", images_dir, i))
            fprintf(stderr, "WARNING: String %s truncated\n",
                    hdr_frame_filename);

          // Save jpg
          cairo_surface_t *surface1 = NULL;
          surface1 =
              cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
          if (cairo_surface_status(surface1) != CAIRO_STATUS_SUCCESS) {
            fprintf(stderr, "Failed to create cairo_surface_t!\n");
            continue;
          }

          unsigned char *cairo_data = cairo_image_surface_get_data(surface1);

          rgbf_to_cairo(frames[i].image, cairo_data, width, height);
          cairo_surface_mark_dirty(surface1);

          save_cairo_surface_as_jpeg(surface1, hdr_frame_filename,
                                     config.jpeg_quality);
          cairo_surface_destroy(surface1);

          printf("HDR exposure image %s saved.\n", hdr_frame_filename);
        }
      }

      printf("Use frames for HDR merge: %d\n", use_max_images);

      // Take exposure t0
      int pixel_count = width * height;
      int channels = 4; // RGBA

      size_t needed_hdr_image_size =
          (size_t)(pixel_count * channels * sizeof(float));
      if (needed_hdr_image_size > hdr_image_size) {
        float *temp_hdr_image =
            (float *)allsky_safe_realloc(hdr_image, needed_hdr_image_size);
        if (!temp_hdr_image) {
          fprintf(stderr, "hdr_image realloc failed");
          allsky_safe_free(hdr_image);
          hdr_image = NULL;

          continue;
        }

        hdr_image = temp_hdr_image;
      }
      hdr_image_size = needed_hdr_image_size;

      printf("===== HDR fusion =====\n");

      if (config.hdr == 0) {
        /* No HDR */
        memcpy(hdr_image, frames[0].image,
               pixel_count * channels * sizeof(float));
        printf("Nothing over-exposed => No HDR needed.\n");
      } else {
        /* HDR */
        /* Multiscale exposure fusion
         * Recommended Y_max_expected values per Scene:
         *
         * - Deep night (clear stars):          Y_max_expected = 1.0f
         * - Night with fog / thin clouds:      Y_max_expected = 1.2f
         * - Dämmerung (blue hour):             Y_max_expected = 1.5f
         * - Daytime (cloudy, no sun in image): Y_max_expected = 2.0f
         * - Daytime (sun visible in image):    Y_max_expected = 3.0f
         *
         * Note: If Y_max_expected = 0, it will be automatically detected from
         * image data.
         */
        float y_max_expected = config.hdr_y_max_expected;
        if (y_max_expected > 0.0f && use_max_images >= 2) {
          // Skaliere basierend auf der Anzahl der Bilder (nur wenn Wert > 0)
          float scale_factor =
              1.0f +
              (use_max_images - 1) * 0.2f; // z.B. 2 Bilder = 1.2, 3 = 1.4, etc.
          y_max_expected *= scale_factor;
        }
        TIMED_BLOCK("hdr_multi_scale_fusion_rgbf1", {
          /* Get channel scaling factors used in white balance */
          float channel_scale_r =
              is_night ? (float)config.nighttime_white_balance_red
                       : (float)config.daytime_white_balance_red;
          float channel_scale_g = 1.0f; // Green is always 1.0
          float channel_scale_b =
              is_night ? (float)config.nighttime_white_balance_blue
                       : (float)config.daytime_white_balance_blue;

          if (overexposed_images == 0)
            use_max_images = 1;

          if (use_max_images > 3) {
            use_max_images = 3;
          }

          hdr_multi_scale_fusion_laplacian_rgbf1(
              frames, use_max_images, width, height,
              (float)config.hdr_clipping_threshold, y_max_expected, hdr_image,
              images_dir, config.hdr_weight_maps, config.hdr_weight_stats,
              config.hdr_chroma_mode, config.hdr_contrast_weight_strength,
              config.hdr_pyramid_levels_override,
              (float)config.hdr_weight_sigma,
              (float)config.hdr_weight_clip_factor, channel_scale_r,
              channel_scale_g, channel_scale_b);
        });

        /* High Dynamic Range Tone Mapping */
        if (config.hdrmt) {
          // Set parameters based on day/night mode
          int levels, start_level;
          float strength, strength_boost, midtones, shadow_protect,
              highlight_protect;
          float epsilon = 1e-6f;
          float gain_cap = 4.0f;

          if (is_night) {
            // Nacht (Deep Sky): levels=6..7, start_level=3..4,
            // strength=0.35..0.6, boost=0.2..0.4, highlight_protect=0.3
            levels = 6;
            start_level = 3;
            strength = 0.45f;
            strength_boost = 0.3f;
            midtones = 0.5f;
            shadow_protect = 0.0f;
            highlight_protect = 0.3f;
          } else {
            // Tag/Wolken: levels=5..6, start_level=2..3, strength=0.25..0.45,
            // boost=0.2, shadow_protect=0.1, highlight_protect=0.2
            levels = 5;
            start_level = 2;
            strength = 0.35f;
            strength_boost = 0.2f;
            midtones = 0.5f;
            shadow_protect = 0.1f;
            highlight_protect = 0.2f;
          }

          TIMED_BLOCK("hdrmt_rgbf1", {
            hdrmt_rgbf1(hdr_image, width, height, levels, start_level, strength,
                        strength_boost, midtones, shadow_protect,
                        highlight_protect, epsilon, gain_cap);
          });
        }

        /* Normalize to 0.0..1.0f */
        TIMED_BLOCK("hdr_normalize_range_rgbf1", {
          hdr_normalize_range_rgbf1(hdr_image, width, height, 0.0f, 1.0f);
        });
      }

      if (debug_pipeline_images) {
        save_debug_pipeline_image(hdr_image, width, height, 801, images_dir, 1);
      }

      printf("===== Postprocessing =====\n");

      /* Upload HDR image to GPU */
      float *hdr_image_device = NULL;
      if (upload_rgbf_to_device(hdr_image, width, height, &hdr_image_device) !=
          0) {
        printf("WARNING: GPU upload failed, using CPU processing\n");
        hdr_image_device = NULL; // Signal for CPU fallback
      }

      /* Autostretch */
      if (config.autostretch) {
        TIMED_BLOCK("autostretch_rgbf1", {
          autostretch_rgbf1(hdr_image_device, width, height, config.autostretch_min_val, config.autostretch_max_val);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 808,
                                    images_dir, 1);
        }
      }

      /* Move blackpoint */
      if ((is_night ? config.nighttime_move_blackpoint
                    : config.daytime_move_blackpoint) > 0.0f) {
        TIMED_BLOCK("adjust_black_point_rgbf1", {
          adjust_black_point_rgbf1(
              hdr_image_device, width, height,
              is_night ? config.nighttime_move_blackpoint_min_shift_pct
                       : config.daytime_move_blackpoint_min_shift_pct,
              is_night ? config.nighttime_move_blackpoint_max_shift_pct
                       : config.daytime_move_blackpoint_max_shift_pct,
              is_night ? config.nighttime_move_blackpoint_dark_threshold
                       : config.daytime_move_blackpoint_dark_threshold);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 803,
                                    images_dir, 1);
        }
      }
#if 0
      /* Auto color calibration */
      if (is_night) {
        TIMED_BLOCK("auto_color_calibration_rgbf1", {
          auto_color_calibration_rgbf1(hdr_image_device, width, height, 40);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 809,
                                    images_dir, 1);
        }
      } else {
#endif
      /* Ambience Color Calibration */
      if (config.ambience_color_calibration) {

        /* Scene detection */
        SceneType scene = detect_scene_rgbf1(hdr_image_device, width, height);

        int do_ambience_color_calibration = 0;
        float subframe_percent = 40.0f;
        float luma_clip_high = 0.80f;
        float sat_clip_high = 0.30f;
        float sigma_factor = 1.0f;
        float mix_factor = 0.25f;

        switch (scene) {
        case SCENE_NIGHT_CLEAR:
          subframe_percent = 40.0f;
          luma_clip_high = 0.80f;
          sat_clip_high = 0.30f;
          sigma_factor = 1.0f;
          mix_factor = 0.25f;
          do_ambience_color_calibration = 1;
          printf("Scene: SCENE_NIGHT_CLEAR\n");
          break;

        case SCENE_NIGHT_CLOUDY:
          subframe_percent = 15.0f;
          luma_clip_high = 0.75f;
          sat_clip_high = 0.25f;
          sigma_factor = 0.8f;
          mix_factor = 0.10f;
          do_ambience_color_calibration = 1;
          printf("Scene: SCENE_NIGHT_CLOUDY\n");
          break;

        case SCENE_NIGHT_MOON:
          subframe_percent = 20.0f;
          luma_clip_high = 0.70f;
          sat_clip_high = 0.20f;
          sigma_factor = 1.2f;
          mix_factor = 0.20f;
          do_ambience_color_calibration = 1;
          printf("Scene: SCENE_NIGHT_MOON\n");
          break;

        case SCENE_NIGHT_AURORA:
          /* lighter correction, bigger subframe */
          subframe_percent = 60.0f;
          luma_clip_high = 0.70f;
          sat_clip_high = 0.60f;
          sigma_factor = 1.0f;
          mix_factor = 0.15f;
          do_ambience_color_calibration = 1;
          printf("Scene: SCENE_NIGHT_AURORA\n");
          break;

        case SCENE_TWILIGHT:
          /* DO NOT correct color → skip */
          printf("Scene: SCENE_TWILIGHT\n");
          break;

        case SCENE_DAY_CLEAR:
        case SCENE_DAY_CLOUDY:
          /* daytime */
          subframe_percent = 20.0f;
          luma_clip_high = 0.30f;
          sat_clip_high = 0.80f;
          sigma_factor = 0.6f;
          mix_factor = 0.08f;
          do_ambience_color_calibration = 1;
          printf("Scene: SCENE_DAY_CLEAR or SCENE_DAY_CLOUDY\n");
          break;
        }

        if (do_ambience_color_calibration) {
          TIMED_BLOCK("ambience_color_calibration_rgbf1", {
            ambience_color_calibration_rgbf1(
                hdr_image_device, width, height, subframe_percent,
                luma_clip_high, sat_clip_high, sigma_factor, mix_factor);
          });

          if (debug_pipeline_images) {
            save_debug_pipeline_image(hdr_image_device, width, height, 802,
                                      images_dir, 1);
          }
        }
      }

      /* ACDNR noise reduction */
      if (config.acdnr_filter) {
        TIMED_BLOCK("acdnr_filter_rgbf1", {
          acdnr_filter_rgbf1(hdr_image_device, width, height,
                             config.acdnr_filter_lum_stddev,
                             config.acdnr_filter_lum_amount,
                             config.acdnr_filter_lum_iterations,
                             config.acdnr_filter_lum_kernel_size,
                             config.acdnr_filter_chrom_stddev,
                             config.acdnr_filter_chrom_amount,
                             config.acdnr_filter_chrom_iterations,
                             config.acdnr_filter_chrom_kernel_size);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 802,
                                    images_dir, 1);
        }
      }

      /* Wavelet sharpening */
      if (config.wavelet_sharpen) {
        TIMED_BLOCK("wavelet_sharpen_rgbf", {
          wavelet_sharpen_rgbf1(hdr_image_device, width, height,
                                config.wavelet_sharpen_gain_small,
                                config.wavelet_sharpen_gain_medium,
                                config.wavelet_sharpen_gain_large);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 810,
                                    images_dir, 1);
        }
      }

      /* Clarity filter */
      if (config.clarity_filter) {
        TIMED_BLOCK("clarity_filter_rgbf_masked", {
          clarity_filter_rgbf_masked(
              hdr_image_device, width, height, config.clarity_filter_strength,
              config.clarity_filter_radius, config.clarity_filter_midtone_width,
              config.clarity_filter_preserve_highlights,
              config.clarity_filter_mask_mode);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 807,
                                    images_dir, 1);
        }
      }

      /* Dehaze filter */
      if (config.dehaze_amount > 0.0) {
        TIMED_BLOCK("dehaze_rgbf1", {
          perceptual_dehaze_rgbf1_multiscale_full(hdr_image_device, width,
                                                  height, config.dehaze_amount,
                                                  config.dehaze_estimate);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 806,
                                    images_dir, 1);
        }
      }

      /* Adaptive Gamma based on number of HDR subframes */
      if (config.gamma != 1.0) {
        float base_gamma = config.gamma;
        /* Linear decay based on number of HDR subframes */
        // float gamma = base_gamma - 0.07f * (use_max_images);
        //  Quadratic decay based on number of HDR subframes */
        float gamma =
            base_gamma -
            0.07f * (1.0f - 1.0f / (1.0f + (use_max_images - 1) * 0.3f));
        // Logarithmic decay based on number of HDR subframes */
        // float gamma = base_gamma - 0.05f * logf(1.0f + (use_max_images - 1) *
        // 0.5f);
        printf("Gamma: %.3f (use_max_images: %d)\n", gamma, use_max_images);
        if (gamma < 0.1f)
          gamma = 0.1f;

        TIMED_BLOCK("apply_gamma_correction_rgbf1", {
          apply_gamma_correction_rgbf1(hdr_image_device, width, height, gamma);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 804,
                                    images_dir, 1);
        }
      }

      /* Boost colors */
      if (config.saturation != 1.0) {
        TIMED_BLOCK("adjust_saturation_rgbf1", {
          adjust_saturation_rgbf1(hdr_image_device, width, height,
                                  config.saturation);
        });

        if (debug_pipeline_images) {
          save_debug_pipeline_image(hdr_image_device, width, height, 805,
                                    images_dir, 1);
        }
      }

      /*********** Non-GPU postprocessing *********/

      /* SQM estimation */
      int num_stars = 0;
      float estimated_sqm = 0.0;
      if (config.sqm && (frames[0].exposure > 1.0) &&
          (frames[0].median_g <= 0.21)) {
        /* Only reliable at astronomical night and long exposures */
        /* Generate star model (gaussian) */
        int tpl_size = config.sqm_star_size;
        float sigma = config.sqm_star_sigma;
        float *template_data = generate_gaussian_template(tpl_size, sigma);

        star_position_t *stars = NULL;

        TIMED_BLOCK("find_stars_template_rgbf1", {
          int result = find_stars_template(
              frames[0].image, width, height, template_data, tpl_size, tpl_size,
              config.image_zenith_x, config.image_zenith_y, config.sqm_radius,
              config.sqm_threshold, config.sqm_max_stars, &stars, &num_stars);

          if (result == 0) {
#if 0
            // Debug output of stars
            for (int i = 0; i < num_stars; i++) {
              printf("Star %d: (%d, %d)\n", i + 1, stars[i].x, stars[i].y);
            }
#endif
            char stars_file[PATH_MAX + 1];
            if (allsky_safe_snprintf(stars_file, sizeof(stars_file),
                                     "%s/stars-%s%s.json", today_directory,
                                     today_str, now_str))
              fprintf(stderr, "WARNING: String %s truncated\n", stars_file);

            stars_to_json(stars, num_stars, stars_file);
          } else
            fprintf(stderr, "Star detection failed with code %d\n", result);
        });

        allsky_safe_free(stars);
        allsky_safe_free(template_data);

        if (num_stars > 0) {
          /* Estimate SQM */
          estimate_sqm_corrected(num_stars, frames[0].exposure,
                                 frames[0].gain * 0.1f, config.sqm_intercept,
                                 config.sqm_slope, &estimated_sqm);
          printf("Estimated SQM: %.2f mag/arcsec²\n", estimated_sqm);
        }
      }

      /* Measure focus */
      float sharpness = 0.0f;
      measure_focus_laplacian_rgba(hdr_image, width, height,
                                   config.focus_center_x, config.focus_center_y,
                                   config.focus_radius, &sharpness);

      /* Calculate next exposure and gain */
      // Exposure t0 for exposure calculation
      exposure_sec = frames[0].exposure;
      if (frames[0].median_g >= 0) {
        adjust_exposure_gain(frames[0].median_g,
                             is_night ? config.nighttime_meanbrightness
                                      : config.daytime_meanbrightness,
                             config.camera_max_exposure,
                             config.camera_min_exposure, config.camera_max_gain,
                             config.camera_min_gain, &exposure_sec, &gain);
      }

      // Format timestamp into datetime string
      char datetime[50];
      snprintf(datetime, sizeof(datetime), "%04d.%02d.%02d  %02d:%02d:%02d",
               tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
               tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec);

      /* Draw text on image */
      draw_text_red_freetype_rgbf1(hdr_image_device, width, height, datetime, config.font_size, 10, 10, config.font_path);


      /* Final 16-bit tonemapping (convert to 0-65535.0)*/
      TIMED_BLOCK("tonemap_rgbf1_to_rgbf16", {
        tonemap_rgbf1_to_rgbf16(hdr_image_device, width, height);
      });

      if (debug_pipeline_images) {
        save_debug_pipeline_image(hdr_image_device, width, height, 99,
                                  images_dir, 0);
      }

      /* Save images/latest_image.jpg */
      char latest_image_filepath[PATH_MAX + 1];
      if (allsky_safe_snprintf(latest_image_filepath,
                               sizeof(latest_image_filepath),
                               "%s/latest_image.jpg", images_dir))
        fprintf(stderr, "WARNING: String %s truncated\n",
                latest_image_filepath);
      TIMED_BLOCK("save_jpeg_rgbf16", {
        if (save_jpeg_rgbf16(hdr_image_device, width, height,
                             config.jpeg_quality, 1.0f,
                             latest_image_filepath) != 0) {
          fprintf(stderr, "ERROR: Failed to save latest image to %s\n",
                  latest_image_filepath);
        }
        printf("Latest image %s saved\n", latest_image_filepath);
      });

      /* Thumbnail creation */
      if (config.thumbnail) {
        char thumbnail_filepath[PATH_MAX + 1];
        if (allsky_safe_snprintf(thumbnail_filepath, sizeof(thumbnail_filepath),
                                 "%s/image-%s%s.jpg", thumbnail_directory,
                                 today_str, now_str))
          fprintf(stderr, "WARNING: String %s truncated\n", thumbnail_filepath);

        float thumbnail_width_scale =
            (float)config.thumbnail_width / (float)width;
        printf("thumbnail_width_scale: %f\n", thumbnail_width_scale);

        if (save_jpeg_rgbf16(hdr_image_device, width, height,
                             config.jpeg_quality, thumbnail_width_scale,
                             thumbnail_filepath) != 0) {
          fprintf(stderr, "ERROR: Failed to save thumbnail to %s\n",
                  thumbnail_filepath);
        }
      }

      /* Download HDR image from GPU and free GPU memory */
      if (hdr_image_device != NULL) {
        download_rgbf_from_device(hdr_image_device, width, height, hdr_image);
        free_rgbf_on_device(hdr_image_device);
      }

      /* Move images/latest_image.* to images/YYYYMMDD directory */
      char archive_image_filepath[PATH_MAX + 1];
      if (allsky_safe_snprintf(
              archive_image_filepath, sizeof(archive_image_filepath),
              "%s/image-%s%s.jpg", today_directory, today_str, now_str))
        fprintf(stderr, "WARNING: String %s truncated\n",
                archive_image_filepath);
      if (allsky_safe_rename(latest_image_filepath, archive_image_filepath)) {
        continue;
      }
      printf("Image %s archived to %s\n", latest_image_filepath,
             archive_image_filepath);

      /* Convert to 8-bit cairo surface */
      cairo_surface_t *hdr_surface = NULL;
      hdr_surface =
          cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
      if (cairo_surface_status(hdr_surface) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "Can not create cairo surface.\n");

        continue;
      }

      unsigned char *cairo_data = cairo_image_surface_get_data(hdr_surface);
      rgbf_to_cairo(hdr_image, cairo_data, width, height);
      cairo_surface_mark_dirty(hdr_surface);

      /* Create horizon panorama  */
      if (config.panorama) {
        char latest_panorama_filepath[PATH_MAX + 1];

        // ./images/latest_panorama.jpg
        if (allsky_safe_snprintf(latest_panorama_filepath,
                                 sizeof(latest_panorama_filepath),
                                 "%s/latest_panorama.jpg", images_dir))
          fprintf(stderr, "WARNING: String %s truncated\n",
                  latest_panorama_filepath);

        cairo_surface_t *panorama_surface = NULL;
        panorama_surface = convert_allsky_to_panorama(
            hdr_surface, config.panorama_width, config.panorama_height,
            config.panorama_center_x, config.panorama_center_y,
            config.panorama_horizontal_start);

        save_cairo_surface_as_jpeg(panorama_surface, latest_panorama_filepath,
                                   config.jpeg_quality);

        char archive_panorama_filepath[PATH_MAX + 1];
        if (allsky_safe_snprintf(
                archive_panorama_filepath, sizeof(archive_panorama_filepath),
                "%s/panorama-%s%s.jpg", today_directory, today_str, now_str))
          fprintf(stderr, "WARNING: String %s truncated\n",
                  archive_panorama_filepath);

        if (rename(latest_panorama_filepath, archive_panorama_filepath)) {
          /* Release cairo image memory */
          cairo_surface_destroy(panorama_surface);
          cairo_surface_destroy(hdr_surface);

          continue;
        }

        /* Save thumbnail */
        if (config.thumbnail) {
          char panorama_thumbnail_filepath[PATH_MAX + 1];
          if (allsky_safe_snprintf(panorama_thumbnail_filepath,
                                   sizeof(panorama_thumbnail_filepath),
                                   "%s/panorama-%s%s.jpg", thumbnail_directory,
                                   today_str, now_str))
            fprintf(stderr, "WARNING: String %s truncated\n",
                    panorama_thumbnail_filepath);
          create_thumbnail_from_surface(panorama_surface,
                                        panorama_thumbnail_filepath,
                                        config.thumbnail_width);
        }

        cairo_surface_destroy(panorama_surface);
      }

      /* Release cairo image memory */
      cairo_surface_destroy(hdr_surface);

      /* Save image meta data */
      image_metadata.timestamp = now;
      image_metadata.timezone_offset = timezone_offset;
      image_metadata.width = width;
      image_metadata.height = height;
      image_metadata.exposure_t0 = frames[0].exposure;
      image_metadata.exposure_t1 = frames[1].exposure;
      image_metadata.exposure_t2 = frames[2].exposure;
      image_metadata.exposure_t3 = frames[3].exposure;
      image_metadata.exposure_t4 = frames[4].exposure;
      image_metadata.sigma_noise_t0 = frames[0].sigma_noise;
      image_metadata.sigma_noise_t1 = frames[1].sigma_noise;
      image_metadata.sigma_noise_t2 = frames[2].sigma_noise;
      image_metadata.sigma_noise_t3 = frames[3].sigma_noise;
      image_metadata.sigma_noise_t4 = frames[4].sigma_noise;
      image_metadata.gain =
          gain; // FIXME, das ist hier schon der neu berechnete gain
      image_metadata.focus = sharpness;
      image_metadata.capture_interval = config.capture_interval;
      image_metadata.night_mode = is_night;
      image_metadata.hdr = use_max_images;
      image_metadata.sensor_temperature = 9.0;
      image_metadata.mean_brightness = frames[0].median_g;
      image_metadata.target_brightness = is_night
                                             ? config.nighttime_meanbrightness
                                             : config.daytime_meanbrightness;
      image_metadata.stars = num_stars;
      image_metadata.sqm = estimated_sqm;
      image_metadata.sun_altitude =
          sun_altitude(now, config.longitude, config.latitude);
      image_metadata.moon_altitude =
          moon_altitude(now, config.longitude, config.latitude);
      image_metadata.moon_phase_percentage = moon_phase_percentage(now);

      char metadata_filepath[PATH_MAX + 1];
      if (allsky_safe_snprintf(metadata_filepath, sizeof(metadata_filepath),
                               "%s/image-%s%s.json", today_directory, today_str,
                               now_str))
        fprintf(stderr, "WARNING: String %s truncated\n", metadata_filepath);
      save_image_metadata_json(metadata_filepath, &image_metadata);

      /* Save images/latest_image.json */
      struct json_object *jroot = json_object_new_object();

      json_object_object_add(jroot, "date", json_object_new_string(today_str));
      json_object_object_add(jroot, "image", json_object_new_string(now_str));
      json_object_object_add(jroot, "hdr", json_object_new_string(now_str));
      json_object_object_add(jroot, "image_timestamp",
                             json_object_new_int(now));
      json_object_object_add(jroot, "timezone_offset",
                             json_object_new_int(timezone_offset));
      if (config.panorama)
        json_object_object_add(jroot, "panorama",
                               json_object_new_string(now_str));
      json_object_object_add(jroot, "filetype", json_object_new_string("jpg"));

      char latest_json_filepath[PATH_MAX + 1];
      if (allsky_safe_snprintf(latest_json_filepath,
                               sizeof(latest_json_filepath),
                               "%s/latest_image.json", images_dir))
        fprintf(stderr, "WARNING: String %s truncated\n", latest_json_filepath);
      FILE *file = fopen(latest_json_filepath, "w");
      if (file) {
        // fprintf(file, "%s\n", json_object_to_json_string_ext(jroot,
        // JSON_C_TO_STRING_NOSLASHESCAPE));
        fprintf(file, "%s\n",
                json_object_to_json_string_ext(jroot, JSON_C_TO_STRING_SPACED));
        // fprintf(file, "%s\n", json_object_to_json_string_ext(jroot,
        // JSON_C_TO_STRING_PRETTY));
        fclose(file);
      }

      json_object_put(jroot);

      /* Add data to database */
      char db_filepath[PATH_MAX + 1];
      if (allsky_safe_snprintf(db_filepath, sizeof(db_filepath),
                               "%s/database_%s.db", database_dir, today_str))
        fprintf(stderr, "WARNING: String %s truncated\n", db_filepath);

      if (create_daily_database(db_filepath) != 0) {
        fprintf(stderr, "ERROR: Database database_%s.db creation failed.\n",
                today_str);
        continue;
      }

      if (insert_measurement(
              db_filepath, now, timezone_offset, frames[0].exposure, gain,
              frames[0].median_g, frames[0].median_r, frames[0].median_g,
              frames[0].median_b, frames[0].sigma_noise, use_max_images,
              is_night, num_stars, estimated_sqm, sharpness) != 0) {
        fprintf(stderr, "SQL insert failed\n");
      }
      printf("Data added to database %s\n", db_filepath);

      /* Run AI processing */
      if (run_ai_processing(latest_image_filepath) != 0) {
        fprintf(stderr, "ERROR: Failed to run AI processing.\n");
      }
      printf("AI processing started\n");

      // Reset debug pipeline images flag at end of processing pipeline
      debug_pipeline_images = 0;

      //	indigo_device_disconnect(&client, indigo_camera);

      printf("\nWaiting");
      fflush(stdout);
    }

    sleep(1);
    printf(".");
    fflush(stdout);
  }

  cleanup_resources();

  return EXIT_SUCCESS;
}
