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
#include "aurora_database.h"
#include "aurora_fetch.h"
#include "config.h"
#include "coordinates.h"
#include <getopt.h>
#include <math.h>
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int debug = 0;
int fetch_kp = 0;
int fetch_mag = 0;
int fetch_plasma = 0;

#ifndef HAVE_TIMEGM
time_t timegm(struct tm *tm) {
  char *tz = getenv("TZ");
  setenv("TZ", "", 1); // UTC
  tzset();
  time_t t = mktime(tm);
  if (tz)
    setenv("TZ", tz, 1);
  else
    unsetenv("TZ");
  tzset();
  return t;
}
#endif

static double round_to_precision(double value, int precision);

/* Missing values (fetch failed, no current data) are NAN and written as JSON
 * null / SQL NULL, never as 0 (a valid Bz) or -1. */
static json_object *json_number_or_null(double value, int precision) {
  return isnan(value) ? NULL : json_object_new_double(round_to_precision(value, precision));
}

/* Fetch url and print a size and a short excerpt with --debug: the RTSW
 * files are megabytes of JSON. */
static int fetch_json(const char *url, char **out_json) {
  int rc = fetch_aurora_json(url, out_json);

  if (rc != 0) {
    fprintf(stderr, "Failed to fetch %s (code %d)\n", url, rc);
    return rc;
  }
  if (debug)
    printf("\n%s: %zu bytes\n%.300s%s\n", url, strlen(*out_json), *out_json,
           strlen(*out_json) > 300 ? " ..." : "");
  return 0;
}

static double round_to_precision(double value, int precision) {
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%.*f", precision,
           value); // Round to precision

  return strtod(buffer, NULL);
}

/**
 * Parse timestamp string in format YYYYMMDDhhmmss (UTC)
 */
time_t parse_timestamp(const char *str) {
  struct tm t = {0};

  if (strptime(str, "%Y%m%d%H%M%S", &t) == NULL) {
    fprintf(stderr, "Invalid timestamp format. Use YYYYMMDDhhmmss.\n");
    return -1;
  }

  t.tm_isdst = -1;

  return timegm(&t); // returns UTC time_t
}

/**
 * Print usage help
 */
void print_help(const char *progname) {
  printf("Usage: %s [OPTIONS]\n", progname);
  printf("  -c, --config      Configuration file\n");
  printf("  -l, --longitude   Longitude of observation site (e.g. 7.0 or "
         "122.5W)\n");
  printf("  -b, --latitude    Latitude of observation site (e.g. 50.2 or "
         "60.0N)\n");
  printf(
      "  -t, --timestamp   Optional UTC timestamp (format: YYYYMMDDhhmmss)\n");
  printf("  -f, --outfile     Write output to file\n");
  printf("      --kp            Include planetary Kp index\n");
  printf("      --mag           Include solar wind magnetic field (Bt, Bz)\n");
  printf("      --plasma        Include solar wind plasma data (density, speed, temperature)\n");
  printf("  Values that are not requested or not available are null.\n");
  printf("  -d, --databasedir Database directory\n");
  printf("  -v, --debug       Enable debug output\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'},
      {"longitude", required_argument, 0, 'l'},
      {"latitude", required_argument, 0, 'b'},
      {"timestamp", required_argument, 0, 't'},
      {"databasedir", required_argument, 0, 'd'},
      {"debug", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
      {"outfile", required_argument, 0, 'f'},
      {"kp", no_argument, &fetch_kp, 1},
      {"mag", no_argument, &fetch_mag, 1},
      {"plasma", no_argument, &fetch_plasma, 1},
      {0, 0, 0, 0}};

  double latitude = 0.0, longitude = 0.0;
  time_t timestamp = time(NULL);
  int lat_set = 0, lon_set = 0;
  char *timestamp_str = NULL;
  char *outfile = NULL;
  char *database_dir = NULL;
  const char *config_path = NULL;

  int opt;
  while ((opt = getopt_long(argc, argv, "c:l:b:t:f:d:vh", long_options,
                            NULL)) != -1) {
    switch (opt) {
    case 'c':
      config_path = optarg;
      break;
    case 'l':
      if (parse_lat_lon(optarg, 0, &longitude) != 0) {
        fprintf(stderr, "Invalid longitude format: %s\n", optarg);
        return 1;
      }
      lon_set = 1;
      break;
    case 'b':
      if (parse_lat_lon(optarg, 1, &latitude) != 0) {
        fprintf(stderr, "Invalid latitude format: %s\n", optarg);
        return 1;
      }
      lat_set = 1;
      break;
    case 't':
      timestamp_str = optarg;
      break;
    case 'f':
      outfile = optarg;
      break;
    case 'd':
      database_dir = optarg;
      break;
    case 'v':
      debug = 1;
      break;
    case 'h':
      print_help(argv[0]);
      return 0;
    case 0:
      break; // Flag options (like --kp)
    default:
      print_help(argv[0]);
      return 1;
    }
  }

  if (!config_path) {
    fprintf(stderr, "ERROR: No configuration file. Call: %s --config <pfad>\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  /* Load configuration */
  config_t config;
  if (load_config(config_path, &config) != 0) {
    fprintf(stderr, "ERROR: Failed to load configuration file %s.\n",
            config_path);
    return EXIT_FAILURE;
  }

  if (!lat_set || !lon_set) {
    latitude = config.location_latitude;
    longitude = config.location_longitude;
  }

  if (timestamp_str) {
    timestamp = parse_timestamp(timestamp_str);
    if (timestamp == (time_t)-1)
      return 1;
  }

  if (debug) {
    printf("Latitude:  %.6f\n", latitude);
    printf("Longitude: %.6f\n", longitude);
    printf("Timestamp: %ld (%s)\n", timestamp, ctime(&timestamp));
  }

  char *json_raw = NULL;
  int best_val = -1, max_val = -1;
  double avg_val = -1.0;
  time_t obs_time = 0, fore_time = 0;
  double kp_val = NAN, bt_val = NAN, bz_val = NAN, density = NAN, speed = NAN,
         temp = NAN;

  /* OVATION aurora probability (required) */
  if (fetch_json(URL_OVATION, &json_raw) != 0)
    return 2;

  if (parse_noaa_json(json_raw, latitude, longitude, timestamp, &best_val, &max_val,
                      &avg_val, &obs_time, &fore_time) != 0) {
    fprintf(stderr, "Failed to parse NOAA OVATION data.\n");
    free(json_raw);
    return 3;
  }
  free(json_raw);
  json_raw = NULL;

  /* Optional values: a failure only leaves the value missing */
  if (fetch_kp) {
    /* 1-minute estimate covers ~6 hours, the 3-hourly product 7 days */
    if (fetch_json(URL_KP_1M, &json_raw) == 0 &&
        parse_kp_json(json_raw, timestamp, &kp_val) != 0)
      kp_val = NAN;
    free(json_raw);
    json_raw = NULL;

    if (isnan(kp_val)) {
      if (fetch_json(URL_KP_3H, &json_raw) == 0 &&
          parse_kp_json(json_raw, timestamp, &kp_val) != 0)
        kp_val = NAN;
      free(json_raw);
      json_raw = NULL;
    }
    if (isnan(kp_val))
      fprintf(stderr, "No Kp index available for the requested time.\n");
    else if (debug)
      printf("Kp index: %.2f\n", kp_val);
  }

  if (fetch_mag) {
    if (fetch_json(URL_RTSW_MAG, &json_raw) == 0 &&
        parse_mag_json(json_raw, timestamp, &bt_val, &bz_val) != 0) {
      bt_val = bz_val = NAN;
      fprintf(stderr, "No current solar wind magnetic field data.\n");
    }
    free(json_raw);
    json_raw = NULL;
    if (debug && !isnan(bt_val))
      printf("Bt: %.1f nT, Bz: %.1f nT\n", bt_val, bz_val);
  }

  if (fetch_plasma) {
    if (fetch_json(URL_RTSW_WIND, &json_raw) == 0 &&
        parse_plasma_json(json_raw, timestamp, &density, &speed, &temp) != 0) {
      density = speed = temp = NAN;
      fprintf(stderr, "No current solar wind plasma data.\n");
    }
    free(json_raw);
    json_raw = NULL;
    if (debug && !isnan(speed))
      printf("Density: %.1f cm^-3, Speed: %.1f km/s, Temp: %.0f K\n", density,
             speed, temp);
  }

  json_object *jout = json_object_new_object();

  // NOAA Daten
  json_object_object_add(jout, "source", json_object_new_string("NOAA"));
  json_object_object_add(jout, "observation_time",
                         json_object_new_int64(obs_time));
  json_object_object_add(jout, "forecast_time",
                         json_object_new_int64(fore_time));
  json_object_object_add(jout, "probability_percent",
                         json_object_new_int(best_val));
  json_object_object_add(jout, "probability_max", json_object_new_int(max_val));
  json_object_object_add(
      jout, "probability_avg",
      json_object_new_double(round_to_precision(avg_val, 1)));

  json_object_object_add(jout, "kp_index", json_number_or_null(kp_val, 1));
  json_object_object_add(jout, "bt", json_number_or_null(bt_val, 1));
  json_object_object_add(jout, "bz", json_number_or_null(bz_val, 1));
  json_object_object_add(jout, "density", json_number_or_null(density, 1));
  json_object_object_add(jout, "speed", json_number_or_null(speed, 0));
  json_object_object_add(jout, "temperature", json_number_or_null(temp, 0));

  const char *json_str =
      json_object_to_json_string_ext(jout, JSON_C_TO_STRING_PRETTY);

  if (outfile) {
    FILE *fp = fopen(outfile, "w");

    if (!fp) {
      fprintf(stderr, "Failed to write JSON to file: %s\n", outfile);
      return 1;
    } else {
      fprintf(fp, "%s\n", json_str);
      fclose(fp);

      if (debug) {
        printf("JSON written to file: %s\n", outfile);
      }
    }
  }

  // Always print to stdout
  printf("%s\n", json_str);

  json_object_put(jout);

  if (database_dir) {
    /* Today string */
    /* The daily database is named after the UTC day, like the one of the
     * camera and the sensors: a local day would put rows into another
     * file than the camera's, and move them on a time zone change. */
    struct tm tm_utc;
    gmtime_r(&timestamp, &tm_utc);
    char today_str[40];
    snprintf(today_str, sizeof(today_str), "%04d%02d%02d",
             tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday);

    /* Add data to database */
    char db_path[512];
    snprintf(db_path, sizeof(db_path), "%s/database_%s.db", database_dir,
             today_str);

    if (create_aurora_database(db_path) != 0) {
      fprintf(stderr, "Database creation failed\n");
      return 1;
    }

    if (insert_measurement(db_path, timestamp, best_val,
                           max_val, avg_val,
                           kp_val, bt_val, bz_val,
                           density, speed,
                           temp) != 0) {
      fprintf(stderr, "SQL insert failed\n");
      return 1;
    }
  }

  return 0;
}
