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
#include "config.h"
#include "coordinate_parser.h"
#include "meteo_database.h"
#include "open_meteo_fetch.h"
#include <getopt.h>
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int debug = 0;

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

  return timegm(&t);
}

/**
 * Print usage help
 */
void print_help(const char *progname) {
  printf("Usage: %s [OPTIONS]\n", progname);
  printf("  -c, --config      Configuration file\n");
  printf(
      "  -t, --timestamp   Optional UTC timestamp (format: YYYYMMDDhhmmss)\n");
  printf("  -f, --outfile     Write output to file\n");
  printf("  -d, --databasedir Database directory\n");
  printf("  -v, --debug       Enable debug output\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'},
      {"timestamp", required_argument, 0, 't'},
      {"debug", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
      {"outfile", required_argument, 0, 'f'},
      {"databasedir", required_argument, 0, 'd'},

      {0, 0, 0, 0}};

  double latitude = 0.0, longitude = 0.0;
  time_t timestamp = time(NULL);
  char *timestamp_str = NULL;
  char *outfile = NULL;
  char *database_dir = NULL;
  const char *config_path = NULL;

  int opt;
  while ((opt = getopt_long(argc, argv, "c:t:f:d:vh", long_options, NULL)) !=
         -1) {
    switch (opt) {
    case 'c':
      config_path = optarg;
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
  printf("-----------------------\n");

  latitude = config.latitude;
  longitude = config.longitude;

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

  open_meteo_weather_t weather;
  char *json_raw = NULL;

  if (fetch_open_meteo_weather(latitude, longitude, timestamp, &json_raw) !=
      0) {
    fprintf(stderr, "Fehler beim Abrufen der Wetterdaten\n");

    return 1;
  }

  if (debug) {
    printf("\nResponse:\n%s\n\n", json_raw);
  }

  int ret = parse_open_meteo_weather(json_raw, time(NULL), &weather);
  if (ret != 0) {
    printf("ret=%d\n", ret);
    fprintf(stderr, "Fehler beim Parsen der Wetterdaten\n");
    if (!json_raw)
      free(json_raw);

    return 0;
  }
  free(json_raw);

  json_object *jout = json_object_new_object();

  json_object_object_add(jout, "source", json_object_new_string("Open-Meteo"));

  json_object_object_add(
      jout, "temperature",
      json_object_new_double(round_to_precision(weather.temperature, 1)));
  json_object_object_add(
      jout, "humidity",
      json_object_new_double(round_to_precision(weather.humidity, 0)));
  json_object_object_add(
      jout, "dew_point",
      json_object_new_double(round_to_precision(weather.dew_point, 1)));
  json_object_object_add(
      jout, "pressure_msl",
      json_object_new_double(round_to_precision(weather.pressure_msl, 1)));
  json_object_object_add(
      jout, "surface_pressure",
      json_object_new_double(round_to_precision(weather.surface_pressure, 1)));
  json_object_object_add(
      jout, "cloud_cover",
      json_object_new_double(round_to_precision(weather.cloud_cover, 0)));
  json_object_object_add(
      jout, "cloud_low",
      json_object_new_double(round_to_precision(weather.cloud_low, 0)));
  json_object_object_add(
      jout, "cloud_mid",
      json_object_new_double(round_to_precision(weather.cloud_mid, 0)));
  json_object_object_add(
      jout, "cloud_high",
      json_object_new_double(round_to_precision(weather.cloud_high, 0)));
  json_object_object_add(
      jout, "visibility",
      json_object_new_double(round_to_precision(weather.visibility, 0)));
  json_object_object_add(
      jout, "wind_speed_10m",
      json_object_new_double(round_to_precision(weather.wind_speed_10m, 1)));
  json_object_object_add(
      jout, "wind_dir_10m",
      json_object_new_double(round_to_precision(weather.wind_dir_10m, 0)));
  json_object_object_add(
      jout, "wind_speed_300hPa",
      json_object_new_double(round_to_precision(weather.wind_speed_300hPa, 1)));

  const char *json_str =
      json_object_to_json_string_ext(jout, JSON_C_TO_STRING_PRETTY);

  if (outfile) {
    FILE *fp = fopen(outfile, "w");
    if (!fp) {
      fprintf(stderr, "Failed to write Open-Meteo JSON to file: %s\n", outfile);
      return 1;
    } else {
      fprintf(fp, "%s\n", json_str);
      fclose(fp);
      if (debug) {
        printf("Open-Meteo JSON written to file: %s\n", outfile);
      }
    }
  }

  // Always print to stdout
  printf("%s\n", json_str);

  json_object_put(jout);

  if (database_dir) {
    /* Today string */
    struct tm *tm_info = localtime(&timestamp);
    char today_str[40];
    snprintf(today_str, sizeof(today_str), "%04d%02d%02d",
             tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday);

    /* Add data to database */
    char db_path[512];
    snprintf(db_path, sizeof(db_path), "%s/database_%s.db", database_dir,
             today_str);

    if (create_meteo_database(db_path) != 0) {
      fprintf(stderr, "Database creation failed\n");
      return 1;
    }

    if (insert_measurement(
            db_path, timestamp, weather.temperature, weather.humidity,
            weather.dew_point, weather.pressure_msl, weather.surface_pressure,
            weather.cloud_cover, weather.cloud_low, weather.cloud_mid,
            weather.cloud_high, weather.visibility, weather.wind_speed_10m,
            weather.wind_dir_10m, weather.wind_speed_300hPa) != 0) {
      fprintf(stderr, "SQL insert failed\n");
      return 1;
    }

    printf("Data added to database %s\n", db_path);
  }

  return 0;
}
