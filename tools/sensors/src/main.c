/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
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
/*
 * allsky360-sensors: reads the RS485/Modbus sensors of the station once and
 * stores the values in the daily database (table 'sensors'). Started
 * periodically by allsky360-sensors.timer.
 *
 * Sensors (addresses and registers in config.ini, keys modbus_*):
 *  - weather sensor Halisense HLS-UWS-SDTHA: temperature, humidity, pressure,
 *    wind, noise, PM2.5/PM10, illuminance, rain
 *  - anemometer (wind speed) and wind vane (wind direction)
 *  - temperature/humidity sensor in the control cabinet
 *
 * Exit status: 0 = at least one sensor read (or none configured),
 *              2 = no configured sensor answered, 1 = configuration, bus or
 *              database error.
 */
#include <getopt.h>
#include <json-c/json.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "atomic_file.h"
#include "modbus_rtu.h"
#include "sensors_config.h"
#include "sensors_database.h"

/* Register layout of the weather sensor, offsets from modbus_weather_register
 * (DFRobot SEN0658 / HLS-UWS-SDTHA, see examples/weather-sensor) */
#define WS_WIND_SPEED 0    /* / 10 = m/s */
#define WS_WIND_DIR_DEG 3  /* 0..360 deg */
#define WS_HUMIDITY 4      /* / 10 = %RH */
#define WS_TEMPERATURE 5   /* signed, / 10 = deg C */
#define WS_NOISE 6         /* / 10 = dB */
#define WS_PM25 7          /* ug/m3 */
#define WS_PM10 8          /* ug/m3 */
#define WS_PRESSURE 9      /* hPa */
#define WS_ILLUMINANCE 10  /* 32 bit, high word first */
#define WS_RAINFALL 13     /* / 10 = mm */
#define WS_SOLAR 14        /* unverified, raw */
#define WS_BLOCK_COUNT 15

static int verbose = 0;

/* Plausibility limits: values outside are sensor or wiring errors and are
 * stored as NULL rather than as a wrong number. */
static double plausible(const char *name, double v, double min, double max) {
  if (isnan(v) || (v >= min && v <= max))
    return v;
  fprintf(stderr, "WARNING: %s = %g outside %g..%g, discarded\n", name, v, min, max);
  return NAN;
}

/* Magnus formula, valid for -45..60 deg C */
static double dew_point(double t, double rh) {
  if (isnan(t) || isnan(rh) || rh <= 0.0)
    return NAN;
  const double a = 17.62, b = 243.12;
  double g = log(rh / 100.0) + a * t / (b + t);
  return b * g / (a - g);
}

static const char *modbus_error(int rc) {
  switch (rc) {
  case MODBUS_ERR_TIMEOUT: return "no valid answer";
  case MODBUS_ERR_EXCEPTION: return "Modbus exception";
  default: return "invalid request";
  }
}

static int read_weather(modbus_bus_t *bus, const sensors_config_t *c,
                        sensors_reading_t *r) {
  uint16_t v[WS_BLOCK_COUNT];
  int rc = modbus_read_registers(bus, (uint8_t)c->weather_address,
                                 (uint8_t)c->function,
                                 (uint16_t)c->weather_register, WS_BLOCK_COUNT, v);
  if (rc != MODBUS_OK) {
    fprintf(stderr, "WARNING: weather sensor (address %d): %s\n",
            c->weather_address, modbus_error(rc));
    return -1;
  }

  r->ws_wind_speed = plausible("weather wind speed", v[WS_WIND_SPEED] / 10.0, 0, 60);
  double dir = plausible("weather wind direction", v[WS_WIND_DIR_DEG], 0, 360);
  if (!isnan(dir))
    r->ws_wind_direction = fmod(dir + c->weather_wind_direction_offset, 360.0);
  r->humidity = plausible("humidity", v[WS_HUMIDITY] / 10.0, 0, 100);
  r->temperature = plausible("temperature", (int16_t)v[WS_TEMPERATURE] / 10.0, -50, 80);
  r->noise = plausible("noise", v[WS_NOISE] / 10.0, 0, 150);
  r->pm25 = v[WS_PM25];
  r->pm10 = v[WS_PM10];
  r->pressure = plausible("pressure", v[WS_PRESSURE], 300, 1100);
  r->illuminance = (double)(((uint32_t)v[WS_ILLUMINANCE] << 16) | v[WS_ILLUMINANCE + 1]);
  r->rainfall = v[WS_RAINFALL] / 10.0;
  r->solar_irradiance_raw = v[WS_SOLAR];
  r->dew_point = dew_point(r->temperature, r->humidity);
  return 0;
}

static int read_wind_speed(modbus_bus_t *bus, const sensors_config_t *c,
                           sensors_reading_t *r) {
  uint16_t raw;
  int rc = modbus_read_registers(bus, (uint8_t)c->wind_speed_address,
                                 (uint8_t)c->function,
                                 (uint16_t)c->wind_speed_register, 1, &raw);
  if (rc != MODBUS_OK) {
    fprintf(stderr, "WARNING: anemometer (address %d): %s\n",
            c->wind_speed_address, modbus_error(rc));
    return -1;
  }
  r->wind_speed = plausible("wind speed", raw / c->wind_speed_scale, 0, 60);
  return isnan(r->wind_speed) ? -1 : 0;
}

static int read_wind_direction(modbus_bus_t *bus, const sensors_config_t *c,
                               sensors_reading_t *r) {
  uint16_t raw;
  int rc = modbus_read_registers(bus, (uint8_t)c->wind_direction_address,
                                 (uint8_t)c->function,
                                 (uint16_t)c->wind_direction_register, 1, &raw);
  if (rc != MODBUS_OK) {
    fprintf(stderr, "WARNING: wind vane (address %d): %s\n",
            c->wind_direction_address, modbus_error(rc));
    return -1;
  }
  if (isnan(plausible("wind direction", raw, 0, 360)))
    return -1;
  r->wind_direction = (double)((raw % 360 + c->wind_direction_offset) % 360);
  return 0;
}

static int read_cabinet(modbus_bus_t *bus, const sensors_config_t *c,
                        sensors_reading_t *r) {
  uint16_t v[2];
  int rc = modbus_read_registers(bus, (uint8_t)c->cabinet_address,
                                 (uint8_t)c->function,
                                 (uint16_t)c->cabinet_register, 2, v);
  if (rc != MODBUS_OK) {
    fprintf(stderr, "WARNING: cabinet sensor (address %d): %s\n",
            c->cabinet_address, modbus_error(rc));
    return -1;
  }
  r->cabinet_humidity = plausible("cabinet humidity", v[0] / 10.0, 0, 100);
  r->cabinet_temperature = plausible("cabinet temperature", (int16_t)v[1] / 10.0, -40, 85);
  r->cabinet_dew_point = dew_point(r->cabinet_temperature, r->cabinet_humidity);
  return 0;
}

static void json_add(json_object *o, const char *key, double value) {
  json_object_object_add(o, key, isnan(value) ? NULL : json_object_new_double(value));
}

static char *reading_to_json(const sensors_reading_t *r) {
  json_object *o = json_object_new_object();
  json_object_object_add(o, "source", json_object_new_string("modbus"));
  json_object_object_add(o, "timestamp", json_object_new_int64((int64_t)r->timestamp));
  json_add(o, "temperature", r->temperature);
  json_add(o, "humidity", r->humidity);
  json_add(o, "dew_point", r->dew_point);
  json_add(o, "pressure", r->pressure);
  json_add(o, "ws_wind_speed", r->ws_wind_speed);
  json_add(o, "ws_wind_direction", r->ws_wind_direction);
  json_add(o, "noise", r->noise);
  json_add(o, "pm25", r->pm25);
  json_add(o, "pm10", r->pm10);
  json_add(o, "illuminance", r->illuminance);
  json_add(o, "rainfall", r->rainfall);
  json_add(o, "solar_irradiance_raw", r->solar_irradiance_raw);
  json_add(o, "wind_speed", r->wind_speed);
  json_add(o, "wind_direction", r->wind_direction);
  json_add(o, "cabinet_temperature", r->cabinet_temperature);
  json_add(o, "cabinet_humidity", r->cabinet_humidity);
  json_add(o, "cabinet_dew_point", r->cabinet_dew_point);
  char *text = strdup(json_object_to_json_string_ext(o, JSON_C_TO_STRING_PRETTY));
  json_object_put(o);
  return text;
}

static void usage(const char *prog) {
  printf("Usage: %s -c CONFIG [options]\n"
         "  -c, --config FILE      config.ini (modbus_* keys, database_directory)\n"
         "  -d, --databasedir DIR  override database_directory\n"
         "  -f, --outfile FILE     also write the reading as JSON (atomic)\n"
         "  -n, --dry-run          read and print, do not write the database\n"
         "  -v, --verbose          show the configuration and every retry\n"
         "  -h, --help             this help\n",
         prog);
}

int main(int argc, char *argv[]) {
  const char *config_path = NULL, *database_dir = NULL, *outfile = NULL;
  int dry_run = 0;

  static struct option long_opts[] = {
      {"config", required_argument, 0, 'c'},
      {"databasedir", required_argument, 0, 'd'},
      {"outfile", required_argument, 0, 'f'},
      {"dry-run", no_argument, 0, 'n'},
      {"verbose", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};
  int opt;
  while ((opt = getopt_long(argc, argv, "c:d:f:nvh", long_opts, NULL)) != -1) {
    switch (opt) {
    case 'c': config_path = optarg; break;
    case 'd': database_dir = optarg; break;
    case 'f': outfile = optarg; break;
    case 'n': dry_run = 1; break;
    case 'v': verbose = 1; break;
    case 'h': usage(argv[0]); return 0;
    default: usage(argv[0]); return 1;
    }
  }
  if (!config_path) {
    usage(argv[0]);
    return 1;
  }

  sensors_config_t config;
  sensors_config_defaults(&config);
  if (sensors_config_load(config_path, &config) != 0)
    return 1;
  if (database_dir)
    snprintf(config.database_directory, sizeof(config.database_directory), "%s",
             database_dir);
  if (verbose)
    sensors_config_print(&config);

  int enabled = (config.weather_address > 0) + (config.wind_speed_address > 0) +
                (config.wind_direction_address > 0) + (config.cabinet_address > 0);
  if (enabled == 0) {
    printf("No Modbus sensor configured (all modbus_*_address are 0), nothing to do.\n");
    return 0;
  }

  modbus_bus_t bus;
  if (modbus_open(&bus, config.device, config.baudrate, config.gpio_chip,
                  config.gpio_line, config.timeout_ms, config.retries) != 0)
    return 1;
  bus.verbose = verbose;

  sensors_reading_t r;
  sensors_reading_init(&r, time(NULL));
  int ok = 0;
  if (config.weather_address > 0 && read_weather(&bus, &config, &r) == 0)
    ok++;
  if (config.wind_speed_address > 0 && read_wind_speed(&bus, &config, &r) == 0)
    ok++;
  if (config.wind_direction_address > 0 && read_wind_direction(&bus, &config, &r) == 0)
    ok++;
  if (config.cabinet_address > 0 && read_cabinet(&bus, &config, &r) == 0)
    ok++;
  modbus_close(&bus);

  char *json = reading_to_json(&r);
  if (dry_run || verbose)
    printf("%s\n", json);
  printf("%d of %d sensors read\n", ok, enabled);

  int ret = ok > 0 ? 0 : 2;

  if (ok > 0 && outfile && write_text_file_atomic(outfile, json) != 0) {
    fprintf(stderr, "ERROR: cannot write %s\n", outfile);
    ret = 1;
  }
  free(json);

  if (ok > 0 && !dry_run) {
    /* One database per UTC day, like the camera daemon */
    struct tm tm_utc;
    gmtime_r(&r.timestamp, &tm_utc);
    char db_path[512];
    snprintf(db_path, sizeof(db_path), "%s/database_%04d%02d%02d.db",
             config.database_directory, tm_utc.tm_year + 1900, tm_utc.tm_mon + 1,
             tm_utc.tm_mday);
    if (sensors_database_store(db_path, &r) != 0)
      ret = 1;
    else if (verbose)
      printf("stored in %s\n", db_path);
  }
  return ret;
}
