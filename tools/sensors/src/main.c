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
 * allsky360-sensors: reads the RS485/Modbus sensors of the station and
 * stores the values in the daily database (table 'sensors').
 *
 * Sensors (addresses and registers in config.ini, keys modbus_*):
 *  - weather sensor Halisense HLS-UWS-SDTHA: temperature, humidity, pressure,
 *    wind, noise, PM2.5/PM10, illuminance, rain
 *  - anemometer (wind speed) and wind vane (wind direction)
 *  - temperature/humidity sensor in the control cabinet
 *
 * Daemon mode (--daemon, allsky360-sensors.service): anemometer and vane are
 * sampled every modbus_wind_sample_interval seconds into a ring buffer in RAM (wind_average.c); every
 * modbus_interval seconds the other sensors are read and one row is stored
 * with the wind mean over modbus_wind_average_minutes (scalar mean speed,
 * speed-weighted vector mean direction). Only that row touches the disk.
 *
 * Without --daemon: one reading of every sensor, instantaneous values, for
 * manual checks (the daemon holds the bus lock, stop it first).
 *
 * Calm: samples below modbus_wind_calm_speed do not count for the direction
 * (the vane turns at random); without enough wind the last direction is kept
 * (in the daemon from RAM, at start and in single mode from today's or
 * yesterday's database).
 *
 * Exit status: 0 = at least one sensor read (or none configured; daemon:
 *              stopped by a signal), 2 = no configured sensor answered,
 *              1 = configuration, bus or database error.
 */
#include <errno.h>
#include <getopt.h>
#include <json-c/json.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "atomic_file.h"
#include "modbus_rtu.h"
#include "sensors_config.h"
#include "sensors_database.h"
#include "wind_average.h"

/* Daemon: seconds of samples before the first stored reading, and the least
 * windy time in the window (seconds, as samples) that gives a mean direction */
#define DAEMON_WARMUP_S 120
#define MIN_DIRECTION_SECONDS 30

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

/* Implausible register value (wind samples, reported by the caller) */
#define SAMPLE_IMPLAUSIBLE -100

/* One anemometer sample without any output (the daemon reads it every
 * second and reports failures itself). @return MODBUS_OK, a MODBUS_ERR_* or
 * SAMPLE_IMPLAUSIBLE */
static int sample_wind_speed(modbus_bus_t *bus, const sensors_config_t *c, double *speed) {
  uint16_t raw;
  *speed = NAN;
  int rc = modbus_read_registers(bus, (uint8_t)c->wind_speed_address,
                                 (uint8_t)c->function,
                                 (uint16_t)c->wind_speed_register, 1, &raw);
  if (rc != MODBUS_OK)
    return rc;
  double v = raw / c->wind_speed_scale;
  if (v < 0.0 || v > 60.0)
    return SAMPLE_IMPLAUSIBLE;
  *speed = v;
  return MODBUS_OK;
}

static int sample_wind_direction(modbus_bus_t *bus, const sensors_config_t *c,
                                 double *direction) {
  uint16_t raw;
  *direction = NAN;
  int rc = modbus_read_registers(bus, (uint8_t)c->wind_direction_address,
                                 (uint8_t)c->function,
                                 (uint16_t)c->wind_direction_register, 1, &raw);
  if (rc != MODBUS_OK)
    return rc;
  if (raw > 360)
    return SAMPLE_IMPLAUSIBLE;
  *direction = (double)((raw % 360 + c->wind_direction_offset) % 360);
  return MODBUS_OK;
}

static const char *sample_error(int rc) {
  return rc == SAMPLE_IMPLAUSIBLE ? "value outside the plausible range" : modbus_error(rc);
}

static int read_wind_speed(modbus_bus_t *bus, const sensors_config_t *c,
                           sensors_reading_t *r) {
  int rc = sample_wind_speed(bus, c, &r->wind_speed);
  if (rc != MODBUS_OK)
    fprintf(stderr, "WARNING: anemometer (address %d): %s\n", c->wind_speed_address,
            sample_error(rc));
  return rc == MODBUS_OK ? 0 : -1;
}

static int read_wind_direction(modbus_bus_t *bus, const sensors_config_t *c,
                               sensors_reading_t *r) {
  int rc = sample_wind_direction(bus, c, &r->wind_direction);
  if (rc != MODBUS_OK)
    fprintf(stderr, "WARNING: wind vane (address %d): %s\n", c->wind_direction_address,
            sample_error(rc));
  return rc == MODBUS_OK ? 0 : -1;
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

/* Daily database of a UTC day, like the camera daemon */
static void database_path(char *buf, size_t size, const char *dir, time_t t) {
  struct tm tm_utc;
  gmtime_r(&t, &tm_utc);
  snprintf(buf, size, "%s/database_%04d%02d%02d.db", dir, tm_utc.tm_year + 1900,
           tm_utc.tm_mon + 1, tm_utc.tm_mday);
}

static void format_utc(char *buf, size_t size, time_t t) {
  struct tm tm_utc;
  gmtime_r(&t, &tm_utc);
  strftime(buf, size, "%Y-%m-%d %H:%M:%S UTC", &tm_utc);
}

/* Last stored wind direction before 'before': today's database, then
 * yesterday's (first reading after 00:00 UTC). @return 0 if found */
static int last_stored_direction(const sensors_config_t *c, time_t before,
                                 double *direction, time_t *when) {
  for (int day = 0; day < 2; day++) {
    char path[512];
    database_path(path, sizeof(path), c->database_directory, before - day * 86400);
    if (sensors_database_last_value(path, "wind_direction", before, direction, when) == 0)
      return 0;
  }
  return -1;
}

/* Single mode, calm: replace the vane reading by the last stored direction.
 * Without an earlier direction (first run in calm) it stays NULL, since the
 * measured value is random. Without a wind speed (anemometer failed) the
 * measured direction is kept. */
static void hold_direction_in_calm(const sensors_config_t *c, sensors_reading_t *r) {
  if (c->wind_calm_speed <= 0.0 || isnan(r->wind_speed) || isnan(r->wind_direction) ||
      r->wind_speed >= c->wind_calm_speed)
    return;

  double measured = r->wind_direction, last;
  time_t last_t;
  int found = last_stored_direction(c, r->timestamp, &last, &last_t);
  r->wind_direction = found == 0 ? last : NAN;

  if (verbose) {
    if (found == 0) {
      char when[32];
      format_utc(when, sizeof(when), last_t);
      printf("calm (%.1f m/s < %g): wind direction %.0f deg kept from %s, measured %.0f deg ignored\n",
             r->wind_speed, c->wind_calm_speed, last, when, measured);
    } else {
      printf("calm (%.1f m/s < %g): no earlier wind direction, measured %.0f deg ignored\n",
             r->wind_speed, c->wind_calm_speed, measured);
    }
  }
}

static void json_add(json_object *o, const char *key, double value) {
  json_object_object_add(o, key, isnan(value) ? NULL : json_object_new_double(value));
}

static char *reading_to_json(const sensors_config_t *c, const sensors_reading_t *r) {
  json_object *o = json_object_new_object();
  json_object_object_add(o, "source", json_object_new_string("modbus"));
  json_object_object_add(o, "timestamp", json_object_new_int64((int64_t)r->timestamp));
  /* Sensors with an address in config.ini: the web page hides the others
   * instead of showing their values as missing */
  json_object *cfg = json_object_new_object();
  json_object_object_add(cfg, "weather", json_object_new_boolean(c->weather_address > 0));
  json_object_object_add(cfg, "anemometer", json_object_new_boolean(c->wind_speed_address > 0));
  json_object_object_add(cfg, "wind_vane", json_object_new_boolean(c->wind_direction_address > 0));
  json_object_object_add(cfg, "cabinet", json_object_new_boolean(c->cabinet_address > 0));
  json_object_object_add(o, "configured", cfg);
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

/* JSON file and database row of one reading. @return 0, or 1 on an error */
static int store_reading(const sensors_config_t *c, const sensors_reading_t *r,
                         const char *outfile, int dry_run) {
  int ret = 0;
  char *json = reading_to_json(c, r);
  if (dry_run || verbose)
    printf("%s\n", json);
  if (outfile && write_text_file_atomic(outfile, json) != 0) {
    fprintf(stderr, "ERROR: cannot write %s\n", outfile);
    ret = 1;
  }
  free(json);

  if (!dry_run) {
    char db_path[512];
    database_path(db_path, sizeof(db_path), c->database_directory, r->timestamp);
    if (sensors_database_store(db_path, r) != 0)
      ret = 1;
    else if (verbose)
      printf("stored in %s\n", db_path);
  }
  return ret;
}

/* ---- Daemon ---------------------------------------------------------- */

static volatile sig_atomic_t stop_requested = 0;

static void on_signal(int sig) {
  (void)sig;
  stop_requested = 1;
}

/* Failures of the periodic wind samples. A single missed sample costs
 * nothing (the next one follows a second later), so failures are only counted
 * for the line of the next stored reading. Only a sensor that stays silent for
 * FAILING_LOG_S gets a warning, and then a line when it answers again: no log
 * lines for short dropouts in the journal, which also lives on the SD card. */
#define FAILING_LOG_S 60

typedef struct {
  const char *name;
  int address;
  time_t failing_since; /* 0 = last read was good */
  int warned;
  int failed; /* since the last stored reading */
} sample_stats_t;

static void count_sample(sample_stats_t *s, int rc, time_t now) {
  if (rc == MODBUS_OK) {
    if (s->warned)
      printf("%s (address %d) answers again\n", s->name, s->address);
    s->failing_since = 0;
    s->warned = 0;
    return;
  }
  s->failed++;
  if (!s->failing_since)
    s->failing_since = now;
  if (!s->warned && now - s->failing_since >= FAILING_LOG_S) {
    fprintf(stderr, "WARNING: %s (address %d): %s for %d s\n", s->name, s->address,
            sample_error(rc), (int)(now - s->failing_since));
    s->warned = 1;
  }
}

static time_t next_boundary(time_t t, int interval) { return (t / interval + 1) * interval; }

static double round_to(double v, double step) { return isnan(v) ? v : round(v / step) * step; }

static int run_daemon(modbus_bus_t *bus, const sensors_config_t *c, const char *outfile,
                      int dry_run) {
  const int have_speed = c->wind_speed_address > 0;
  const int have_direction = c->wind_direction_address > 0;
  const int window_s = c->wind_average_minutes * 60;

  wind_buffer_t *wind = malloc(sizeof(*wind));
  if (!wind) {
    fprintf(stderr, "ERROR: out of memory\n");
    return 1;
  }
  wind_buffer_init(wind);
  setvbuf(stdout, NULL, _IOLBF, 0); /* journal gets every line at once */

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = on_signal; /* no SA_RESTART: interrupts the sleep */
  sigaction(SIGTERM, &sa, NULL);
  sigaction(SIGINT, &sa, NULL);

  time_t start = time(NULL);
  /* Direction kept in calm air: starts with the last stored one, then lives
   * in RAM */
  double held_direction = NAN;
  time_t held_t;
  if (have_direction && last_stored_direction(c, start, &held_direction, &held_t) != 0)
    held_direction = NAN;

  sample_stats_t speed_stats = {"anemometer", c->wind_speed_address, 0, 0, 0};
  sample_stats_t direction_stats = {"wind vane", c->wind_direction_address, 0, 0, 0};

  time_t next_record = next_boundary(start + DAEMON_WARMUP_S, c->interval);
  char when[32];
  format_utc(when, sizeof(when), next_record);
  /* At least 3 samples, e.g. 30 s of wind = 3 samples at 10 s */
  int min_direction_samples =
      (MIN_DIRECTION_SECONDS + c->wind_sample_interval - 1) / c->wind_sample_interval;
  if (min_direction_samples < 3)
    min_direction_samples = 3;
  time_t next_sample = 0;

  printf("daemon: wind sampled every %d s, mean over %d min; a reading every %d s, "
         "first at %s\n",
         c->wind_sample_interval, c->wind_average_minutes, c->interval, when);

  struct timespec tick;
  clock_gettime(CLOCK_MONOTONIC, &tick);

  while (!stop_requested) {
    time_t now = time(NULL);

    if ((have_speed || have_direction) && now >= next_sample) {
      next_sample = now + c->wind_sample_interval;
      /* One attempt per sample: retries (up to modbus_retries x timeout) would
       * only block the loop for seconds, a missed sample costs little */
      double speed = NAN, direction = NAN;
      bus->retries = 1;
      if (have_speed)
        count_sample(&speed_stats, sample_wind_speed(bus, c, &speed), now);
      if (have_direction)
        count_sample(&direction_stats, sample_wind_direction(bus, c, &direction), now);
      bus->retries = c->retries; /* weather and cabinet sensor: with retries */
      wind_buffer_add(wind, now, speed, direction);
      if (verbose > 1)
        printf("sample: %.1f m/s, %.0f deg\n", speed, direction);
    }

    if (now >= next_record) {
      sensors_reading_t r;
      sensors_reading_init(&r, next_record);
      int ok = 0;
      if (c->weather_address > 0 && read_weather(bus, c, &r) == 0)
        ok++;
      if (c->cabinet_address > 0 && read_cabinet(bus, c, &r) == 0)
        ok++;

      wind_average_t avg;
      wind_buffer_average(wind, now, window_s, c->wind_calm_speed, min_direction_samples,
                          &avg);
      r.wind_speed = round_to(avg.speed, 0.01);
      ok += avg.speed_samples > 0;
      const char *direction_note = "";
      if (avg.direction_samples == 0) {
        r.wind_direction = NAN; /* vane silent (or not configured) */
      } else {
        ok++;
        if (!isnan(avg.direction)) {
          held_direction = r.wind_direction = round(avg.direction);
        } else {
          r.wind_direction = held_direction; /* calm: keep the last one */
          direction_note = isnan(held_direction) ? " (calm, no earlier direction)"
                                                 : " (calm, last direction kept)";
        }
      }

      int enabled = (c->weather_address > 0) + (c->cabinet_address > 0) + have_speed +
                    have_direction;
      format_utc(when, sizeof(when), r.timestamp);
      printf("%s: %d of %d sensors", when, ok, enabled);
      if (have_speed || have_direction)
        printf("; wind %d min: %.2f m/s (max %.1f), %.0f deg%s, consistency %.2f, "
               "%d of %d samples windy",
               c->wind_average_minutes, avg.speed, avg.speed_max, r.wind_direction,
               direction_note, avg.consistency, avg.used_for_direction, avg.samples);
      if (speed_stats.failed || direction_stats.failed)
        printf("; failed reads: anemometer %d, vane %d", speed_stats.failed,
               direction_stats.failed);
      printf("\n");
      speed_stats.failed = direction_stats.failed = 0;

      if (ok > 0)
        store_reading(c, &r, outfile, dry_run);
      next_record = next_boundary(now, c->interval);
    }

    /* Next second on the monotonic clock; after a long overrun (a sensor
     * that times out with retries) start again from now instead of catching
     * up with a burst of samples */
    tick.tv_sec += 1;
    struct timespec mono;
    clock_gettime(CLOCK_MONOTONIC, &mono);
    if (mono.tv_sec > tick.tv_sec + 1)
      tick = mono;
    while (!stop_requested &&
           clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &tick, NULL) == EINTR)
      ;
  }

  printf("daemon: stopped\n");
  free(wind);
  return 0;
}

static void usage(const char *prog) {
  printf("Usage: %s -c CONFIG [options]\n"
         "  -c, --config FILE      config.ini (modbus_* keys, database_directory)\n"
         "  -d, --databasedir DIR  override database_directory\n"
         "  -f, --outfile FILE     also write the reading as JSON (atomic)\n"
         "  -D, --daemon           run continuously: wind every modbus_wind_sample_interval\n"
         "                         s (RAM ring buffer), a stored reading every\n"
         "                         modbus_interval s\n"
         "  -n, --dry-run          read and print, do not write the database\n"
         "  -v, --verbose          show the configuration and every retry\n"
         "                         (twice in daemon mode: every wind sample)\n"
         "  -h, --help             this help\n"
         "Without --daemon all sensors are read once (instantaneous values).\n",
         prog);
}

int main(int argc, char *argv[]) {
  const char *config_path = NULL, *database_dir = NULL, *outfile = NULL;
  int dry_run = 0, daemon_mode = 0;

  static struct option long_opts[] = {
      {"config", required_argument, 0, 'c'},
      {"databasedir", required_argument, 0, 'd'},
      {"outfile", required_argument, 0, 'f'},
      {"daemon", no_argument, 0, 'D'},
      {"dry-run", no_argument, 0, 'n'},
      {"verbose", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};
  int opt;
  while ((opt = getopt_long(argc, argv, "c:d:f:Dnvh", long_opts, NULL)) != -1) {
    switch (opt) {
    case 'c': config_path = optarg; break;
    case 'd': database_dir = optarg; break;
    case 'f': outfile = optarg; break;
    case 'D': daemon_mode = 1; break;
    case 'n': dry_run = 1; break;
    case 'v': verbose++; break;
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

  if (daemon_mode) {
    int ret = run_daemon(&bus, &config, outfile, dry_run);
    modbus_close(&bus);
    return ret;
  }

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

  hold_direction_in_calm(&config, &r);
  printf("%d of %d sensors read\n", ok, enabled);
  if (ok == 0)
    return 2;
  return store_reading(&config, &r, outfile, dry_run);
}
