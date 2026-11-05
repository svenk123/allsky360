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
#include <math.h>
#include <sqlite3.h>
#include <stdio.h>

#include "sensors_database.h"

void sensors_reading_init(sensors_reading_t *r, time_t timestamp) {
  r->timestamp = timestamp;
  r->temperature = r->humidity = r->dew_point = r->pressure = NAN;
  r->ws_wind_speed = r->ws_wind_direction = r->noise = NAN;
  r->pm25 = r->pm10 = r->illuminance = r->rainfall = NAN;
  r->solar_irradiance_raw = NAN;
  r->wind_speed = r->wind_direction = NAN;
  r->cabinet_temperature = r->cabinet_humidity = r->cabinet_dew_point = NAN;
}

/* Same table layout for every daily database (one file per UTC day). */
static const char *CREATE_SQL =
    "CREATE TABLE IF NOT EXISTS sensors ("
    "timestamp INTEGER PRIMARY KEY, "
    "temperature REAL, humidity REAL, dew_point REAL, pressure REAL, "
    "ws_wind_speed REAL, ws_wind_direction REAL, noise REAL, "
    "pm25 REAL, pm10 REAL, illuminance REAL, rainfall REAL, "
    "solar_irradiance_raw REAL, "
    "wind_speed REAL, wind_direction REAL, "
    "cabinet_temperature REAL, cabinet_humidity REAL, cabinet_dew_point REAL);";

static const char *INSERT_SQL =
    "INSERT OR REPLACE INTO sensors (timestamp, temperature, humidity, "
    "dew_point, pressure, ws_wind_speed, ws_wind_direction, noise, pm25, pm10, "
    "illuminance, rainfall, solar_irradiance_raw, wind_speed, wind_direction, "
    "cabinet_temperature, cabinet_humidity, cabinet_dew_point) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

static void bind_value(sqlite3_stmt *stmt, int index, double value) {
  if (isnan(value))
    sqlite3_bind_null(stmt, index);
  else
    sqlite3_bind_double(stmt, index, value);
}

int sensors_database_last_value(const char *db_filename, const char *column,
                                time_t before, double *value, time_t *timestamp) {
  sqlite3 *db = NULL;
  /* Read only: a missing database (new day, first run) is not created */
  if (sqlite3_open_v2(db_filename, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
    sqlite3_close(db);
    return -1;
  }
  sqlite3_busy_timeout(db, 5000);

  char sql[256];
  snprintf(sql, sizeof(sql),
           "SELECT timestamp, %s FROM sensors WHERE %s IS NOT NULL AND timestamp < ? "
           "ORDER BY timestamp DESC LIMIT 1;",
           column, column);
  sqlite3_stmt *stmt = NULL;
  int ret = -1;
  if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
    sqlite3_bind_int64(stmt, 1, (sqlite3_int64)before);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      *timestamp = (time_t)sqlite3_column_int64(stmt, 0);
      *value = sqlite3_column_double(stmt, 1);
      ret = 0;
    }
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return ret;
}

int sensors_database_store(const char *db_filename, const sensors_reading_t *r) {
  sqlite3 *db = NULL;
  if (sqlite3_open(db_filename, &db) != SQLITE_OK) {
    fprintf(stderr, "ERROR: cannot open database %s: %s\n", db_filename,
            sqlite3_errmsg(db));
    sqlite3_close(db);
    return 1;
  }
  /* Several processes write to the same daily database (review.md REL-07) */
  sqlite3_busy_timeout(db, 5000);

  char *errmsg = NULL;
  if (sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, &errmsg) != SQLITE_OK ||
      sqlite3_exec(db, CREATE_SQL, NULL, NULL, &errmsg) != SQLITE_OK) {
    fprintf(stderr, "ERROR: database %s: %s\n", db_filename, errmsg);
    sqlite3_free(errmsg);
    sqlite3_close(db);
    return 2;
  }

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db, INSERT_SQL, -1, &stmt, NULL) != SQLITE_OK) {
    fprintf(stderr, "ERROR: prepare insert: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    return 3;
  }

  int i = 1;
  sqlite3_bind_int64(stmt, i++, (sqlite3_int64)r->timestamp);
  bind_value(stmt, i++, r->temperature);
  bind_value(stmt, i++, r->humidity);
  bind_value(stmt, i++, r->dew_point);
  bind_value(stmt, i++, r->pressure);
  bind_value(stmt, i++, r->ws_wind_speed);
  bind_value(stmt, i++, r->ws_wind_direction);
  bind_value(stmt, i++, r->noise);
  bind_value(stmt, i++, r->pm25);
  bind_value(stmt, i++, r->pm10);
  bind_value(stmt, i++, r->illuminance);
  bind_value(stmt, i++, r->rainfall);
  bind_value(stmt, i++, r->solar_irradiance_raw);
  bind_value(stmt, i++, r->wind_speed);
  bind_value(stmt, i++, r->wind_direction);
  bind_value(stmt, i++, r->cabinet_temperature);
  bind_value(stmt, i++, r->cabinet_humidity);
  bind_value(stmt, i++, r->cabinet_dew_point);

  int rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE)
    fprintf(stderr, "ERROR: insert into %s: %s\n", db_filename, sqlite3_errmsg(db));
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return rc == SQLITE_DONE ? 0 : 4;
}
